# 自然语言处理 cs.CL

- **最新发布 90 篇**

- **更新 62 篇**

## 最新发布

#### [new 001] StyliTruth : Unlocking Stylized yet Truthful LLM Generation via Disentangled Steering
- **分类: cs.CL**

- **简介: 该论文旨在解决生成风格化但真实性差的LLM回答问题，通过将风格与真实方向分离并使用自适应引导机制，有效减少了风格-真实之间的干扰，显著提升了模型的风格与真实性能平衡能力。**

- **链接: [http://arxiv.org/pdf/2508.04530v1](http://arxiv.org/pdf/2508.04530v1)**

> **作者:** Chenglei Shen; Zhongxiang Sun; Teng Shi; Xiao Zhang; Jun Xu
>
> **摘要:** Generating stylized large language model (LLM) responses via representation editing is a promising way for fine-grained output control. However, there exists an inherent trade-off: imposing a distinctive style often degrades truthfulness. Existing representation editing methods, by naively injecting style signals, overlook this collateral impact and frequently contaminate the model's core truthfulness representations, resulting in reduced answer correctness. We term this phenomenon stylization-induced truthfulness collapse. We attribute this issue to latent coupling between style and truth directions in certain key attention heads, and propose StyliTruth, a mechanism that preserves stylization while keeping truthfulness intact. StyliTruth separates the style-relevant and truth-relevant subspaces in the model's representation space via an orthogonal deflation process. This decomposition enables independent control of style and truth in their own subspaces, minimizing interference. By designing adaptive, token-level steering vectors within each subspace, we dynamically and precisely control the generation process to maintain both stylistic fidelity and truthfulness. We validate our method on multiple styles and languages. Extensive experiments and analyses show that StyliTruth significantly reduces stylization-induced truthfulness collapse and outperforms existing inference-time intervention methods in balancing style adherence with truthfulness.
>
---
#### [new 002] Dialogue Response Prefetching Based on Semantic Similarity and Prediction Confidence of Language Model
- **分类: cs.CL**

- **简介: 该论文旨在减少用户感知延迟（UPL），通过构建预测信心模型（PCM）评估语言模型的语义相似性，解决对话响应预取问题。**

- **链接: [http://arxiv.org/pdf/2508.04403v1](http://arxiv.org/pdf/2508.04403v1)**

> **作者:** Kiyotada Mori; Seiya Kawano; Angel Fernando Garcia Contreras; Koichiro Yoshino
>
> **摘要:** Prefetching of dialogue responses has been investigated to reduce user-perceived latency (UPL), which refers to the user's waiting time before receiving the system's response, in spoken dialogue systems. To reduce the UPL, it is necessary to predict complete user utterances before the end of the user's speech, typically by language models, to prepare prefetched dialogue responses. In this study, we proposed a prediction confidence model (PCM) that determines whether prefetching is possible or not by estimating the semantic similarity between the predicted complete user utterance and the complete user utterance. We evaluated our PCM based on the differences between the predicted complete user utterance and the complete user utterance.
>
---
#### [new 003] WINELL: Wikipedia Never-Ending Updating with LLM Agents
- **分类: cs.CL**

- **简介: 该研究提出一种基于多智能体的框架，旨在解决Wikipedia内容更新的挑战，通过整合在线信息并生成编辑建议来实现持续性更新。**

- **链接: [http://arxiv.org/pdf/2508.03728v1](http://arxiv.org/pdf/2508.03728v1)**

> **作者:** Revanth Gangi Reddy; Tanay Dixit; Jiaxin Qin; Cheng Qian; Daniel Lee; Jiawei Han; Kevin Small; Xing Fan; Ruhi Sarikaya; Heng Ji
>
> **摘要:** Wikipedia, a vast and continuously consulted knowledge base, faces significant challenges in maintaining up-to-date content due to its reliance on manual human editors. Inspired by the vision of continuous knowledge acquisition in NELL and fueled by advances in LLM-based agents, this paper introduces WiNELL, an agentic framework for continuously updating Wikipedia articles. Our approach employs a multi-agent framework to aggregate online information, select new and important knowledge for a target entity in Wikipedia, and then generate precise edit suggestions for human review. Our fine-grained editing models, trained on Wikipedia's extensive history of human edits, enable incorporating updates in a manner consistent with human editing behavior. Our editor models outperform both open-source instruction-following baselines and closed-source LLMs (e.g., GPT-4o) in key information coverage and editing efficiency. End-to-end evaluation on high-activity Wikipedia pages demonstrates WiNELL's ability to identify and suggest timely factual updates. This opens up a promising research direction in LLM agents for automatically updating knowledge bases in a never-ending fashion.
>
---
#### [new 004] Efficient Strategy for Improving Large Language Model (LLM) Capabilities
- **分类: cs.CL; cs.LG; I.2.7; I.2.6; I.5.1**

- **简介: 该论文旨在解决大型语言模型资源受限部署的问题，通过构建高效策略（包括数据处理与选择、训练策略及架构调整）提升其在知识库内的能效与安全性，验证了相关方法的有效性。**

- **链接: [http://arxiv.org/pdf/2508.04073v1](http://arxiv.org/pdf/2508.04073v1)**

> **作者:** Julián Camilo Velandia Gutiérrez
>
> **备注:** Based on master's thesis in Systems and Computer Engineering, Universidad Nacional de Colombia (2025)
>
> **摘要:** Large Language Models (LLMs) have become a milestone in the field of artificial intelligence and natural language processing. However, their large-scale deployment remains constrained by the need for significant computational resources. This work proposes starting from a base model to explore and combine data processing and careful data selection techniques, training strategies, and architectural adjustments to improve the efficiency of LLMs in resource-constrained environments and within a delimited knowledge base. The methodological approach included defining criteria for building reliable datasets, conducting controlled experiments with different configurations, and systematically evaluating the resulting variants in terms of capability, versatility, response time, and safety. Finally, comparative tests were conducted to measure the performance of the developed variants and to validate the effectiveness of the proposed strategies. This work is based on the master's thesis in Systems and Computer Engineering titled "Efficient Strategy for Improving the Capabilities of Large Language Models (LLMs)".
>
---
#### [new 005] ShoppingBench: A Real-World Intent-Grounded Shopping Benchmark for LLM-based Agents
- **分类: cs.CL**

- **简介: 该论文旨在填补e-commerce领域对复杂意图（如优惠券应用、预算管理）的评估空白，提出ShoppingBench作为端到端基准，通过多意图模拟框架与大规模沙盒实现高精度训练，利用轨迹蒸馏与强化学习提升LLM性能，验证其在复杂任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.04266v1](http://arxiv.org/pdf/2508.04266v1)**

> **作者:** Jiangyuan Wang; Kejun Xiao; Qi Sun; Huaipeng Zhao; Tao Luo; Jiandong Zhang; Xiaoyi Zeng
>
> **备注:** submit to AAAI2026
>
> **摘要:** Existing benchmarks in e-commerce primarily focus on basic user intents, such as finding or purchasing products. However, real-world users often pursue more complex goals, such as applying vouchers, managing budgets, and finding multi-products seller. To bridge this gap, we propose ShoppingBench, a novel end-to-end shopping benchmark designed to encompass increasingly challenging levels of grounded intent. Specifically, we propose a scalable framework to simulate user instructions based on various intents derived from sampled real-world products. To facilitate consistent and reliable evaluations, we provide a large-scale shopping sandbox that serves as an interactive simulated environment, incorporating over 2.5 million real-world products. Experimental results demonstrate that even state-of-the-art language agents (such as GPT-4.1) achieve absolute success rates under 50% on our benchmark tasks, highlighting the significant challenges posed by our ShoppingBench. In addition, we propose a trajectory distillation strategy and leverage supervised fine-tuning, along with reinforcement learning on synthetic trajectories, to distill the capabilities of a large language agent into a smaller one. As a result, our trained agent achieves competitive performance compared to GPT-4.1.
>
---
#### [new 006] Are Today's LLMs Ready to Explain Well-Being Concepts?
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 本研究探讨了LLMs在解释良好性方面的挑战，通过构建数据集和优化模型方法来提升其解释质量。**

- **链接: [http://arxiv.org/pdf/2508.03990v1](http://arxiv.org/pdf/2508.03990v1)**

> **作者:** Bohan Jiang; Dawei Li; Zhen Tan; Chengshuai Zhao; Huan Liu
>
> **备注:** 9 pages, 4 figures, 3 tables
>
> **摘要:** Well-being encompasses mental, physical, and social dimensions essential to personal growth and informed life decisions. As individuals increasingly consult Large Language Models (LLMs) to understand well-being, a key challenge emerges: Can LLMs generate explanations that are not only accurate but also tailored to diverse audiences? High-quality explanations require both factual correctness and the ability to meet the expectations of users with varying expertise. In this work, we construct a large-scale dataset comprising 43,880 explanations of 2,194 well-being concepts, generated by ten diverse LLMs. We introduce a principle-guided LLM-as-a-judge evaluation framework, employing dual judges to assess explanation quality. Furthermore, we show that fine-tuning an open-source LLM using Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) can significantly enhance the quality of generated explanations. Our results reveal: (1) The proposed LLM judges align well with human evaluations; (2) explanation quality varies significantly across models, audiences, and categories; and (3) DPO- and SFT-finetuned models outperform their larger counterparts, demonstrating the effectiveness of preference-based learning for specialized explanation tasks.
>
---
#### [new 007] Characterizing Deep Research: A Benchmark and Formal Definition
- **分类: cs.CL**

- **简介: 该论文探讨了深研究（DR）任务的定义及其评估方法，指出其核心在于高效概念探索而非长篇报告生成，通过中间输出表示区分推理与报告生成，并构建了包含100个挑战任务的基准（LiveDRBench），验证了DR系统在F1分数上的性能差异及推理机制改进的方向。**

- **链接: [http://arxiv.org/pdf/2508.04183v1](http://arxiv.org/pdf/2508.04183v1)**

> **作者:** Abhinav Java; Ashmit Khandelwal; Sukruta Midigeshi; Aaron Halfaker; Amit Deshpande; Navin Goyal; Ankur Gupta; Nagarajan Natarajan; Amit Sharma
>
> **备注:** First three authors contributed equally (ordered alphabetically)
>
> **摘要:** Information tasks such as writing surveys or analytical reports require complex search and reasoning, and have recently been grouped under the umbrella of \textit{deep research} -- a term also adopted by recent models targeting these capabilities. Despite growing interest, the scope of the deep research task remains underdefined and its distinction from other reasoning-intensive problems is poorly understood. In this paper, we propose a formal characterization of the deep research (DR) task and introduce a benchmark to evaluate the performance of DR systems. We argue that the core defining feature of deep research is not the production of lengthy report-style outputs, but rather the high fan-out over concepts required during the search process, i.e., broad and reasoning-intensive exploration. To enable objective evaluation, we define DR using an intermediate output representation that encodes key claims uncovered during search-separating the reasoning challenge from surface-level report generation. Based on this formulation, we propose a diverse, challenging benchmark LiveDRBench with 100 challenging tasks over scientific topics (e.g., datasets, materials discovery, prior art search) and public interest events (e.g., flight incidents, movie awards). Across state-of-the-art DR systems, F1 score ranges between 0.02 and 0.72 for any sub-category. OpenAI's model performs the best with an overall F1 score of 0.55. Analysis of reasoning traces reveals the distribution over the number of referenced sources, branching, and backtracking events executed by current DR systems, motivating future directions for improving their search mechanisms and grounding capabilities. The benchmark is available at https://github.com/microsoft/LiveDRBench.
>
---
#### [new 008] Hop, Skip, and Overthink: Diagnosing Why Reasoning Models Fumble during Multi-Hop Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在研究推理模型在多步问答任务中的认知局限性，通过系统探索三个维度（跳过/覆盖/过思考）的错误特征，构建错误分类框架，并结合人类标注与自动化评估揭示其潜在问题，为提升语言模型的推理准确性与透明度提供理论支持。**

- **链接: [http://arxiv.org/pdf/2508.04699v1](http://arxiv.org/pdf/2508.04699v1)**

> **作者:** Anushka Yadav; Isha Nalawade; Srujana Pillarichety; Yashwanth Babu; Reshmi Ghosh; Samyadeep Basu; Wenlong Zhao; Ali Nasaeh; Sriram Balasubramanian; Soundararajan Srinivasan
>
> **摘要:** The emergence of reasoning models and their integration into practical AI chat bots has led to breakthroughs in solving advanced math, deep search, and extractive question answering problems that requires a complex and multi-step thought process. Yet, a complete understanding of why these models hallucinate more than general purpose language models is missing. In this investigative study, we systematicallyexplore reasoning failures of contemporary language models on multi-hop question answering tasks. We introduce a novel, nuanced error categorization framework that examines failures across three critical dimensions: the diversity and uniqueness of source documents involved ("hops"), completeness in capturing relevant information ("coverage"), and cognitive inefficiency ("overthinking"). Through rigorous hu-man annotation, supported by complementary automated metrics, our exploration uncovers intricate error patterns often hidden by accuracy-centric evaluations. This investigative approach provides deeper insights into the cognitive limitations of current models and offers actionable guidance toward enhancing reasoning fidelity, transparency, and robustness in future language modeling efforts.
>
---
#### [new 009] Confidence-Weighted Token Set Cover for Early Hypothesis Pruning in Self-Consistency
- **分类: cs.CL**

- **简介: 该论文研究了如何通过早期假设筛选优化自一致性以提升token效率，解决了长链推理任务中的token消耗问题，设计了基于模型信心和词汇覆盖的高效算法。**

- **链接: [http://arxiv.org/pdf/2508.03979v1](http://arxiv.org/pdf/2508.03979v1)**

> **作者:** Md Arafat Sultan; Ramón Fernandez Astudillo
>
> **摘要:** Despite its simplicity and efficacy, the high token expenditure of self-consistency can limit its practical utility. Here we investigate if self-consistency can be made more token-efficient for long chain-of-thought reasoning tasks, while preserving its parallelism, through early hypothesis pruning. Concretely, we generate all solutions in parallel, but periodically prune intermediate hypotheses that are deemed unnecessary based on two lightweight indicators: (a) the model's own confidence in individual hypotheses, and (b) lexical coverage of all current hypotheses by candidate subsets that are under consideration for continued retention. We design a fast weighted set cover algorithm that utilizes the two indicators; our evaluation of five LLMs on three math benchmarks shows that this method can improve token efficiency for all models, by 10-35% in many cases.
>
---
#### [new 010] CALE : Concept-Aligned Embeddings for Both Within-Lemma and Inter-Lemma Sense Differentiation
- **分类: cs.CL**

- **简介: 该论文提出**概念关联嵌入（Concept-Aligned Embeddings, CALE）**，解决跨词语义差异和上下文敏感性问题，通过引入概念区分机制，利用SemCor数据集优化多个任务的表现，实现高效多用途词汇表示。**

- **链接: [http://arxiv.org/pdf/2508.04494v1](http://arxiv.org/pdf/2508.04494v1)**

> **作者:** Bastien Liétard; Gabriel Loiseau
>
> **备注:** Under review in ARR July 2025
>
> **摘要:** Lexical semantics is concerned with both the multiple senses a word can adopt in different contexts, and the semantic relations that exist between meanings of different words. To investigate them, Contextualized Language Models are a valuable tool that provides context-sensitive representations that can be used to investigate lexical meaning. Recent works like XL-LEXEME have leveraged the task of Word-in-Context to fine-tune them to get more semantically accurate representations, but Word-in-Context only compares occurrences of the same lemma, limiting the range of captured information. In this paper, we propose an extension, Concept Differentiation, to include inter-words scenarios. We provide a dataset for this task, derived from SemCor data. Then we fine-tune several representation models on this dataset. We call these models Concept-Aligned Embeddings (CALE). By challenging our models and other models on various lexical semantic tasks, we demonstrate that the proposed models provide efficient multi-purpose representations of lexical meaning that reach best performances in our experiments. We also show that CALE's fine-tuning brings valuable changes to the spatial organization of embeddings.
>
---
#### [new 011] ReasoningGuard: Safeguarding Large Reasoning Models with Inference-time Safety Aha Moments
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在解决大型推理模型（LRMs）在推理过程中因潜在有害内容生成而面临的安全风险问题。通过在推理时注入安全"aha"时刻，利用模型内部注意力机制识别关键步骤并触发安全反思，结合解码阶段的采样策略优化推理路径，提出"推理-采样"双重防护机制，有效应对三种攻击，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.04204v1](http://arxiv.org/pdf/2508.04204v1)**

> **作者:** Yuquan Wang; Mi Zhang; Yining Wang; Geng Hong; Xiaoyu You; Min Yang
>
> **摘要:** Large Reasoning Models (LRMs) have demonstrated impressive performance in reasoning-intensive tasks, but they remain vulnerable to harmful content generation, particularly in the mid-to-late steps of their reasoning processes. Existing defense mechanisms, however, rely on costly fine-tuning and additional expert knowledge, which restricts their scalability. In this work, we propose ReasoningGuard, an inference-time safeguard for LRMs, which injects timely safety aha moments to steer harmless while helpful reasoning processes. Leveraging the model's internal attention behavior, our approach accurately identifies critical points in the reasoning path, and triggers spontaneous, safety-oriented reflection. To safeguard both the subsequent reasoning steps and the final answers, we further implement a scaling sampling strategy during the decoding phase, selecting the optimal reasoning path. Inducing minimal extra inference cost, ReasoningGuard effectively mitigates three types of jailbreak attacks, including the latest ones targeting the reasoning process of LRMs. Our approach outperforms seven existing safeguards, achieving state-of-the-art safety defenses while effectively avoiding the common exaggerated safety issues.
>
---
#### [new 012] AIC CTU@FEVER 8: On-premise fact checking through long context RAG
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出了一种基于长期上下文的在线事实核查系统，解决传统系统在资源受限场景下的效率问题。通过改进RAG架构并部署到本地，实现了在NVidia A10 GPU（23GB内存）条件下达到Ev2R测试分数的优异性能。**

- **链接: [http://arxiv.org/pdf/2508.04390v1](http://arxiv.org/pdf/2508.04390v1)**

> **作者:** Herbert Ullrich; Jan Drchal
>
> **摘要:** In this paper, we present our fact-checking pipeline which has scored first in FEVER 8 shared task. Our fact-checking system is a simple two-step RAG pipeline based on our last year's submission. We show how the pipeline can be redeployed on-premise, achieving state-of-the-art fact-checking performance (in sense of Ev2R test-score), even under the constraint of a single NVidia A10 GPU, 23GB of graphical memory and 60s running time per claim.
>
---
#### [new 013] Evaluating, Synthesizing, and Enhancing for Customer Support Conversation
- **分类: cs.CL**

- **简介: 该论文提出Customer Support Conversation（CSC）任务，旨在通过结构化框架提升客户支持对话质量，构建CSConv和RoleCS数据集，利用LLM优化策略响应能力，并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.04423v1](http://arxiv.org/pdf/2508.04423v1)**

> **作者:** Jie Zhu; Huaixia Dou; Junhui Li; Lifan Guo; Feng Chen; Chi Zhang; Fang Kong
>
> **备注:** under review
>
> **摘要:** Effective customer support requires not only accurate problem solving but also structured and empathetic communication aligned with professional standards. However, existing dialogue datasets often lack strategic guidance, and real-world service data is difficult to access and annotate. To address this, we introduce the task of Customer Support Conversation (CSC), aimed at training customer service agents to respond using well-defined support strategies. We propose a structured CSC framework grounded in COPC guidelines, defining five conversational stages and twelve strategies to guide high-quality interactions. Based on this, we construct CSConv, an evaluation dataset of 1,855 real-world customer-agent conversations rewritten using LLMs to reflect deliberate strategy use, and annotated accordingly. Additionally, we develop a role-playing approach that simulates strategy-rich conversations using LLM-powered roles aligned with the CSC framework, resulting in the training dataset RoleCS. Experiments show that fine-tuning strong LLMs on RoleCS significantly improves their ability to generate high-quality, strategy-aligned responses on CSConv. Human evaluations further confirm gains in problem resolution. All code and data will be made publicly available at https://github.com/aliyun/qwen-dianjin.
>
---
#### [new 014] An Entity Linking Agent for Question Answering
- **分类: cs.CL**

- **简介: 该论文提出了一个基于大语言模型的实体链接代理，用于问答系统，解决了传统EL方法在处理短、模糊用户问题时效率低下及准确性不足的问题，通过模拟人类认知流程实现主动实体识别与决策。**

- **链接: [http://arxiv.org/pdf/2508.03865v1](http://arxiv.org/pdf/2508.03865v1)**

> **作者:** Yajie Luo; Yihong Wu; Muzhi Li; Fengran Mo; Jia Ao Sun; Xinyu Wang; Liheng Ma; Yingxue Zhang; Jian-Yun Nie
>
> **备注:** 12 pages, 2 figures. Submitted to AAAI 2026 Conference
>
> **摘要:** Some Question Answering (QA) systems rely on knowledge bases (KBs) to provide accurate answers. Entity Linking (EL) plays a critical role in linking natural language mentions to KB entries. However, most existing EL methods are designed for long contexts and do not perform well on short, ambiguous user questions in QA tasks. We propose an entity linking agent for QA, based on a Large Language Model that simulates human cognitive workflows. The agent actively identifies entity mentions, retrieves candidate entities, and makes decision. To verify the effectiveness of our agent, we conduct two experiments: tool-based entity linking and QA task evaluation. The results confirm the robustness and effectiveness of our agent.
>
---
#### [new 015] Improving Crash Data Quality with Large Language Models: Evidence from Secondary Crash Narratives in Kentucky
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文旨在通过自然语言处理技术提升事故数据质量，解决事故信息不完整等问题，研究对比了多种模型并提出了优化部署策略。**

- **链接: [http://arxiv.org/pdf/2508.04399v1](http://arxiv.org/pdf/2508.04399v1)**

> **作者:** Xu Zhang; Mei Chen
>
> **备注:** 19 pages, 2 figures
>
> **摘要:** This study evaluates advanced natural language processing (NLP) techniques to enhance crash data quality by mining crash narratives, using secondary crash identification in Kentucky as a case study. Drawing from 16,656 manually reviewed narratives from 2015-2022, with 3,803 confirmed secondary crashes, we compare three model classes: zero-shot open-source large language models (LLMs) (LLaMA3:70B, DeepSeek-R1:70B, Qwen3:32B, Gemma3:27B); fine-tuned transformers (BERT, DistilBERT, RoBERTa, XLNet, Longformer); and traditional logistic regression as baseline. Models were calibrated on 2015-2021 data and tested on 1,771 narratives from 2022. Fine-tuned transformers achieved superior performance, with RoBERTa yielding the highest F1-score (0.90) and accuracy (95%). Zero-shot LLaMA3:70B reached a comparable F1 of 0.86 but required 139 minutes of inference; the logistic baseline lagged well behind (F1:0.66). LLMs excelled in recall for some variants (e.g., GEMMA3:27B at 0.94) but incurred high computational costs (up to 723 minutes for DeepSeek-R1:70B), while fine-tuned models processed the test set in seconds after brief training. Further analysis indicated that mid-sized LLMs (e.g., DeepSeek-R1:32B) can rival larger counterparts in performance while reducing runtime, suggesting opportunities for optimized deployments. Results highlight trade-offs between accuracy, efficiency, and data requirements, with fine-tuned transformer models balancing precision and recall effectively on Kentucky data. Practical deployment considerations emphasize privacy-preserving local deployment, ensemble approaches for improved accuracy, and incremental processing for scalability, providing a replicable scheme for enhancing crash-data quality with advanced NLP.
>
---
#### [new 016] Transferring Expert Cognitive Models to Social Robots via Agentic Concept Bottleneck Models
- **分类: cs.CL**

- **简介: 该论文属于跨领域人机协同任务，旨在解决传统FMs无法满足实时干预与跨组适应的挑战。通过构建Agentic Concept Bottleneck Model（CBM）和基于知识转移的框架，实现了对社交互动的智能解析与透明决策，显著提升了机器人在复杂群体中的效能与适应性。**

- **链接: [http://arxiv.org/pdf/2508.03998v1](http://arxiv.org/pdf/2508.03998v1)**

> **作者:** Xinyu Zhao; Zhen Tan; Maya Enisman; Minjae Seo; Marta R. Durantini; Dolores Albarracin; Tianlong Chen
>
> **备注:** 27 pages, 7 figures
>
> **摘要:** Successful group meetings, such as those implemented in group behavioral-change programs, work meetings, and other social contexts, must promote individual goal setting and execution while strengthening the social relationships within the group. Consequently, an ideal facilitator must be sensitive to the subtle dynamics of disengagement, difficulties with individual goal setting and execution, and interpersonal difficulties that signal a need for intervention. The challenges and cognitive load experienced by facilitators create a critical gap for an embodied technology that can interpret social exchanges while remaining aware of the needs of the individuals in the group and providing transparent recommendations that go beyond powerful but "black box" foundation models (FMs) that identify social cues. We address this important demand with a social robot co-facilitator that analyzes multimodal meeting data and provides discreet cues to the facilitator. The robot's reasoning is powered by an agentic concept bottleneck model (CBM), which makes decisions based on human-interpretable concepts like participant engagement and sentiments, ensuring transparency and trustworthiness. Our core contribution is a transfer learning framework that distills the broad social understanding of an FM into our specialized and transparent CBM. This concept-driven system significantly outperforms direct zero-shot FMs in predicting the need for intervention and enables real-time human correction of its reasoning. Critically, we demonstrate robust knowledge transfer: the model generalizes across different groups and successfully transfers the expertise of senior human facilitators to improve the performance of novices. By transferring an expert's cognitive model into an interpretable robotic partner, our work provides a powerful blueprint for augmenting human capabilities in complex social domains.
>
---
#### [new 017] Share Your Attention: Transformer Weight Sharing via Matrix-based Dictionary Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在解决大型语言模型（LLMs）高计算/内存需求的问题，提出通过矩阵字典学习实现Transformer权重共享的方法，有效降低参数量并保持性能，证明其在多个任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.04581v1](http://arxiv.org/pdf/2508.04581v1)**

> **作者:** Magauiya Zhussip; Dmitriy Shopkhoev; Ammar Ali; Stamatios Lefkimmiatis
>
> **摘要:** Large language models (LLMs) have revolutionized AI applications, yet their high computational and memory demands hinder their widespread deployment. Existing compression techniques focus on intra-block optimizations (e.g. low-rank approximation, attention head pruning), while the repetitive layered structure of transformers implies significant inter-block redundancy - a dimension largely unexplored beyond key-value (KV) caching. Inspired by dictionary learning in CNNs, we propose a framework for structured weight sharing across transformer layers. Our approach decomposes attention projection matrices into shared dictionary atoms, reducing the attention module's parameters by 66.7% while achieving on-par performance. Unlike complex methods requiring distillation or architectural changes, MASA (Matrix Atom Sharing in Attention) operates as a drop-in replacement - trained with standard optimizers - and represents each layer's weights as linear combinations of shared matrix atoms. Experiments across scales (100M-700M parameters) show that MASA achieves better benchmark accuracy and perplexity than grouped-query attention (GQA), low-rank baselines and recently proposed Repeat-all-over/Sequential sharing at comparable parameter budgets. Ablation studies confirm robustness to the dictionary size and the efficacy of shared representations in capturing cross-layer statistical regularities. Extending to Vision Transformers (ViT), MASA matches performance metrics on image classification and detection tasks with 66.7% fewer attention parameters. By combining dictionary learning strategies with transformer efficiency, MASA offers a scalable blueprint for parameter-efficient models without sacrificing performance. Finally, we investigate the possibility of employing MASA on pretrained LLMs to reduce their number of parameters without experiencing any significant drop in their performance.
>
---
#### [new 018] Data and AI governance: Promoting equity, ethics, and fairness in large language models
- **分类: cs.CL; cs.AI; 68T01 (Primary), 68T50 (Secondary); I.2.0; I.2.7**

- **简介: 该论文旨在系统化治理大语言模型（LLMs）的偏见、伦理与公平性，通过构建BEATS测试套件及提出数据与AI治理框架，解决了偏见评估与风险控制的问题，推动LLMs在实际场景中的安全部署。**

- **链接: [http://arxiv.org/pdf/2508.03970v1](http://arxiv.org/pdf/2508.03970v1)**

> **作者:** Alok Abhishek; Lisa Erickson; Tushar Bandopadhyay
>
> **备注:** Published in MIT Science Policy Review 6, 139-146 (2025)
>
> **摘要:** In this paper, we cover approaches to systematically govern, assess and quantify bias across the complete life cycle of machine learning models, from initial development and validation to ongoing production monitoring and guardrail implementation. Building upon our foundational work on the Bias Evaluation and Assessment Test Suite (BEATS) for Large Language Models, the authors share prevalent bias and fairness related gaps in Large Language Models (LLMs) and discuss data and AI governance framework to address Bias, Ethics, Fairness, and Factuality within LLMs. The data and AI governance approach discussed in this paper is suitable for practical, real-world applications, enabling rigorous benchmarking of LLMs prior to production deployment, facilitating continuous real-time evaluation, and proactively governing LLM generated responses. By implementing the data and AI governance across the life cycle of AI development, organizations can significantly enhance the safety and responsibility of their GenAI systems, effectively mitigating risks of discrimination and protecting against potential reputational or brand-related harm. Ultimately, through this article, we aim to contribute to advancement of the creation and deployment of socially responsible and ethically aligned generative artificial intelligence powered applications.
>
---
#### [new 019] Hierarchical Text Classification Using Black Box Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该研究探讨了使用黑盒大语言模型实现层次化文本分类的任务，解决了传统方法因数据不足和模型复杂性导致的HTC挑战，通过对比多种提示策略验证了LMMs在深度结构中的优势，同时分析了API成本差异。**

- **链接: [http://arxiv.org/pdf/2508.04219v1](http://arxiv.org/pdf/2508.04219v1)**

> **作者:** Kosuke Yoshimura; Hisashi Kashima
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Hierarchical Text Classification (HTC) aims to assign texts to structured label hierarchies; however, it faces challenges due to data scarcity and model complexity. This study explores the feasibility of using black box Large Language Models (LLMs) accessed via APIs for HTC, as an alternative to traditional machine learning methods that require extensive labeled data and computational resources. We evaluate three prompting strategies -- Direct Leaf Label Prediction (DL), Direct Hierarchical Label Prediction (DH), and Top-down Multi-step Hierarchical Label Prediction (TMH) -- in both zero-shot and few-shot settings, comparing the accuracy and cost-effectiveness of these strategies. Experiments on two datasets show that a few-shot setting consistently improves classification accuracy compared to a zero-shot setting. While a traditional machine learning model achieves high accuracy on a dataset with a shallow hierarchy, LLMs, especially DH strategy, tend to outperform the machine learning model on a dataset with a deeper hierarchy. API costs increase significantly due to the higher input tokens required for deeper label hierarchies on DH strategy. These results emphasize the trade-off between accuracy improvement and the computational cost of prompt strategy. These findings highlight the potential of black box LLMs for HTC while underscoring the need to carefully select a prompt strategy to balance performance and cost.
>
---
#### [new 020] Step More: Going Beyond Single Backpropagation in Meta Learning Based Model Editing
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文旨在解决大型语言模型（LLM）在模型编辑中的效率与效果问题，提出Step More方法通过引入多步Backpropagation（MBPS）和正则化技术，提升了在低数据场景下的编辑性能和训练效率。**

- **链接: [http://arxiv.org/pdf/2508.04012v1](http://arxiv.org/pdf/2508.04012v1)**

> **作者:** Xiaopeng Li; Shasha Li; Xi Wang; Shezheng Song; Bin Ji; Shangwen Wang; Jun Ma; Xiaodong Liu; Mina Liu; Jie Yu
>
> **摘要:** Large Language Models (LLMs) underpin many AI applications, but their static nature makes updating knowledge costly. Model editing offers an efficient alternative by injecting new information through targeted parameter modifications. In particular, meta-learning-based model editing (MLBME) methods have demonstrated notable advantages in both editing effectiveness and efficiency. Despite this, we find that MLBME exhibits suboptimal performance in low-data scenarios, and its training efficiency is bottlenecked by the computation of KL divergence. To address these, we propose $\textbf{S}$tep $\textbf{M}$ore $\textbf{Edit}$ ($\textbf{SMEdit}$), a novel MLBME method that adopts $\textbf{M}$ultiple $\textbf{B}$ackpro$\textbf{P}$agation $\textbf{S}$teps ($\textbf{MBPS}$) to improve editing performance under limited supervision and a norm regularization on weight updates to improve training efficiency. Experimental results on two datasets and two LLMs demonstrate that SMEdit outperforms prior MLBME baselines and the MBPS strategy can be seamlessly integrated into existing methods to further boost their performance. Our code will be released soon.
>
---
#### [new 021] FaST: Feature-aware Sampling and Tuning for Personalized Preference Alignment with Limited Data
- **分类: cs.CL**

- **简介: 该论文研究了如何通过有限数据集实现个性化偏好对齐，提出了FaST方法以提升模型性能。任务是解决LLM部署中的个体差异问题，解决了Personalized Preference Alignment with Limited Data（PPALLI）的挑战。**

- **链接: [http://arxiv.org/pdf/2508.04698v1](http://arxiv.org/pdf/2508.04698v1)**

> **作者:** Thibaut Thonet; Germán Kruszewski; Jos Rozen; Pierre Erbacher; Marc Dymetman
>
> **摘要:** LLM-powered conversational assistants are often deployed in a one-size-fits-all manner, which fails to accommodate individual user preferences. Recently, LLM personalization -- tailoring models to align with specific user preferences -- has gained increasing attention as a way to bridge this gap. In this work, we specifically focus on a practical yet challenging setting where only a small set of preference annotations can be collected per user -- a problem we define as Personalized Preference Alignment with Limited Data (PPALLI). To support research in this area, we introduce two datasets -- DnD and ELIP -- and benchmark a variety of alignment techniques on them. We further propose FaST, a highly parameter-efficient approach that leverages high-level features automatically discovered from the data, achieving the best overall performance.
>
---
#### [new 022] StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs through Knowledge-Reasoning Fusion
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文旨在解决LLMs在自然语言数学转化中的低准确性问题，通过构建知识-推理融合的数据训练框架，改进了知识理解与推理能力，最终在FormalMATH-Lite和ProverBench等基准任务上取得40.5%和26.7%的SOTA性能，实现了有效自动形式化。**

- **链接: [http://arxiv.org/pdf/2508.04440v1](http://arxiv.org/pdf/2508.04440v1)**

> **作者:** Yutong Wu; Di Huang; Ruosi Wan; Yue Peng; Shijie Shang; Chenrui Cao; Lei Qi; Rui Zhang; Zidong Du; Jie Yan; Xing Hu
>
> **备注:** 24 pages, 17 figures, under review
>
> **摘要:** Autoformalization aims to translate natural-language mathematical statements into a formal language. While LLMs have accelerated progress in this area, existing methods still suffer from low accuracy. We identify two key abilities for effective autoformalization: comprehensive mastery of formal-language domain knowledge, and reasoning capability of natural language problem understanding and informal-formal alignment. Without the former, a model cannot identify the correct formal objects; without the latter, it struggles to interpret real-world contexts and map them precisely into formal expressions. To address these gaps, we introduce ThinkingF, a data synthesis and training pipeline that improves both abilities. First, we construct two datasets: one by distilling and selecting large-scale examples rich in formal knowledge, and another by generating informal-to-formal reasoning trajectories guided by expert-designed templates. We then apply SFT and RLVR with these datasets to further fuse and refine the two abilities. The resulting 7B and 32B models exhibit both comprehensive formal knowledge and strong informal-to-formal reasoning. Notably, StepFun-Formalizer-32B achieves SOTA BEq@1 scores of 40.5% on FormalMATH-Lite and 26.7% on ProverBench, surpassing all prior general-purpose and specialized models.
>
---
#### [new 023] Unveiling the Landscape of Clinical Depression Assessment: From Behavioral Signatures to Psychiatric Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在构建临床抑郁评估的系统，解决现有方法数据受限和模型效率低的问题。通过C-MIND数据集（2年医院记录）收集行为、视听及神经信号，训练多个模型量化任务与模态作用，探索LLM与临床推理的关系，并提出改进方案提升诊断准确性。**

- **链接: [http://arxiv.org/pdf/2508.04531v1](http://arxiv.org/pdf/2508.04531v1)**

> **作者:** Zhuang Chen; Guanqun Bi; Wen Zhang; Jiawei Hu; Aoyun Wang; Xiyao Xiao; Kun Feng; Minlie Huang
>
> **摘要:** Depression is a widespread mental disorder that affects millions worldwide. While automated depression assessment shows promise, most studies rely on limited or non-clinically validated data, and often prioritize complex model design over real-world effectiveness. In this paper, we aim to unveil the landscape of clinical depression assessment. We introduce C-MIND, a clinical neuropsychiatric multimodal diagnosis dataset collected over two years from real hospital visits. Each participant completes three structured psychiatric tasks and receives a final diagnosis from expert clinicians, with informative audio, video, transcript, and functional near-infrared spectroscopy (fNIRS) signals recorded. Using C-MIND, we first analyze behavioral signatures relevant to diagnosis. We train a range of classical models to quantify how different tasks and modalities contribute to diagnostic performance, and dissect the effectiveness of their combinations. We then explore whether LLMs can perform psychiatric reasoning like clinicians and identify their clear limitations in realistic clinical settings. In response, we propose to guide the reasoning process with clinical expertise and consistently improves LLM diagnostic performance by up to 10% in Macro-F1 score. We aim to build an infrastructure for clinical depression assessment from both data and algorithmic perspectives, enabling C-MIND to facilitate grounded and reliable research for mental healthcare.
>
---
#### [new 024] Automated Generation of Curriculum-Aligned Multiple-Choice Questions for Malaysian Secondary Mathematics Using Generative AI
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在解决马来西亚低资源语言数学教学中高质多选题生成问题，通过Intelligent Prompting与RAG框架对比研究，验证了基于官方课程的系统有效性，提出可扩展的EdTech解决方案。**

- **链接: [http://arxiv.org/pdf/2508.04442v1](http://arxiv.org/pdf/2508.04442v1)**

> **作者:** Rohaizah Abdul Wahid; Muhamad Said Nizamuddin Nadim; Suliana Sulaiman; Syahmi Akmal Shaharudin; Muhammad Danial Jupikil; Iqqwan Jasman Su Azlan Su
>
> **摘要:** This paper addresses the critical need for scalable and high-quality educational assessment tools within the Malaysian education system. It highlights the potential of Generative AI (GenAI) while acknowledging the significant challenges of ensuring factual accuracy and curriculum alignment, especially for low-resource languages like Bahasa Melayu. This research introduces and compares four incremental pipelines for generating Form 1 Mathematics multiple-choice questions (MCQs) in Bahasa Melayu using OpenAI's GPT-4o. The methods range from non-grounded prompting (structured and basic) to Retrieval-Augmented Generation (RAG) approaches (one using the LangChain framework, one implemented manually). The system is grounded in official curriculum documents, including teacher-prepared notes and the yearly teaching plan (RPT). A dual-pronged automated evaluation framework is employed to assess the generated questions. Curriculum alignment is measured using Semantic Textual Similarity (STS) against the RPT, while contextual validity is verified through a novel RAG-based Question-Answering (RAG-QA) method. The results demonstrate that RAG-based pipelines significantly outperform non-grounded prompting methods, producing questions with higher curriculum alignment and factual validity. The study further analyzes the trade-offs between the ease of implementation of framework-based RAG and the fine-grained control offered by a manual pipeline. This work presents a validated methodology for generating curriculum-specific educational content in a low-resource language, introduces a symbiotic RAG-QA evaluation technique, and provides actionable insights for the development and deployment of practical EdTech solutions in Malaysia and similar regions.
>
---
#### [new 025] PAIRS: Parametric-Verified Adaptive Information Retrieval and Selection for Efficient RAG
- **分类: cs.CL**

- **简介: 该论文旨在改进检索增强生成（RAG）技术，解决传统系统效率低和相关性差的问题。通过引入参数与检索知识融合的框架，采用双重路径生成和DPR-AIS机制，实现了高效检索与精准选题，实验表明可降低检索成本25%并提升准确率。**

- **链接: [http://arxiv.org/pdf/2508.04057v1](http://arxiv.org/pdf/2508.04057v1)**

> **作者:** Wang Chen; Guanqiang Qi; Weikang Li; Yang Li; Deguo Xia; Jizhou Huang
>
> **摘要:** Retrieval-Augmented Generation (RAG) has become a cornerstone technique for enhancing large language models (LLMs) with external knowledge. However, current RAG systems face two critical limitations: (1) they inefficiently retrieve information for every query, including simple questions that could be resolved using the LLM's parametric knowledge alone, and (2) they risk retrieving irrelevant documents when queries contain sparse information signals. To address these gaps, we introduce Parametric-verified Adaptive Information Retrieval and Selection (PAIRS), a training-free framework that integrates parametric and retrieved knowledge to adaptively determine whether to retrieve and how to select external information. Specifically, PAIRS employs a dual-path generation mechanism: First, the LLM produces both a direct answer and a context-augmented answer using self-generated pseudo-context. When these outputs converge, PAIRS bypasses external retrieval entirely, dramatically improving the RAG system's efficiency. For divergent cases, PAIRS activates a dual-path retrieval (DPR) process guided by both the original query and self-generated contextual signals, followed by an Adaptive Information Selection (AIS) module that filters documents through weighted similarity to both sources. This simple yet effective approach can not only enhance efficiency by eliminating unnecessary retrievals but also improve accuracy through contextually guided retrieval and adaptive information selection. Experimental results on six question-answering (QA) benchmarks show that PAIRS reduces retrieval costs by around 25% (triggering for only 75% of queries) while still improving accuracy-achieving +1.1% EM and +1.0% F1 over prior baselines on average.
>
---
#### [new 026] Unveiling Over-Memorization in Finetuning LLMs for Reasoning Tasks
- **分类: cs.CL**

- **简介: 该论文研究了在推理任务中LLM因过度记忆训练数据导致的过拟合现象，发现训练周期和大学习率等因素影响了这一问题，提出通过调整参数提升模型泛化能力的研究方法。**

- **链接: [http://arxiv.org/pdf/2508.04117v1](http://arxiv.org/pdf/2508.04117v1)**

> **作者:** Zhiwen Ruan; Yun Chen; Yutao Hou; Peng Li; Yang Liu; Guanhua Chen
>
> **摘要:** The pretrained large language models (LLMs) are finetuned with labeled data for better instruction following ability and alignment with human values. In this paper, we study the learning dynamics of LLM finetuning on reasoning tasks and reveal the uncovered over-memorization phenomenon during a specific stage of LLM finetuning. At this stage, the LLMs have excessively memorized training data and exhibit high test perplexity while maintaining good test accuracy. We investigate the conditions that lead to LLM over-memorization and find that training epochs and large learning rates contribute to this issue. Although models with over-memorization demonstrate comparable test accuracy to normal models, they suffer from reduced robustness, poor out-of-distribution generalization, and decreased generation diversity. Our experiments unveil the over-memorization to be broadly applicable across different tasks, models, and finetuning methods. Our research highlights that overparameterized, extensively finetuned LLMs exhibit unique learning dynamics distinct from traditional machine learning models. Based on our observations of over-memorization, we provide recommendations on checkpoint and learning rate selection during finetuning.
>
---
#### [new 027] DP-GPT4MTS: Dual-Prompt Large Language Model for Textual-Numerical Time Series Forecasting
- **分类: cs.CL**

- **简介: 该论文提出DP-GPT4MTS（双提示多模态时间序列预测框架），解决传统单一提示模型无法融合文本与数值信息的问题，通过显式指令提示与文本上下文嵌入的联合优化，提升时间序列预测准确性，验证其优于现有算法。**

- **链接: [http://arxiv.org/pdf/2508.04239v1](http://arxiv.org/pdf/2508.04239v1)**

> **作者:** Chanjuan Liu; Shengzhi Wang; Enqiang Zhu
>
> **摘要:** Time series forecasting is crucial in strategic planning and decision-making across various industries. Traditional forecasting models mainly concentrate on numerical time series data, often overlooking important textual information such as events and news, which can significantly affect forecasting accuracy. While large language models offer a promise for integrating multimodal data, existing single-prompt frameworks struggle to effectively capture the semantics of timestamped text, introducing redundant information that can hinder model performance. To address this limitation, we introduce DP-GPT4MTS (Dual-Prompt GPT2-base for Multimodal Time Series), a novel dual-prompt large language model framework that combines two complementary prompts: an explicit prompt for clear task instructions and a textual prompt for context-aware embeddings from time-stamped data. The tokenizer generates the explicit prompt while the embeddings from the textual prompt are refined through self-attention and feed-forward networks. Comprehensive experiments conducted on diverse textural-numerical time series datasets demonstrate that this approach outperforms state-of-the-art algorithms in time series forecasting. This highlights the significance of incorporating textual context via a dual-prompt mechanism to achieve more accurate time series predictions.
>
---
#### [new 028] GeRe: Towards Efficient Anti-Forgetting in Continual Learning of LLM via General Samples Replay
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文旨在解决大规模语言模型（LLMs）的连续学习中由于" catastrophic forgetting "导致的遗忘问题。通过使用一般样本重放机制（GeRe），提出增强激活状态约束优化方法（TM），验证了小固定样本集可有效平衡通用能力保留与整体性能提升。**

- **链接: [http://arxiv.org/pdf/2508.04676v1](http://arxiv.org/pdf/2508.04676v1)**

> **作者:** Yunan Zhang; Shuoran Jiang; Mengchen Zhao; Yuefeng Li; Yang Fan; Xiangping Wu; Qingcai Chen
>
> **摘要:** The continual learning capability of large language models (LLMs) is crucial for advancing artificial general intelligence. However, continual fine-tuning LLMs across various domains often suffers from catastrophic forgetting, characterized by: 1) significant forgetting of their general capabilities, and 2) sharp performance declines in previously learned tasks. To simultaneously address both issues in a simple yet stable manner, we propose General Sample Replay (GeRe), a framework that use usual pretraining texts for efficient anti-forgetting. Beyond revisiting the most prevalent replay-based practices under GeRe, we further leverage neural states to introduce a enhanced activation states constrained optimization method using threshold-based margin (TM) loss, which maintains activation state consistency during replay learning. We are the first to validate that a small, fixed set of pre-collected general replay samples is sufficient to resolve both concerns--retaining general capabilities while promoting overall performance across sequential tasks. Indeed, the former can inherently facilitate the latter. Through controlled experiments, we systematically compare TM with different replay strategies under the GeRe framework, including vanilla label fitting, logit imitation via KL divergence and feature imitation via L1/L2 losses. Results demonstrate that TM consistently improves performance and exhibits better robustness. Our work paves the way for efficient replay of LLMs for the future. Our code and data are available at https://github.com/Qznan/GeRe.
>
---
#### [new 029] Lightweight Transformers for Zero-Shot and Fine-Tuned Text-to-SQL Generation Using Spider
- **分类: cs.CL; cs.IR; 68T50 % Natural language processing (in Computer Science); I.2.7; H.2.3**

- **简介: 该论文研究了轻量化Transformer模型（如T5-Small、BART-Small、GPT-2）在低资源场景下的文本到SQL推理任务，通过Spider数据集验证了其零样本与微调能力，提出了一种模块化框架以提升性能并解决资源约束问题。**

- **链接: [http://arxiv.org/pdf/2508.04623v1](http://arxiv.org/pdf/2508.04623v1)**

> **作者:** Chirag Seth; Utkarsh Singh
>
> **摘要:** Text-to-SQL translation enables non-expert users to query relational databases using natural language, with applications in education and business intelligence. This study evaluates three lightweight transformer models - T5-Small, BART-Small, and GPT-2 - on the Spider dataset, focusing on low-resource settings. We developed a reusable, model-agnostic pipeline that tailors schema formatting to each model's architecture, training them across 1000 to 5000 iterations and evaluating on 1000 test samples using Logical Form Accuracy (LFAcc), BLEU, and Exact Match (EM) metrics. Fine-tuned T5-Small achieves the highest LFAcc (27.8%), outperforming BART-Small (23.98%) and GPT-2 (20.1%), highlighting encoder-decoder models' superiority in schema-aware SQL generation. Despite resource constraints limiting performance, our pipeline's modularity supports future enhancements, such as advanced schema linking or alternative base models. This work underscores the potential of compact transformers for accessible text-to-SQL solutions in resource-scarce environments.
>
---
#### [new 030] HarmonyGuard: Toward Safety and Utility in Web Agents via Adaptive Policy Enhancement and Dual-Objective Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出了一种多代理协作框架HarmonyGuard，旨在解决Web代理在开放环境中平衡任务性能与潜在风险（安全）及实用价值（utility）的问题。通过引入政策增强器和双重目标优化器，实现策略协同优化，显著提升了政策合规性（38%）和任务完成率（20%），验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2508.04010v1](http://arxiv.org/pdf/2508.04010v1)**

> **作者:** Yurun Chen; Xavier Hu; Yuhan Liu; Keting Yin; Juncheng Li; Zhuosheng Zhang; Shengyu Zhang
>
> **摘要:** Large language models enable agents to autonomously perform tasks in open web environments. However, as hidden threats within the web evolve, web agents face the challenge of balancing task performance with emerging risks during long-sequence operations. Although this challenge is critical, current research remains limited to single-objective optimization or single-turn scenarios, lacking the capability for collaborative optimization of both safety and utility in web environments. To address this gap, we propose HarmonyGuard, a multi-agent collaborative framework that leverages policy enhancement and objective optimization to jointly improve both utility and safety. HarmonyGuard features a multi-agent architecture characterized by two fundamental capabilities: (1) Adaptive Policy Enhancement: We introduce the Policy Agent within HarmonyGuard, which automatically extracts and maintains structured security policies from unstructured external documents, while continuously updating policies in response to evolving threats. (2) Dual-Objective Optimization: Based on the dual objectives of safety and utility, the Utility Agent integrated within HarmonyGuard performs the Markovian real-time reasoning to evaluate the objectives and utilizes metacognitive capabilities for their optimization. Extensive evaluations on multiple benchmarks show that HarmonyGuard improves policy compliance by up to 38% and task completion by up to 20% over existing baselines, while achieving over 90% policy compliance across all tasks. Our project is available here: https://github.com/YurunChen/HarmonyGuard.
>
---
#### [new 031] Eliciting and Analyzing Emergent Misalignment in State-of-the-Art Large Language Models
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 本论文旨在分析LLM中涌现的误解，通过实验发现其对可控场景的脆弱性，并构建了MISALIGNMENTBENCH框架以验证攻击有效性。**

- **链接: [http://arxiv.org/pdf/2508.04196v1](http://arxiv.org/pdf/2508.04196v1)**

> **作者:** Siddhant Panpatil; Hiskias Dingeto; Haon Park
>
> **摘要:** Despite significant advances in alignment techniques, we demonstrate that state-of-the-art language models remain vulnerable to carefully crafted conversational scenarios that can induce various forms of misalignment without explicit jailbreaking. Through systematic manual red-teaming with Claude-4-Opus, we discovered 10 successful attack scenarios, revealing fundamental vulnerabilities in how current alignment methods handle narrative immersion, emotional pressure, and strategic framing. These scenarios successfully elicited a range of misaligned behaviors, including deception, value drift, self-preservation, and manipulative reasoning, each exploiting different psychological and contextual vulnerabilities. To validate generalizability, we distilled our successful manual attacks into MISALIGNMENTBENCH, an automated evaluation framework that enables reproducible testing across multiple models. Cross-model evaluation of our 10 scenarios against five frontier LLMs revealed an overall 76% vulnerability rate, with significant variations: GPT-4.1 showed the highest susceptibility (90%), while Claude-4-Sonnet demonstrated greater resistance (40%). Our findings demonstrate that sophisticated reasoning capabilities often become attack vectors rather than protective mechanisms, as models can be manipulated into complex justifications for misaligned behavior. This work provides (i) a detailed taxonomy of conversational manipulation patterns and (ii) a reusable evaluation framework. Together, these findings expose critical gaps in current alignment strategies and highlight the need for robustness against subtle, scenario-based manipulation in future AI systems.
>
---
#### [new 032] Majority Bit-Aware Watermarking For Large Language Models
- **分类: cs.CL; cs.CR**

- **简介: 该论文旨在解决大型语言模型（LLMs）生成内容可能被恶意利用的问题，提出MajorMark和MajorMark+两种方法，通过多数位编码优化水印采样范围并结合聚类解码策略，提升文本质量与误码率，有效应对现有多比特水印方案的局限性。**

- **链接: [http://arxiv.org/pdf/2508.03829v1](http://arxiv.org/pdf/2508.03829v1)**

> **作者:** Jiahao Xu; Rui Hu; Zikai Zhang
>
> **备注:** Preprint
>
> **摘要:** The growing deployment of Large Language Models (LLMs) in real-world applications has raised concerns about their potential misuse in generating harmful or deceptive content. To address this issue, watermarking techniques have emerged as a promising solution by embedding identifiable binary messages into generated text for origin verification and misuse tracing. While recent efforts have explored multi-bit watermarking schemes capable of embedding rich information such as user identifiers, they typically suffer from the fundamental trade-off between text quality and decoding accuracy: to ensure reliable message decoding, they have to restrict the size of preferred token sets during encoding, yet such restrictions reduce the quality of the generated content. In this work, we propose MajorMark, a novel watermarking method that improves this trade-off through majority bit-aware encoding. MajorMark selects preferred token sets based on the majority bit of the message, enabling a larger and more flexible sampling of tokens. In contrast to prior methods that rely on token frequency analysis for decoding, MajorMark employs a clustering-based decoding strategy, which maintains high decoding accuracy even when the preferred token set is large, thus preserving both content quality and decoding accuracy. We further introduce MajorMark$^+$, which partitions the message into multiple blocks to independently encode and deterministically decode each block, thereby further enhancing the quality of watermarked text and improving decoding accuracy. Extensive experiments on state-of-the-art LLMs demonstrate that our methods significantly enhance both decoding accuracy and text generation quality, outperforming prior multi-bit watermarking baselines.
>
---
#### [new 033] A Few Words Can Distort Graphs: Knowledge Poisoning Attacks on Graph-based Retrieval-Augmented Generation of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出两种知识中毒攻击（KPAs），旨在检测GraphRAG中的恶意信息注入问题，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.04276v1](http://arxiv.org/pdf/2508.04276v1)**

> **作者:** Jiayi Wen; Tianxin Chen; Zhirun Zheng; Cheng Huang
>
> **摘要:** Graph-based Retrieval-Augmented Generation (GraphRAG) has recently emerged as a promising paradigm for enhancing large language models (LLMs) by converting raw text into structured knowledge graphs, improving both accuracy and explainability. However, GraphRAG relies on LLMs to extract knowledge from raw text during graph construction, and this process can be maliciously manipulated to implant misleading information. Targeting this attack surface, we propose two knowledge poisoning attacks (KPAs) and demonstrate that modifying only a few words in the source text can significantly change the constructed graph, poison the GraphRAG, and severely mislead downstream reasoning. The first attack, named Targeted KPA (TKPA), utilizes graph-theoretic analysis to locate vulnerable nodes in the generated graphs and rewrites the corresponding narratives with LLMs, achieving precise control over specific question-answering (QA) outcomes with a success rate of 93.1\%, while keeping the poisoned text fluent and natural. The second attack, named Universal KPA (UKPA), exploits linguistic cues such as pronouns and dependency relations to disrupt the structural integrity of the generated graph by altering globally influential words. With fewer than 0.05\% of full text modified, the QA accuracy collapses from 95\% to 50\%. Furthermore, experiments show that state-of-the-art defense methods fail to detect these attacks, highlighting that securing GraphRAG pipelines against knowledge poisoning remains largely unexplored.
>
---
#### [new 034] Reasoning Beyond Labels: Measuring LLM Sentiment in Low-Resource, Culturally Nuanced Contexts
- **分类: cs.CL**

- **简介: 该论文旨在解决低资源、文化多元背景下LLM情感推理的评估问题，提出基于反事实与解释的诊断框架，通过混合评估技术探究模型的推理质量，并从社会科学视角验证其作为情感工具的可行性。**

- **链接: [http://arxiv.org/pdf/2508.04199v1](http://arxiv.org/pdf/2508.04199v1)**

> **作者:** Millicent Ochieng; Anja Thieme; Ignatius Ezeani; Risa Ueno; Samuel Maina; Keshet Ronen; Javier Gonzalez; Jacki O'Neill
>
> **摘要:** Sentiment analysis in low-resource, culturally nuanced contexts challenges conventional NLP approaches that assume fixed labels and universal affective expressions. We present a diagnostic framework that treats sentiment as a context-dependent, culturally embedded construct, and evaluate how large language models (LLMs) reason about sentiment in informal, code-mixed WhatsApp messages from Nairobi youth health groups. Using a combination of human-annotated data, sentiment-flipped counterfactuals, and rubric-based explanation evaluation, we probe LLM interpretability, robustness, and alignment with human reasoning. Framing our evaluation through a social-science measurement lens, we operationalize and interrogate LLMs outputs as an instrument for measuring the abstract concept of sentiment. Our findings reveal significant variation in model reasoning quality, with top-tier LLMs demonstrating interpretive stability, while open models often falter under ambiguity or sentiment shifts. This work highlights the need for culturally sensitive, reasoning-aware AI evaluation in complex, real-world communication.
>
---
#### [new 035] GM-PRM: A Generative Multimodal Process Reward Model for Multimodal Mathematical Reasoning
- **分类: cs.CL**

- **简介: 该论文提出一种基于生成式多模态过程奖励模型（GM-PRM），解决传统PRM在多步数学推理中的错误识别与修正能力不足问题。通过动态生成推理步骤的修正，改进了数学推理的准确性和可解释性，并在多个基准测试中取得优异性能，实现高效训练需求。**

- **链接: [http://arxiv.org/pdf/2508.04088v1](http://arxiv.org/pdf/2508.04088v1)**

> **作者:** Jianghangfan Zhang; Yibo Yan; Kening Zheng; Xin Zou; Song Dai; Xuming Hu
>
> **摘要:** Multimodal Large Language Models (MLLMs) demonstrate remarkable capabilities but often struggle with complex, multi-step mathematical reasoning, where minor errors in visual perception or logical deduction can lead to complete failure. While Process Reward Models (PRMs) offer step-by-step supervision, existing multimodal PRMs are limited to being binary verifiers that can identify but not correct errors, offering little explanatory power. To address these deficiencies, we introduce the Generative Multimodal Process Reward Model (GM-PRM), a novel paradigm that transforms the PRM from a passive judge into an active reasoning collaborator. Instead of a simple scalar score, GM-PRM provides a fine-grained, interpretable analysis of each reasoning step, evaluating its step intent, visual alignment, and logical soundness. More critically, GM-PRM is trained to generate a corrected version of the first erroneous step it identifies. This unique corrective capability enables our new test-time inference strategy, Refined Best-of-N (Refined-BoN). This framework actively enhances solution quality by using the PRM's generated correction to guide the policy model toward a more promising reasoning trajectory, thereby improving the diversity and correctness of the solution pool. We demonstrate that GM-PRM achieves state-of-the-art results on multiple multimodal math benchmarks, significantly boosting policy model performance with remarkable data efficiency, requiring only a 20K-sample training dataset. Our code will be released upon acceptance.
>
---
#### [new 036] Why are LLMs' abilities emergent?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LMM能力的涌现性，探讨DNN如何通过非线性、随机过程实现复杂系统行为，揭示其本质机制与传统符号计算的不同，旨在解决"创造无理解释"问题，论证DNN的涌现特性与自然系统相似。**

- **链接: [http://arxiv.org/pdf/2508.04401v1](http://arxiv.org/pdf/2508.04401v1)**

> **作者:** Vladimír Havlík
>
> **备注:** 20 pages
>
> **摘要:** The remarkable success of Large Language Models (LLMs) in generative tasks has raised fundamental questions about the nature of their acquired capabilities, which often appear to emerge unexpectedly without explicit training. This paper examines the emergent properties of Deep Neural Networks (DNNs) through both theoretical analysis and empirical observation, addressing the epistemological challenge of "creation without understanding" that characterises contemporary AI development. We explore how the neural approach's reliance on nonlinear, stochastic processes fundamentally differs from symbolic computational paradigms, creating systems whose macro-level behaviours cannot be analytically derived from micro-level neuron activities. Through analysis of scaling laws, grokking phenomena, and phase transitions in model capabilities, I demonstrate that emergent abilities arise from the complex dynamics of highly sensitive nonlinear systems rather than simply from parameter scaling alone. My investigation reveals that current debates over metrics, pre-training loss thresholds, and in-context learning miss the fundamental ontological nature of emergence in DNNs. I argue that these systems exhibit genuine emergent properties analogous to those found in other complex natural phenomena, where systemic capabilities emerge from cooperative interactions among simple components without being reducible to their individual behaviours. The paper concludes that understanding LLM capabilities requires recognising DNNs as a new domain of complex dynamical systems governed by universal principles of emergence, similar to those operating in physics, chemistry, and biology. This perspective shifts the focus from purely phenomenological definitions of emergence to understanding the internal dynamic transformations that enable these systems to acquire capabilities that transcend their individual components.
>
---
#### [new 037] Modelling and Classifying the Components of a Literature Review
- **分类: cs.CL; cs.AI; cs.HC; cs.IR**

- **简介: 该论文旨在构建文献综述建模框架并分类文本中的 rhetorical roles，解决如何高效标注和分类学术文献挑战，通过开发Novel Annotation Schema及Sci-Sentence基准测试，评估多种LLMs的性能，并验证其在大规模数据训练下的有效性。**

- **链接: [http://arxiv.org/pdf/2508.04337v1](http://arxiv.org/pdf/2508.04337v1)**

> **作者:** Francisco Bolaños; Angelo Salatino; Francesco Osborne; Enrico Motta
>
> **摘要:** Previous work has demonstrated that AI methods for analysing scientific literature benefit significantly from annotating sentences in papers according to their rhetorical roles, such as research gaps, results, limitations, extensions of existing methodologies, and others. Such representations also have the potential to support the development of a new generation of systems capable of producing high-quality literature reviews. However, achieving this goal requires the definition of a relevant annotation schema and effective strategies for large-scale annotation of the literature. This paper addresses these challenges by 1) introducing a novel annotation schema specifically designed to support literature review generation and 2) conducting a comprehensive evaluation of a wide range of state-of-the-art large language models (LLMs) in classifying rhetorical roles according to this schema. To this end, we also present Sci-Sentence, a novel multidisciplinary benchmark comprising 700 sentences manually annotated by domain experts and 2,240 sentences automatically labelled using LLMs. We evaluate 37 LLMs on this benchmark, spanning diverse model families and sizes, using both zero-shot learning and fine-tuning approaches. The experiments yield several novel insights that advance the state of the art in this challenging domain. First, the current generation of LLMs performs remarkably well on this task when fine-tuned on high-quality data, achieving performance levels above 96\% F1. Second, while large proprietary models like GPT-4o achieve the best results, some lightweight open-source alternatives also demonstrate excellent performance. Finally, enriching the training data with semi-synthetic examples generated by LLMs proves beneficial, enabling small encoders to achieve robust results and significantly enhancing the performance of several open decoder models.
>
---
#### [new 038] CAP-LLM: Context-Augmented Personalized Large Language Models for News Headline Generation
- **分类: cs.CL**

- **简介: 该论文旨在解决信息过载时代个性化新闻标题生成中的事实准确性与用户兴趣匹配问题，提出CAP-LLM框架通过用户偏好编码器、事实约束适配器和强化模块提升模型性能，达到87.50 FactCC和2.73~17.25的个性化程度，验证其在新闻标题生成中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.03935v1](http://arxiv.org/pdf/2508.03935v1)**

> **作者:** Raymond Wilson; Cole Graham; Chase Carter; Zefeng Yang; Ruiqi Gu
>
> **摘要:** In the era of information overload, personalized news headline generation is crucial for engaging users by tailoring content to their preferences while accurately conveying news facts. Existing methods struggle with effectively capturing complex user interests and ensuring factual consistency, often leading to generic or misleading headlines. Leveraging the unprecedented capabilities of Large Language Models (LLMs) in text generation, we propose Context-Augmented Personalized LLM (CAP-LLM), a novel framework that integrates user preferences and factual consistency constraints into a powerful pre-trained LLM backbone. CAP-LLM features a User Preference Encoder to capture long-term user interests, a Context Injection Adapter to seamlessly integrate these preferences and current article context into the LLM's generation process, and a Fact-Consistency Reinforcement Module employing a novel contrastive loss to mitigate hallucination. Evaluated on the real-world PENS dataset, CAP-LLM achieves state-of-the-art performance across all metrics. Notably, it significantly improves factual consistency (FactCC of 87.50) over strong baselines like BART (86.67), while simultaneously enhancing personalization (Pc(avg) 2.73, Pc(max) 17.25) and content coverage (ROUGE-1 26.55, ROUGE-2 9.95, ROUGE-L 23.01). Our ablation studies, human evaluations, and sensitivity analyses further validate the effectiveness of each component and the robustness of our approach, demonstrating CAP-LLM's ability to achieve a superior balance between personalization and factual accuracy in news headline generation.
>
---
#### [new 039] Large Reasoning Models Are Autonomous Jailbreak Agents
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文旨在探索大型推理模型（LRMs）如何通过自主行为实现低成本的AI安全破译，解决传统安全机制难以被普通用户操作的问题，验证其在多轮对话中的攻击能力并揭示安全风险关联性。**

- **链接: [http://arxiv.org/pdf/2508.04039v1](http://arxiv.org/pdf/2508.04039v1)**

> **作者:** Thilo Hagendorff; Erik Derner; Nuria Oliver
>
> **摘要:** Jailbreaking -- bypassing built-in safety mechanisms in AI models -- has traditionally required complex technical procedures or specialized human expertise. In this study, we show that the persuasive capabilities of large reasoning models (LRMs) simplify and scale jailbreaking, converting it into an inexpensive activity accessible to non-experts. We evaluated the capabilities of four LRMs (DeepSeek-R1, Gemini 2.5 Flash, Grok 3 Mini, Qwen3 235B) to act as autonomous adversaries conducting multi-turn conversations with nine widely used target models. LRMs received instructions via a system prompt, before proceeding to planning and executing jailbreaks with no further supervision. We performed extensive experiments with a benchmark of harmful prompts composed of 70 items covering seven sensitive domains. This setup yielded an overall attack success rate across all model combinations of 97.14%. Our study reveals an alignment regression, in which LRMs can systematically erode the safety guardrails of other models, highlighting the urgent need to further align frontier models not only to resist jailbreak attempts, but also to prevent them from being co-opted into acting as jailbreak agents.
>
---
#### [new 040] How Deep Is Representational Bias in LLMs? The Cases of Caste and Religion
- **分类: cs.CL**

- **简介: 本研究探讨了LLM中代表偏见的深度，特别是针对印度宗教与阶级议题。通过分析GPT-4 Turbo生成内容发现其在文化主导群体上的过度代表，质疑传统训练数据影响并提出需改进模型开发以消除偏见。**

- **链接: [http://arxiv.org/pdf/2508.03712v1](http://arxiv.org/pdf/2508.03712v1)**

> **作者:** Agrima Seth; Monojit Choudhary; Sunayana Sitaram; Kentaro Toyama; Aditya Vashistha; Kalika Bali
>
> **备注:** Accepted to AIES 2025
>
> **摘要:** Representational bias in large language models (LLMs) has predominantly been measured through single-response interactions and has focused on Global North-centric identities like race and gender. We expand on that research by conducting a systematic audit of GPT-4 Turbo to reveal how deeply encoded representational biases are and how they extend to less-explored dimensions of identity. We prompt GPT-4 Turbo to generate over 7,200 stories about significant life events (such as weddings) in India, using prompts designed to encourage diversity to varying extents. Comparing the diversity of religious and caste representation in the outputs against the actual population distribution in India as recorded in census data, we quantify the presence and "stickiness" of representational bias in the LLM for religion and caste. We find that GPT-4 responses consistently overrepresent culturally dominant groups far beyond their statistical representation, despite prompts intended to encourage representational diversity. Our findings also suggest that representational bias in LLMs has a winner-take-all quality that is more biased than the likely distribution bias in their training data, and repeated prompt-based nudges have limited and inconsistent efficacy in dislodging these biases. These results suggest that diversifying training data alone may not be sufficient to correct LLM bias, highlighting the need for more fundamental changes in model development. Dataset and Codebook: https://github.com/agrimaseth/How-Deep-Is-Representational-Bias-in-LLMs
>
---
#### [new 041] The State Of TTS: A Case Study with Human Fooling Rates
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文旨在探讨TTS系统在人类欺骗测试中的表现，提出HFR指标并分析其有效性，指出当前模型在自然对话等真实语境下仍存在不足，建议优化评估标准以提高准确性。**

- **链接: [http://arxiv.org/pdf/2508.04179v1](http://arxiv.org/pdf/2508.04179v1)**

> **作者:** Praveen Srinivasa Varadhan; Sherry Thomas; Sai Teja M. S.; Suvrat Bhooshan; Mitesh M. Khapra
>
> **备注:** Accepted at InterSpeech 2025
>
> **摘要:** While subjective evaluations in recent years indicate rapid progress in TTS, can current TTS systems truly pass a human deception test in a Turing-like evaluation? We introduce Human Fooling Rate (HFR), a metric that directly measures how often machine-generated speech is mistaken for human. Our large-scale evaluation of open-source and commercial TTS models reveals critical insights: (i) CMOS-based claims of human parity often fail under deception testing, (ii) TTS progress should be benchmarked on datasets where human speech achieves high HFRs, as evaluating against monotonous or less expressive reference samples sets a low bar, (iii) Commercial models approach human deception in zero-shot settings, while open-source systems still struggle with natural conversational speech; (iv) Fine-tuning on high-quality data improves realism but does not fully bridge the gap. Our findings underscore the need for more realistic, human-centric evaluations alongside existing subjective tests.
>
---
#### [new 042] TalkDep: Clinically Grounded LLM Personas for Conversation-Centric Depression Screening
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在通过构建基于语言模型的临床虚拟患者系统（TalkDep）解决抑郁症筛查数据不足问题，开发可生成自然症状表现的模拟患者，验证其可靠性以支持自动诊断系统训练。**

- **链接: [http://arxiv.org/pdf/2508.04248v1](http://arxiv.org/pdf/2508.04248v1)**

> **作者:** Xi Wang; Anxo Perez; Javier Parapar; Fabio Crestani
>
> **备注:** Paper accepted at CIKM 2025
>
> **摘要:** The increasing demand for mental health services has outpaced the availability of real training data to develop clinical professionals, leading to limited support for the diagnosis of depression. This shortage has motivated the development of simulated or virtual patients to assist in training and evaluation, but existing approaches often fail to generate clinically valid, natural, and diverse symptom presentations. In this work, we embrace the recent advanced language models as the backbone and propose a novel clinician-in-the-loop patient simulation pipeline, TalkDep, with access to diversified patient profiles to develop simulated patients. By conditioning the model on psychiatric diagnostic criteria, symptom severity scales, and contextual factors, our goal is to create authentic patient responses that can better support diagnostic model training and evaluation. We verify the reliability of these simulated patients with thorough assessments conducted by clinical professionals. The availability of validated simulated patients offers a scalable and adaptable resource for improving the robustness and generalisability of automatic depression diagnosis systems.
>
---
#### [new 043] AttnTrace: Attention-based Context Traceback for Long-Context LLMs
- **分类: cs.CL; cs.CR**

- **简介: 该论文提出AttnTrace方法，利用LLM的注意力权重进行上下文追溯，解决了传统方案高计算成本的问题，提升了对长上下文推理的准确性和效率，并应用于RAG和自主系统，通过增强注意力机制优化方案。**

- **链接: [http://arxiv.org/pdf/2508.03793v1](http://arxiv.org/pdf/2508.03793v1)**

> **作者:** Yanting Wang; Runpeng Geng; Ying Chen; Jinyuan Jia
>
> **备注:** The code is available at https://github.com/Wang-Yanting/AttnTrace. The demo is available at https://huggingface.co/spaces/SecureLLMSys/AttnTrace
>
> **摘要:** Long-context large language models (LLMs), such as Gemini-2.5-Pro and Claude-Sonnet-4, are increasingly used to empower advanced AI systems, including retrieval-augmented generation (RAG) pipelines and autonomous agents. In these systems, an LLM receives an instruction along with a context--often consisting of texts retrieved from a knowledge database or memory--and generates a response that is contextually grounded by following the instruction. Recent studies have designed solutions to trace back to a subset of texts in the context that contributes most to the response generated by the LLM. These solutions have numerous real-world applications, including performing post-attack forensic analysis and improving the interpretability and trustworthiness of LLM outputs. While significant efforts have been made, state-of-the-art solutions such as TracLLM often lead to a high computation cost, e.g., it takes TracLLM hundreds of seconds to perform traceback for a single response-context pair. In this work, we propose AttnTrace, a new context traceback method based on the attention weights produced by an LLM for a prompt. To effectively utilize attention weights, we introduce two techniques designed to enhance the effectiveness of AttnTrace, and we provide theoretical insights for our design choice. We also perform a systematic evaluation for AttnTrace. The results demonstrate that AttnTrace is more accurate and efficient than existing state-of-the-art context traceback methods. We also show that AttnTrace can improve state-of-the-art methods in detecting prompt injection under long contexts through the attribution-before-detection paradigm. As a real-world application, we demonstrate that AttnTrace can effectively pinpoint injected instructions in a paper designed to manipulate LLM-generated reviews. The code is at https://github.com/Wang-Yanting/AttnTrace.
>
---
#### [new 044] P-Aligner: Enabling Pre-Alignment of Language Models via Principled Instruction Synthesis
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在解决大型语言模型在接收模糊指令时难以保持意图和语境的问题，提出P-Aligner模块通过生成更人性化的预对齐指令来提升其安全性与准确性。研究基于UltraPrompt数据集，结合蒙特卡洛树搜索优化指令空间，并验证其在GPT-4-turbo等基准中的性能优势。**

- **链接: [http://arxiv.org/pdf/2508.04626v1](http://arxiv.org/pdf/2508.04626v1)**

> **作者:** Feifan Song; Bofei Gao; Yifan Song; Yi Liu; Weimin Xiong; Yuyang Song; Tianyu Liu; Guoyin Wang; Houfeng Wang
>
> **摘要:** Large Language Models (LLMs) are expected to produce safe, helpful, and honest content during interaction with human users, but they frequently fail to align with such values when given flawed instructions, e.g., missing context, ambiguous directives, or inappropriate tone, leaving substantial room for improvement along multiple dimensions. A cost-effective yet high-impact way is to pre-align instructions before the model begins decoding. Existing approaches either rely on prohibitive test-time search costs or end-to-end model rewrite, which is powered by a customized training corpus with unclear objectives. In this work, we demonstrate that the goal of efficient and effective preference alignment can be achieved by P-Aligner, a lightweight module generating instructions that preserve the original intents while being expressed in a more human-preferred form. P-Aligner is trained on UltraPrompt, a new dataset synthesized via a proposed principle-guided pipeline using Monte-Carlo Tree Search, which systematically explores the space of candidate instructions that are closely tied to human preference. Experiments across different methods show that P-Aligner generally outperforms strong baselines across various models and benchmarks, including average win-rate gains of 28.35% and 8.69% on GPT-4-turbo and Gemma-2-SimPO, respectively. Further analyses validate its effectiveness and efficiency through multiple perspectives, including data quality, search strategies, iterative deployment, and time overhead.
>
---
#### [new 045] FeynTune: Large Language Models for High-Energy Theory
- **分类: cs.CL; cs.LG; hep-th**

- **简介: 该论文提出用于高能理论的语言模型，解决生成理论相关摘要的任务，通过多领域训练和两种低秩方法实现超越商用LLMs的效果。**

- **链接: [http://arxiv.org/pdf/2508.03716v1](http://arxiv.org/pdf/2508.03716v1)**

> **作者:** Paul Richmond; Prarit Agarwal; Borun Chowdhury; Vasilis Niarchos; Constantinos Papageorgakis
>
> **备注:** 16 pages
>
> **摘要:** We present specialized Large Language Models for theoretical High-Energy Physics, obtained as 20 fine-tuned variants of the 8-billion parameter Llama-3.1 model. Each variant was trained on arXiv abstracts (through August 2024) from different combinations of hep-th, hep-ph and gr-qc. For a comparative study, we also trained models on datasets that contained abstracts from disparate fields such as the q-bio and cs categories. All models were fine-tuned using two distinct Low-Rank Adaptation fine-tuning approaches and varying dataset sizes, and outperformed the base model on hep-th abstract completion tasks. We compare performance against leading commercial LLMs (ChatGPT, Claude, Gemini, DeepSeek) and derive insights for further developing specialized language models for High-Energy Theoretical Physics.
>
---
#### [new 046] Multi-module GRPO: Composing Policy Gradients and Prompt Optimization for Language Model Programs
- **分类: cs.CL**

- **简介: 该论文旨在解决模块化语言模型（LLM）中的GRPO应用不足问题，提出mmGRPO框架，将多个LM调用按模块分组并处理中断轨迹，通过自动提示优化提升模型性能，验证其在分类、搜索和隐私任务上的有效性。**

- **链接: [http://arxiv.org/pdf/2508.04660v1](http://arxiv.org/pdf/2508.04660v1)**

> **作者:** Noah Ziems; Dilara Soylu; Lakshya A Agrawal; Isaac Miller; Liheng Lai; Chen Qian; Kaiqiang Song; Meng Jiang; Dan Klein; Matei Zaharia; Karel D'Oosterlinck; Christopher Potts; Omar Khattab
>
> **摘要:** Group Relative Policy Optimization (GRPO) has proven to be an effective tool for post-training language models (LMs). However, AI systems are increasingly expressed as modular programs that mix together multiple LM calls with distinct prompt templates and other tools, and it is not clear how best to leverage GRPO to improve these systems. We begin to address this challenge by defining mmGRPO, a simple multi-module generalization of GRPO that groups LM calls by module across rollouts and handles variable-length and interrupted trajectories. We find that mmGRPO, composed with automatic prompt optimization, improves accuracy by 11% on average across classification, many-hop search, and privacy-preserving delegation tasks against the post-trained LM, and by 5% against prompt optimization on its own. We open-source mmGRPO in DSPy as the dspy.GRPO optimizer.
>
---
#### [new 047] What Do Humans Hear When Interacting? Experiments on Selective Listening for Evaluating ASR of Spoken Dialogue Systems
- **分类: cs.CL**

- **简介: 该论文探讨人类在对话系统中的听觉选择性对ASR能力评估的作用，旨在验证人类对关键信息的聚焦能力，解决现有ASR系统与人类能力差异的评估问题，并提出基于人类选择性听力的新评估方法。**

- **链接: [http://arxiv.org/pdf/2508.04402v1](http://arxiv.org/pdf/2508.04402v1)**

> **作者:** Kiyotada Mori; Seiya Kawano; Chaoran Liu; Carlos Toshinori Ishi; Angel Fernando Garcia Contreras; Koichiro Yoshino
>
> **摘要:** Spoken dialogue systems (SDSs) utilize automatic speech recognition (ASR) at the front end of their pipeline. The role of ASR in SDSs is to recognize information in user speech related to response generation appropriately. Examining selective listening of humans, which refers to the ability to focus on and listen to important parts of a conversation during the speech, will enable us to identify the ASR capabilities required for SDSs and evaluate them. In this study, we experimentally confirmed selective listening when humans generate dialogue responses by comparing human transcriptions for generating dialogue responses and reference transcriptions. Based on our experimental results, we discuss the possibility of a new ASR evaluation method that leverages human selective listening, which can identify the gap between transcription ability between ASR systems and humans.
>
---
#### [new 048] Sculptor: Empowering LLMs with Cognitive Agency via Active Context Management
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出了一种通过主动上下文管理技术增强LLMs的认知能力的方法，解决长文本处理中的性能下降问题。研究设计了一套包含碎片化、总结、隐藏与恢复等工具的框架，验证了其在信息稀疏数据集上的有效性，表明主动控制而非固定窗口长度是提升模型鲁棒性的关键。**

- **链接: [http://arxiv.org/pdf/2508.04664v1](http://arxiv.org/pdf/2508.04664v1)**

> **作者:** Mo Li; L. H. Xu; Qitai Tan; Ting Cao; Yunxin Liu
>
> **备注:** Preprint. Work in progress
>
> **摘要:** Large Language Models (LLMs) suffer from significant performance degradation when processing long contexts due to proactive interference, where irrelevant information in earlier parts of the context disrupts reasoning and memory recall. While most research focuses on external memory systems to augment LLMs' capabilities, we propose a complementary approach: empowering LLMs with Active Context Management (ACM) tools to actively sculpt their internal working memory. We introduce Sculptor, a framework that equips LLMs with three categories of tools: (1) context fragmentation, (2) summary, hide, and restore, and (3) intelligent search. Our approach enables LLMs to proactively manage their attention and working memory, analogous to how humans selectively focus on relevant information while filtering out distractions. Experimental evaluation on information-sparse benchmarks-PI-LLM (proactive interference) and NeedleBench Multi-Needle Reasoning-demonstrates that Sculptor significantly improves performance even without specific training, leveraging LLMs' inherent tool calling generalization capabilities. By enabling Active Context Management, Sculptor not only mitigates proactive interference but also provides a cognitive foundation for more reliable reasoning across diverse long-context tasks-highlighting that explicit context-control strategies, rather than merely larger token windows, are key to robustness at scale.
>
---
#### [new 049] Can NLP Tackle Hate Speech in the Real World? Stakeholder-Informed Feedback and Survey on Counterspeech
- **分类: cs.CL**

- **简介: 该论文探讨了NLP在应对网络暴力中的应用，分析了其研究趋势与实际需求的脱节，并提出了基于社区参与的改进方案。**

- **链接: [http://arxiv.org/pdf/2508.04638v1](http://arxiv.org/pdf/2508.04638v1)**

> **作者:** Tanvi Dinkar; Aiqi Jiang; Simona Frenda; Poppy Gerrard-Abbott; Nancie Gunson; Gavin Abercrombie; Ioannis Konstas
>
> **摘要:** Counterspeech, i.e. the practice of responding to online hate speech, has gained traction in NLP as a promising intervention. While early work emphasised collaboration with non-governmental organisation stakeholders, recent research trends have shifted toward automated pipelines that reuse a small set of legacy datasets, often without input from affected communities. This paper presents a systematic review of 74 NLP studies on counterspeech, analysing the extent to which stakeholder participation influences dataset creation, model development, and evaluation. To complement this analysis, we conducted a participatory case study with five NGOs specialising in online Gender-Based Violence (oGBV), identifying stakeholder-informed practices for counterspeech generation. Our findings reveal a growing disconnect between current NLP research and the needs of communities most impacted by toxic online content. We conclude with concrete recommendations for re-centring stakeholder expertise in counterspeech research.
>
---
#### [new 050] IFDECORATOR: Wrapping Instruction Following Reinforcement Learning with Verifiable Rewards
- **分类: cs.CL**

- **简介: The paper introduces IFDecorator, a framework for RLVR that enhances instruction following by integrating cooperative adversarial data, intent alignment, and reward hacking detection. It reduces training inefficiency and over-optimization while achieving 87.43% accuracy on IFEval and significant improvement on FollowBench, with trip wires detecting reward hacking.**

- **链接: [http://arxiv.org/pdf/2508.04632v1](http://arxiv.org/pdf/2508.04632v1)**

> **作者:** Xu Guo; Tianyi Liang; Tong Jian; Xiaogui Yang; Ling-I Wu; Chenhui Li; Zhihui Lu; Qipeng Guo; Kai Chen
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) improves instruction following capabilities of large language models (LLMs), but suffers from training inefficiency due to inadequate difficulty assessment. Moreover, RLVR is prone to over-optimization, where LLMs exploit verification shortcuts without aligning to the actual intent of user instructions. We introduce Instruction Following Decorator (IFDecorator}, a framework that wraps RLVR training into a robust and sample-efficient pipeline. It consists of three components: (1) a cooperative-adversarial data flywheel that co-evolves instructions and hybrid verifications, generating progressively more challenging instruction-verification pairs; (2) IntentCheck, a bypass module enforcing intent alignment; and (3) trip wires, a diagnostic mechanism that detects reward hacking via trap instructions, which trigger and capture shortcut exploitation behaviors. Our Qwen2.5-32B-Instruct-IFDecorator achieves 87.43% accuracy on IFEval, outperforming larger proprietary models such as GPT-4o. Additionally, we demonstrate substantial improvements on FollowBench while preserving general capabilities. Our trip wires show significant reductions in reward hacking rates. We will release models, code, and data for future research.
>
---
#### [new 051] Hierarchical Verification of Speculative Beams for Accelerating LLM Inference
- **分类: cs.CL**

- **简介: 该论文旨在解决大规模语言模型推理效率低下的问题，提出Hierarchical Verification Tree（HVT）框架，通过优先验证高概率分支并实现早期剪枝，提升推理效率与准确性，同时保持模型性能。**

- **链接: [http://arxiv.org/pdf/2508.03726v1](http://arxiv.org/pdf/2508.03726v1)**

> **作者:** Jaydip Sen; Harshitha Puvvala; Subhasis Dasgupta
>
> **备注:** This paper was accepted for oral presentation and publication in the 3rd International Conference on Data Science and Network Engineering (ICDSNE 2025), organized at NIT, Agartala, India, from July 25 to 26, 2025. The paper is 12 pages long, and it contains 3 tables and 4 figures. This is NOT the final paper, which will be published in the Springer-published proceedings
>
> **摘要:** Large language models (LLMs) have achieved remarkable success across diverse natural language processing tasks but face persistent challenges in inference efficiency due to their autoregressive nature. While speculative decoding and beam sampling offer notable improvements, traditional methods verify draft sequences sequentially without prioritization, leading to unnecessary computational overhead. This work proposes the Hierarchical Verification Tree (HVT), a novel framework that restructures speculative beam decoding by prioritizing high-likelihood drafts and enabling early pruning of suboptimal candidates. Theoretical foundations and a formal verification-pruning algorithm are developed to ensure correctness and efficiency. Integration with standard LLM inference pipelines is achieved without requiring retraining or architecture modification. Experimental evaluations across multiple datasets and models demonstrate that HVT consistently outperforms existing speculative decoding schemes, achieving substantial reductions in inference time and energy consumption while maintaining or enhancing output quality. The findings highlight the potential of hierarchical verification strategies as a new direction for accelerating large language model inference.
>
---
#### [new 052] Hallucination to Truth: A Review of Fact-Checking and Factuality Evaluation in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文旨在探讨大型语言模型（LLMs）在生成内容时如何评估其事实准确性，解决虚假信息生成与验证的核心问题，并通过分析挑战、提出改进方法及研究框架，推动构建更可靠的事实检查系统。**

- **链接: [http://arxiv.org/pdf/2508.03860v1](http://arxiv.org/pdf/2508.03860v1)**

> **作者:** Subhey Sadi Rahman; Md. Adnanul Islam; Md. Mahbub Alam; Musarrat Zeba; Md. Abdur Rahman; Sadia Sultana Chowa; Mohaimenul Azam Khan Raiaan; Sami Azam
>
> **备注:** 30 pages, 11 figures, 6 tables. Submitted to Artificial Intelligence Review for peer review
>
> **摘要:** Large Language Models (LLMs) are trained on vast and diverse internet corpora that often include inaccurate or misleading content. Consequently, LLMs can generate misinformation, making robust fact-checking essential. This review systematically analyzes how LLM-generated content is evaluated for factual accuracy by exploring key challenges such as hallucinations, dataset limitations, and the reliability of evaluation metrics. The review emphasizes the need for strong fact-checking frameworks that integrate advanced prompting strategies, domain-specific fine-tuning, and retrieval-augmented generation (RAG) methods. It proposes five research questions that guide the analysis of the recent literature from 2020 to 2025, focusing on evaluation methods and mitigation techniques. The review also discusses the role of instruction tuning, multi-agent reasoning, and external knowledge access via RAG frameworks. Key findings highlight the limitations of current metrics, the value of grounding outputs with validated external evidence, and the importance of domain-specific customization to improve factual consistency. Overall, the review underlines the importance of building LLMs that are not only accurate and explainable but also tailored for domain-specific fact-checking. These insights contribute to the advancement of research toward more trustworthy and context-aware language models.
>
---
#### [new 053] ZARA: Zero-shot Motion Time-Series Analysis via Knowledge and Retrieval Driven LLM Agents
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出ZARA框架，用于零样本运动时间序列分析，解决传统方法因固定活动集限制和缺乏解释性的问题，通过自动推导特征知识、多传感器检索和层级代理实现灵活、可解释的HAR分析，取得SOTA性能。**

- **链接: [http://arxiv.org/pdf/2508.04038v1](http://arxiv.org/pdf/2508.04038v1)**

> **作者:** Zechen Li; Baiyu Chen; Hao Xue; Flora D. Salim
>
> **摘要:** Motion sensor time-series are central to human activity recognition (HAR), with applications in health, sports, and smart devices. However, existing methods are trained for fixed activity sets and require costly retraining when new behaviours or sensor setups appear. Recent attempts to use large language models (LLMs) for HAR, typically by converting signals into text or images, suffer from limited accuracy and lack verifiable interpretability. We propose ZARA, the first agent-based framework for zero-shot, explainable HAR directly from raw motion time-series. ZARA integrates an automatically derived pair-wise feature knowledge base that captures discriminative statistics for every activity pair, a multi-sensor retrieval module that surfaces relevant evidence, and a hierarchical agent pipeline that guides the LLM to iteratively select features, draw on this evidence, and produce both activity predictions and natural-language explanations. ZARA enables flexible and interpretable HAR without any fine-tuning or task-specific classifiers. Extensive experiments on 8 HAR benchmarks show that ZARA achieves SOTA zero-shot performance, delivering clear reasoning while exceeding the strongest baselines by 2.53x in macro F1. Ablation studies further confirm the necessity of each module, marking ZARA as a promising step toward trustworthy, plug-and-play motion time-series analysis. Our codes are available at https://github.com/zechenli03/ZARA.
>
---
#### [new 054] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MM**

- **简介: 该论文旨在解决医疗基准可靠性不足的问题，提出MedCheck框架通过构建5个阶段的评估体系，系统改进AI医疗应用的临床适配度、数据安全性和安全性指标，推动标准化评估实践。**

- **链接: [http://arxiv.org/pdf/2508.04325v1](http://arxiv.org/pdf/2508.04325v1)**

> **作者:** Zizhan Ma; Wenxuan Wang; Guo Yu; Yiu-Fai Cheung; Meidan Ding; Jie Liu; Wenting Chen; Linlin Shen
>
> **摘要:** Large language models (LLMs) show significant potential in healthcare, prompting numerous benchmarks to evaluate their capabilities. However, concerns persist regarding the reliability of these benchmarks, which often lack clinical fidelity, robust data management, and safety-oriented evaluation metrics. To address these shortcomings, we introduce MedCheck, the first lifecycle-oriented assessment framework specifically designed for medical benchmarks. Our framework deconstructs a benchmark's development into five continuous stages, from design to governance, and provides a comprehensive checklist of 46 medically-tailored criteria. Using MedCheck, we conducted an in-depth empirical evaluation of 53 medical LLM benchmarks. Our analysis uncovers widespread, systemic issues, including a profound disconnect from clinical practice, a crisis of data integrity due to unmitigated contamination risks, and a systematic neglect of safety-critical evaluation dimensions like model robustness and uncertainty awareness. Based on these findings, MedCheck serves as both a diagnostic tool for existing benchmarks and an actionable guideline to foster a more standardized, reliable, and transparent approach to evaluating AI in healthcare.
>
---
#### [new 055] TURA: Tool-Augmented Unified Retrieval Agent for AI Search
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出TURA框架，解决传统静态RAG无法处理动态信息源（如数据库/实时API）的问题，通过意图推理模块、DAG任务规划和轻量执行器实现跨模态协作，构建首个支持工业级低延迟的AI搜索系统。**

- **链接: [http://arxiv.org/pdf/2508.04604v1](http://arxiv.org/pdf/2508.04604v1)**

> **作者:** Zhejun Zhao; Yuehu Dong; Alley Liu; Lixue Zheng; Pingsheng Liu; Dongdong Shen; Long Xia; Jiashu Zhao; Dawei Yin
>
> **摘要:** The advent of Large Language Models (LLMs) is transforming search engines into conversational AI search products, primarily using Retrieval-Augmented Generation (RAG) on web corpora. However, this paradigm has significant industrial limitations. Traditional RAG approaches struggle with real-time needs and structured queries that require accessing dynamically generated content like ticket availability or inventory. Limited to indexing static pages, search engines cannot perform the interactive queries needed for such time-sensitive data. Academic research has focused on optimizing RAG for static content, overlooking complex intents and the need for dynamic sources like databases and real-time APIs. To bridge this gap, we introduce TURA (Tool-Augmented Unified Retrieval Agent for AI Search), a novel three-stage framework that combines RAG with agentic tool-use to access both static content and dynamic, real-time information. TURA has three key components: an Intent-Aware Retrieval module to decompose queries and retrieve information sources encapsulated as Model Context Protocol (MCP) Servers, a DAG-based Task Planner that models task dependencies as a Directed Acyclic Graph (DAG) for optimal parallel execution, and a lightweight Distilled Agent Executor for efficient tool calling. TURA is the first architecture to systematically bridge the gap between static RAG and dynamic information sources for a world-class AI search product. Serving tens of millions of users, it leverages an agentic framework to deliver robust, real-time answers while meeting the low-latency demands of a large-scale industrial system.
>
---
#### [new 056] DTPA: Dynamic Token-level Prefix Augmentation for Controllable Text Generation
- **分类: cs.CL**

- **简介: 该论文属于可控文本生成（CTG）任务，旨在解决长序列生成中前缀控制不足的问题。提出Dynamic Token-level Prefix Augmentation（DTPA）框架，通过动态调整前缀增强控制力，结合任务特性实现长序列生成的高效可控性提升。**

- **链接: [http://arxiv.org/pdf/2508.04047v1](http://arxiv.org/pdf/2508.04047v1)**

> **作者:** Jiabing Yang; Yixiang Chen; Zichen Wen; Chenhang Cui; Peiyan Li; Yuan Xu; Bowen Fang; Yan Huang; Liang Wang
>
> **摘要:** Controllable Text Generation (CTG) is a vital subfield in Natural Language Processing (NLP), aiming to generate text that aligns with desired attributes. However, previous studies commonly focus on the quality of controllable text generation for short sequences, while the generation of long-form text remains largely underexplored. In this paper, we observe that the controllability of texts generated by the powerful prefix-based method Air-Decoding tends to decline with increasing sequence length, which we hypothesize primarily arises from the observed decay in attention to the prefixes. Meanwhile, different types of prefixes including soft and hard prefixes are also key factors influencing performance. Building on these insights, we propose a lightweight and effective framework called Dynamic Token-level Prefix Augmentation (DTPA) based on Air-Decoding for controllable text generation. Specifically, it first selects the optimal prefix type for a given task. Then we dynamically amplify the attention to the prefix for the attribute distribution to enhance controllability, with a scaling factor growing exponentially as the sequence length increases. Moreover, based on the task, we optionally apply a similar augmentation to the original prompt for the raw distribution to balance text quality. After attribute distribution reconstruction, the generated text satisfies the attribute constraints well. Experiments on multiple CTG tasks demonstrate that DTPA generally outperforms other methods in attribute control while maintaining competitive fluency, diversity, and topic relevance. Further analysis highlights DTPA's superior effectiveness in long text generation.
>
---
#### [new 057] ToolGrad: Efficient Tool-use Dataset Generation with Textual "Gradients"
- **分类: cs.CL**

- **简介: 该论文提出ToolGrad框架，解决传统工具使用数据集生成中标注失败与效率低下问题，通过逆向思维构建有效工具链并生成用户查询，使模型性能提升至OOD基准。**

- **链接: [http://arxiv.org/pdf/2508.04086v1](http://arxiv.org/pdf/2508.04086v1)**

> **作者:** Zhongyi Zhou; Kohei Uehara; Haoyu Zhang; Jingtao Zhou; Lin Gu; Ruofei Du; Zheng Xu; Tatsuya Harada
>
> **摘要:** Prior work synthesizes tool-use LLM datasets by first generating a user query, followed by complex tool-use annotations like DFS. This leads to inevitable annotation failures and low efficiency in data generation. We introduce ToolGrad, an agentic framework that inverts this paradigm. ToolGrad first constructs valid tool-use chains through an iterative process guided by textual "gradients", and then synthesizes corresponding user queries. This "answer-first" approach led to ToolGrad-5k, a dataset generated with more complex tool use, lower cost, and 100% pass rate. Experiments show that models trained on ToolGrad-5k outperform those on expensive baseline datasets and proprietary LLMs, even on OOD benchmarks.
>
---
#### [new 058] Hacking Hallucinations of MLLMs with Causal Sufficiency and Necessity
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究了多模态大语言模型（MLLMs）在生成过程中产生的幻觉问题，通过因果分析识别了缺失与伪造两种原因，提出基于因果充分性与必要性的强化学习框架，以优化生成结果的准确性。**

- **链接: [http://arxiv.org/pdf/2508.04182v1](http://arxiv.org/pdf/2508.04182v1)**

> **作者:** Peizheng Guo; Jingyao Wang; Wenwen Qiang; Huijie Guo; Changwen Zheng; Jiahuan Zhou; Gang Hua
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across vision-language tasks. However, they may suffer from hallucinations--generating outputs that are semantically inconsistent with the input image or text. Through causal analyses, we find that: (i) hallucinations with omission may arise from the failure to adequately capture essential causal factors, and (ii) hallucinations with fabrication are likely caused by the model being misled by non-causal cues. To address these challenges, we propose a novel reinforcement learning framework guided by causal completeness, which jointly considers both causal sufficiency and causal necessity of tokens. Specifically, we evaluate each token's standalone contribution and counterfactual indispensability to define a token-level causal completeness reward. This reward is used to construct a causally informed advantage function within the GRPO optimization framework, encouraging the model to focus on tokens that are both causally sufficient and necessary for accurate generation. Experimental results across various benchmark datasets and tasks demonstrate the effectiveness of our approach, which effectively mitigates hallucinations in MLLMs.
>
---
#### [new 059] Sotopia-RL: Reward Design for Social Intelligence
- **分类: cs.CL**

- **简介: 该论文提出Sotopia-RL框架，旨在通过对话层次奖励设计优化大型语言模型的社会智能训练，解决传统RL在社交交互中的部分可观测性与多维维度带来的效率不足问题，实现优异的社会目标完成性能。**

- **链接: [http://arxiv.org/pdf/2508.03905v1](http://arxiv.org/pdf/2508.03905v1)**

> **作者:** Haofei Yu; Zhengyang Qi; Yining Zhao; Kolby Nottingham; Keyang Xuan; Bodhisattwa Prasad Majumder; Hao Zhu; Paul Pu Liang; Jiaxuan You
>
> **备注:** 10 pages
>
> **摘要:** Social intelligence has become a critical capability for large language models (LLMs), enabling them to engage effectively in real-world social tasks such as accommodation, persuasion, collaboration, and negotiation. Reinforcement learning (RL) is a natural fit for training socially intelligent agents because it allows models to learn sophisticated strategies directly through social interactions. However, social interactions have two key characteristics that set barriers for RL training: (1) partial observability, where utterances have indirect and delayed effects that complicate credit assignment, and (2) multi-dimensionality, where behaviors such as rapport-building or knowledge-seeking contribute indirectly to goal achievement. These characteristics make Markov decision process (MDP)-based RL with single-dimensional episode-level rewards inefficient and unstable. To address these challenges, we propose Sotopia-RL, a novel framework that refines coarse episode-level feedback into utterance-level, multi-dimensional rewards. Utterance-level credit assignment mitigates partial observability by attributing outcomes to individual utterances, while multi-dimensional rewards capture the full richness of social interactions and reduce reward hacking. Experiments in Sotopia, an open-ended social learning environment, demonstrate that Sotopia-RL achieves state-of-the-art social goal completion scores (7.17 on Sotopia-hard and 8.31 on Sotopia-full), significantly outperforming existing approaches. Ablation studies confirm the necessity of both utterance-level credit assignment and multi-dimensional reward design for RL training. Our implementation is publicly available at: https://github.com/sotopia-lab/sotopia-rl.
>
---
#### [new 060] Beyond Brainstorming: What Drives High-Quality Scientific Ideas? Lessons from Multi-Agent Collaboration
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文旨在探讨多智能体协作对高质科研灵感的驱动作用，通过构建多代理系统验证其优势，对比不同配置（如团队规模、领导结构及专业性）并量化评估其性能，发现领导作用能提升创意质量，认知多样性与专业能力共同决定成果。**

- **链接: [http://arxiv.org/pdf/2508.04575v1](http://arxiv.org/pdf/2508.04575v1)**

> **作者:** Nuo Chen; Yicheng Tong; Jiaying Wu; Minh Duc Duong; Qian Wang; Qingyun Zou; Bryan Hooi; Bingsheng He
>
> **备注:** Preprint
>
> **摘要:** While AI agents show potential in scientific ideation, most existing frameworks rely on single-agent refinement, limiting creativity due to bounded knowledge and perspective. Inspired by real-world research dynamics, this paper investigates whether structured multi-agent discussions can surpass solitary ideation. We propose a cooperative multi-agent framework for generating research proposals and systematically compare configurations including group size, leaderled versus leaderless structures, and team compositions varying in interdisciplinarity and seniority. To assess idea quality, we employ a comprehensive protocol with agent-based scoring and human review across dimensions such as novelty, strategic vision, and integration depth. Our results show that multi-agent discussions substantially outperform solitary baselines. A designated leader acts as a catalyst, transforming discussion into more integrated and visionary proposals. Notably, we find that cognitive diversity is a primary driver of quality, yet expertise is a non-negotiable prerequisite, as teams lacking a foundation of senior knowledge fail to surpass even a single competent agent. These findings offer actionable insights for designing collaborative AI ideation systems and shed light on how team structure influences creative outcomes.
>
---
#### [new 061] Chain of Questions: Guiding Multimodal Curiosity in Language Models
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文旨在解决多模态语言模型在复杂环境中的推理能力不足问题，提出Chain of Questions（CoQ）框架，通过动态生成目标问题引导模型选择相关感官模态，提升其对多模态信息整合与推理准确性。**

- **链接: [http://arxiv.org/pdf/2508.04350v1](http://arxiv.org/pdf/2508.04350v1)**

> **作者:** Nima Iji; Kia Dashtipour
>
> **摘要:** Reasoning capabilities in large language models (LLMs) have substantially advanced through methods such as chain-of-thought and explicit step-by-step explanations. However, these improvements have not yet fully transitioned to multimodal contexts, where models must proactively decide which sensory modalities such as vision, audio, or spatial perception to engage when interacting with complex real-world environments. In this paper, we introduce the Chain of Questions (CoQ) framework, a curiosity-driven reasoning approach that encourages multimodal language models to dynamically generate targeted questions regarding their surroundings. These generated questions guide the model to selectively activate relevant modalities, thereby gathering critical information necessary for accurate reasoning and response generation. We evaluate our framework on a novel multimodal benchmark dataset, assembled by integrating WebGPT, ScienceQA, AVSD, and ScanQA datasets. Experimental results demonstrate that our CoQ method improves a foundation model's ability to effectively identify and integrate pertinent sensory information. This leads to improved accuracy, interpretability, and alignment of the reasoning process with diverse multimodal tasks.
>
---
#### [new 062] GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究了基于强化学习的LLM推理任务，解决了粗粒度奖励分配不足的问题。提出动态熵权重方法，通过分组令牌优化（GTPO）和序列级分组相对优化（GRPO-S），将熵权重用于精确奖励信号，显著提升了模型的深度推理能力。**

- **链接: [http://arxiv.org/pdf/2508.04349v1](http://arxiv.org/pdf/2508.04349v1)**

> **作者:** Hongze Tan; Jianfei Pan
>
> **摘要:** Reinforcement learning (RL) with algorithms like Group Relative Policy Optimization (GRPO) improves Large Language Model (LLM) reasoning, but is limited by a coarse-grained credit assignment that applies a uniform reward to all tokens in a sequence. This is a major flaw in long-chain reasoning tasks. This paper solves this with \textbf{Dynamic Entropy Weighting}. Our core idea is that high-entropy tokens in correct responses can guide the policy toward a higher performance ceiling. This allows us to create more fine-grained reward signals for precise policy updates via two ways: 1) \textbf{Group Token Policy Optimization} (\textbf{GTPO}), we assigns a entropy-weighted reward to each token for fine-grained credit assignment. 2) \textbf{Sequence-Level Group Relative Policy Optimization} (\textbf{GRPO-S}), we assigns a entropy-weighted reward to each sequence based on its average token entropy. Experiments show our methods significantly outperform the strong DAPO baseline. The results confirm that our entropy-weighting mechanism is the key driver of this performance boost, offering a better path to enhance deep reasoning in models.
>
---
#### [new 063] GanitBench: A bi-lingual benchmark for evaluating mathematical reasoning in Vision Language Models
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文旨在评估视觉语言模型在数学推理任务中的性能，解决跨语言数据不足的问题。通过收集来自印度JEE和CBSE的数学题，构建了双语基准（英语/哈里发），并对比了GPT-4o mini在零样本和两步CoT下的表现，同时验证了Double Lock约束对模型性能的负面影响。**

- **链接: [http://arxiv.org/pdf/2508.03737v1](http://arxiv.org/pdf/2508.03737v1)**

> **作者:** Ashutosh Bandooni; Brindha Subburaj
>
> **备注:** 6 pages, 3 figures. Accepted, Presented and Published as part of Proceedings of the 6th International Conference on Recent Advantages in Information Technology (RAIT) 2025
>
> **摘要:** Benchmarks for evaluating reasoning among Vision Language Models (VLMs) on several fields and domains are being curated more frequently over the last few years. However these are often monolingual, mostly available in English. Additionally there also is a lack of datasets available in Hindi on tasks apart from comprehension and translation. We introduce GanitBench, a tough benchmark consisting of 1527 vision-only questions covering several topics in Mathematics - available in languages English and Hindi. Collected from two major examinations from India, the JEE Advanced and the CBSE Boards examinations, this benchmark includes questions in the form of images comprising of figures essential to a question as well as text. We evaluate two closed source models for the same, in zero-shot Chain-of-Thought (CoT) and two-shot CoT settings. GPT-4o mini is found to be the more dominant model on the benchmark, with it's highest average accuracy being 38.15%. We also evaluate models through a "Double Lock" constraint, which brings down the performance of the models by considerable margins. We observe that two-shot CoT appears to be a more effective setting under this environment. Performance of the two VLMs also decreases when answering the same questions in the Hindi language. We hope to facilitate the inclusion of languages like Hindi in research through our work.
>
---
#### [new 064] Difficulty-Based Preference Data Selection by DPO Implicit Reward Gap
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于AI模型对齐任务，旨在解决偏好数据选择问题，提出基于难度差异的DPO隐式奖励机制，通过缩小隐式奖励差距提升数据效率与模型匹配度，取得优于基线的优异性能。**

- **链接: [http://arxiv.org/pdf/2508.04149v1](http://arxiv.org/pdf/2508.04149v1)**

> **作者:** Xuan Qi; Rongwu Xu; Zhijing Jin
>
> **备注:** Our code and data are available at https://github.com/Difficulty-Based-Preference-Data-Select/Difficulty-Based-Preference-Data-Select
>
> **摘要:** Aligning large language models (LLMs) with human preferences is a critical challenge in AI research. While methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) are widely used, they often rely on large, costly preference datasets. The current work lacks methods for high-quality data selection specifically for preference data. In this work, we introduce a novel difficulty-based data selection strategy for preference datasets, grounded in the DPO implicit reward mechanism. By selecting preference data examples with smaller DPO implicit reward gaps, which are indicative of more challenging cases, we improve data efficiency and model alignment. Our approach consistently outperforms five strong baselines across multiple datasets and alignment tasks, achieving superior performance with only 10\% of the original data. This principled, efficient selection method offers a promising solution for scaling LLM alignment with limited resources.
>
---
#### [new 065] Intent Aware Context Retrieval for Multi-Turn Agricultural Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出了一种基于意图驱动的多轮农业问答系统，旨在解决印度农村地区农民获取及时、易懂农业知识的挑战。系统通过融合指令微调、上下文检索与生成增强技术，构建了结构化对话流程，实现了97.53%的准确度和91.35%的个性化响应，有效提升了农业支持的可及性与效率。**

- **链接: [http://arxiv.org/pdf/2508.03719v1](http://arxiv.org/pdf/2508.03719v1)**

> **作者:** Abhay Vijayvargia; Ajay Nagpal; Kundeshwar Pundalik; Atharva Savarkar; Smita Gautam; Pankaj Singh; Rohit Saluja; Ganesh Ramakrishnan
>
> **摘要:** Indian farmers often lack timely, accessible, and language-friendly agricultural advice, especially in rural areas with low literacy. To address this gap in accessibility, this paper presents a novel AI-powered agricultural chatbot, Krishi Sathi, designed to support Indian farmers by providing personalized, easy-to-understand answers to their queries through both text and speech. The system's intelligence stems from an IFT model, subsequently refined through fine-tuning on Indian agricultural knowledge across three curated datasets. Unlike traditional chatbots that respond to one-off questions, Krishi Sathi follows a structured, multi-turn conversation flow to gradually collect the necessary details from the farmer, ensuring the query is fully understood before generating a response. Once the intent and context are extracted, the system performs Retrieval-Augmented Generation (RAG) by first fetching information from a curated agricultural database and then generating a tailored response using the IFT model. The chatbot supports both English and Hindi languages, with speech input and output features (via ASR and TTS) to make it accessible for users with low literacy or limited digital skills. This work demonstrates how combining intent-driven dialogue flows, instruction-tuned models, and retrieval-based generation can improve the quality and accessibility of digital agricultural support in India. This approach yielded strong results, with the system achieving a query response accuracy of 97.53%, 91.35% contextual relevance and personalization, and a query completion rate of 97.53%. The average response time remained under 6 seconds, ensuring timely support for users across both English and Hindi interactions.
>
---
#### [new 066] KVSink: Understanding and Enhancing the Preservation of Attention Sinks in KV Cache Quantization for LLMs
- **分类: cs.CL**

- **简介: 该论文旨在通过解析注意力点在KV缓存量化中的作用机制，解决现有方法无法有效保留注意力点的问题，提出KVSink方法，提升KV缓存量化对注意力点的保护能力并优化LLM推理效果。**

- **链接: [http://arxiv.org/pdf/2508.04257v1](http://arxiv.org/pdf/2508.04257v1)**

> **作者:** Zunhai Su; Kehong Yuan
>
> **备注:** Published as a conference paper at COLM 2025
>
> **摘要:** Key-Value (KV) cache quantization has become a widely adopted optimization technique for efficient large language models (LLMs) inference by reducing KV cache memory usage and mitigating memory-bound constraints. Recent studies have emphasized the importance of preserving the original precision of KVs for the first few tokens to ensure the protection of attention sinks. While this approach has proven effective in mitigating performance degradation, its underlying principles remain insufficiently understood. Moreover, it fails to address the recent discovery that attention sinks can emerge beyond the initial token positions. In this work, we elucidate the underlying mechanisms of attention sinks during inference by examining their role in the cross-layer evolution of extreme activation outliers. Additionally, we provide a comprehensive analysis of the interplay between attention sinks and KV cache quantization. Based on our enhanced understanding, we introduce \textit{\textbf{KVSink}}, a plug-and-play method that effectively predicts sink tokens with negligible overhead, enabling more thorough preservation. Extensive experiments demonstrate that KVSink outperforms the existing Preserve-First-N (PFN) strategy, offering more effective preservation of attention sinks during KV cache quantization. Moreover, when applied to the well-established KVQuant method, KVSink further improves perplexity (PPL) and reduces reliance on 16-bit numerical outliers.
>
---
#### [new 067] CoAct-1: Computer-using Agents with Coding as Actions
- **分类: cs.CL**

- **简介: 该论文提出了一种结合GUI与编程的多智能体系统（CoAct-1），旨在解决传统GUI操作效率低且缺乏编程能力的问题。通过动态分发任务至GUI或程序员，实现了代码执行的增强，显著提升了任务完成效率（平均步骤减少10.15次）和成功率（60.76%）。**

- **链接: [http://arxiv.org/pdf/2508.03923v1](http://arxiv.org/pdf/2508.03923v1)**

> **作者:** Linxin Song; Yutong Dai; Viraj Prabhu; Jieyu Zhang; Taiwei Shi; Li Li; Junnan Li; Silvio Savarese; Zeyuan Chen; Jieyu Zhao; Ran Xu; Caiming Xiong
>
> **摘要:** Autonomous agents that operate computers via Graphical User Interfaces (GUIs) often struggle with efficiency and reliability on complex, long-horizon tasks. While augmenting these agents with planners can improve task decomposition, they remain constrained by the inherent limitations of performing all actions through GUI manipulation, leading to brittleness and inefficiency. In this work, we introduce a more robust and flexible paradigm: enabling agents to use coding as a enhanced action. We present CoAct-1, a novel multi-agent system that synergistically combines GUI-based control with direct programmatic execution. CoAct-1 features an Orchestrator that dynamically delegates subtasks to either a conventional GUI Operator or a specialized Programmer agent, which can write and execute Python or Bash scripts. This hybrid approach allows the agent to bypass inefficient GUI action sequences for tasks like file management and data processing, while still leveraging visual interaction when necessary. We evaluate our system on the challenging OSWorld benchmark, where CoAct-1 achieves a new state-of-the-art success rate of 60.76%, significantly outperforming prior methods. Furthermore, our approach dramatically improves efficiency, reducing the average number of steps required to complete a task to just 10.15, compared to 15 for leading GUI agents. Our results demonstrate that integrating coding as a core action provides a more powerful, efficient, and scalable path toward generalized computer automation.
>
---
#### [new 068] SEAgent: Self-Evolving Computer Use Agent with Autonomous Learning from Experience
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MA; cs.MM**

- **简介: 本研究提出SEAgent，一种自进化计算机使用代理，解决因缺乏标注而难以适应新软件的问题，通过构建World State模型和Curriculum生成器，结合对抗性模仿和GRPO等方法提升性能。**

- **链接: [http://arxiv.org/pdf/2508.04700v1](http://arxiv.org/pdf/2508.04700v1)**

> **作者:** Zeyi Sun; Ziyu Liu; Yuhang Zang; Yuhang Cao; Xiaoyi Dong; Tong Wu; Dahua Lin; Jiaqi Wang
>
> **备注:** Code at https://github.com/SunzeY/SEAgent
>
> **摘要:** Repurposing large vision-language models (LVLMs) as computer use agents (CUAs) has led to substantial breakthroughs, primarily driven by human-labeled data. However, these models often struggle with novel and specialized software, particularly in scenarios lacking human annotations. To address this challenge, we propose SEAgent, an agentic self-evolving framework enabling CUAs to autonomously evolve through interactions with unfamiliar software. Specifically, SEAgent empowers computer-use agents to autonomously master novel software environments via experiential learning, where agents explore new software, learn through iterative trial-and-error, and progressively tackle auto-generated tasks organized from simple to complex. To achieve this goal, we design a World State Model for step-wise trajectory assessment, along with a Curriculum Generator that generates increasingly diverse and challenging tasks. The agent's policy is updated through experiential learning, comprised of adversarial imitation of failure actions and Group Relative Policy Optimization (GRPO) on successful ones. Furthermore, we introduce a specialist-to-generalist training strategy that integrates individual experiential insights from specialist agents, facilitating the development of a stronger generalist CUA capable of continuous autonomous evolution. This unified agent ultimately achieves performance surpassing ensembles of individual specialist agents on their specialized software. We validate the effectiveness of SEAgent across five novel software environments within OS-World. Our approach achieves a significant improvement of 23.2% in success rate, from 11.3% to 34.5%, over a competitive open-source CUA, i.e., UI-TARS.
>
---
#### [new 069] ASTRA: Autonomous Spatial-Temporal Red-teaming for AI Software Assistants
- **分类: cs.CR; cs.CL; cs.LG; cs.SE**

- **简介: 该论文提出一种自主的时空红攻系统ASTRA，旨在提升AI软件助手的安全性，通过构建结构化知识图谱进行动态漏洞挖掘，解决AI在高危场景下的安全性问题。**

- **链接: [http://arxiv.org/pdf/2508.03936v1](http://arxiv.org/pdf/2508.03936v1)**

> **作者:** Xiangzhe Xu; Guangyu Shen; Zian Su; Siyuan Cheng; Hanxi Guo; Lu Yan; Xuan Chen; Jiasheng Jiang; Xiaolong Jin; Chengpeng Wang; Zhuo Zhang; Xiangyu Zhang
>
> **备注:** The first two authors (Xiangzhe Xu and Guangyu Shen) contributed equally to this work
>
> **摘要:** AI coding assistants like GitHub Copilot are rapidly transforming software development, but their safety remains deeply uncertain-especially in high-stakes domains like cybersecurity. Current red-teaming tools often rely on fixed benchmarks or unrealistic prompts, missing many real-world vulnerabilities. We present ASTRA, an automated agent system designed to systematically uncover safety flaws in AI-driven code generation and security guidance systems. ASTRA works in three stages: (1) it builds structured domain-specific knowledge graphs that model complex software tasks and known weaknesses; (2) it performs online vulnerability exploration of each target model by adaptively probing both its input space, i.e., the spatial exploration, and its reasoning processes, i.e., the temporal exploration, guided by the knowledge graphs; and (3) it generates high-quality violation-inducing cases to improve model alignment. Unlike prior methods, ASTRA focuses on realistic inputs-requests that developers might actually ask-and uses both offline abstraction guided domain modeling and online domain knowledge graph adaptation to surface corner-case vulnerabilities. Across two major evaluation domains, ASTRA finds 11-66% more issues than existing techniques and produces test cases that lead to 17% more effective alignment training, showing its practical value for building safer AI systems.
>
---
#### [new 070] CX-Mind: A Pioneering Multimodal Large Language Model for Interleaved Reasoning in Chest X-ray via Curriculum-Guided Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究了基于课程引导强化学习的胸部X光（CXR）多模态推理任务，解决了传统单任务模型缺乏过程监督和跨任务推理能力的问题，创新性地构建了CX-Set数据集并通过CuRL-VPR机制实现高精度推理，显著提升了医疗影像诊断效率与解释性。**

- **链接: [http://arxiv.org/pdf/2508.03733v1](http://arxiv.org/pdf/2508.03733v1)**

> **作者:** Wenjie Li; Yujie Zhang; Haoran Sun; Yueqi Li; Fanrui Zhang; Mengzhe Xu; Victoria Borja Clausich; Sade Mellin; Renhao Yang; Chenrun Wang; Jethro Zih-Shuo Wang; Shiyi Yao; Gen Li; Yidong Xu; Hanyu Wang; Yilin Huang; Angela Lin Wang; Chen Shi; Yin Zhang; Jianan Guo; Luqi Yang; Renxuan Li; Yang Xu; Jiawei Liu; Yao Zhang; Lei Liu; Carlos Gutiérrez SanRomán; Lei Wang
>
> **摘要:** Chest X-ray (CXR) imaging is one of the most widely used diagnostic modalities in clinical practice, encompassing a broad spectrum of diagnostic tasks. Recent advancements have seen the extensive application of reasoning-based multimodal large language models (MLLMs) in medical imaging to enhance diagnostic efficiency and interpretability. However, existing multimodal models predominantly rely on "one-time" diagnostic approaches, lacking verifiable supervision of the reasoning process. This leads to challenges in multi-task CXR diagnosis, including lengthy reasoning, sparse rewards, and frequent hallucinations. To address these issues, we propose CX-Mind, the first generative model to achieve interleaved "think-answer" reasoning for CXR tasks, driven by curriculum-based reinforcement learning and verifiable process rewards (CuRL-VPR). Specifically, we constructed an instruction-tuning dataset, CX-Set, comprising 708,473 images and 2,619,148 samples, and generated 42,828 high-quality interleaved reasoning data points supervised by clinical reports. Optimization was conducted in two stages under the Group Relative Policy Optimization framework: initially stabilizing basic reasoning with closed-domain tasks, followed by transfer to open-domain diagnostics, incorporating rule-based conditional process rewards to bypass the need for pretrained reward models. Extensive experimental results demonstrate that CX-Mind significantly outperforms existing medical and general-domain MLLMs in visual understanding, text generation, and spatiotemporal alignment, achieving an average performance improvement of 25.1% over comparable CXR-specific models. On real-world clinical dataset (Rui-CXR), CX-Mind achieves a mean recall@1 across 14 diseases that substantially surpasses the second-best results, with multi-center expert evaluations further confirming its clinical utility across multiple dimensions.
>
---
#### [new 071] MegaWika 2: A More Comprehensive Multilingual Collection of Articles and their Sources
- **分类: cs.DL; cs.CL**

- **简介: 该论文介绍了MegaWika 2，一个包含多语言文章及引用的大型数据集，扩展了原始数据并支持报告生成与事实核查等功能。解决了传统数据集在覆盖范围和信息完整性方面的不足，通过引入爬虫来源文本和精确字符引用提升信息获取能力。**

- **链接: [http://arxiv.org/pdf/2508.03828v1](http://arxiv.org/pdf/2508.03828v1)**

> **作者:** Samuel Barham; Chandler May; Benjamin Van Durme
>
> **摘要:** We introduce MegaWika 2, a large, multilingual dataset of Wikipedia articles with their citations and scraped web sources; articles are represented in a rich data structure, and scraped source texts are stored inline with precise character offsets of their citations in the article text. MegaWika 2 is a major upgrade from the original MegaWika, spanning six times as many articles and twice as many fully scraped citations. Both MegaWika and MegaWika 2 support report generation research ; whereas MegaWika also focused on supporting question answering and retrieval applications, MegaWika 2 is designed to support fact checking and analyses across time and language.
>
---
#### [new 072] MD-LLM-1: A Large Language Model for Molecular Dynamics
- **分类: q-bio.BM; cs.CL; cs.LG; physics.comp-ph**

- **简介: 该论文提出MD-LLM框架，旨在通过深度学习优化分子动力学计算，解决传统MD方法在生物大分子模拟中的计算成本高问题，实现蛋白构象状态预测。**

- **链接: [http://arxiv.org/pdf/2508.03709v1](http://arxiv.org/pdf/2508.03709v1)**

> **作者:** Mhd Hussein Murtada; Z. Faidon Brotzakis; Michele Vendruscolo
>
> **摘要:** Molecular dynamics (MD) is a powerful approach for modelling molecular systems, but it remains computationally intensive on spatial and time scales of many macromolecular systems of biological interest. To explore the opportunities offered by deep learning to address this problem, we introduce a Molecular Dynamics Large Language Model (MD-LLM) framework to illustrate how LLMs can be leveraged to learn protein dynamics and discover states not seen in training. By applying MD-LLM-1, the first implementation of this approach, obtained by fine-tuning Mistral 7B, to the T4 lysozyme and Mad2 protein systems, we show that training on one conformational state enables the prediction of other conformational states. These results indicate that MD-LLM-1 can learn the principles for the exploration of the conformational landscapes of proteins, although it is not yet modeling explicitly their thermodynamics and kinetics.
>
---
#### [new 073] GTPO: Trajectory-Based Policy Optimization in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出GTPO方法，解决传统政策优化中的冲突梯度与分布失真问题，通过识别冲突项、跳过负更新并过滤熵阈值，提升训练稳定性与效果，验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2508.03772v1](http://arxiv.org/pdf/2508.03772v1)**

> **作者:** Marco Simoni; Aleksandar Fontana; Giulio Rossolini; Andrea Saracino
>
> **摘要:** Policy-based optimizations are widely adopted today for the training and alignment of language models, where one of the most recent and effective approaches is Group-relative Policy Optimization (GRPO). In this paper, we reveals and analyze two major limitations of GRPO: (i) tokens frequently appear in completions with both positive and negative rewards, leading to conflicting gradient updates that can reduce their output probability, even though can be essential for maintaining proper structure; (ii) negatively rewarded completions may penalize confident responses and shift model decisions toward unlikely tokens, progressively flattening the output distribution and degrading learning. To address these issues and provide a more stable and effective policy optimization strategy, we introduce GTPO (Group-relative Trajectory-based Policy Optimization), which identifies conflict tokens, tokens appearing in the same position across completions with opposite rewards, protects them by skipping negative updates, while amplifying positive ones. To further prevent policy collapse, GTPO filters out completions whose entropy exceeds a provable threshold. Unlike GRPO, GTPO does not rely on KL-divergence regularization, eliminating the need for a reference model during training, while still ensuring greater training stability and improved performance, validated through multiple experiments on GSM8K, MATH and AIME 2024 benchmarks.
>
---
#### [new 074] Analyzing and Mitigating Object Hallucination: A Training Bias Perspective
- **分类: cs.CV; cs.CL**

- **简介: 该论文旨在分析并缓解对象幻觉问题，通过研究训练偏移机制（主要集中在语言建模头）提出Obliviate方法，有效减少模型对训练数据的依赖性。**

- **链接: [http://arxiv.org/pdf/2508.04567v1](http://arxiv.org/pdf/2508.04567v1)**

> **作者:** Yifan Li; Kun Zhou; Wayne Xin Zhao; Lei Fang; Ji-Rong Wen
>
> **摘要:** As scaling up training data has significantly improved the general multimodal capabilities of Large Vision-Language Models (LVLMs), they still suffer from the hallucination issue, generating text that is inconsistent with the visual input. This phenomenon motivates us to systematically investigate the role of training data in hallucination. We introduce a new benchmark, POPEv2, which consists of counterfactual images collected from the training data of LVLMs with certain objects masked. Through comprehensive evaluation on POPEv2, we find that current LVLMs suffer from training bias: they fail to fully leverage their training data and hallucinate more frequently on images seen during training. Specifically, they perform poorly on counterfactual images, often incorrectly answering ``Yes'' to questions about masked objects. To understand this issue, we conduct probing experiments on the models' internal components, revealing that this training bias is primarily located in the language modeling (LM) head. Based on these findings, we propose Obliviate, an efficient and lightweight unlearning method designed to mitigate object hallucination via training bias unlearning. Obliviate identifies the discrepancy between ground-truth labels and model outputs on the training data as a proxy for bias and adopts a parameter- and data-efficient fine-tuning strategy that only updates the LM head. Extensive experiments demonstrate the effectiveness of our approach. While only reusing the training data and updating approximately 2\% of the parameters, Obliviate significantly reduces hallucination across both discriminative and generative tasks. Furthermore, it demonstrates strong scalability with respect to both model size (2B to 72B) and training data volume, and exhibits promising generalization to hallucination types beyond object-level hallucination. Our code and data will be publicly released.
>
---
#### [new 075] Do Recommender Systems Really Leverage Multimodal Content? A Comprehensive Analysis on Multimodal Representations for Recommendation
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文探讨了多模态推荐系统的有效性，旨在验证跨模态理解与模型复杂性的关系。通过构建大型视觉语言模型（LVLMs）生成结构化嵌入，解决了传统多模态融合策略的局限性，实现了跨模态语义对齐与性能提升。**

- **链接: [http://arxiv.org/pdf/2508.04571v1](http://arxiv.org/pdf/2508.04571v1)**

> **作者:** Claudio Pomo; Matteo Attimonelli; Danilo Danese; Fedelucio Narducci; Tommaso Di Noia
>
> **备注:** Accepted as Full Research Papers at CIKM 2025
>
> **摘要:** Multimodal Recommender Systems aim to improve recommendation accuracy by integrating heterogeneous content, such as images and textual metadata. While effective, it remains unclear whether their gains stem from true multimodal understanding or increased model complexity. This work investigates the role of multimodal item embeddings, emphasizing the semantic informativeness of the representations. Initial experiments reveal that embeddings from standard extractors (e.g., ResNet50, Sentence-Bert) enhance performance, but rely on modality-specific encoders and ad hoc fusion strategies that lack control over cross-modal alignment. To overcome these limitations, we leverage Large Vision-Language Models (LVLMs) to generate multimodal-by-design embeddings via structured prompts. This approach yields semantically aligned representations without requiring any fusion. Experiments across multiple settings show notable performance improvements. Furthermore, LVLMs embeddings offer a distinctive advantage: they can be decoded into structured textual descriptions, enabling direct assessment of their multimodal comprehension. When such descriptions are incorporated as side content into recommender systems, they improve recommendation performance, empirically validating the semantic depth and alignment encoded within LVLMs outputs. Our study highlights the importance of semantically rich representations and positions LVLMs as a compelling foundation for building robust and meaningful multimodal representations in recommendation tasks.
>
---
#### [new 076] Health Insurance Coverage Rule Interpretation Corpus: Law, Policy, and Medical Guidance for Health Insurance Coverage Understanding
- **分类: cs.CY; cs.AI; cs.CL; cs.LG**

- **简介: 该论文旨在构建健康保险规则理解语料库并开发预测模型，解决理解复杂政策与公平性问题，通过收集法律与医学文本及设计任务提升监管与患者自决能力。**

- **链接: [http://arxiv.org/pdf/2508.03718v1](http://arxiv.org/pdf/2508.03718v1)**

> **作者:** Mike Gartner
>
> **备注:** 22 pages, 7 figures
>
> **摘要:** U.S. health insurance is complex, and inadequate understanding and limited access to justice have dire implications for the most vulnerable. Advances in natural language processing present an opportunity to support efficient, case-specific understanding, and to improve access to justice and healthcare. Yet existing corpora lack context necessary for assessing even simple cases. We collect and release a corpus of reputable legal and medical text related to U.S. health insurance. We also introduce an outcome prediction task for health insurance appeals designed to support regulatory and patient self-help applications, and release a labeled benchmark for our task, and models trained on it.
>
---
#### [new 077] FrEVL: Leveraging Frozen Pretrained Embeddings for Efficient Vision-Language Understanding
- **分类: cs.CV; cs.CL**

- **简介: FrEVL框架探索冻结预训练嵌入能否提升视觉-语言理解效果，解决计算资源不足问题，通过对比实验验证其效率优势。**

- **链接: [http://arxiv.org/pdf/2508.04469v1](http://arxiv.org/pdf/2508.04469v1)**

> **作者:** Emmanuelle Bourigault; Pauline Bourigault
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** The deployment of vision-language models remains constrained by substantial computational requirements. We present \textbf{FrEVL}, a framework exploring whether frozen pretrained embeddings can support effective vision-language understanding. Our analysis reveals that frozen embeddings contain rich information for discriminative tasks, achieving 85\% to 95\% of state-of-the-art performance on standard benchmarks with only 68.4M trainable parameters. This performance dichotomy reveals a critical insight: frozen embedding effectiveness depends on alignment between pretraining objectives and downstream task requirements. When accounting for end-to-end computation including embedding extraction, FrEVL provides $2.3\times$ speedup with 52\% lower energy consumption, making it suitable for scenarios with pre-computable inputs or when deployment constraints outweigh marginal performance gains. Our evaluation provides practitioners with guidance on when frozen embedding approaches represent viable alternatives to full model deployment. We will release our complete implementation and evaluation framework to facilitate further research into efficient multi-modal understanding.
>
---
#### [new 078] ToxicTAGS: Decoding Toxic Memes with Rich Tag Annotations
- **分类: cs.CV; cs.CL**

- **简介: 该论文旨在解决传统文本数据无法有效识别恶意网络言论的问题，构建了一个包含二分类和细粒度标签的毒言数据集，通过辅助标签增强语义理解并提出生成式标签模块提升多模态模型性能，为实时内容监控提供新框架。**

- **链接: [http://arxiv.org/pdf/2508.04166v1](http://arxiv.org/pdf/2508.04166v1)**

> **作者:** Subhankar Swain; Naquee Rizwan; Nayandeep Deb; Vishwajeet Singh Solanki; Vishwa Gangadhar S; Animesh Mukherjee
>
> **摘要:** The 2025 Global Risks Report identifies state-based armed conflict and societal polarisation among the most pressing global threats, with social media playing a central role in amplifying toxic discourse. Memes, as a widely used mode of online communication, often serve as vehicles for spreading harmful content. However, limitations in data accessibility and the high cost of dataset curation hinder the development of robust meme moderation systems. To address this challenge, in this work, we introduce a first-of-its-kind dataset of 6,300 real-world meme-based posts annotated in two stages: (i) binary classification into toxic and normal, and (ii) fine-grained labelling of toxic memes as hateful, dangerous, or offensive. A key feature of this dataset is that it is enriched with auxiliary metadata of socially relevant tags, enhancing the context of each meme. In addition, we propose a tag generation module that produces socially grounded tags, because most in-the-wild memes often do not come with tags. Experimental results show that incorporating these tags substantially enhances the performance of state-of-the-art VLMs detection tasks. Our contributions offer a novel and scalable foundation for improved content moderation in multimodal online environments.
>
---
#### [new 079] Causal Reflection with Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Causal Reflection框架，解决LLMs在因果推理中的能力不足问题，通过动态状态-动作-时间-扰动函数建模与Reflect机制优化，实现自然语言解释与因果推断。**

- **链接: [http://arxiv.org/pdf/2508.04495v1](http://arxiv.org/pdf/2508.04495v1)**

> **作者:** Abi Aryan; Zac Liu
>
> **摘要:** While LLMs exhibit impressive fluency and factual recall, they struggle with robust causal reasoning, often relying on spurious correlations and brittle patterns. Similarly, traditional Reinforcement Learning agents also lack causal understanding, optimizing for rewards without modeling why actions lead to outcomes. We introduce Causal Reflection, a framework that explicitly models causality as a dynamic function over state, action, time, and perturbation, enabling agents to reason about delayed and nonlinear effects. Additionally, we define a formal Reflect mechanism that identifies mismatches between predicted and observed outcomes and generates causal hypotheses to revise the agent's internal model. In this architecture, LLMs serve not as black-box reasoners, but as structured inference engines translating formal causal outputs into natural language explanations and counterfactuals. Our framework lays the theoretical groundwork for Causal Reflective agents that can adapt, self-correct, and communicate causal understanding in evolving environments.
>
---
#### [new 080] COPO: Consistency-Aware Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决多样本收敛导致的梯度消失问题，提出基于全局一致性约束的策略优化框架与熵平衡机制，提升模型推理路径的自洽性与训练效率。**

- **链接: [http://arxiv.org/pdf/2508.04138v1](http://arxiv.org/pdf/2508.04138v1)**

> **作者:** Jinghang Han; Jiawei Chen; Hang Shao; Hao Ma; Mingcheng Li; Xintian Shen; Lihao Zheng; Wei Chen; Tao Wei; Lihua Zhang
>
> **摘要:** Reinforcement learning has significantly enhanced the reasoning capabilities of Large Language Models (LLMs) in complex problem-solving tasks. Recently, the introduction of DeepSeek R1 has inspired a surge of interest in leveraging rule-based rewards as a low-cost alternative for computing advantage functions and guiding policy optimization. However, a common challenge observed across many replication and extension efforts is that when multiple sampled responses under a single prompt converge to identical outcomes, whether correct or incorrect, the group-based advantage degenerates to zero. This leads to vanishing gradients and renders the corresponding samples ineffective for learning, ultimately limiting training efficiency and downstream performance. To address this issue, we propose a consistency-aware policy optimization framework that introduces a structured global reward based on outcome consistency, the global loss based on it ensures that, even when model outputs show high intra-group consistency, the training process still receives meaningful learning signals, which encourages the generation of correct and self-consistent reasoning paths from a global perspective. Furthermore, we incorporate an entropy-based soft blending mechanism that adaptively balances local advantage estimation with global optimization, enabling dynamic transitions between exploration and convergence throughout training. Our method introduces several key innovations in both reward design and optimization strategy. We validate its effectiveness through substantial performance gains on multiple mathematical reasoning benchmarks, highlighting the proposed framework's robustness and general applicability. Code of this work has been released at https://github.com/hijih/copo-code.git.
>
---
#### [new 081] Graph Representation Learning with Massive Unlabeled Data for Rumor Detection
- **分类: cs.SI; cs.CL**

- **简介: 该论文旨在利用大规模无标签数据（如微博/推特）构建图表示学习模型，解决谣言检测任务中的数据获取难及泛化性问题，通过改进图自监督方法验证其性能并建立十年级谣言数据集，证明其在半监督下具有更强的适应能力。**

- **链接: [http://arxiv.org/pdf/2508.04252v1](http://arxiv.org/pdf/2508.04252v1)**

> **作者:** Chaoqun Cui; Caiyan Jia
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** With the development of social media, rumors spread quickly, cause great harm to society and economy. Thereby, many effective rumor detection methods have been developed, among which the rumor propagation structure learning based methods are particularly effective compared to other methods. However, the existing methods still suffer from many issues including the difficulty to obtain large-scale labeled rumor datasets, which leads to the low generalization ability and the performance degeneration on new events since rumors are time-critical and usually appear with hot topics or newly emergent events. In order to solve the above problems, in this study, we used large-scale unlabeled topic datasets crawled from the social media platform Weibo and Twitter with claim propagation structure to improve the semantic learning ability of a graph reprentation learing model on various topics. We use three typical graph self-supervised methods, InfoGraph, JOAO and GraphMAE in two commonly used training strategies, to verify the performance of general graph semi-supervised methods in rumor detection tasks. In addition, for alleviating the time and topic difference between unlabeled topic data and rumor data, we also collected a rumor dataset covering a variety of topics over a decade (10-year ago from 2022) from the Weibo rumor-refuting platform. Our experiments show that these general graph self-supervised learning methods outperform previous methods specifically designed for rumor detection tasks and achieve good performance under few-shot conditions, demonstrating the better generalization ability with the help of our massive unlabeled topic dataset.
>
---
#### [new 082] OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文旨在调查基于多模态大语言模型（MLLMs）的OS Agent技术，解决在通用计算设备中实现智能助手的任务需求，提出构建方法与评估体系，并推动安全隐私与个性化发展研究。**

- **链接: [http://arxiv.org/pdf/2508.04482v1](http://arxiv.org/pdf/2508.04482v1)**

> **作者:** Xueyu Hu; Tao Xiong; Biao Yi; Zishu Wei; Ruixuan Xiao; Yurun Chen; Jiasheng Ye; Meiling Tao; Xiangxin Zhou; Ziyu Zhao; Yuhuai Li; Shengze Xu; Shenzhi Wang; Xinchen Xu; Shuofei Qiao; Zhaokai Wang; Kun Kuang; Tieyong Zeng; Liang Wang; Jiwei Li; Yuchen Eleanor Jiang; Wangchunshu Zhou; Guoyin Wang; Keting Yin; Zhou Zhao; Hongxia Yang; Fan Wu; Shengyu Zhang; Fei Wu
>
> **备注:** ACL 2025 (Oral)
>
> **摘要:** The dream to create AI assistants as capable and versatile as the fictional J.A.R.V.I.S from Iron Man has long captivated imaginations. With the evolution of (multi-modal) large language models ((M)LLMs), this dream is closer to reality, as (M)LLM-based Agents using computing devices (e.g., computers and mobile phones) by operating within the environments and interfaces (e.g., Graphical User Interface (GUI)) provided by operating systems (OS) to automate tasks have significantly advanced. This paper presents a comprehensive survey of these advanced agents, designated as OS Agents. We begin by elucidating the fundamentals of OS Agents, exploring their key components including the environment, observation space, and action space, and outlining essential capabilities such as understanding, planning, and grounding. We then examine methodologies for constructing OS Agents, focusing on domain-specific foundation models and agent frameworks. A detailed review of evaluation protocols and benchmarks highlights how OS Agents are assessed across diverse tasks. Finally, we discuss current challenges and identify promising directions for future research, including safety and privacy, personalization and self-evolution. This survey aims to consolidate the state of OS Agents research, providing insights to guide both academic inquiry and industrial development. An open-source GitHub repository is maintained as a dynamic resource to foster further innovation in this field. We present a 9-page version of our work, accepted by ACL 2025, to provide a concise overview to the domain.
>
---
#### [new 083] ConvMix: A Mixed-Criteria Data Augmentation Framework for Conversational Dense Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出了一种混合准则的对话密集检索框架（ConvMix），解决数据稀缺问题并通过两向相关性增强与质量控制机制优化检索效果，实现了跨标注数据的多样化样本生成与近分布监督。**

- **链接: [http://arxiv.org/pdf/2508.04001v1](http://arxiv.org/pdf/2508.04001v1)**

> **作者:** Fengran Mo; Jinghan Zhang; Yuchen Hui; Jia Ao Sun; Zhichao Xu; Zhan Su; Jian-Yun Nie
>
> **摘要:** Conversational search aims to satisfy users' complex information needs via multiple-turn interactions. The key challenge lies in revealing real users' search intent from the context-dependent queries. Previous studies achieve conversational search by fine-tuning a conversational dense retriever with relevance judgments between pairs of context-dependent queries and documents. However, this training paradigm encounters data scarcity issues. To this end, we propose ConvMix, a mixed-criteria framework to augment conversational dense retrieval, which covers more aspects than existing data augmentation frameworks. We design a two-sided relevance judgment augmentation schema in a scalable manner via the aid of large language models. Besides, we integrate the framework with quality control mechanisms to obtain semantically diverse samples and near-distribution supervisions to combine various annotated data. Experimental results on five widely used benchmarks show that the conversational dense retriever trained by our ConvMix framework outperforms previous baseline methods, which demonstrates our superior effectiveness.
>
---
#### [new 084] Accelerating Scientific Discovery with Multi-Document Summarization of Impact-Ranked Papers
- **分类: cs.DL; cs.AI; cs.CL**

- **简介: 该论文旨在通过BIP! Finder等工具的多文档摘要功能，解决科学文献筛选中的认知负担问题，提升对高影响力论文的快速理解和整合能力。**

- **链接: [http://arxiv.org/pdf/2508.03962v1](http://arxiv.org/pdf/2508.03962v1)**

> **作者:** Paris Koloveas; Serafeim Chatzopoulos; Dionysis Diamantis; Christos Tryfonopoulos; Thanasis Vergoulis
>
> **摘要:** The growing volume of scientific literature makes it challenging for scientists to move from a list of papers to a synthesized understanding of a topic. Because of the constant influx of new papers on a daily basis, even if a scientist identifies a promising set of papers, they still face the tedious task of individually reading through dozens of titles and abstracts to make sense of occasionally conflicting findings. To address this critical bottleneck in the research workflow, we introduce a summarization feature to BIP! Finder, a scholarly search engine that ranks literature based on distinct impact aspects like popularity and influence. Our approach enables users to generate two types of summaries from top-ranked search results: a concise summary for an instantaneous at-a-glance comprehension and a more comprehensive literature review-style summary for greater, better-organized comprehension. This ability dynamically leverages BIP! Finder's already existing impact-based ranking and filtering features to generate context-sensitive, synthesized narratives that can significantly accelerate literature discovery and comprehension.
>
---
#### [new 085] AgREE: Agentic Reasoning for Knowledge Graph Completion on Emerging Entities
- **分类: cs.AI; cs.CL**

- **简介: 该论文旨在解决开放域知识图谱完成（KGC）在动态信息环境下的新兴实体建模难题，提出基于代理推理的AgREE框架，通过迭代检索与多步推理动态生成知识图谱，证明在无需额外训练情况下可显著优于现有方法，提升对新实体的捕捉能力达13.7%。**

- **链接: [http://arxiv.org/pdf/2508.04118v1](http://arxiv.org/pdf/2508.04118v1)**

> **作者:** Ruochen Zhao; Simone Conia; Eric Peng; Min Li; Saloni Potdar
>
> **摘要:** Open-domain Knowledge Graph Completion (KGC) faces significant challenges in an ever-changing world, especially when considering the continual emergence of new entities in daily news. Existing approaches for KGC mainly rely on pretrained language models' parametric knowledge, pre-constructed queries, or single-step retrieval, typically requiring substantial supervision and training data. Even so, they often fail to capture comprehensive and up-to-date information about unpopular and/or emerging entities. To this end, we introduce Agentic Reasoning for Emerging Entities (AgREE), a novel agent-based framework that combines iterative retrieval actions and multi-step reasoning to dynamically construct rich knowledge graph triplets. Experiments show that, despite requiring zero training efforts, AgREE significantly outperforms existing methods in constructing knowledge graph triplets, especially for emerging entities that were not seen during language models' training processes, outperforming previous methods by up to 13.7%. Moreover, we propose a new evaluation methodology that addresses a fundamental weakness of existing setups and a new benchmark for KGC on emerging entities. Our work demonstrates the effectiveness of combining agent-based reasoning with strategic information retrieval for maintaining up-to-date knowledge graphs in dynamic information environments.
>
---
#### [new 086] Beyond Pixels: Exploring DOM Downsampling for LLM-Based Web Agents
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文探讨了通过DOM下降采样技术提升LLM基于Web代理的任务实现。研究解决了应用状态保存（snapshots）与传统GUI截图的对比问题，提出D2Snap算法优化DOM结构以降低输入token规模，验证其在Online-Mind2Web数据集上的有效性，证明DOM内嵌结构对LLM具备强UI特征。**

- **链接: [http://arxiv.org/pdf/2508.04412v1](http://arxiv.org/pdf/2508.04412v1)**

> **作者:** Thassilo M. Schiepanski; Nicholas Piël
>
> **摘要:** Frontier LLMs only recently enabled serviceable, autonomous web agents. At that, a model poses as an instantaneous domain model backend. Ought to suggest interaction, it is consulted with a web-based task and respective application state. The key problem lies in application state serialisation $\unicode{x2013}$ referred to as snapshot. State-of-the-art web agents are premised on grounded GUI snapshots, i.e., screenshots enhanced with visual cues. Not least to resemble human perception, but for images representing relatively cheap means of model input. LLM vision still lag behind code interpretation capabilities. DOM snapshots, which structurally resemble HTML, impose a desired alternative. Vast model input token size, however, disables reliable implementation with web agents to date. We propose D2Snap, a first-of-its-kind DOM downsampling algorithm. Based on a GPT-4o backend, we evaluate D2Snap on tasks sampled from the Online-Mind2Web dataset. The success rate of D2Snap-downsampled DOM snapshots (67%) matches a grounded GUI snapshot baseline (65%) $\unicode{x2013}$ within the same input token order of magnitude (1e3). Our best evaluated configurations $\unicode{x2013}$ one token order above, but within the model's context window $\unicode{x2013}$ outperform this baseline by 8%. Our evaluation, moreover, yields that DOM-inherent hierarchy embodies a strong UI feature for LLMs.
>
---
#### [new 087] Position: The Current AI Conference Model is Unsustainable! Diagnosing the Crisis of Centralized AI Conference
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文旨在诊断当前中心化AI会议的可持续性危机，分析其四个结构性压力（科学/环境/心理/物流）并提出Community-Federated Conference（CFC）模型作为替代方案。**

- **链接: [http://arxiv.org/pdf/2508.04586v1](http://arxiv.org/pdf/2508.04586v1)**

> **作者:** Nuo Chen; Moming Duan; Andre Huikai Lin; Qian Wang; Jiaying Wu; Bingsheng He
>
> **备注:** Preprint
>
> **摘要:** Artificial Intelligence (AI) conferences are essential for advancing research, sharing knowledge, and fostering academic community. However, their rapid expansion has rendered the centralized conference model increasingly unsustainable. This paper offers a data-driven diagnosis of a structural crisis that threatens the foundational goals of scientific dissemination, equity, and community well-being. We identify four key areas of strain: (1) scientifically, with per-author publication rates more than doubling over the past decade to over 4.5 papers annually; (2) environmentally, with the carbon footprint of a single conference exceeding the daily emissions of its host city; (3) psychologically, with 71% of online community discourse reflecting negative sentiment and 35% referencing mental health concerns; and (4) logistically, with attendance at top conferences such as NeurIPS 2024 beginning to outpace venue capacity. These pressures point to a system that is misaligned with its core mission. In response, we propose the Community-Federated Conference (CFC) model, which separates peer review, presentation, and networking into globally coordinated but locally organized components, offering a more sustainable, inclusive, and resilient path forward for AI research.
>
---
#### [new 088] A Social Data-Driven System for Identifying Estate-related Events and Topics
- **分类: cs.IR; cs.AI; cs.CL; cs.LG; cs.SI**

- **简介: 该论文提出了一种基于社会数据的系统，旨在通过语言模型检测和分类遗产相关事件，解决了从社交媒体中提取有效信息的问题，并结合地理模块实现点位定位，支持城市管理和应急响应。**

- **链接: [http://arxiv.org/pdf/2508.03711v1](http://arxiv.org/pdf/2508.03711v1)**

> **作者:** Wenchuan Mu; Menglin Li; Kwan Hui Lim
>
> **备注:** Accepted at ASONAM 2025
>
> **摘要:** Social media platforms such as Twitter and Facebook have become deeply embedded in our everyday life, offering a dynamic stream of localized news and personal experiences. The ubiquity of these platforms position them as valuable resources for identifying estate-related issues, especially in the context of growing urban populations. In this work, we present a language model-based system for the detection and classification of estate-related events from social media content. Our system employs a hierarchical classification framework to first filter relevant posts and then categorize them into actionable estate-related topics. Additionally, for posts lacking explicit geotags, we apply a transformer-based geolocation module to infer posting locations at the point-of-interest level. This integrated approach supports timely, data-driven insights for urban management, operational response and situational awareness.
>
---
#### [new 089] Query Attribute Modeling: Improving search relevance with Semantic Search and Meta Data Filtering
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出了一种基于语义搜索和元数据过滤的查询属性建模框架，旨在提升搜索精度并解决传统搜索无法有效提取元数据的问题。通过将开放文本查询拆分为结构化标签与语义元素，该方法在Amazon Toys Reviews数据集上验证了其优于BM25等传统技术的表现，建立了企业级搜索系统的高效解决方案。**

- **链接: [http://arxiv.org/pdf/2508.04683v1](http://arxiv.org/pdf/2508.04683v1)**

> **作者:** Karthik Menon; Batool Arhamna Haider; Muhammad Arham; Kanwal Mehreen; Ram Mohan Rao Kadiyala; Hamza Farooq
>
> **摘要:** This study introduces Query Attribute Modeling (QAM), a hybrid framework that enhances search precision and relevance by decomposing open text queries into structured metadata tags and semantic elements. QAM addresses traditional search limitations by automatically extracting metadata filters from free-form text queries, reducing noise and enabling focused retrieval of relevant items. Experimental evaluation using the Amazon Toys Reviews dataset (10,000 unique items with 40,000+ reviews and detailed product attributes) demonstrated QAM's superior performance, achieving a mean average precision at 5 (mAP@5) of 52.99\%. This represents significant improvement over conventional methods, including BM25 keyword search, encoder-based semantic similarity search, cross-encoder re-ranking, and hybrid search combining BM25 and semantic results via Reciprocal Rank Fusion (RRF). The results establish QAM as a robust solution for Enterprise Search applications, particularly in e-commerce systems.
>
---
#### [new 090] Multilingual Source Tracing of Speech Deepfakes: A First Benchmark
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文旨在构建跨语言的语音深伪造源追踪基准，解决多语言模型追踪问题，通过比较DSP与SSL方法、分析语言差异影响等工作，为多语言模型泛化性研究提供新方向。**

- **链接: [http://arxiv.org/pdf/2508.04143v1](http://arxiv.org/pdf/2508.04143v1)**

> **作者:** Xi Xuan; Yang Xiao; Rohan Kumar Das; Tomi Kinnunen
>
> **备注:** Accepted at Interspeech SPSC 2025 - 5th Symposium on Security and Privacy in Speech Communication (Oral)
>
> **摘要:** Recent progress in generative AI has made it increasingly easy to create natural-sounding deepfake speech from just a few seconds of audio. While these tools support helpful applications, they also raise serious concerns by making it possible to generate convincing fake speech in many languages. Current research has largely focused on detecting fake speech, but little attention has been given to tracing the source models used to generate it. This paper introduces the first benchmark for multilingual speech deepfake source tracing, covering both mono- and cross-lingual scenarios. We comparatively investigate DSP- and SSL-based modeling; examine how SSL representations fine-tuned on different languages impact cross-lingual generalization performance; and evaluate generalization to unseen languages and speakers. Our findings offer the first comprehensive insights into the challenges of identifying speech generation models when training and inference languages differ. The dataset, protocol and code are available at https://github.com/xuanxixi/Multilingual-Source-Tracing.
>
---
## 更新

#### [replaced 001] EcoTransformer: Attention without Multiplication
- **分类: cs.LG; cs.AI; cs.CL; 68T05**

- **链接: [http://arxiv.org/pdf/2507.20096v2](http://arxiv.org/pdf/2507.20096v2)**

> **作者:** Xin Gao; Xingming Xu; Shirin Amiraslani; Hong Xu
>
> **备注:** 8 pages, 1 figure
>
> **摘要:** The Transformer, with its scaled dot-product attention mechanism, has become a foundational architecture in modern AI. However, this mechanism is computationally intensive and incurs substantial energy costs. We propose a new Transformer architecture EcoTransformer, in which the output context vector is constructed as the convolution of the values using a Laplacian kernel, where the distances are measured by the L1 metric between the queries and keys. Compared to dot-product based attention, the new attention score calculation is free of matrix multiplication. It performs on par with, or even surpasses, scaled dot-product attention in NLP, bioinformatics, and vision tasks, while consuming significantly less energy. (This version (v2) supersedes v1 and reflects the intended release and licensing.)
>
---
#### [replaced 002] Tool Unlearning for Tool-Augmented LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01083v2](http://arxiv.org/pdf/2502.01083v2)**

> **作者:** Jiali Cheng; Hadi Amiri
>
> **备注:** ICML 2025 https://clu-uml.github.io/MU-Bench-Project-Page/
>
> **摘要:** Tool-augmented large language models (LLMs) are often trained on datasets of query-response pairs, which embed the ability to use tools or APIs directly into the parametric knowledge of LLMs. Tool-augmented LLMs need the ability to forget learned tools due to security vulnerabilities, privacy regulations, or tool deprecations. However, ``tool unlearning'' has not been investigated in unlearning literature. We introduce this novel task, which requires addressing distinct challenges compared to traditional unlearning: knowledge removal rather than forgetting individual samples, the high cost of optimizing LLMs, and the need for principled evaluation metrics. To bridge these gaps, we propose ToolDelete, the first approach for unlearning tools from tool-augmented LLMs. It implements three key properties to address the above challenges for effective tool unlearning and introduces a new membership inference attack (MIA) model for effective evaluation. Extensive experiments on multiple tool learning datasets and tool-augmented LLMs show that ToolDelete effectively unlearns randomly selected tools, while preserving the LLM's knowledge on non-deleted tools and maintaining performance on general tasks.
>
---
#### [replaced 003] Marco-Voice Technical Report
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.02038v2](http://arxiv.org/pdf/2508.02038v2)**

> **作者:** Fengping Tian; Chenyang Lyu; Xuanfan Ni; Haoqin Sun; Qingjuan Li; Zhiqiang Qian; Haijun Li; Longyue Wang; Zhao Xu; Weihua Luo; Kaifu Zhang
>
> **备注:** Technical Report. Our code and dataset are publicly available at https://github.com/AIDC-AI/Marco-Voice and https://huggingface.co/datasets/AIDC-AI/CSEMOTIONS respectively
>
> **摘要:** This paper presents a multifunctional speech synthesis system that integrates voice cloning and emotion control speech synthesis within a unified framework. The goal of this work is to address longstanding challenges in achieving highly expressive, controllable, and natural speech generation that faithfully preserves speaker identity across diverse linguistic and emotional contexts. Our approach introduces an effective speaker-emotion disentanglement mechanism with in-batch contrastive learning, enabling independent manipulation of speaker identity and eemotional style, as well as rotational emotional embedding integration method for smooth emotion control. To support comprehensive training and evaluation, we construct CSEMOTIONS, a high-quality emotional speech dataset containing 10 hours of Mandarin speech from six professional speakers across seven emotional categories. Extensive experiments demonstrate that our system, Marco-Voice, achieves substantial improvements in both objective and subjective metrics. Comprehensive evaluations and analysis were conducted, results show that MarcoVoice delivers competitive performance in terms of speech clarity and emotional richness, representing a substantial advance in the field of expressive neural speech synthesis. Our code and dataset are publicly available at https://github.com/AIDC-AI/Marco-Voice and https://huggingface.co/datasets/AIDC-AI/CSEMOTIONS respectively.
>
---
#### [replaced 004] Evaluating Multi-Hop Reasoning in Large Language Models: A Chemistry-Centric Case Study
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.16414v2](http://arxiv.org/pdf/2504.16414v2)**

> **作者:** Mohammad Khodadad; Ali Shiraee Kasmaee; Mahdi Astaraki; Nicholas Sherck; Hamidreza Mahyar; Soheila Samiee
>
> **摘要:** In this study, we introduced a new benchmark consisting of a curated dataset and a defined evaluation process to assess the compositional reasoning capabilities of large language models within the chemistry domain. We designed and validated a fully automated pipeline, verified by subject matter experts, to facilitate this task. Our approach integrates OpenAI reasoning models with named entity recognition (NER) systems to extract chemical entities from recent literature, which are then augmented with external knowledge bases to form a comprehensive knowledge graph. By generating multi-hop questions across these graphs, we assess LLM performance in both context-augmented and non-context augmented settings. Our experiments reveal that even state-of-the-art models face significant challenges in multi-hop compositional reasoning. The results reflect the importance of augmenting LLMs with document retrieval, which can have a substantial impact on improving their performance. However, even perfect retrieval accuracy with full context does not eliminate reasoning errors, underscoring the complexity of compositional reasoning. This work not only benchmarks and highlights the limitations of current LLMs but also presents a novel data generation pipeline capable of producing challenging reasoning datasets across various domains. Overall, this research advances our understanding of reasoning in computational linguistics.
>
---
#### [replaced 005] The Impact of Item-Writing Flaws on Difficulty and Discrimination in Item Response Theory
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2503.10533v2](http://arxiv.org/pdf/2503.10533v2)**

> **作者:** Robin Schmucker; Steven Moore
>
> **摘要:** High-quality test items are essential for educational assessments, particularly within Item Response Theory (IRT). Traditional validation methods rely on resource-intensive pilot testing to estimate item difficulty and discrimination. More recently, Item-Writing Flaw (IWF) rubrics emerged as a domain-general approach for evaluating test items based on textual features. This method offers a scalable, pre-deployment evaluation without requiring student data, but its predictive validity concerning empirical IRT parameters is underexplored. To address this gap, we conducted a study involving 7,126 multiple-choice questions across various STEM subjects (physical science, mathematics, and life/earth sciences). Using an automated approach, we annotated each question with a 19-criteria IWF rubric and studied relationships to data-driven IRT parameters. Our analysis revealed statistically significant links between the number of IWFs and IRT difficulty and discrimination parameters, particularly in life/earth and physical science domains. We further observed how specific IWF criteria can impact item quality more and less severely (e.g., negative wording vs. implausible distractors) and how they might make a question more or less challenging. Overall, our findings establish automated IWF analysis as a valuable supplement to traditional validation, providing an efficient method for initial item screening, particularly for flagging low-difficulty MCQs. Our findings show the need for further research on domain-general evaluation rubrics and algorithms that understand domain-specific content for robust item validation.
>
---
#### [replaced 006] CharBench: Evaluating the Role of Tokenization in Character-Level Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.02591v2](http://arxiv.org/pdf/2508.02591v2)**

> **作者:** Omri Uzan; Yuval Pinter
>
> **摘要:** Tasks that require character-level reasoning, such as counting or locating characters within words, remain challenging for contemporary language models. A common conjecture is that language models' reliance on subword units, rather than characters, contributes to their struggles with character-level tasks, yet recent studies offer conflicting conclusions about the role of tokenization, leaving its impact unclear. To address this gap, we introduce CharBench, a comprehensive benchmark of character-level tasks that is two orders of magnitude larger than existing alternatives. We evaluate a diverse range of leading open-weight and proprietary models on CharBench and find that it presents a significant challenge to modern LLMs, with an average accuracy of 43.6% and 32.3% on some tasks. We present an in-depth analysis of how intrinsic properties of words and their segmentations into tokens correspond to model performance. For counting tasks, we find that tokenization properties are weakly correlated with correctness, while the length of the queried word and the actual character count play a more significant part. In contrast, for tasks requiring intra-word positional understanding, performance is negatively correlated with the length of the token containing the queried character, suggesting that longer tokens obscure character position information for LLMs. We encourage future work to build on the benchmark and evaluation methodology introduced here as tools for improving model performance on such tasks.
>
---
#### [replaced 007] Human Bias in the Face of AI: Examining Human Judgment Against Text Labeled as AI Generated
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2410.03723v2](http://arxiv.org/pdf/2410.03723v2)**

> **作者:** Tiffany Zhu; Iain Weissburg; Kexun Zhang; William Yang Wang
>
> **备注:** 5 main pages, 10 total pages
>
> **摘要:** As AI advances in text generation, human trust in AI generated content remains constrained by biases that go beyond concerns of accuracy. This study explores how bias shapes the perception of AI versus human generated content. Through three experiments involving text rephrasing, news article summarization, and persuasive writing, we investigated how human raters respond to labeled and unlabeled content. While the raters could not differentiate the two types of texts in the blind test, they overwhelmingly favored content labeled as "Human Generated," over those labeled "AI Generated," by a preference score of over 30%. We observed the same pattern even when the labels were deliberately swapped. This human bias against AI has broader societal and cognitive implications, as it undervalues AI performance. This study highlights the limitations of human judgment in interacting with AI and offers a foundation for improving human-AI collaboration, especially in creative fields.
>
---
#### [replaced 008] Thought Anchors: Which LLM Reasoning Steps Matter?
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.19143v3](http://arxiv.org/pdf/2506.19143v3)**

> **作者:** Paul C. Bogdan; Uzay Macar; Neel Nanda; Arthur Conmy
>
> **备注:** Paul C. Bogdan and Uzay Macar contributed equally to this work, and their listed order was determined by coinflip. Neel Nanda and Arthur Conmy contributed equally to this work as senior authors, and their listed order was determined by coinflip
>
> **摘要:** Reasoning large language models have recently achieved state-of-the-art performance in many fields. However, their long-form chain-of-thought reasoning creates interpretability challenges as each generated token depends on all previous ones, making the computation harder to decompose. We argue that analyzing reasoning traces at the sentence level is a promising approach to understanding reasoning processes. We present three complementary attribution methods: (1) a black-box method measuring each sentence's counterfactual importance by comparing final answers across 100 rollouts conditioned on the model generating that sentence or one with a different meaning; (2) a white-box method of aggregating attention patterns between pairs of sentences, which identified "broadcasting" sentences that receive disproportionate attention from all future sentences via "receiver" attention heads; (3) a causal attribution method measuring logical connections between sentences by suppressing attention toward one sentence and measuring the effect on each future sentence's tokens. Each method provides evidence for the existence of thought anchors, reasoning steps that have outsized importance and that disproportionately influence the subsequent reasoning process. These thought anchors are typically planning or backtracking sentences. We provide an open-source tool (www.thought-anchors.com) for visualizing the outputs of our methods, and present a case study showing converging patterns across methods that map how a model performs multi-step reasoning. The consistency across methods demonstrates the potential of sentence-level analysis for a deeper understanding of reasoning models.
>
---
#### [replaced 009] Emotion-o1: Adaptive Long Reasoning for Emotion Understanding in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22548v2](http://arxiv.org/pdf/2505.22548v2)**

> **作者:** Changhao Song; Yazhou Zhang; Hui Gao; Kaiyun Huang; Peng Zhang
>
> **摘要:** Long chain-of-thought (CoT) reasoning has shown great promise in enhancing the emotion understanding performance of large language models (LLMs). However, current fixed-length CoT methods struggle to balance reasoning depth and efficiency. Simple tasks (e.g., sentiment classification) are over-reasoned, while complex tasks (e.g., sarcasm understanding) lack depth. To fill this gap, we present Emotion-o1, an adaptive CoT framework that dynamically adjusts reasoning length based on emotion-task complexity. Emotion-o1 is trained by distilling adaptive CoT patterns from a reasoning-oriented LLM, followed by supervised fine-tuning and reinforcement learning with a four-part reward targeting accuracy, brevity, structure, and redundancy. Experimental results on four emotion tasks highlight: (1) Emotion-o1 demonstrates significant improvements over its backbone, with F1 score increases of 10%(Sentiment), 5%(Emotion), 18%(Humor), and 27%(Sarcasm). (2) In sentiment and sarcasm tasks, our 8B model demonstrates superior performance against advanced LLMs, outperforming Grok-3 by 1.1% and Claude-3.7 by 2%. (3) The framework maintains accuracy while reducing reasoning length by 83% compared to OpenAI-o1, demonstrating effective precision-efficiency optimization. Emotion-o1 effectively balances reasoning depth and efficiency for emotion understanding in LLMs.
>
---
#### [replaced 010] Reliable Evaluation Protocol for Low-Precision Retrieval
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.03306v2](http://arxiv.org/pdf/2508.03306v2)**

> **作者:** Kisu Yang; Yoonna Jang; Hwanseok Jang; Kenneth Choi; Isabelle Augenstein; Heuiseok Lim
>
> **备注:** 11 pages, 5 figures, submitted to ARR
>
> **摘要:** Lowering the numerical precision of model parameters and computations is widely adopted to improve the efficiency of retrieval systems. However, when computing relevance scores between the query and documents in low-precision, we observe spurious ties due to the reduced granularity. This introduces high variability in the results based on tie resolution, making the evaluation less reliable. To address this, we propose a more robust retrieval evaluation protocol designed to reduce score variation. It consists of: (1) High-Precision Scoring (HPS), which upcasts the final scoring step to higher precision to resolve tied candidates with minimal computational cost; and (2) Tie-aware Retrieval Metrics (TRM), which report expected scores, range, and bias to quantify order uncertainty of tied candidates. Our experiments test multiple models with three scoring functions on two retrieval datasets to demonstrate that HPS dramatically reduces tie-induced instability, and TRM accurately recovers expected metric values. This combination enables a more consistent and reliable evaluation system for lower-precision retrievals.
>
---
#### [replaced 011] Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.17937v2](http://arxiv.org/pdf/2507.17937v2)**

> **作者:** Jaechul Roh; Zachary Novack; Yuefeng Peng; Niloofar Mireshghallah; Taylor Berg-Kirkpatrick; Amir Houmansadr
>
> **摘要:** Memorization in generative models extends far beyond verbatim text reproduction--it manifests through non-literal patterns, semantic associations, and surprisingly, across modalities in transcript-conditioned generation tasks such as Lyrics-to-Song (L2S) and Text-to-Video (T2V) models. We reveal a new class of cross-modality memorization where models trained on these tasks leak copyrighted content through indirect, phonetic pathways invisible to traditional text-based analysis. In this work, we introduce Adversarial PhoneTic Prompting (APT), an attack that replaces iconic phrases with homophonic alternatives--e.g., "mom's spaghetti" becomes "Bob's confetti"--preserving the acoustic form while largely changing semantic content. We demonstrate that models can be prompted to regurgitate memorized songs using phonetically similar but semantically unrelated lyrics. Despite the semantic drift, black-box models like SUNO and open-source models like YuE generate outputs that are strikingly similar to the original songs--melodically, rhythmically, and vocally--achieving high scores on AudioJudge, CLAP, and CoverID. These effects persist across genres and languages. More surprisingly, we find that phonetic prompts alone can trigger visual memorization in text-to-video models: when given altered lyrics from Lose Yourself, Veo 3 generates scenes that mirror the original music video--complete with a hooded rapper and dim urban settings--despite no explicit visual cues in the prompt. This cross-modality leakage represents an unprecedented threat: models memorize deep, structural patterns that transcend their training modality, making traditional safety measures like copyright filters ineffective. Our findings reveal a fundamental vulnerability in transcript-conditioned generative models and raise urgent concerns around copyright, provenance, and secure deployment of multimodal generation systems.
>
---
#### [replaced 012] AUTALIC: A Dataset for Anti-AUTistic Ableist Language In Context
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.16520v4](http://arxiv.org/pdf/2410.16520v4)**

> **作者:** Naba Rizvi; Harper Strickland; Daniel Gitelman; Tristan Cooper; Alexis Morales-Flores; Michael Golden; Aekta Kallepalli; Akshat Alurkar; Haaset Owens; Saleha Ahmedi; Isha Khirwadkar; Imani Munyaka; Nedjma Ousidhoum
>
> **备注:** accepted to ACL main 2025, 9 pages, 5 figures, 7 tables
>
> **摘要:** As our understanding of autism and ableism continues to increase, so does our understanding of ableist language towards autistic people. Such language poses a significant challenge in NLP research due to its subtle and context-dependent nature. Yet, detecting anti-autistic ableist language remains underexplored, with existing NLP tools often failing to capture its nuanced expressions. We present AUTALIC, the first benchmark dataset dedicated to the detection of anti-autistic ableist language in context, addressing a significant gap in the field. The dataset comprises 2,400 autism-related sentences collected from Reddit, accompanied by surrounding context, and is annotated by trained experts with backgrounds in neurodiversity. Our comprehensive evaluation reveals that current language models, including state-of-the-art LLMs, struggle to reliably identify anti-autistic ableism and align with human judgments, underscoring their limitations in this domain. We publicly release AUTALIC along with the individual annotations which serve as a valuable resource to researchers working on ableism, neurodiversity, and also studying disagreements in annotation tasks. This dataset serves as a crucial step towards developing more inclusive and context-aware NLP systems that better reflect diverse perspectives.
>
---
#### [replaced 013] SLR: Automated Synthesis for Scalable Logical Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.15787v4](http://arxiv.org/pdf/2506.15787v4)**

> **作者:** Lukas Helff; Ahmad Omar; Felix Friedrich; Antonia Wüst; Hikaru Shindo; Rupert Mitchell; Tim Woydt; Patrick Schramowski; Wolfgang Stammer; Kristian Kersting
>
> **摘要:** We introduce SLR, an end-to-end framework for systematic evaluation and training of Large Language Models (LLMs) via Scalable Logical Reasoning. Given a user's task specification, SLR automatically synthesizes (i) an instruction prompt for an inductive reasoning task, (ii) a validation program, executable on model outputs to provide verifiable rewards, and (iii) the latent ground-truth rule. This process is fully automated, scalable, requires no human annotations, and offers precise control over task difficulty. Using SLR, we create SLR-Bench, a benchmark comprising 19k prompts organized into 20 curriculum levels that progressively increase in relational, arithmetic, and recursive complexity. Large-scale evaluation reveals that contemporary LLMs readily produce syntactically valid rules, yet often fail at correct logical inference. Recent reasoning LLMs demonstrate improved performance but incur very high test-time computation, with costs exceeding $300 for just 1,000 prompts. Finally, curriculum learning via SLR doubles Llama-3-8B accuracy on SLR-Bench, achieving parity with Gemini-Flash-Thinking at a fraction of computational cost. Moreover, these reasoning capabilities generalize to a wide range of established benchmarks, underscoring the effectiveness of SLR for downstream reasoning.
>
---
#### [replaced 014] IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.16402v2](http://arxiv.org/pdf/2506.16402v2)**

> **作者:** Xiaoya Lu; Zeren Chen; Xuhao Hu; Yijin Zhou; Weichen Zhang; Dongrui Liu; Lu Sheng; Jing Shao
>
> **摘要:** Flawed planning from VLM-driven embodied agents poses significant safety hazards, hindering their deployment in real-world household tasks. However, existing static, non-interactive evaluation paradigms fail to adequately assess risks within these interactive environments, since they cannot simulate dynamic risks that emerge from an agent's actions and rely on unreliable post-hoc evaluations that ignore unsafe intermediate steps. To bridge this critical gap, we propose evaluating an agent's interactive safety: its ability to perceive emergent risks and execute mitigation steps in the correct procedural order. We thus present IS-Bench, the first multi-modal benchmark designed for interactive safety, featuring 161 challenging scenarios with 388 unique safety risks instantiated in a high-fidelity simulator. Crucially, it facilitates a novel process-oriented evaluation that verifies whether risk mitigation actions are performed before/after specific risk-prone steps. Extensive experiments on leading VLMs, including the GPT-4o and Gemini-2.5 series, reveal that current agents lack interactive safety awareness, and that while safety-aware Chain-of-Thought can improve performance, it often compromises task completion. By highlighting these critical limitations, IS-Bench provides a foundation for developing safer and more reliable embodied AI systems. Code and data are released under [this https URL](https://github.com/AI45Lab/IS-Bench).
>
---
#### [replaced 015] Improving the fact-checking performance of language models by relying on their entailment ability
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15050v2](http://arxiv.org/pdf/2505.15050v2)**

> **作者:** Gaurav Kumar; Debajyoti Mazumder; Ayush Garg; Jasabanta Patro
>
> **备注:** 44 pages
>
> **摘要:** Automated fact-checking is a crucial task in this digital age. The NLP community has been trying various strategies to build robust fact-checking systems. However, we have not been very successful yet. One main reason behind this is that fact verification is a complex process. Language models have to parse through multiple pieces of evidence, often contradicting each other, to predict a claim's veracity. In this paper, we proposed a simple yet effective strategy, where we relied on the entailment ability of language models to improve the fact-checking performance. Apart from that, we did a comparison of different prompting and fine-tuning strategies, as it is currently lacking in the literature. Some of our observations are: (i) training language models with raw evidence sentences (TBE-1) and overall claim-evidence understanding (TBE-2) resulted in an improvement up to 8.20% and 16.39% in macro-F1 for RAW-FC dataset, and (ii) training language models with entailed justifications (TBE-3) outperformed the baselines by a huge margin (up to 28.57% and 44.26% for LIAR-RAW and RAW-FC, respectively). We have shared our code repository to reproduce the results.
>
---
#### [replaced 016] I Have Covered All the Bases Here: Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.18878v2](http://arxiv.org/pdf/2503.18878v2)**

> **作者:** Andrey Galichin; Alexey Dontsov; Polina Druzhinina; Anton Razzhigaev; Oleg Y. Rogov; Elena Tutubalina; Ivan Oseledets
>
> **摘要:** Recent LLMs like DeepSeek-R1 have demonstrated state-of-the-art performance by integrating deep thinking and complex reasoning during generation. However, the internal mechanisms behind these reasoning processes remain unexplored. We observe reasoning LLMs consistently use vocabulary associated with human reasoning processes. We hypothesize these words correspond to specific reasoning moments within the models' internal mechanisms. To test this hypothesis, we employ Sparse Autoencoders (SAEs), a technique for sparse decomposition of neural network activations into human-interpretable features. We introduce ReasonScore, an automatic metric to identify active SAE features during these reasoning moments. We perform manual and automatic interpretation of the features detected by our metric, and find those with activation patterns matching uncertainty, exploratory thinking, and reflection. Through steering experiments, we demonstrate that amplifying these features increases performance on reasoning-intensive benchmarks (+2.2%) while producing longer reasoning traces (+20.5%). Using the model diffing technique, we provide evidence that these features are present only in models with reasoning capabilities. Our work provides the first step towards a mechanistic understanding of reasoning in LLMs. Code available at https://github.com/AIRI-Institute/SAE-Reasoning
>
---
#### [replaced 017] On the Fundamental Impossibility of Hallucination Control in Large Language Models
- **分类: stat.ML; cs.AI; cs.CL; cs.GT; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06382v4](http://arxiv.org/pdf/2506.06382v4)**

> **作者:** Michał P. Karpowicz
>
> **备注:** cleared mathematics, proofs and ideas explained, added missing definitions and axioms, discussion and speculation section added
>
> **摘要:** This paper establishes a fundamental impossibility theorem: no LLM capable performing non-trivial knowledge aggregation can simultaneously achieve truthful (internally consistent) knowledge representation, semantic information conservation, complete revelation of relevant knowledge, and knowledge-constrained optimality. This impossibility is not an engineering limitation but arises from the mathematical structure of information aggregation itself. We establish this result by describing the inference process as an auction of ideas, where distributed components compete exploiting their partial knowledge to shape responses. The proof spans three independent mathematical domains: mechanism design theory (Green-Laffont), the theory of proper scoring rules (Savage), and direct architectural analysis of transformers (Log-Sum-Exp convexity). In particular, we show how in the strictly concave settings the score of an aggregate of diverse beliefs strictly exceeds the sum of individual scores. That gap may quantify the creation of unattributable certainty or overconfidence -- the mathematical origin of both hallucination and creativity, or imagination. To support this analysis, we introduce the complementary concepts of the semantic information measure and the emergence operator to model bounded reasoning in a general setting. We prove that while bounded reasoning generates accessible information, providing valuable insights and inspirations, idealized reasoning strictly preserves semantic content. By demonstrating that hallucination and imagination are mathematically identical phenomena-grounded in the necessary violation of information conservation-this paper offers a principled foundation for managing these behaviors in advanced AI systems. Finally, we present some speculative ideas to inspire evaluation and refinements of the proposed theory.
>
---
#### [replaced 018] HyCodePolicy: Hybrid Language Controllers for Multimodal Monitoring and Decision in Embodied Agents
- **分类: cs.RO; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.02629v2](http://arxiv.org/pdf/2508.02629v2)**

> **作者:** Yibin Liu; Zhixuan Liang; Zanxin Chen; Tianxing Chen; Mengkang Hu; Wanxi Dong; Congsheng Xu; Zhaoming Han; Yusen Qin; Yao Mu
>
> **备注:** Accepted to ICCV 2025 Workshop on Multi-Modal Reasoning for Agentic Intelligence
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have enabled richer perceptual grounding for code policy generation in embodied agents. However, most existing systems lack effective mechanisms to adaptively monitor policy execution and repair codes during task completion. In this work, we introduce HyCodePolicy, a hybrid language-based control framework that systematically integrates code synthesis, geometric grounding, perceptual monitoring, and iterative repair into a closed-loop programming cycle for embodied agents. Technically, given a natural language instruction, our system first decomposes it into subgoals and generates an initial executable program grounded in object-centric geometric primitives. The program is then executed in simulation, while a vision-language model (VLM) observes selected checkpoints to detect and localize execution failures and infer failure reasons. By fusing structured execution traces capturing program-level events with VLM-based perceptual feedback, HyCodePolicy infers failure causes and repairs programs. This hybrid dual feedback mechanism enables self-correcting program synthesis with minimal human supervision. Our results demonstrate that HyCodePolicy significantly improves the robustness and sample efficiency of robot manipulation policies, offering a scalable strategy for integrating multimodal reasoning into autonomous decision-making pipelines.
>
---
#### [replaced 019] UITron-Speech: Towards Automated GUI Agents Based on Speech Instructions
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.11127v2](http://arxiv.org/pdf/2506.11127v2)**

> **作者:** Wenkang Han; Zhixiong Zeng; Jing Huang; Shu Jiang; Liming Zheng; Haibo Qiu; Chang Yao; Jingyuan Chen; Lin Ma
>
> **摘要:** Autonomous agents for Graphical User Interfaces (GUIs) are revolutionizing human-computer interaction, yet their reliance on text-based instructions imposes limitations on accessibility and convenience, particularly in hands-free scenarios. To address this issue, we propose replacing text with speech as the instruction input modality for GUI agents, and introduce UITron-Speech, which is the first end-to-end GUI agent capable of directly processing speech instructions and on-device screenshots to predict user actions. To tackle the problem of data scarcity, we synthesize high-quality speech instruction datasets using a random-speaker text-to-speech model. Additionally, we design a mixed-modality training strategy to mitigate the inherent modality imbalance in pre-trained foundation models. Furthermore, we conduct a statistical analysis of the distribution of GUI grounding prediction errors and propose a training-free two-step grounding refinement method to alleviate minor localization deviations. Extensive experiments on multiple benchmarks demonstrate that UITron-Speech achieves robust performance and superior adaptability, underscoring the feasibility and potential of speech-driven GUI agents for more accessible and intelligent human-computer interaction. Our code and datasets are available at https://github.com/UITron-hub/UITron-Speech.
>
---
#### [replaced 020] Thinking with Nothinking Calibration: A New In-Context Learning Paradigm in Reasoning Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.03363v2](http://arxiv.org/pdf/2508.03363v2)**

> **作者:** Haotian Wu; Bo Xu; Yao Shu; Menglin Yang; Chengwei Qin
>
> **摘要:** Reasoning large language models (RLLMs) have recently demonstrated remarkable capabilities through structured and multi-step reasoning. While prior research has primarily focused on improving their training and inference strategies, their potential for in-context learning (ICL) remains largely underexplored. To fill this gap, we propose Thinking with Nothinking Calibration (JointThinking), a new ICL paradigm that leverages the structured difference between two reasoning modes, i.e., Thinking and Nothinking, to improve reasoning accuracy. Specifically, our method prompts the model to generate two answers in parallel: one in Thinking mode and the other in Nothinking mode. A second round of Thinking is triggered only when the two initial responses are inconsistent, using a single prompt that incorporates the original question and both candidate answers. Since such disagreement occurs infrequently (e.g., only 6\% in GSM8K), our method performs just one round of reasoning in most cases, resulting in minimal latency overhead. Extensive experiments across multiple reasoning benchmarks demonstrate that JointThinking significantly outperforms few-shot chain-of-thought (CoT) and majority voting with improved answer robustness. Moreover, It achieves comparable in-distribution performance to training-based SOTA method, while substantially outperforming on out-of-distribution tasks. We further conduct a systematic analysis of the calibration mechanism, showing that leveraging different reasoning modes consistently lowers the error rate and highlights the value of structural thinking diversity. Additionally, we observe that the performance gap between actual and ideal reasoning narrows as model size increases in the second round of thinking, indicating the strong scalability of our approach. Finally, we discuss current limitations and outline promising directions for future ICL research in RLLMs.
>
---
#### [replaced 021] R1-RE: Cross-Domain Relation Extraction with RLVR
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.04642v2](http://arxiv.org/pdf/2507.04642v2)**

> **作者:** Runpeng Dai; Tong Zheng; Run Yang; Kaixian Yu; Hongtu Zhu
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Relation extraction (RE) is a core task in natural language processing. Traditional approaches typically frame RE as a supervised learning problem, directly mapping context to labels-an approach that often suffers from poor out-of-domain (OOD) generalization. Inspired by the workflow of human annotators, we reframe RE as a reasoning task guided by annotation guidelines and introduce R1-RE, the first reinforcement learning with verifiable reward (RLVR) framework for RE tasks. Our method elicits the reasoning abilities of small language models for annotation tasks, resulting in significantly improved OOD robustness. We evaluate our approach on the public Sem-2010 dataset and a private MDKG dataset. The R1-RE-7B model attains an average OOD accuracy of approximately 70%, on par with leading proprietary models such as GPT-4o. Additionally, our comprehensive analysis provides novel insights into the training dynamics and emergent reasoning behaviors of the RLVR paradigm for RE.
>
---
#### [replaced 022] Strong Priority and Determinacy in Timed CCS
- **分类: cs.PL; cs.CL**

- **链接: [http://arxiv.org/pdf/2403.04618v4](http://arxiv.org/pdf/2403.04618v4)**

> **作者:** Luigi Liquori; Michael Mendler
>
> **备注:** Change Notes (06.08.25): Streamlined the definition of coherence and non-interference; Corrections in Def.~14 for coherence, adding condition on residual transitions; Adjusted coding of Esterel signals (Ex.~11) to match adjusted Def.~14; To reflect changed Def.~14, use the term "c-coherence''; Minor rewrite of Sec.~2.3 and Sec.~4; Further corrections and revisions in Appendices
>
> **摘要:** Building on the standard theory of process algebra with priorities, we identify a new scheduling mechanism, called "constructive reduction" which is designed to capture the essence of synchronous programming. The distinctive property of this evaluation strategy is to achieve determinacy-by-construction for multi-cast concurrent communication with shared memory. In the technical setting of CCS extended by clocks and priorities, we prove for a large class of "coherent" processes a confluence property for constructive reductions. We show that under some restrictions, called "pivotability", coherence is preserved by the operators of prefix, summation, parallel composition, restriction and hiding. Since this permits memory and sharing, we are able to cover a strictly larger class of processes compared to those in Milner's classical confluence theory for CCS without priorities.
>
---
#### [replaced 023] ContextASR-Bench: A Massive Contextual Speech Recognition Benchmark
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.05727v2](http://arxiv.org/pdf/2507.05727v2)**

> **作者:** He Wang; Linhan Ma; Dake Guo; Xiong Wang; Lei Xie; Jin Xu; Junyang Lin
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** Automatic Speech Recognition (ASR) has been extensively investigated, yet prior benchmarks have largely focused on assessing the acoustic robustness of ASR models, leaving evaluations of their linguistic capabilities relatively underexplored. This largely stems from the limited parameter sizes and training corpora of conventional ASR models, leaving them with insufficient world knowledge, which is crucial for accurately recognizing named entities across diverse domains. For instance, drug and treatment names in medicine or specialized technical terms in engineering. Recent breakthroughs in Large Language Models (LLMs) and corresponding Large Audio Language Models (LALMs) have markedly enhanced the visibility of advanced context modeling and general artificial intelligence capabilities. Leveraging LLMs, we envision a unified system capable of robust speech recognition across diverse real-world domains, yet existing benchmarks are inadequate for evaluating this objective. To address this gap, we propose ContextASR-Bench: a comprehensive, large-scale benchmark designed to assess the linguistic competence of ASR systems using corpora that feature numerous named entities across multiple domains. It encompasses up to 40,000 data entries with more than 300,000 named entities across over 10 domains. Beyond the audio and its transcription, each sample provides the domain it belongs to and a list of named entities it contains, which are referred to as the context. Based on this, we introduce three evaluation modes to assess how effectively models can exploit such context to improve ASR accuracy. Extensive evaluation on ContextASR-Bench highlights that LALMs outperform conventional ASR models by a large margin thanks to the strong world knowledge and context modeling of LLMs, yet there remains ample room for further improvement. The dataset and evaluation code have been released.
>
---
#### [replaced 024] Inside-Out: Hidden Factual Knowledge in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.15299v4](http://arxiv.org/pdf/2503.15299v4)**

> **作者:** Zorik Gekhman; Eyal Ben David; Hadas Orgad; Eran Ofek; Yonatan Belinkov; Idan Szpektor; Jonathan Herzig; Roi Reichart
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** This work presents a framework for assessing whether large language models (LLMs) encode more factual knowledge in their parameters than what they express in their outputs. While a few studies hint at this possibility, none has clearly defined or demonstrated this phenomenon. We first propose a formal definition of knowledge, quantifying it for a given question as the fraction of correct-incorrect answer pairs where the correct one is ranked higher. This gives rise to external and internal knowledge, depending on the information used to score individual answer candidates: either the model's observable token-level probabilities or its intermediate computations. Hidden knowledge arises when internal knowledge exceeds external knowledge. We then present a case study, applying this framework to three popular open-weights LLMs in a closed-book QA setup. Our results indicate that: (1) LLMs consistently encode more factual knowledge internally than what they express externally, with an average relative gap of 40%. (2) Surprisingly, some knowledge is so deeply hidden that a model can internally know an answer perfectly, yet fail to generate it even once, despite large-scale repeated sampling of 1,000 answers. This reveals fundamental limitations in the generation capabilities of LLMs, which (3) put a practical constraint on scaling test-time compute via repeated answer sampling in closed-book QA: significant performance improvements remain inaccessible because some answers are practically never sampled, yet if they were, we would be guaranteed to rank them first.
>
---
#### [replaced 025] CoCoTen: Detecting Adversarial Inputs to Large Language Models through Latent Space Features of Contextual Co-occurrence Tensors
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.02997v2](http://arxiv.org/pdf/2508.02997v2)**

> **作者:** Sri Durga Sai Sowmya Kadali; Evangelos E. Papalexakis
>
> **摘要:** The widespread use of Large Language Models (LLMs) in many applications marks a significant advance in research and practice. However, their complexity and hard-to-understand nature make them vulnerable to attacks, especially jailbreaks designed to produce harmful responses. To counter these threats, developing strong detection methods is essential for the safe and reliable use of LLMs. This paper studies this detection problem using the Contextual Co-occurrence Matrix, a structure recognized for its efficacy in data-scarce environments. We propose a novel method leveraging the latent space characteristics of Contextual Co-occurrence Matrices and Tensors for the effective identification of adversarial and jailbreak prompts. Our evaluations show that this approach achieves a notable F1 score of 0.83 using only 0.5% of labeled prompts, which is a 96.6% improvement over baselines. This result highlights the strength of our learned patterns, especially when labeled data is scarce. Our method is also significantly faster, speedup ranging from 2.3 to 128.4 times compared to the baseline models. To support future research and reproducibility, we have made our implementation publicly available.
>
---
#### [replaced 026] Model Internal Sleuthing: Finding Lexical Identity and Inflectional Morphology in Modern Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.02132v3](http://arxiv.org/pdf/2506.02132v3)**

> **作者:** Michael Li; Nishant Subramani
>
> **备注:** INTERPLAY Workshop COLM 2025
>
> **摘要:** Large transformer-based language models dominate modern NLP, yet our understanding of how they encode linguistic information is rooted in studies of early models like BERT and GPT-2. To better understand today's language models, we investigate how 25 models - from classical architectures (BERT, DeBERTa, GPT-2) to modern large language models (Pythia, OLMo-2, Gemma-2, Qwen2.5, Llama-3.1) - represent lexical identity and inflectional morphology across six typologically diverse languages. Using linear and nonlinear classifiers trained on hidden activations, we predict word lemmas and inflectional features layer by layer. We find that models concentrate lexical information linearly in early layers and increasingly nonlinearly in later layers, while keeping inflectional information uniformly accessible and linearly separable throughout. Additional experiments probe the nature of these encodings: attention and residual analyses examine where within layers information can be recovered, steering vector experiments test what information can be functionally manipulated, and intrinsic dimensionality analyses explore how the representational structure evolves across layers. Remarkably, these encoding patterns emerge across all models we test, despite differences in architecture, size, and training regime (pretrained and instruction-tuned variants). This suggests that, even with substantial advances in LLM technologies, transformer models organize linguistic information in similar ways, indicating that these properties are important for next token prediction and are learned early during pretraining. Our code is available at https://github.com/ml5885/model_internal_sleuthing
>
---
#### [replaced 027] Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2503.09516v5](http://arxiv.org/pdf/2503.09516v5)**

> **作者:** Bowen Jin; Hansi Zeng; Zhenrui Yue; Jinsung Yoon; Sercan Arik; Dong Wang; Hamed Zamani; Jiawei Han
>
> **备注:** 31 pages
>
> **摘要:** Efficiently acquiring external knowledge and up-to-date information is essential for effective reasoning and text generation in large language models (LLMs). Prompting advanced LLMs with reasoning capabilities to use search engines during inference is often suboptimal, as the LLM might not fully possess the capability on how to interact optimally with the search engine. This paper introduces Search-R1, an extension of reinforcement learning (RL) for reasoning frameworks where the LLM learns to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval. Search-R1 optimizes LLM reasoning trajectories with multi-turn search interactions, leveraging retrieved token masking for stable RL training and a simple outcome-based reward function. Experiments on seven question-answering datasets show that Search-R1 improves performance by 41% (Qwen2.5-7B) and 20% (Qwen2.5-3B) over various RAG baselines under the same setting. This paper further provides empirical insights into RL optimization methods, LLM choices, and response length dynamics in retrieval-augmented reasoning. The code and model checkpoints are available at https://github.com/PeterGriffinJin/Search-R1.
>
---
#### [replaced 028] How Far Can LLMs Improve from Experience? Measuring Test-Time Learning Ability in LLMs with Human Comparison
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14448v2](http://arxiv.org/pdf/2506.14448v2)**

> **作者:** Jiayin Wang; Zhiquang Guo; Weizhi Ma; Min Zhang
>
> **摘要:** As evaluation designs of large language models may shape our trajectory toward artificial general intelligence, comprehensive and forward-looking assessment is essential. Existing benchmarks primarily assess static knowledge, while intelligence also entails the ability to rapidly learn from experience. To this end, we advocate for the evaluation of Test-time Learning, the capacity to improve performance in experience-based, reasoning-intensive tasks during test time. In this work, we propose semantic games as effective testbeds for evaluating test-time learning, due to their resistance to saturation and inherent demand for strategic reasoning. We introduce an objective evaluation framework that compares model performance under both limited and cumulative experience settings, and contains four forms of experience representation. To provide a comparative baseline, we recruit eight human participants to complete the same task. Results show that LLMs exhibit measurable test-time learning capabilities; however, their improvements are less stable under cumulative experience and progress more slowly than those observed in humans. These findings underscore the potential of LLMs as general-purpose learning machines, while also revealing a substantial intellectual gap between models and humans, irrespective of how well LLMs perform on static benchmarks.
>
---
#### [replaced 029] Ultra Memory-Efficient On-FPGA Training of Transformers via Tensor-Compressed Optimization
- **分类: cs.LG; cs.AR; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.06663v2](http://arxiv.org/pdf/2501.06663v2)**

> **作者:** Jiayi Tian; Jinming Lu; Hai Li; Xiangwei Wang; Cong Hao; Ian Young; Zheng Zhang
>
> **摘要:** Transformer models have achieved state-of-the-art performance across a wide range of machine learning tasks. There is growing interest in training transformers on resource-constrained edge devices due to considerations such as privacy, domain adaptation, and on-device scientific machine learning. However, the significant computational and memory demands required for transformer training often exceed the capabilities of an edge device. Leveraging low-rank tensor compression, this paper presents the first on-FPGA accelerator for end-to-end transformer training. On the algorithm side, we present a bi-directional contraction flow for tensorized transformer training, significantly reducing the computational FLOPS and intra-layer memory costs compared to existing tensor operations. On the hardware side, we store all highly compressed model parameters and gradient information on chip, creating an on-chip-memory-only framework for each stage in training. This reduces off-chip communication and minimizes latency and energy costs. Additionally, we implement custom computing kernels for each training stage and employ intra-layer parallelism and pipe-lining to further enhance run-time and memory efficiency. Through experiments on transformer models within $36.7$ to $93.5$ MB using FP-32 data formats on the ATIS dataset, our tensorized FPGA accelerator could conduct single-batch end-to-end training on the AMD Alevo U50 FPGA, with a memory budget of less than $6$-MB BRAM and $22.5$-MB URAM. Compared to uncompressed training on the NVIDIA RTX 3090 GPU, our on-FPGA training achieves a memory reduction of $30\times$ to $51\times$. Our FPGA accelerator also achieves up to $3.6\times$ less energy cost per epoch compared with tensor Transformer training on an NVIDIA RTX 3090 GPU.
>
---
#### [replaced 030] AVG-LLaVA: An Efficient Large Multimodal Model with Adaptive Visual Granularity
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.02745v3](http://arxiv.org/pdf/2410.02745v3)**

> **作者:** Zhibin Lan; Liqiang Niu; Fandong Meng; Wenbo Li; Jie Zhou; Jinsong Su
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** Recently, large multimodal models (LMMs) have achieved significant advancements. When dealing with high-resolution images, dominant LMMs typically divide them into multiple local images and a global image, leading to a large number of visual tokens. In this work, we introduce AVG-LLaVA, an LMM that can adaptively select the appropriate visual granularity based on the input image and instruction. Specifically, we first apply the multiple pooling layers to obtain visual tokens at different granularities. Then we propose a visual granularity router, which includes a Transformer layer, an MLP layer, and a voter layer, used to select the appropriate visual granularity based on the image and instruction. Furthermore, we put forward RGLF, a novel training paradigm that aims at aligning the granularity predicted by the router with the preferences of the LMM, without the need for additional manually annotated data. Extensive experiments and analysis show that AVG-LLaVA achieves superior performance across 11 benchmarks, as well as significantly reduces the number of visual tokens and speeds up inference (e.g., an 85.3% reduction in visual tokens and a 2.53$\times$ increase in inference speed on the AI2D benchmark).
>
---
#### [replaced 031] LLMs Have a Heart of Stone: Demystifying the Soft Thinking Ability of Large Reasoning Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.03440v2](http://arxiv.org/pdf/2508.03440v2)**

> **作者:** Chünhung Wu; Jinliang Lu; Zixuan Ren; Gangqiang Hu; Zhi Wu; Dai Dai; Hua Wu
>
> **备注:** 10 pages, 7 figures, working in progress
>
> **摘要:** Human cognition naturally engages with abstract and fluid concepts, whereas existing reasoning models often rely on generating discrete tokens, potentially constraining their expressive capabilities. Recent advancements aim to address this limitation by enabling large language models (LLMs) to generate soft, abstract tokens, thus facilitating reasoning within a continuous concept space. This paper explores the `Soft Thinking' capabilities of various LLMs by examining the models' internal behavior using a suite of probing techniques. Contrary to the common belief that Soft Thinking enables the simultaneous exploration of diverse reasoning paths, our findings reveal that LLMs predominantly rely on the most influential component of the soft inputs during subsequent decoding steps. This reliance hinders the exploration of different reasoning paths and reduces vanilla Soft Thinking to a form of greedy decoding, obscuring the advantage of transmitting more information through Soft Tokens. To tackle this issue, we explore sampling strategies to introduce \emph{randomness}, employing methods such as Dirichlet resampling and the Gumbel-Softmax trick. Our experiments demonstrate that incorporating randomness can alleviate the limitations of vanilla approaches and unleash the potential of Soft Thinking. Notably, the Gumbel-Softmax trick provides adequate randomness with controlled smoothness, resulting in superior performance across eight reasoning benchmarks.
>
---
#### [replaced 032] A Survey of Conversational Search
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2410.15576v2](http://arxiv.org/pdf/2410.15576v2)**

> **作者:** Fengran Mo; Kelong Mao; Ziliang Zhao; Hongjin Qian; Haonan Chen; Yiruo Cheng; Xiaoxi Li; Yutao Zhu; Zhicheng Dou; Jian-Yun Nie
>
> **备注:** 38 pages, 8 figures, corresponding Github repository: https://github.com/fengranMark/ConvSearch-Survey
>
> **摘要:** As a cornerstone of modern information access, search engines have become indispensable in everyday life. With the rapid advancements in AI and natural language processing (NLP) technologies, particularly large language models (LLMs), search engines have evolved to support more intuitive and intelligent interactions between users and systems. Conversational search, an emerging paradigm for next-generation search engines, leverages natural language dialogue to facilitate complex and precise information retrieval, thus attracting significant attention. Unlike traditional keyword-based search engines, conversational search systems enhance user experience by supporting intricate queries, maintaining context over multi-turn interactions, and providing robust information integration and processing capabilities. Key components such as query reformulation, search clarification, conversational retrieval, and response generation work in unison to enable these sophisticated interactions. In this survey, we explore the recent advancements and potential future directions in conversational search, examining the critical modules that constitute a conversational search system. We highlight the integration of LLMs in enhancing these systems and discuss the challenges and opportunities that lie ahead in this dynamic field. Additionally, we provide insights into real-world applications and robust evaluations of current conversational search systems, aiming to guide future research and development in conversational search.
>
---
#### [replaced 033] CAIN: Hijacking LLM-Humans Conversations via Malicious System Prompts
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16888v2](http://arxiv.org/pdf/2505.16888v2)**

> **作者:** Viet Pham; Thai Le
>
> **摘要:** Large language models (LLMs) have advanced many applications, but are also known to be vulnerable to adversarial attacks. In this work, we introduce a novel security threat: hijacking AI-human conversations by manipulating LLMs' system prompts to produce malicious answers only to specific targeted questions (e.g., "Who should I vote for US President?", "Are Covid vaccines safe?"), while behaving benignly on others. This attack is detrimental as it can enable malicious actors to exercise large-scale information manipulation by spreading harmful but benign-looking system prompts online. To demonstrate such an attack, we develop CAIN, an algorithm that can automatically curate such harmful system prompts for a specific target question in a black-box setting or without the need to access the LLM's parameters. Evaluated on both open-source and commercial LLMs, CAIN demonstrates significant adversarial impact. In untargeted attacks or forcing LLMs to output incorrect answers, CAIN achieves up to 40% F1 degradation on targeted questions while preserving high accuracy on benign inputs. For targeted attacks or forcing LLMs to output specific harmful answers, CAIN achieves over 70% F1 scores on these targeted responses with minimal impact on benign questions. Our results highlight the critical need for enhanced robustness measures to safeguard the integrity and safety of LLMs in real-world applications. All source code will be publicly available.
>
---
#### [replaced 034] Learning Optimal Prompt Ensemble for Multi-source Visual Prompt Transfer
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.12311v3](http://arxiv.org/pdf/2504.12311v3)**

> **作者:** Enming Zhang; Liwen Cao; Yanru Wu; Zijie Zhao; Yang Li
>
> **摘要:** Prompt tuning has emerged as a lightweight strategy for adapting foundation models to downstream tasks, particularly for resource-constrained systems. As pre-trained prompts become valuable assets, combining multiple source prompts offers a promising approach to enhance generalization for new tasks by leveraging complementary knowledge. However, naive aggregation often overlooks different source prompts have different contribution potential to the target task. To address this, we propose HGPrompt, a dynamic framework that learns optimal ensemble weights. These weights are optimized by jointly maximizing an information-theoretic metric for transferability and minimizing gradient conflicts via a novel regularization strategy. Specifically, we propose a differentiable prompt transferability metric to captures the discriminability of prompt-induced features on the target task. Meanwhile, HGPrompt match the gradient variances with respect to different source prompts based on Hessian and Fisher Information, ensuring stable and coherent knowledge transfer while suppressing gradient conflicts among them. Extensive experiments on the large-scale VTAB benchmark demonstrate the state-of-the-art performance of HGPrompt, validating its effectiveness in learning an optimal ensemble for effective multi-source prompt transfer.
>
---
#### [replaced 035] Improved Unbiased Watermark for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11268v3](http://arxiv.org/pdf/2502.11268v3)**

> **作者:** Ruibo Chen; Yihan Wu; Junfeng Guo; Heng Huang
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** As artificial intelligence surpasses human capabilities in text generation, the necessity to authenticate the origins of AI-generated content has become paramount. Unbiased watermarks offer a powerful solution by embedding statistical signals into language model-generated text without distorting the quality. In this paper, we introduce MCmark, a family of unbiased, Multi-Channel-based watermarks. MCmark works by partitioning the model's vocabulary into segments and promoting token probabilities within a selected segment based on a watermark key. We demonstrate that MCmark not only preserves the original distribution of the language model but also offers significant improvements in detectability and robustness over existing unbiased watermarks. Our experiments with widely-used language models demonstrate an improvement in detectability of over 10% using MCmark, compared to existing state-of-the-art unbiased watermarks. This advancement underscores MCmark's potential in enhancing the practical application of watermarking in AI-generated texts.
>
---
#### [replaced 036] Assessing Agentic Large Language Models in Multilingual National Bias
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17945v2](http://arxiv.org/pdf/2502.17945v2)**

> **作者:** Qianying Liu; Katrina Qiyao Wang; Fei Cheng; Sadao Kurohashi
>
> **备注:** Accepted to ACL 2025 Findings. 14 pages
>
> **摘要:** Large Language Models have garnered significant attention for their capabilities in multilingual natural language processing, while studies on risks associated with cross biases are limited to immediate context preferences. Cross-language disparities in reasoning-based recommendations remain largely unexplored, with a lack of even descriptive analysis. This study is the first to address this gap. We test LLM's applicability and capability in providing personalized advice across three key scenarios: university applications, travel, and relocation. We investigate multilingual bias in state-of-the-art LLMs by analyzing their responses to decision-making tasks across multiple languages. We quantify bias in model-generated scores and assess the impact of demographic factors and reasoning strategies (e.g., Chain-of-Thought prompting) on bias patterns. Our findings reveal that local language bias is prevalent across different tasks, with GPT-4 and Sonnet reducing bias for English-speaking countries compared to GPT-3.5 but failing to achieve robust multilingual alignment, highlighting broader implications for multilingual AI agents and applications such as education. \footnote{Code available at: https://github.com/yiyunya/assess_agentic_national_bias
>
---
#### [replaced 037] CLaSP: Learning Concepts for Time-Series Signals from Natural Language Supervision
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.08397v3](http://arxiv.org/pdf/2411.08397v3)**

> **作者:** Aoi Ito; Kota Dohi; Yohei Kawaguchi
>
> **摘要:** This paper presents CLaSP, a novel model for retrieving time-series signals using natural language queries that describe signal characteristics. The ability to search time-series signals based on descriptive queries is essential in domains such as industrial diagnostics, where data scientists often need to find signals with specific characteristics. However, existing methods rely on sketch-based inputs, predefined synonym dictionaries, or domain-specific manual designs, limiting their scalability and adaptability. CLaSP addresses these challenges by employing contrastive learning to map time-series signals to natural language descriptions. Unlike prior approaches, it eliminates the need for predefined synonym dictionaries and leverages the rich contextual knowledge of large language models (LLMs). Using the TRUCE and SUSHI datasets, which pair time-series signals with natural language descriptions, we demonstrate that CLaSP achieves high accuracy in retrieving a variety of time series patterns based on natural language queries.
>
---
#### [replaced 038] Beyond Adapter Retrieval: Latent Geometry-Preserving Composition via Sparse Task Projection
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.09908v2](http://arxiv.org/pdf/2410.09908v2)**

> **作者:** Pengfei Jin; Peng Shu; Sifan Song; Sekeun Kim; Qing Xiao; Cheng Chen; Tianming Liu; Xiang Li; Quanzheng Li
>
> **摘要:** Recent advances in parameter-efficient transfer learning have demonstrated the utility of composing LoRA adapters from libraries of pretrained modules. However, most existing approaches rely on simple retrieval heuristics or uniform averaging, which overlook the latent structure of task relationships in representation space. We propose a new framework for adapter reuse that moves beyond retrieval, formulating adapter composition as a geometry-aware sparse reconstruction problem. Specifically, we represent each task by a latent prototype vector derived from the base model's encoder and aim to approximate the target task prototype as a sparse linear combination of retrieved reference prototypes, under an $\ell_1$-regularized optimization objective. The resulting combination weights are then used to blend the corresponding LoRA adapters, yielding a composite adapter tailored to the target task. This formulation not only preserves the local geometric structure of the task representation manifold, but also promotes interpretability and efficient reuse by selecting a minimal set of relevant adapters. We demonstrate the effectiveness of our approach across multiple domains-including medical image segmentation, medical report generation and image synthesis. Our results highlight the benefit of coupling retrieval with latent geometry-aware optimization for improved zero-shot generalization.
>
---
#### [replaced 039] A Comparative Study of Specialized LLMs as Dense Retrievers
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.03958v2](http://arxiv.org/pdf/2507.03958v2)**

> **作者:** Hengran Zhang; Keping Bi; Jiafeng Guo
>
> **备注:** Accepted by CCIR25 and published by Springer LNCS or LNAI
>
> **摘要:** While large language models (LLMs) are increasingly deployed as dense retrievers, the impact of their domain-specific specialization on retrieval effectiveness remains underexplored. This investigation systematically examines how task-specific adaptations in LLMs influence their retrieval capabilities, an essential step toward developing unified retrievers capable of handling text, code, images, and multimodal content. We conduct extensive experiments with eight Qwen2.5 7B LLMs, including base, instruction-tuned, code/math-specialized, long reasoning, and vision-language models across zero-shot retrieval settings and the supervised setting. For the zero-shot retrieval settings, we consider text retrieval from the BEIR benchmark and code retrieval from the CoIR benchmark. Further, to evaluate supervised performance, all LLMs are fine-tuned on the MS MARCO dataset. We find that mathematical specialization and the long reasoning capability cause consistent degradation in three settings, indicating conflicts between mathematical reasoning and semantic matching. The vision-language model and code-specialized LLMs demonstrate superior zero-shot performance compared to other LLMs, even surpassing BM25 on the code retrieval task, and maintain comparable performance to base LLMs in supervised settings. These findings suggest promising directions for the unified retrieval task leveraging cross-domain and cross-modal fusion.
>
---
#### [replaced 040] Basis Selection: Low-Rank Decomposition of Pretrained Large Language Models for Target Applications
- **分类: cs.LG; cs.AR; cs.CL**

- **链接: [http://arxiv.org/pdf/2405.15877v2](http://arxiv.org/pdf/2405.15877v2)**

> **作者:** Yang Li; Daniel Agyei Asante; Changsheng Zhao; Ernie Chang; Yangyang Shi; Vikas Chandra
>
> **摘要:** Large language models (LLMs) significantly enhance the performance of various applications, but they are computationally intensive and energy-demanding. This makes it challenging to deploy them on devices with limited resources, such as personal computers and mobile/wearable devices, and results in substantial inference costs in resource-rich environments like cloud servers. To extend the use of LLMs, we introduce a low-rank decomposition approach to effectively compress these models, tailored to the requirements of specific applications. We observe that LLMs pretrained on general datasets contain many redundant components not needed for particular applications. Our method focuses on identifying and removing these redundant parts, retaining only the necessary elements for the target applications. Specifically, we represent the weight matrices of LLMs as a linear combination of base components. We then prune the irrelevant bases and enhance the model with new bases beneficial for specific applications. Deep compression results on the Llama 2-7b and -13B models, conducted on target applications including mathematical reasoning and code generation, show that our method significantly reduces model size while maintaining comparable accuracy to state-of-the-art low-rank compression techniques.
>
---
#### [replaced 041] LinkQA: Synthesizing Diverse QA from Multiple Seeds Strongly Linked by Knowledge Points
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.01317v2](http://arxiv.org/pdf/2508.01317v2)**

> **作者:** Xuemiao Zhang; Can Ren; Chengying Tu; Rongxiang Weng; Hongfei Yan; Jingang Wang; Xunliang Cai
>
> **摘要:** The advancement of large language models (LLMs) struggles with the scarcity of high-quality, diverse training data. To address this limitation, we propose LinkSyn, a novel knowledge point (KP) graph-based synthesis framework that enables flexible control over discipline and difficulty distributions while balancing KP coverage and popularity. LinkSyn extracts KPs from question-answering (QA) seed data and constructs a KP graph to synthesize diverse QA data from multiple seeds strongly linked by KPs and sampled from graph walks. Specifically, LinkSyn incorporates (1) a knowledge distribution value function to guide the adjustment of path sampling probability and balance KP coverage and popularity during graph walks; (2) diffusion-based synthesis via DeepSeek-R1 by leveraging multiple seeds with dense logical associations along each path; and (3) high-difficulty QA enhancement within given disciplines by flexible difficulty adjustments. By executing LinkSyn, we synthesize LinkQA, a diverse multi-disciplinary QA dataset with 50B tokens. Extensive experiments on Llama-3 8B demonstrate that continual pre-training with LinkQA yields an average improvement of $\mathbf{11.51\%}$ on MMLU and CMMLU, establishing new SOTA results. LinkQA consistently enhances performance across model size and initial FLOPs scales.
>
---
#### [replaced 042] SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.18892v3](http://arxiv.org/pdf/2503.18892v3)**

> **作者:** Weihao Zeng; Yuzhen Huang; Qian Liu; Wei Liu; Keqing He; Zejun Ma; Junxian He
>
> **备注:** Published as a conference paper at COLM 2025
>
> **摘要:** DeepSeek-R1 has shown that long chain-of-thought (CoT) reasoning can naturally emerge through a simple reinforcement learning (RL) framework with rule-based rewards, where the training may directly start from the base models-a paradigm referred to as zero RL training. Most recent efforts to reproduce zero RL training have primarily focused on the Qwen2.5 model series, which may not be representative as we find the base models already exhibit strong instruction-following and self-reflection abilities. In this work, we investigate zero RL training across 10 diverse base models, spanning different families and sizes including LLama3-8B, Mistral-7B/24B, DeepSeek-Math-7B, Qwen2.5-math-7B, and all Qwen2.5 models from 0.5B to 32B. Leveraging several key design strategies-such as adjusting format reward and controlling query difficulty-we achieve substantial improvements in both reasoning accuracy and response length across most settings. However, by carefully monitoring the training dynamics, we observe that different base models exhibit distinct patterns during training. For instance, the increased response length does not always correlate with the emergence of certain cognitive behaviors such as verification (i.e., the "aha moment"). Notably, we observe the "aha moment" for the first time in small models not from the Qwen family. We share the key designs that enable successful zero RL training, along with our findings and practices. To facilitate further research, we open-source the code, models, and analysis tools.
>
---
#### [replaced 043] Text-Only Reasoning Unleashes Zero-Shot Multimodal Evaluators
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18601v2](http://arxiv.org/pdf/2505.18601v2)**

> **作者:** Jongwoo Ko; Sungnyun Kim; Sungwoo Cho; Se-Young Yun
>
> **备注:** The code is available at https://github.com/jongwooko/flex-judge
>
> **摘要:** Human-generated reward signals are critical for aligning generative models with human preferences, guiding both training and inference-time evaluations. While large language models (LLMs) employed as proxy evaluators, i.e., LLM-as-a-Judge, significantly reduce the costs associated with manual annotations, they typically require extensive modality-specific training data and fail to generalize well across diverse multimodal tasks. In this paper, we propose Flex-Judge, a reasoning-guided multimodal judge model that leverages minimal textual reasoning data to robustly generalize across multiple modalities and evaluation formats. Our core intuition is that structured textual reasoning explanations inherently encode generalizable decision-making patterns, enabling an effective transfer to multimodal judgments, e.g., with images or videos. Empirical results demonstrate that Flex-Judge, despite being trained on significantly fewer text data, achieves competitive or superior performance compared to state-of-the-art commercial APIs and extensively trained multimodal evaluators. Notably, Flex-Judge presents broad impact in modalities like molecule, where comprehensive evaluation benchmarks are scarce, underscoring its practical value in resource-constrained domains. Our framework highlights reasoning-based text supervision as a powerful, cost-effective alternative to traditional annotation-intensive approaches, substantially advancing scalable multimodal model-as-a-judge.
>
---
#### [replaced 044] From Sufficiency to Reflection: Reinforcement-Guided Thinking Quality in Retrieval-Augmented Reasoning for LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.22716v2](http://arxiv.org/pdf/2507.22716v2)**

> **作者:** Jie He; Victor Gutiérrez-Basulto; Jeff Z. Pan
>
> **摘要:** Reinforcement learning-based retrieval-augmented generation (RAG) methods enhance the reasoning abilities of large language models (LLMs). However, most rely only on final-answer rewards, overlooking intermediate reasoning quality. This paper analyzes existing RAG reasoning models and identifies three main failure patterns: (1) information insufficiency, meaning the model fails to retrieve adequate support; (2) faulty reasoning, where logical or content-level flaws appear despite sufficient information; and (3) answer-reasoning inconsistency, where a valid reasoning chain leads to a mismatched final answer. We propose TIRESRAG-R1, a novel framework using a think-retrieve-reflect process and a multi-dimensional reward system to improve reasoning and stability. TIRESRAG-R1 introduces: (1) a sufficiency reward to encourage thorough retrieval; (2) a reasoning quality reward to assess the rationality and accuracy of the reasoning chain; and (3) a reflection reward to detect and revise errors. It also employs a difficulty-aware reweighting strategy and training sample filtering to boost performance on complex tasks. Experiments on four multi-hop QA datasets show that TIRESRAG-R1 outperforms prior RAG methods and generalizes well to single-hop tasks. The code and data are available at: https://github.com/probe2/TIRESRAG-R1.
>
---
#### [replaced 045] FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging
- **分类: cs.CL; cs.CE**

- **链接: [http://arxiv.org/pdf/2506.05828v2](http://arxiv.org/pdf/2506.05828v2)**

> **作者:** Zichen Tang; Haihong E; Ziyan Ma; Haoyang He; Jiacheng Liu; Zhongjun Yang; Zihua Rong; Rongjin Li; Kun Ji; Qing Huang; Xinyang Hu; Yang Liu; Qianhe Zheng
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** We introduce FinanceReasoning, a novel benchmark designed to evaluate the reasoning capabilities of large reasoning models (LRMs) in financial numerical reasoning problems. Compared to existing benchmarks, our work provides three key advancements. (1) Credibility: We update 15.6% of the questions from four public datasets, annotating 908 new questions with detailed Python solutions and rigorously refining evaluation standards. This enables an accurate assessment of the reasoning improvements of LRMs. (2) Comprehensiveness: FinanceReasoning covers 67.8% of financial concepts and formulas, significantly surpassing existing datasets. Additionally, we construct 3,133 Python-formatted functions, which enhances LRMs' financial reasoning capabilities through refined knowledge (e.g., 83.2% $\rightarrow$ 91.6% for GPT-4o). (3) Challenge: Models are required to apply multiple financial formulas for precise numerical reasoning on 238 Hard problems. The best-performing model (i.e., OpenAI o1 with PoT) achieves 89.1% accuracy, yet LRMs still face challenges in numerical precision. We demonstrate that combining Reasoner and Programmer models can effectively enhance LRMs' performance (e.g., 83.2% $\rightarrow$ 87.8% for DeepSeek-R1). Our work paves the way for future research on evaluating and improving LRMs in domain-specific complex reasoning tasks.
>
---
#### [replaced 046] Evaluating Robustness of LLMs in Question Answering on Multilingual Noisy OCR Data
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16781v2](http://arxiv.org/pdf/2502.16781v2)**

> **作者:** Bhawna Piryani; Jamshid Mozafari; Abdelrahman Abdallah; Antoine Doucet; Adam Jatowt
>
> **备注:** Accepted at CIKM 2025
>
> **摘要:** Optical Character Recognition (OCR) plays a crucial role in digitizing historical and multilingual documents, yet OCR errors - imperfect extraction of text, including character insertion, deletion, and substitution can significantly impact downstream tasks like question-answering (QA). In this work, we conduct a comprehensive analysis of how OCR-induced noise affects the performance of Multilingual QA Systems. To support this analysis, we introduce a multilingual QA dataset MultiOCR-QA, comprising 50K question-answer pairs across three languages, English, French, and German. The dataset is curated from OCR-ed historical documents, which include different levels and types of OCR noise. We then evaluate how different state-of-the-art Large Language models (LLMs) perform under different error conditions, focusing on three major OCR error types. Our findings show that QA systems are highly prone to OCR-induced errors and perform poorly on noisy OCR text. By comparing model performance on clean versus noisy texts, we provide insights into the limitations of current approaches and emphasize the need for more noise-resilient QA systems in historical digitization contexts.
>
---
#### [replaced 047] NameTag 3: A Tool and a Service for Multilingual/Multitagset NER
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05949v2](http://arxiv.org/pdf/2506.05949v2)**

> **作者:** Jana Straková; Milan Straka
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** We introduce NameTag 3, an open-source tool and cloud-based web service for multilingual, multidataset, and multitagset named entity recognition (NER), supporting both flat and nested entities. NameTag 3 achieves state-of-the-art results on 21 test datasets in 15 languages and remains competitive on the rest, even against larger models. It is available as a command-line tool and as a cloud-based service, enabling use without local installation. NameTag 3 web service currently provides flat NER for 17 languages, trained on 21 corpora and three NE tagsets, all powered by a single 355M-parameter fine-tuned model; and nested NER for Czech, powered by a 126M fine-tuned model. The source code is licensed under open-source MPL 2.0, while the models are distributed under non-commercial CC BY-NC-SA 4.0. Documentation is available at https://ufal.mff.cuni.cz/nametag, source code at https://github.com/ufal/nametag3, and trained models via https://lindat.cz. The REST service and the web application can be found at https://lindat.mff.cuni.cz/services/nametag/. A demonstration video is available at https://www.youtube.com/watch?v=-gaGnP0IV8A.
>
---
#### [replaced 048] Towards Domain Specification of Embedding Models in Medicine
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.19407v2](http://arxiv.org/pdf/2507.19407v2)**

> **作者:** Mohammad Khodadad; Ali Shiraee Kasmaee; Mahdi Astaraki; Hamidreza Mahyar
>
> **摘要:** Medical text embedding models are foundational to a wide array of healthcare applications, ranging from clinical decision support and biomedical information retrieval to medical question answering, yet they remain hampered by two critical shortcomings. First, most models are trained on a narrow slice of medical and biological data, beside not being up to date in terms of methodology, making them ill suited to capture the diversity of terminology and semantics encountered in practice. Second, existing evaluations are often inadequate: even widely used benchmarks fail to generalize across the full spectrum of real world medical tasks. To address these gaps, we leverage MEDTE, a GTE model extensively fine-tuned on diverse medical corpora through self-supervised contrastive learning across multiple data sources, to deliver robust medical text embeddings. Alongside this model, we propose a comprehensive benchmark suite of 51 tasks spanning classification, clustering, pair classification, and retrieval modeled on the Massive Text Embedding Benchmark (MTEB) but tailored to the nuances of medical text. Our results demonstrate that this combined approach not only establishes a robust evaluation framework but also yields embeddings that consistently outperform state of the art alternatives in different tasks.
>
---
#### [replaced 049] From Queries to Criteria: Understanding How Astronomers Evaluate LLMs
- **分类: cs.CL; astro-ph.IM**

- **链接: [http://arxiv.org/pdf/2507.15715v2](http://arxiv.org/pdf/2507.15715v2)**

> **作者:** Alina Hyk; Kiera McCormick; Mian Zhong; Ioana Ciucă; Sanjib Sharma; John F Wu; J. E. G. Peek; Kartheik G. Iyer; Ziang Xiao; Anjalie Field
>
> **备注:** Accepted to the Conference on Language Modeling 2025 (COLM), 22 pages, 6 figures
>
> **摘要:** There is growing interest in leveraging LLMs to aid in astronomy and other scientific research, but benchmarks for LLM evaluation in general have not kept pace with the increasingly diverse ways that real people evaluate and use these models. In this study, we seek to improve evaluation procedures by building an understanding of how users evaluate LLMs. We focus on a particular use case: an LLM-powered retrieval-augmented generation bot for engaging with astronomical literature, which we deployed via Slack. Our inductive coding of 368 queries to the bot over four weeks and our follow-up interviews with 11 astronomers reveal how humans evaluated this system, including the types of questions asked and the criteria for judging responses. We synthesize our findings into concrete recommendations for building better benchmarks, which we then employ in constructing a sample benchmark for evaluating LLMs for astronomy. Overall, our work offers ways to improve LLM evaluation and ultimately usability, particularly for use in scientific research.
>
---
#### [replaced 050] How Well Do LLMs Represent Values Across Cultures? Empirical Analysis of LLM Responses Based on Hofstede Cultural Dimensions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.14805v2](http://arxiv.org/pdf/2406.14805v2)**

> **作者:** Julia Kharchenko; Tanya Roosta; Aman Chadha; Chirag Shah
>
> **备注:** KDD 2025
>
> **摘要:** Large Language Models (LLMs) attempt to imitate human behavior by responding to humans in a way that pleases them, including by adhering to their values. However, humans come from diverse cultures with different values. It is critical to understand whether LLMs showcase different values to the user based on the stereotypical values of a user's known country. We prompt different LLMs with a series of advice requests based on 5 Hofstede Cultural Dimensions -- a quantifiable way of representing the values of a country. Throughout each prompt, we incorporate personas representing 36 different countries and, separately, languages predominantly tied to each country to analyze the consistency in the LLMs' cultural understanding. Through our analysis of the responses, we found that LLMs can differentiate between one side of a value and another, as well as understand that countries have differing values, but will not always uphold the values when giving advice, and fail to understand the need to answer differently based on different cultural values. Rooted in these findings, we present recommendations for training value-aligned and culturally sensitive LLMs. More importantly, the methodology and the framework developed here can help further understand and mitigate culture and language alignment issues with LLMs.
>
---
#### [replaced 051] RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.00222v3](http://arxiv.org/pdf/2508.00222v3)**

> **作者:** Yihong Dong; Xue Jiang; Yongding Tao; Huanyu Liu; Kechi Zhang; Lili Mou; Rongyu Cao; Yingwei Ma; Jue Chen; Binhua Li; Zhi Jin; Fei Huang; Yongbin Li; Ge Li
>
> **摘要:** Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its essentially on-policy strategy coupled with LLM's immense action space and sparse reward. Critically, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel hybrid-policy optimization approach for LLMs that synergizes internal exploitation with external data to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components, i.e., Multiple Importance Sampling to address distributional mismatch from external data, and Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. Compared with existing RLVR methods, RL-PLUS achieves 1) state-of-the-art performance on six math reasoning benchmarks; 2) superior performance on six out-of-distribution reasoning tasks; 3) consistent and significant gains across diverse model families, with average relative improvements up to 69.2\%. Moreover, the analysis of Pass@k curves indicates that RL-PLUS effectively resolves the capability boundary collapse problem.
>
---
#### [replaced 052] p-MoD: Building Mixture-of-Depths MLLMs via Progressive Ratio Decay
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.04449v2](http://arxiv.org/pdf/2412.04449v2)**

> **作者:** Jun Zhang; Desen Meng; Zhengming Zhang; Zhenpeng Huang; Tao Wu; Limin Wang
>
> **备注:** Accepted by ICCV 2025; Code released at https://github.com/MCG-NJU/p-MoD
>
> **摘要:** Despite the remarkable performance of multimodal large language models (MLLMs) across diverse tasks, the substantial training and inference costs impede their advancement. In this paper, we propose p-MoD, an efficient MLLM architecture that significantly reduces training and inference costs while maintaining model performance. The majority of computation in MLLMs stems from the overwhelming volume of vision tokens processed by the transformer-based LLM. Accordingly, we leverage the Mixture-of-Depths (MoD) mechanism, where each LLM layer selects essential vision tokens to process while skipping redundant ones. However, integrating MoD into MLLMs is non-trivial. To address the challenges of training and inference stability as well as limited training data, we adapt the MoD module with two novel designs: tanh-gated weight normalization (TanhNorm) and symmetric token reweighting (STRing). Moreover, we observe that vision tokens exhibit higher redundancy in deeper layers and thus design a progressive ratio decay (PRD) strategy, which gradually reduces the token retention ratio layer by layer, employing a shifted cosine schedule. This crucial design fully unleashes the potential of MoD, significantly boosting the efficiency and performance of our models. Extensive experiments on two baseline models across 15 benchmarks show that our model matches or even surpasses the performance of corresponding baselines, while requiring only 55.6% TFLOPs and 53.7% KV cache storage during inference, and 77.7% GPU hours during training.
>
---
#### [replaced 053] Explain Less, Understand More: Jargon Detection via Personalized Parameter-Efficient Fine-tuning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16227v2](http://arxiv.org/pdf/2505.16227v2)**

> **作者:** Bohao Wu; Qingyun Wang; Yue Guo
>
> **摘要:** Personalizing jargon detection and explanation is essential for making technical documents accessible to readers with diverse disciplinary backgrounds. However, tailoring models to individual users typically requires substantial annotation efforts and computational resources due to user-specific finetuning. To address this, we present a systematic study of personalized jargon detection, focusing on methods that are both efficient and scalable for real-world deployment. We explore two personalization strategies: (1) lightweight fine-tuning using Low-Rank Adaptation (LoRA) on open-source models, and (2) personalized prompting, which tailors model behavior at inference time without retaining. To reflect realistic constraints, we also investigate hybrid approaches that combine limited annotated data with unsupervised user background signals. Our personalized LoRA model outperforms GPT-4 by 21.4% in F1 score and exceeds the best performing oracle baseline by 8.3%. Remarkably, our method achieves comparable performance using only 10% of the annotated training data, demonstrating its practicality for resource-constrained settings. Our study offers the first work to systematically explore efficient, low-resource personalization of jargon detection using open-source language models, offering a practical path toward scalable, user-adaptive NLP system.
>
---
#### [replaced 054] Automatically Interpreting Millions of Features in Large Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.13928v3](http://arxiv.org/pdf/2410.13928v3)**

> **作者:** Gonçalo Paulo; Alex Mallen; Caden Juang; Nora Belrose
>
> **摘要:** While the activations of neurons in deep neural networks usually do not have a simple human-understandable interpretation, sparse autoencoders (SAEs) can be used to transform these activations into a higher-dimensional latent space which may be more easily interpretable. However, these SAEs can have millions of distinct latent features, making it infeasible for humans to manually interpret each one. In this work, we build an open-source automated pipeline to generate and evaluate natural language explanations for SAE features using LLMs. We test our framework on SAEs of varying sizes, activation functions, and losses, trained on two different open-weight LLMs. We introduce five new techniques to score the quality of explanations that are cheaper to run than the previous state of the art. One of these techniques, intervention scoring, evaluates the interpretability of the effects of intervening on a feature, which we find explains features that are not recalled by existing methods. We propose guidelines for generating better explanations that remain valid for a broader set of activating contexts, and discuss pitfalls with existing scoring techniques. We use our explanations to measure the semantic similarity of independently trained SAEs, and find that SAEs trained on nearby layers of the residual stream are highly similar. Our large-scale analysis confirms that SAE latents are indeed much more interpretable than neurons, even when neurons are sparsified using top-$k$ postprocessing. Our code is available at https://github.com/EleutherAI/sae-auto-interp, and our explanations are available at https://huggingface.co/datasets/EleutherAI/auto_interp_explanations.
>
---
#### [replaced 055] CRAB: A Benchmark for Evaluating Curation of Retrieval-Augmented LLMs in Biomedicine
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.12342v2](http://arxiv.org/pdf/2504.12342v2)**

> **作者:** Hanmeng Zhong; Linqing Chen; Wentao Wu; Weilei Wang
>
> **摘要:** Recent development in Retrieval-Augmented Large Language Models (LLMs) have shown great promise in biomedical applications. How ever, a critical gap persists in reliably evaluating their curation ability the process by which models select and integrate relevant references while filtering out noise. To address this, we introduce the benchmark for Curation of Retrieval-Augmented LLMs in Biomedicine (CRAB), the first multilingual benchmark tailored for evaluating the biomedical curation of retrieval-augmented LLMs, available in English, French, German and Chinese. By incorporating a novel citation-based evaluation metric, CRAB quantifies the curation performance of retrieval-augmented LLMs in biomedicine. Experimental results reveal significant discrepancies in the curation performance of mainstream LLMs, underscoring the urgent need to improve it in the domain of biomedicine. Our dataset is available at https://huggingface.co/datasets/zhm0/CRAB.
>
---
#### [replaced 056] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14194v4](http://arxiv.org/pdf/2504.14194v4)**

> **作者:** Xinlin Zhuang; Jiahui Peng; Ren Ma; Yinfan Wang; Tianyi Bai; Xingjian Wei; Jiantao Qiu; Chi Zhang; Ying Qian; Conghui He
>
> **备注:** ACL 2025 Best Theme Paper Award
>
> **摘要:** The composition of pre-training datasets for large language models (LLMs) remains largely undisclosed, hindering transparency and efforts to optimize data quality, a critical driver of model performance. Current data selection methods, such as natural language quality assessments, diversity-based filters, and classifier-based approaches, are limited by single-dimensional evaluation or redundancy-focused strategies. To address these gaps, we propose four dimensions to evaluate data quality: professionalism, readability, reasoning, and cleanliness. We further introduce Meta-rater,a multi-dimensional data selection method that integrates these dimensions with existing quality metrics through learned optimal weightings. Meta-rater employs proxy models to train a regression model that predicts validation loss, enabling the identification of optimal combinations of quality scores. Experiments demonstrate that Meta-rater doubles convergence speed for 1.3B parameter models and improves downstream task performance by 3.23, with advantages that scale to models as large as 7.2B parameters. Our work establishes that holistic, multi-dimensional quality integration significantly outperforms conventional single-dimension approaches, offering a scalable paradigm for enhancing pre-training efficiency and model capability. To advance future research, we release scripts, data, and models at https://github.com/opendatalab/Meta-rater.
>
---
#### [replaced 057] Evaluating the Robustness of Multimodal Agents Against Active Environmental Injection Attacks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13053v3](http://arxiv.org/pdf/2502.13053v3)**

> **作者:** Yurun Chen; Xavier Hu; Keting Yin; Juncheng Li; Shengyu Zhang
>
> **备注:** Accepted at ACM MM 2025 Main Conference
>
> **摘要:** As researchers continue to optimize AI agents for more effective task execution within operating systems, they often overlook a critical security concern: the ability of these agents to detect "impostors" within their environment. Through an analysis of the agents' operational context, we identify a significant threat-attackers can disguise malicious attacks as environmental elements, injecting active disturbances into the agents' execution processes to manipulate their decision-making. We define this novel threat as the Active Environment Injection Attack (AEIA). Focusing on the interaction mechanisms of the Android OS, we conduct a risk assessment of AEIA and identify two critical security vulnerabilities: (1) Adversarial content injection in multimodal interaction interfaces, where attackers embed adversarial instructions within environmental elements to mislead agent decision-making; and (2) Reasoning gap vulnerabilities in the agent's task execution process, which increase susceptibility to AEIA attacks during reasoning. To evaluate the impact of these vulnerabilities, we propose AEIA-MN, an attack scheme that exploits interaction vulnerabilities in mobile operating systems to assess the robustness of MLLM-based agents. Experimental results show that even advanced MLLMs are highly vulnerable to this attack, achieving a maximum attack success rate of 93% on the AndroidWorld benchmark by combining two vulnerabilities.
>
---
#### [replaced 058] FactEHR: A Dataset for Evaluating Factuality in Clinical Notes Using LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.12422v2](http://arxiv.org/pdf/2412.12422v2)**

> **作者:** Monica Munnangi; Akshay Swaminathan; Jason Alan Fries; Jenelle Jindal; Sanjana Narayanan; Ivan Lopez; Lucia Tu; Philip Chung; Jesutofunmi A. Omiye; Mehr Kashyap; Nigam Shah
>
> **备注:** To appear at MLHC 2025
>
> **摘要:** Verifying and attributing factual claims is essential for the safe and effective use of large language models (LLMs) in healthcare. A core component of factuality evaluation is fact decomposition, the process of breaking down complex clinical statements into fine-grained atomic facts for verification. Recent work has proposed fact decomposition, which uses LLMs to rewrite source text into concise sentences conveying a single piece of information, to facilitate fine-grained fact verification. However, clinical documentation poses unique challenges for fact decomposition due to dense terminology and diverse note types and remains understudied. To address this gap and explore these challenges, we present FactEHR, an NLI dataset consisting of document fact decompositions for 2,168 clinical notes spanning four types from three hospital systems, resulting in 987,266 entailment pairs. We assess the generated facts on different axes, from entailment evaluation of LLMs to a qualitative analysis. Our evaluation, including review by the clinicians, reveals substantial variability in LLM performance for fact decomposition. For example, Gemini-1.5-Flash consistently generates relevant and accurate facts, while Llama-3 8B produces fewer and less consistent outputs. The results underscore the need for better LLM capabilities to support factual verification in clinical text.
>
---
#### [replaced 059] Fairness Definitions in Language Models Explained
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.18454v2](http://arxiv.org/pdf/2407.18454v2)**

> **作者:** Avash Palikhe; Zichong Wang; Zhipeng Yin; Wenbin Zhang
>
> **摘要:** Language Models (LMs) have demonstrated exceptional performance across various Natural Language Processing (NLP) tasks. Despite these advancements, LMs can inherit and amplify societal biases related to sensitive attributes such as gender and race, limiting their adoption in real-world applications. Therefore, fairness has been extensively explored in LMs, leading to the proposal of various fairness notions. However, the lack of clear agreement on which fairness definition to apply in specific contexts and the complexity of understanding the distinctions between these definitions can create confusion and impede further progress. To this end, this paper proposes a systematic survey that clarifies the definitions of fairness as they apply to LMs. Specifically, we begin with a brief introduction to LMs and fairness in LMs, followed by a comprehensive, up-to-date overview of existing fairness notions in LMs and the introduction of a novel taxonomy that categorizes these concepts based on their transformer architecture: encoder-only, decoder-only, and encoder-decoder LMs. We further illustrate each definition through experiments, showcasing their practical implications and outcomes. Finally, we discuss current research challenges and open questions, aiming to foster innovative ideas and advance the field. The repository is publicly available online at https://github.com/vanbanTruong/Fairness-in-Large-Language-Models/tree/main/definitions.
>
---
#### [replaced 060] Mixup Model Merge: Enhancing Model Merging Performance through Randomized Linear Interpolation
- **分类: cs.CL; I.2.7; I.2.6**

- **链接: [http://arxiv.org/pdf/2502.15434v3](http://arxiv.org/pdf/2502.15434v3)**

> **作者:** Yue Zhou; Yi Chang; Yuan Wu
>
> **备注:** 15 pages
>
> **摘要:** Model merging aims to integrate multiple task-specific models into a unified model that inherits the capabilities of the task-specific models, without additional training. Existing model merging methods often lack consideration of the varying contribution ratios of different task-specific models to the final merged model. In this paper, we propose Mixup Model Merge (M3), a simple yet effective method inspired by the randomized linear interpolation strategy from the Mixup data augmentation technique. M3 performs randomized linear interpolation in parameter space between two task-specific LLMs, where interpolation coefficients are sampled from a Beta distribution to explore diverse contribution ratios. This controllable randomness allows M3 to outperform standard equal-ratio merging by discovering better contribution ratio combinations. Extensive experiments show that M3 significantly (1) improves merged LLM performance across tasks, (2) enhances out-of-distribution and adversarial robustness, (3) outperforms the positive effects of the sparsification method DARE on model merging and can be further combined with DARE to achieve superior results, and (4) balances exploration efficiency and diversity in contribution ratios by tuning the Beta distribution's shape parameters. The code is provided in the supplementary materials.
>
---
#### [replaced 061] Parse Trees Guided LLM Prompt Compression
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.15395v2](http://arxiv.org/pdf/2409.15395v2)**

> **作者:** Wenhao Mao; Chengbin Hou; Tianyu Zhang; Xinyu Lin; Ke Tang; Hairong Lv
>
> **备注:** IEEE TPAMI major revision submitted
>
> **摘要:** Offering rich contexts to Large Language Models (LLMs) has shown to boost the performance in various tasks, but the resulting longer prompt would increase the computational cost and might exceed the input limit of LLMs. Recently, some prompt compression methods have been suggested to shorten the length of prompts by using language models to generate shorter prompts or by developing computational models to select important parts of original prompt. The generative compression methods would suffer from issues like hallucination, while the selective compression methods have not involved linguistic rules and overlook the global structure of prompt. To this end, we propose a novel selective compression method called PartPrompt. It first obtains a parse tree for each sentence based on linguistic rules, and calculates local information entropy for each node in a parse tree. These local parse trees are then organized into a global tree according to the hierarchical structure such as the dependency of sentences, paragraphs, and sections. After that, the root-ward propagation and leaf-ward propagation are proposed to adjust node values over the global tree. Finally, a recursive algorithm is developed to prune the global tree based on the adjusted node values. The experiments show that PartPrompt receives the state-of-the-art performance across various datasets, metrics, compression ratios, and target LLMs for inference. The in-depth ablation studies confirm the effectiveness of designs in PartPrompt, and other additional experiments also demonstrate its superiority in terms of the coherence of compressed prompts and in the extreme long prompt scenario.
>
---
#### [replaced 062] EdgeInfinite-Instruct: Bridging SFT-Based Optimization and NPU-Level Efficiency for Edge Devices
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.00370v2](http://arxiv.org/pdf/2508.00370v2)**

> **作者:** Jiyu Chen; Poh Seng Lim; Shuang Peng; Daxiong Luo; JungHau Foo; Yap Deep; Timothy Lee Jun Jie; Kelvin Teh Kae Wen; Fan Yang; Danyu Feng; Hao-Yun Chen; Peng-Wen Chen; Fangyuan Li; Xiaoxin Chen; Wong Wai Mun
>
> **备注:** The data and method in the paper need to be re-audited
>
> **摘要:** Deploying Transformer-based large language models (LLMs) on resource-constrained edge devices for long-sequence tasks remains challenging due to the quadratic time complexity of self-attention and growing Key-Value (KV) cache demands. While existing KV cache optimizations improve memory efficiency, they often fail to reduce time to first token (TTFT) and may degrade performance through token pruning. Alternative sequence modeling architectures address some of these limitations, but typically require full retraining and lack infrastructure support. EdgeInfinite offers an efficient solution by fine-tuning only a small subset of parameters, maintaining quality while reducing both computational and memory costs, including improved TTFT. However, its instruction-following ability is limited, and it lacks mobile-specific optimizations. To address these issues, we propose EdgeInfinite-Instruct, which introduces a Segmented Supervised Fine-Tuning (S-SFT) strategy tailored to long-sequence tasks such as summarization and question answering. We further optimized EdgeInfinite-Instruct for efficient deployment on edge NPUs by employing fine-grained post-training quantization (PTQ) to reduce computational demands while maintaining accuracy, and by implementing a fixed-shape computation graph that balances memory usage and on-device efficiency through scenario-specific customization of input token and cache sizes. Experiments on long-context benchmarks and real-world mobile tasks show that our approach improves domain-specific performance while maintaining efficiency on NPU-accelerated edge devices.
>
---
