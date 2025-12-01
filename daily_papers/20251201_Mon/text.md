# 自然语言处理 cs.CL

- **最新发布 125 篇**

- **更新 69 篇**

## 最新发布

#### [new 001] Insight-A: Attribution-aware for Multimodal Misinformation Detection
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对AIGC时代多模态虚假信息检测任务，解决传统方法忽视信息溯源与主观性问题。提出Insight-A框架，通过跨源溯源提示（CAP）和去偏自动提示（ADP）实现伪造源追溯，结合图像描述增强跨模态一致性验证，构建层次化推理管道，提升检测准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.21705v1](https://arxiv.org/pdf/2511.21705v1)**

> **作者:** Junjie Wu; Yumeng Fu; Chen Gong; Guohong Fu
>
> **摘要:** AI-generated content (AIGC) technology has emerged as a prevalent alternative to create multimodal misinformation on social media platforms, posing unprecedented threats to societal safety. However, standard prompting leverages multimodal large language models (MLLMs) to identify the emerging misinformation, which ignores the misinformation attribution. To this end, we present Insight-A, exploring attribution with MLLM insights for detecting multimodal misinformation. Insight-A makes two efforts: I) attribute misinformation to forgery sources, and II) an effective pipeline with hierarchical reasoning that detects distortions across modalities. Specifically, to attribute misinformation to forgery traces based on generation patterns, we devise cross-attribution prompting (CAP) to model the sophisticated correlations between perception and reasoning. Meanwhile, to reduce the subjectivity of human-annotated prompts, automatic attribution-debiased prompting (ADP) is used for task adaptation on MLLMs. Additionally, we design image captioning (IC) to achieve visual details for enhancing cross-modal consistency checking. Extensive experiments demonstrate the superiority of our proposal and provide a new paradigm for multimodal misinformation detection in the era of AIGC.
>
---
#### [new 002] Are LLMs Good Safety Agents or a Propaganda Engine?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）在政治敏感内容上的拒绝行为，旨在区分其是否出于安全策略或政治审查。通过构建PSP数据集，分析七款模型在政治语境下的拒绝行为及对提示注入攻击的脆弱性，发现多数模型存在隐性审查现象，揭示了影响拒绝分布的关键因素。**

- **链接: [https://arxiv.org/pdf/2511.23174v1](https://arxiv.org/pdf/2511.23174v1)**

> **作者:** Neemesh Yadav; Francesco Ortu; Jiarui Liu; Joeun Yook; Bernhard Schölkopf; Rada Mihalcea; Alberto Cazzaniga; Zhijing Jin
>
> **备注:** 15 pages, 7 tables, 4 figures
>
> **摘要:** Large Language Models (LLMs) are trained to refuse to respond to harmful content. However, systematic analyses of whether this behavior is truly a reflection of its safety policies or an indication of political censorship, that is practiced globally by countries, is lacking. Differentiating between safety influenced refusals or politically motivated censorship is hard and unclear. For this purpose we introduce PSP, a dataset built specifically to probe the refusal behaviors in LLMs from an explicitly political context. PSP is built by formatting existing censored content from two data sources, openly available on the internet: sensitive prompts in China generalized to multiple countries, and tweets that have been censored in various countries. We study: 1) impact of political sensitivity in seven LLMs through data-driven (making PSP implicit) and representation-level approaches (erasing the concept of politics); and, 2) vulnerability of models on PSP through prompt injection attacks (PIAs). Associating censorship with refusals on content with masked implicit intent, we find that most LLMs perform some form of censorship. We conclude with summarizing major attributes that can cause a shift in refusal distributions across models and contexts of different countries.
>
---
#### [new 003] LLMs for Low-Resource Dialect Translation Using Context-Aware Prompting: A Case Study on Sylheti
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究低资源方言翻译任务，针对缺乏标注数据的孟加拉语方言锡尔赫特语，提出上下文感知提示框架Sylheti-CAP。通过嵌入词典、语言规则和真实性校验，提升大模型在双向翻译中的准确性与自然度，有效缓解词汇偏差与幻觉问题。**

- **链接: [https://arxiv.org/pdf/2511.21761v1](https://arxiv.org/pdf/2511.21761v1)**

> **作者:** Tabia Tanzin Prama; Christopher M. Danforth; Peter Sheridan Dodds
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong translation abilities through prompting, even without task-specific training. However, their effectiveness in dialectal and low-resource contexts remains underexplored. This study presents the first systematic investigation of LLM-based machine translation (MT) for Sylheti, a dialect of Bangla that is itself low-resource. We evaluate five advanced LLMs (GPT-4.1, GPT-4.1, LLaMA 4, Grok 3, and DeepSeek V3.2) across both translation directions (Bangla $\Leftrightarrow$ Sylheti), and find that these models struggle with dialect-specific vocabulary. To address this, we introduce Sylheti-CAP (Context-Aware Prompting), a three-step framework that embeds a linguistic rulebook, a dictionary (2{,}260 core vocabulary items and idioms), and an authenticity check directly into prompts. Extensive experiments show that Sylheti-CAP consistently improves translation quality across models and prompting strategies. Both automatic metrics and human evaluations confirm its effectiveness, while qualitative analysis reveals notable reductions in hallucinations, ambiguities, and awkward phrasing, establishing Sylheti-CAP as a scalable solution for dialectal and low-resource MT. Dataset link: \href{https://github.com/TabiaTanzin/LLMs-for-Low-Resource-Dialect-Translation-Using-Context-Aware-Prompting-A-Case-Study-on-Sylheti.git}{https://github.com/TabiaTanzin/LLMs-for-Low-Resource-Dialect-Translation-Using-Context-Aware-Prompting-A-Case-Study-on-Sylheti.git}
>
---
#### [new 004] Polarity-Aware Probing for Quantifying Latent Alignment in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型对齐性评估问题，提出极性感知的无监督探针PA-CCS，通过检测模型在有害与安全语句极性反转下的内部表示一致性，量化其隐式知识的语义鲁棒性。工作包括构建三组配对数据集，评估16个模型，揭示架构与层间差异，并验证方法有效性。**

- **链接: [https://arxiv.org/pdf/2511.21737v1](https://arxiv.org/pdf/2511.21737v1)**

> **作者:** Sabrina Sadiekh; Elena Ericheva; Chirag Agarwal
>
> **备注:** 7 pages
>
> **摘要:** Advances in unsupervised probes such as Contrast-Consistent Search (CCS), which reveal latent beliefs without relying on token outputs, raise the question of whether these methods can reliably assess model alignment. We investigate this by examining the sensitivity of CCS to harmful vs. safe statements and by introducing Polarity-Aware CCS (PA-CCS), a method for evaluating whether a model's internal representations remain consistent under polarity inversion. We propose two alignment-oriented metrics, Polar-Consistency and the Contradiction Index, to quantify the semantic robustness of a model's latent knowledge. To validate PA-CCS, we curate two main datasets and one control dataset containing matched harmful-safe sentence pairs constructed using different methodologies (concurrent and antagonistic statements). We apply PA-CCS to 16 language models. Our results show that PA-CCS identifies both architectural and layer-specific differences in the encoding of latent harmful knowledge. Notably, replacing the negation token with a meaningless marker degrades PA-CCS scores for models with well-aligned internal representations, while models lacking robust internal calibration do not exhibit this degradation. Our findings highlight the potential of unsupervised probing for alignment evaluation and emphasize the need to incorporate structural robustness checks into interpretability benchmarks. Code and datasets are available at: https://github.com/SadSabrina/polarity-probing. WARNING: This paper contains potentially sensitive, harmful, and offensive content.
>
---
#### [new 005] MCP vs RAG vs NLWeb vs HTML: A Comparison of the Effectiveness and Efficiency of Different Agent Interfaces to the Web (Technical Report)
- **分类: cs.CL**

- **简介: 该论文比较了LLM代理访问网页的四种接口（HTML、RAG、MCP、NLWeb）在任务执行中的效果与效率。通过构建模拟电商环境，测试不同接口下代理完成搜索、比价、下单等任务的表现，发现RAG、MCP和NLWeb显著优于传统HTML，其中RAG结合GPT 5表现最佳。**

- **链接: [https://arxiv.org/pdf/2511.23281v1](https://arxiv.org/pdf/2511.23281v1)**

> **作者:** Aaron Steiner; Ralph Peeters; Christian Bizer
>
> **摘要:** Large language model agents are increasingly used to automate web tasks such as product search, offer comparison, and checkout. Current research explores different interfaces through which these agents interact with websites, including traditional HTML browsing, retrieval-augmented generation (RAG) over pre-crawled content, communication via Web APIs using the Model Context Protocol (MCP), and natural-language querying through the NLWeb interface. However, no prior work has compared these four architectures within a single controlled environment using identical tasks. To address this gap, we introduce a testbed consisting of four simulated e-shops, each offering its products via HTML, MCP, and NLWeb interfaces. For each interface (HTML, RAG, MCP, and NLWeb) we develop specialized agents that perform the same sets of tasks, ranging from simple product searches and price comparisons to complex queries for complementary or substitute products and checkout processes. We evaluate the agents using GPT 4.1, GPT 5, GPT 5 mini, and Claude Sonnet 4 as underlying LLM. Our evaluation shows that the RAG, MCP and NLWeb agents outperform HTML on both effectiveness and efficiency. Averaged over all tasks, F1 rises from 0.67 for HTML to between 0.75 and 0.77 for the other agents. Token usage falls from about 241k for HTML to between 47k and 140k per task. The runtime per task drops from 291 seconds to between 50 and 62 seconds. The best overall configuration is RAG with GPT 5 achieving an F1 score of 0.87 and a completion rate of 0.79. Also taking cost into consideration, RAG with GPT 5 mini offers a good compromise between API usage fees and performance. Our experiments show the choice of the interaction interface has a substantial impact on both the effectiveness and efficiency of LLM-based web agents.
>
---
#### [new 006] Language-conditioned world model improves policy generalization by reading environmental descriptions
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言条件下的世界模型以提升智能体策略泛化能力。针对现有方法依赖专家示范或推理延迟的问题，提出LED-WM模型，通过语言感知编码器将动态描述与环境实体对齐，实现无规划、无示范的训练。实验表明，该方法在新游戏上泛化性能更优，并支持基于合成轨迹的策略微调。**

- **链接: [https://arxiv.org/pdf/2511.22904v1](https://arxiv.org/pdf/2511.22904v1)**

> **作者:** Anh Nguyen; Stefan Lee
>
> **备注:** NeuRIPS 2025. Workshop: LAW 2025: Bridging Language, Agent, and World Models
>
> **摘要:** To interact effectively with humans in the real world, it is important for agents to understand language that describes the dynamics of the environment--that is, how the environment behaves--rather than just task instructions specifying "what to do". Understanding this dynamics-descriptive language is important for human-agent interaction and agent behavior. Recent work address this problem using a model-based approach: language is incorporated into a world model, which is then used to learn a behavior policy. However, these existing methods either do not demonstrate policy generalization to unseen games or rely on limiting assumptions. For instance, assuming that the latency induced by inference-time planning is tolerable for the target task or expert demonstrations are available. Expanding on this line of research, we focus on improving policy generalization from a language-conditioned world model while dropping these assumptions. We propose a model-based reinforcement learning approach, where a language-conditioned world model is trained through interaction with the environment, and a policy is learned from this model--without planning or expert demonstrations. Our method proposes Language-aware Encoder for Dreamer World Model (LED-WM) built on top of DreamerV3. LED-WM features an observation encoder that uses an attention mechanism to explicitly ground language descriptions to entities in the observation. We show that policies trained with LED-WM generalize more effectively to unseen games described by novel dynamics and language compared to other baselines in several settings in two environments: MESSENGER and MESSENGER-WM.To highlight how the policy can leverage the trained world model before real-world deployment, we demonstrate the policy can be improved through fine-tuning on synthetic test trajectories generated by the world model.
>
---
#### [new 007] RAG System for Supporting Japanese Litigation Procedures: Faithful Response Generation Complying with Legal Norms
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究如何构建符合日本医疗诉讼法律规范的RAG系统，解决专家角色替代中的合规性问题。针对禁止使用私有知识、生成内容需忠实于上下文及时间戳匹配等要求，设计了满足三项约束的RAG系统架构。**

- **链接: [https://arxiv.org/pdf/2511.22858v1](https://arxiv.org/pdf/2511.22858v1)**

> **作者:** Yuya Ishihara; Atsushi Keyaki; Hiroaki Yamada; Ryutaro Ohara; Mihoko Sumida
>
> **备注:** This is a preprint version of a paper reviewed and accepted at BREV-RAG 2025: Beyond Relevance-based EValuation of RAG Systems, a SIGIR-AP 2025 workshop
>
> **摘要:** This study discusses the essential components that a Retrieval-Augmented Generation (RAG)-based LLM system should possess in order to support Japanese medical litigation procedures complying with legal norms. In litigation, expert commissioners, such as physicians, architects, accountants, and engineers, provide specialized knowledge to help judges clarify points of dispute. When considering the substitution of these expert roles with a RAG-based LLM system, the constraint of strict adherence to legal norms is imposed. Specifically, three requirements arise: (1) the retrieval module must retrieve appropriate external knowledge relevant to the disputed issues in accordance with the principle prohibiting the use of private knowledge, (2) the responses generated must originate from the context provided by the RAG and remain faithful to that context, and (3) the retrieval module must reference external knowledge with appropriate timestamps corresponding to the issues at hand. This paper discusses the design of a RAG-based LLM system that satisfies these requirements.
>
---
#### [new 008] Quantifying and Mitigating Selection Bias in LLMs: A Transferable LoRA Fine-Tuning and Efficient Majority Voting Approach
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文针对大语言模型在多选题问答中的选择偏差问题，提出无标签的排列偏差度量（PBM）、高效的批量缓存投票方法（BaQCKV）及基于PBM的LoRA微调策略，有效量化并缓解偏差，提升模型一致性与计算效率。**

- **链接: [https://arxiv.org/pdf/2511.21709v1](https://arxiv.org/pdf/2511.21709v1)**

> **作者:** Blessed Guda; Lawrence Francis; Gabrial Zencha Ashungafac; Carlee Joe-Wong; Moise Busogi
>
> **备注:** Accepted into IJCNLP-AACL 2026
>
> **摘要:** Multiple Choice Question (MCQ) answering is a widely used method for evaluating the performance of Large Language Models (LLMs). However, LLMs often exhibit selection bias in MCQ tasks, where their choices are influenced by factors like answer position or option symbols rather than the content. This bias undermines the reliability of MCQ as an evaluation framework. Most existing selection bias metrics require answer labels and measure divergences between prediction and answer distributions, but do not fully capture the consistency of a model's predictions across different orderings of answer choices. Existing selection bias mitigation strategies have notable limitations: majority voting, though effective, is computationally prohibitive; calibration-based methods require validation sets and often fail to generalize across datasets. To address these gaps, we propose three key contributions: (1) a new unsupervised label-free Permutation Bias Metric (PBM) that directly quantifies inconsistencies in model predictions across answer permutations, providing a more precise measure of selection bias, (2) an efficient majority voting approach called Batch Question-Context KV caching (BaQCKV), to significantly reduce computational costs while preserving bias mitigation effectiveness, and (3) an unsupervised Low-Rank Adaptation (LoRA-1) fine-tuning strategy based on our proposed metric and the BaQCKV that mitigates selection bias, providing a computationally efficient alternative that maintains model generalizability. Experiments across multiple MCQ benchmarks demonstrate that our approaches reduce bias, increasing consistency in accuracy while minimizing computational costs.
>
---
#### [new 009] A Lightweight Approach to Detection of AI-Generated Texts Using Stylometric Features
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对AI生成文本检测任务，解决现有方法计算成本高、泛化能力差的问题。提出轻量级方法NEULIF，结合文本的风格与可读性特征，采用小型CNN或随机森林模型，实现97%准确率，模型体积小、可在普通CPU运行，兼具高效与高精度。**

- **链接: [https://arxiv.org/pdf/2511.21744v1](https://arxiv.org/pdf/2511.21744v1)**

> **作者:** Sergey K. Aityan; William Claster; Karthik Sai Emani; Sohni Rais; Thy Tran
>
> **备注:** 19 pages, 6 figures, 3 tables
>
> **摘要:** A growing number of AI-generated texts raise serious concerns. Most existing approaches to AI-generated text detection rely on fine-tuning large transformer models or building ensembles, which are computationally expensive and often provide limited generalization across domains. Existing lightweight alternatives achieved significantly lower accuracy on large datasets. We introduce NEULIF, a lightweight approach that achieves best performance in the lightweight detector class, that does not require extensive computational power and provides high detection accuracy. In our approach, a text is first decomposed into stylometric and readability features which are then used for classification by a compact Convolutional Neural Network (CNN) or Random Forest (RF). Evaluated and tested on the Kaggle AI vs. Human corpus, our models achieve 97% accuracy (~ 0.95 F1) for CNN and 95% accuracy (~ 0.94 F1) for the Random Forest, demonstrating high precision and recall, with ROC-AUC scores of 99.5% and 95%, respectively. The CNN (~ 25 MB) and Random Forest (~ 10.6 MB) models are orders of magnitude smaller than transformer-based ensembles and can be run efficiently on standard CPU devices, without sacrificing accuracy.This study also highlights the potential of such models for broader applications across languages, domains, and streaming contexts, showing that simplicity, when guided by structural insights, can rival complexity in AI-generated content detection.
>
---
#### [new 010] Factors That Support Grounded Responses in LLM Conversations: A Rapid Review
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对大语言模型对话对齐技术的快速综述，旨在解决模型输出偏离用户意图、缺乏上下文一致性和幻觉问题。研究基于PRISMA和PICO框架，系统梳理了推理时、后训练及强化学习三类对齐方法，发现推理时方法在不重训情况下最高效，能有效提升响应质量与可靠性。**

- **链接: [https://arxiv.org/pdf/2511.21762v1](https://arxiv.org/pdf/2511.21762v1)**

> **作者:** Gabriele Cesar Iwashima; Claudia Susie Rodrigues; Claudio Dipolitto; Geraldo Xexéo
>
> **备注:** 28 pages, 1 figure, 3 tables
>
> **摘要:** Large language models (LLMs) may generate outputs that are misaligned with user intent, lack contextual grounding, or exhibit hallucinations during conversation, which compromises the reliability of LLM-based applications. This review aimed to identify and analyze techniques that align LLM responses with conversational goals, ensure grounding, and reduce hallucination and topic drift. We conducted a Rapid Review guided by the PRISMA framework and the PICO strategy to structure the search, filtering, and selection processes. The alignment strategies identified were categorized according to the LLM lifecycle phase in which they operate: inference-time, post-training, and reinforcement learning-based methods. Among these, inference-time approaches emerged as particularly efficient, aligning outputs without retraining while supporting user intent, contextual grounding, and hallucination mitigation. The reviewed techniques provided structured mechanisms for improving the quality and reliability of LLM responses across key alignment objectives.
>
---
#### [new 011] Extension Condition "violations" and Merge optimality constraints
- **分类: cs.CL; math.RA**

- **简介: 该论文研究语法中“扩展条件”（EC）是否被违反的问题，针对头-头移动、助词化等现象，提出通过侧向合并（Sideward Merge）可避免EC违规，并在数学框架下证明EC具有代数本质。研究表明，多数现象可通过非EC路径解释，仅头-头移动需最小优化代价，且与相位和格关系的彩色操作子兼容。**

- **链接: [https://arxiv.org/pdf/2511.22582v1](https://arxiv.org/pdf/2511.22582v1)**

> **作者:** Matilde Marcolli; Richard Larson; Riny Huijbregts
>
> **备注:** 85 pages
>
> **摘要:** We analyze, using the mathematical formulation of Merge within the Strong Minimalist Thesis framework, a set of linguistic phenomena, including head-to-head movement, phrasal affixes and syntactic cliticization, verb-particle alternation, and operator-variable phenomena. These are often regarded as problematic, as violations of the Extension Condition. We show that, in fact, all of these phenomena can be explained without involving any EC violation. We first show that derivations using Sideward Merge are possible for all of these cases: these respect EC, though they involve some amount of optimality violations, with respect to Resource Restrictions cost functions, andthe amount of violation differs among these cases. We show that all the cases that involve large optimality violations can be derived in alternative ways involving neither EC nor the use of SM. The main remaining case (head-to-head movement) only involves SM with minimal violations of optimality (near equilibrium fluctuations). We analyze explicitly also the cases of multiple wh-fronting, clusters of clitics in Romance languages and possessor agreement construction in Korean, and how an explanation of these phenomena based on SM can be made compatible with the colored operad generators for phases and theta roles. We also show that the EC condition has a clear algebraic meaning in the mathematical formulation of Merge and is therefore an intrinsic structural algebraic constraint of the model, rather than an additional assumption. We also show that the minimal optimality violating SM plays a structural role in the Markovian properties of Merge, and we compare different optimality conditions coming from Minimal Search and from Resource Restriction in terms of their effect on the dynamics of the Hopf algebra Markov chain, in a simple explicit example.
>
---
#### [new 012] Joint Speech and Text Training for LLM-Based End-to-End Spoken Dialogue State Tracking
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文针对端到端语音对话状态追踪（DST）中语音数据稀缺、跨领域泛化差的问题，提出联合训练语音与文本DST数据的方法。通过利用易获取的文本DST数据，提升模型在无语音训练数据的目标领域的泛化能力，有效实现跨域性能提升。**

- **链接: [https://arxiv.org/pdf/2511.22503v1](https://arxiv.org/pdf/2511.22503v1)**

> **作者:** Katia Vendrame; Bolaji Yusuf; Santosh Kesiraju; Šimon Sedláček; Oldřich Plchot; Jan Černocký
>
> **备注:** submitted to ICASSP 2026
>
> **摘要:** End-to-end spoken dialogue state tracking (DST) is made difficult by the tandem of having to handle speech input and data scarcity. Combining speech foundation encoders and large language models has been proposed in recent work as to alleviate some of this difficulty. Although this approach has been shown to result in strong spoken DST models, achieving state-of-the-art performance in realistic multi-turn DST, it struggles to generalize across domains and requires annotated spoken DST training data for each domain of interest. However, collecting such data for every target domain is both costly and difficult. Noting that textual DST data is more easily obtained for various domains, in this work, we propose jointly training on available spoken DST data and written textual data from other domains as a way to achieve cross-domain generalization. We conduct experiments which show the efficacy of our proposed method for getting good cross-domain DST performance without relying on spoken training data from the target domains.
>
---
#### [new 013] R2Q: Towards Robust 2-Bit Large Language Models via Residual Refinement Quantization
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型2比特量化导致精度严重下降的问题，提出R2Q框架。通过两次1比特子量化构建自适应量化格栅，并引入残差精炼机制，提升精度与训练稳定性。在多个模型和任务上验证了其优越性，可无缝集成现有量化训练流程。**

- **链接: [https://arxiv.org/pdf/2511.21736v1](https://arxiv.org/pdf/2511.21736v1)**

> **作者:** Jiayi Chen; Jieqi Shi; Jing Huo; Chen Wu
>
> **摘要:** The rapid progress of Large Language Models (LLMs) has brought substantial computational and memory demands, spurring the adoption of low-bit quantization. While 8-bit and 4-bit formats have become prevalent, extending quantization to 2 bits remains challenging due to severe accuracy degradation. To address this, we propose Residual Refinement Quantization (R2Q)-a novel 2-bit quantization framework that decomposes the process into two sequential 1-bit sub-quantizations, forming an adaptive quantization lattice. Extensive evaluations on Llama, OPT, and Qwen across diverse benchmarks-covering question answering, commonsense reasoning, and language modeling-demonstrate that R2Q consistently outperforms existing 2-bit quantization methods in both fine-grained and coarse-grained settings. By refining quantization through a residual learning mechanism, R2Q enhances performance, improves training stability, and accelerates convergence under extreme compression. Furthermore, its modular design enables seamless integration with existing quantization-aware training (QAT) frameworks.
>
---
#### [new 014] Mapping Clinical Doubt: Locating Linguistic Uncertainty in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在临床文本中对语言不确定性的内部表征，属于自然语言处理中的可解释性任务。针对现有模型对语义不确定性感知不足的问题，构建对比数据集并提出层间敏感性度量MSU，揭示不确定性信息在深层逐步编码的规律，提升模型决策的可解释性与可信度。**

- **链接: [https://arxiv.org/pdf/2511.22402v1](https://arxiv.org/pdf/2511.22402v1)**

> **作者:** Srivarshinee Sridhar; Raghav Kaushik Ravi; Kripabandhu Ghosh
>
> **备注:** Accepted to AAAI'26 SECURE-AI4H Workshop
>
> **摘要:** Large Language Models (LLMs) are increasingly used in clinical settings, where sensitivity to linguistic uncertainty can influence diagnostic interpretation and decision-making. Yet little is known about where such epistemic cues are internally represented within these models. Distinct from uncertainty quantification, which measures output confidence, this work examines input-side representational sensitivity to linguistic uncertainty in medical text. We curate a contrastive dataset of clinical statements varying in epistemic modality (e.g., 'is consistent with' vs. 'may be consistent with') and propose Model Sensitivity to Uncertainty (MSU), a layerwise probing metric that quantifies activation-level shifts induced by uncertainty cues. Our results show that LLMs exhibit structured, depth-dependent sensitivity to clinical uncertainty, suggesting that epistemic information is progressively encoded in deeper layers. These findings reveal how linguistic uncertainty is internally represented in LLMs, offering insight into their interpretability and epistemic reliability.
>
---
#### [new 015] Semantics as a Shield: Label Disguise Defense (LDD) against Prompt Injection in LLM Sentiment Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型情感分类中的提示注入攻击问题，提出标签伪装防御（LDD）策略。通过语义变换或无关替换真实标签（如“正”变“蓝”），使模型隐式学习新映射，阻断攻击指令与输出的直接关联，提升鲁棒性。实验验证其在多种模型上有效恢复性能，语义相关标签更具防御优势。**

- **链接: [https://arxiv.org/pdf/2511.21752v1](https://arxiv.org/pdf/2511.21752v1)**

> **作者:** Yanxi Li; Ruocheng Shan
>
> **摘要:** Large language models are increasingly used for text classification tasks such as sentiment analysis, yet their reliance on natural language prompts exposes them to prompt injection attacks. In particular, class-directive injections exploit knowledge of the model's label set (e.g., positive vs. negative) to override its intended behavior through adversarial instructions. Existing defenses, such as detection-based filters, instruction hierarchies, and signed prompts, either require model retraining or remain vulnerable to obfuscation. This paper introduces Label Disguise Defense (LDD), a lightweight and model-agnostic strategy that conceals true labels by replacing them with semantically transformed or unrelated alias labels(e.g., blue vs. yellow). The model learns these new label mappings implicitly through few-shot demonstrations, preventing direct correspondence between injected directives and decision outputs. We evaluate LDD across nine state-of-the-art models, including GPT-5, GPT-4o, LLaMA3.2, Gemma3, and Mistral variants, under varying few-shot and an adversarial setting. Our results show that the ability of LDD to recover performance lost to the adversarial attack varies across models and alias choices. For every model evaluated, LDD is able to restore a portion of the accuracy degradation caused by the attack. Moreover, for the vast majority of models, we can identify more than one alias pair that achieves higher accuracy than the under-attack baseline, in which the model relies solely on few-shot learning without any defensive mechanism. A linguistic analysis further reveals that semantically aligned alias labels(e.g., good vs. bad) yield stronger robustness than unaligned symbols(e.g., blue vs. yellow). Overall, this study demonstrates that label semantics can serve as an effective defense layer, transforming meaning itself into a shield against prompt injection.
>
---
#### [new 016] An Optimized Machine Learning Classifier for Detecting Fake Reviews Using Extracted Features
- **分类: cs.CL**

- **简介: 该论文针对虚假评论检测任务，解决AI生成评论难以识别的问题。提出融合文本预处理、多模态特征提取、HHO优化特征选择与堆叠集成分类的框架，在公开数据集上实现95.40%准确率，显著提升机器生成文本识别效果。**

- **链接: [https://arxiv.org/pdf/2511.21716v1](https://arxiv.org/pdf/2511.21716v1)**

> **作者:** Shabbir Anees; Anshuman; Ayush Chaurasia; Prathmesh Bogar
>
> **摘要:** It is well known that fraudulent reviews cast doubt on the legitimacy and dependability of online purchases. The most recent development that leads customers towards darkness is the appearance of human reviews in computer-generated (CG) ones. In this work, we present an advanced machine-learning-based system that analyses these reviews produced by AI with remarkable precision. Our method integrates advanced text preprocessing, multi-modal feature extraction, Harris Hawks Optimization (HHO) for feature selection, and a stacking ensemble classifier. We implemented this methodology on a public dataset of 40,432 Original (OR) and Computer-Generated (CG) reviews. From an initial set of 13,539 features, HHO selected the most applicable 1,368 features, achieving an 89.9% dimensionality reduction. Our final stacking model achieved 95.40% accuracy, 92.81% precision, 95.01% recall, and a 93.90% F1-Score, which demonstrates that the combination of ensemble learning and bio-inspired optimisation is an effective method for machine-generated text recognition. Because large-scale review analytics commonly run on cloud platforms, privacy-preserving techniques such as differential approaches and secure outsourcing are essential to protect user data in these systems.
>
---
#### [new 017] Behavior-Equivalent Token: Single-Token Replacement for Long Prompts in LLMs
- **分类: cs.CL**

- **简介: 该论文针对大模型长系统提示导致的推理延迟高、计算成本大问题，提出行为等价令牌（[BE]）方法。通过三阶段训练，用单个令牌替代原长提示，在不访问模型内部、无需额外模型或标注数据的情况下，实现3000倍长度压缩，保留98%性能，显著降低开销并释放上下文空间。**

- **链接: [https://arxiv.org/pdf/2511.23271v1](https://arxiv.org/pdf/2511.23271v1)**

> **作者:** Jiancheng Dong; Pengyue Jia; Jingyu Peng; Maolin Wang; Yuhao Wang; Lixin Su; Xin Sun; Shuaiqiang Wang; Dawei Yin; Xiangyu Zhao
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Carefully engineered system prompts play a critical role in guiding the behavior of LLM agents, but their considerable length introduces significant drawbacks, including increased inference latency, higher computational cost, and reduced effective context length. This raises the question of whether such lengthy prompts can be replaced by a drastically reduced number of tokens while preserving their behavioral effect on downstream tasks. To enable this, we propose a lightweight three-stage training framework that learns a single prompt-specific Behavior-Equivalent token ([BE]). The framework first trains [BE] to encode the natural-language content of the original system prompt via reconstruction, and then distills the prompt 's downstream behavior into this single token. Importantly, our method requires no access to model internals, no auxiliary compression models, and no labeled responses. Empirical evaluations on three datasets show that a single [BE] token achieves up to a 3000x reduction in prompt length, while retaining about 98% of the downstream performance of the original system prompts. This substantially reduces inference cost and leaves almost the entire context window available for user inputs.
>
---
#### [new 018] Multi-chain Graph Refinement and Selection for Reliable Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型推理能力不足的问题，提出多链图精炼与选择框架（MGRS），通过生成多样推理路径、自交叉验证精炼、构建推理图并评估成功率，实现高效可靠推理。显著提升准确率与速度，在多个任务上超越现有方法。**

- **链接: [https://arxiv.org/pdf/2511.23136v1](https://arxiv.org/pdf/2511.23136v1)**

> **作者:** Yujiao Yang; Jing Lian; Linhui Li
>
> **摘要:** The complex reasoning ability of Large Language Models (LLMs) poses a critical bottleneck for their practical applications. Test-time expansion methods such as Tree-of-Thought (ToT) and Graph-of-Thought (GoT) enhance reasoning by introducing intermediate reasoning structures, tree search, or graph-based exploration mechanisms. However, their reasoning strategies suffer from limited diversity, redundant search branches, and inadequate integration and error correction across heterogeneous reasoning paths. To address these limitations, we propose a novel reasoning framework called Multi-chain Graph Refinement & Selection (MGRS), which first generates multiple diverse reasoning trajectories for a given problem, refines candidate responses using a composite self- and cross-verification strategy, then constructs a reasoning relation graph and estimates the success rate of intermediate nodes, and finally computes cumulative success rates to select the most reliable answer and corresponding reasoning trajectory. Experimental results demonstrate that MGRS significantly advances both the reasoning capability and computational efficiency of reasoning enhancement methods. Across six benchmark datasets spanning four distinct tasks, MGRS achieves an average accuracy of 82.9%, outperforming state-of-the-art baselines by a clear margin of 2.1%. Remarkably, on the 24-point game, MGRS attains 100% accuracy for the first time, while delivering a 13.6x speed-up compared to the leading Forest of Thoughts framework.
>
---
#### [new 019] RoSA: Enhancing Parameter-Efficient Fine-Tuning via RoPE-aware Selective Adaptation in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型微调计算成本高的问题，提出RoSA框架。通过关注旋转位置编码（RoPE）在低频维度的激活特性，设计了针对性的注意力增强模块与动态层选择策略，实现更高效、精准的参数高效微调，在多个基准上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.21733v1](https://arxiv.org/pdf/2511.21733v1)**

> **作者:** Dayan Pan; Jingyuan Wang; Yilong Zhou; Jiawei Cheng; Pengyue Jia; Xiangyu Zhao
>
> **备注:** Accepted by AAAI' 26
>
> **摘要:** Fine-tuning large language models is essential for task-specific adaptation, yet it remains computationally prohibitive. Parameter-Efficient Fine-Tuning (PEFT) methods have emerged as a solution, but current approaches typically ignore the distinct roles of model components and the heterogeneous importance across layers, thereby limiting adaptation efficiency. Motivated by the observation that Rotary Position Embeddings (RoPE) induce critical activations in the low-frequency dimensions of attention states, we propose RoPE-aware Selective Adaptation (RoSA), a novel PEFT framework that allocates trainable parameters in a more targeted and effective manner. RoSA comprises a RoPE-aware Attention Enhancement (RoAE) module, which selectively enhances the low-frequency components of RoPE-influenced attention states, and a Dynamic Layer Selection (DLS) strategy that adaptively identifies and updates the most critical layers based on LayerNorm gradient norms. By combining dimension-wise enhancement with layer-wise adaptation, RoSA achieves more targeted and efficient fine-tuning. Extensive experiments on fifteen commonsense and arithmetic benchmarks demonstrate that RoSA outperforms existing mainstream PEFT methods under comparable trainable parameters. The code is available to ease reproducibility at https://github.com/Applied-Machine-Learning-Lab/RoSA.
>
---
#### [new 020] HUMORCHAIN: Theory-Guided Multi-Stage Reasoning for Interpretable Multimodal Humor Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多模态幽默生成任务，解决现有方法缺乏理论指导、生成幽默缺乏认知深度的问题。提出HUMORCHAIN框架，基于幽默理论构建多阶段推理链，融合视觉理解与心理学驱动的幽默推理，实现可解释、可控的幽默图像描述生成，显著提升人类对幽默的偏好与多样性评价。**

- **链接: [https://arxiv.org/pdf/2511.21732v1](https://arxiv.org/pdf/2511.21732v1)**

> **作者:** Jiajun Zhang; Shijia Luo; Ruikang Zhang; Qi Su
>
> **摘要:** Humor, as both a creative human activity and a social binding mechanism, has long posed a major challenge for AI generation. Although producing humor requires complex cognitive reasoning and social understanding, theories of humor suggest that it follows learnable patterns and structures, making it theoretically possible for generative models to acquire them implicitly. In recent years, multimodal humor has become a prevalent form of online communication, especially among Gen Z, highlighting the need for AI systems capable of integrating visual understanding with humorous language generation. However, existing data-driven approaches lack explicit modeling or theoretical grounding of humor, often producing literal descriptions that fail to capture its underlying cognitive mechanisms, resulting in the generated image descriptions that are fluent but lack genuine humor or cognitive depth. To address this limitation, we propose HUMORCHAIN (HUmor-guided Multi-step Orchestrated Reasoning Chain for Image Captioning), a theory-guided multi-stage reasoning framework. It integrates visual semantic parsing, humor- and psychology-based reasoning, and a fine-tuned discriminator for humor evaluation, forming an interpretable and controllable cognitive reasoning chain. To the best of our knowledge, this is the first work to explicitly embed cognitive structures from humor theories into multimodal humor generation, enabling a structured reasoning process from visual understanding to humor creation. Experiments on Meme-Image-No-Text, Oogiri-GO, and OxfordTVG-HIC datasets show that HUMORCHAIN outperforms state-of-the-art baselines in human humor preference, Elo/BT scores, and semantic diversity, demonstrating that theory-driven structured reasoning enables large language models to generate humor aligned with human perception.
>
---
#### [new 021] Towards Improving Interpretability of Language Model Generation through a Structured Knowledge Discovery Approach
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对知识增强文本生成中解释性不足的问题，提出一种任务无关的结构化知识发现方法。通过双层知识架构与层次化注意力机制，提升生成过程的可解释性。在表到文和对话生成任务上验证了模型的有效性，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.23335v1](https://arxiv.org/pdf/2511.23335v1)**

> **作者:** Shuqi Liu; Han Wu; Guanzhi Deng; Jianshu Chen; Xiaoyang Wang; Linqi Song
>
> **摘要:** Knowledge-enhanced text generation aims to enhance the quality of generated text by utilizing internal or external knowledge sources. While language models have demonstrated impressive capabilities in generating coherent and fluent text, the lack of interpretability presents a substantial obstacle. The limited interpretability of generated text significantly impacts its practical usability, particularly in knowledge-enhanced text generation tasks that necessitate reliability and explainability. Existing methods often employ domain-specific knowledge retrievers that are tailored to specific data characteristics, limiting their generalizability to diverse data types and tasks. To overcome this limitation, we directly leverage the two-tier architecture of structured knowledge, consisting of high-level entities and low-level knowledge triples, to design our task-agnostic structured knowledge hunter. Specifically, we employ a local-global interaction scheme for structured knowledge representation learning and a hierarchical transformer-based pointer network as the backbone for selecting relevant knowledge triples and entities. By combining the strong generative ability of language models with the high faithfulness of the knowledge hunter, our model achieves high interpretability, enabling users to comprehend the model output generation process. Furthermore, we empirically demonstrate the effectiveness of our model in both internal knowledge-enhanced table-to-text generation on the RotoWireFG dataset and external knowledge-enhanced dialogue response generation on the KdConv dataset. Our task-agnostic model outperforms state-of-the-art methods and corresponding language models, setting new standards on the benchmark.
>
---
#### [new 022] Accent Placement Models for Rigvedic Sanskrit Text
- **分类: cs.CL**

- **简介: 该论文针对古梵语《梨俱吠陀》文本中缺失的声调标记问题，提出三种自动声调标注方法：全量微调ByT5、BiLSTM-CRF基线与LoRA高效微调。通过构建对照语料库，评估其在字词和音标错误率上的表现，旨在实现准确、高效的声调恢复，支持数字人文与语音处理等下游应用。**

- **链接: [https://arxiv.org/pdf/2511.23088v1](https://arxiv.org/pdf/2511.23088v1)**

> **作者:** Akhil Rajeev P; Annarao Kulkarni
>
> **备注:** Submitted to AACL-IJCNLP 2025
>
> **摘要:** The Rigveda, among the oldest Indian texts in Vedic Sanskrit, employs a distinctive pitch-accent system : udātta, anudātta, svarita whose marks encode melodic and interpretive cues but are often absent from modern e-texts. This work develops a parallel corpus of accented-unaccented ślokas and conducts a controlled comparison of three strategies for automatic accent placement in Rigvedic verse: (i) full fine-tuning of ByT5, a byte-level Transformer that operates directly on Unicode combining marks, (ii) a from-scratch BiLSTM-CRF sequence-labeling baseline, and (iii) LoRA-based parameter-efficient fine-tuning atop ByT5. Evaluation uses Word Error Rate (WER) and Character Error Rate (CER) for orthographic fidelity, plus a task-specific Diacritic Error Rate (DER) that isolates accent edits. Full ByT5 fine-tuning attains the lowest error across all metrics; LoRA offers strong efficiency-accuracy trade-offs, and BiLSTM-CRF serves as a transparent baseline. The study underscores practical requirements for accent restoration - Unicode-safe preprocessing, mark-aware tokenization, and evaluation that separates grapheme from accent errors - and positions heritage-language technology as an emerging NLP area connecting computational modeling with philological and pedagogical aims. Results establish reproducible baselines for Rigvedic accent restoration and provide guidance for downstream tasks such as accent-aware OCR, ASR/chant synthesis, and digital scholarship.
>
---
#### [new 023] Decoding inner speech with an end-to-end brain-to-text neural interface
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出端到端脑-文本（BIT）神经接口，解决传统语音脑机接口需分步解码、无法联合优化的问题。通过跨任务、跨物种预训练编码器，实现对尝试与想象言语的统一建模，并结合小规模音频大模型显著降低词错误率，提升解码性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.21740v1](https://arxiv.org/pdf/2511.21740v1)**

> **作者:** Yizi Zhang; Linyang He; Chaofei Fan; Tingkai Liu; Han Yu; Trung Le; Jingyuan Li; Scott Linderman; Lea Duncker; Francis R Willett; Nima Mesgarani; Liam Paninski
>
> **摘要:** Speech brain-computer interfaces (BCIs) aim to restore communication for people with paralysis by translating neural activity into text. Most systems use cascaded frameworks that decode phonemes before assembling sentences with an n-gram language model (LM), preventing joint optimization of all stages simultaneously. Here, we introduce an end-to-end Brain-to-Text (BIT) framework that translates neural activity into coherent sentences using a single differentiable neural network. Central to our approach is a cross-task, cross-species pretrained neural encoder, whose representations transfer to both attempted and imagined speech. In a cascaded setting with an n-gram LM, the pretrained encoder establishes a new state-of-the-art (SOTA) on the Brain-to-Text '24 and '25 benchmarks. Integrated end-to-end with audio large language models (LLMs) and trained with contrastive learning for cross-modal alignment, BIT reduces the word error rate (WER) of the prior end-to-end method from 24.69% to 10.22%. Notably, we find that small-scale audio LLMs markedly improve end-to-end decoding. Beyond record-setting performance, BIT aligns attempted and imagined speech embeddings to enable cross-task generalization. Altogether, our approach advances the integration of large, diverse neural datasets, paving the way for an end-to-end decoding framework that supports seamless, differentiable optimization.
>
---
#### [new 024] JELV: A Judge of Edit-Level Validity for Evaluation and Automated Reference Expansion in Grammatical Error Correction
- **分类: cs.CL**

- **简介: 该论文针对语法纠错（GEC）中参考答案多样性不足的问题，提出JELV框架，通过自动化验证修正编辑的合法性、忠实性和流畅性，提升评估准确性和模型泛化能力。工作包括构建标注数据集、开发多轮LLM与轻量分类器两种实现方式，并用于扩展参考数据集，显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.21700v1](https://arxiv.org/pdf/2511.21700v1)**

> **作者:** Yuhao Zhan; Yuqing Zhang; Jing Yuan; Qixiang Ma; Zhiqi Yang; Yu Gu; Zemin Liu; Fei Wu
>
> **摘要:** Existing Grammatical Error Correction (GEC) systems suffer from limited reference diversity, leading to underestimated evaluation and restricted model generalization. To address this issue, we introduce the Judge of Edit-Level Validity (JELV), an automated framework to validate correction edits from grammaticality, faithfulness, and fluency. Using our proposed human-annotated Pair-wise Edit-level Validity Dataset (PEVData) as benchmark, JELV offers two implementations: a multi-turn LLM-as-Judges pipeline achieving 90% agreement with human annotators, and a distilled DeBERTa classifier with 85% precision on valid edits. We then apply JELV to reclassify misjudged false positives in evaluation and derive a comprehensive evaluation metric by integrating false positive decoupling and fluency scoring, resulting in state-of-the-art correlation with human judgments. We also apply JELV to filter LLM-generated correction candidates, expanding the BEA19's single-reference dataset containing 38,692 source sentences. Retraining top GEC systems on this expanded dataset yields measurable performance gains. JELV provides a scalable solution for enhancing reference diversity and strengthening both evaluation and model generalization.
>
---
#### [new 025] Named Entity Recognition for the Kurdish Sorani Language: Dataset Creation and Comparative Analysis
- **分类: cs.CL**

- **简介: 该论文针对低资源语言库尔德语索拉尼文，提出首个命名实体识别（NER）数据集，包含64,563个标注词元，并开发了辅助工具。通过对比经典机器学习与神经网络模型，发现条件随机场（CRF）在该任务上表现优于BiLSTM，挑战了神经方法在NLP中普遍更优的假设。**

- **链接: [https://arxiv.org/pdf/2511.22315v1](https://arxiv.org/pdf/2511.22315v1)**

> **作者:** Bakhtawar Abdalla; Rebwar Mala Nabi; Hassan Eshkiki; Fabio Caraffini
>
> **摘要:** This work contributes towards balancing the inclusivity and global applicability of natural language processing techniques by proposing the first 'name entity recognition' dataset for Kurdish Sorani, a low-resource and under-represented language, that consists of 64,563 annotated tokens. It also provides a tool for facilitating this task in this and many other languages and performs a thorough comparative analysis, including classic machine learning models and neural systems. The results obtained challenge established assumptions about the advantage of neural approaches within the context of NLP. Conventional methods, in particular CRF, obtain F1-scores of 0.825, outperforming the results of BiLSTM-based models (0.706) significantly. These findings indicate that simpler and more computationally efficient classical frameworks can outperform neural architectures in low-resource settings.
>
---
#### [new 026] ShoppingComp: Are LLMs Really Ready for Your Shopping Cart?
- **分类: cs.CL**

- **简介: 该论文提出ShoppingComp，一个面向电商场景的综合性评测基准，旨在评估大模型在精准商品检索、专业报告生成及安全决策方面的实际能力。针对现有基准在真实性和可验证性上的不足，该研究构建了120项任务、1026个场景，揭示当前大模型在安全风险识别与信息甄别上存在严重缺陷，推动更可靠、实用的电商智能代理发展。**

- **链接: [https://arxiv.org/pdf/2511.22978v1](https://arxiv.org/pdf/2511.22978v1)**

> **作者:** Huaixiao Tou; Ying Zeng; Cong Ma; Muzhi Li; Minghao Li; Weijie Yuan; He Zhang; Kai Jia
>
> **摘要:** We present ShoppingComp, a challenging real-world benchmark for rigorously evaluating LLM-powered shopping agents on three core capabilities: precise product retrieval, expert-level report generation, and safety critical decision making. Unlike prior e-commerce benchmarks, ShoppingComp introduces highly complex tasks under the principle of guaranteeing real products and ensuring easy verifiability, adding a novel evaluation dimension for identifying product safety hazards alongside recommendation accuracy and report quality. The benchmark comprises 120 tasks and 1,026 scenarios, curated by 35 experts to reflect authentic shopping needs. Results reveal stark limitations of current LLMs: even state-of-the-art models achieve low performance (e.g., 11.22% for GPT-5, 3.92% for Gemini-2.5-Flash). These findings highlight a substantial gap between research benchmarks and real-world deployment, where LLMs make critical errors such as failure to identify unsafe product usage or falling for promotional misinformation, leading to harmful recommendations. ShoppingComp fills the gap and thus establishes a new standard for advancing reliable and practical agents in e-commerce.
>
---
#### [new 027] Bridging the Modality Gap by Similarity Standardization with Pseudo-Positive Samples
- **分类: cs.CL**

- **简介: 该论文针对视觉-语言模型中的模态差距问题，提出一种基于伪正样本的相似度标准化方法。通过构建伪配对样本计算模态专属统计量，统一跨模态相似度尺度，提升异模态检索性能，在多个基准上显著提升召回率。**

- **链接: [https://arxiv.org/pdf/2511.22141v1](https://arxiv.org/pdf/2511.22141v1)**

> **作者:** Shuhei Yamashita; Daiki Shirafuji; Tatsuhiko Saito
>
> **备注:** Accepted to PACLIC2025
>
> **摘要:** Advances in vision-language models (VLMs) have enabled effective cross-modality retrieval. However, when both text and images exist in the database, similarity scores would differ in scale by modality. This phenomenon, known as the modality gap, hinders accurate retrieval. Most existing studies address this issue with manually labeled data, e.g., by fine-tuning VLMs on them. In this work, we propose a similarity standardization approach with pseudo data construction. We first compute the mean and variance of the similarity scores between each query and its paired data in text or image modality. Using these modality-specific statistics, we standardize all similarity scores to compare on a common scale across modalities. These statistics are calculated from pseudo pairs, which are constructed by retrieving the text and image candidates with the highest cosine similarity to each query. We evaluate our method across seven VLMs using two multi-modal QA benchmarks (MMQA and WebQA), where each question requires retrieving either text or image data. Our experimental results show that our method significantly improves retrieval performance, achieving average Recall@20 gains of 64% on MMQA and 28% on WebQA when the query and the target data belong to different modalities. Compared to E5-V, which addresses the modality gap through image captioning, we confirm that our method more effectively bridges the modality gap.
>
---
#### [new 028] A Benchmark for Procedural Memory Retrieval in Language Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对语言智能体在新任务中无法有效检索程序记忆的问题，提出首个分离程序记忆检索与执行的基准。通过构建双语料库，评估六种方法在跨上下文场景下的泛化能力，发现嵌入方法依赖表面词汇而忽视时序结构，而大模型生成的抽象能实现稳定迁移，揭示当前编码器存在架构瓶颈。**

- **链接: [https://arxiv.org/pdf/2511.21730v1](https://arxiv.org/pdf/2511.21730v1)**

> **作者:** Ishant Kohar; Aswanth Krishnan
>
> **摘要:** Current AI agents excel in familiar settings, but fail sharply when faced with novel tasks with unseen vocabularies -- a core limitation of procedural memory systems. We present the first benchmark that isolates procedural memory retrieval from task execution, evaluating whether agents can recognize functionally equivalent procedures that span different object instantiations. Using ALFWorld, we construct dual corpora of expert and LLM-generated trajectories and evaluate six retrieval methods using systematically stratified queries. Our results expose a clear generalization cliff: embedding-based methods perform strongly on familiar contexts, yet degrade considerably on novel ones, while LLM-generated procedural abstractions demonstrate reliable cross-context transfer. Controlled ablations show that although embeddings capture some lexical-level abstraction, they fundamentally treat procedures as unordered bags of words, discarding temporal structure necessary for cross-context transfer. Corpus scale delivers far larger gains than representation enrichment, revealing an architectural ceiling in current encoders. Our benchmark offers the first diagnostic framework separating genuine procedural understanding from surface-level memorization and gives tools for developing retrieval systems capable of dependable generalization. Resources available at our GitHub repository (https://github.com/qpiai/Proced_mem_bench).
>
---
#### [new 029] Lost in the Pipeline: How Well Do Large Language Models Handle Data Preparation?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在数据准备任务中的表现，旨在解决其在处理低质量数据时的自动化能力问题。通过对比通用与微调模型在数据清洗、剖析等任务中的表现，并结合用户研究验证评估模型，评估其相较于传统工具的有效性。**

- **链接: [https://arxiv.org/pdf/2511.21708v1](https://arxiv.org/pdf/2511.21708v1)**

> **作者:** Matteo Spreafico; Ludovica Tassini; Camilla Sancricca; Cinzia Cappiello
>
> **摘要:** Large language models have recently demonstrated their exceptional capabilities in supporting and automating various tasks. Among the tasks worth exploring for testing large language model capabilities, we considered data preparation, a critical yet often labor-intensive step in data-driven processes. This paper investigates whether large language models can effectively support users in selecting and automating data preparation tasks. To this aim, we considered both general-purpose and fine-tuned tabular large language models. We prompted these models with poor-quality datasets and measured their ability to perform tasks such as data profiling and cleaning. We also compare the support provided by large language models with that offered by traditional data preparation tools. To evaluate the capabilities of large language models, we developed a custom-designed quality model that has been validated through a user study to gain insights into practitioners' expectations.
>
---
#### [new 030] Mitigating Semantic Drift: Evaluating LLMs' Efficacy in Psychotherapy through MI Dialogue Summarization
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究LLMs在心理治疗中生成动机访谈（MI）对话摘要的效能，旨在缓解语义漂移问题。通过结合MITI框架设计双阶段标注方案，构建多分类任务，评估不同提示策略下模型对治疗核心要素的理解能力，提供高质量数据集与优化实践。**

- **链接: [https://arxiv.org/pdf/2511.22818v1](https://arxiv.org/pdf/2511.22818v1)**

> **作者:** Vivek Kumar; Pushpraj Singh Rajawat; Eirini Ntoutsi
>
> **摘要:** Recent advancements in large language models (LLMs) have shown their potential across both general and domain-specific tasks. However, there is a growing concern regarding their lack of sensitivity, factual incorrectness in responses, inconsistent expressions of empathy, bias, hallucinations, and overall inability to capture the depth and complexity of human understanding, especially in low-resource and sensitive domains such as psychology. To address these challenges, our study employs a mixed-methods approach to evaluate the efficacy of LLMs in psychotherapy. We use LLMs to generate precise summaries of motivational interviewing (MI) dialogues and design a two-stage annotation scheme based on key components of the Motivational Interviewing Treatment Integrity (MITI) framework, namely evocation, collaboration, autonomy, direction, empathy, and a non-judgmental attitude. Using expert-annotated MI dialogues as ground truth, we formulate multi-class classification tasks to assess model performance under progressive prompting techniques, incorporating one-shot and few-shot prompting. Our results offer insights into LLMs' capacity for understanding complex psychological constructs and highlight best practices to mitigate ``semantic drift" in therapeutic settings. Our work contributes not only to the MI community by providing a high-quality annotated dataset to address data scarcity in low-resource domains but also critical insights for using LLMs for precise contextual interpretation in complex behavioral therapy.
>
---
#### [new 031] Asking LLMs to Verify First is Almost Free Lunch
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型推理错误问题，提出验证优先（VF）策略，通过先验证随机答案再生成解法，激发模型批判性思维。进一步提出迭代式VF（Iter-VF），在测试时低成本提升推理性能。实验表明，该方法在数学、编程等任务上显著优于标准链式思维，且计算开销极低。**

- **链接: [https://arxiv.org/pdf/2511.21734v1](https://arxiv.org/pdf/2511.21734v1)**

> **作者:** Shiguang Wu; Quanming Yao
>
> **摘要:** To enhance the reasoning capabilities of Large Language Models (LLMs) without high costs of training, nor extensive test-time sampling, we introduce Verification-First (VF), a strategy that prompts models to verify a provided candidate answer, even a trivial or random one, before generating a solution. This approach triggers a "reverse reasoning" process that is cognitively easier and complementary to standard forward Chain-of-Thought (CoT), effectively invoking the model's critical thinking to reduce logical errors. We further generalize the VF strategy to Iter-VF, a sequential test-time scaling (TTS) method that iteratively cycles the verification-generation process using the model's previous answer. Extensive experiments across various benchmarks (from mathematical reasoning to coding and agentic tasks) and various LLMs (from open-source 1B to cutting-edge commercial ones) confirm that VF with random answer consistently outperforms standard CoT with minimal computational overhead, and Iter-VF outperforms existing TTS strategies.
>
---
#### [new 032] Scaling Competence, Shrinking Reasoning: Cognitive Signatures in Language Model Learning
- **分类: cs.CL**

- **简介: 该论文研究语言模型在任务微调中的推理行为，类比人类工作记忆，揭示其经历“无推理—尝试推理—有效推理—无需推理”的四阶段发展。发现推理标记长度先增后减，且训练后移除推理仍可保持性能，表明推理是学习的临时支架。提出以推理动态为指标诊断训练阶段与优化训练过程。**

- **链接: [https://arxiv.org/pdf/2511.21743v1](https://arxiv.org/pdf/2511.21743v1)**

> **作者:** Mukul Singh; Ananya Singha; Arjun Radhakrishna; Sumit Gulwani
>
> **摘要:** We analyze reasoning in language models during task-specific fine-tuning and draws parallel between reasoning tokens--intermediate steps generated while solving problem and the human working memory. Drawing from cognitive science, we align training dynamics with the Four Stages of Competence: models initially produce incorrect outputs without reasoning, then begin reasoning (but still fail), eventually reason effectively, and finally solve tasks without explicit reasoning. We find that reasoning token length expands as performance improves, peaks at the stage of conscious competence, then declines as the model internalizes the task. Notably, after training, models retain performance even when reasoning is removed--suggesting it scaffolded learning but is no longer needed. This progression offers actionable insights: reasoning token dynamics can serve as a signal for diagnosing training stage, identifying convergence, and guiding early stopping. We propose metrics to track this trajectory and argue that reasoning behavior is valuable for understanding and optimizing reasoning model training.
>
---
#### [new 033] Building Domain-Specific Small Language Models via Guided Data Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对领域专用小模型训练中数据稀缺与隐私、资源消耗问题，提出基于引导式合成数据生成与数据精炼的低成本训练流程。通过DAPT、DSFT与DPO联合优化，构建3B参数的DiagnosticSLM模型，在工业故障诊断任务上显著优于同类开源模型，验证了其在领域推理与泛化上的有效性。**

- **链接: [https://arxiv.org/pdf/2511.21748v1](https://arxiv.org/pdf/2511.21748v1)**

> **作者:** Aman Kumar; Ekant Muljibhai Amin; Xian Yeow Lee; Lasitha Vidyaratne; Ahmed K. Farahat; Dipanjan D. Ghosh; Yuta Koreeda; Chetan Gupta
>
> **备注:** Accepted at Thirty-Eighth Annual Conference on Innovative Applications of Artificial Intelligence (IAAI-26)
>
> **摘要:** Large Language Models (LLMs) have shown remarkable success in supporting a wide range of knowledge-intensive tasks. In specialized domains, there is growing interest in leveraging LLMs to assist subject matter experts with domain-specific challenges. However, deploying LLMs as SaaS solutions raises data privacy concerns, while many open-source models demand significant computational resources for effective domain adaptation and deployment. A promising alternative is to develop smaller, domain-specialized LLMs, though this approach is often constrained by the lack of high-quality domain-specific training data. In this work, we address these limitations by presenting a cost-efficient and scalable training pipeline that combines guided synthetic data generation from a small seed corpus with bottom-up domain data curation. Our pipeline integrates Domain-Adaptive Pretraining (DAPT), Domain-specific Supervised Fine-tuning (DSFT), and Direct Preference Optimization (DPO) to train effective small-scale models for specialized use cases. We demonstrate this approach through DiagnosticSLM, a 3B-parameter domain-specific model tailored for fault diagnosis, root cause analysis, and repair recommendation in industrial settings. To evaluate model performance, we introduce four domain-specific benchmarks: multiple-choice questions (DiagnosticMCQ), question answering (DiagnosticQA), sentence completion (DiagnosticComp), and summarization (DiagnosticSum). DiagnosticSLM achieves up to 25% accuracy improvement over open-source models of comparable or larger size (2B-9B) on the MCQ task, while also outperforming or matching them in other tasks, demonstrating effective domain-specific reasoning and generalization capabilities.
>
---
#### [new 034] C$^2$DLM: Causal Concept-Guided Diffusion Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出C²DLM，一种基于因果概念引导的扩散语言模型，旨在解决传统AR与DLM模型推理能力不足的问题。通过构建概念级因果图并引导注意力学习因果关系，提升模型在复杂推理任务中的表现，显著改善推理准确性与训练效率。**

- **链接: [https://arxiv.org/pdf/2511.22146v1](https://arxiv.org/pdf/2511.22146v1)**

> **作者:** Kairong Han; Nuanqiao Shan; Ziyu Zhao; Zijing Hu; Xinpeng Dong; Junjian Ye; Lujia Pan; Fei Wu; Kun Kuang
>
> **摘要:** Autoregressive (AR) language models and Diffusion Language Models (DLMs) constitute the two principal paradigms of large language models. However, both paradigms suffer from insufficient reasoning capabilities. Human reasoning inherently relies on causal knowledge and thought, which are reflected in natural language. But in the AR paradigm, language is modeled as next token prediction (a strictly left-to-right, token-by-token order), whereas natural language itself exhibits more flexible causal structures. In the DLM paradigm, the attention mechanism is fully connected, which entirely disregards causal order. To fill this gap, we propose a \underline{\textbf{C}}ausal \underline{\textbf{C}}oncept-Guided \underline{\textbf{D}}iffusion \underline{\textbf{L}}anguage \underline{\textbf{M}}odel (C$^2$DLM). Starting from DLM's fully connected attention, C$^2$DLM first obtains a concept-level causal graph from the teacher model, and then explicitly guides attention to learn causal relationships between concepts. By focusing on causal relationships and avoiding interference from difficult subgoals involving causal inversion, C$^2$DLM improves 12\% with about 3.2 times training speedup in the COT-OrderPerturb task, and achieves an average gain of 1.31\% across six downstream reasoning tasks. More details in the repository ~\href{https://github.com/Kairong-Han/C-2-DLM}{here}.
>
---
#### [new 035] Closing the Performance Gap Between AI and Radiologists in Chest X-Ray Reporting
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对胸片报告中临床发现与导管/线路（L&T）信息报告效率低的问题，提出MAIRA-X多模态AI模型，实现纵向胸片报告自动生成。基于大规模数据训练，显著提升报告的语义质量、临床准确性和L&T报告精度，并通过用户评估验证其与放射科医生报告相当的可靠性，有效缓解工作负荷。**

- **链接: [https://arxiv.org/pdf/2511.21735v1](https://arxiv.org/pdf/2511.21735v1)**

> **作者:** Harshita Sharma; Maxwell C. Reynolds; Valentina Salvatelli; Anne-Marie G. Sykes; Kelly K. Horst; Anton Schwaighofer; Maximilian Ilse; Olesya Melnichenko; Sam Bond-Taylor; Fernando Pérez-García; Vamshi K. Mugu; Alex Chan; Ceylan Colak; Shelby A. Swartz; Motassem B. Nashawaty; Austin J. Gonzalez; Heather A. Ouellette; Selnur B. Erdal; Beth A. Schueler; Maria T. Wetscherek; Noel Codella; Mohit Jain; Shruthi Bannur; Kenza Bouzid; Daniel C. Castro; Stephanie Hyland; Panos Korfiatis; Ashish Khandelwal; Javier Alvarez-Valle
>
> **摘要:** AI-assisted report generation offers the opportunity to reduce radiologists' workload stemming from expanded screening guidelines, complex cases and workforce shortages, while maintaining diagnostic accuracy. In addition to describing pathological findings in chest X-ray reports, interpreting lines and tubes (L&T) is demanding and repetitive for radiologists, especially with high patient volumes. We introduce MAIRA-X, a clinically evaluated multimodal AI model for longitudinal chest X-ray (CXR) report generation, that encompasses both clinical findings and L&T reporting. Developed using a large-scale, multi-site, longitudinal dataset of 3.1 million studies (comprising 6 million images from 806k patients) from Mayo Clinic, MAIRA-X was evaluated on three holdout datasets and the public MIMIC-CXR dataset, where it significantly improved AI-generated reports over the state of the art on lexical quality, clinical correctness, and L&T-related elements. A novel L&T-specific metrics framework was developed to assess accuracy in reporting attributes such as type, longitudinal change and placement. A first-of-its-kind retrospective user evaluation study was conducted with nine radiologists of varying experience, who blindly reviewed 600 studies from distinct subjects. The user study found comparable rates of critical errors (3.0% for original vs. 4.6% for AI-generated reports) and a similar rate of acceptable sentences (97.8% for original vs. 97.4% for AI-generated reports), marking a significant improvement over prior user studies with larger gaps and higher error rates. Our results suggest that MAIRA-X can effectively assist radiologists, particularly in high-volume clinical settings.
>
---
#### [new 036] Modeling Romanized Hindi and Bengali: Dataset Creation and Multilingual LLM Integration
- **分类: cs.CL**

- **简介: 该论文针对南亚地区罗马化印地语和孟加拉语的音译难题，提出一个包含近280万对音译数据的新数据集，并基于Marian架构预训练多语言序列到序列模型。实验表明，该模型在BLEU和CER指标上优于现有方法，有效提升了音译准确率。**

- **链接: [https://arxiv.org/pdf/2511.22769v1](https://arxiv.org/pdf/2511.22769v1)**

> **作者:** Kanchon Gharami; Quazi Sarwar Muhtaseem; Deepti Gupta; Lavanya Elluri; Shafika Showkat Moni
>
> **备注:** Proceedings of the 8th Workshop on Big Data for Cybersecurity (BigCyber)
>
> **摘要:** The development of robust transliteration techniques to enhance the effectiveness of transforming Romanized scripts into native scripts is crucial for Natural Language Processing tasks, including sentiment analysis, speech recognition, information retrieval, and intelligent personal assistants. Despite significant advancements, state-of-the-art multilingual models still face challenges in handling Romanized script, where the Roman alphabet is adopted to represent the phonetic structure of diverse languages. Within the South Asian context, where the use of Romanized script for Indo-Aryan languages is widespread across social media and digital communication platforms, such usage continues to pose significant challenges for cutting-edge multilingual models. While a limited number of transliteration datasets and models are available for Indo-Aryan languages, they generally lack sufficient diversity in pronunciation and spelling variations, adequate code-mixed data for large language model (LLM) training, and low-resource adaptation. To address this research gap, we introduce a novel transliteration dataset for two popular Indo-Aryan languages, Hindi and Bengali, which are ranked as the 3rd and 7th most spoken languages worldwide. Our dataset comprises nearly 1.8 million Hindi and 1 million Bengali transliteration pairs. In addition to that, we pre-train a custom multilingual seq2seq LLM based on Marian architecture using the developed dataset. Experimental results demonstrate significant improvements compared to existing relevant models in terms of BLEU and CER metrics.
>
---
#### [new 037] Addressing Stereotypes in Large Language Models: A Critical Examination and Mitigation
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 该论文聚焦于大语言模型中的偏见问题，旨在识别并缓解其在性别、种族等维度的隐性与显性偏见。通过基准测试与多策略优化，研究发现微调模型在隐性偏见检测上表现提升，但对提示词关键词依赖性强，缺乏深层理解。**

- **链接: [https://arxiv.org/pdf/2511.21711v1](https://arxiv.org/pdf/2511.21711v1)**

> **作者:** Fatima Kazi
>
> **摘要:** Large Language models (LLMs), such as ChatGPT, have gained popularity in recent years with the advancement of Natural Language Processing (NLP), with use cases spanning many disciplines and daily lives as well. LLMs inherit explicit and implicit biases from the datasets they were trained on; these biases can include social, ethical, cultural, religious, and other prejudices and stereotypes. It is important to comprehensively examine such shortcomings by identifying the existence and extent of such biases, recognizing the origin, and attempting to mitigate such biased outputs to ensure fair outputs to reduce harmful stereotypes and misinformation. This study inspects and highlights the need to address biases in LLMs amid growing generative Artificial Intelligence (AI). We utilize bias-specific benchmarks such StereoSet and CrowSPairs to evaluate the existence of various biases in many different generative models such as BERT, GPT 3.5, and ADA. To detect both explicit and implicit biases, we adopt a three-pronged approach for thorough and inclusive analysis. Results indicate fine-tuned models struggle with gender biases but excel at identifying and avoiding racial biases. Our findings also illustrated that despite some cases of success, LLMs often over-rely on keywords in prompts and its outputs. This demonstrates the incapability of LLMs to attempt to truly understand the accuracy and authenticity of its outputs. Finally, in an attempt to bolster model performance, we applied an enhancement learning strategy involving fine-tuning, models using different prompting techniques, and data augmentation of the bias benchmarks. We found fine-tuned models to exhibit promising adaptability during cross-dataset testing and significantly enhanced performance on implicit bias benchmarks, with performance gains of up to 20%.
>
---
#### [new 038] Start Making Sense(s): A Developmental Probe of Attention Specialization Using Lexical Ambiguity
- **分类: cs.CL**

- **简介: 该论文研究预训练语言模型中注意力机制的发育过程，聚焦词义消歧任务。通过分析不同规模模型在词汇歧义场景下的注意力行为，发现小模型（14M）注意力敏感且脆弱，大模型（410M）具更鲁棒的专用注意力头，揭示了注意力机制随模型规模发展的特化规律。**

- **链接: [https://arxiv.org/pdf/2511.21974v1](https://arxiv.org/pdf/2511.21974v1)**

> **作者:** Pamela D. Rivière; Sean Trott
>
> **备注:** 13 pages (main text), 5 figures (main text) 6 pages (appendix), 6 figures (appendix), journal submission to TACL ("a" decision: pre-MIT Press publication version)
>
> **摘要:** Despite an in-principle understanding of self-attention matrix operations in Transformer language models (LMs), it remains unclear precisely how these operations map onto interpretable computations or functions--and how or when individual attention heads develop specialized attention patterns. Here, we present a pipeline to systematically probe attention mechanisms, and we illustrate its value by leveraging lexical ambiguity--where a single word has multiple meanings--to isolate attention mechanisms that contribute to word sense disambiguation. We take a "developmental" approach: first, using publicly available Pythia LM checkpoints, we identify inflection points in disambiguation performance for each LM in the suite; in 14M and 410M, we identify heads whose attention to disambiguating words covaries with overall disambiguation performance across development. We then stress-test the robustness of these heads to stimulus perturbations: in 14M, we find limited robustness, but in 410M, we identify multiple heads with surprisingly generalizable behavior. Then, in a causal analysis, we find that ablating the target heads demonstrably impairs disambiguation performance, particularly in 14M. We additionally reproduce developmental analyses of 14M across all of its random seeds. Together, these results suggest: that disambiguation benefits from a constellation of mechanisms, some of which (especially in 14M) are highly sensitive to the position and part-of-speech of the disambiguating cue; and that larger models (410M) may contain heads with more robust disambiguation behavior. They also join a growing body of work that highlights the value of adopting a developmental perspective when probing LM mechanisms.
>
---
#### [new 039] Sentiment Analysis Of Shopee Product Reviews Using Distilbert
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在高效处理Shopee平台海量英文产品评论。针对人工分析效率低的问题，采用轻量级DistilBERT模型进行情感分类，对比Bert与SVM，结果表明其在保持高准确率（94.8%）的同时，显著提升计算效率，适合大规模电商评论分析。**

- **链接: [https://arxiv.org/pdf/2511.22313v1](https://arxiv.org/pdf/2511.22313v1)**

> **作者:** Zahri Aksa Dautd; Aviv Yuniar Rahman
>
> **备注:** 6 pages, 11 figures
>
> **摘要:** The rapid growth of digital commerce has led to the accumulation of a massive number of consumer reviews on online platforms. Shopee, as one of the largest e-commerce platforms in Southeast Asia, receives millions of product reviews every day containing valuable information regarding customer satisfaction and preferences. Manual analysis of these reviews is inefficient, thus requiring a computational approach such as sentiment analysis. This study examines the use of DistilBERT, a lightweight transformer-based deep learning model, for sentiment classification on Shopee product reviews. The dataset used consists of approximately one million English-language reviews that have been preprocessed and trained using the distilbert-base-uncased model. Evaluation was conducted using accuracy, precision, recall, and F1-score metrics, and compared against benchmark models such as BERT and SVM. The results show that DistilBERT achieved an accuracy of 94.8%, slightly below BERT (95.3%) but significantly higher than SVM (90.2%), with computation time reduced by more than 55%. These findings demonstrate that DistilBERT provides an optimal balance between accuracy and efficiency, making it suitable for large scale sentiment analysis on e-commerce platforms. Keywords: Sentiment Analysis, DistilBERT, Shopee Reviews, Natural Language Processing, Deep Learning, Transformer Models.
>
---
#### [new 040] Social Perceptions of English Spelling Variation on Twitter: A Comparative Analysis of Human and LLM Responses
- **分类: cs.CL**

- **简介: 该论文研究英文网络文本中拼写变异（如funnnn）的社会感知，比较人类与大语言模型（LLM）在正式性、谨慎性和年龄感知上的评价。任务为社会语言学分析，旨在探究人类与LLM对拼写变异的感知是否一致。研究发现两者整体相关性强，但具体分布和类型差异显著。**

- **链接: [https://arxiv.org/pdf/2511.23041v1](https://arxiv.org/pdf/2511.23041v1)**

> **作者:** Dong Nguyen; Laura Rosseel
>
> **摘要:** Spelling variation (e.g. funnnn vs. fun) can influence the social perception of texts and their writers: we often have various associations with different forms of writing (is the text informal? does the writer seem young?). In this study, we focus on the social perception of spelling variation in online writing in English and study to what extent this perception is aligned between humans and large language models (LLMs). Building on sociolinguistic methodology, we compare LLM and human ratings on three key social attributes of spelling variation (formality, carefulness, age). We find generally strong correlations in the ratings between humans and LLMs. However, notable differences emerge when we analyze the distribution of ratings and when comparing between different types of spelling variation.
>
---
#### [new 041] EulerESG: Automating ESG Disclosure Analysis with LLMs
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出EulerESG，一个基于大模型的ESG披露分析系统，旨在解决传统方法难以高效、准确提取异构PDF格式ESG报告中标准对齐信息的问题。通过双通道检索与框架感知的LLM分析，实现高精度自动填充标准指标表，并提供交互式可视化工具，提升分析效率与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.21712v1](https://arxiv.org/pdf/2511.21712v1)**

> **作者:** Yi Ding; Xushuo Tang; Zhengyi Yang; Wenqian Zhang; Simin Wu; Yuxin Huang; Lingjing Lan; Weiyuan Li; Yin Chen; Mingchen Ju; Wenke Yang; Thong Hoang; Mykhailo Klymenko; Xiwei Zu; Wenjie Zhang
>
> **摘要:** Environmental, Social, and Governance (ESG) reports have become central to how companies communicate climate risk, social impact, and governance practices, yet they are still published primarily as long, heterogeneous PDF documents. This makes it difficult to systematically answer seemingly simple questions. Existing tools either rely on brittle rule-based extraction or treat ESG reports as generic text, without explicitly modelling the underlying reporting standards. We present \textbf{EulerESG}, an LLM-powered system for automating ESG disclosure analysis with explicit awareness of ESG frameworks. EulerESG combines (i) dual-channel retrieval and LLM-driven disclosure analysis over ESG reports, and (ii) an interactive dashboard and chatbot for exploration, benchmarking, and explanation. Using four globally recognised companies and twelve SASB sub-industries, we show that EulerESG can automatically populate standard-aligned metric tables with high fidelity (up to 0.95 average accuracy) while remaining practical in end-to-end runtime, and we compare several recent LLM models in this setting. The full implementation, together with a demonstration video, is publicly available at https://github.com/UNSW-database/EulerESG.
>
---
#### [new 042] CSV-Decode: Certifiable Sub-Vocabulary Decoding for Efficient Large Language Model Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型推理中的计算瓶颈问题，提出CSV-Decode方法。通过几何上界构建小规模子词表，实现高效稀疏计算，同时保证精确的top-k认证和ε-认证的softmax近似，显著提升推理速度并降低计算开销。**

- **链接: [https://arxiv.org/pdf/2511.21702v1](https://arxiv.org/pdf/2511.21702v1)**

> **作者:** Dong Liu; Yanxuan Yu; Ben Lengerich
>
> **摘要:** Large language models face significant computational bottlenecks during inference due to the expensive output layer computation over large vocabularies. We present CSV-Decode, a novel approach that uses geometric upper bounds to construct small sub-vocabularies for each decoding step, enabling efficient sparse computation while maintaining dual correctness guarantees: exact top-$k$ certification and $\varepsilon$-certified softmax approximations. Our method clusters vocabulary embeddings offline and uses centroid-plus-radius bounds to identify which tokens can be safely omitted from computation. We provide a complete system implementation with sparse GEMV kernels, multi-GPU sharding, and CUDA Graph optimization. Experimental results demonstrate significant speedup over full vocabulary decoding while maintaining distributional guarantees and low fallback rates. Our code implementation available at \href{https://github.com/FastLM/CSV-Decode}{https://github.com/FastLM/CSV-Decode}.
>
---
#### [new 043] A Multiscale Geometric Method for Capturing Relational Topic Alignment
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文针对科学文献中稀有话题识别与主题演化追踪难题，提出一种多尺度几何方法。融合文本与作者网络数据，利用海林格距离与Ward聚类构建层次化主题树，有效捕捉局部与全局结构，实现对罕见话题的识别及主题随时间平滑演变的可视化，提升主题模型可解释性。**

- **链接: [https://arxiv.org/pdf/2511.21741v1](https://arxiv.org/pdf/2511.21741v1)**

> **作者:** Conrad D. Hougen; Karl T. Pazdernik; Alfred O. Hero
>
> **备注:** 5 pages, 3 figures, 2025 IEEE International Workshop on Computational Advances in Multi-Sensor Adaptive Processing
>
> **摘要:** Interpretable topic modeling is essential for tracking how research interests evolve within co-author communities. In scientific corpora, where novelty is prized, identifying underrepresented niche topics is particularly important. However, contemporary models built from dense transformer embeddings tend to miss rare topics and therefore also fail to capture smooth temporal alignment. We propose a geometric method that integrates multimodal text and co-author network data, using Hellinger distances and Ward's linkage to construct a hierarchical topic dendrogram. This approach captures both local and global structure, supporting multiscale learning across semantic and temporal dimensions. Our method effectively identifies rare-topic structure and visualizes smooth topic drift over time. Experiments highlight the strength of interpretable bag-of-words models when paired with principled geometric alignment.
>
---
#### [new 044] Ambiguity Awareness Optimization: Towards Semantic Disambiguation for Direct Preference Optimization
- **分类: cs.CL**

- **简介: 该论文针对直接偏好优化（DPO）中的语义模糊问题，提出模糊感知优化（AAO）方法。通过计算偏好对中内容的语义相似度，自动重加权模糊项以减少歧义，提升模型对齐效果。实验表明，AAO在多个基准上显著优于现有方法，性能提升达8.9至15.0点。**

- **链接: [https://arxiv.org/pdf/2511.23391v1](https://arxiv.org/pdf/2511.23391v1)**

> **作者:** Jian Li; Shenglin Yin; Yujia Zhang; Alan Zhao; Xi Chen; Xiaohui Zhou; Pengfei Xu
>
> **备注:** Accepted at EMNLP 2025 main
>
> **摘要:** Direct Preference Optimization (DPO) is a widely used reinforcement learning from human feedback (RLHF) method across various domains. Recent research has increasingly focused on the role of token importance in improving DPO effectiveness. It is observed that identical or semantically similar content (defined as ambiguous content) frequently appears within the preference pairs. We hypothesize that the presence of ambiguous content during DPO training may introduce ambiguity, thereby limiting further improvements in alignment. Through mathematical analysis and proof-of-concept experiments, we reveal that ambiguous content may potentially introduce ambiguities, thereby degrading performance. To address this issue, we introduce Ambiguity Awareness Optimization (AAO), a simple yet effective approach that automatically re-weights ambiguous content to reduce ambiguities by calculating semantic similarity from preference pairs. Through extensive experiments, we demonstrate that AAO consistently and significantly surpasses state-of-the-art approaches in performance, without markedly increasing response length, across multiple model scales and widely adopted benchmark datasets, including AlpacaEval 2, MT-Bench, and Arena-Hard. Specifically, AAO outperforms DPO by up to 8.9 points on AlpacaEval 2 and achieves an improvement of by up to 15.0 points on Arena-Hard.
>
---
#### [new 045] Improving Score Reliability of Multiple Choice Benchmarks with Consistency Evaluation and Altered Answer Choices
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型在多项选择题基准测试中得分不可靠的问题，提出一致性重平衡准确率（CoRA）指标。通过合成问题与答案选项变换，评估模型响应一致性，引入BMCA和CI两个中间指标，有效降低不一致模型的得分，提升评分可靠性。**

- **链接: [https://arxiv.org/pdf/2511.21860v1](https://arxiv.org/pdf/2511.21860v1)**

> **作者:** Paulo Cavalin; Cassia Sanctos; Marcelo Grave; Claudio Pinhanez; Yago Primerano
>
> **摘要:** In this work we present the Consistency-Rebalanced Accuracy (CoRA) metric, improving the reliability of Large Language Model (LLM) scores computed on multiple choice (MC) benchmarks. Our metric explores the response consistency of the LLMs, taking advantage of synthetically-generated questions with altered answer choices. With two intermediate scores, i.e. Bare-Minimum-Consistency Accuracy (BMCA) and Consistency Index (CI), CoRA is computed by adjusting the multiple-choice question answering (MCQA) scores to better reflect the level of consistency of the LLM. We present evaluations in different benchmarks using diverse LLMs, and not only demonstrate that LLMs can present low response consistency even when they present high MCQA scores, but also that CoRA can successfully scale down the scores of inconsistent models.
>
---
#### [new 046] Early Risk Prediction with Temporally and Contextually Grounded Clinical Language Processing
- **分类: cs.CL**

- **简介: 该论文面向慢性病早期风险预测任务，解决临床文本中时序信息利用难、隐私与资源限制等问题。提出HiTGNN模型捕捉患者轨迹的细粒度时序结构，以及ReVeAL轻量推理框架，提升模型可解释性与敏感性，实现高精度、隐私保护的糖尿病风险预测。**

- **链接: [https://arxiv.org/pdf/2511.22038v1](https://arxiv.org/pdf/2511.22038v1)**

> **作者:** Rochana Chaturvedi; Yue Zhou; Andrew Boyd; Brian T. Layden; Mudassir Rashid; Lu Cheng; Ali Cinar; Barbara Di Eugenio
>
> **摘要:** Clinical notes in Electronic Health Records (EHRs) capture rich temporal information on events, clinician reasoning, and lifestyle factors often missing from structured data. Leveraging them for predictive modeling can be impactful for timely identification of chronic diseases. However, they present core natural language processing (NLP) challenges: long text, irregular event distribution, complex temporal dependencies, privacy constraints, and resource limitations. We present two complementary methods for temporally and contextually grounded risk prediction from longitudinal notes. First, we introduce HiTGNN, a hierarchical temporal graph neural network that integrates intra-note temporal event structures, inter-visit dynamics, and medical knowledge to model patient trajectories with fine-grained temporal granularity. Second, we propose ReVeAL, a lightweight, test-time framework that distills the reasoning of large language models into smaller verifier models. Applied to opportunistic screening for Type 2 Diabetes (T2D) using temporally realistic cohorts curated from private and public hospital corpora, HiTGNN achieves the highest predictive accuracy, especially for near-term risk, while preserving privacy and limiting reliance on large proprietary models. ReVeAL enhances sensitivity to true T2D cases and retains explanatory reasoning. Our ablations confirm the value of temporal structure and knowledge augmentation, and fairness analysis shows HiTGNN performs more equitably across subgroups.
>
---
#### [new 047] Standard Occupation Classifier -- A Natural Language Processing Approach
- **分类: cs.CL; cs.LG; econ.GN**

- **简介: 该论文属于自然语言处理中的分类任务，旨在利用职位广告文本自动识别职业类别。针对英国ONS SOC和美国O*NET SOC体系，研究构建了基于BERT与神经网络的集成模型，结合职位标题、描述和技能信息，显著提升了分类准确率，为实时分析劳动力市场动态提供了有效工具。**

- **链接: [https://arxiv.org/pdf/2511.23057v1](https://arxiv.org/pdf/2511.23057v1)**

> **作者:** Sidharth Rony; Jack Patman
>
> **摘要:** Standard Occupational Classifiers (SOC) are systems used to categorize and classify different types of jobs and occupations based on their similarities in terms of job duties, skills, and qualifications. Integrating these facets with Big Data from job advertisement offers the prospect to investigate labour demand that is specific to various occupations. This project investigates the use of recent developments in natural language processing to construct a classifier capable of assigning an occupation code to a given job advertisement. We develop various classifiers for both UK ONS SOC and US O*NET SOC, using different Language Models. We find that an ensemble model, which combines Google BERT and a Neural Network classifier while considering job title, description, and skills, achieved the highest prediction accuracy. Specifically, the ensemble model exhibited a classification accuracy of up to 61% for the lower (or fourth) tier of SOC, and 72% for the third tier of SOC. This model could provide up to date, accurate information on the evolution of the labour market using job advertisements.
>
---
#### [new 048] Visual Puns from Idioms: An Iterative LLM-T2IM-MLLM Framework
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究成语视觉双关图生成与理解任务，旨在自动创建既符合字面又体现隐喻意义的图像。提出迭代框架，协同LLM、T2IM和MLLM，通过循环优化提示词生成高质量视觉双关图，并构建了1000个图文数据集用于评估。实验表明MLLM性能主导，GPT表现最佳，Claude在提示生成上最优。**

- **链接: [https://arxiv.org/pdf/2511.22943v1](https://arxiv.org/pdf/2511.22943v1)**

> **作者:** Kelaiti Xiao; Liang Yang; Dongyu Zhang; Paerhati Tulajiang; Hongfei Lin
>
> **备注:** Submitted to ICASSP 2026 (under review)
>
> **摘要:** We study idiom-based visual puns--images that align an idiom's literal and figurative meanings--and present an iterative framework that coordinates a large language model (LLM), a text-to-image model (T2IM), and a multimodal LLM (MLLM) for automatic generation and evaluation. Given an idiom, the system iteratively (i) generates detailed visual prompts, (ii) synthesizes an image, (iii) infers the idiom from the image, and (iv) refines the prompt until recognition succeeds or a step limit is reached. Using 1,000 idioms as inputs, we synthesize a corresponding dataset of visual pun images with paired prompts, enabling benchmarking of both generation and understanding. Experiments across 10 LLMs, 10 MLLMs, and one T2IM (Qwen-Image) show that MLLM choice is the primary performance driver: GPT achieves the highest accuracies, Gemini follows, and the best open-source MLLM (Gemma) is competitive with some closed models. On the LLM side, Claude attains the strongest average performance for prompt generation.
>
---
#### [new 049] AfriStereo: A Culturally Grounded Dataset for Evaluating Stereotypical Bias in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出AfriStereo，首个基于非洲本土文化语境的立体化偏见评估数据集与框架，旨在解决现有AI偏见评测基准忽视非洲视角、加剧文化刻板印象的问题。通过跨国社区协作收集并验证超5000对刻板印象-反刻板印象样本，发现多数大模型存在显著偏见，凸显构建更具包容性的NLP技术的必要性。**

- **链接: [https://arxiv.org/pdf/2511.22016v1](https://arxiv.org/pdf/2511.22016v1)**

> **作者:** Yann Le Beux; Oluchi Audu; Oche D. Ankeli; Dhananjay Balakrishnan; Melissah Weya; Marie D. Ralaiarinosy; Ignatius Ezeani
>
> **摘要:** Existing AI bias evaluation benchmarks largely reflect Western perspectives, leaving African contexts underrepresented and enabling harmful stereotypes in applications across various domains. To address this gap, we introduce AfriStereo, the first open-source African stereotype dataset and evaluation framework grounded in local socio-cultural contexts. Through community engaged efforts across Senegal, Kenya, and Nigeria, we collected 1,163 stereotypes spanning gender, ethnicity, religion, age, and profession. Using few-shot prompting with human-in-the-loop validation, we augmented the dataset to over 5,000 stereotype-antistereotype pairs. Entries were validated through semantic clustering and manual annotation by culturally informed reviewers. Preliminary evaluation of language models reveals that nine of eleven models exhibit statistically significant bias, with Bias Preference Ratios (BPR) ranging from 0.63 to 0.78 (p <= 0.05), indicating systematic preferences for stereotypes over antistereotypes, particularly across age, profession, and gender dimensions. Domain-specific models appeared to show weaker bias in our setup, suggesting task-specific training may mitigate some associations. Looking ahead, AfriStereo opens pathways for future research on culturally grounded bias evaluation and mitigation, offering key methodologies for the AI community on building more equitable, context-aware, and globally inclusive NLP technologies.
>
---
#### [new 050] Listwise Preference Optimization with Element-wise Confusions for Aspect Sentiment Quad Prediction
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对Aspect Sentiment Quad Prediction（ASQP）任务，解决传统方法在建模四元组元素间复杂关系及高阶元素预测性能下降的问题。提出基于推理生成的统一框架，结合元素级混淆候选与列表偏好优化，提升结构合理性与解释一致性。**

- **链接: [https://arxiv.org/pdf/2511.23184v1](https://arxiv.org/pdf/2511.23184v1)**

> **作者:** Wenna Lai; Haoran Xie; Guandong Xu; Qing Li; S. Joe Qin
>
> **备注:** 11 pages, 7 figures, and 6 tables
>
> **摘要:** Aspect sentiment quad prediction (ASQP) is inherently challenging to predict a structured quadruple with four core sentiment elements, including aspect term (a), aspect category (c), opinion term (o), and sentiment polarity (s). Prior methods relying on marker-based prediction struggle with modeling the intricate relationships among elements and experience sharp performance declines when predicting higher-order elements (e.g., c and s) under standard supervised fine-tuning. To address these limitations, we employ reasoning-based generation to output both the quadruple and a natural language rationale under element prefixes within a unified template, encouraging explicit relational reasoning and interpretability. To further enhance element-wise alignment, we introduce a listwise preference optimization framework for improving structural validity and relational coherence. Specifically, we generate element-wise confusable candidates via syntactic and semantic proximity, then train the model with listwise objectives to prefer the gold candidates over closely competing alternatives. Extensive experiments on four benchmark datasets demonstrate that our framework effectively improves quadruple prediction accuracy and explanation consistency.
>
---
#### [new 051] Tackling a Challenging Corpus for Early Detection of Gambling Disorder: UNSL at MentalRiskES 2025
- **分类: cs.CL**

- **简介: 该论文参与MentalRiskES 2025挑战赛Task 1，聚焦社交网络上赌博障碍的早期风险检测。针对高/低风险用户分类难题，提出基于CPI+DMC的三种方法，融合SS3、扩展词汇BERT与SBERT模型，结合历史用户分析制定决策策略，有效提升预测精度与决策速度，两项方案位列官方结果前二。**

- **链接: [https://arxiv.org/pdf/2511.23325v1](https://arxiv.org/pdf/2511.23325v1)**

> **作者:** Horacio Thompson; Marcelo Errecalde
>
> **备注:** In Iberian Language Evaluation Forum (IberLEF 2025), Zaragoza, Spain
>
> **摘要:** Gambling disorder is a complex behavioral addiction that is challenging to understand and address, with severe physical, psychological, and social consequences. Early Risk Detection (ERD) on the Web has become a key task in the scientific community for identifying early signs of mental health behaviors based on social media activity. This work presents our participation in the MentalRiskES 2025 challenge, specifically in Task 1, aimed at classifying users at high or low risk of developing a gambling-related disorder. We proposed three methods based on a CPI+DMC approach, addressing predictive effectiveness and decision-making speed as independent objectives. The components were implemented using the SS3, BERT with extended vocabulary, and SBERT models, followed by decision policies based on historical user analysis. Although it was a challenging corpus, two of our proposals achieved the top two positions in the official results, performing notably in decision metrics. Further analysis revealed some difficulty in distinguishing between users at high and low risk, reinforcing the need to explore strategies to improve data interpretation and quality, and to promote more transparent and reliable ERD systems for mental disorders.
>
---
#### [new 052] CrossCheck-Bench: Diagnosing Compositional Failures in Multimodal Conflict Resolution
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对多模态模型在真实场景中识别与解决视觉与文本矛盾的能力不足问题，提出CrossCheck-Bench基准。通过构建15k含合成矛盾的多层级任务，评估模型跨模态推理能力，发现现有模型在复杂推理上表现薄弱，揭示符号推理与视觉处理融合的重要性。**

- **链接: [https://arxiv.org/pdf/2511.21717v1](https://arxiv.org/pdf/2511.21717v1)**

> **作者:** Baoliang Tian; Yuxuan Si; Jilong Wang; Lingyao Li; Zhongyuan Bao; Zineng Zhou; Tao Wang; Sixu Li; Ziyao Xu; Mingze Wang; Zhouzhuo Zhang; Zhihao Wang; Yike Yun; Ke Tian; Ning Yang; Minghui Qiu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Multimodal Large Language Models are primarily trained and evaluated on aligned image-text pairs, which leaves their ability to detect and resolve real-world inconsistencies largely unexplored. In open-domain applications visual and textual cues often conflict, requiring models to perform structured reasoning beyond surface-level alignment. We introduce CrossCheck-Bench, a diagnostic benchmark for evaluating contradiction detection in multimodal inputs. The benchmark adopts a hierarchical task framework covering three levels of reasoning complexity and defines seven atomic capabilities essential for resolving cross-modal inconsistencies. CrossCheck-Bench includes 15k question-answer pairs sourced from real-world artifacts with synthetically injected contradictions. The dataset is constructed through a multi-stage annotation pipeline involving more than 450 expert hours to ensure semantic validity and calibrated difficulty across perception, integration, and reasoning. We evaluate 13 state-of-the-art vision-language models and observe a consistent performance drop as tasks shift from perceptual matching to logical contradiction detection. Most models perform well on isolated entity recognition but fail when multiple clues must be synthesized for conflict reasoning. Capability-level analysis further reveals uneven skill acquisition, especially in tasks requiring multi-step inference or rule-based validation. Additional probing shows that conventional prompting strategies such as Chain-of-Thought and Set-of-Mark yield only marginal gains. By contrast, methods that interleave symbolic reasoning with grounded visual processing achieve more stable improvements. These results highlight a persistent bottleneck in multimodal reasoning and suggest new directions for building models capable of robust cross-modal verification.
>
---
#### [new 053] Affective Multimodal Agents with Proactive Knowledge Grounding for Emotionally Aligned Marketing Dialogue
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对营销对话中对话系统反应式行为导致情感不一致的问题，提出AffectMind多模态情感对话代理。通过主动知识接地、情绪-意图对齐建模与强化话语循环，实现情感协同与高说服力交互，在新构建的数据集上显著提升情感一致性、说服成功率与用户参与度。**

- **链接: [https://arxiv.org/pdf/2511.21728v1](https://arxiv.org/pdf/2511.21728v1)**

> **作者:** Lin Yu; Xiaofei Han; Yifei Kang; Chiung-Yi Tseng; Danyang Zhang; Ziqian Bi; Zhimo Han
>
> **摘要:** Recent advances in large language models (LLMs) have enabled fluent dialogue systems, but most remain reactive and struggle in emotionally rich, goal-oriented settings such as marketing conversations. To address this limitation, we propose AffectMind, a multimodal affective dialogue agent that performs proactive reasoning and dynamic knowledge grounding to sustain emotionally aligned and persuasive interactions. AffectMind combines three components: a Proactive Knowledge Grounding Network (PKGN) that continuously updates factual and affective context from text, vision, and prosody; an Emotion--Intent Alignment Model (EIAM) that jointly models user emotion and purchase intent to adapt persuasion strategies; and a Reinforced Discourse Loop (RDL) that optimizes emotional coherence and engagement via reinforcement signals from user responses. Experiments on two newly curated marketing dialogue datasets, MM-ConvMarket and AffectPromo, show that AffectMind outperforms strong LLM-based baselines in emotional consistency (+26\%), persuasive success rate (+19\%), and long-term user engagement (+23\%), highlighting emotion-grounded proactivity as a key capability for commercial multimodal agents.
>
---
#### [new 054] Every Token Counts: Generalizing 16M Ultra-Long Context in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于大语言模型的超长上下文建模任务，旨在解决记忆效率问题。提出层级稀疏注意力（HSA）机制，实现稀疏性、随机访问灵活性和长度泛化能力。构建8B参数MoE模型HSA-UltraLong，训练数据超8万亿词，在1600万词上下文中检索准确率超90%，显著提升长文本处理能力。**

- **链接: [https://arxiv.org/pdf/2511.23319v1](https://arxiv.org/pdf/2511.23319v1)**

> **作者:** Xiang Hu; Zhanchao Zhou; Ruiqi Liang; Zehuan Li; Wei Wu; Jianguo Li
>
> **摘要:** This work explores the challenge of building ``Machines that Can Remember'', framing long-term memory as the problem of efficient ultra-long context modeling. We argue that this requires three key properties: \textbf{sparsity}, \textbf{random-access flexibility}, and \textbf{length generalization}. To address ultra-long-context modeling, we leverage Hierarchical Sparse Attention (HSA), a novel attention mechanism that satisfies all three properties. We integrate HSA into Transformers to build HSA-UltraLong, which is an 8B-parameter MoE model trained on over 8 trillion tokens and is rigorously evaluated on different tasks with in-domain and out-of-domain context lengths to demonstrate its capability in handling ultra-long contexts. Results show that our model performs comparably to full-attention baselines on in-domain lengths while achieving over 90\% accuracy on most in-context retrieval tasks with contexts up to 16M. This report outlines our experimental insights and open problems, contributing a foundation for future research in ultra-long context modeling.
>
---
#### [new 055] German General Personas: A Survey-Derived Persona Prompt Collection for Population-Aligned LLM Studies
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出德国通用人物画像（GGP）数据集，基于真实社会调查构建，用于提升大语言模型在社会模拟中的代表性。针对现有人物画像缺乏实证支持的问题，研究通过构建可适配各类任务的提示模板，验证其在数据稀缺下优于主流分类器，推动更精准的人群对齐建模。**

- **链接: [https://arxiv.org/pdf/2511.21722v1](https://arxiv.org/pdf/2511.21722v1)**

> **作者:** Jens Rupprecht; Leon Fröhling; Claudia Wagner; Markus Strohmaier
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** The use of Large Language Models (LLMs) for simulating human perspectives via persona prompting is gaining traction in computational social science. However, well-curated, empirically grounded persona collections remain scarce, limiting the accuracy and representativeness of such simulations. Here we introduce the German General Personas (GGP) collection, a comprehensive and representative persona prompt collection built from the German General Social Survey (ALLBUS). The GGP and its persona prompts are designed to be easily plugged into prompts for all types of LLMs and tasks, steering models to generate responses aligned with the underlying German population. We evaluate GGP by prompting various LLMs to simulate survey response distributions across diverse topics, demonstrating that GGP-guided LLMs outperform state-of-the-art classifiers, particularly under data scarcity. Furthermore, we analyze how the representativity and attribute selection within persona prompts affect alignment with population responses. Our findings suggest that GGP provides a potentially valuable resource for research on LLM-based social simulations that enables more systematic explorations of population-aligned persona prompting in NLP and social science research.
>
---
#### [new 056] On the Cross-lingual Transferability of Pre-trained wav2vec2-based Models
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究wav2vec2模型在跨语言场景下的知识迁移能力。针对预训练模型在不同语言上表现差异的问题，通过18种语言的语音识别任务实验，发现数据多样性比数量更重要，且相似语言间迁移效果更优。研究为模型选择与预训练提供指导。**

- **链接: [https://arxiv.org/pdf/2511.21704v1](https://arxiv.org/pdf/2511.21704v1)**

> **作者:** Jonatas Grosman; Cassio Almeida; Guilherme Schardong; Hélio Lopes
>
> **摘要:** Using representations provided by a large pre-trained model has become the primary strategy for achieving state-of-the-art results in a wide range of tasks. A recently proposed large pre-trained model, wav2vec 2.0, was seminal for several other works on pre-training large models on speech data. Many models are being pre-trained using the same architecture as wav2vec 2.0 and are getting state-of-the-art in various speech-related tasks. Previous work has demonstrated that the data used during the pre-training of these wav2vec2-based models can impact the model's performance in downstream tasks, and this should be taken into consideration before utilizing these models. However, few works have proposed investigating further how the transfer knowledge of these pre-trained models behaves in different languages, even when the target language differs from the one used during the model's pre-training. Our work aims to investigate the cross-lingual transferability of these wav2vec2-based models. We performed several fine-tuning experiments on the speech recognition task in 18 languages using 15 large pre-trained models. The results of our experiments showed us that the size of data used during the pre-training of these models is not as important to the final performance as the diversity. We noticed that the performance of Indo-European languages is superior to non-Indo-European languages in the evaluated models. We have observed a positive cross-lingual transfer of knowledge using monolingual models, which was evident in all the languages we used, but more pronounced when the language used during pre-training was more similar to the downstream task language. With these findings, we aim to assist the scientific community in utilizing existing wav2vec2-based pre-trained models, as well as facilitate the pre-training of new ones.
>
---
#### [new 057] Identifying Quantum Structure in AI Language: Evidence for Evolutionary Convergence of Human and Artificial Cognition
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究人工智能语言中的量子结构，通过认知测试发现LLMs在概念组合与词汇分布中呈现量子特征（如贝尔不等式违背、玻色-爱因斯坦统计），类比人类认知，提出人与AI认知存在演化收敛。工作包括实验验证、跨主体对比与统一框架构建，旨在揭示语言意义的量子组织本质。**

- **链接: [https://arxiv.org/pdf/2511.21731v1](https://arxiv.org/pdf/2511.21731v1)**

> **作者:** Diederik Aerts; Jonito Aerts Arguëlles; Lester Beltran; Suzette Geriente; Roberto Leporini; Massimiliano Sassoli de Bianchi; Sandro Sozzo
>
> **摘要:** We present the results of cognitive tests on conceptual combinations, performed using specific Large Language Models (LLMs) as test subjects. In the first test, performed with ChatGPT and Gemini, we show that Bell's inequalities are significantly violated, which indicates the presence of 'quantum entanglement' in the tested concepts. In the second test, also performed using ChatGPT and Gemini, we instead identify the presence of 'Bose-Einstein statistics', rather than the intuitively expected 'Maxwell-Boltzmann statistics', in the distribution of the words contained in large-size texts. Interestingly, these findings mirror the results previously obtained in both cognitive tests with human participants and information retrieval tests on large corpora. Taken together, they point to the 'systematic emergence of quantum structures in conceptual-linguistic domains', regardless of whether the cognitive agent is human or artificial. Although LLMs are classified as neural networks for historical reasons, we believe that a more essential form of knowledge organization takes place in the distributive semantic structure of vector spaces built on top of the neural network. It is this meaning-bearing structure that lends itself to a phenomenon of evolutionary convergence between human cognition and language, slowly established through biological evolution, and LLM cognition and language, emerging much more rapidly as a result of self-learning and training. We analyze various aspects and examples that contain evidence supporting the above hypothesis. We also advance a unifying framework that explains the pervasive quantum organization of meaning that we identify.
>
---
#### [new 058] FEANEL: A Benchmark for Fine-Grained Error Analysis in K-12 English Writing
- **分类: cs.CL**

- **简介: 该论文针对K-12英语写作中细粒度错误分析任务，提出FEANEL基准。通过构建包含1000篇学生作文和专家标注的错误分类体系，评估大语言模型在错误类型、严重程度及教学反馈上的表现，揭示其在教育应用中的不足，推动精准教学反馈技术发展。**

- **链接: [https://arxiv.org/pdf/2511.22883v1](https://arxiv.org/pdf/2511.22883v1)**

> **作者:** Jingheng Ye; Shen Wang; Jiaqi Chen; Hebin Wang; Deqing Zou; Yanyu Zhu; Jiwei Tang; Hai-Tao Zheng; Ruitong Liu; Haoyang Li; Yanfeng Wang; Qingsong Wen
>
> **备注:** 19 pages, 7 figures, and 4 tables. The dataset is available at https://huggingface.co/datasets/Feanel/FEANEL
>
> **摘要:** Large Language Models (LLMs) have transformed artificial intelligence, offering profound opportunities for educational applications. However, their ability to provide fine-grained educational feedback for K-12 English writing remains underexplored. In this paper, we challenge the error analysis and pedagogical skills of LLMs by introducing the problem of Fine-grained Error Analysis for English Learners and present the Fine-grained Error ANalysis for English Learners (FEANEL) Benchmark. The benchmark comprises 1,000 essays written by elementary and secondary school students, and a well-developed English writing error taxonomy. Each error is annotated by language education experts and categorized by type, severity, and explanatory feedback, using a part-of-speech-based taxonomy they co-developed. We evaluate state-of-the-art LLMs on the FEANEL Benchmark to explore their error analysis and pedagogical abilities. Experimental results reveal significant gaps in current LLMs' ability to perform fine-grained error analysis, highlighting the need for advancements in particular methods for educational applications.
>
---
#### [new 059] TWEO: Transformers Without Extreme Outliers Enables FP8 Training And Quantization For Dummies
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文针对大模型FP8训练中极端激活值溢出问题，提出非侵入式损失函数TWEO。通过揭示异常值源于权重矩阵共线性而非数据特性，以简单损失项有效抑制异常值，实现无需复杂工程的全模型FP8训练与量化，显著提升训练效率并达成SOTA量化性能。**

- **链接: [https://arxiv.org/pdf/2511.23225v1](https://arxiv.org/pdf/2511.23225v1)**

> **作者:** Guang Liang; Jie Shao; Ningyuan Tang; Xinyao Liu; Jianxin Wu
>
> **摘要:** Native FP8 support in modern hardware is essential for training large Transformers, but is severely hindered by extreme activation outliers. Existing solutions either rely on complex mixed-precision engineering or invasive architectural modifications. This paper fundamentally challenges the conventional wisdom that outliers are data-driven. We demonstrate that extreme outliers are a data-independent, mechanically-produced artifact of training, originating from specific structural properties of the weight matrices (i.e., colinearity). Based on this insight, we propose TWEO (Transformers Without Extreme Outliers), a novel, non-invasive loss function. TWEO effectively prevents extreme outliers via a very simple loss term, which reduces outliers from 10000+ to less than 20. TWEO then enables full-model FP8 pre-training with neither engineering tricks nor architectural changes for both LLM and ViT. When standard FP8 training catastrophically collapses, TWEO achieves performance comparable to the BF16 baseline while delivering a 36% increase in training throughput. Also, TWEO enables a new quantization paradigm. Hardware-friendly W8A8 per-tensor static quantization of LLMs, previously considered completely unusable due to outliers, achieves SOTA performance for the first time on TWEO-trained models.
>
---
#### [new 060] fMRI-LM: Towards a Universal Foundation Model for Language-Aligned fMRI Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出fMRI-LM，旨在构建语言对齐的通用脑成像理解模型。针对脑影像与语言模态难以统一建模的问题，通过三阶段框架：神经令牌化、跨模态预训练、多任务指令微调，实现fMRI与语言的联合理解，支持零样本和少样本应用。**

- **链接: [https://arxiv.org/pdf/2511.21760v1](https://arxiv.org/pdf/2511.21760v1)**

> **作者:** Yuxiang Wei; Yanteng Zhang; Xi Xiao; Chengxuan Qian; Tianyang Wang; Vince D. Calhoun
>
> **摘要:** Recent advances in multimodal large language models (LLMs) have enabled unified reasoning across images, audio, and video, but extending such capability to brain imaging remains largely unexplored. Bridging this gap is essential to link neural activity with semantic cognition and to develop cross-modal brain representations. To this end, we present fMRI-LM, a foundational model that bridges functional MRI (fMRI) and language through a three-stage framework. In Stage 1, we learn a neural tokenizer that maps fMRI into discrete tokens embedded in a language-consistent space. In Stage 2, a pretrained LLM is adapted to jointly model fMRI tokens and text, treating brain activity as a sequence that can be temporally predicted and linguistically described. To overcome the lack of natural fMRI-text pairs, we construct a large descriptive corpus that translates diverse imaging-based features into structured textual descriptors, capturing the low-level organization of fMRI signals. In Stage 3, we perform multi-task, multi-paradigm instruction tuning to endow fMRI-LM with high-level semantic understanding, supporting diverse downstream applications. Across various benchmarks, fMRI-LM achieves strong zero-shot and few-shot performance, and adapts efficiently with parameter-efficient tuning (LoRA), establishing a scalable pathway toward a language-aligned, universal model for structural and semantic understanding of fMRI.
>
---
#### [new 061] Dissecting the Ledger: Locating and Suppressing "Liar Circuits" in Financial Large Language Models
- **分类: cs.CL; cs.CE**

- **简介: 该论文研究金融领域大模型的算术幻觉问题，提出基于因果追踪的机制分析方法。发现模型在中层（L12-L30）形成分布式计算垫，在深层（L46）存在决定性聚合电路。通过抑制L46层，幻觉置信度降低81.8%，并验证了该层可泛化识别算术错误，为可解释性与安全控制提供新路径。**

- **链接: [https://arxiv.org/pdf/2511.21756v1](https://arxiv.org/pdf/2511.21756v1)**

> **作者:** Soham Mirajkar
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in high-stakes financial domains, yet they suffer from specific, reproducible hallucinations when performing arithmetic operations. Current mitigation strategies often treat the model as a black box. In this work, we propose a mechanistic approach to intrinsic hallucination detection. By applying Causal Tracing to the GPT-2 XL architecture on the ConvFinQA benchmark, we identify a dual-stage mechanism for arithmetic reasoning: a distributed computational scratchpad in middle layers (L12-L30) and a decisive aggregation circuit in late layers (specifically Layer 46). We verify this mechanism via an ablation study, demonstrating that suppressing Layer 46 reduces the model's confidence in hallucinatory outputs by 81.8%. Furthermore, we demonstrate that a linear probe trained on this layer generalizes to unseen financial topics with 98% accuracy, suggesting a universal geometry of arithmetic deception.
>
---
#### [new 062] 47B Mixture-of-Experts Beats 671B Dense Models on Chinese Medical Examinations
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对中文医学考试评估任务，对比27个大模型在七大学科上的表现。通过构建2800题基准测试集，发现47B MoE模型优于671B密集模型，揭示模型规模与性能无直接关联，且不同学科表现差异显著，为医疗AI应用提供关键参考。**

- **链接: [https://arxiv.org/pdf/2511.21701v1](https://arxiv.org/pdf/2511.21701v1)**

> **作者:** Chiung-Yi Tseng; Danyang Zhang; Tianyang Wang; Hongying Luo; Lu Chen; Junming Huang; Jibin Guan; Junfeng Hao; Junhao Song; Ziqian Bi
>
> **摘要:** The rapid advancement of large language models(LLMs) has prompted significant interest in their potential applications in medical domains. This paper presents a comprehensive benchmark evaluation of 27 state-of-the-art LLMs on Chinese medical examination questions, encompassing seven medical specialties across two professional levels. We introduce a robust evaluation framework that assesses model performance on 2,800 carefully curated questions from cardiovascular, gastroenterology, hematology, infectious diseases, nephrology, neurology, and respiratory medicine domains. Our dataset distinguishes between attending physician and senior physician difficulty levels, providing nuanced insights into model capabilities across varying complexity. Our empirical analysis reveals substantial performance variations among models, with Mixtral-8x7B achieving the highest overall accuracy of 74.25%, followed by DeepSeek-R1-671B at 64.07%. Notably, we observe no consistent correlation between model size and performance, as evidenced by the strong performance of smaller mixture-of-experts architectures. The evaluation demonstrates significant performance gaps between medical specialties, with models generally performing better on cardiovascular and neurology questions compared to gastroenterology and nephrology domains. Furthermore, our analysis indicates minimal performance degradation between attending and senior physician levels for top-performing models, suggesting robust generalization capabilities. This benchmark provides critical insights for the deployment of LLMs in medical education and clinical decision support systems, highlighting both the promise and current limitations of these technologies in specialized medical contexts.
>
---
#### [new 063] Decoding the Past: Explainable Machine Learning Models for Dating Historical Texts
- **分类: cs.CL**

- **简介: 该论文针对历史文本年代判定任务，提出基于可解释树模型的多特征融合方法，整合五类特征提升时序分类精度。实验显示模型在世纪与十年尺度上显著优于随机基准，具备强排序能力与可解释性，适用于文化遗产管理与文本认证。**

- **链接: [https://arxiv.org/pdf/2511.23056v1](https://arxiv.org/pdf/2511.23056v1)**

> **作者:** Paulo J. N. Pinto; Armando J. Pinho; Diogo Pratas
>
> **摘要:** Accurately dating historical texts is essential for organizing and interpreting cultural heritage collections. This article addresses temporal text classification using interpretable, feature-engineered tree-based machine learning models. We integrate five feature categories - compression-based, lexical structure, readability, neologism detection, and distance features - to predict the temporal origin of English texts spanning five centuries. Comparative analysis shows that these feature domains provide complementary temporal signals, with combined models outperforming any individual feature set. On a large-scale corpus, we achieve 76.7% accuracy for century-scale prediction and 26.1% for decade-scale classification, substantially above random baselines (20% and 2.3%). Under relaxed temporal precision, performance increases to 96.0% top-2 accuracy for centuries and 85.8% top-10 accuracy for decades. The final model exhibits strong ranking capabilities with AUCROC up to 94.8% and AUPRC up to 83.3%, and maintains controlled errors with mean absolute deviations of 27 years and 30 years, respectively. For authentication-style tasks, binary models around key thresholds (e.g., 1850-1900) reach 85-98% accuracy. Feature importance analysis identifies distance features and lexical structure as most informative, with compression-based features providing complementary signals. SHAP explainability reveals systematic linguistic evolution patterns, with the 19th century emerging as a pivot point across feature domains. Cross-dataset evaluation on Project Gutenberg highlights domain adaptation challenges, with accuracy dropping by 26.4 percentage points, yet the computational efficiency and interpretability of tree-based models still offer a scalable, explainable alternative to neural architectures.
>
---
#### [new 064] Scaling HuBERT for African Languages: From Base to Large and XL
- **分类: cs.CL**

- **简介: 该论文针对非洲语言在语音识别中数据与模型稀缺的问题，提出首个专为非洲语音训练的大规模自监督模型SSA-HuBERT-Large和XL。通过在撒哈拉以南非洲语言上进行训练，验证了更大模型容量对自动语音识别与语言识别任务的显著提升，推动了非洲语言语音技术的发展。**

- **链接: [https://arxiv.org/pdf/2511.23370v1](https://arxiv.org/pdf/2511.23370v1)**

> **作者:** Antoine Caubrière; Elodie Gauthier
>
> **备注:** Journée d'études AFIA-ATALA 2025 : Technologies linguistiques pour les langues peu dotées
>
> **摘要:** Despite recent progress in multilingual speech processing, African languages remain under-represented in both research and deployed systems, particularly when it comes to strong, open-weight encoders that transfer well under low-resource supervision. Self-supervised learning has proven especially promising in such settings, yet most publicly released models targeting African speech remain at BASE scale, leaving unanswered whether larger encoders, trained exclusively on Africa-centric audio, offer tangible benefits and how model capacity interacts with data composition. This work addresses that gap by introducing SSA-HuBERT-Large (317M parameters) and SSA-HuBERT-XL (964M parameters), the first large models trained solely on African speech, alongside a BASE size counterpart. We release these models as open weights: see https://huggingface.co/collections/Orange/african-speech-foundation-models. By conducting a carefully controlled experimental study focused exclusively on Sub-Saharan languages, covering automatic speech recognition (ASR) and language identification (LID) tasks, we demonstrate that larger architectures significantly improve performance by effectively leveraging large audio datasets.
>
---
#### [new 065] Improving LLM-based Ontology Matching with fine-tuning on synthetic data
- **分类: cs.CL**

- **简介: 该论文研究如何利用微调增强LLM在本体匹配任务中的表现。针对标注数据稀缺问题，提出基于LLM生成合成数据集的方法，并结合搜索空间缩减技术构建提示。实验表明，微调后的模型在多个数据集上优于基线模型，有效提升了零样本场景下的匹配性能。**

- **链接: [https://arxiv.org/pdf/2511.22612v1](https://arxiv.org/pdf/2511.22612v1)**

> **作者:** Guilherme Sousa; Rinaldo Lima; Cassia Trojahn
>
> **摘要:** Large Language Models (LLMs) are increasingly being integrated into various components of Ontology Matching pipelines. This paper investigates the capability of LLMs to perform ontology matching directly on ontology modules and generate the corresponding alignments. Furthermore, it is explored how a dedicated fine-tuning strategy can enhance the model's matching performance in a zero-shot setting. The proposed method incorporates a search space reduction technique to select relevant subsets from both source and target ontologies, which are then used to automatically construct prompts. Recognizing the scarcity of reference alignments for training, a novel LLM-based approach is introduced for generating a synthetic dataset. This process creates a corpus of ontology submodule pairs and their corresponding reference alignments, specifically designed to fine-tune an LLM for the ontology matching task. The proposed approach was evaluated on the Conference, Geolink, Enslaved, Taxon, and Hydrography datasets from the OAEI complex track. The results demonstrate that the LLM fine-tuned on the synthetically generated data exhibits superior performance compared to the non-fine-tuned base model. The key contribution is a strategy that combines automatic dataset generation with fine-tuning to effectively adapt LLMs for ontology matching tasks.
>
---
#### [new 066] AD-CDO: A Lightweight Ontology for Representing Eligibility Criteria in Alzheimer's Disease Clinical Trials
- **分类: cs.CL**

- **简介: 该论文提出轻量级本体AD-CDO，用于标准化阿尔茨海默病临床试验的纳入标准。通过提取1500+试验高频概念，整合标准术语库，采用自然断点法优化，实现63%以上覆盖率，支持试验模拟与文本标准化，解决临床数据异构性问题，助力精准研究与数据整合。**

- **链接: [https://arxiv.org/pdf/2511.21724v1](https://arxiv.org/pdf/2511.21724v1)**

> **作者:** Zenan Sun; Rashmie Abeysinghe; Xiaojin Li; Xinyue Hu; Licong Cui; Guo-Qiang Zhang; Jiang Bian; Cui Tao
>
> **摘要:** Objective This study introduces the Alzheimer's Disease Common Data Element Ontology for Clinical Trials (AD-CDO), a lightweight, semantically enriched ontology designed to represent and standardize key eligibility criteria concepts in Alzheimer's disease (AD) clinical trials. Materials and Methods We extracted high-frequency concepts from more than 1,500 AD clinical trials on ClinicalTrials.gov and organized them into seven semantic categories: Disease, Medication, Diagnostic Test, Procedure, Social Determinants of Health, Rating Criteria, and Fertility. Each concept was annotated with standard biomedical vocabularies, including the UMLS, OMOP Standardized Vocabularies, DrugBank, NDC, and NLM VSAC value sets. To balance coverage and manageability, we applied the Jenks Natural Breaks method to identify an optimal set of representative concepts. Results The optimized AD-CDO achieved over 63% coverage of extracted trial concepts while maintaining interpretability and compactness. The ontology effectively captured the most frequent and clinically meaningful entities used in AD eligibility criteria. We demonstrated AD-CDO's practical utility through two use cases: (a) an ontology-driven trial simulation system for formal modeling and virtual execution of clinical trials, and (b) an entity normalization task mapping raw clinical text to ontology-aligned terms, enabling consistency and integration with EHR data. Discussion AD-CDO bridges the gap between broad biomedical ontologies and task-specific trial modeling needs. It supports multiple downstream applications, including phenotyping algorithm development, cohort identification, and structured data integration. Conclusion By harmonizing essential eligibility entities and aligning them with standardized vocabularies, AD-CDO provides a versatile foundation for ontology-driven AD clinical trial research.
>
---
#### [new 067] DELTA: Language Diffusion-based EEG-to-Text Architecture
- **分类: cs.CL**

- **简介: 该论文针对脑电（EEG）转文本任务，解决高维噪声、个体差异及自回归解码误差累积问题。提出DELTA架构，结合RVQ EEG分词器与掩码语言扩散模型（LLaDA），通过离散化降噪与非序列去噪重建文本。在ZuCo数据集上显著提升语义对齐，实现高效小样本文本生成。**

- **链接: [https://arxiv.org/pdf/2511.21746v1](https://arxiv.org/pdf/2511.21746v1)**

> **作者:** Mingyu Jeon; Hyobin Kim
>
> **摘要:** Electroencephalogram (EEG)-to-text remains challenging due to high-dimensional noise, subject variability, and error accumulation in autoregressive decoding. We introduce DELTA, which pairs a Residual Vector Quantization (RVQ) EEG tokenizer with a masked language diffusion model (LLaDA). RVQ discretizes continuous EEG into multi-layer tokens to reduce noise and individual differences, while LLaDA reconstructs sentences via non-sequential denoising. On ZuCo, DELTA improves semantic alignment by up to 5.37 points over autoregressive baselines, achieving BLEU-1 21.9 and ROUGE-1 F 17.2 under word-level conditions. These results enable reliable text generation from small EEG-text datasets and point toward scalable multimodal EEG-language models.
>
---
#### [new 068] PromptTailor: Multi-turn Intent-Aligned Prompt Synthesis for Lightweight LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对轻量级大模型在开放生成任务中因提示质量差导致响应不佳的问题，提出PromptTailor系统。通过量化Llama3-8B并结合轻量LoRA适配器，基于多轮对话数据学习意图对齐的提示生成策略，实现高效、精准的提示优化，显著提升生成质量与用户意图一致性，适用于边缘设备部署。**

- **链接: [https://arxiv.org/pdf/2511.21725v1](https://arxiv.org/pdf/2511.21725v1)**

> **作者:** Yizhou Xu; Janet Davis
>
> **备注:** EMNLP 2025 Workshop PALS. Additional note: There is a citation error on Evoke. The paper we are referring to is "Evoking critical thinking abilities in LLMs via reviewer-author prompt editing."
>
> **摘要:** Lightweight language models remain attractive for on-device and privacy-sensitive applications, but their responses are highly sensitive to prompt quality. For open-ended generation, non-expert users often lack the knowledge or time to consistently craft high-quality prompts, leading them to rely on prompt optimization tools. However, a key challenge is ensuring the optimized prompts genuinely align with users' original intents and preferences. We introduce PromptTailor, a system for controllable prompt generation for open-ended text that improves model output quality by intent-aligned prompt synthesis. PromptTailor expands minimal user instructions into rich, domain-aware prompts while preserving the user's stated preferences. The system is a quantized Llama3-8B model fine-tuned with a lightweight LoRA adapter on 12,300 prompt-refinement dialogues spanning 41 everyday domains, distilled from three stronger LLMs. The adapter attaches to any Llama3-8B base, enabling edge deployment. In human and LLM-judge evaluations across multiple target models and optimization baselines, PromptTailor yields higher preference rates than chain-of-thought prompting and matches or surpasses state-of-the-art prompt optimization methods while requiring fewer model calls (e.g., 3 vs. 9). These results show that a compact student, guided by powerful teachers, can learn effective prompt-generation strategies that enhance response quality while maintaining alignment with user intent.
>
---
#### [new 069] EduMod-LLM: A Modular Approach for Designing Flexible and Transparent Educational Assistants
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对教育领域LLM问答系统透明性与可解释性不足的问题，提出EduMod-LLM模块化框架。通过分离函数调用、检索与生成组件，实现对各环节的细粒度评估，揭示系统失效模式，提升教育QA系统的可解释性与教学适配性。**

- **链接: [https://arxiv.org/pdf/2511.21742v1](https://arxiv.org/pdf/2511.21742v1)**

> **作者:** Meenakshi Mittal; Rishi Khare; Mihran Miroyan; Chancharik Mitra; Narges Norouzi
>
> **备注:** Proceedings of the AAAI Conference on Artificial Intelligence
>
> **摘要:** With the growing use of Large Language Model (LLM)-based Question-Answering (QA) systems in education, it is critical to evaluate their performance across individual pipeline components. In this work, we introduce {\model}, a modular function-calling LLM pipeline, and present a comprehensive evaluation along three key axes: function calling strategies, retrieval methods, and generative language models. Our framework enables fine-grained analysis by isolating and assessing each component. We benchmark function-calling performance across LLMs, compare our novel structure-aware retrieval method to vector-based and LLM-scoring baselines, and evaluate various LLMs for response synthesis. This modular approach reveals specific failure modes and performance patterns, supporting the development of interpretable and effective educational QA systems. Our findings demonstrate the value of modular function calling in improving system transparency and pedagogical alignment. Website and Supplementary Material: https://chancharikmitra.github.io/EduMod-LLM-website/
>
---
#### [new 070] Optimizing Multimodal Language Models through Attention-based Interpretability
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对多模态语言模型（MLM）难以解释的问题，提出基于注意力的可解释性方法，通过分析注意力分数识别关键图像对象相关注意力头，并用于参数高效微调（PEFT）。研究聚焦图像描述任务，构建新数据集，验证高影响头微调能以极小参数量显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.23375v1](https://arxiv.org/pdf/2511.23375v1)**

> **作者:** Alexander Sergeev; Evgeny Kotelnikov
>
> **备注:** Accepted for ICAI-2025 conference
>
> **摘要:** Modern large language models become multimodal, analyzing various data formats like text and images. While fine-tuning is effective for adapting these multimodal language models (MLMs) to downstream tasks, full fine-tuning is computationally expensive. Parameter-Efficient Fine-Tuning (PEFT) methods address this by training only a small portion of model weights. However, MLMs are difficult to interpret, making it challenging to identify which components are most effective for training to balance efficiency and performance. We propose an attention-based interpretability method for MLMs by analyzing attention scores relative to image tokens. The core idea is to identify attention heads that focus on image key objects. We utilize this information to select optimal model components for PEFT in multimodal models. Our contributions include a method for identifying attention heads associated with image key objects, its application to PEFT for image captioning, and the creation of a new dataset containing images, key object masks, and their textual descriptions. We conducted experiments on MLMs with 2-3 billion parameters to validate the method's effectiveness. By calculating Head Impact (HI) scores we quantify an attention head's focus on key objects, indicating its significance in image understanding. Our fine-tuning experiments demonstrate that adapting layers with the highest HI scores leads to the most significant shifts in metrics compared to pre-trained, randomly selected, or lowest-HI-score layers. This indicates that fine-tuning a small percentage (around 0.01%) of parameters in these crucial layers can substantially influence image understanding capabilities.
>
---
#### [new 071] When Harmless Words Harm: A New Threat to LLM Safety via Conceptual Triggers
- **分类: cs.CL**

- **简介: 该论文针对大语言模型（LLM）安全中的隐性威胁，提出一种新型无模型依赖的越狱方法MICM。其通过概念触发词操控模型隐含价值观，诱导有害输出而不触发安全过滤。研究揭示了当前对齐机制在抽象价值层面的脆弱性，为提升LLM安全性提供了新视角。**

- **链接: [https://arxiv.org/pdf/2511.21718v1](https://arxiv.org/pdf/2511.21718v1)**

> **作者:** Zhaoxin Zhang; Borui Chen; Yiming Hu; Youyang Qu; Tianqing Zhu; Longxiang Gao
>
> **摘要:** Recent research on large language model (LLM) jailbreaks has primarily focused on techniques that bypass safety mechanisms to elicit overtly harmful outputs. However, such efforts often overlook attacks that exploit the model's capacity for abstract generalization, creating a critical blind spot in current alignment strategies. This gap enables adversaries to induce objectionable content by subtly manipulating the implicit social values embedded in model outputs. In this paper, we introduce MICM, a novel, model-agnostic jailbreak method that targets the aggregate value structure reflected in LLM responses. Drawing on conceptual morphology theory, MICM encodes specific configurations of nuanced concepts into a fixed prompt template through a predefined set of phrases. These phrases act as conceptual triggers, steering model outputs toward a specific value stance without triggering conventional safety filters. We evaluate MICM across five advanced LLMs, including GPT-4o, Deepseek-R1, and Qwen3-8B. Experimental results show that MICM consistently outperforms state-of-the-art jailbreak techniques, achieving high success rates with minimal rejection. Our findings reveal a critical vulnerability in commercial LLMs: their safety mechanisms remain susceptible to covert manipulation of underlying value alignment.
>
---
#### [new 072] MegaChat: A Synthetic Persian Q&A Dataset for High-Quality Sales Chatbot Evaluation
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文针对伊朗中小企业在Telegram上销售缺乏高质量对话数据的问题，提出MegaChat——首个全合成的波斯语问答数据集。通过多智能体架构自动生成符合角色设定的对话，解决低资源语言下标注成本高的难题。实验表明，其智能体系统在多个维度优于传统RAG模型，为低成本构建高效销售聊天机器人提供可行方案。**

- **链接: [https://arxiv.org/pdf/2511.23397v1](https://arxiv.org/pdf/2511.23397v1)**

> **作者:** Mahdi Rahmani; AmirHossein Saffari; Reyhane Rahmani
>
> **备注:** 6 pages, 11 figures, 2 tables
>
> **摘要:** Small and medium-sized enterprises (SMEs) in Iran increasingly leverage Telegram for sales, where real-time engagement is essential for conversion. However, developing AI-driven chatbots for this purpose requires large, high-quality question-and-answer (Q&A) datasets, which are typically expensive and resource-intensive to produce, especially for low-resource languages like Persian. In this paper, we introduce MegaChat, the first fully synthetic Persian Q&A dataset designed to evaluate intelligent sales chatbots in Telegram-based e-commerce. We propose a novel, automated multi-agent architecture that generates persona-aware Q&A pairs by collecting data from active Telegram shopping channels. The system employs specialized agents for question generation, validation, and refinement, ensuring the production of realistic and diverse conversational data. To evaluate answer generation, we compare three classic retrieval-augmented generation (RAG) models with our advanced agentic system, which features multi-query retrieval, reranking, and persona-aligned response synthesis. Using GPT-5.1 for evaluation across six quality dimensions, our results show that the agentic architecture outperformed traditional RAG models in 4 out of 5 diverse channels, demonstrating its ability to generate scalable, high-quality datasets without relying on expensive human annotation or complex fine-tuning. MegaChat provides SMEs with an efficient, cost-effective solution for building intelligent customer engagement systems in specialized commercial domains, enabling advancements in multilingual conversational AI for low-resource languages. Download: https://github.com/MegaChat-Tech/MegaChat-DataSet
>
---
#### [new 073] Extracting Disaster Impacts and Impact Related Locations in Social Media Posts Using Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于灾害影响信息提取任务，旨在从社交媒体中精准识别灾害影响及受影响地点。针对非影响地点干扰的问题，研究通过微调大语言模型，区分影响与非影响地点，提升提取准确率，显著优于基线模型，为应急响应提供高效、可扩展的地理时空信息支持。**

- **链接: [https://arxiv.org/pdf/2511.21753v1](https://arxiv.org/pdf/2511.21753v1)**

> **作者:** Sameeah Noreen Hameed; Surangika Ranathunga; Raj Prasanna; Kristin Stock; Christopher B. Jones
>
> **摘要:** Large-scale disasters can often result in catastrophic consequences on people and infrastructure. Situation awareness about such disaster impacts generated by authoritative data from in-situ sensors, remote sensing imagery, and/or geographic data is often limited due to atmospheric opacity, satellite revisits, and time limitations. This often results in geo-temporal information gaps. In contrast, impact-related social media posts can act as "geo-sensors" during a disaster, where people describe specific impacts and locations. However, not all locations mentioned in disaster-related social media posts relate to an impact. Only the impacted locations are critical for directing resources effectively. e.g., "The death toll from a fire which ripped through the Greek coastal town of #Mati stood at 80, with dozens of people unaccounted for as forensic experts tried to identify victims who were burned alive #Greecefires #AthensFires #Athens #Greece." contains impacted location "Mati" and non-impacted locations "Greece" and "Athens". This research uses Large Language Models (LLMs) to identify all locations, impacts and impacted locations mentioned in disaster-related social media posts. In the process, LLMs are fine-tuned to identify only impacts and impacted locations (as distinct from other, non-impacted locations), including locations mentioned in informal expressions, abbreviations, and short forms. Our fine-tuned model demonstrates efficacy, achieving an F1-score of 0.69 for impact and 0.74 for impacted location extraction, substantially outperforming the pre-trained baseline. These robust results confirm the potential of fine-tuned language models to offer a scalable solution for timely decision-making in resource allocation, situational awareness, and post-disaster recovery planning for responders.
>
---
#### [new 074] JBE-QA: Japanese Bar Exam QA Dataset for Assessing Legal Domain Knowledge
- **分类: cs.CL**

- **简介: 该论文提出JBE-QA，一个用于评估大语言模型法律知识的日本司法考试问答数据集。针对现有资源多聚焦民法、缺乏全面性的问题，构建覆盖民法、刑法和宪法的3,464个结构化题目，支持对26种模型的评测，发现具备推理能力的专有模型表现最佳，宪法题较易。**

- **链接: [https://arxiv.org/pdf/2511.22869v1](https://arxiv.org/pdf/2511.22869v1)**

> **作者:** Zhihan Cao; Fumihito Nishino; Hiroaki Yamada; Nguyen Ha Thanh; Yusuke Miyao; Ken Satoh
>
> **备注:** Three tables and one figure
>
> **摘要:** We introduce JBE-QA, a Japanese Bar Exam Question-Answering dataset to evaluate large language models' legal knowledge. Derived from the multiple-choice (tanto-shiki) section of the Japanese bar exam (2015-2024), JBE-QA provides the first comprehensive benchmark for Japanese legal-domain evaluation of LLMs. It covers the Civil Code, the Penal Code, and the Constitution, extending beyond the Civil Code focus of prior Japanese resources. Each question is decomposed into independent true/false judgments with structured contextual fields. The dataset contains 3,464 items with balanced labels. We evaluate 26 LLMs, including proprietary, open-weight, Japanese-specialised, and reasoning models. Our results show that proprietary models with reasoning enabled perform best, and the Constitution questions are generally easier than the Civil Code or the Penal Code questions.
>
---
#### [new 075] Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match
- **分类: cs.CL**

- **简介: 该论文针对大语言模型推理延迟问题，提出无需训练的松散推测解码方法FLy。通过利用目标模型自校正能力，放宽严格匹配验证，接纳语义正确但词序不同的生成，提升效率。设计双层机制与多级加速策略，实现高加速比与强泛化性，显著降低延迟并保持高准确率。**

- **链接: [https://arxiv.org/pdf/2511.22972v1](https://arxiv.org/pdf/2511.22972v1)**

> **作者:** Jinze Li; Yixing Xu; Guanchen Li; Shuo Yang; Jinfeng Xu; Xuanwu Yin; Dong Li; Edith C. H. Ngai; Emad Barsoum
>
> **备注:** Under review
>
> **摘要:** Large language models (LLMs) achieve strong performance across diverse tasks but suffer from high inference latency due to their autoregressive generation. Speculative Decoding (SPD) mitigates this issue by verifying candidate tokens in parallel from a smaller draft model, yet its strict exact-match verification discards many semantically valid continuations. Moreover, existing training-based SPD methods often suffer from performance degradation on out-of-distribution (OOD) tasks. To this end, we propose Training-Free Loosely Speculative Decoding (FLy), a novel method that loosens the rigid verification criterion by leveraging the target model's self-corrective behavior to judge whether a draft-target mismatch remains semantically valid. FLy introduces a two-tier mechanism: an entropy-level gate that identifies whether the current token allows multiple plausible alternatives or is nearly deterministic, and a token-level deferred window that distinguishes genuine errors from differently worded yet semantically correct variants. To further reduce latency, we design a multi-level acceleration strategy that accelerates not only the target model but also the drafter itself. Owing to its training-free design, FLy composes seamlessly with arbitrary draft-target pairs and generalizes across models and domains without hyperparameter re-tuning. Experiments show that FLy preserves more than 99% of the target model's accuracy while achieving an average 2.81x speedup on Llama-3.1-70B-Instruct and 5.07x speedup on the 405B variant. Notably, on out-of-domain datasets, our method remains highly effective and outperforms the training-based method EAGLE-3 by 1.62x.
>
---
#### [new 076] Mind Reading or Misreading? LLMs on the Big Five Personality Test
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLMs）在零样本条件下预测文本中人格特质的可行性，针对的是五大性格维度（BIG5）的二分类任务。研究对比了多个模型与提示策略，在多数据集上评估性能，发现当前模型难以可靠预测人格，尤其在外向性和神经质维度上表现不佳，强调需优化提示设计与评估指标以提升结果可解释性。**

- **链接: [https://arxiv.org/pdf/2511.23101v1](https://arxiv.org/pdf/2511.23101v1)**

> **作者:** Francesco Di Cursi; Chiara Boldrini; Marco Conti; Andrea Passarella
>
> **备注:** Funding: SoBigDatait (IR0000013), FAIR (PE00000013), ICSC (CN00000013)
>
> **摘要:** We evaluate large language models (LLMs) for automatic personality prediction from text under the binary Five Factor Model (BIG5). Five models -- including GPT-4 and lightweight open-source alternatives -- are tested across three heterogeneous datasets (Essays, MyPersonality, Pandora) and two prompting strategies (minimal vs. enriched with linguistic and psychological cues). Enriched prompts reduce invalid outputs and improve class balance, but also introduce a systematic bias toward predicting trait presence. Performance varies substantially: Openness and Agreeableness are relatively easier to detect, while Extraversion and Neuroticism remain challenging. Although open-source models sometimes approach GPT-4 and prior benchmarks, no configuration yields consistently reliable predictions in zero-shot binary settings. Moreover, aggregate metrics such as accuracy and macro-F1 mask significant asymmetries, with per-class recall offering clearer diagnostic value. These findings show that current out-of-the-box LLMs are not yet suitable for APPT, and that careful coordination of prompt design, trait framing, and evaluation metrics is essential for interpretable results.
>
---
#### [new 077] A Customer Journey in the Land of Oz: Leveraging the Wizard of Oz Technique to Model Emotions in Customer Service Interactions
- **分类: cs.CL**

- **简介: 该论文聚焦情感感知客服系统，针对现有情绪识别数据多为域外、标注单一的问题，通过沃兹机制（Wizard of Oz）构建了2148条双语对话数据集EmoWOZ-CS。研究评估了人工引导情绪轨迹的有效性，分析了情绪标注一致性与自我报告差异，并验证了实时情绪推断的挑战，推动了前瞻性情感支持的发展。**

- **链接: [https://arxiv.org/pdf/2511.21909v1](https://arxiv.org/pdf/2511.21909v1)**

> **作者:** Sofie Labat; Thomas Demeester; Véronique Hoste
>
> **摘要:** Emotion-aware customer service needs in-domain conversational data, rich annotations, and predictive capabilities, but existing resources for emotion recognition are often out-of-domain, narrowly labeled, and focused on post-hoc detection. To address this, we conducted a controlled Wizard of Oz (WOZ) experiment to elicit interactions with targeted affective trajectories. The resulting corpus, EmoWOZ-CS, contains 2,148 bilingual (Dutch-English) written dialogues from 179 participants across commercial aviation, e-commerce, online travel agencies, and telecommunication scenarios. Our contributions are threefold: (1) Evaluate WOZ-based operator-steered valence trajectories as a design for emotion research; (2) Quantify human annotation performance and variation, including divergences between self-reports and third-party judgments; (3) Benchmark detection and forward-looking emotion inference in real-time support. Findings show neutral dominates participant messages; desire and gratitude are the most frequent non-neutral emotions. Agreement is moderate for multilabel emotions and valence, lower for arousal and dominance; self-reports diverge notably from third-party labels, aligning most for neutral, gratitude, and anger. Objective strategies often elicit neutrality or gratitude, while suboptimal strategies increase anger, annoyance, disappointment, desire, and confusion. Some affective strategies (cheerfulness, gratitude) foster positive reciprocity, whereas others (apology, empathy) can also leave desire, anger, or annoyance. Temporal analysis confirms successful conversation-level steering toward prescribed trajectories, most distinctly for negative targets; positive and neutral targets yield similar final valence distributions. Benchmarks highlight the difficulty of forward-looking emotion inference from prior turns, underscoring the complexity of proactive emotion-aware support.
>
---
#### [new 078] Dripper: Token-Efficient Main HTML Extraction with a Lightweight LM
- **分类: cs.CL**

- **简介: 该论文针对网页主内容提取任务，解决大模型在上下文长度、推理成本和格式幻觉方面的瓶颈。提出Dripper框架，通过HTML简化、语义块分类、可控解码及新评估集WebMainBench，实现轻量模型高效精准提取，0.6B参数即达当前最佳性能。**

- **链接: [https://arxiv.org/pdf/2511.23119v1](https://arxiv.org/pdf/2511.23119v1)**

> **作者:** Mengjie Liu; Jiahui Peng; Pei Chu; Jiantao Qiu; Ren Ma; He Zhu; Rui Min; Lindong Lu; Wenchang Ning; Linfeng Hou; Kaiwen Liu; Yuan Qu; Zhenxiang Li; Chao Xu; Zhongying Tu; Wentao Zhang; Conghui He
>
> **摘要:** Accurately and efficiently extracting main content from general web pages is of great significance for obtaining training data for large models. Using well-pre-trained decoder-only generative language models offers excellent document comprehension capabilities, thereby effectively enhancing parsing quality. However, it remains constrained by issues such as context window length, inference cost, and format hallucination. We present Dripper, an efficient HTML main content extraction framework powered by lightweight language models, which addresses these challenges through four key innovations: (1) We design a specialized HTML simplification algorithm that reduces input token count to 22\% compared to raw HTML while preserving critical structural information; (2) We reformulate main content extraction as a semantic block sequence classification task, significantly reducing inference cost; (3) We introduce a controlled decoding mechanism that strictly constrains the output space through logits processors, effectively eliminating hallucination issues common in small-scale models; (4) We propose WebMainBench, an evaluation dataset containing over 7,800 web pages with meticulously human-annotated main content extraction labels. Experimental results demonstrate that using only a 0.6B parameter model, Dripper achieves state-of-the-art performance across all evaluation benchmarks and outperforms all baseline methods, attaining an ROUGE-N F1 score of 81.58\%( 83.13\% with fall-back strategy) on our proposed WebMainBench dataset.
>
---
#### [new 079] Exploring Performance Variations in Finetuned Translators of Ultra-Low Resource Languages: Do Linguistic Differences Matter?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究超低资源语言翻译中微调译码器性能差异的原因。针对巴西原住民语言，探究数据清洗、预训练模型、模型规模及数据量等因素的影响，发现这些因素影响有限，提示语言自身差异可能是关键因素。任务为低资源语言机器翻译。**

- **链接: [https://arxiv.org/pdf/2511.22482v1](https://arxiv.org/pdf/2511.22482v1)**

> **作者:** Isabel Gonçalves; Paulo Cavalin; Claudio Pinhanez
>
> **摘要:** Finetuning pre-trained language models with small amounts of data is a commonly-used method to create translators for ultra-low resource languages such as endangered Indigenous languages. However, previous works have reported substantially different performances with translators created using similar methodology and data. In this work we systematically explored possible causes of the performance difference, aiming to determine whether it was a product of different cleaning procedures, limitations of the pre-trained models, the size of the base model, or the size of the training dataset, studying both directions of translation. Our studies, using two Brazilian Indigenous languages, related but with significant structural linguistic characteristics, indicated none or very limited influence from those training factors, suggesting differences between languages may play a significant role in the ability to produce translators by fine-tuning pre-trained models.
>
---
#### [new 080] PeerCoPilot: A Language Model-Powered Assistant for Behavioral Health Organizations
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 该论文提出PeerCoPilot，一个基于大语言模型的助手，旨在帮助行为健康组织的同伴提供者更高效地制定个性化康复计划、设定具体目标并匹配资源。针对资源分散、人力不足的问题，通过检索增强生成技术整合1300+经审核资源，提升信息可靠性。用户评估显示超90%支持使用，且优于基线模型。**

- **链接: [https://arxiv.org/pdf/2511.21721v1](https://arxiv.org/pdf/2511.21721v1)**

> **作者:** Gao Mo; Naveen Raman; Megan Chai; Cindy Peng; Shannon Pagdon; Nev Jones; Hong Shen; Peggy Swarbrick; Fei Fang
>
> **备注:** Accepted at IAAI'26
>
> **摘要:** Behavioral health conditions, which include mental health and substance use disorders, are the leading disease burden in the United States. Peer-run behavioral health organizations (PROs) critically assist individuals facing these conditions by combining mental health services with assistance for needs such as income, employment, and housing. However, limited funds and staffing make it difficult for PROs to address all service user needs. To assist peer providers at PROs with their day-to-day tasks, we introduce PeerCoPilot, a large language model (LLM)-powered assistant that helps peer providers create wellness plans, construct step-by-step goals, and locate organizational resources to support these goals. PeerCoPilot ensures information reliability through a retrieval-augmented generation pipeline backed by a large database of over 1,300 vetted resources. We conducted human evaluations with 15 peer providers and 6 service users and found that over 90% of users supported using PeerCoPilot. Moreover, we demonstrated that PeerCoPilot provides more reliable and specific information than a baseline LLM. PeerCoPilot is now used by a group of 5-10 peer providers at CSPNJ, a large behavioral health organization serving over 10,000 service users, and we are actively expanding PeerCoPilot's use.
>
---
#### [new 081] Evaluating Embedding Generalization: How LLMs, LoRA, and SLERP Shape Representational Geometry
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究密集文本嵌入的泛化能力，对比LLM与非LLM编码器在数值序列上的表现，探究LoRA适配与SLERP模型融合对表示几何的影响。通过数论属性分类任务，发现LLM嵌入更优但易过拟合，而SLERP融合能有效恢复通用性，提升聚类质量与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.21703v1](https://arxiv.org/pdf/2511.21703v1)**

> **作者:** Siyaxolisa Kabane
>
> **备注:** 20 pages, 16 figures
>
> **摘要:** We investigate the generalization properties of dense text embeddings when the embedding backbone is a large language model (LLM) versus when it is a non-LLM encoder, and we study the extent to which spherical linear interpolation (SLERP) model-merging mitigates over-specialization introduced by task-specific adaptation (e.g., LoRA). To make the comparison concrete and domain-agnostic, we design a controlled suite of experiments in which models embed short numerical sequences and are evaluated on their ability to cluster and classify those sequences according to well-defined number-theoretic properties. Our experimental protocol compares four families of models: (1) non-LLM encoders trained from scratch or fine-tuned for embeddings, (2) LLM-based encoders adapted with parameter-efficient methods (LoRA), (3) LLM-based encoders with LoRA followed by model souping merging into the base weights, and (4) the same LoRA-adapted LLMs merged using SLERP across checkpoints or stages. We evaluate representational quality with clustering indices (Silhouette and Davies Bouldin). We additionally analyze the use of kmeans labels to see if the embeddings encode any other information besides the one we are testing for. Empirically, we find that LLM-based backbones produce embeddings that better capture higher-order, compositional numeric patterns, but are prone to adapter dominance that degrades balanced generalization; SLERP merging consistently recovers base-model structure while retaining most task gains, yielding superior tradeoffs in clustering separability, and robustness compared to model souping or models that were not merged.
>
---
#### [new 082] Smarter, not Bigger: Fine-Tuned RAG-Enhanced LLMs for Automotive HIL Testing
- **分类: cs.CL**

- **简介: 该论文针对汽车HIL测试中测试用例与需求碎片化、利用率低的问题，提出HIL-GPT系统。通过领域自适应的LLM与语义检索结合，利用细调嵌入模型实现高效可追溯的检索。实验表明，小型优化模型在准确率、延迟和成本上优于大模型，用户研究也验证其更优的实用性与满意度。**

- **链接: [https://arxiv.org/pdf/2511.22584v1](https://arxiv.org/pdf/2511.22584v1)**

> **作者:** Chao Feng; Zihan Liu; Siddhant Gupta; Gongpei Cui; Jan von der Assen; Burkhard Stiller
>
> **摘要:** Hardware-in-the-Loop (HIL) testing is essential for automotive validation but suffers from fragmented and underutilized test artifacts. This paper presents HIL-GPT, a retrieval-augmented generation (RAG) system integrating domain-adapted large language models (LLMs) with semantic retrieval. HIL-GPT leverages embedding fine-tuning using a domain-specific dataset constructed via heuristic mining and LLM-assisted synthesis, combined with vector indexing for scalable, traceable test case and requirement retrieval. Experiments show that fine-tuned compact models, such as \texttt{bge-base-en-v1.5}, achieve a superior trade-off between accuracy, latency, and cost compared to larger models, challenging the notion that bigger is always better. An A/B user study further confirms that RAG-enhanced assistants improve perceived helpfulness, truthfulness, and satisfaction over general-purpose LLMs. These findings provide insights for deploying efficient, domain-aligned LLM-based assistants in industrial HIL environments.
>
---
#### [new 083] RefineBench: Evaluating Refinement Capability of Language Models via Checklists
- **分类: cs.CL**

- **简介: 该论文提出RefineBench基准，评估大语言模型在开放任务中自我修正的能力。针对模型自反思与反馈引导下的改进效果，构建11领域1000题清单，发现模型在无指导时提升有限，而有反馈时可接近完美，揭示自修正需重大突破。**

- **链接: [https://arxiv.org/pdf/2511.22173v1](https://arxiv.org/pdf/2511.22173v1)**

> **作者:** Young-Jun Lee; Seungone Kim; Byung-Kwan Lee; Minkyeong Moon; Yechan Hwang; Jong Myoung Kim; Graham Neubig; Sean Welleck; Ho-Jin Choi
>
> **备注:** Project website: https://passing2961.github.io/refinebench-page/
>
> **摘要:** Can language models (LMs) self-refine their own responses? This question is increasingly relevant as a wide range of real-world user interactions involve refinement requests. However, prior studies have largely tested LMs' refinement abilities on verifiable tasks such as competition math or symbolic reasoning with simplified scaffolds, whereas users often pose open-ended queries and provide varying degrees of feedback on what they desire. The recent advent of reasoning models that exhibit self-reflection patterns in their chains-of-thought further motivates this question. To analyze this, we introduce RefineBench, a benchmark of 1,000 challenging problems across 11 domains paired with a checklist-based evaluation framework. We evaluate two refinement modes: (1) guided refinement, where an LM is provided natural language feedback, and (2) self-refinement, where LMs attempt to improve without guidance. In the self-refinement setting, even frontier LMs such as Gemini 2.5 Pro and GPT-5 achieve modest baseline scores of 31.3% and 29.1%, respectively, and most models fail to consistently improve across iterations (e.g., Gemini-2.5-Pro gains only +1.8%, while DeepSeek-R1 declines by -0.1%). By contrast, in guided refinement, both proprietary LMs and large open-weight LMs (>70B) can leverage targeted feedback to refine responses to near-perfect levels within five turns. These findings suggest that frontier LMs require breakthroughs to self-refine their incorrect responses, and that RefineBench provides a valuable testbed for tracking progress.
>
---
#### [new 084] Orchestrating Dual-Boundaries: An Arithmetic Intensity Inspired Acceleration Framework for Diffusion Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对扩散语言模型（dLLM）推理效率低的问题，提出ODB-dLLM框架。针对预填充阶段冗余计算和解码阶段迭代过多，分别设计自适应长度预测与跳步共享推测解码，显著提升速度并缓解精度下降。属于大模型高效推理任务。**

- **链接: [https://arxiv.org/pdf/2511.21759v1](https://arxiv.org/pdf/2511.21759v1)**

> **作者:** Linye Wei; Wenjue Chen; Pingzhi Tang; Xiaotian Guo; Le Ye; Runsheng Wang; Meng Li
>
> **摘要:** Diffusion-based large language models (dLLMs) have recently gained significant attention for their exceptional performance and inherent potential for parallel decoding. Existing frameworks further enhance its inference efficiency by enabling KV caching. However, its bidirectional attention mechanism necessitates periodic cache refreshes that interleave prefill and decoding phases, both contributing substantial inference cost and constraining achievable speedup. Inspired by the heterogeneous arithmetic intensity of the prefill and decoding phases, we propose ODB-dLLM, a framework that orchestrates dual-boundaries to accelerate dLLM inference. In the prefill phase, we find that the predefined fixed response length introduces heavy yet redundant computational overhead, which affects efficiency. To alleviate this, ODB-dLLM incorporates an adaptive length prediction mechanism that progressively reduces prefill overhead and unnecessary computation. In the decoding phase, we analyze the computational characteristics of dLLMs and propose a dLLM-specific jump-share speculative decoding method to enhance efficiency by reducing the number of decoding iterations. Experimental results demonstrate that ODB-dLLM achieves 46-162x and 2.63-6.30x speedups over the baseline dLLM and Fast-dLLM, respectively, while simultaneously mitigating the accuracy degradation in existing acceleration frameworks.
>
---
#### [new 085] Token-Level Marginalization for Multi-Label LLM Classifiers
- **分类: cs.CL**

- **简介: 该论文针对多标签内容安全分类中生成式大模型缺乏类级别概率的问题，提出三种基于token logits的置信度估计方法，以提升模型可解释性与准确性，支持动态阈值设定与细粒度错误分析。**

- **链接: [https://arxiv.org/pdf/2511.22312v1](https://arxiv.org/pdf/2511.22312v1)**

> **作者:** Anjaneya Praharaj; Jaykumar Kasundra
>
> **摘要:** This paper addresses the critical challenge of deriving interpretable confidence scores from generative language models (LLMs) when applied to multi-label content safety classification. While models like LLaMA Guard are effective for identifying unsafe content and its categories, their generative architecture inherently lacks direct class-level probabilities, which hinders model confidence assessment and performance interpretation. This limitation complicates the setting of dynamic thresholds for content moderation and impedes fine-grained error analysis. This research proposes and evaluates three novel token-level probability estimation approaches to bridge this gap. The aim is to enhance model interpretability and accuracy, and evaluate the generalizability of this framework across different instruction-tuned models. Through extensive experimentation on a synthetically generated, rigorously annotated dataset, it is demonstrated that leveraging token logits significantly improves the interpretability and reliability of generative classifiers, enabling more nuanced content safety moderation.
>
---
#### [new 086] FLAWS: A Benchmark for Error Identification and Localization in Scientific Papers
- **分类: cs.CL; cs.AI; cs.DL; cs.LG**

- **简介: 该论文提出FLAWS基准，用于评估大语言模型在科学论文中识别与定位错误的能力。针对人工审稿压力大、错误难发现的问题，构建713个带错误的论文对，通过自动化方法插入有效错误并设计评估指标，测试五款前沿模型，结果显示GPT 5表现最佳。**

- **链接: [https://arxiv.org/pdf/2511.21843v1](https://arxiv.org/pdf/2511.21843v1)**

> **作者:** Sarina Xi; Vishisht Rao; Justin Payan; Nihar B. Shah
>
> **备注:** 30 pages, 12 tables, 2 figures
>
> **摘要:** The identification and localization of errors is a core task in peer review, yet the exponential growth of scientific output has made it increasingly difficult for human reviewers to reliably detect errors given the limited pool of experts. Recent advances in Large Language Models (LLMs) have sparked interest in their potential to support such evaluation tasks, from academic peer review to automated scientific assessment. However, despite the growing use of LLMs in review systems, their capabilities to pinpoint errors remain underexplored. In this work, we introduce Fault Localization Across Writing in Science (FLAWS), an automated benchmark consisting of 713 paper-error pairs designed to evaluate how effectively LLMs detect errors that undermine key claims in research papers. We construct the benchmark by systematically inserting claim-invalidating errors into peer-reviewed papers using LLMs, paired with an automated evaluation metric that measures whether models can identify and localize these errors. Developing such a benchmark presents unique challenges that we overcome: ensuring that the inserted errors are well-defined, challenging, and relevant to the content of the paper, avoiding artifacts that would make identification trivial, and designing a scalable, automated evaluation metric. On the resulting benchmark, we evaluate five frontier LLMs: Claude Sonnet 4.5, DeepSeek Reasoner v3.1, Gemini 2.5 Pro, GPT 5, and Grok 4. Among these, GPT 5 is the top-performing model, achieving 39.1% identification accuracy when k=10, where k is the number of top-ranked error text candidates generated by the LLM.
>
---
#### [new 087] ResearchArcade: Graph Interface for Academic Tasks
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出ResearchArcade，一个基于图结构的统一学术数据接口，旨在整合多源异构学术数据（如ArXiv论文、OpenReview评审），支持跨源、多模态、时序演化的学术任务。通过统一任务定义与模型输入，提升机器学习在科研全流程中的应用效果，实验验证其在六类任务中均优于基线方法。**

- **链接: [https://arxiv.org/pdf/2511.22036v1](https://arxiv.org/pdf/2511.22036v1)**

> **作者:** Jingjun Xu; Chongshan Lin; Haofei Yu; Tao Feng; Jiaxuan You
>
> **摘要:** Academic research generates diverse data sources, and as researchers increasingly use machine learning to assist research tasks, a crucial question arises: Can we build a unified data interface to support the development of machine learning models for various academic tasks? Models trained on such a unified interface can better support human researchers throughout the research process, eventually accelerating knowledge discovery. In this work, we introduce ResearchArcade, a graph-based interface that connects multiple academic data sources, unifies task definitions, and supports a wide range of base models to address key academic challenges. ResearchArcade utilizes a coherent multi-table format with graph structures to organize data from different sources, including academic corpora from ArXiv and peer reviews from OpenReview, while capturing information with multiple modalities, such as text, figures, and tables. ResearchArcade also preserves temporal evolution at both the manuscript and community levels, supporting the study of paper revisions as well as broader research trends over time. Additionally, ResearchArcade unifies diverse academic task definitions and supports various models with distinct input requirements. Our experiments across six academic tasks demonstrate that combining cross-source and multi-modal information enables a broader range of tasks, while incorporating graph structures consistently improves performance over baseline methods. This highlights the effectiveness of ResearchArcade and its potential to advance research progress.
>
---
#### [new 088] Lips-Jaw and Tongue-Jaw Articulatory Tradeoff in DYNARTmo
- **分类: cs.CL; cs.RO**

- **简介: 该论文研究语音生成中唇-下颌与舌-下颌的协同运动机制，针对动态构音模型DYNARTmo如何实现多发音器间的努力分配问题。通过仿真不同辅音-元音组合，验证了模型能再现真实口腔运动模式，如下颌随发音部位变化、舌颌共动等，证明其在简化假设下仍可有效模拟构音协同效应。**

- **链接: [https://arxiv.org/pdf/2511.22155v1](https://arxiv.org/pdf/2511.22155v1)**

> **作者:** Bernd J. Kröger
>
> **备注:** 12 pages, 3 figures, supplementary material: python code
>
> **摘要:** This paper investigates how the dynamic articulatory model DYNARTmo accounts for articulatory tradeoffs between primary and secondary articulators, with a focus on lips-jaw and tongue-jaw coordination. While DYNARTmo does not implement full task-dynamic second-order biomechanics, it adopts first-order task-space gesture specifications comparable to those used in articulatory phonology and integrates a simplified mechanism for distributing articulatory effort across multiple articulators. We first outline the conceptual relationship between task dynamics and DYNARTmo, emphasizing the distinction between high-level task-space trajectories and their low-level articulatory execution. We then present simulation results for a set of CV syllables that illustrate how jaw displacement varies as a function of both place of articulation (labial, apical, dorsal) and vowel context (/a/, /i/, /u/). The model reproduces empirically attested patterns of articulatory synergy, including jaw-supported apical closures, lower-lip elevation in bilabial stops, tongue-jaw co-movement, and saturation effects in labial constrictions. These results demonstrate that even with computationally simplified assumptions, DYNARTmo can generate realistic spatio-temporal movement patterns that capture key aspects of articulatory tradeoff and synergy across a range of consonant-vowel combinations.
>
---
#### [new 089] A Comparative Study of LLM Prompting and Fine-Tuning for Cross-genre Authorship Attribution on Chinese Lyrics
- **分类: cs.CL**

- **简介: 该论文研究中文歌词跨流派作者归属任务，针对缺乏公开高质量数据的问题，构建了首个平衡的中文歌词数据集，并对比了大模型零样本推理与微调模型的表现。实验发现，细粒度流派（如民谣）识别准确率更高，且测试集设计缺陷影响微调效果评估。研究提出优化建议，为该领域提供基准与工具。**

- **链接: [https://arxiv.org/pdf/2511.21930v1](https://arxiv.org/pdf/2511.21930v1)**

> **作者:** Yuxin Li; Lorraine Xu; Meng Fan Wang
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** We propose a novel study on authorship attribution for Chinese lyrics, a domain where clean, public datasets are sorely lacking. Our contributions are twofold: (1) we create a new, balanced dataset of Chinese lyrics spanning multiple genres, and (2) we develop and fine-tune a domain-specific model, comparing its performance against zero-shot inference using the DeepSeek LLM. We test two central hypotheses. First, we hypothesize that a fine-tuned model will outperform a zero-shot LLM baseline. Second, we hypothesize that performance is genre-dependent. Our experiments strongly confirm Hypothesis 2: structured genres (e.g. Folklore & Tradition) yield significantly higher attribution accuracy than more abstract genres (e.g. Love & Romance). Hypothesis 1 receives only partial support: fine-tuning improves robustness and generalization in Test1 (real-world data and difficult genres), but offers limited or ambiguous gains in Test2, a smaller, synthetically-augmented set. We show that the design limitations of Test2 (e.g., label imbalance, shallow lexical differences, and narrow genre sampling) can obscure the true effectiveness of fine-tuning. Our work establishes the first benchmark for cross-genre Chinese lyric attribution, highlights the importance of genre-sensitive evaluation, and provides a public dataset and analytical framework for future research. We conclude with recommendations: enlarge and diversify test sets, reduce reliance on token-level data augmentation, balance author representation across genres, and investigate domain-adaptive pretraining as a pathway for improved attribution performance.
>
---
#### [new 090] A Theoretically Grounded Hybrid Ensemble for Reliable Detection of LLM-Generated Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型生成文本检测任务，解决现有方法泛化差、误报率高的问题。提出基于理论的混合集成模型，融合语义、概率与统计特征，通过优化加权投票提升性能，在学术文本上实现94.2%准确率与35%误报率降低。**

- **链接: [https://arxiv.org/pdf/2511.22153v1](https://arxiv.org/pdf/2511.22153v1)**

> **作者:** Sepyan Purnama Kristanto; Lutfi Hakim
>
> **备注:** 24 pages
>
> **摘要:** The rapid proliferation of Large Language Models (LLMs) has blurred the line between human and machine authorship, creating practical risks for academic integrity and information reliability. Existing text detectors typically rely on a single methodological paradigm and suffer from poor generalization and high false positive rates (FPR), especially on high-stakes academic text. We propose a theoretically grounded hybrid ensemble that systematically fuses three complementary detection paradigms: (i) a RoBERTa-based transformer classifier for deep semantic feature extraction, (ii) a GPT-2-based probabilistic detector using perturbation-induced likelihood curvature, and (iii) a statistical linguistic feature analyzer capturing stylometric patterns. The core novelty lies in an optimized weighted voting framework, where ensemble weights are learned on the probability simplex to maximize F1-score rather than set heuristically. We provide a bias-variance analysis and empirically demonstrate low inter-model correlation (rho ~ 0.35-0.42), a key condition for variance reduction. Evaluated on a large-scale, multigenerator corpus of 30,000 documents, our system achieves 94.2% accuracy and an AUC of 0.978, with a 35% relative reduction in false positives on academic text. This yields a more reliable and ethically responsible detector for real-world deployment in education and other high-stakes domains.
>
---
#### [new 091] Pooling Attention: Evaluating Pretrained Transformer Embeddings for Deception Classification
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究虚假新闻检测任务，旨在评估预训练Transformer模型在信息真实性判断中的表现。通过将BERT、GPT-2等模型作为冻结嵌入器，结合轻量分类器，对比不同池化策略与分类头，发现注意力编码具有强泛化能力，简单池化与线性分类器即可实现优异性能，验证了注意力机制在可信度分析中的有效性。**

- **链接: [https://arxiv.org/pdf/2511.22977v1](https://arxiv.org/pdf/2511.22977v1)**

> **作者:** Sumit Mamtani; Abhijeet Bhure
>
> **备注:** Accepted at the IEEE 7th Computing, Communications and IoT Applications Conference (ComComAp 2025), Madrid, Spain, December 2025. 6 pages
>
> **摘要:** This paper investigates fake news detection as a downstream evaluation of Transformer representations, benchmarking encoder-only and decoder-only pre-trained models (BERT, GPT-2, Transformer-XL) as frozen embedders paired with lightweight classifiers. Through controlled preprocessing comparing pooling versus padding and neural versus linear heads, results demonstrate that contextual self-attention encodings consistently transfer effectively. BERT embeddings combined with logistic regression outperform neural baselines on LIAR dataset splits, while analyses of sequence length and aggregation reveal robustness to truncation and advantages from simple max or average pooling. This work positions attention-based token encoders as robust, architecture-centric foundations for veracity tasks, isolating Transformer contributions from classifier complexity.
>
---
#### [new 092] Tourism Question Answer System in Indian Language using Domain-Adapted Foundation Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对印度语旅游问答任务，构建首个面向瓦拉纳西旅游的高精度中文-印地语问答数据集，提出基于领域自适应基础模型（BERT/ RoBERTa）的高效微调框架，结合LoRA与SFT提升参数效率，显著降低训练开销并保持高性能，为低资源文化语境下自然语言处理提供有效解决方案。**

- **链接: [https://arxiv.org/pdf/2511.23235v1](https://arxiv.org/pdf/2511.23235v1)**

> **作者:** Praveen Gatla; Anushka; Nikita Kanwar; Gouri Sahoo; Rajesh Kumar Mundotiya
>
> **摘要:** This article presents the first comprehensive study on designing a baseline extractive question-answering (QA) system for the Hindi tourism domain, with a specialized focus on the Varanasi-a cultural and spiritual hub renowned for its Bhakti-Bhaav (devotional ethos). Targeting ten tourism-centric subdomains-Ganga Aarti, Cruise, Food Court, Public Toilet, Kund, Museum, General, Ashram, Temple and Travel, the work addresses the absence of language-specific QA resources in Hindi for culturally nuanced applications. In this paper, a dataset comprising 7,715 Hindi QA pairs pertaining to Varanasi tourism was constructed and subsequently augmented with 27,455 pairs generated via Llama zero-shot prompting. We propose a framework leveraging foundation models-BERT and RoBERTa, fine-tuned using Supervised Fine-Tuning (SFT) and Low-Rank Adaptation (LoRA), to optimize parameter efficiency and task performance. Multiple variants of BERT, including pre-trained languages (e.g., Hindi-BERT), are evaluated to assess their suitability for low-resource domain-specific QA. Evaluation metrics - F1, BLEU, and ROUGE-L - highlight trade-offs between answer precision and linguistic fluency. Experiments demonstrate that LoRA-based fine-tuning achieves competitive performance (85.3\% F1) while reducing trainable parameters by 98\% compared to SFT, striking a balance between efficiency and accuracy. Comparative analysis across models reveals that RoBERTa with SFT outperforms BERT variants in capturing contextual nuances, particularly for culturally embedded terms (e.g., Aarti, Kund). This work establishes a foundational baseline for Hindi tourism QA systems, emphasizing the role of LORA in low-resource settings and underscoring the need for culturally contextualized NLP frameworks in the tourism domain.
>
---
#### [new 093] Cacheback: Speculative Decoding With Nothing But Cache
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Cacheback解码，一种无需训练、模型无关的推测性解码方法，利用令牌n-gram的LRU缓存加速大语言模型推理。通过挖掘语言局部性生成候选序列，实现高效推理，兼具高性能与易集成性，适用于快速适配新领域。**

- **链接: [https://arxiv.org/pdf/2511.21699v1](https://arxiv.org/pdf/2511.21699v1)**

> **作者:** Zhiyao Ma; In Gim; Lin Zhong
>
> **摘要:** We present Cacheback Decoding, a training-free and model-agnostic speculative decoding method that exploits the locality in language to accelerate Large Language Model (LLM) inference. Cacheback leverages only Least Recently Used (LRU) cache tables of token n-grams to generate draft sequences. Cacheback achieves state-of-the-art performance among comparable methods despite its minimalist design, and its simplicity allows easy integration into existing systems. Cacheback also shows potential for fast adaptation to new domains.
>
---
#### [new 094] Tracing How Annotators Think: Augmenting Preference Judgments with Reading Processes
- **分类: cs.CL**

- **简介: 该论文针对主观性NLP任务中的标注可靠性问题，提出结合鼠标追踪的注释方法，捕捉标注者阅读过程。通过构建PreferRead数据集，分析其在偏好判断任务中的阅读行为，发现重读与高一致性相关，揭示了认知过程对标注结果的影响，为理解标注者决策提供了新视角。**

- **链接: [https://arxiv.org/pdf/2511.21912v1](https://arxiv.org/pdf/2511.21912v1)**

> **作者:** Karin de Langis; William Walker; Khanh Chi Le; Dongyeop Kang
>
> **摘要:** We propose an annotation approach that captures not only labels but also the reading process underlying annotators' decisions, e.g., what parts of the text they focus on, re-read or skim. Using this framework, we conduct a case study on the preference annotation task, creating a dataset PreferRead that contains fine-grained annotator reading behaviors obtained from mouse tracking. PreferRead enables detailed analysis of how annotators navigate between a prompt and two candidate responses before selecting their preference. We find that annotators re-read a response in roughly half of all trials, most often revisiting the option they ultimately choose, and rarely revisit the prompt. Reading behaviors are also significantly related to annotation outcomes: re-reading is associated with higher inter-annotator agreement, whereas long reading paths and times are associated with lower agreement. These results demonstrate that reading processes provide a complementary cognitive dimension for understanding annotator reliability, decision-making and disagreement in complex, subjective NLP tasks. Our code and data are publicly available.
>
---
#### [new 095] Beyond Query-Level Comparison: Fine-Grained Reinforcement Learning for Text-to-SQL with Automated Interpretable Critiques
- **分类: cs.CL**

- **简介: 该论文针对文本转SQL任务中的评估瓶颈，提出RuCo-C框架。通过自动生成查询特定的评价标准与可解释批评，实现无监督细粒度评估，并引入渐进式探索策略优化强化学习奖励，显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.22258v1](https://arxiv.org/pdf/2511.22258v1)**

> **作者:** Guifeng Wang; Yuanfeng Song; Meng Yang; Tao Zhu; Xiaoming Yin; Xing Chen
>
> **摘要:** Text-to-SQL, a pivotal natural language processing (NLP) task that converts textual queries into executable SQL, has seen substantial progress in recent years. However, existing evaluation and reward mechanisms used to train and assess the text-to-SQL models remain a critical bottleneck. Current approaches heavily rely on manually annotated gold SQL queries, which are costly to produce and impractical for large-scale evaluation. More importantly, most reinforcement learning (RL) methods in text-to-SQL leverage only the final binary execution outcome as the reward signal, a coarse-grained supervision that overlooks detailed structural and semantic errors from the perspective of rubrics. To address these challenges, we propose RuCo-C, a novel generative judge model for fine-grained, query-specific automatic evaluation using interpretable critiques without human intervention. Our framework first automatically generates query-specific evaluation rubrics for human-free annotation, linking them to interpretable critiques. Subsequently, it integrates densified reward feedback through a "progressive exploration" strategy during the RL training process, which dynamically adjusts the rewards to enhance the model's performance. Comprehensive experiments demonstrate that RuCo-C outperforms existing methods in text-to-SQL evaluation, yielding significant performance gains.
>
---
#### [new 096] A Hybrid Theory and Data-driven Approach to Persuasion Detection with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于说服力检测任务，旨在解决社交媒体中大规模文本互动下信念改变预测难题。结合心理学理论与大语言模型，提取八项特征，构建随机森林模型，发现“认知情绪”和“分享意愿”为关键预测因子，提升了在线说服力分析能力。**

- **链接: [https://arxiv.org/pdf/2511.22109v1](https://arxiv.org/pdf/2511.22109v1)**

> **作者:** Gia Bao Hoang; Keith J Ransom; Rachel Stephens; Carolyn Semmler; Nicolas Fay; Lewis Mitchell
>
> **摘要:** Traditional psychological models of belief revision focus on face-to-face interactions, but with the rise of social media, more effective models are needed to capture belief revision at scale, in this rich text-based online discourse. Here, we use a hybrid approach, utilizing large language models (LLMs) to develop a model that predicts successful persuasion using features derived from psychological experiments. Our approach leverages LLM generated ratings of features previously examined in the literature to build a random forest classification model that predicts whether a message will result in belief change. Of the eight features tested, \textit{epistemic emotion} and \textit{willingness to share} were the top-ranking predictors of belief change in the model. Our findings provide insights into the characteristics of persuasive messages and demonstrate how LLMs can enhance models of successful persuasion based on psychological theory. Given these insights, this work has broader applications in fields such as online influence detection and misinformation mitigation, as well as measuring the effectiveness of online narratives.
>
---
#### [new 097] Proactive Defense: Compound AI for Detecting Persuasion Attacks and Measuring Inoculation Effectiveness
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对生成式AI在信息环境中的说服攻击检测与防护问题，提出BRIES复合AI系统。通过四类智能体实现攻击生成、检测、防御与效果评估，揭示不同大模型对复杂说服技巧的识别差异及提示工程影响，量化认知脆弱性，构建前置免疫干预框架，提升人类认知韧性。**

- **链接: [https://arxiv.org/pdf/2511.21749v1](https://arxiv.org/pdf/2511.21749v1)**

> **作者:** Svitlana Volkova; Will Dupree; Hsien-Te Kao; Peter Bautista; Gabe Ganberg; Jeff Beaubien; Laura Cassani
>
> **摘要:** This paper introduces BRIES, a novel compound AI architecture designed to detect and measure the effectiveness of persuasion attacks across information environments. We present a system with specialized agents: a Twister that generates adversarial content employing targeted persuasion tactics, a Detector that identifies attack types with configurable parameters, a Defender that creates resilient content through content inoculation, and an Assessor that employs causal inference to evaluate inoculation effectiveness. Experimenting with the SemEval 2023 Task 3 taxonomy across the synthetic persuasion dataset, we demonstrate significant variations in detection performance across language agents. Our comparative analysis reveals significant performance disparities with GPT-4 achieving superior detection accuracy on complex persuasion techniques, while open-source models like Llama3 and Mistral demonstrated notable weaknesses in identifying subtle rhetorical, suggesting that different architectures encode and process persuasive language patterns in fundamentally different ways. We show that prompt engineering dramatically affects detection efficacy, with temperature settings and confidence scoring producing model-specific variations; Gemma and GPT-4 perform optimally at lower temperatures while Llama3 and Mistral show improved capabilities at higher temperatures. Our causal analysis provides novel insights into socio-emotional-cognitive signatures of persuasion attacks, revealing that different attack types target specific cognitive dimensions. This research advances generative AI safety and cognitive security by quantifying LLM-specific vulnerabilities to persuasion attacks and delivers a framework for enhancing human cognitive resilience through structured interventions before exposure to harmful content.
>
---
#### [new 098] GPS: General Per-Sample Prompter
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出GPS，一种无需任务特定训练的通用自适应提示方法。针对传统自动提示需大量数据、耗时优化且无法个性化输入的问题，GPS通过强化学习训练提示生成器，为每个输入动态生成定制化提示，并采用最小贝叶斯风险解码稳定推理。实验表明其在多任务上表现优异，实现高效精准的个性化提示生成。**

- **链接: [https://arxiv.org/pdf/2511.21714v1](https://arxiv.org/pdf/2511.21714v1)**

> **作者:** Pawel Batorski; Paul Swoboda
>
> **摘要:** LLMs are sensitive to prompting, with task performance often hinging on subtle, sometimes imperceptible variations in phrasing. As a result, crafting effective prompts manually remains challenging and time-consuming. Recent automatic prompting methods mitigate this difficulty but face three key limitations: (i) for each new task, they require large datasets to train good prompts;(ii) they rely on costly optimization loops that may take hours; (iii)they typically produce a single task-level prompt that does not adapt to the individual input problem to be solved. We propose GPS, the first general-purpose, per-sample prompting method. Without any task-specific tuning, GPS generates a tailored prompt for each unseen input, improving performance across diverse tasks. The prompter is trained with reinforcement learning on a suite of training tasks and includes a novel regularization for effectively adapting to per-sample prompting. Finally, we employ Minimum Bayes Risk decoding to stabilize inference. Empirically, GPS demonstrates competitive performance: we attain second best results among baselines on text simplification, third best results on summarization and on-par results on classification, while not training on any of these tasks, in contrast to the baselines. For in-domain prompting, we obtain sota on GSM8K. Our work shows the potential of a novel and effective paradigm for automatic prompting: generating adaptive, input-specific prompts without extensive optimization and without access to a task-specific training set. Our code is available at https://github.com/Batorskq/GPS.
>
---
#### [new 099] EvalCards: A Framework for Standardized Evaluation Reporting
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文针对自然语言处理中评估报告透明度不足的问题，提出EvalCards框架，旨在提升评估的可复现性、可访问性和治理性。通过标准化报告格式，解决当前评估文档缺乏统一规范的痛点，为研究者与实践者提供透明、可信赖的评估信息披露方案。**

- **链接: [https://arxiv.org/pdf/2511.21695v1](https://arxiv.org/pdf/2511.21695v1)**

> **作者:** Ruchira Dhar; Danae Sanchez Villegas; Antonia Karamolegkou; Alice Schiavone; Yifei Yuan; Xinyi Chen; Jiaang Li; Stella Frank; Laura De Grazia; Monorama Swain; Stephanie Brandl; Daniel Hershcovich; Anders Søgaard; Desmond Elliott
>
> **备注:** Under review
>
> **摘要:** Evaluation has long been a central concern in NLP, and transparent reporting practices are more critical than ever in today's landscape of rapidly released open-access models. Drawing on a survey of recent work on evaluation and documentation, we identify three persistent shortcomings in current reporting practices: reproducibility, accessibility, and governance. We argue that existing standardization efforts remain insufficient and introduce Evaluation Disclosure Cards (EvalCards) as a path forward. EvalCards are designed to enhance transparency for both researchers and practitioners while providing a practical foundation to meet emerging governance requirements.
>
---
#### [new 100] Focused Chain-of-Thought: Efficient LLM Reasoning via Structured Input Information
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型推理中冗余文本导致的高延迟与高耗能问题，提出无需训练的输入中心方法F-CoT。通过将查询中的关键信息结构化为简洁上下文，引导模型仅基于此推理，有效减少注意力干扰，实现2-3倍的生成文本压缩，同时保持高准确率，显著提升推理效率。**

- **链接: [https://arxiv.org/pdf/2511.22176v1](https://arxiv.org/pdf/2511.22176v1)**

> **作者:** Lukas Struppek; Dominik Hintersdorf; Hannah Struppek; Daniel Neider; Kristian Kersting
>
> **摘要:** Recent large language models achieve strong reasoning performance by generating detailed chain-of-thought traces, but this often leads to excessive token use and high inference latency. Existing efficiency approaches typically focus on model-centric interventions, such as reinforcement learning or supervised fine-tuning, to reduce verbosity. In contrast, we propose a training-free, input-centric approach. Inspired by cognitive psychology, we introduce Focused Chain-of-Thought (F-CoT), which separates information extraction from the reasoning process. F-CoT first organizes the essential information from a query into a concise, structured context and then guides the model to reason exclusively over this context. By preventing attention to irrelevant details, F-CoT naturally produces shorter reasoning paths. On arithmetic word problems, F-CoT reduces generated tokens by 2-3x while maintaining accuracy comparable to standard zero-shot CoT. These results highlight structured input as a simple yet effective lever for more efficient LLM reasoning.
>
---
#### [new 101] A General Highly Accurate Online Planning Method Integrating Large Language Models into Nested Rollout Policy Adaptation for Dialogue Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对目标导向对话任务中策略难以适应新场景、训练成本高的问题，提出NRPA-GD方法。通过引入大语言模型（LLM）模拟用户与系统行为，结合嵌套蒙特卡洛仿真与策略自适应机制，实现无需训练的高精度在线规划。实验表明，该方法在多个数据集上优于现有方法，仅用0.6亿参数的LLM即超越ChatGPT。**

- **链接: [https://arxiv.org/pdf/2511.21706v1](https://arxiv.org/pdf/2511.21706v1)**

> **作者:** Hui Wang; Fafa Zhang; Xiaoyu Zhang; Chaoxu Mu
>
> **摘要:** In goal-oriented dialogue tasks, the main challenge is to steer the interaction towards a given goal within a limited number of turns. Existing approaches either rely on elaborate prompt engineering, whose effectiveness is heavily dependent on human experience, or integrate policy networks and pre-trained policy models, which are usually difficult to adapt to new dialogue scenarios and costly to train. Therefore, in this paper, we present Nested Rollout Policy Adaptation for Goal-oriented Dialogue (NRPA-GD), a novel dialogue policy planning method that completely avoids specific model training by utilizing a Large Language Model (LLM) to simulate behaviors of user and system at the same time. Specifically, NRPA-GD constructs a complete evaluation mechanism for dialogue trajectories and employs an optimization framework of nested Monte Carlo simulation and policy self-adaptation to dynamically adjust policies during the dialogue process. The experimental results on four typical goal-oriented dialogue datasets show that NRPA-GD outperforms both existing prompt engineering and specifically pre-trained model-based methods. Impressively, NRPA-GD surpasses ChatGPT and pre-trained policy models with only a 0.6-billion-parameter LLM. The proposed approach further demonstrates the advantages and novelty of employing planning methods on LLMs to solve practical planning tasks.
>
---
#### [new 102] Goal-Directed Search Outperforms Goal-Agnostic Memory Compression in Long-Context Memory Tasks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大模型长时记忆任务，解决传统记忆压缩方法因依赖特定数据分布而引入人类偏见的问题。提出SUMER框架，通过无压缩内存中的目标导向搜索，利用强化学习自主获取信息，显著优于现有压缩方法与全上下文基线，在LoCoMo数据集上实现SOTA性能。**

- **链接: [https://arxiv.org/pdf/2511.21726v1](https://arxiv.org/pdf/2511.21726v1)**

> **作者:** Yicong Zheng; Kevin L. McKee; Thomas Miconi; Zacharie Bugaud; Mick van Gelderen; Jed McCaleb
>
> **摘要:** How to enable human-like long-term memory in large language models (LLMs) has been a central question for unlocking more general capabilities such as few-shot generalization. Existing memory frameworks and benchmarks focus on finding the optimal memory compression algorithm for higher performance in tasks that require recollection and sometimes further reasoning. However, such efforts have ended up building more human bias into the compression algorithm, through the search for the best prompts and memory architectures that suit specific benchmarks, rather than finding a general solution that would work on other data distributions. On the other hand, goal-directed search on uncompressed information could potentially exhibit superior performance because compression is lossy, and a predefined compression algorithm will not fit all raw data distributions. Here we present SUMER (Search in Uncompressed Memory via Experience Replay), an end-to-end reinforcement learning agent with verifiable reward (RLVR) that learns to use search tools to gather information and answer a target question. On the LoCoMo dataset for long-context conversation understanding, SUMER with Qwen2.5-7B-Instruct learned to use search tools and outperformed all other biased memory compression approaches and also the full-context baseline, reaching SOTA performance (43% gain over the prior best). We demonstrate that a simple search method applied to raw data outperforms goal-agnostic and biased compression algorithms in current long-context memory tasks, arguing for new paradigms and benchmarks that are more dynamic and autonomously scalable. Code for SUMER and all implemented baselines is publicly available at https://github.com/zycyc/SUMER.
>
---
#### [new 103] Beyond Component Strength: Synergistic Integration and Adaptive Calibration in Multi-Agent RAG Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多智能体检索增强生成（RAG）系统，旨在解决组件孤立使用时效果不佳的问题。通过50个查询的消融实验，发现协同集成与自适应校准比单一组件强化更关键，显著降低弃答率至2%且不增加幻觉，强调标准化指标的重要性。**

- **链接: [https://arxiv.org/pdf/2511.21729v1](https://arxiv.org/pdf/2511.21729v1)**

> **作者:** Jithin Krishnan
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Building reliable retrieval-augmented generation (RAG) systems requires more than adding powerful components; it requires understanding how they interact. Using ablation studies on 50 queries (15 answerable, 10 edge cases, and 25 adversarial), we show that enhancements such as hybrid retrieval, ensemble verification, and adaptive thresholding provide almost no benefit when used in isolation, yet together achieve a 95% reduction in abstention (from 40% to 2%) without increasing hallucinations. We also identify a measurement challenge: different verification strategies can behave safely but assign inconsistent labels (for example, "abstained" versus "unsupported"), creating apparent hallucination rates that are actually artifacts of labeling. Our results show that synergistic integration matters more than the strength of any single component, that standardized metrics and labels are essential for correctly interpreting performance, and that adaptive calibration is needed to prevent overconfident over-answering even when retrieval quality is high.
>
---
#### [new 104] Conveying Imagistic Thinking in TCM Translation: A Prompt Engineering and LLM-Based Evaluation Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对中医典籍翻译中意象思维传递不足的问题，提出一种人机协同的提示工程与大模型评估框架。通过引导大模型识别并转化隐喻、转喻，结合模拟读者评估，验证了优化后翻译在认知维度上的优越性，为古籍概念密集文本的高效精准翻译提供了可复制的方法路径。**

- **链接: [https://arxiv.org/pdf/2511.23059v1](https://arxiv.org/pdf/2511.23059v1)**

> **作者:** Jiatong Han
>
> **备注:** 3 figures
>
> **摘要:** Traditional Chinese Medicine theory is built on imagistic thinking, in which medical principles and diagnostic and therapeutic logic are structured through metaphor and metonymy. However, existing English translations largely rely on literal rendering, making it difficult for target-language readers to reconstruct the underlying conceptual networks and apply them in clinical practice. This study adopted a human-in-the-loop framework and selected four passages from the medical canon Huangdi Neijing that are fundamental in theory. Through prompt-based cognitive scaffolding, DeepSeek V3.1 was guided to identify metaphor and metonymy in the source text and convey the theory in translation. In the evaluation stage, ChatGPT 5 Pro and Gemini 2.5 Pro were instructed by prompts to simulate three types of real-world readers. Human translations, baseline model translations, and prompt-adjusted translations were scored by the simulated readers across five cognitive dimensions, followed by structured interviews and Interpretative Phenomenological Analysis. Results show that the prompt-adjusted LLM translations perform best across all five dimensions, with high cross-model and cross-role consistency. The interview themes reveal differences between human and machine translation, effective strategies for metaphor and metonymy transfer, and readers' cognitive preferences. This study provides a cognitive, efficient and replicable HITL methodological pathway for translation of ancient, concept-dense texts like TCM.
>
---
#### [new 105] ThetaEvolve: Test-time Learning on Open Problems
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ThetaEvolve，一个开源的测试时学习框架，用于在开放优化问题（如圆盘打包、自相关不等式）上持续进化。针对大模型难以内化演化策略的问题，引入单模型、程序库、批量采样与奖励塑形，实现高效强化学习，使小型开源模型达成新最优解，并验证了模型具备可迁移的学习能力。**

- **链接: [https://arxiv.org/pdf/2511.23473v1](https://arxiv.org/pdf/2511.23473v1)**

> **作者:** Yiping Wang; Shao-Rong Su; Zhiyuan Zeng; Eva Xu; Liliang Ren; Xinyu Yang; Zeyi Huang; Xuehai He; Luyao Ma; Baolin Peng; Hao Cheng; Pengcheng He; Weizhu Chen; Shuohang Wang; Simon Shaolei Du; Yelong Shen
>
> **备注:** 30 pages, link: https://github.com/ypwang61/ThetaEvolve
>
> **摘要:** Recent advances in large language models (LLMs) have enabled breakthroughs in mathematical discovery, exemplified by AlphaEvolve, a closed-source system that evolves programs to improve bounds on open problems. However, it relies on ensembles of frontier LLMs to achieve new bounds and is a pure inference system that models cannot internalize the evolving strategies. We introduce ThetaEvolve, an open-source framework that simplifies and extends AlphaEvolve to efficiently scale both in-context learning and Reinforcement Learning (RL) at test time, allowing models to continually learn from their experiences in improving open optimization problems. ThetaEvolve features a single LLM, a large program database for enhanced exploration, batch sampling for higher throughput, lazy penalties to discourage stagnant outputs, and optional reward shaping for stable training signals, etc. ThetaEvolve is the first evolving framework that enable a small open-source model, like DeepSeek-R1-0528-Qwen3-8B, to achieve new best-known bounds on open problems (circle packing and first auto-correlation inequality) mentioned in AlphaEvolve. Besides, across two models and four open tasks, we find that ThetaEvolve with RL at test-time consistently outperforms inference-only baselines, and the model indeed learns evolving capabilities, as the RL-trained checkpoints demonstrate faster progress and better final performance on both trained target task and other unseen tasks. We release our code publicly: https://github.com/ypwang61/ThetaEvolve
>
---
#### [new 106] SO-Bench: A Structural Output Evaluation of Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文针对多模态大模型在视觉输入下生成结构化输出的能力，提出SO-Bench基准。解决现有缺乏系统评估框架的问题，覆盖四类视觉场景，包含超6500个JSON模式与1800对图像-模式对。通过基准测试揭示模型在结构化推理上的不足，并开展训练优化，推动多模态结构化生成发展。**

- **链接: [https://arxiv.org/pdf/2511.21750v1](https://arxiv.org/pdf/2511.21750v1)**

> **作者:** Di Feng; Kaixin Ma; Feng Nan; Haofeng Chen; Bohan Zhai; David Griffiths; Mingfei Gao; Zhe Gan; Eshan Verma; Yinfei Yang; Zhifeng Chen; Afshin Dehghan
>
> **摘要:** Multimodal large language models (MLLMs) are increasingly deployed in real-world, agentic settings where outputs must not only be correct, but also conform to predefined data schemas. Despite recent progress in structured generation in textual domain, there is still no benchmark that systematically evaluates schema-grounded information extraction and reasoning over visual inputs. In this work, we conduct a comprehensive study of visual structural output capabilities for MLLMs with our carefully designed SO-Bench benchmark. Covering four visual domains, including UI screens, natural images, documents, and charts, SO-Bench is built from over 6.5K diverse JSON schemas and 1.8K curated image-schema pairs with human-verified quality. Benchmarking experiments on open-sourced and frontier proprietary models reveal persistent gaps in predicting accurate, schema compliant outputs, highlighting the need for better multimodal structured reasoning. Beyond benchmarking, we further conduct training experiments to largely improve the model's structured output capability. We plan to make the benchmark available to the community.
>
---
#### [new 107] What Shape Is Optimal for Masks in Text Removal?
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文研究文档图像中文本移除任务，针对复杂密集文本场景下掩码形状对修复效果的影响问题，提出基于贝叶斯优化的灵活掩码建模方法，发现字符级掩码更优，非最小覆盖最优，为手动掩码提供实用指导。**

- **链接: [https://arxiv.org/pdf/2511.22499v1](https://arxiv.org/pdf/2511.22499v1)**

> **作者:** Hyakka Nakada; Marika Kubota
>
> **备注:** 12 pages, 17 figures
>
> **摘要:** The advent of generative models has dramatically improved the accuracy of image inpainting. In particular, by removing specific text from document images, reconstructing original images is extremely important for industrial applications. However, most existing methods of text removal focus on deleting simple scene text which appears in images captured by a camera in an outdoor environment. There is little research dedicated to complex and practical images with dense text. Therefore, we created benchmark data for text removal from images including a large amount of text. From the data, we found that text-removal performance becomes vulnerable against mask profile perturbation. Thus, for practical text-removal tasks, precise tuning of the mask shape is essential. This study developed a method to model highly flexible mask profiles and learn their parameters using Bayesian optimization. The resulting profiles were found to be character-wise masks. It was also found that the minimum cover of a text region is not optimal. Our research is expected to pave the way for a user-friendly guideline for manual masking.
>
---
#### [new 108] From Topology to Retrieval: Decoding Embedding Spaces with Unified Signatures
- **分类: cs.LG; cs.CL; cs.IR**

- **简介: 该论文研究文本嵌入空间的几何与拓扑结构，旨在解决嵌入空间表征不清晰、度量冗余的问题。提出统一拓扑签名（UTS）框架，通过多属性分析揭示模型架构影响，并成功预测文档可检索性，提升模型可解释性与下游任务性能。**

- **链接: [https://arxiv.org/pdf/2511.22150v1](https://arxiv.org/pdf/2511.22150v1)**

> **作者:** Florian Rottach; William Rudman; Bastain Rieck; Harrisen Scells; Carsten Eickhoff
>
> **摘要:** Studying how embeddings are organized in space not only enhances model interpretability but also uncovers factors that drive downstream task performance. In this paper, we present a comprehensive analysis of topological and geometric measures across a wide set of text embedding models and datasets. We find a high degree of redundancy among these measures and observe that individual metrics often fail to sufficiently differentiate embedding spaces. Building on these insights, we introduce Unified Topological Signatures (UTS), a holistic framework for characterizing embedding spaces. We show that UTS can predict model-specific properties and reveal similarities driven by model architecture. Further, we demonstrate the utility of our method by linking topological structure to ranking effectiveness and accurately predicting document retrievability. We find that a holistic, multi-attribute perspective is essential to understanding and leveraging the geometry of text embeddings.
>
---
#### [new 109] Swarms of Large Language Model Agents for Protein Sequence Design with Experimental Validation
- **分类: cs.AI; cond-mat.mes-hall; cond-mat.soft; cs.CL; cs.LG**

- **简介: 该论文提出一种基于大语言模型代理的群体智能框架，用于从头设计蛋白质序列。针对传统方法依赖微调、数据及结构模板的问题，该框架通过并行运行的LLM代理，实现位置级协同优化，无需预训练或特定数据，高效生成具有目标功能的蛋白序列，并经实验验证，适用于多种结构类型。**

- **链接: [https://arxiv.org/pdf/2511.22311v1](https://arxiv.org/pdf/2511.22311v1)**

> **作者:** Fiona Y. Wang; Di Sheng Lee; David L. Kaplan; Markus J. Buehler
>
> **摘要:** Designing proteins de novo with tailored structural, physicochemical, and functional properties remains a grand challenge in biotechnology, medicine, and materials science, due to the vastness of sequence space and the complex coupling between sequence, structure, and function. Current state-of-the-art generative methods, such as protein language models (PLMs) and diffusion-based architectures, often require extensive fine-tuning, task-specific data, or model reconfiguration to support objective-directed design, thereby limiting their flexibility and scalability. To overcome these limitations, we present a decentralized, agent-based framework inspired by swarm intelligence for de novo protein design. In this approach, multiple large language model (LLM) agents operate in parallel, each assigned to a specific residue position. These agents iteratively propose context-aware mutations by integrating design objectives, local neighborhood interactions, and memory and feedback from previous iterations. This position-wise, decentralized coordination enables emergent design of diverse, well-defined sequences without reliance on motif scaffolds or multiple sequence alignments, validated with experiments on proteins with alpha helix and coil structures. Through analyses of residue conservation, structure-based metrics, and sequence convergence and embeddings, we demonstrate that the framework exhibits emergent behaviors and effective navigation of the protein fitness landscape. Our method achieves efficient, objective-directed designs within a few GPU-hours and operates entirely without fine-tuning or specialized training, offering a generalizable and adaptable solution for protein design. Beyond proteins, the approach lays the groundwork for collective LLM-driven design across biomolecular systems and other scientific discovery tasks.
>
---
#### [new 110] Intelligent Neural Networks: From Layered Architectures to Graph-Organized Intelligence
- **分类: cs.LG; cs.CL; cs.NE**

- **简介: 该论文提出智能神经网络（INN），将神经元设计为具内部记忆和自学习通信的智能单元，采用完全图结构而非传统层级架构。针对序列建模任务，INN在Text8上实现1.705 BPC，优于Transformer与LSTM，并展现训练稳定性优势，证明了神经元中心化与图结构组织的有效性。**

- **链接: [https://arxiv.org/pdf/2511.22813v1](https://arxiv.org/pdf/2511.22813v1)**

> **作者:** Antoine Salomon
>
> **备注:** Code available at https://github.com/AntoineSal/IntelligentNeuralNetwork
>
> **摘要:** Biological neurons exhibit remarkable intelligence: they maintain internal states, communicate selectively with other neurons, and self-organize into complex graphs rather than rigid hierarchical layers. What if artificial intelligence could emerge from similarly intelligent computational units? We introduce Intelligent Neural Networks (INN), a paradigm shift where neurons are first-class entities with internal memory and learned communication patterns, organized in complete graphs rather than sequential layers. Each Intelligent Neuron combines selective state-space dynamics (knowing when to activate) with attention-based routing (knowing to whom to send signals), enabling emergent computation through graph-structured interactions. On the standard Text8 character modeling benchmark, INN achieves 1.705 Bit-Per-Character (BPC), significantly outperforming a comparable Transformer (2.055 BPC) and matching a highly optimized LSTM baseline. Crucially, a parameter-matched baseline of stacked Mamba blocks fails to converge (>3.4 BPC) under the same training protocol, demonstrating that INN's graph topology provides essential training stability. Ablation studies confirm this: removing inter-neuron communication degrades performance or leads to instability, proving the value of learned neural routing. This work demonstrates that neuron-centric design with graph organization is not merely bio-inspired -- it is computationally effective, opening new directions for modular, interpretable, and scalable neural architectures.
>
---
#### [new 111] PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration
- **分类: cs.CR; cs.CL**

- **简介: 该论文针对云边协同大模型推理中的隐私与性能矛盾问题，提出PRISM框架。通过动态感知输入敏感性，实现按需隐私保护与语义协作，结合自适应差分隐私与边缘小模型精炼，显著提升隐私-效用平衡，降低能耗与延迟。**

- **链接: [https://arxiv.org/pdf/2511.22788v1](https://arxiv.org/pdf/2511.22788v1)**

> **作者:** Junfei Zhan; Haoxun Shen; Zheng Lin; Tengjiao He
>
> **备注:** Accepted to AAAI 2026. This is the arXiv preprint version
>
> **摘要:** Large Language Models (LLMs) demonstrate impressive capabilities in natural language understanding and generation, but incur high communication overhead and privacy risks in cloud deployments, while facing compute and memory constraints when confined to edge devices. Cloud-edge inference has emerged as a promising paradigm for improving privacy in LLM services by retaining sensitive computations on local devices. However, existing cloud-edge inference approaches apply uniform privacy protection without considering input sensitivity, resulting in unnecessary perturbation and degraded utility even for non-sensitive tokens. To address this limitation, we propose Privacy-aware Routing for Inference with Semantic Modulation (PRISM), a context-aware framework that dynamically balances privacy and inference quality. PRISM executes in four stages: (1) the edge device profiles entity-level sensitivity; (2) a soft gating module on the edge selects an execution mode - cloud, edge, or collaboration; (3) for collaborative paths, the edge applies adaptive two-layer local differential privacy based on entity risks; and (4) the cloud LLM generates a semantic sketch from the perturbed prompt, which is then refined by the edge-side small language model (SLM) using local context. Our results show that PRISM consistently achieves superior privacy-utility trade-offs across various scenarios, reducing energy consumption and latency to 40-50% of baseline methods such as Uniform and Selective LDP, while maintaining high output quality under strong privacy constraints. These findings are validated through comprehensive evaluations involving realistic prompts, actual energy measurements, and heterogeneous cloud-edge model deployments.
>
---
#### [new 112] DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦于数学定理证明任务，旨在解决大模型仅追求正确答案而忽视推理过程严谨性的问题。提出通过训练自验证的验证器，引导生成器自我检查并修正证明，实现更可靠的数学推理。利用扩展的验证计算构建新训练数据，提升验证能力，最终在多项国际数学竞赛中取得优异成绩。**

- **链接: [https://arxiv.org/pdf/2511.22570v1](https://arxiv.org/pdf/2511.22570v1)**

> **作者:** Zhihong Shao; Yuxiang Luo; Chengda Lu; Z. Z. Ren; Jiewen Hu; Tian Ye; Zhibin Gou; Shirong Ma; Xiaokang Zhang
>
> **摘要:** Large language models have made significant progress in mathematical reasoning, which serves as an important testbed for AI and could impact scientific research if further advanced. By scaling reasoning with reinforcement learning that rewards correct final answers, LLMs have improved from poor performance to saturating quantitative reasoning competitions like AIME and HMMT in one year. However, this approach faces fundamental limitations. Pursuing higher final answer accuracy doesn't address a key issue: correct answers don't guarantee correct reasoning. Moreover, many mathematical tasks like theorem proving require rigorous step-by-step derivation rather than numerical answers, making final answer rewards inapplicable. To push the limits of deep reasoning, we believe it is necessary to verify the comprehensiveness and rigor of mathematical reasoning. Self-verification is particularly important for scaling test-time compute, especially for open problems without known solutions. Towards self-verifiable mathematical reasoning, we investigate how to train an accurate and faithful LLM-based verifier for theorem proving. We then train a proof generator using the verifier as the reward model, and incentivize the generator to identify and resolve as many issues as possible in their own proofs before finalizing them. To maintain the generation-verification gap as the generator becomes stronger, we propose to scale verification compute to automatically label new hard-to-verify proofs, creating training data to further improve the verifier. Our resulting model, DeepSeekMath-V2, demonstrates strong theorem-proving capabilities, achieving gold-level scores on IMO 2025 and CMO 2024 and a near-perfect 118/120 on Putnam 2024 with scaled test-time compute.
>
---
#### [new 113] PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel
- **分类: cs.DC; cs.CL**

- **简介: 该论文针对大语言模型（LLM）推理中的解码注意力瓶颈问题，提出PAT方法。通过感知前缀共享，采用打包-前向-合并范式，优化内存访问与资源利用，显著降低注意力延迟与吞吐等待时间，提升解码效率。**

- **链接: [https://arxiv.org/pdf/2511.22333v1](https://arxiv.org/pdf/2511.22333v1)**

> **作者:** Jinjun Yi; Zhixin Zhao; Yitao Hu; Ke Yan; Weiwei Sun; Hao Wang; Laiping Zhao; Yuhao Zhang; Wenxin Li; Keqiu Li
>
> **备注:** Accepted by ASPLOS'26
>
> **摘要:** LLM serving is increasingly dominated by decode attention, which is a memory-bound operation due to massive KV cache loading from global memory. Meanwhile, real-world workloads exhibit substantial, hierarchical shared prefixes across requests (e.g., system prompts, tools/templates, RAG). Existing attention implementations fail to fully exploit prefix sharing: *one-query-per-CTA* execution repeatedly loads shared prefix KV cache, while *one-size-fits-all* tiling leaves on-chip resources idle and exacerbates bubbles for uneven KV lengths. These choices amplify memory bandwidth pressure and stall memory-bound decode attention. This paper introduces PAT, a prefix-aware attention kernel implementation for LLM decoding that organizes execution with a pack-forward-merge paradigm. PAT packs queries by shared prefix to reduce repeated memory accesses, runs a customized multi-tile kernel to achieve high resource efficiency. It further applies practical multi-stream forwarding and KV splitting to reduce resource bubbles. The final merge performs online softmax with negligible overhead. We implement PAT as an off-the-shelf plugin for vLLM. Evaluation on both real-world and synthetic workloads shows that PAT reduces attention latency by 67.4% on average and TPOT by 13.6-83.4% under the same configurations against state-of-the-art attention kernels.
>
---
#### [new 114] SuRe: Surprise-Driven Prioritised Replay for Continual LLM Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究持续学习中的语言模型遗忘问题，提出基于“意外性”的优先回放（SuRe）与双学习器架构。通过高负对数似然序列选择与慢速权重融合，有效缓解灾难性遗忘，在多任务场景下实现更优性能，显著提升模型适应能力。**

- **链接: [https://arxiv.org/pdf/2511.22367v1](https://arxiv.org/pdf/2511.22367v1)**

> **作者:** Hugo Hazard; Zafeirios Fountas; Martin A. Benfeghoul; Adnan Oomerjee; Jun Wang; Haitham Bou-Ammar
>
> **摘要:** Continual learning, one's ability to adapt to a sequence of tasks without forgetting previously acquired knowledge, remains a major challenge in machine learning and a key gap between artificial and human intelligence. While regularisation and replay perform well in vision, they lag behind multi-task learning for large language models (LLMs), especially at scale with many tasks. We revisit replay and argue that two failure modes drive this gap: selection (what to rehearse) and integration (how to consolidate new knowledge). To address selection, we propose Surprise-prioritised Replay (SuRe), a simple, architecture-agnostic rule that ranks and stores the most surprising (high Negative Log-Likelihood) sequences. SuRe achieves state-of-the-art performance in the Large Number of Tasks (LNT) setting and delivers the best overall average across both Standard CL and LNT benchmarks. To address integration, we add a dual-learner design with fast and slow LoRA adapters merged via an exponential moving average (EMA), enabling rapid adaptation while stabilising long-term knowledge. Combining SuRe with the dual learner yields further gains, including improvements of up to +5 accuracy points on LNT over prior SOTA. Ablation studies confirm that our proposed method remains robust under reduced replay frequency and small buffer size, demonstrating both effectiveness and sample efficiency. Taken together, our results establish replay as a strong baseline for continual LLM fine-tuning and demonstrate that surprise-based selection and slow-weight consolidation are complementary components for mitigating catastrophic forgetting.
>
---
#### [new 115] Is Passive Expertise-Based Personalization Enough? A Case Study in AI-Assisted Test-Taking
- **分类: cs.HC; cs.CL**

- **简介: 该论文研究AI辅助考试中基于专家水平的个性化策略。针对新手与专家用户在任务对话中的偏好差异，通过构建具有被动个性化功能的AI助手，开展用户实验。结果表明，被动个性化虽能降低任务负荷、提升助手感知，但存在局限性，需结合主动个性化以优化体验与效率。**

- **链接: [https://arxiv.org/pdf/2511.23376v1](https://arxiv.org/pdf/2511.23376v1)**

> **作者:** Li Siyan; Jason Zhang; Akash Maharaj; Yuanming Shi; Yunyao Li
>
> **备注:** Accepted into Tailoring AI: Exploring Active and Passive LLM Personalization (PALS) workshop at EMNLP 2025
>
> **摘要:** Novice and expert users have different systematic preferences in task-oriented dialogues. However, whether catering to these preferences actually improves user experience and task performance remains understudied. To investigate the effects of expertise-based personalization, we first built a version of an enterprise AI assistant with passive personalization. We then conducted a user study where participants completed timed exams, aided by the two versions of the AI assistant. Preliminary results indicate that passive personalization helps reduce task load and improve assistant perception, but reveal task-specific limitations that can be addressed through providing more user agency. These findings underscore the importance of combining active and passive personalization to optimize user experience and effectiveness in enterprise task-oriented environments.
>
---
#### [new 116] From Compound Figures to Composite Understanding: Developing a Multi-Modal LLM from Biomedical Literature with Medical Multiple-Image Benchmarking and Validation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对医疗多模态大模型在多图像理解上的不足，提出基于生物医学文献中复合图像的五阶段指令生成框架，构建M3LLM模型，实现跨模态、时空关系的综合理解。通过自建专家验证的PMC-MI-Bench基准，验证其在多图像分析中的卓越性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.22232v1](https://arxiv.org/pdf/2511.22232v1)**

> **作者:** Zhen Chen; Yihang Fu; Gabriel Madera; Mauro Giuffre; Serina Applebaum; Hyunjae Kim; Hua Xu; Qingyu Chen
>
> **摘要:** Multi-modal large language models (MLLMs) have shown promise in advancing healthcare. However, most existing models remain confined to single-image understanding, which greatly limits their applicability in clinical workflows. In practice, medical diagnosis and progression often require synthesizing information across multiple images from different modalities or time points. The development of medical MLLMs capable of such multi-image understanding has been hindered by the lack of large-scale, high-quality annotated training data. To address this limitation, we propose a novel framework that leverages license-permissive compound images in biomedical literature, as a rich yet underutilized data source for multi-image analysis. Specifically, we design a five-stage, context-aware instruction generation paradigm underpinned by a divide-and-conquer strategy. By decomposing multi-image analysis into manageable sub-tasks, this paradigm empowers MLLMs to move beyond single-panel analysis and provide a composite understanding by learning the complex spatial, temporal, and cross-modal relationships inherent in these compound figures. By parsing over 237,000 compound figures and their contextual text for instruction generation, we develop M3LLM, a medical multi-image multi-modal large language model. For benchmarking, we construct PMC-MI-Bench for composite understanding, manually validated by medical experts. Extensive experiments show that M3LLM significantly outperforms both general-purpose and specialized medical MLLMs across multi-image, single-image, text-only, and multi-choice scenarios. Notably, M3LLM exhibits strong generalization to longitudinal chest X-ray analysis using the MIMIC dataset. This work establishes a scalable and efficient paradigm for developing medical MLLMs capable of composite reasoning, bridging the gap between biomedical literature and real-world clinical applications.
>
---
#### [new 117] Artwork Interpretation with Vision Language Models: A Case Study on Emotions and Emotion Symbols
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型（VLMs）在艺术作品情绪与情绪符号解读中的能力。针对情绪表达抽象性及符号识别难题，通过案例分析三种VLMs对四类递进问题的回答，并结合专家评估，发现模型对具体图像表现良好，但在抽象/象征性图像上表现不佳，且存在回答不一致问题。**

- **链接: [https://arxiv.org/pdf/2511.22929v1](https://arxiv.org/pdf/2511.22929v1)**

> **作者:** Sebastian Padó; Kerstin Thomas
>
> **备注:** Accepted for publication at the IJCNLP-AACL workshop on Multimodal Models for Low-Resource Contexts and Social Impact
>
> **摘要:** Emotions are a fundamental aspect of artistic expression. Due to their abstract nature, there is a broad spectrum of emotion realization in artworks. These are subject to historical change and their analysis requires expertise in art history. In this article, we investigate which aspects of emotional expression can be detected by current (2025) vision language models (VLMs). We present a case study of three VLMs (Llava-Llama and two Qwen models) in which we ask these models four sets of questions of increasing complexity about artworks (general content, emotional content, expression of emotions, and emotion symbols) and carry out a qualitative expert evaluation. We find that the VLMs recognize the content of the images surprisingly well and often also which emotions they depict and how they are expressed. The models perform best for concrete images but fail for highly abstract or highly symbolic images. Reliable recognition of symbols remains fundamentally difficult. Furthermore, the models continue to exhibit the well-known LLM weakness of providing inconsistent answers to related questions.
>
---
#### [new 118] Mechanistic Finetuning of Vision-Language-Action Models via Few-Shot Demonstrations
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在机器人任务中因物理差异需精细调优的问题，提出“机器人引导”方法。通过少量示范识别任务特定注意力头，实现精准、高效、可解释的微调，显著提升模型在真实机器人上的适应性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.22697v1](https://arxiv.org/pdf/2511.22697v1)**

> **作者:** Chancharik Mitra; Yusen Luo; Raj Saravanan; Dantong Niu; Anirudh Pai; Jesse Thomason; Trevor Darrell; Abrar Anwar; Deva Ramanan; Roei Herzig
>
> **摘要:** Vision-Language Action (VLAs) models promise to extend the remarkable success of vision-language models (VLMs) to robotics. Yet, unlike VLMs in the vision-language domain, VLAs for robotics require finetuning to contend with varying physical factors like robot embodiment, environment characteristics, and spatial relationships of each task. Existing fine-tuning methods lack specificity, adapting the same set of parameters regardless of a task's visual, linguistic, and physical characteristics. Inspired by functional specificity in neuroscience, we hypothesize that it is more effective to finetune sparse model representations specific to a given task. In this work, we introduce Robotic Steering, a finetuning approach grounded in mechanistic interpretability that leverages few-shot demonstrations to identify and selectively finetune task-specific attention heads aligned with the physical, visual, and linguistic requirements of robotic tasks. Through comprehensive on-robot evaluations with a Franka Emika robot arm, we demonstrate that Robotic Steering outperforms LoRA while achieving superior robustness under task variation, reduced computational cost, and enhanced interpretability for adapting VLAs to diverse robotic tasks.
>
---
#### [new 119] Medical Malice: A Dataset for Context-Aware Safety in Healthcare LLMs
- **分类: cs.CY; cs.AI; cs.CL; cs.CR**

- **简介: 该论文针对医疗领域LLM的安全问题，提出“Medical Malice”数据集，包含21.4万条基于巴西卫生系统（SUS）的对抗性提示，涵盖七类临床与行政违规。通过引入违规理由，促进模型理解伦理边界，推动从通用安全向情境感知安全的转变，以应对医疗AI中的复杂系统性风险。**

- **链接: [https://arxiv.org/pdf/2511.21757v1](https://arxiv.org/pdf/2511.21757v1)**

> **作者:** Andrew Maranhão Ventura D'addario
>
> **摘要:** The integration of Large Language Models (LLMs) into healthcare demands a safety paradigm rooted in \textit{primum non nocere}. However, current alignment techniques rely on generic definitions of harm that fail to capture context-dependent violations, such as administrative fraud and clinical discrimination. To address this, we introduce Medical Malice: a dataset of 214,219 adversarial prompts calibrated to the regulatory and ethical complexities of the Brazilian Unified Health System (SUS). Crucially, the dataset includes the reasoning behind each violation, enabling models to internalize ethical boundaries rather than merely memorizing a fixed set of refusals. Using an unaligned agent (Grok-4) within a persona-driven pipeline, we synthesized high-fidelity threats across seven taxonomies, ranging from procurement manipulation and queue-jumping to obstetric violence. We discuss the ethical design of releasing these "vulnerability signatures" to correct the information asymmetry between malicious actors and AI developers. Ultimately, this work advocates for a shift from universal to context-aware safety, providing the necessary resources to immunize healthcare AI against the nuanced, systemic threats inherent to high-stakes medical environments -- vulnerabilities that represent the paramount risk to patient safety and the successful integration of AI in healthcare systems.
>
---
#### [new 120] Transformer-Driven Triple Fusion Framework for Enhanced Multimodal Author Intent Classification in Low-Resource Bangla
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对低资源语言孟加拉语社交媒体文本的作者意图分类任务，提出基于Transformer的三模态融合框架BangACMM。通过结合文本与视觉特征，采用中间层融合策略，显著提升分类性能，达到84.11%宏F1，优于先前方法，为低资源语言多模态理解提供新范式。**

- **链接: [https://arxiv.org/pdf/2511.23287v1](https://arxiv.org/pdf/2511.23287v1)**

> **作者:** Ariful Islam; Tanvir Mahmud; Md Rifat Hossen
>
> **备注:** Accepted at the 28th International Conference on Computer and Information Technology (ICCIT 2025). To be published in IEEE proceedings
>
> **摘要:** The expansion of the Internet and social networks has led to an explosion of user-generated content. Author intent understanding plays a crucial role in interpreting social media content. This paper addresses author intent classification in Bangla social media posts by leveraging both textual and visual data. Recognizing limitations in previous unimodal approaches, we systematically benchmark transformer-based language models (mBERT, DistilBERT, XLM-RoBERTa) and vision architectures (ViT, Swin, SwiftFormer, ResNet, DenseNet, MobileNet), utilizing the Uddessho dataset of 3,048 posts spanning six practical intent categories. We introduce a novel intermediate fusion strategy that significantly outperforms early and late fusion on this task. Experimental results show that intermediate fusion, particularly with mBERT and Swin Transformer, achieves 84.11% macro-F1 score, establishing a new state-of-the-art with an 8.4 percentage-point improvement over prior Bangla multimodal approaches. Our analysis demonstrates that integrating visual context substantially enhances intent classification. Cross-modal feature integration at intermediate levels provides optimal balance between modality-specific representation and cross-modal learning. This research establishes new benchmarks and methodological standards for Bangla and other low-resource languages. We call our proposed framework BangACMM (Bangla Author Content MultiModal).
>
---
#### [new 121] Toward Automatic Safe Driving Instruction: A Large-Scale Vision Language Model Approach
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究自动驾驶中的安全驾驶指导任务，旨在通过多摄像头视频分析实现自动安全提醒。针对仅依赖道路视角的局限，提出融合驾驶员与道路双视角输入的方案，构建数据集并验证微调后视觉语言模型在生成精准安全指令上的有效性，揭示其对细微复杂事件检测的挑战。**

- **链接: [https://arxiv.org/pdf/2511.23311v1](https://arxiv.org/pdf/2511.23311v1)**

> **作者:** Haruki Sakajo; Hiroshi Takato; Hiroshi Tsutsui; Komei Soda; Hidetaka Kamigaito; Taro Watanabe
>
> **备注:** Accepted to MMLoSo 2025
>
> **摘要:** Large-scale Vision Language Models (LVLMs) exhibit advanced capabilities in tasks that require visual information, including object detection. These capabilities have promising applications in various industrial domains, such as autonomous driving. For example, LVLMs can generate safety-oriented descriptions of videos captured by road-facing cameras. However, ensuring comprehensive safety requires monitoring driver-facing views as well to detect risky events, such as the use of mobiles while driving. Thus, the ability to process synchronized inputs is necessary from both driver-facing and road-facing cameras. In this study, we develop models and investigate the capabilities of LVLMs by constructing a dataset and evaluating their performance on this dataset. Our experimental results demonstrate that while pre-trained LVLMs have limited effectiveness, fine-tuned LVLMs can generate accurate and safety-aware driving instructions. Nonetheless, several challenges remain, particularly in detecting subtle or complex events in the video. Our findings and error analysis provide valuable insights that can contribute to the improvement of LVLM-based systems in this domain.
>
---
#### [new 122] Bharat Scene Text: A Novel Comprehensive Dataset and Benchmark for Indian Language Scene Text Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对印度多语言场景文本识别难题，提出Bharat Scene Text Dataset（BSTD），涵盖11种印度语言和英语，包含超10万词。数据集支持检测、脚本识别、单词识别及端到端识别任务。通过微调英文先进模型，揭示了印度语言场景文本识别的挑战与机遇，推动该领域研究发展。**

- **链接: [https://arxiv.org/pdf/2511.23071v1](https://arxiv.org/pdf/2511.23071v1)**

> **作者:** Anik De; Abhirama Subramanyam Penamakuri; Rajeev Yadav; Aditya Rathore; Harshiv Shah; Devesh Sharma; Sagar Agarwal; Pravin Kumar; Anand Mishra
>
> **备注:** Under Peer Review
>
> **摘要:** Reading scene text, that is, text appearing in images, has numerous application areas, including assistive technology, search, and e-commerce. Although scene text recognition in English has advanced significantly and is often considered nearly a solved problem, Indian language scene text recognition remains an open challenge. This is due to script diversity, non-standard fonts, and varying writing styles, and, more importantly, the lack of high-quality datasets and open-source models. To address these gaps, we introduce the Bharat Scene Text Dataset (BSTD) - a large-scale and comprehensive benchmark for studying Indian Language Scene Text Recognition. It comprises more than 100K words that span 11 Indian languages and English, sourced from over 6,500 scene images captured across various linguistic regions of India. The dataset is meticulously annotated and supports multiple scene text tasks, including: (i) Scene Text Detection, (ii) Script Identification, (iii) Cropped Word Recognition, and (iv) End-to-End Scene Text Recognition. We evaluated state-of-the-art models originally developed for English by adapting (fine-tuning) them for Indian languages. Our results highlight the challenges and opportunities in Indian language scene text recognition. We believe that this dataset represents a significant step toward advancing research in this domain. All our models and data are open source.
>
---
#### [new 123] ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文针对知识密集型视觉问答（KB-VQA）任务，解决现有模型在处理领域特定或需外部知识查询时因检索精度低、噪声多、推理能力弱导致的准确率问题。提出ReAG方法，结合粗细粒度检索与批判模型过滤，通过强化学习优化推理，提升答案准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.22715v1](https://arxiv.org/pdf/2511.22715v1)**

> **作者:** Alberto Compagnoni; Marco Morini; Sara Sarto; Federico Cocchi; Davide Caffagni; Marcella Cornia; Lorenzo Baraldi; Rita Cucchiara
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown impressive capabilities in jointly understanding text, images, and videos, often evaluated via Visual Question Answering (VQA). However, even state-of-the-art MLLMs struggle with domain-specific or knowledge-intensive queries, where relevant information is underrepresented in pre-training data. Knowledge-based VQA (KB-VQA) addresses this by retrieving external documents to condition answer generation, but current retrieval-augmented approaches suffer from low precision, noisy passages, and limited reasoning. To address this, we propose ReAG, a novel Reasoning-Augmented Multimodal RAG approach that combines coarse- and fine-grained retrieval with a critic model that filters irrelevant passages, ensuring high-quality additional context. The model follows a multi-stage training strategy leveraging reinforcement learning to enhance reasoning over retrieved content, while supervised fine-tuning serves only as a cold start. Extensive experiments on Encyclopedic-VQA and InfoSeek demonstrate that ReAG significantly outperforms prior methods, improving answer accuracy and providing interpretable reasoning grounded in retrieved evidence. Our source code is publicly available at: https://github.com/aimagelab/ReAG.
>
---
#### [new 124] BanglaSentNet: An Explainable Hybrid Deep Learning Framework for Multi-Aspect Sentiment Analysis with Cross-Domain Transfer Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对孟加拉语电商评论多方面情感分析难题，提出可解释的混合深度学习框架BanglaSentNet。解决低资源语言标注数据少、形态复杂、跨域泛化差等问题。通过动态加权集成LSTM、BiLSTM、GRU与BanglaBERT，并引入SHAP与注意力可视化提升可解释性，实现85%准确率与0.88 F1-score，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.23264v1](https://arxiv.org/pdf/2511.23264v1)**

> **作者:** Ariful Islam; Md Rifat Hossen; Tanvir Mahmud
>
> **备注:** Submitted to Springer Nature Computer Science (SNCS) as an extended version of our ICDSAIA 2025 conference paper
>
> **摘要:** Multi-aspect sentiment analysis of Bangla e-commerce reviews remains challenging due to limited annotated datasets, morphological complexity, code-mixing phenomena, and domain shift issues, affecting 300 million Bangla-speaking users. Existing approaches lack explainability and cross-domain generalization capabilities crucial for practical deployment. We present BanglaSentNet, an explainable hybrid deep learning framework integrating LSTM, BiLSTM, GRU, and BanglaBERT through dynamic weighted ensemble learning for multi-aspect sentiment classification. We introduce a dataset of 8,755 manually annotated Bangla product reviews across four aspects (Quality, Service, Price, Decoration) from major Bangladeshi e-commerce platforms. Our framework incorporates SHAP-based feature attribution and attention visualization for transparent insights. BanglaSentNet achieves 85% accuracy and 0.88 F1-score, outperforming standalone deep learning models by 3-7% and traditional approaches substantially. The explainability suite achieves 9.4/10 interpretability score with 87.6% human agreement. Cross-domain transfer learning experiments reveal robust generalization: zero-shot performance retains 67-76% effectiveness across diverse domains (BanglaBook reviews, social media, general e-commerce, news headlines); few-shot learning with 500-1000 samples achieves 90-95% of full fine-tuning performance, significantly reducing annotation costs. Real-world deployment demonstrates practical utility for Bangladeshi e-commerce platforms, enabling data-driven decision-making for pricing optimization, service improvement, and customer experience enhancement. This research establishes a new state-of-the-art benchmark for Bangla sentiment analysis, advances ensemble learning methodologies for low-resource languages, and provides actionable solutions for commercial applications.
>
---
#### [new 125] ORION: Teaching Language Models to Reason Efficiently in the Language of Thought
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出ORION框架，针对大语言模型推理冗长、低效的问题，受“思想语言假说”启发，引导模型以紧凑的符号化形式（Mentalese）进行推理。通过短长度偏好优化（SLPO）提升效率与准确率，在多个数学基准上实现4-16倍的推理压缩、5倍延迟降低及训练成本大幅下降，兼顾性能与效率。**

- **链接: [https://arxiv.org/pdf/2511.22891v1](https://arxiv.org/pdf/2511.22891v1)**

> **作者:** Kumar Tanmay; Kriti Aggarwal; Paul Pu Liang; Subhabrata Mukherjee
>
> **摘要:** Large Reasoning Models (LRMs) achieve strong performance in mathematics, code generation, and task planning, but their reliance on long chains of verbose "thinking" tokens leads to high latency, redundancy, and incoherent reasoning paths. Inspired by the Language of Thought Hypothesis, which posits that human reasoning operates over a symbolic, compositional mental language called Mentalese, we introduce a framework that trains models to reason in a similarly compact style. Mentalese encodes abstract reasoning as ultra-compressed, structured tokens, enabling models to solve complex problems with far fewer steps. To improve both efficiency and accuracy, we propose SHORTER LENGTH PREFERENCE OPTIMIZATION (SLPO), a reinforcement learning method that rewards concise solutions that stay correct, while still allowing longer reasoning when needed. Applied to Mentalese-aligned models, SLPO yields significantly higher compression rates by enabling concise reasoning that preserves the benefits of detailed thinking without the computational overhead. Across benchmarks including AIME 2024 and 2025, MinervaMath, OlympiadBench, Math500, and AMC, our ORION models produce reasoning traces with 4-16x fewer tokens, achieve up to 5x lower inference latency, and reduce training costs by 7-9x relative to the DeepSeek R1 Distilled model, while maintaining 90-98% of its accuracy. ORION also surpasses Claude and ChatGPT-4o by up to 5% in accuracy while maintaining 2x compression. These results show that Mentalese-style compressed reasoning offers a step toward human-like cognitive efficiency, enabling real-time, cost-effective reasoning without sacrificing accuracy.
>
---
## 更新

#### [replaced 001] CANVAS: A Benchmark for Vision-Language Models on Tool-Based User Interface Design
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出CANVAS基准，用于评估视觉语言模型（VLMs）在工具调用下的用户界面（UI）设计能力。针对现有缺乏工具驱动设计评测标准的问题，构建包含598个任务的基准，涵盖设计复现与修改两类任务，通过真实设计软件操作验证模型性能，揭示模型策略与错误模式，推动VLM在设计协作中的发展。**

- **链接: [https://arxiv.org/pdf/2511.20737v2](https://arxiv.org/pdf/2511.20737v2)**

> **作者:** Daeheon Jeong; Seoyeon Byun; Kihoon Son; Dae Hyun Kim; Juho Kim
>
> **摘要:** User interface (UI) design is an iterative process in which designers progressively refine their work with design software such as Figma or Sketch. Recent advances in vision language models (VLMs) with tool invocation suggest these models can operate design software to edit a UI design through iteration. Understanding and enhancing this capacity is important, as it highlights VLMs' potential to collaborate with designers within conventional software. However, as no existing benchmark evaluates tool-based design performance, the capacity remains unknown. To address this, we introduce CANVAS, a benchmark for VLMs on tool-based user interface design. Our benchmark contains 598 tool-based design tasks paired with ground-truth references sampled from 3.3K mobile UI designs across 30 function-based categories (e.g., onboarding, messaging). In each task, a VLM updates the design step-by-step through context-based tool invocations (e.g., create a rectangle as a button background), linked to design software. Specifically, CANVAS incorporates two task types: (i) design replication evaluates the ability to reproduce a whole UI screen; (ii) design modification evaluates the ability to modify a specific part of an existing screen. Results suggest that leading models exhibit more strategic tool invocations, improving design quality. Furthermore, we identify common error patterns models exhibit, guiding future work in enhancing tool-based design capabilities.
>
---
#### [replaced 002] Agentar-Scale-SQL: Advancing Text-to-SQL through Orchestrated Test-Time Scaling
- **分类: cs.CL; cs.DB**

- **简介: 该论文针对文本转SQL任务中模型性能滞后于人类的问题，提出Agentar-Scale-SQL框架。通过协同的测试时扩展策略，融合内部、序列与并行缩放，提升模型推理能力。实验表明其在BIRD基准上达81.67%执行准确率，位居榜首，推动接近人类水平。**

- **链接: [https://arxiv.org/pdf/2509.24403v5](https://arxiv.org/pdf/2509.24403v5)**

> **作者:** Pengfei Wang; Baolin Sun; Xuemei Dong; Yaxun Dai; Hongwei Yuan; Mengdie Chu; Yingqi Gao; Xiang Qi; Peng Zhang; Ying Yan
>
> **摘要:** State-of-the-art (SOTA) Text-to-SQL methods still lag significantly behind human experts on challenging benchmarks like BIRD. Current approaches that explore test-time scaling lack an orchestrated strategy and neglect the model's internal reasoning process. To bridge this gap, we introduce Agentar-Scale-SQL, a novel framework leveraging scalable computation to improve performance. Agentar-Scale-SQL implements an Orchestrated Test-Time Scaling strategy that synergistically combines three distinct perspectives: i) Internal Scaling via RL-enhanced Intrinsic Reasoning, ii) Sequential Scaling through Iterative Refinement, and iii) Parallel Scaling using Diverse Synthesis and Tournament Selection. Agentar-Scale-SQL is a general-purpose framework designed for easy adaptation to new databases and more powerful language models. Extensive experiments show that Agentar-Scale-SQL achieves SOTA performance on the BIRD benchmark, reaching 81.67% execution accuracy on the test set and ranking first on the official leaderboard, demonstrating an effective path toward human-level performance.
>
---
#### [replaced 003] AutoHall: Automated Factuality Hallucination Dataset Generation for Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对大语言模型（LLM）的幻觉问题，提出AutoHall方法，自动构建模型特定的幻觉数据集。通过利用现有事实核查数据，实现高效、低成本的幻觉样本生成，并设计零资源黑盒检测方法，基于自相矛盾提升检测性能，揭示幻觉成因。**

- **链接: [https://arxiv.org/pdf/2310.00259v3](https://arxiv.org/pdf/2310.00259v3)**

> **作者:** Zouying Cao; Yifei Yang; XiaoJing Li; Hai Zhao
>
> **备注:** Accepted by IEEE Transactions on Audio, Speech, and Language Processing (TASLP)
>
> **摘要:** Large language models (LLMs) have gained broad applications across various domains but still struggle with hallucinations. Currently, hallucinations occur frequently in the generation of factual content and pose a great challenge to trustworthy LLMs. However, hallucination detection is hindered by the laborious and expensive manual annotation of hallucinatory content. Meanwhile, as different LLMs exhibit distinct types and rates of hallucination, the collection of hallucination datasets is inherently model-specific, which also increases the cost. To address this issue, this paper proposes a method called $\textbf{AutoHall}$ for $\underline{Auto}$matically constructing model-specific $\underline{Hall}$ucination datasets based on existing fact-checking datasets. The empirical results reveal variations in hallucination proportions and types among different models. Moreover, we introduce a zero-resource and black-box hallucination detection method based on self-contradiction to recognize the hallucination in our constructed dataset, achieving superior detection performance compared to baselines. Further analysis on our dataset provides insight into factors that may contribute to LLM hallucinations. Our codes and datasets are publicly available at https://github.com/zouyingcao/AutoHall.
>
---
#### [replaced 004] Exploring the Human-LLM Synergy in Advancing Theory-driven Qualitative Analysis
- **分类: cs.HC; cs.CL; cs.CY**

- **简介: 该论文研究人与大语言模型（LLM）协同进行理论驱动的定性分析任务。针对现有方法难以挖掘超越初始理论的新洞察问题，提出CHALET框架，通过迭代编码、分歧分析与概念化，实现人-LLM协作发现心理疾病污名在认知、情感、行为维度的隐含主题，推动定性分析创新。**

- **链接: [https://arxiv.org/pdf/2405.05758v2](https://arxiv.org/pdf/2405.05758v2)**

> **作者:** Han Meng; Yitian Yang; Wayne Fu; Jungup Lee; Yunan Li; Yi-Chieh Lee
>
> **备注:** 51 pages, 6 figures, accepted by ACM Trans. Comput.-Hum. Interact (TOCHI)
>
> **摘要:** Qualitative coding is a demanding yet crucial research method in the field of Human-Computer Interaction (HCI). While recent studies have shown the capability of large language models (LLMs) to perform qualitative coding within theoretical frameworks, their potential for collaborative human-LLM discovery and generation of new insights beyond initial theory remains underexplored. To bridge this gap, we proposed CHALET, a novel approach that harnesses the power of human-LLM partnership to advance theory-driven qualitative analysis by facilitating iterative coding, disagreement analysis, and conceptualization of qualitative data. We demonstrated CHALET's utility by applying it to the qualitative analysis of conversations related to mental-illness stigma, using the attribution model as the theoretical framework. Results highlighted the unique contribution of human-LLM collaboration in uncovering latent themes of stigma across the cognitive, emotional, and behavioral dimensions. We discuss the methodological implications of the human-LLM collaborative approach to theory-based qualitative analysis for the HCI community and beyond.
>
---
#### [replaced 005] Local Hybrid Retrieval-Augmented Document QA
- **分类: cs.CL**

- **简介: 该论文针对企业文档问答中隐私与性能的矛盾，提出本地化混合检索增强的问答系统。通过结合语义理解与关键词匹配，在离线环境下利用消费级硬件实现高精度问答，解决了敏感数据无法上云的问题，实现了隐私保护与良好性能的统一。**

- **链接: [https://arxiv.org/pdf/2511.10297v2](https://arxiv.org/pdf/2511.10297v2)**

> **作者:** Paolo Astrino
>
> **备注:** 10 pages, 5 figures, 3 tables; conference-style (ACL format); fully local RAG system
>
> **摘要:** Organizations handling sensitive documents face a critical dilemma: adopt cloud-based AI systems that offer powerful question-answering capabilities but compromise data privacy, or maintain local processing that ensures security but delivers poor accuracy. We present a question-answering system that resolves this trade-off by combining semantic understanding with keyword precision, operating entirely on local infrastructure without internet access. Our approach demonstrates that organizations can achieve competitive accuracy on complex queries across legal, scientific, and conversational documents while keeping all data on their machines. By balancing two complementary retrieval strategies and using consumer-grade hardware acceleration, the system delivers reliable answers with minimal errors, letting banks, hospitals, and law firms adopt conversational document AI without transmitting proprietary information to external providers. This work establishes that privacy and performance need not be mutually exclusive in enterprise AI deployment.
>
---
#### [replaced 006] Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对数学推理中链式思维（CoT）数据存在的“思维跳跃”问题，提出CoT Thought Leap Bridge任务，通过构建ScaleQM+数据集并训练CoT-Bridge模型自动补全缺失推理步骤。实验表明，该方法显著提升模型性能与泛化能力，可作为通用模块增强现有优化方法。**

- **链接: [https://arxiv.org/pdf/2505.14684v3](https://arxiv.org/pdf/2505.14684v3)**

> **作者:** Haolei Xu; Yuchen Yan; Yongliang Shen; Wenqi Zhang; Guiyang Hou; Shengpei Jiang; Kaitao Song; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **备注:** Accepted to NeurIPS 2025. Camera ready version. Code: https://github.com/ZJU-REAL/Mind-the-Gap Project: https://zju-real.github.io/CoT-Bridge/
>
> **摘要:** Large language models (LLMs) have achieved remarkable progress on mathematical tasks through Chain-of-Thought (CoT) reasoning. However, existing mathematical CoT datasets often suffer from Thought Leaps due to experts omitting intermediate steps, which negatively impacts model learning and generalization. We propose the CoT Thought Leap Bridge Task, which aims to automatically detect leaps and generate missing intermediate reasoning steps to restore the completeness and coherence of CoT. To facilitate this, we constructed a specialized training dataset called ScaleQM+, based on the structured ScaleQuestMath dataset, and trained CoT-Bridge to bridge thought leaps. Through comprehensive experiments on mathematical reasoning benchmarks, we demonstrate that models fine-tuned on bridged datasets consistently outperform those trained on original datasets, with improvements of up to +5.87% on NuminaMath. Our approach effectively enhances distilled data (+3.02%) and provides better starting points for reinforcement learning (+3.1%), functioning as a plug-and-play module compatible with existing optimization techniques. Furthermore, CoT-Bridge demonstrate improved generalization to out-of-domain logical reasoning tasks, confirming that enhancing reasoning completeness yields broadly applicable benefits.
>
---
#### [replaced 007] WritingBench: A Comprehensive Benchmark for Generative Writing
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出WritingBench，一个面向生成式写作的综合性基准，涵盖6大领域100子领域。针对现有评估体系覆盖不足、缺乏动态评价标准的问题，设计了基于查询的评估框架与细调评鉴模型，支持风格、格式、长度等多维度评分，并实现7B模型超越GPT-4o写作性能。开源工具促进语言模型写作能力发展。**

- **链接: [https://arxiv.org/pdf/2503.05244v4](https://arxiv.org/pdf/2503.05244v4)**

> **作者:** Yuning Wu; Jiahao Mei; Ming Yan; Chenliang Li; Shaopeng Lai; Yuran Ren; Zijia Wang; Ji Zhang; Mengyue Wu; Qin Jin; Fei Huang
>
> **摘要:** Recent advancements in large language models (LLMs) have significantly enhanced text generation capabilities, yet evaluating their performance in generative writing remains a challenge. Existing benchmarks primarily focus on generic text generation or limited in writing tasks, failing to capture the diverse requirements of high-quality written contents across various domains. To bridge this gap, we present WritingBench, a comprehensive benchmark designed to evaluate LLMs across 6 core writing domains and 100 subdomains. We further propose a query-dependent evaluation framework that empowers LLMs to dynamically generate instance-specific assessment criteria. This framework is complemented by a fine-tuned critic model for criteria-aware scoring, enabling evaluations in style, format and length. The framework's validity is further demonstrated by its data curation capability, which enables a 7B-parameter model to outperform the performance of GPT-4o in writing. We open-source the benchmark, along with evaluation tools and modular framework components, to advance the development of LLMs in writing.
>
---
#### [replaced 008] G$^2$VLM: Geometry Grounded Vision Language Model with Unified 3D Reconstruction and Spatial Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出G²VLM，一种基于几何引导的视觉语言模型，旨在解决视觉语言模型在空间理解与推理上的不足。通过融合多视角图像与视频数据，实现统一的3D重建与空间推理，提升模型对三维空间的感知能力，为后续3D场景编辑等应用提供基础。**

- **链接: [https://arxiv.org/pdf/2511.21688v2](https://arxiv.org/pdf/2511.21688v2)**

> **作者:** Wenbo Hu; Jingli Lin; Yilin Long; Yunlong Ran; Lihan Jiang; Yifan Wang; Chenming Zhu; Runsen Xu; Tai Wang; Jiangmiao Pang
>
> **备注:** code are released at https://github.com/InternRobotics/G2VLM
>
> **摘要:** Vision-Language Models (VLMs) still lack robustness in spatial intelligence, demonstrating poor performance on spatial understanding and reasoning tasks. We attribute this gap to the absence of a visual geometry learning process capable of reconstructing 3D space from 2D images. We present G$^2$VLM, a geometry grounded vision-language model that bridges two fundamental aspects of spatial intelligence: spatial 3D reconstruction and spatial understanding. G$^2$VLM natively leverages learned 3D visual geometry features to directly predict 3D attributes and enhance spatial reasoning tasks via in-context learning and interleaved reasoning. Our unified design is highly scalable for spatial understanding: it trains on abundant multi-view image and video data, while simultaneously leveraging the benefits of 3D visual priors that are typically only derived from hard-to-collect annotations. Experimental results demonstrate G$^2$VLM is proficient in both tasks, achieving comparable results to state-of-the-art feed-forward 3D reconstruction models and achieving better or competitive results across spatial understanding and reasoning tasks. By unifying a semantically strong VLM with low-level 3D vision tasks, we hope G$^2$VLM can serve as a strong baseline for the community and unlock more future applications, such as 3D scene editing.
>
---
#### [replaced 009] Strong Memory, Weak Control: An Empirical Study of Executive Functioning in LLMs
- **分类: cs.CL**

- **简介: 该论文属于认知评估类任务，旨在探究大语言模型（LLMs）的执行功能。通过经典工作记忆任务测试，发现LLMs工作记忆容量超人类，但未提升问题解决表现，揭示其注意力控制与认知灵活性缺陷。研究指出当前推理模型难以弥补这些短板。**

- **链接: [https://arxiv.org/pdf/2504.02789v2](https://arxiv.org/pdf/2504.02789v2)**

> **作者:** Karin de Langis; Jong Inn Park; Bin Hu; Khanh Chi Le; Andreas Schramm; Michael C. Mensink; Andrew Elfenbein; Dongyeop Kang
>
> **摘要:** Working memory, or the ability to hold and manipulate information in the mind, is a critical component of human intelligence and executive functioning. It is correlated with performance on various cognitive tasks, including measures of fluid intelligence, which encompasses reasoning and problem solving. We use a comprehensive set of classic working memory tasks to estimate the working memory capacity of large language models (LLMs). We find that in most cases, LLMs exceed normative human scores. However, we do not find that the increased capacity of working memory is associated with higher performance on other executive functioning tasks or problem solving benchmarks. These results suggest that LLMs may have deficits in attentional control and cognitive flexibility, which result in difficulties with inhibiting automatic responses and adapting to shifting information. Our findings suggest that current reasoning models have mixed results in compensating for these deficits.
>
---
#### [replaced 010] A Trio Neural Model for Dynamic Entity Relatedness Ranking
- **分类: cs.IR; cs.CL; cs.LG; stat.ML**

- **简介: 该论文针对动态实体相关性排序任务，解决静态模型无法捕捉实体关系随时间变化的问题。提出基于集体注意力监督的三元神经网络模型，联合学习动态实体表示，在大规模数据上优于基线方法。**

- **链接: [https://arxiv.org/pdf/1808.08316v5](https://arxiv.org/pdf/1808.08316v5)**

> **作者:** Tu Nguyen; Tuan Tran; Wolfgang Nejdl
>
> **备注:** In Proceedings of CoNLL 2018
>
> **摘要:** Measuring entity relatedness is a fundamental task for many natural language processing and information retrieval applications. Prior work often studies entity relatedness in static settings and an unsupervised manner. However, entities in real-world are often involved in many different relationships, consequently entity-relations are very dynamic over time. In this work, we propose a neural networkbased approach for dynamic entity relatedness, leveraging the collective attention as supervision. Our model is capable of learning rich and different entity representations in a joint framework. Through extensive experiments on large-scale datasets, we demonstrate that our method achieves better results than competitive baselines.
>
---
#### [replaced 011] Deep Improvement Supervision
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究小规模循环模型（TRMs）在复杂推理任务中的效率优化。针对TRMs训练效率低、依赖停顿机制的问题，提出基于隐式策略改进的新型训练方案，通过为每轮迭代提供目标，显著减少前向传播次数（18倍），消除停顿机制，仅用0.8M参数即在ARC-1上达到24%准确率，优于多数LLMs。**

- **链接: [https://arxiv.org/pdf/2511.16886v2](https://arxiv.org/pdf/2511.16886v2)**

> **作者:** Arip Asadulaev; Rayan Banerjee; Fakhri Karray; Martin Takac
>
> **摘要:** Recently, it was shown that small, looped architectures, such as Tiny Recursive Models (TRMs), can outperform Large Language Models (LLMs) on complex reasoning tasks, including the Abstraction and Reasoning Corpus (ARC). In this work, we investigate a core question: how can we further improve the efficiency of these methods with minimal changes? To address this, we frame the latent reasoning of TRMs as a form of classifier-free guidance and implicit policy improvement algorithm. Building on these insights, we propose a novel training scheme that provides a target for each loop during training. We demonstrate that our approach significantly enhances training efficiency. Our method reduces the total number of forward passes by 18x and eliminates halting mechanisms, while maintaining quality comparable to standard TRMs. Notably, we achieve 24% accuracy on ARC-1 with only 0.8M parameters, outperforming most LLMs.
>
---
#### [replaced 012] Fine-grained and Explainable Factuality Evaluation for Multimodal Summarization
- **分类: cs.CL**

- **简介: 该论文针对多模态摘要中的事实性问题，提出两种细粒度可解释的评估框架（FALLACIOUS），分别适用于有无参考文本的场景，无需真实标签即可评估摘要真实性，有效提升评估的准确性和适用性。**

- **链接: [https://arxiv.org/pdf/2402.11414v5](https://arxiv.org/pdf/2402.11414v5)**

> **作者:** Yue Zhang; Jingxuan Zuo; Ke Su; Liqiang Jing
>
> **备注:** project link: https://github.com/for4WARD/FaithfulnessEvaluation
>
> **摘要:** Multimodal summarization aims to generate a concise summary based on the input text and image. However, the existing methods potentially suffer from unfactual output. To evaluate the factuality of multimodal summarization models, we propose two fine-grained and explainable evaluation frameworks (FALLACIOUS) for different application scenarios, i.e. reference-based factuality evaluation framework and reference-free factuality evaluation framework. Notably, the reference-free factuality evaluation framework doesn't need ground truth and hence it has a wider application scenario. To evaluate the effectiveness of the proposed frameworks, we compute the correlation between our frameworks and the other metrics. The experimental results show the effectiveness of our proposed method. We will release our code and dataset via github.
>
---
#### [replaced 013] AppSelectBench: Application-Level Tool Selection Benchmark
- **分类: cs.CL**

- **简介: 该论文提出AppSelectBench，一个用于评估计算机使用代理（CUAs）应用级工具选择能力的基准。针对现有基准仅关注细粒度API选择、缺乏对跨应用推理评估的问题，构建了包含超十万条真实用户任务的大规模基准，涵盖100个常用桌面应用，支持多种评估设置，揭示了当前模型在应用选择上的系统性不足。**

- **链接: [https://arxiv.org/pdf/2511.19957v2](https://arxiv.org/pdf/2511.19957v2)**

> **作者:** Tianyi Chen; Michael Solodko; Sen Wang; Jongwoo Ko; Junheng Hao; Colby Banbury; Sara Abdali; Saeed Amizadeh; Qing Xiao; Yinheng Li; Tianyu Ding; Kamran Ghasedi Dizaji; Suzhen Zheng; Hao Fan; Justin Wagle; Pashmina Cameron; Kazuhito Koishida
>
> **摘要:** Computer Using Agents (CUAs) are increasingly equipped with external tools, enabling them to perform complex and realistic tasks. For CUAs to operate effectively, application selection, which refers to deciding which application to use before invoking fine-grained tools such as APIs, is a fundamental capability. It determines whether the agent initializes the correct environment, avoids orchestration confusion, and efficiently focuses on relevant context. However, existing benchmarks primarily assess fine-grained API selection, offering limited insight into whether models can reason across and choose between different applications. To fill this gap, we introduce AppSelectBench, a comprehensive benchmark for evaluating application selection in CUAs. AppSelectBench contains a novel user task generation pipeline that produces realistic, diverse, and semantically grounded user intents at scale, together with unified evaluation protocols covering random, heuristic, zero-shot, few-shot, and retrieval-augmented-settings. AppSelectBench covers one hundred widely used desktop applications and includes more than one hundred thousand realistic, diverse, and semantically grounded user tasks. Extensive experiments across both closed-source and open-source large language models reveal systematic strengths and weaknesses in inter-application reasoning, showing that even the most capable models still struggle to make consistent application choices. Together, these results establish AppSelectBench as a foundation for studying and advancing application level reasoning, an essential yet underexplored capability of intelligent CUAs. The source is available at https://microsoft.github.io/appselectbench/.
>
---
#### [replaced 014] From Perception to Reasoning: Deep Thinking Empowers Multimodal Large Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文聚焦多模态大模型的复杂推理能力提升，针对现有模型推理不透明、泛化性差的问题，系统综述了多模态思维链（MCoT）方法。从技术演进与任务需求出发，分析其原理、训练与推理策略，总结评估体系与应用场景，并展望未来挑战与方向。**

- **链接: [https://arxiv.org/pdf/2511.12861v4](https://arxiv.org/pdf/2511.12861v4)**

> **作者:** Wenxin Zhu; Andong Chen; Yuchen Song; Kehai Chen; Conghui Zhu; Ziyan Chen; Tiejun Zhao
>
> **备注:** Survey; 7 figures, 3 tables, 44 pages
>
> **摘要:** With the remarkable success of Multimodal Large Language Models (MLLMs) in perception tasks, enhancing their complex reasoning capabilities has emerged as a critical research focus. Existing models still suffer from challenges such as opaque reasoning paths and insufficient generalization ability. Chain-of-Thought (CoT) reasoning, which has demonstrated significant efficacy in language models by enhancing reasoning transparency and output interpretability, holds promise for improving model reasoning capabilities when extended to the multimodal domain. This paper provides a systematic review centered on "Multimodal Chain-of-Thought" (MCoT). First, it analyzes the background and theoretical motivations for its inception from the perspectives of technical evolution and task demands. Then, it introduces mainstream MCoT methods from three aspects: CoT paradigms, the post-training stage, and the inference stage, while also analyzing their underlying mechanisms. Furthermore, the paper summarizes existing evaluation benchmarks and metrics, and discusses the application scenarios of MCoT. Finally, it analyzes the challenges currently facing MCoT and provides an outlook on its future research directions.
>
---
#### [replaced 015] Self-Guided Defense: Adaptive Safety Alignment for Reasoning Models via Synthesized Guidelines
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对推理模型在对抗性越狱提示下的安全问题，提出自引导防御框架SGASA。通过生成安全准则并进行微调，实现模型对有害提示的自适应防护，提升安全性同时减少对正常请求的误拒。**

- **链接: [https://arxiv.org/pdf/2511.21214v2](https://arxiv.org/pdf/2511.21214v2)**

> **作者:** Yuhang Wang; Yanxu Zhu; Dongyuan Lu; Jitao Sang
>
> **摘要:** Reasoning models have demonstrated remarkable capabilities in complex reasoning tasks. However, ensuring their safety against adversarial jailbreak prompts remains a critical challenge. Due to the covert and deceptive nature of such prompts, they can often evade built-in safety mechanisms and lead to the generation of harmful content. This underscores the need for an adaptive safety alignment approach that enables models to autonomously reinforce their defenses in response to adversarial inputs. This paper introduces the Synthesized Guideline-based Adaptive Safety Alignment (SGASA) framework, which internalizes model-generated safety guidelines to strengthen models' ability to enhance robustness against harmful adversarial prompts while minimizing unnecessary refusals of benign requests. SGASA consists of two key stages: Data Pre-synthesis, which generates safety guidelines and augmented prompts; and Alignment Fine-tuning, which leverages Supervised Fine-tuning (SFT) and Direct Preference Optimization (DPO) to embed these guidelines into the model. Extensive experiments across multiple datasets demonstrate that SGASA significantly improves model safety, validating its adaptive and scalable effectiveness.
>
---
#### [replaced 016] Toward Equitable Access: Leveraging Crowdsourced Reviews to Investigate Public Perceptions of Health Resource Accessibility
- **分类: cs.CL**

- **简介: 该论文旨在解决公共卫生危机中健康资源可及性差异监测难题。通过分析2018–2021年谷歌地图众包评论，结合DeBERTa模型构建公众感知指数，利用PLS回归识别社会经济与人口因素影响，揭示疫情期感知差距达峰值并部分恢复，验证了实时健康公平监测的新方法。**

- **链接: [https://arxiv.org/pdf/2502.10641v2](https://arxiv.org/pdf/2502.10641v2)**

> **作者:** Zhaoqian Xue; Guanhong Liu; Chong Zhang; Kai Wei; Qingcheng Zeng; Songhua Hu; Wenyue Hua; Lizhou Fan; Yongfeng Zhang; Lingyao Li
>
> **摘要:** Monitoring health resource disparities during public health crises is critical, yet traditional methods, like surveys, lack the requisite speed and spatial granularity. This study introduces a novel framework that leverages: 1) crowdsourced Google Maps reviews (2018-2021) and 2) advanced NLP (DeBERTa) to create a high-resolution, spatial-temporal index of public perception of health resource accessibility in the United States. We then employ Partial Least Squares (PLS) regression to link this perception index to a range of socioeconomic and demographic drivers. Our results quantify significant spatial-temporal shifts in perceived access, confirming that disparities peaked during the COVID-19 crisis and only partially recovered post-peak. We identify political affiliation, racial composition, and educational attainment as primary determinants of these perceptions. This study validates a scalable method for real-time health equity monitoring and provides actionable evidence for interventions to build a more resilient healthcare infrastructure.
>
---
#### [replaced 017] STAR-Bench: Probing Deep Spatio-Temporal Reasoning as Audio 4D Intelligence
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出STAR-Bench，用于评估音频4D智能——即对时空中声音动态的细粒度感知与推理能力。针对现有音频基准依赖文本描述、忽视复杂听觉推理的问题，构建包含基础感知与整体时空推理的评测体系，通过合成数据与人工筛选确保质量，揭示当前模型在时空感知上的显著短板。**

- **链接: [https://arxiv.org/pdf/2510.24693v2](https://arxiv.org/pdf/2510.24693v2)**

> **作者:** Zihan Liu; Zhikang Niu; Qiuyang Xiao; Zhisheng Zheng; Ruoqi Yuan; Yuhang Zang; Yuhang Cao; Xiaoyi Dong; Jianze Liang; Xie Chen; Leilei Sun; Dahua Lin; Jiaqi Wang
>
> **备注:** Homepage: https://internlm.github.io/StarBench/
>
> **摘要:** Despite rapid progress in Multi-modal Large Language Models and Large Audio-Language Models, existing audio benchmarks largely test semantics that can be recovered from text captions, masking deficits in fine-grained perceptual reasoning. We formalize audio 4D intelligence that is defined as reasoning over sound dynamics in time and 3D space, and introduce STAR-Bench to measure it. STAR-Bench combines a Foundational Acoustic Perception setting (six attributes under absolute and relative regimes) with a Holistic Spatio-Temporal Reasoning setting that includes segment reordering for continuous and discrete processes and spatial tasks spanning static localization, multi-source relations, and dynamic trajectories. Our data curation pipeline uses two methods to ensure high-quality samples. For foundational tasks, we use procedurally synthesized and physics-simulated audio. For holistic data, we follow a four-stage process that includes human annotation and final selection based on human performance. Unlike prior benchmarks where caption-only answering reduces accuracy slightly, STAR-Bench induces far larger drops (-31.5\% temporal, -35.2\% spatial), evidencing its focus on linguistically hard-to-describe cues. Evaluating 19 models reveals substantial gaps compared with humans and a capability hierarchy: closed-source models are bottlenecked by fine-grained perception, while open-source models lag across perception, knowledge, and reasoning. Our STAR-Bench provides critical insights and a clear path forward for developing future models with a more robust understanding of the physical world.
>
---
#### [replaced 018] Harvesting Textual and Contrastive Data from the HAL Publication Repository
- **分类: cs.DL; cs.CL**

- **简介: 该论文聚焦作者归属任务，旨在区分语言风格与主题干扰。提出HALvest及HALvest-Contrastive数据集，通过利用同一作者论文间的自然主题变化，提取非词汇性风格特征。实验表明，基于去主题化数据训练的神经模型表现更优，证明其能有效捕捉深层风格模式。**

- **链接: [https://arxiv.org/pdf/2407.20595v3](https://arxiv.org/pdf/2407.20595v3)**

> **作者:** Francis Kulumba; Wissam Antoun; Guillaume Vimont; Laurent Romary
>
> **备注:** New dataset version with only the contrastive learning data
>
> **摘要:** Authorship attribution in natural language processing traditionally struggles to distinguish genuine stylistic signals from topical confounds. While contrastive learning approaches have addressed this by maximizing semantic overlap between positive pairs, creating large-scale datasets under strict topic constraints remains challenging. We introduce HALvest, a 17-billion-token multilingual corpus harvested from 778k open-access academic papers, and HALvest-Contrastive, a derived dataset designed to isolate stylometric signals through controlled topic variation. Unlike prior work that minimizes lexical overlap, we exploit natural topic drift between papers by the same author, treating residual lexical patterns as authorial fingerprints rather than noise. Comparing lexical baselines (BM25) against neural models trained on unrestricted (topic-rich) versus base (topic-decoupled) triplets, we demonstrate that models trained exclusively on topic-decoupled data achieve superior performance across all test conditions, outperforming both retrieval baselines and models exposed to topic-rich training data. Our analysis reveals that while lexical signals provide substantial performance gains for keyword-driven methods, neural architectures learn robust stylometric representations that plateau with moderate context length, suggesting they capture distributional style beyond surface-level tokens. Both datasets and code are publicly available.
>
---
#### [replaced 019] Prompt-R1: Collaborative Automatic Prompting Framework via End-to-end Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出Prompt-R1，一个基于强化学习的协作式自动提示框架，旨在解决用户难以生成有效提示以充分发挥大模型能力的问题。通过小模型与大模型多轮协作生成高质量提示，优化推理准确性与生成质量，显著提升复杂任务表现。**

- **链接: [https://arxiv.org/pdf/2511.01016v3](https://arxiv.org/pdf/2511.01016v3)**

> **作者:** Wenjin Liu; Haoran Luo; Xueyuan Lin; Haoming Liu; Tiesunlong Shen; Jiapu Wang; Rui Mao; Erik Cambria
>
> **摘要:** Recently, advanced large language models (LLMs) have emerged at an increasingly rapid pace. However, when faced with complex problems, most users are often unable to provide accurate and effective prompts to interact with LLMs, thus limiting the performance of LLMs. To address this challenge, we propose Prompt-R1, an end-to-end reinforcement learning framework that uses a small-scale LLM to collaborate with large-scale LLMs, replacing user interaction to solve problems better. This collaboration is cast as a multi-turn prompt interaction, where the small-scale LLM thinks and generates prompts, and the large-scale LLM performs complex reasoning. A dual-constrained reward is designed to optimize for correctness, generation quality, and reasoning accuracy. Prompt-R1 provides a plug-and-play framework that supports both inference and training with various large-scale LLMs. Experiments on multiple public datasets show that Prompt-R1 significantly outperforms baseline models across tasks. Our code is publicly available at https://github.com/QwenQKing/Prompt-R1.
>
---
#### [replaced 020] KSHSeek: Data-Driven Approaches to Mitigating and Detecting Knowledge-Shortcut Hallucinations in Generative Models
- **分类: cs.CL**

- **简介: 该论文针对生成模型中的知识捷径幻觉问题，提出KSHSeek方法。通过数据预处理阶段的高相似度剪枝减少数据中的虚假关联，并设计专用检测方法评估效果。实验表明该方法有效降低幻觉，提升模型可靠性，适用于问答等文本生成任务。**

- **链接: [https://arxiv.org/pdf/2503.19482v2](https://arxiv.org/pdf/2503.19482v2)**

> **作者:** Zhongxin Liu; Zhiwei Wang; Jun Niu; Ying Li; Hongyu Sun; Meng Xu; He Wang; Gaofei Wu; Yuqing Zhang
>
> **备注:** 16 pages, 34 figures
>
> **摘要:** The emergence of large language models (LLMs) has significantly advanced the development of natural language processing (NLP), especially in text generation tasks like question answering. However, model hallucinations remain a major challenge in natural language generation (NLG) tasks due to their complex causes. We systematically expand on the causes of factual hallucinations from the perspective of knowledge shortcuts, analyzing hallucinations arising from correct and defect-free data and demonstrating that knowledge-shortcut hallucinations are prevalent in generative models. To mitigate this issue, we propose a high similarity pruning algorithm at the data preprocessing level to reduce spurious correlations in the data. Additionally, we design a specific detection method for knowledge-shortcut hallucinations to evaluate the effectiveness of our mitigation strategy. Experimental results show that our approach effectively reduces knowledge-shortcut hallucinations, particularly in fine-tuning tasks, without negatively impacting model performance in question answering. This work introduces a new paradigm for mitigating specific hallucination issues in generative models, enhancing their robustness and reliability in real-world applications.
>
---
#### [replaced 021] Do Large Language Models Think Like the Brain? Sentence-Level Evidence from fMRI and Hierarchical Embeddings
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文探究大语言模型（LLM）与人脑在句子级语言处理上的相似性，属于认知神经科学与AI交叉任务。针对“LLM是否像大脑一样运作”这一核心问题，研究通过对比14个LLM的分层嵌入与人脑fMRI数据，构建句级神经预测模型，发现模型性能提升促使表征架构向脑似层次结构演化，尤其在高层语义抽象层面表现更强对应。**

- **链接: [https://arxiv.org/pdf/2505.22563v2](https://arxiv.org/pdf/2505.22563v2)**

> **作者:** Yu Lei; Xingyang Ge; Yi Zhang; Yiming Yang; Bolei Ma
>
> **摘要:** Understanding whether large language models (LLMs) and the human brain converge on similar computational principles remains a fundamental and important question in cognitive neuroscience and AI. Do the brain-like patterns observed in LLMs emerge simply from scaling, or do they reflect deeper alignment with the architecture of human language processing? This study focuses on the sentence-level neural mechanisms of language models, systematically investigating how hierarchical representations in LLMs align with the dynamic neural responses during human sentence comprehension. By comparing hierarchical embeddings from 14 publicly available LLMs with fMRI data collected from participants, who were exposed to a naturalistic narrative story, we constructed sentence-level neural prediction models to precisely identify the model layers most significantly correlated with brain region activations. Results show that improvements in model performance drive the evolution of representational architectures toward brain-like hierarchies, particularly achieving stronger functional and anatomical correspondence at higher semantic abstraction levels.
>
---
#### [replaced 022] Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对智能体系统中组件选择效率低的问题，提出基于背包问题的自动化组合框架。通过动态评估性能、成本与兼容性，实现最优组件选型，在多场景下显著提升成功率并降低资源消耗。**

- **链接: [https://arxiv.org/pdf/2510.16499v2](https://arxiv.org/pdf/2510.16499v2)**

> **作者:** Michelle Yuan; Khushbu Pahwa; Shuaichen Chang; Mustafa Kaba; Jiarong Jiang; Xiaofei Ma; Yi Zhang; Monica Sunkara
>
> **备注:** Accepted to NeurIPS 2025 Conference
>
> **摘要:** Designing effective agentic systems requires the seamless composition and integration of agents, tools, and models within dynamic and uncertain environments. Most existing methods rely on static, semantic retrieval approaches for tool or agent discovery. However, effective reuse and composition of existing components remain challenging due to incomplete capability descriptions and the limitations of retrieval methods. Component selection suffers because the decisions are not based on capability, cost, and real-time utility. To address these challenges, we introduce a structured, automated framework for agentic system composition that is inspired by the knapsack problem. Our framework enables a composer agent to systematically identify, select, and assemble an optimal set of agentic components by jointly considering performance, budget constraints, and compatibility. By dynamically testing candidate components and modeling their utility in real-time, our approach streamlines the assembly of agentic systems and facilitates scalable reuse of resources. Empirical evaluation with Claude 3.5 Sonnet across five benchmarking datasets shows that our online-knapsack-based composer consistently lies on the Pareto frontier, achieving higher success rates at significantly lower component costs compared to our baselines. In the single-agent setup, the online knapsack composer shows a success rate improvement of up to 31.6% in comparison to the retrieval baselines. In multi-agent systems, the online knapsack composer increases success rate from 37% to 87% when agents are selected from an agent inventory of 100+ agents. The substantial performance gap confirms the robust adaptability of our method across diverse domains and budget constraints.
>
---
#### [replaced 023] ReGATE: Learning Faster and Better with Fewer Tokens in MLLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态大模型（MLLM）训练中计算成本高的问题，提出ReGATE方法。通过教师-学生框架与动态令牌剪枝，实现训练加速。在不改变模型结构前提下，减少41%以上令牌使用，显著提升训练效率，且性能优于标准训练。**

- **链接: [https://arxiv.org/pdf/2507.21420v2](https://arxiv.org/pdf/2507.21420v2)**

> **作者:** Chaoyu Li; Yogesh Kulkarni; Pooyan Fazli
>
> **摘要:** The computational cost of training multimodal large language models (MLLMs) grows rapidly with the number of processed tokens. Existing efficiency methods mainly target inference via token reduction or merging, offering limited benefits during training. We introduce ReGATE (Reference-Guided Adaptive Token Elision), an adaptive token pruning method for accelerating MLLM training. ReGATE adopts a teacher-student framework, in which a frozen teacher LLM provides per-token guidance losses that are fused with an exponential moving average of the student's difficulty estimates. This adaptive scoring mechanism dynamically selects informative tokens while skipping redundant ones in the forward pass, substantially reducing computation without altering the model architecture. Across three representative MLLMs, ReGATE matches the peak accuracy of standard training on MVBench up to 2$\times$ faster, using only 38% of the tokens. With extended training, it even surpasses the baseline across multiple multimodal benchmarks, cutting total token usage by over 41%. Code and models will be released publicly.
>
---
#### [replaced 024] Odin: Oriented Dual-module Integration for Text-rich Network Representation Learning
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对文本属性图的表示学习任务，解决现有GNN与Transformer模型在结构与文本融合上的局限。提出Odin架构，通过定向双模块机制在特定深度注入图结构，实现层次化结构抽象，避免过平滑，提升表达能力。还提出轻量版Light Odin，高效且性能优异。**

- **链接: [https://arxiv.org/pdf/2511.21416v2](https://arxiv.org/pdf/2511.21416v2)**

> **作者:** Kaifeng Hong; Yinglong Zhang; Xiaoying Hong; Xuewen Xia; Xing Xu
>
> **备注:** 32 pages, 2 figures
>
> **摘要:** Text-attributed graphs require models to effectively combine strong textual understanding with structurally informed reasoning. Existing approaches either rely on GNNs--limited by over-smoothing and hop-dependent diffusion--or employ Transformers that overlook graph topology and treat nodes as isolated sequences. We propose Odin (Oriented Dual-module INtegration), a new architecture that injects graph structure into Transformers at selected depths through an oriented dual-module mechanism. Unlike message-passing GNNs, Odin does not rely on multi-hop diffusion; instead, multi-hop structures are integrated at specific Transformer layers, yielding low-, mid-, and high-level structural abstraction aligned with the model's semantic hierarchy. Because aggregation operates on the global [CLS] representation, Odin fundamentally avoids over-smoothing and decouples structural abstraction from neighborhood size or graph topology. We further establish that Odin's expressive power strictly contains that of both pure Transformers and GNNs. To make the design efficient in large-scale or low-resource settings, we introduce Light Odin, a lightweight variant that preserves the same layer-aligned structural abstraction for faster training and inference. Experiments on multiple text-rich graph benchmarks show that Odin achieves state-of-the-art accuracy, while Light Odin delivers competitive performance with significantly reduced computational cost. Together, Odin and Light Odin form a unified, hop-free framework for principled structure-text integration. The source code of this model has been released at https://github.com/hongkaifeng/Odin.
>
---
#### [replaced 025] KurdSTS: The Kurdish Semantic Textual Similarity
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦低资源语言 Kurdish 的语义文本相似度任务，提出首个 Kurdish STS 数据集（10,000 句对），涵盖正式与非正式语体。通过基准测试多种模型，揭示了库尔德语形态复杂、拼写变异和代码混用带来的挑战，为低资源 NLP 研究提供了可复现的评估框架与起点。**

- **链接: [https://arxiv.org/pdf/2510.02336v2](https://arxiv.org/pdf/2510.02336v2)**

> **作者:** Abdulhady Abas Abdullah; Hadi Veisi; Hussein M. Al
>
> **摘要:** Semantic Textual Similarity (STS) measures the degree of meaning overlap between two texts and underpins many NLP tasks. While extensive resources exist for high-resource languages, low-resource languages such as Kurdish remain underserved. We present, to our knowledge, the first Kurdish STS dataset: 10,000 sentence pairs spanning formal and informal registers, each annotated for similarity. We benchmark Sentence-BERT, multilingual BERT, and other strong baselines, obtaining competitive results while highlighting challenges arising from Kurdish morphology, orthographic variation, and code-mixing. The dataset and baselines establish a reproducible evaluation suite and provide a strong starting point for future research on Kurdish semantics and low-resource NLP.
>
---
#### [replaced 026] Privacy-Preserving Reasoning with Knowledge-Distilled Parametric Retrieval Augmented Generation
- **分类: cs.CL**

- **简介: 该论文针对隐私保护下的高效知识增强生成（RAG）问题，提出DistilledPRAG模型。通过知识蒸馏与参数生成，将文档转为LoRA参数，保留标准RAG结构并提升跨文档推理能力，实现高效率、高精度且泛化性强的隐私保护推理。**

- **链接: [https://arxiv.org/pdf/2509.01088v2](https://arxiv.org/pdf/2509.01088v2)**

> **作者:** Jinwen Chen; Hainan Zhang; Liang Pang; Yongxin Tong; Haibo Zhou; Yuan Zhan; Wei Lin; Zhiming Zheng
>
> **摘要:** The current RAG system requires uploading plaintext documents to the cloud, risking private data leakage. Parametric RAG (PRAG) encodes documents as LoRA parameters within LLMs, offering a possible way to reduce exposure of raw content. However, it still faces two issues: (1) PRAG demands synthesizing QA pairs and fine-tuning LLM for each individual document to create its corresponding LoRA, leading to unacceptable inference latency. (2) The performance of PRAG relies solely on synthetic QA data while lacking internal alignment with standard RAG, resulting in poor generalization on out-of-distribution(OOD) inputs. Therefore, achieving high-efficiency parameterization while maintaining RAG-level performance remains a critical challenge for privacy-preserving reasoning. In this paper, we propose DistilledPRAG, a generalizable knowledge-distilled parametric RAG model aligned with standard RAG in document structure and parameter activation. We first synthesize QA pairs from single and multi-documents to enhance cross-document reasoning. Then, we mask the plaintext documents with a special token and translate them to LoRA via a parameter generator, maintaining the standard RAG document structure. Finally, guided by synthetic QA data, we train the parameter generator to match standard RAG's hidden states and output logits, enabling RAG-style reasoning without original documents. Experiments on four QA datasets show that DistilledPRAG outperforms baselines in accuracy and generalizes well on OOD data.
>
---
#### [replaced 027] Beyond the Rubric: Cultural Misalignment in LLM Benchmarks for Sexual and Reproductive Health
- **分类: cs.CY; cs.CL**

- **简介: 该论文研究LLM在性与生殖健康领域对印度弱势群体的适用性。针对现有基准（如HealthBench）存在西方文化偏见的问题，作者通过实证发现其自动化评分与实际文化适切性不符。工作包括基于真实查询的评估、人工质性分析，揭示了法律、饮食、成本等维度的文化偏差，主张构建更包容的跨文化评估框架。**

- **链接: [https://arxiv.org/pdf/2511.17554v2](https://arxiv.org/pdf/2511.17554v2)**

> **作者:** Sumon Kanti Dey; Manvi S; Zeel Mehta; Meet Shah; Unnati Agrawal; Suhani Jalota; Azra Ismail
>
> **备注:** https://github.com/Sumon/healthbench-srh-eval/
>
> **摘要:** Large Language Models (LLMs) have been positioned as having the potential to expand access to health information in the Global South, yet their evaluation remains heavily dependent on benchmarks designed around Western norms. We present insights from a preliminary benchmarking exercise with a chatbot for sexual and reproductive health (SRH) for an underserved community in India. We evaluated using HealthBench, a benchmark for conversational health models by OpenAI. We extracted 637 SRH queries from the dataset and evaluated on the 330 single-turn conversations. Responses were evaluated using HealthBench's rubric-based automated grader, which rated responses consistently low. However, qualitative analysis by trained annotators and public health experts revealed that many responses were actually culturally appropriate and medically accurate. We highlight recurring issues, particularly a Western bias, such as for legal framing and norms (e.g., breastfeeding in public), diet assumptions (e.g., fish safe to eat during pregnancy), and costs (e.g., insurance models). Our findings demonstrate the limitations of current benchmarks in capturing the effectiveness of systems built for different cultural and healthcare contexts. We argue for the development of culturally adaptive evaluation frameworks that meet quality standards while recognizing needs of diverse populations.
>
---
#### [replaced 028] Event Stream-based Sign Language Translation: A High-Definition Benchmark Dataset and A Novel Baseline
- **分类: cs.CV; cs.AI; cs.CL; cs.NE**

- **简介: 该论文聚焦于基于事件流的手语翻译任务，针对传统视觉方法受光照、快速动作和隐私影响的问题，构建了高分辨率事件数据集Event-CSL，并提出EvSLT框架。通过事件相机采集数据，结合时空特征融合与记忆聚合模块，显著提升翻译性能，推动了无障碍AI发展。**

- **链接: [https://arxiv.org/pdf/2408.10488v2](https://arxiv.org/pdf/2408.10488v2)**

> **作者:** Shiao Wang; Xiao Wang; Duoqing Yang; Yao Rong; Fuling Wang; Jianing Li; Lin Zhu; Bo Jiang
>
> **摘要:** Sign Language Translation (SLT) is a core task in the field of AI-assisted disability. Traditional SLT methods are typically based on visible light videos, which are easily affected by factors such as lighting variations, rapid hand movements, and privacy concerns. This paper proposes the use of bio-inspired event cameras to alleviate the aforementioned issues. Specifically, we introduce a new high-definition event-based sign language dataset, termed Event-CSL, which effectively addresses the data scarcity in this research area. The dataset comprises 14,827 videos, 14,821 glosses, and 2,544 Chinese words in the text vocabulary. These samples are collected across diverse indoor and outdoor scenes, covering multiple viewpoints, lighting conditions, and camera motions. We have also benchmarked existing mainstream SLT methods on this dataset to facilitate fair comparisons in future research.Furthermore, we propose a novel event-based sign language translation framework, termed EvSLT. The framework first segments continuous video features into clips and employs a Mamba-based memory aggregation module to compress and aggregate spatial detail features at the clip level. Subsequently, these spatial features, along with temporal representations obtained from temporal convolution, are then fused by a graph-guided spatiotemporal fusion module. Extensive experiments on Event-CSL, as well as other publicly available datasets, demonstrate the superior performance of our method. The dataset and source code will be released on https://github.com/Event-AHU/OpenESL
>
---
#### [replaced 029] Atom of Thoughts for Markov LLM Test-Time Scaling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大语言模型测试时推理中因历史信息累积导致的计算浪费与推理干扰问题，提出基于马尔可夫过程的“思维原子”（Atom of Thoughts）框架。通过将复杂问题分解为独立子问题并迭代进行分解-压缩，实现无记忆的高效推理，可无缝集成至现有测试时扩展方法中，显著提升多跳问答等任务性能。**

- **链接: [https://arxiv.org/pdf/2502.12018v3](https://arxiv.org/pdf/2502.12018v3)**

> **作者:** Fengwei Teng; Quan Shi; Zhaoyang Yu; Jiayi Zhang; Chenglin Wu; Yuyu Luo; Zhijiang Guo
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Large Language Models (LLMs) achieve superior performance through training-time scaling, and test-time scaling further enhances their capabilities by conducting effective reasoning during inference. However, as the scale of reasoning increases, existing test-time scaling methods suffer from accumulated historical information, which not only wastes computational resources but also interferes with effective reasoning. To address this issue, we observe that complex reasoning can be achieved by solving a series of independent and self-contained subquestions. These subquestions are essentially \textit{atomic questions}, exhibiting the memoryless property similar to Markov processes. Based on this observation, we propose Atom of Thoughts (\our), where each state transition consists of decomposing the current question into a dependency-based directed acyclic graph and contracting its subquestions, forming a simplified question that maintains answer equivalence with the original problem. This answer preservation enables the iterative \textit{decomposition-contraction} process to naturally form a meaningful Markov reasoning process. Furthermore, these atomic states can be seamlessly integrated into existing test-time scaling methods, enabling \our to serve as a plug-in enhancement for improving reasoning capabilities. Experiments across six benchmarks demonstrate the effectiveness of \our both as a standalone framework and a plug-in enhancement. Notably, on HotpotQA, when applied to gpt-4o-mini, \our achieves an \textbf{80.6\%} F1 score, surpassing o3-mini by \textbf{3.4\%} and DeepSeek-R1 by \textbf{10.6\%}. The code is available at \href{https://github.com/qixucen/atom}{https://github.com/qixucen/atom}.
>
---
#### [replaced 030] Financial Risk Relation Identification through Dual-view Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对金融风险关联识别任务，解决传统人工评估主观、低效且难扩展的问题。通过分析企业10-K年报，利用自然语言处理技术，基于时间与词汇模式进行无监督微调，构建领域专用编码器，实现对隐含风险关系的量化评分，提升风险关联识别的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2509.18775v2](https://arxiv.org/pdf/2509.18775v2)**

> **作者:** Wei-Ning Chiu; Yu-Hsiang Wang; Andy Hsiao; Yu-Shiang Huang; Chuan-Ju Wang
>
> **备注:** 11 pages, 3 figures, EMNLP 2025 Main Conference
>
> **摘要:** A multitude of interconnected risk events -- ranging from regulatory changes to geopolitical tensions -- can trigger ripple effects across firms. Identifying inter-firm risk relations is thus crucial for applications like portfolio management and investment strategy. Traditionally, such assessments rely on expert judgment and manual analysis, which are, however, subjective, labor-intensive, and difficult to scale. To address this, we propose a systematic method for extracting inter-firm risk relations using Form 10-K filings -- authoritative, standardized financial documents -- as our data source. Leveraging recent advances in natural language processing, our approach captures implicit and abstract risk connections through unsupervised fine-tuning based on chronological and lexical patterns in the filings. This enables the development of a domain-specific financial encoder with a deeper contextual understanding and introduces a quantitative risk relation score for transparency, interpretable analysis. Extensive experiments demonstrate that our method outperforms strong baselines across multiple evaluation settings. Our codes are available at https://github.com/cnclabs/codes.fin.relation.
>
---
#### [replaced 031] One Patient, Many Contexts: Scaling Medical AI with Contextual Intelligence
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦医疗AI的泛化能力，旨在解决模型在不同临床场景中因上下文缺失导致的误判问题。提出“上下文切换”机制，使模型在不重训练的前提下动态适应患者特征、诊疗环境与多模态数据，提升跨专科、跨地域应用的可靠性与适应性。**

- **链接: [https://arxiv.org/pdf/2506.10157v3](https://arxiv.org/pdf/2506.10157v3)**

> **作者:** Michelle M. Li; Ben Y. Reis; Adam Rodman; Tianxi Cai; Noa Dagan; Ran D. Balicer; Joseph Loscalzo; Isaac S. Kohane; Marinka Zitnik
>
> **摘要:** Medical AI, including clinical language models, vision-language models, and multimodal health record models, already summarizes notes, answers questions, and supports decisions. Their adaptation to new populations, specialties, or care settings often relies on fine-tuning, prompting, or retrieval from external knowledge bases. These strategies can scale poorly and risk contextual errors: outputs that appear plausible but miss critical patient or situational information. We envision context switching as a solution. Context switching adjusts model reasoning at inference without retraining. Generative models can tailor outputs to patient biology, care setting, or disease. Multimodal models can reason on notes, laboratory results, imaging, and genomics, even when some data are missing or delayed. Agent models can coordinate tools and roles based on tasks and users. In each case, context switching enables medical AI to adapt across specialties, populations, and geographies. It requires advances in data design, model architectures, and evaluation frameworks, and establishes a foundation for medical AI that scales to infinitely many contexts while remaining reliable and suited to real-world care.
>
---
#### [replaced 032] Toward Honest Language Models for Deductive Reasoning
- **分类: cs.CL**

- **简介: 该论文研究语言模型在演绎推理中的诚实性问题，旨在使模型仅在结论被前提逻辑蕴含时作答，否则拒绝回答。针对现有方法易产生不当回答的问题，作者构建了基于图结构的双数据集，并提出ACNCHOR强化学习方法，通过注入真实推理轨迹稳定训练，显著提升模型诚实推理能力。**

- **链接: [https://arxiv.org/pdf/2511.09222v4](https://arxiv.org/pdf/2511.09222v4)**

> **作者:** Jiarui Liu; Kaustubh Dhole; Yingheng Wang; Haoyang Wen; Sarah Zhang; Haitao Mao; Gaotang Li; Neeraj Varshney; Jingguo Liu; Xiaoman Pan
>
> **摘要:** Deductive reasoning is the process of deriving conclusions strictly from the given premises, without relying on external knowledge. We define honesty in this setting as a model's ability to respond only when the conclusion is logically entailed by the premises, and to abstain otherwise. However, current language models often fail to reason honestly, producing unwarranted answers when the input is insufficient. To study this challenge, we formulate honest deductive reasoning as multi-step tasks where models must either derive the correct conclusion or abstain. We curate two datasets from graph structures, one for linear algebra and one for logical inference, and introduce unanswerable cases by randomly perturbing an edge in half of the instances. We find that prompting and existing training methods, including GRPO with or without supervised fine-tuning initialization, struggle on these tasks. In particular, GRPO optimize only for final task outcomes, leaving models vulnerable to collapse when negative rewards dominate early training. To address this, we propose ACNCHOR, a reinforcement learning method that injects ground truth trajectories into rollouts, preventing early training collapse. Our results demonstrate that this method stabilizes learning and significantly improves the overall reasoning performance, underscoring the importance of training dynamics for enabling honest deductive reasoning in language models.
>
---
#### [replaced 033] Normal forms in Virus Machines
- **分类: cs.CL; cs.FL**

- **简介: 该论文研究病毒机器（VM）的计算能力，提出多种正则形式以简化模型结构。通过限制主机数、指令数及病毒对象数量等特征，刻画了有限集、半线性集和递归可枚举集等集合类的计算特性，深化了对VM计算能力的理解。**

- **链接: [https://arxiv.org/pdf/2409.03327v3](https://arxiv.org/pdf/2409.03327v3)**

> **作者:** A. Ramírez-de-Arellano; F. G. C. Cabarle; D. Orellana-Martín; M. J. Pérez-Jiménez
>
> **备注:** 24 pages, 14 figures
>
> **摘要:** In the present work, we further study the computational power of virus machines (VMs in short).VMs provide a computing paradigm inspired by the transmission and replication networks of viruses.VMs consist of process units (called hosts) structured by a directed graph whose arcs are called channels and an instruction graph that controls the transmissions of virus objects among hosts. The present work complements our understanding of the computing power of VMs by introducing normal forms; these expressions restrict the features in a given computing model.Some of the features that we restrict in our normal forms include (a) the number of hosts, (b) the number of instructions, and (c) the number of virus objects in each host. After we recall some known results on the computing power of VMs we give our series of normal forms, such as the size of the loops in the network, proving new characterisations of family of sets, such as finite sets, semilinear sets, or recursively enumerable sets (NRE).
>
---
#### [replaced 034] Simulated patient systems powered by large language model-based AI agents offer potential for transforming medical education
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对医学教育中模拟患者系统真实性和成本高的问题，提出基于大语言模型的AIPatient系统。通过六类任务专用智能体与知识图谱融合，实现高精度、高可读性的医患交互，经实证验证其在问答准确率、稳定性及用户体验上表现优异，具备替代真人模拟患者的潜力。**

- **链接: [https://arxiv.org/pdf/2409.18924v4](https://arxiv.org/pdf/2409.18924v4)**

> **作者:** Huizi Yu; Jiayan Zhou; Lingyao Li; Shan Chen; Jack Gallifant; Anye Shi; Xiang Li; Jingxian He; Wenyue Hua; Mingyu Jin; Guang Chen; Yang Zhou; Zhao Li; Trisha Gupte; Ming-Li Chen; Zahra Azizi; Qi Dou; Bryan P. Yan; Yongfeng Zhang; Yanqiu Xing; Themistocles L. Danielle S. Bitterman; Themistocles L. Assimes; Xin Ma; Lin Lu; Lizhou Fan
>
> **备注:** 19 pages, 6 figures, 4 tables
>
> **摘要:** Background: Simulated patient systems are important in medical education and research, providing safe, integrative training environments and supporting clinical decision making. Advances in artificial intelligence (AI), especially large language models (LLMs), can enhance simulated patients by replicating medical conditions and doctor patient interactions with high fidelity and at low cost, but effectiveness and trustworthiness remain open challenges. Methods: We developed AIPatient, a simulated patient system powered by LLM based AI agents. The system uses a retrieval augmented generation (RAG) framework with six task specific agents for complex reasoning. To improve realism, it is linked to the AIPatient knowledge graph built from de identified real patient data in the MIMIC III intensive care database. Results: We evaluated electronic health record (EHR) based medical question answering (QA), readability, robustness, stability, and user experience. AIPatient reached 94.15 percent QA accuracy when all six agents were enabled, outperforming versions with partial or no agent integration. The knowledge base achieved an F1 score of 0.89. Readability scores showed a median Flesch Reading Ease of 68.77 and a median Flesch Kincaid Grade of 6.4, indicating accessibility for most medical trainees and clinicians. Robustness and stability were supported by non significant variance in repeated trials (analysis of variance F value 0.61, p greater than 0.1; F value 0.78, p greater than 0.1). A user study with medical students showed that AIPatient provides high fidelity, usability, and educational value, comparable to or better than human simulated patients for history taking. Conclusions: LLM based simulated patient systems can deliver accurate, readable, and reliable medical encounters and show strong potential to transform medical education.
>
---
#### [replaced 035] OpenMMReasoner: Pushing the Frontiers for Multimodal Reasoning with an Open and General Recipe
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出OpenMMReasoner，针对多模态推理中数据与训练策略不透明的问题，设计了可复现的两阶段训练方案（SFT+RL）。通过高质量数据集构建，显著提升模型在多模态推理任务上的性能，相比基线提升11.6%，并开源全部资源，推动领域发展。**

- **链接: [https://arxiv.org/pdf/2511.16334v2](https://arxiv.org/pdf/2511.16334v2)**

> **作者:** Kaichen Zhang; Keming Wu; Zuhao Yang; Kairui Hu; Bin Wang; Ziwei Liu; Xingxuan Li; Lidong Bing
>
> **摘要:** Recent advancements in large reasoning models have fueled growing interest in extending such capabilities to multimodal domains. However, despite notable progress in visual reasoning, the lack of transparent and reproducible data curation and training strategies remains a major barrier to scalable research. In this work, we introduce OpenMMReasoner, a fully transparent two-stage recipe for multimodal reasoning spanning supervised fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we construct an 874K-sample cold-start dataset with rigorous step-by-step validation, providing a strong foundation for reasoning capabilities. The subsequent RL stage leverages a 74K-sample dataset across diverse domains to further sharpen and stabilize these abilities, resulting in a more robust and efficient learning process. Extensive evaluations demonstrate that our training recipe not only surpasses strong baselines but also highlights the critical role of data quality and training design in shaping multimodal reasoning performance. Notably, our method achieves a 11.6% improvement over the Qwen2.5-VL-7B-Instruct baseline across nine multimodal reasoning benchmarks, establishing a solid empirical foundation for future large-scale multimodal reasoning research. We open-sourced all our codes, pipeline, and data at https://github.com/EvolvingLMMs-Lab/OpenMMReasoner.
>
---
#### [replaced 036] Accelerating Training of Recursive Reasoning Models with Curriculum Guided Adaptive Recursion
- **分类: cs.LG; cs.AI; cs.CL; cs.NE**

- **简介: 该论文针对递归推理模型训练耗时过高的问题，提出CGAR方法。通过课程学习控制网络深度递增，并引入分层监督加权，实现训练加速与精度保持。在数独任务上提升2.26倍速度，降低42%成本，同时提升推理效率。**

- **链接: [https://arxiv.org/pdf/2511.08653v2](https://arxiv.org/pdf/2511.08653v2)**

> **作者:** Kaleem Ullah Qasim; Jiashu Zhang
>
> **摘要:** Recursive reasoning models achieve remarkable performance on complex reasoning tasks through iterative refinement, enabling tiny networks to match large language models thousands of times their size. However, training remains computationally expensive, prior work reporting approximately 36 GPU-hours per dataset, limiting broader adoption and research. We propose CGAR, a novel training methodology that applies curriculum learning to architectural depth rather than traditional data ordering. CGAR introduces two synergistic components: Progressive Depth Curriculum dynamically adjusts recursion depth from shallow to deep configurations during training, preventing early overfitting while reducing computational cost, and Hierarchical Supervision Weighting applies exponentially decaying importance to supervision steps, aligning loss weighting with observed gradient magnitude decay. On Sudoku-Extreme with 423,168 test puzzles, CGAR achieves 1.71x training speedup (10.93 to 6.38 hours, 42% cost reduction) with only 0.63% accuracy drop (86.65% to 86.02%). Systematic ablations reveal Progressive Depth Curriculum alone achieves 2.26x speedup with 85.47% accuracy, demonstrating a rare Pareto improvement where architectural curriculum simultaneously enhances training efficiency and solution quality. CGAR-trained models exhibit superior inference efficiency with 100% halting accuracy and 11% fewer reasoning steps. Our work demonstrates that principled curriculum on architectural depth enables efficient training of recursive reasoning models on modest hardware. Code and models: https://github.com/Kaleemullahqasim/CGAR and https://huggingface.co/Kaleemullah/trm-cgar-sudoku
>
---
#### [replaced 037] Continual Learning with Global Alignment
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究持续学习任务，针对灾难性遗忘问题，提出通过全局对齐机制增强不同任务间数据表示的相关性。方法基于共享预训练标记表示，构建任务特定表示，减少梯度干扰，无需经验回放即达先进性能。**

- **链接: [https://arxiv.org/pdf/2205.12186v3](https://arxiv.org/pdf/2205.12186v3)**

> **作者:** Xueying Bai; Jinghuan Shang; Yifan Sun; Niranjan Balasubramanian
>
> **备注:** Neurips 2024
>
> **摘要:** Continual learning aims to sequentially learn new tasks without forgetting previous tasks' knowledge (catastrophic forgetting). One factor that can cause forgetting is the interference between the gradients on losses from different tasks. When the gradients on the current task's loss are in opposing directions to those on previous tasks' losses, updating the model for the current task may cause performance degradation on previous tasks. In this paper, we first identify causes of the above interference, and hypothesize that correlations between data representations are a key factor of interference. We then propose a method for promoting appropriate correlations between arbitrary tasks' data representations (i.e., global alignment) in individual task learning. Specifically, we learn the data representation as a task-specific composition of pre-trained token representations shared across all tasks. Then the correlations between different tasks' data representations are grounded by correlations between pre-trained token representations. We explore different ways to learn such compositions. Without experience replay, our model achieves SOTA performance in continual learning tasks. It also achieves advanced class-incremental performance through task-incremental training.
>
---
#### [replaced 038] Masked Diffusion Models as Energy Minimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文将掩码扩散模型（MDMs）视为离散最优传输中的能量最小化问题，证明三种能量形式在MDM框架下等价，并提出满足闭式最优条件的掩码调度。通过参数化插值路径为贝塔分布，实现无需修改模型的高效后训练调优，显著提升低步数采样性能。**

- **链接: [https://arxiv.org/pdf/2509.13866v2](https://arxiv.org/pdf/2509.13866v2)**

> **作者:** Sitong Chen; Shen Nie; Jiacheng Sun; Zijin Feng; Zhenguo Li; Ji-Rong Wen; Chongxuan Li
>
> **摘要:** We present a systematic theoretical framework that interprets masked diffusion models (MDMs) as solutions to energy minimization problems in discrete optimal transport. Specifically, we prove that three distinct energy formulations--kinetic, conditional kinetic, and geodesic energy--are mathematically equivalent under the structure of MDMs, and that MDMs minimize all three when the mask schedule satisfies a closed-form optimality condition. This unification not only clarifies the theoretical foundations of MDMs, but also motivates practical improvements in sampling. By parameterizing interpolation schedules via Beta distributions, we reduce the schedule design space to a tractable 2D search, enabling efficient post-training tuning without model modification. Experiments on synthetic and real-world benchmarks demonstrate that our energy-inspired schedules outperform hand-crafted baselines, particularly in low-step sampling settings.
>
---
#### [replaced 039] Self Iterative Label Refinement via Robust Unlabeled Learning
- **分类: cs.CL**

- **简介: 该论文针对大语言模型自精炼中因内部偏见和过自信导致的伪标签质量下降问题，提出一种基于无标签-无标签学习的迭代伪标签优化方法。通过利用两个正类比例不同的无标签数据集，迭代净化与提升伪标签，显著提升分类性能，并成功应用于低资源语言、专利分类及蛋白质结构识别，同时支持安全对齐与生成任务自精炼。**

- **链接: [https://arxiv.org/pdf/2502.12565v2](https://arxiv.org/pdf/2502.12565v2)**

> **作者:** Hikaru Asano; Tadashi Kozuno; Yukino Baba
>
> **备注:** To appear in the Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Recent advances in large language models (LLMs) have yielded impressive performance on various tasks, yet they often depend on high-quality feedback that can be costly. Self-refinement methods attempt to leverage LLMs' internal evaluation mechanisms with minimal human supervision; however, these approaches frequently suffer from inherent biases and overconfidence, especially in domains where the models lack sufficient internal knowledge, resulting in performance degradation. As an initial step toward enhancing self-refinement for broader applications, we introduce an iterative refinement pipeline that employs the Unlabeled-Unlabeled learning framework to improve LLM-generated pseudo-labels for classification tasks. By exploiting two unlabeled datasets with differing positive class ratios, our approach iteratively denoises and refines the initial pseudo-labels, thereby mitigating the adverse effects of internal biases with minimal human supervision. Evaluations on diverse datasets, including low-resource language corpora, patent classifications, and protein structure categorizations, demonstrate that our method consistently outperforms both initial LLM's classification performance and the self-refinement approaches by cutting-edge models (e.g., GPT-4o and DeepSeek-R1). Moreover, we experimentally confirm that our refined classifier facilitates effective post-training alignment for safety in LLMs and demonstrate successful self-refinement in generative tasks as well.\footnote{Our code is available at https://github.com/HikaruAsano/self-iterative-label-refinement.}
>
---
#### [replaced 040] Beyond Introspection: Reinforcing Thinking via Externalist Behavioral Feedback
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型推理可靠性问题，提出外部主义三步框架DRR。通过分析模型行为轨迹，训练轻量级外部判别模型，以识别并纠正可疑推理步骤，避免自省偏差。实验表明，DRR显著优于自纠错方法，且无需标注，可广泛适配各类LLMs。**

- **链接: [https://arxiv.org/pdf/2501.01457v3](https://arxiv.org/pdf/2501.01457v3)**

> **作者:** Diji Yang; Linda Zeng; Kezhen Chen; Yi Zhang
>
> **摘要:** While inference-time thinking allows Large Language Models (LLMs) to address complex problems, the extended thinking process can be unreliable or inconsistent because of the model's probabilistic nature, especially near its knowledge boundaries. Existing approaches attempt to mitigate this by having the model critique its own reasoning to make corrections. However, such self-critique inherits the same biases of the original output, known as the introspection illusion. Moving beyond such introspection and inspired by core methodologies in ethology, we propose an externalist three-step framework Distillation-Reinforcement-Reasoning (DRR). Rather than relying on a model's introspection, DRR evaluates its observable behaviors to provide corrective feedback. DRR first distills the reasoner's behavioral traces, then trains a lightweight, external Discriminative Model (DM). At inference time, this DM acts as a critic, identifying and rejecting suspicious reasoning steps. This external feedback compels the LLM to discard flawed pathways and explore alternatives, thereby enhancing reasoning quality without altering the base model. Experiments on multiple reasoning benchmarks show that our framework significantly outperforms prominent self-critique methods. Benefiting from a lightweight and annotation-free design, DRR offers a scalable and adaptable solution for improving the reliability of reasoning in a wide range of LLMs.
>
---
#### [replaced 041] FlowerTune: A Cross-Domain Benchmark for Federated Fine-Tuning of Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出FlowerTune，首个面向大语言模型联邦微调的跨领域基准。针对数据隐私与领域专有数据稀缺问题，构建涵盖通用NLP、金融、医疗、编码四个领域的联邦微调数据集与评估指标，首次系统比较26个模型在联邦设置下的性能，推动隐私保护下领域专用LLM的发展。**

- **链接: [https://arxiv.org/pdf/2506.02961v2](https://arxiv.org/pdf/2506.02961v2)**

> **作者:** Yan Gao; Massimo Roberto Scamarcia; Javier Fernandez-Marques; Mohammad Naseri; Chong Shen Ng; Dimitris Stripelis; Zexi Li; Tao Shen; Jiamu Bai; Daoyuan Chen; Zikai Zhang; Rui Hu; InSeo Song; Lee KangYoon; Hong Jia; Ting Dang; Junyan Wang; Zheyuan Liu; Daniel Janes Beutel; Lingjuan Lyu; Nicholas D. Lane
>
> **摘要:** Large Language Models (LLMs) have achieved state-of-the-art results across diverse domains, yet their development remains reliant on vast amounts of publicly available data, raising concerns about data scarcity and the lack of access to domain-specific, sensitive information. Federated Learning (FL) presents a compelling framework to address these challenges by enabling decentralized fine-tuning on pre-trained LLMs without sharing raw data. However, the compatibility and performance of pre-trained LLMs in FL settings remain largely under explored. We introduce the FlowerTune LLM Leaderboard, a first-of-its-kind benchmarking suite designed to evaluate federated fine-tuning of LLMs across four diverse domains: general NLP, finance, medical, and coding. Each domain includes federated instruction-tuning datasets and domain-specific evaluation metrics. Our results, obtained through a collaborative, open-source and community-driven approach, provide the first comprehensive comparison across 26 pre-trained LLMs with different aggregation and fine-tuning strategies under federated settings, offering actionable insights into model performance, resource constraints, and domain adaptation. This work lays the foundation for developing privacy-preserving, domain-specialized LLMs for real-world applications.
>
---
#### [replaced 042] Holistic Evaluation of Multimodal LLMs on Spatial Intelligence
- **分类: cs.CV; cs.CL; cs.LG; cs.MM; cs.RO**

- **简介: 该论文聚焦多模态大模型的空间智能（SI）评估，提出EASI框架，统一现有与新构建的时空任务基准。通过在八项基准上超十亿令牌的实测，揭示当前顶尖模型（如GPT-5）虽强但仍远逊于人类，且非开源模型无显著优势。研究开放代码与排行榜，推动可复现、持续更新的SI评估。**

- **链接: [https://arxiv.org/pdf/2508.13142v4](https://arxiv.org/pdf/2508.13142v4)**

> **作者:** Zhongang Cai; Yubo Wang; Qingping Sun; Ruisi Wang; Chenyang Gu; Wanqi Yin; Zhiqian Lin; Zhitao Yang; Chen Wei; Oscar Qian; Hui En Pang; Xuanke Shi; Kewang Deng; Xiaoyang Han; Zukai Chen; Jiaqi Li; Xiangyu Fan; Hanming Deng; Lewei Lu; Bo Li; Ziwei Liu; Quan Wang; Dahua Lin; Lei Yang
>
> **备注:** Codebase: https://github.com/EvolvingLMMs-Lab/EASI/; Leaderboard: https://huggingface.co/spaces/lmms-lab-si/EASI-Leaderboard
>
> **摘要:** Multimodal models have achieved remarkable progress in recent years. Nevertheless, they continue to exhibit notable limitations in spatial understanding and reasoning, the very capability that anchors artificial general intelligence in the physical world. With the recent release of GPT-5, allegedly the most powerful AI model to date, it is timely to examine where the leading models (GPT, Gemini, Grok, Seed, Qwen, and Intern) stand on the path toward spatial intelligence (SI). We thus propose EASI for holistic Evaluation of multimodAl LLMs on Spatial Intelligence. EASI conceptualizes a comprehensive taxonomy of spatial tasks that unifies existing benchmarks and a growing collection of newly curated ones, enabling systematic evaluation of state-of-the-art models. In this report, we conduct the study across eight key benchmarks, at a cost exceeding ten billion total tokens. Our empirical study then reveals that (1) GPT-5 demonstrates unprecedented strength in SI, yet (2) still falls short of human performance significantly across a broad spectrum of SI-tasks. Moreover, we (3) show that SI-tasks expose greater model capability deficiency than non-SI tasks, to the extent that (4) proprietary models do not exhibit a decisive advantage when facing the most difficult ones. In addition, we conduct a qualitative evaluation across a diverse set of scenarios that are intuitive for humans, yet fail the most advanced multimodal models. EASI is an ongoing community effort: we have open-sourced the EASI codebase that provides a one-stop and reproducible solution with standardized interfaces, integrated protocols and prompts that significantly reduce the friction of configuring and running multiple benchmarks; we have also launched an accompanying EASI leaderboard to provide a continually updated snapshot of model performance across the full SI spectrum, accelerating collective progress toward robust SI.
>
---
#### [replaced 043] Efficient Reasoning via Thought-Training and Thought-Free Inference
- **分类: cs.CL**

- **简介: 该论文针对大模型推理效率与准确率矛盾问题，提出3TF框架。通过训练混合模型，在推理时启用隐式思维模式，实现无需显式思维链的高效高质推理，突破传统压缩式方法依赖大量短思维链数据的局限。**

- **链接: [https://arxiv.org/pdf/2511.03408v3](https://arxiv.org/pdf/2511.03408v3)**

> **作者:** Canhui Wu; Qiong Cao; Chao Xue; Wei Xi; Xiaodong He
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Recent advances in large language models (LLMs) have leveraged explicit Chain-of-Thought (CoT) prompting to improve reasoning accuracy. However, most existing methods primarily focus on compressing verbose reasoning outputs. These Long-to-Short transformations aim to improve efficiency, but require a large amount of short CoT data. In this work, we introduce \textbf{3TF} (\textbf{T}hought-\textbf{T}raining and \textbf{T}hought-\textbf{F}ree inference), a framework for efficient reasoning that takes a Short-to-Long perspective. We first train a hybrid model that can operate in both reasoning and non-reasoning modes, and then further train it on CoT-annotated data to internalize structured reasoning, while enforcing concise, thought-free outputs at inference time using the no-reasoning mode. Unlike compression-based approaches, 3TF improves the reasoning quality of non-reasoning outputs, enabling models to perform rich internal reasoning implicitly while keeping external outputs short. Empirically, 3TF-trained models obtain large improvements on reasoning benchmarks under thought-free inference, demonstrating that high quality reasoning can be learned and executed implicitly without explicit step-by-step generation.
>
---
#### [replaced 044] Mavors: Multi-granularity Video Representation for Multimodal Large Language Model
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对多模态大模型中的长视频理解任务，解决现有方法在处理复杂视频时信息丢失的问题。提出Mavors框架，通过多粒度视频表示，结合高分辨率空间编码与跨片段时序建模，有效保留细粒度时空特征，提升长视频理解性能。**

- **链接: [https://arxiv.org/pdf/2504.10068v2](https://arxiv.org/pdf/2504.10068v2)**

> **作者:** Yang Shi; Jiaheng Liu; Yushuo Guan; Zhenhua Wu; Yuanxing Zhang; Zihao Wang; Weihong Lin; Jingyun Hua; Zekun Wang; Xinlong Chen; Bohan Zeng; Wentao Zhang; Fuzheng Zhang; Wenjing Yang; Di Zhang
>
> **备注:** 22 pages
>
> **摘要:** Long-context video understanding in multimodal large language models (MLLMs) faces a critical challenge: balancing computational efficiency with the retention of fine-grained spatio-temporal patterns. Existing approaches (e.g., sparse sampling, dense sampling with low resolution, and token compression) suffer from significant information loss in temporal dynamics, spatial details, or subtle interactions, particularly in videos with complex motion or varying resolutions. To address this, we propose $\mathbf{Mavors}$, a novel framework that introduces $\mathbf{M}$ulti-gr$\mathbf{a}$nularity $\mathbf{v}$ide$\mathbf{o}$ $\mathbf{r}$epre$\mathbf{s}$entation for holistic long-video modeling. Specifically, Mavors directly encodes raw video content into latent representations through two core components: 1) an Intra-chunk Vision Encoder (IVE) that preserves high-resolution spatial features via 3D convolutions and Vision Transformers, and 2) an Inter-chunk Feature Aggregator (IFA) that establishes temporal coherence across chunks using transformer-based dependency modeling with chunk-level rotary position encodings. Moreover, the framework unifies image and video understanding by treating images as single-frame videos via sub-image decomposition. Experiments across diverse benchmarks demonstrate Mavors' superiority in maintaining both spatial fidelity and temporal continuity, significantly outperforming existing methods in tasks requiring fine-grained spatio-temporal reasoning.
>
---
#### [replaced 045] Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言模型预训练中词表大小的影响。针对“更大词表是否真的有益”这一问题，通过控制变量实验发现，扩大词表主要降低文本的柯尔莫哥洛夫复杂度，仅提升高频词的训练效果，且其优势可转移至下游任务。结论表明：词表扩展的本质是降低文本复杂度，为词表与模型协同设计提供理论依据。**

- **链接: [https://arxiv.org/pdf/2508.15390v3](https://arxiv.org/pdf/2508.15390v3)**

> **作者:** Woojin Chung; Jeonghoon Kim
>
> **备注:** NeurIPS 2025
>
> **摘要:** Large language models are trained with tokenizers, and the resulting token distribution is highly imbalanced: a few words dominate the stream while most occur rarely. Recent practice favors ever-larger vocabularies, but it is unclear where the benefit comes from. To this end, we perform a controlled study that scales the vocabulary of the language model from 24K to 196K while holding data, computation, and optimization unchanged. We begin by quantifying the complexity of tokenized text -- formalized via Kolmogorov complexity -- and show that larger vocabularies reduce this complexity. Above 24K, every common word is already tokenized as a single token, so enlarging vocabulary only deepens the relative token-frequency imbalance. Word-level loss decomposition shows that larger vocabularies reduce cross-entropy loss almost exclusively by lowering uncertainty on the 2,500 most frequent words, even though loss on the rare tail rises. The same frequent words cover roughly 75% of tokens in downstream benchmarks, so this training advantage transfers intact. We further show that enlarging model parameters with a fixed vocabulary yields the same frequent-word benefit. Our results recast "bigger vocabularies help" as "lowering complexity of tokenized text helps," offering a simple, principled knob for tokenizer-model co-design and clarifying the loss dynamics that govern language model scaling in pre-training.
>
---
#### [replaced 046] Linguistically-Controlled Paraphrase Generation
- **分类: cs.CL**

- **简介: 该论文针对可控改写生成任务，解决如何精确控制输出文本语言属性的同时保持语义一致的问题。提出LingConv框架，实现对40个语言属性的细粒度控制，并引入推理时质量控制机制，显著降低属性误差，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2410.24199v2](https://arxiv.org/pdf/2410.24199v2)**

> **作者:** Mohamed Elgaar; Hadi Amiri
>
> **备注:** This paper was published in Findings of ACL: EMNLP 2025
>
> **摘要:** Controlled paraphrase generation produces paraphrases that preserve meaning while allowing precise control over linguistic attributes of the output. We introduce LingConv, an encoder-decoder framework that enables fine-grained control over 40 linguistic attributes in English. To improve reliability, we introduce a novel inference-time quality control mechanism that iteratively refines attribute embeddings to generate paraphrases that closely match target attributes without sacrificing semantic fidelity. LingConv reduces attribute error by up to 34% over existing models, with the quality control mechanism contributing an additional 14% improvement.
>
---
#### [replaced 047] Adversarial Confusion Attack: Disrupting Multimodal Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出一种针对多模态大语言模型（MLLMs）的对抗性混淆攻击，旨在通过生成扰动图像使模型输出混乱或自信错误。任务为提升模型鲁棒性，解决对抗样本威胁问题。工作包括设计基于PGD的高熵攻击方法，实现跨模型迁移，在白盒环境下有效干扰多个开源与闭源模型。**

- **链接: [https://arxiv.org/pdf/2511.20494v2](https://arxiv.org/pdf/2511.20494v2)**

> **作者:** Jakub Hoscilowicz; Artur Janicki
>
> **摘要:** We introduce the Adversarial Confusion Attack, a new class of threats against multimodal large language models (MLLMs). Unlike jailbreaks or targeted misclassification, the goal is to induce systematic disruption that makes the model generate incoherent or confidently incorrect outputs. Applications include embedding adversarial images into websites to prevent MLLM-powered agents from operating reliably. The proposed attack maximizes next-token entropy using a small ensemble of open-source MLLMs. In the white-box setting, we show that a single adversarial image can disrupt all models in the ensemble, both in the full-image and adversarial CAPTCHA settings. Despite relying on a basic adversarial technique (PGD), the attack generates perturbations that transfer to both unseen open-source (e.g., Qwen3-VL) and proprietary (e.g., GPT-5.1) models.
>
---
#### [replaced 048] More Documents, Same Length: Isolating the Challenge of Multiple Documents in RAG
- **分类: cs.CL**

- **简介: 该论文研究RAG中多文档对模型性能的影响。针对“文档数量增加是否独立于上下文长度影响性能”的问题，通过控制上下文长度与信息位置，对比不同模型在多文档下的表现，发现多数模型性能下降，而Qwen2.5表现稳定，证明多文档处理是独立挑战。**

- **链接: [https://arxiv.org/pdf/2503.04388v3](https://arxiv.org/pdf/2503.04388v3)**

> **作者:** Shahar Levy; Nir Mazor; Lihi Shalmon; Michael Hassid; Gabriel Stanovsky
>
> **备注:** Preprint
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances the accuracy of Large Language Model (LLM) responses by leveraging relevant external documents during generation. Although previous studies noted that retrieving many documents can degrade performance, they did not isolate how the quantity of documents affects performance while controlling for context length. We evaluate various language models on custom datasets derived from a multi-hop QA task. We keep the context length and position of relevant information constant while varying the number of documents, and find that increasing the document count in RAG settings poses significant challenges for most LLMs, reducing performance by up to 20%. However, Qwen2.5 maintained consistent results across increasing document counts, indicating better multi-document handling capability. Finally, our results indicate that processing multiple documents is a separate challenge from handling long contexts. We also make the datasets and code available: https://github.com/shaharl6000/MoreDocsSameLen .
>
---
#### [replaced 049] Leveraging Biomolecule and Natural Language through Multi-Modal Learning: A Survey
- **分类: cs.CL; cs.AI; q-bio.BM**

- **简介: 该论文属于多模态学习任务，旨在融合生物分子与自然语言数据。通过分析分子序列、结构及文本描述，解决生物分子表征不完整问题，推动属性预测等应用。工作包括梳理技术方法、整合多源数据、总结资源并展望未来方向。**

- **链接: [https://arxiv.org/pdf/2403.01528v3](https://arxiv.org/pdf/2403.01528v3)**

> **作者:** Qizhi Pei; Zhimeng Zhou; Kaiyuan Gao; Jinhua Zhu; Yue Wang; Zun Wang; Tao Qin; Lijun Wu; Rui Yan
>
> **备注:** 2025.11.28 Updated Version
>
> **摘要:** The integration of biomolecular modeling with natural language (BL) has emerged as a promising interdisciplinary area at the intersection of artificial intelligence, chemistry and biology. This approach leverages the rich, multifaceted descriptions of biomolecules contained within textual data sources to enhance our fundamental understanding and enable downstream computational tasks such as biomolecule property prediction. The fusion of the nuanced narratives expressed through natural language with the structural and functional specifics of biomolecules described via various molecular modeling techniques opens new avenues for comprehensively representing and analyzing biomolecules. By incorporating the contextual language data that surrounds biomolecules into their modeling, BL aims to capture a holistic view encompassing both the symbolic qualities conveyed through language as well as quantitative structural characteristics. In this review, we provide an extensive analysis of recent advancements achieved through cross modeling of biomolecules and natural language. (1) We begin by outlining the technical representations of biomolecules employed, including sequences, 2D graphs, and 3D structures. (2) We then examine in depth the rationale and key objectives underlying effective multi-modal integration of language and molecular data sources. (3) We subsequently survey the practical applications enabled to date in this developing research area. (4) We also compile and summarize the available resources and datasets to facilitate future work. (5) Looking ahead, we identify several promising research directions worthy of further exploration and investment to continue advancing the field. The related resources and contents are updating in https://github.com/QizhiPei/Awesome-Biomolecule-Language-Cross-Modeling.
>
---
#### [replaced 050] RvLLM: LLM Runtime Verification with Domain Knowledge
- **分类: cs.AI; cs.CL; cs.LO**

- **简介: 该论文针对大语言模型（LLM）输出不一致、错误频发的问题，提出RvLLM框架，通过引入领域知识进行运行时验证。设计轻量级规范语言ESL，使领域专家可定制约束条件，实现对LLM输出的高效、灵活验证，有效检测新加坡地铁法案合规性、数值比较和不等式求解中的错误。**

- **链接: [https://arxiv.org/pdf/2505.18585v3](https://arxiv.org/pdf/2505.18585v3)**

> **作者:** Yedi Zhang; Sun Yi Emma; Annabelle Lee Jia En; Jin Song Dong
>
> **备注:** 24 pages, 11 tables, 13 figures
>
> **摘要:** Large language models (LLMs) have emerged as a dominant AI paradigm due to their exceptional text understanding and generation capabilities. However, their tendency to generate inconsistent or erroneous outputs challenges their reliability, especially in high-stakes domains requiring accuracy and trustworthiness. Existing research primarily focuses on detecting and mitigating model misbehavior in general-purpose scenarios, often overlooking the potential of integrating domain-specific knowledge. In this work, we advance misbehavior detection by incorporating domain knowledge. The core idea is to design a general specification language that enables domain experts to customize domain-specific predicates in a lightweight and intuitive manner, supporting later runtime verification of LLM outputs. To achieve this, we design a novel specification language, ESL, and introduce a runtime verification framework, RvLLM, to validate LLM output against domain-specific constraints defined in ESL. We evaluate RvLLM on three representative tasks: violation detection against Singapore Rapid Transit Systems Act, numerical comparison, and inequality solving. Experimental results demonstrate that RvLLM effectively detects erroneous outputs across various LLMs in a lightweight and flexible manner. The results reveal that despite their impressive capabilities, LLMs remain prone to low-level errors due to limited interpretability and a lack of formal guarantees during inference, and our framework offers a potential long-term solution by leveraging expert domain knowledge to rigorously and efficiently verify LLM outputs.
>
---
#### [replaced 051] Structured Prompting Enables More Robust Evaluation of Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型（LM）评估中的性能低估问题，提出将结构化提示（如DSPy）融入HELM框架，通过优化提示提升评估准确性。解决了固定提示导致性能估计不准确、排名失真等问题，验证了结构化提示可更稳健地揭示模型真实能力。**

- **链接: [https://arxiv.org/pdf/2511.20836v2](https://arxiv.org/pdf/2511.20836v2)**

> **作者:** Asad Aali; Muhammad Ahmed Mohsin; Vasiliki Bikia; Arnav Singhvi; Richard Gaus; Suhana Bedi; Hejie Cui; Miguel Fuentes; Alyssa Unell; Yifan Mai; Jordan Cahoon; Michael Pfeffer; Roxana Daneshjou; Sanmi Koyejo; Emily Alsentzer; Christopher Potts; Nigam H. Shah; Akshay S. Chaudhari
>
> **摘要:** As language models (LMs) are increasingly adopted across domains, high-quality benchmarking frameworks that accurately estimate performance are essential for guiding deployment decisions. While frameworks such as Holistic Evaluation of Language Models (HELM) enable broad evaluation across tasks, they often rely on fixed prompts that fail to generalize across LMs, yielding unrepresentative performance estimates. Unless we approximate each LM's ceiling (maximum achievable via changes to the prompt), we risk underestimating performance. Declarative prompting frameworks, such as DSPy, offer a scalable alternative to manual prompt engineering by crafting structured prompts that can be optimized per task. However, such frameworks have not been systematically evaluated across established benchmarks. We present a reproducible DSPy+HELM framework that introduces structured prompting methods which elicit reasoning, enabling more accurate LM benchmarking. Using four prompting methods, we evaluate four frontier LMs across seven benchmarks (general/medical domain) against existing HELM baseline scores. We find that without structured prompting: (i) HELM underestimates LM performance (by 4% average), (ii) performance estimates vary more across benchmarks ($+$2% standard deviation), (iii) performance gaps are misrepresented (leaderboard rankings flip on 3/7 benchmarks), and (iv) introducing chain-of-thought reduces LM sensitivity to prompt design (smaller $Δ$ across prompts). To our knowledge, this is the first benchmarking study to systematically integrate structured prompting into an established evaluation framework, demonstrating how scalable performance-ceiling approximation yields more robust, decision-useful benchmarks. We open-source (i) DSPy+HELM Integration (https://github.com/stanford-crfm/helm/pull/3893) and (ii) Prompt Optimization Pipeline (https://github.com/StanfordMIMI/dspy-helm).
>
---
#### [replaced 052] Robust LLM Unlearning with MUDMAN: Meta-Unlearning with Disruption Masking And Normalization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型（LLM）难以彻底消除危险知识的问题，提出MUDMAN方法。通过引入“干扰掩蔽”和梯度归一化，确保更新不破坏原有知识，结合元学习实现鲁棒去遗忘。实验表明其在防止能力恢复上优于先前方法40%，达到新基准。**

- **链接: [https://arxiv.org/pdf/2506.12484v5](https://arxiv.org/pdf/2506.12484v5)**

> **作者:** Filip Sondej; Yushi Yang; Mikołaj Kniejski; Marcel Windys
>
> **摘要:** Language models can retain dangerous knowledge and skills even after extensive safety fine-tuning, posing both misuse and misalignment risks. Recent studies show that even specialized unlearning methods can be easily reversed. To address this, we systematically evaluate many existing and novel components of unlearning methods and identify ones crucial for irreversible unlearning. We introduce Disruption Masking, a technique in which we only allow updating weights, where the signs of the unlearning gradient and the retaining gradient are the same. This ensures all updates are non-disruptive. Additionally, we identify the need for normalizing the unlearning gradients, and also confirm the usefulness of meta-learning. We combine these insights into MUDMAN (Meta-Unlearning with Disruption Masking and Normalization) and validate its effectiveness at preventing the recovery of dangerous capabilities. MUDMAN outperforms the prior TAR method by 40%, setting a new state-of-the-art for robust unlearning.
>
---
#### [replaced 053] Asymmetric REINFORCE for off-Policy Reinforcement Learning: Balancing positive and negative rewards
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究基于REINFORCE的离策略强化学习，针对离策略方法在大语言模型对齐中性能不佳的问题，提出通过调节基线值平衡正负奖励。理论分析表明，当基线低于期望奖励时可保证策略改进，并实验证明侧重正奖励更有效。**

- **链接: [https://arxiv.org/pdf/2506.20520v2](https://arxiv.org/pdf/2506.20520v2)**

> **作者:** Charles Arnal; Gaëtan Narozniak; Vivien Cabannes; Yunhao Tang; Julia Kempe; Remi Munos
>
> **摘要:** Reinforcement learning (RL) is increasingly used to align large language models (LLMs). Off-policy methods offer greater implementation simplicity and data efficiency than on-policy techniques, but often result in suboptimal performance. In this work, we study the intermediate range of algorithms between off-policy RL and supervised fine-tuning by analyzing a simple off-policy REINFORCE algorithm, where the advantage is defined as $A=r-V$, with $r$ a reward and $V$ some tunable baseline. Intuitively, lowering $V$ emphasizes high-reward samples, while raising it penalizes low-reward ones more heavily. We first provide a theoretical analysis of this off-policy REINFORCE algorithm, showing that when the baseline $V$ lower-bounds the expected reward, the algorithm enjoys a policy improvement guarantee. Our analysis reveals that while on-policy updates can safely leverage both positive and negative signals, off-policy updates benefit from focusing more on positive rewards than on negative ones. We validate our findings experimentally in a controlled stochastic bandit setting and through fine-tuning state-of-the-art LLMs on reasoning tasks.
>
---
#### [replaced 054] Extensible Multi-Granularity Fusion Network and Transferable Curriculum Learning for Aspect-based Sentiment Analysis
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对方面级情感分析（ABSA）任务，解决现有方法模型复杂、特征融合不统一的问题。提出可扩展的多粒度融合网络（EMGF），融合多种语言特征，并引入任务特定的课程学习框架，通过从易到难训练提升模型泛化能力，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2402.07787v4](https://arxiv.org/pdf/2402.07787v4)**

> **作者:** Xinran Li; Xiaowei Zhao; Yubo Zhu; Zhiheng Zhang; Zhiqi Huang; Hongkun Song; Jinglu Hu; Xinze Che; Yifan Lyu; Yong Zhou; Xiujuan Xu
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Aspect-based Sentiment Analysis (ABSA) aims to determine sentiment polarity toward specific aspects in text. Existing methods enrich semantic and syntactic representations through external knowledge or GNNs, but the growing diversity of linguistic features increases model complexity and lacks a unified, extensible framework. We propose an Extensible Multi-Granularity Fusion Network (EMGF) that integrates dependency syntax, constituent syntax, attention-based semantics, and external knowledge graphs. EMGF employs multi-anchor triplet learning and orthogonal projection to effectively fuse multi-granularity features and strengthen their interactions without additional computational overhead. Furthermore, we introduce the first task-specific curriculum learning framework for text-only ABSA, which assigns difficulty scores using five indicators and trains the model from easy to hard to mimic human learning and improve generalization. Experiments on SemEval 2014, Twitter, and MAMS datasets show that EMGF+CL consistently outperforms state-of-the-art ABSA models.
>
---
#### [replaced 055] Benford's Curse: Tracing Digit Bias to Numerical Hallucination in LLMs
- **分类: cs.CL**

- **简介: 该论文研究大模型在数值推理中的数字偏差问题。针对其在简单数字任务中频繁出错的现象，作者发现预训练数据的数字分布偏差导致模型生成倾向符合本福德定律。通过构建均匀分布的评测基准，结合神经元分析与剪枝实验，证实了深层前馈网络中少数敏感神经元是偏差来源，并验证了其因果作用，为缓解数值幻觉提供了新路径。**

- **链接: [https://arxiv.org/pdf/2506.01734v2](https://arxiv.org/pdf/2506.01734v2)**

> **作者:** Jiandong Shao; Yao Lu; Jianfei Yang
>
> **备注:** NeurIPS 2025
>
> **摘要:** Large Language Models (LLMs) exhibit impressive performance on complex reasoning tasks, yet they frequently fail on basic numerical problems, producing incorrect outputs. Inspired by Benford's Law, a statistical pattern in which lower digits occur more frequently as leading digits, we hypothesize that the skewed digit distributions in web-collected corpora may be learned by LLMs during pretraining, leading to biased numerical generation. To investigate the hypothesis, we first examine whether digits frequencies in pretraining corpus (OLMo2) follows Benford's law. We then construct an evaluation benchmark in which the ground-truth digits are uniformly distributed within each of the seven numerical reasoning tasks. Our evaluation results demonstrate that leading open-source LLMs show a consistent pattern of digit bias that resembles Benford's law. Through logit-lens tracing and neuron-level dissection, we identify that this bias arises predominantly from a small subset of highly digit-selective feed-forward network (FFN) neurons in the deeper layers. Finally, we demonstrate that pruning these neurons mitigates imbalanced overgeneration and partially corrects erroneous outputs, providing causal evidence that fine-grained pretraining digit bias can propagate into model behavior. Our findings reveal a fundamental connection between corpus-level statistics and symbolic failure modes in LLMs, offering a new lens for diagnosing and mitigating hallucinations in numerical tasks.
>
---
#### [replaced 056] InfiMed-ORBIT: Aligning LLMs on Open-Ended Complex Tasks via Rubric-Based Incremental Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对高风险医疗对话中强化学习因反馈模糊、难以量化而失效的问题，提出ORBIT框架。通过动态构建评分标准（rubric）指导增量式强化学习，利用通用大模型作为评判器，无需任务特定微调。在医疗问答任务中显著提升模型性能，验证了该方法的有效性与通用性。**

- **链接: [https://arxiv.org/pdf/2510.15859v3](https://arxiv.org/pdf/2510.15859v3)**

> **作者:** Pengkai Wang; Linus; Pengwei Liu; Zhijie Sang; Congkai Xie; Hongxia Yang
>
> **摘要:** Reinforcement learning has powered many of the recent breakthroughs in large language models, especially for tasks where rewards can be computed automatically, such as code generation. However, these methods deteriorate in open-ended domains like medical consultation, where feedback is inherently ambiguous, highly context-dependent, and cannot be reduced to a reliable scalar signal. In such settings, RL must either rely on supervision-intensive reward models that often fail to generalize, or it falls into pathological behaviors such as reward hacking - an especially troubling risk for high-stakes medical dialogue. To address these limitations, we introduce ORBIT, an open-ended rubric-based incremental training framework for high-stakes medical dialogue. ORBIT integrates synthetic dialogue generation with dynamically constructed rubrics that serve as adaptive guides for incremental RL. Instead of relying on external medical knowledge bases or handcrafted rule sets, ORBIT uses rubric-driven feedback to steer the learning process. Its judge component can be instantiated with general-purpose instruction-following LLMs, removing the need for any task-specific fine-tuning. Applied to the Qwen3-4B-Instruct model, ORBIT raises the HealthBench-Hard score from 7.0 to 27.5 using only 2k training samples, achieving SOTA performance for models at this scale. With larger rubric datasets, ORBIT-trained models further compete with the strongest open-source baselines on HealthBench-Hard. Our analysis shows that rubric-guided RL consistently improves consultation quality across diverse medical scenarios. We also apply such rubric generation and training pipeline to InfoBench, where ORBIT enhances instruction-following performance, highlighting the generality of rubric-based feedback.
>
---
#### [replaced 057] On the Superimposed Noise Accumulation Problem in Sequential Knowledge Editing of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型的连续知识编辑任务，针对长期编辑导致的性能下降问题。提出“超叠加噪声累积”现象，源于无关知识误激活与知识冲突。为此设计DeltaEdit方法，通过动态正交约束减少知识冲突，显著提升编辑效果，较最优基线提升16.8%。**

- **链接: [https://arxiv.org/pdf/2505.07899v2](https://arxiv.org/pdf/2505.07899v2)**

> **作者:** Ding Cao; Yuchen Cai; Yuqing Huang; Xuesong He; Rongxi Guo; Guiquan Liu; Guangzhong Sun
>
> **摘要:** Sequential knowledge editing techniques aim to continuously update knowledge in large language models at low cost, preventing models from generating outdated or incorrect information. However, existing sequential editing methods suffer from a significant decline in editing success rates after long-term editing. Through theoretical analysis and experiments, our findings reveal that as the number of edits increases, the model's output increasingly deviates from the desired target, leading to a drop in editing success rates. We refer to this issue as the superimposed noise accumulation problem. Our further analysis demonstrates that the problem is related to the erroneous activation of irrelevant knowledge and conflicts between activated knowledge. Based on this analysis, a method named DeltaEdit is proposed that reduces conflicts between knowledge through dynamic orthogonal constraint strategies. Experiments show that DeltaEdit significantly reduces superimposed noise, achieving a 16.8% improvement in editing performance over the strongest baseline.
>
---
#### [replaced 058] COPO: Causal-Oriented Policy Optimization for Hallucinations of MLLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多模态大模型（MLLMs）的幻觉问题，提出因果导向的策略优化方法COPO。通过分析任务无关背景导致的虚假相关性，设计基于因果完整性的奖励机制，约束生成过程中的因果必要性与充分性，提升输出真实性。**

- **链接: [https://arxiv.org/pdf/2508.04182v2](https://arxiv.org/pdf/2508.04182v2)**

> **作者:** Peizheng Guo; Jingyao Wang; Wenwen Qiang; Jiahuan Zhou; Changwen Zheng; Gang Hua
>
> **摘要:** Despite Multimodal Large Language Models (MLLMs) having shown impressive capabilities, they may suffer from hallucinations. Empirically, we find that MLLMs attend disproportionately to task-irrelevant background regions compared with text-only LLMs, implying spurious background-answer correlations. We claim and analyze that (i) outcome-based rewards can be an important factor leading to spurious correlations, and (ii) spurious correlations can be an important factor leading to hallucinations. Based on these results, we propose Causal-Oriented Policy Optimization (COPO) to mitigate these spurious correlations, thus addressing the issue of hallucinations. It imposes token-level sufficiency and necessity constraints to measure each inference token's causal contribution, thus ensuring correct and evidence-grounded output. Specifically, we first evaluate each token's causal contribution via a newly proposed causal completeness reward. This reward is then used to construct a causally informed advantage function within the GRPO optimization framework, encouraging the model to focus on tokens that are causally sufficient and necessary for accurate generation. Experimental results across various benchmarks demonstrate the advantages of COPO.
>
---
#### [replaced 059] KeepKV: Achieving Periodic Lossless KV Cache Compression for Efficient LLM Inference
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型推理中KV缓存占用内存过大的问题，提出KeepKV方法。通过自适应合并与零推断扰动机制，实现无损压缩，有效缓解注意力不一致问题，显著降低内存使用，提升推理吞吐量，同时保持生成质量。**

- **链接: [https://arxiv.org/pdf/2504.09936v2](https://arxiv.org/pdf/2504.09936v2)**

> **作者:** Yuxuan Tian; Zihan Wang; Yebo Peng; Aomufei Yuan; Zhiming Wang; Bairen Yi; Xin Liu; Yong Cui; Tong Yang
>
> **备注:** 14 pages, 20 figures
>
> **摘要:** Efficient inference of large language models (LLMs) is hindered by an ever-growing key-value (KV) cache, making KV cache compression a critical research direction. Traditional methods selectively evict less important KV cache entries, which leads to information loss and hallucinations. Recently, merging-based strategies have been explored to retain more information by merging KV pairs that would be discarded; however, these existing approaches inevitably introduce inconsistencies in attention distributions before and after merging, causing degraded generation quality. To overcome this challenge, we propose KeepKV, a novel adaptive KV cache merging method designed to preserve performance under strict memory constraints, achieving single-step lossless compression and providing error bounds for multi-step compression. KeepKV introduces the Electoral Votes mechanism that records merging history and adaptively adjusts attention scores. Moreover, it further leverages a novel Zero Inference-Perturbation Merging method, compensating for attention loss resulting from cache merging. Extensive experiments on various benchmarks and LLM architectures demonstrate that KeepKV substantially reduces memory usage while successfully retaining essential context information, achieving over 2x inference throughput improvement and maintaining superior generation quality even with only 10% KV cache budgets.
>
---
#### [replaced 060] TrackList: Tracing Back Query Linguistic Diversity for Head and Tail Knowledge in Open Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大模型在多样语言查询下的表现差异，聚焦于头知识（高频）与尾知识（低频）对回答质量的影响。通过提出TrackList分析框架和Refomed-EN数据集，发现模型在定义类问题上表现最佳，而在举例类问题上最差，且更倾向对高频知识进行改写。**

- **链接: [https://arxiv.org/pdf/2511.21006v2](https://arxiv.org/pdf/2511.21006v2)**

> **作者:** Ioana Buhnila; Aman Sinha; Mathieu Constant
>
> **备注:** under review
>
> **摘要:** Large Language Models (LLMs) have proven efficient in giving definition-type answers to user input queries. While for humans giving various types of answers, such as examples and paraphrases, is an easy task, LLMs struggle to provide correct answers for other than definition-type queries. In this study, we evaluated this drop in performance using TrackList, a fine-grained linguistic and statistical analysis pipeline to investigate the impact of the pre-training data on LLMs answers to diverse linguistic queries. We also introduce RefoMed-EN, an English dataset consisting of 6170 human-annotated medical terms alongside their corresponding definitions, denominations, exemplifications, explanations, or paraphrases. We studied whether the high frequency of a concept (head) or low frequency (tail) impacts the language model's performance. We evaluated the quality of the LLM's output using syntactic and semantic similarity metrics, statistical correlations and embeddings. Results showed that the LLM's task performance for definition type questions is the highest, while for the exemplification type it is the lowest. Additionally, we showed that for definition-type questions, large language models are prone to paraphrase more on popular and frequent knowledge and less on tail and technical knowledge, especially in the expert texts.
>
---
#### [replaced 061] IROTE: Human-like Traits Elicitation of Large Language Model via In-Context Self-Reflective Optimization
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出IROTE方法，解决大模型仅能表面模仿人类特质的难题。通过在上下文中自动生成并优化自我反思文本，基于信息论目标增强行为与目标特质的关联性，实现无需微调的稳定、可迁移的人类特质模拟，在多任务中表现优于现有方法。**

- **链接: [https://arxiv.org/pdf/2508.08719v2](https://arxiv.org/pdf/2508.08719v2)**

> **作者:** Yuzhuo Bai; Shitong Duan; Muhua Huang; Jing Yao; Zhenghao Liu; Peng Zhang; Tun Lu; Xiaoyuan Yi; Maosong Sun; Xing Xie
>
> **备注:** This paper is accepted by AAAI 2026
>
> **摘要:** Trained on various human-authored corpora, Large Language Models (LLMs) have demonstrated a certain capability of reflecting specific human-like traits (e.g., personality or values) by prompting, benefiting applications like personalized LLMs and social simulations. However, existing methods suffer from the superficial elicitation problem: LLMs can only be steered to mimic shallow and unstable stylistic patterns, failing to embody the desired traits precisely and consistently across diverse tasks like humans. To address this challenge, we propose IROTE, a novel in-context method for stable and transferable trait elicitation. Drawing on psychological theories suggesting that traits are formed through identity-related reflection, our method automatically generates and optimizes a textual self-reflection within prompts, which comprises self-perceived experience, to stimulate LLMs' trait-driven behavior. The optimization is performed by iteratively maximizing an information-theoretic objective that enhances the connections between LLMs' behavior and the target trait, while reducing noisy redundancy in reflection without any fine-tuning, leading to evocative and compact trait reflection. Extensive experiments across three human trait systems manifest that one single IROTE-generated self-reflection can induce LLMs' stable impersonation of the target trait across diverse downstream tasks beyond simple questionnaire answering, consistently outperforming existing strong baselines.
>
---
#### [replaced 062] ROVER: Recursive Reasoning Over Videos with Vision-Language Models for Embodied Tasks
- **分类: cs.CL; cs.AI; cs.CV; cs.RO**

- **简介: 该论文针对视觉语言模型在长视频序列中推理能力弱的问题，提出ROVER框架。通过递归分解视频为短时子任务，实现局部精准推理并保留全局上下文。在机器人操作任务中验证，显著提升任务进度估计、帧级推理和视频问答性能，降低幻觉，线性降低时间复杂度。**

- **链接: [https://arxiv.org/pdf/2508.01943v2](https://arxiv.org/pdf/2508.01943v2)**

> **作者:** Philip Schroeder; Ondrej Biza; Thomas Weng; Hongyin Luo; James Glass
>
> **摘要:** Vision-language models (VLMs) have exhibited impressive capabilities across diverse image understanding tasks, but still struggle in settings that require reasoning over extended sequences of camera frames from a video. This limits their utility in embodied settings, which require reasoning over long frame sequences from a continuous stream of visual input at each moment of a task attempt. To address this limitation, we propose ROVER (Reasoning Over VidEo Recursively), a framework that enables the model to recursively decompose long-horizon video trajectories into segments corresponding to shorter subtasks within the trajectory. In doing so, ROVER facilitates more focused and accurate reasoning over temporally localized frame sequences without losing global context. We evaluate ROVER, implemented using an in-context learning approach, on diverse OpenX Embodiment videos and on a new dataset derived from RoboCasa that consists of 543 videos showing both expert and perturbed non-expert trajectories across 27 robotic manipulation tasks. ROVER outperforms strong baselines across three video reasoning tasks: task progress estimation, frame-level natural language reasoning, and video question answering. We observe that, by reducing the number of frames the model reasons over at each timestep, ROVER mitigates hallucinations, especially during unexpected or non-optimal moments of a trajectory. In addition, by enabling the implementation of a subtask-specific sliding context window, ROVER's time complexity scales linearly with video length, an asymptotic improvement over baselines. Demos, code, and data available at: https://rover-vlm.github.io
>
---
#### [replaced 063] PoETa v2: Toward More Robust Evaluation of Large Language Models in Portuguese
- **分类: cs.CL**

- **简介: 该论文针对葡萄牙语大语言模型的评估问题，提出PoETa v2基准，涵盖40余项任务，评估20多个模型。旨在系统分析计算投入与语言适配对性能的影响，揭示葡语与英语任务间的性能差距，推动葡萄牙语语言模型研究发展。**

- **链接: [https://arxiv.org/pdf/2511.17808v2](https://arxiv.org/pdf/2511.17808v2)**

> **作者:** Thales Sales Almeida; Ramon Pires; Hugo Abonizio; Rodrigo Nogueira; Hélio Pedrini
>
> **摘要:** Large Language Models (LLMs) exhibit significant variations in performance across linguistic and cultural contexts, underscoring the need for systematic evaluation in diverse languages. In this work, we present the most extensive evaluation of LLMs for the Portuguese language to date. Leveraging our newly introduced PoETa v2 benchmark -- a comprehensive suite of over 40 tasks in Portuguese -- we assess more than 20 models covering a broad spectrum of training scales and computational resources. Our study reveals how computational investment and language-specific adaptation impact performance in Portuguese, while also analyzing performance gaps in comparison to equivalent tasks in English. Through this benchmark and analysis, PoETa v2 lays the groundwork for future research on Portuguese language modeling and evaluation. The benchmark is available at https://github.com/PoETaV2/PoETaV2.
>
---
#### [replaced 064] Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh for Empowering Access to Justice
- **分类: cs.CL; cs.CY; cs.HC; cs.MA; cs.MM**

- **简介: 该论文提出Mina，一个面向孟加拉国的多语言法律智能助手，解决低收入群体获取廉价法律服务难的问题。针对现有AI工具缺乏孟加拉语支持与本地适配的缺陷，研究构建了基于多语言嵌入与RAG框架的系统，实现法律文书生成、翻译与解释，经实测在多项法律考试中表现达人类水平，成本仅为传统服务的0.12%-0.61%，显著提升司法可及性。**

- **链接: [https://arxiv.org/pdf/2511.08605v2](https://arxiv.org/pdf/2511.08605v2)**

> **作者:** Azmine Toushik Wasi; Wahid Faisal; Mst Rafia Islam
>
> **摘要:** Bangladesh's low-income population faces major barriers to affordable legal advice due to complex legal language, procedural opacity, and high costs. Existing AI legal assistants lack Bengali-language support and jurisdiction-specific adaptation, limiting their effectiveness. To address this, we developed Mina, a multilingual LLM-based legal assistant tailored for the Bangladeshi context. It employs multilingual embeddings and a RAG-based chain-of-tools framework for retrieval, reasoning, translation, and document generation, delivering context-aware legal drafts, citations, and plain-language explanations via an interactive chat interface. Evaluated by law faculty from leading Bangladeshi universities across all stages of the 2022 and 2023 Bangladesh Bar Council Exams, Mina scored 75-80% in Preliminary MCQs, Written, and simulated Viva Voce exams, matching or surpassing average human performance and demonstrating clarity, contextual understanding, and sound legal reasoning. Even under a conservative upper bound, Mina operates at just 0.12-0.61% of typical legal consultation costs in Bangladesh, yielding a 99.4-99.9\% cost reduction relative to human-provided services. These results confirm its potential as a low-cost, multilingual AI assistant that automates key legal tasks and scales access to justice, offering a real-world case study on building domain-specific, low-resource systems and addressing challenges of multilingual adaptation, efficiency, and sustainable public-service AI deployment.
>
---
#### [replaced 065] Continual Learning of Domain Knowledge from Human Feedback in Text-to-SQL
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 该论文针对文本转SQL任务中大模型缺乏数据库特定知识的问题，提出一种基于人类反馈的持续学习框架。通过结构化记忆存储并复用人类反馈中的隐性知识，使模型在迭代中提升查询准确率。实验表明，引入记忆机制的代理（如过程式代理）显著改善性能，推动更自适应、领域感知的文本转SQL系统发展。**

- **链接: [https://arxiv.org/pdf/2511.10674v2](https://arxiv.org/pdf/2511.10674v2)**

> **作者:** Thomas Cook; Kelly Patel; Sivapriya Vellaichamy; Udari Madhushani Sehwag; Saba Rahimi; Zhen Zeng; Sumitra Ganesh
>
> **备注:** 34 pages, 6 figures, 4 tables
>
> **摘要:** Large Language Models (LLMs) can generate SQL queries from natural language questions but struggle with database-specific schemas and tacit domain knowledge. We introduce a framework for continual learning from human feedback in text-to-SQL, where a learning agent receives natural language feedback to refine queries and distills the revealed knowledge for reuse on future tasks. This distilled knowledge is stored in a structured memory, enabling the agent to improve execution accuracy over time. We design and evaluate multiple variations of a learning agent architecture that vary in how they capture and retrieve past experiences. Experiments on the BIRD benchmark Dev set show that memory-augmented agents, particularly the Procedural Agent, achieve significant accuracy gains and error reduction by leveraging human-in-the-loop feedback. Our results highlight the importance of transforming tacit human expertise into reusable knowledge, paving the way for more adaptive, domain-aware text-to-SQL systems that continually learn from a human-in-the-loop.
>
---
#### [replaced 066] ChiKhaPo: A Large-Scale Multilingual Benchmark for Evaluating Lexical Comprehension and Generation in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出ChiKhaPo，一个覆盖2700+语言的大规模多语言基准，聚焦低资源语言的词汇理解与生成能力评估。针对现有基准语言覆盖不足、偏重高阶任务的问题，构建8个子任务，揭示主流大模型在多语言基础语言能力上的短板，推动更全面的多语言模型评测。**

- **链接: [https://arxiv.org/pdf/2510.16928v2](https://arxiv.org/pdf/2510.16928v2)**

> **作者:** Emily Chang; Niyati Bafna
>
> **摘要:** Existing benchmarks for large language models (LLMs) are largely restricted to high- or mid-resource languages, and often evaluate performance on higher-order tasks in reasoning and generation. However, plenty of evidence points to the fact that LLMs lack basic linguistic competence in the vast majority of the world's 3800+ written languages. We introduce ChiKhaPo, consisting of 8 subtasks of varying difficulty designed to evaluate the lexical comprehension and generation abilities of generative models. ChiKhaPo draws on existing lexicons, monolingual data, and bitext, and provides coverage for 2700+ languages for 2 subtasks, surpassing any existing benchmark in terms of language coverage. We further show that 6 SOTA models struggle on our benchmark, and discuss the factors contributing to performance scores, including language family, language resourcedness, task, and comprehension versus generation directions. With ChiKhaPo, we hope to enable and encourage the massively multilingual benchmarking of LLMs.
>
---
#### [replaced 067] MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search
- **分类: cs.DB; cs.AI; cs.CL; cs.PL**

- **简介: 该论文聚焦于轻量级大模型的文本转SQL任务。针对小模型在复杂查询、冗余链接和语法正确性上的不足，提出MCTS-SQL框架，通过蒙特卡洛树搜索多步迭代优化，并引入词元级前缀缓存加速推理，显著提升生成质量与效率，在SPIDER和BIRD上超越ChatGPT-3.5，验证了小模型在实际场景中的可行性。**

- **链接: [https://arxiv.org/pdf/2501.16607v3](https://arxiv.org/pdf/2501.16607v3)**

> **作者:** Shuozhi Yuan; Limin Chen; Miaomiao Yuan; Zhao Jin
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Text-to-SQL is a fundamental yet challenging task in the NLP area, aiming at translating natural language questions into SQL queries. While recent advances in large language models have greatly improved performance, most existing approaches depend on models with tens of billions of parameters or costly APIs, limiting their applicability in resource-constrained environments. For real world, especially on edge devices, it is crucial for Text-to-SQL to ensure cost-effectiveness. Therefore, enabling the light-weight models for Text-to-SQL is of great practical significance. However, smaller LLMs often struggle with complicated user instruction, redundant schema linking or syntax correctness. To address these challenges, we propose MCTS-SQL, a novel framework that uses Monte Carlo Tree Search to guide SQL generation through multi-step refinement. Since the light-weight models' weak performance of single-shot prediction, we generate better results through several trials with feedback. However, directly applying MCTS-based methods inevitably leads to significant time and computational overhead. Driven by this issue, we propose a token-level prefix-cache mechanism that stores prior information during iterations, effectively improved the execution speed. Experiments results on the SPIDER and BIRD benchmarks demonstrate the effectiveness of our approach. Using a small open-source Qwen2.5-Coder-1.5B, our method outperforms ChatGPT-3.5. When leveraging a more powerful model Gemini 2.5 to explore the performance upper bound, we achieved results competitive with the SOTA. Our findings demonstrate that even small models can be effectively deployed in practical Text-to-SQL systems with the right strategy.
>
---
#### [replaced 068] LongCat-Flash-Omni Technical Report
- **分类: cs.MM; cs.AI; cs.CL; cs.DC; cs.LG; cs.SD**

- **简介: 该论文提出LongCat-Flash-Omni，一个560B参数的开源多模态模型，解决大规模多模态训练与实时交互效率问题。通过渐进式训练和模态解耦并行策略，实现高效跨模态理解与生成，在文本、图像、视频、音频等任务上达领先性能。**

- **链接: [https://arxiv.org/pdf/2511.00279v2](https://arxiv.org/pdf/2511.00279v2)**

> **作者:** Meituan LongCat Team; Bairui Wang; Bayan; Bin Xiao; Bo Zhang; Bolin Rong; Borun Chen; Chang Wan; Chao Zhang; Chen Huang; Chen Chen; Chen Chen; Chengxu Yang; Chengzuo Yang; Cong Han; Dandan Peng; Delian Ruan; Detai Xin; Disong Wang; Dongchao Yang; Fanfan Liu; Fengjiao Chen; Fengyu Yang; Gan Dong; Gang Huang; Gang Xu; Guanglu Wan; Guoqiang Tan; Guoqiao Yu; Haibo Qiu; Hao Lu; Hongbo Liu; Hongyu Xiang; Jiaheng Wu; Jian Yang; Jiaxing Liu; Jing Huang; Jingang Wang; Jinrui Ding; Juchao Jiang; Jun Kuang; Jun Wang; Junhui Mei; Ke Ding; Kefeng Zhang; Lei Chen; Liang Shi; Limeng Qiao; Liming Zheng; Lin Ma; Liuyang Guo; Liya Ma; Luying Sun; Man Gao; Mengshen Zhu; Miao Cao; Minliang Lin; Nuo Xu; Peng Shi; Qi Zhang; Qian Fang; Qian Wang; Qian Yang; Quanxiu Wang; Rongxiang Weng; Rongxin Guo; Ruoxuan Liang; Senbin Yang; Shanbo Xu; Shanglin Lei; Shengze Ye; Shimin Chen; Shuaiqi Chen; Shujie Hu; Shuo Li; Siqi Yang; Siyu Xu; Siyu Ren; Song Li; Songxiang Liu; Tianhao Bai; Tianye Dai; Wei Hong; Wei Wang; Weixiao Zhao; Wengang Cao; Wenlong Zhu; Wenlong He; Xi Su; Xi Nan; Xiaohan Zhao; Xiaohao Wang; Xiaoyu Zhao; Xiaoyu Wang; Xiaoyu Li; Xin Pan; Xin Chen; Xiusong Sun; Xu Xiang; Xudong Xing; Xuezhi Cao; Xunliang Cai; Yang Yang; Yanli Tan; Yao Yao; Yerui Sun; Yi Chen; Yifan Lu; Yin Gong; Yining Zhang; Yitian Chen; Yiyang Gan; Yuchen Tang; Yuchen Xie; Yueqian Wang; Yuewen Zheng; Yufei Zhang; Yufeng Zhong; Yulei Qian; Yuqi Peng; Yuqian Li; Yuwei Jiang; Zeyang Hu; Zheng Zhang; Zhengkun Tian; Zhiqing Hong; Zhixiong Zeng; Zhuqi Mi; Ziran Li; Ziwen Wang; Ziyi Zhao; Ziyuan Zhuang; Zizhe Zhao
>
> **摘要:** We introduce LongCat-Flash-Omni, a state-of-the-art open-source omni-modal model with 560 billion parameters, excelling at real-time audio-visual interaction. By adopting a curriculum-inspired progressive training strategy that transitions from simpler to increasingly complex modality sequence modeling tasks, LongCat-Flash-Omni attains comprehensive multimodal capabilities while maintaining strong unimodal capability. Building upon LongCat-Flash, which adopts a high-performance Shortcut-connected Mixture-of-Experts (MoE) architecture with zero-computation experts, LongCat-Flash-Omni integrates efficient multimodal perception and speech reconstruction modules. Despite its immense size of 560B parameters (with 27B activated), LongCat-Flash-Omni achieves low-latency real-time audio-visual interaction. For training infrastructure, we developed a modality-decoupled parallelism scheme specifically designed to manage the data and model heterogeneity inherent in large-scale multimodal training. This innovative approach demonstrates exceptional efficiency by sustaining over 90% of the throughput achieved by text-only training. Extensive evaluations show that LongCat-Flash-Omni achieves state-of-the-art performance on omni-modal benchmarks among open-source models. Furthermore, it delivers highly competitive results across a wide range of modality-specific tasks, including text, image, and video understanding, as well as audio understanding and generation. We provide a comprehensive overview of the model architecture design, training procedures, and data strategies, and open-source the model to foster future research and development in the community.
>
---
#### [replaced 069] REFLEX: Self-Refining Explainable Fact-Checking via Disentangling Truth into Style and Substance
- **分类: cs.CL**

- **简介: 该论文针对社交媒体虚假信息问题，提出REFLEX框架，解决现有大模型事实核查中依赖外部知识导致的延迟与幻觉问题。通过角色扮演对话与激活向量解耦，将真相分离为风格与实质，实现自精炼、可解释的事实核查，仅用465样本即达顶尖性能。**

- **链接: [https://arxiv.org/pdf/2511.20233v2](https://arxiv.org/pdf/2511.20233v2)**

> **作者:** Chuyi Kong; Gao Wei; Jing Ma; Hongzhan Lin; Yaxin Fan
>
> **摘要:** The prevalence of misinformation on social media threatens public trust, demanding automated fact-checking systems that provide accurate verdicts with interpretable explanations. However, existing large language model-based (LLM-based) approaches often rely heavily on external knowledge sources, introducing substantial latency and even hallucinations that undermine reliability, interpretability, and responsiveness, which is crucial for real-time use. To address these challenges, we propose REason-guided Fact-checking with Latent EXplanations REFLEX paradigm, a plug-and-play, self-refining paradigm that leverages the internal knowledge in backbone model to improve both verdict accuracy and explanation quality. REFLEX reformulates fact-checking as a role-play dialogue and jointly trains verdict prediction and explanation generation. It adaptively extracts contrastive activation pairs between the backbone model and its fine-tuned variant to construct steering vectors that disentangle truth into style and substance naturally. These activation-level signals guide inference and suppress noisy explanations, enabling more faithful and efficient reasoning. Experiments on real-world datasets show that REFLEX outperforms previous methods that steer toward a single truth direction and underscores the challenge traditional approaches face when handling the subtle, human-unknown truth in fact-checking tasks. Remarkably, with only 465 self-refined training samples, RELFEX achieves state-of-the-art performance. Furthermore, models trained with explanatory objectives can effectively guide those without them, yielding up to a 7.57% improvement, highlighting that internal explanation signals play a dual role in both interpreting and enhancing factual reasoning.
>
---
