# 自然语言处理 cs.CL

- **最新发布 79 篇**

- **更新 86 篇**

## 最新发布

#### [new 001] A Good Plan is Hard to Find: Aligning Models with Preferences is Misaligned with What Helps Users
- **分类: cs.CL**

- **简介: 该论文研究了大语言模型（LLM）生成计划与用户偏好的对齐问题。任务是评估计划的实用性，而非表面偏好。通过实验发现，用户偏好与实际帮助性不一致，需基于真实交互反馈优化模型，以提升LLM在复杂任务中的实用性。**

- **链接: [http://arxiv.org/pdf/2509.18632v1](http://arxiv.org/pdf/2509.18632v1)**

> **作者:** Nishant Balepur; Matthew Shu; Yoo Yeon Sung; Seraphina Goldfarb-Tarrant; Shi Feng; Fumeng Yang; Rachel Rudinger; Jordan Lee Boyd-Graber
>
> **备注:** EMNLP 2025
>
> **摘要:** To assist users in complex tasks, LLMs generate plans: step-by-step instructions towards a goal. While alignment methods aim to ensure LLM plans are helpful, they train (RLHF) or evaluate (ChatbotArena) on what users prefer, assuming this reflects what helps them. We test this with Planorama: an interface where 126 users answer 300 multi-step questions with LLM plans. We get 4388 plan executions and 5584 comparisons to measure plan helpfulness (QA success) and user preferences on plans, and recreate the setup in agents and reward models to see if they simulate or prefer what helps users. We expose: 1) user/model preferences and agent success do not accurately predict which plans help users, so common alignment feedback can misalign with helpfulness; 2) this gap is not due to user-specific preferences, as users are similarly successful when using plans they prefer/disprefer; 3) surface-level cues like brevity and question similarity strongly link to preferences, but such biases fail to predict helpfulness. In all, we argue aligning helpful LLMs needs feedback from real user interactions, not just preferences of what looks helpful, so we discuss the plan NLP researchers can execute to solve this problem.
>
---
#### [new 002] Online Process Reward Leanring for Agentic Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出在线过程奖励学习（OPRL），用于代理强化学习，解决稀疏奖励下的信用分配问题。通过联合优化隐式过程奖励模型与策略，将轨迹偏好转化为步骤奖励，提升训练稳定性与样本效率，在多个任务中取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.19199v1](http://arxiv.org/pdf/2509.19199v1)**

> **作者:** Xiaoqian Liu; Ke Wang; Yuchuan Wu; Fei Huang; Yongbin Li; Junge Zhang; Jianbin Jiao
>
> **备注:** preprint
>
> **摘要:** Large language models (LLMs) are increasingly trained with reinforcement learning (RL) as autonomous agents that reason and act over long horizons in interactive environments. However, sparse and sometimes unverifiable rewards make temporal credit assignment extremely challenging. Recent work attempts to integrate process supervision into agent learning but suffers from biased annotation, reward hacking, high-variance from overly fine-grained signals or failtures when state overlap is rare. We therefore introduce Online Process Reward Learning (OPRL), a general credit-assignment strategy for agentic RL that integrates seamlessly with standard on-policy algorithms without relying on additional rollouts or explicit step labels. In OPRL, we optimize an implicit process reward model (PRM) alternately with the agent's policy to transform trajectory preferences into implicit step rewards through a trajectory-based DPO objective. These step rewards are then used to compute step-level advantages, which are combined with episode-level advantages from outcome rewards for policy update, creating a self-reinforcing loop. Theoretical findings guarantee that the learned step rewards are consistent with trajectory preferences and act as potential-based shaping rewards, providing bounded gradients to stabilize training. Empirically, we evaluate OPRL on three distinct agent benmarks, including WebShop and VisualSokoban, as well as open-ended social interactions with unverfiable rewards in SOTOPIA. Crucially, OPRL shows superior performance over frontier LLMs and strong RL baselines across domains, achieving state-of-the-art results with higher sample-efficiency and lower variance during training. Further analysis also demonstrates the efficient exploration by OPRL using fewer actions, underscoring its potential for agentic learning in real-world scenarios.
>
---
#### [new 003] Anecdoctoring: Automated Red-Teaming Across Language and Place
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出“anecdoctoring”方法，用于跨语言和地域的自动红队测试，旨在检测生成式AI的虚假信息风险。针对现有数据以英语和美国为中心的问题，收集多语言虚假信息并构建知识图谱增强攻击模型，提升攻击成功率与可解释性。**

- **链接: [http://arxiv.org/pdf/2509.19143v1](http://arxiv.org/pdf/2509.19143v1)**

> **作者:** Alejandro Cuevas; Saloni Dash; Bharat Kumar Nayak; Dan Vann; Madeleine I. G. Daepp
>
> **备注:** To be published in EMNLP 2025
>
> **摘要:** Disinformation is among the top risks of generative artificial intelligence (AI) misuse. Global adoption of generative AI necessitates red-teaming evaluations (i.e., systematic adversarial probing) that are robust across diverse languages and cultures, but red-teaming datasets are commonly US- and English-centric. To address this gap, we propose "anecdoctoring", a novel red-teaming approach that automatically generates adversarial prompts across languages and cultures. We collect misinformation claims from fact-checking websites in three languages (English, Spanish, and Hindi) and two geographies (US and India). We then cluster individual claims into broader narratives and characterize the resulting clusters with knowledge graphs, with which we augment an attacker LLM. Our method produces higher attack success rates and offers interpretability benefits relative to few-shot prompting. Results underscore the need for disinformation mitigations that scale globally and are grounded in real-world adversarial misuse.
>
---
#### [new 004] When Long Helps Short: How Context Length in Supervised Fine-tuning Affects Behavior of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究监督微调中数据长度对大语言模型行为的影响，发现长上下文微调能提升短任务性能。通过分析MHA和FFN的作用机制，揭示了知识偏好偏差，并提出混合训练可缓解此问题，为模型微调提供指导。**

- **链接: [http://arxiv.org/pdf/2509.18762v1](http://arxiv.org/pdf/2509.18762v1)**

> **作者:** Yingming Zheng; Hanqi Li; Kai Yu; Lu Chen
>
> **摘要:** Large language models (LLMs) have achieved impressive performance across natural language processing (NLP) tasks. As real-world applications increasingly demand longer context windows, continued pretraining and supervised fine-tuning (SFT) on long-context data has become a common approach. While the effects of data length in continued pretraining have been extensively studied, their implications for SFT remain unclear. In this work, we systematically investigate how SFT data length influences LLM behavior on short-context tasks. Counterintuitively, we find that long-context SFT improves short-context performance, contrary to the commonly observed degradation from long-context pretraining. To uncover the underlying mechanisms of this phenomenon, we first decouple and analyze two key components, Multi-Head Attention (MHA) and Feed-Forward Network (FFN), and show that both independently benefit from long-context SFT. We further study their interaction and reveal a knowledge preference bias: long-context SFT promotes contextual knowledge, while short-context SFT favors parametric knowledge, making exclusive reliance on long-context SFT suboptimal. Finally, we demonstrate that hybrid training mitigates this bias, offering explainable guidance for fine-tuning LLMs.
>
---
#### [new 005] WolBanking77: Wolof Banking Speech Intent Classification Dataset
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出了WolBanking77，一个用于沃尔夫语（Wolof）银行领域意图分类的数据集，旨在解决低资源语言在NLP和ASR任务中的研究缺口。数据集包含9,791条文本和4小时语音，并提供了多种基线模型的实验结果与分析。**

- **链接: [http://arxiv.org/pdf/2509.19271v1](http://arxiv.org/pdf/2509.19271v1)**

> **作者:** Abdou Karim Kandji; Frédéric Precioso; Cheikh Ba; Samba Ndiaye; Augustin Ndione
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Intent classification models have made a lot of progress in recent years. However, previous studies primarily focus on high-resource languages datasets, which results in a gap for low-resource languages and for regions with a high rate of illiterate people where languages are more spoken than read or written. This is the case in Senegal, for example, where Wolof is spoken by around 90\% of the population, with an illiteracy rate of 42\% for the country. Wolof is actually spoken by more than 10 million people in West African region. To tackle such limitations, we release a Wolof Intent Classification Dataset (WolBanking77), for academic research in intent classification. WolBanking77 currently contains 9,791 text sentences in the banking domain and more than 4 hours of spoken sentences. Experiments on various baselines are conducted in this work, including text and voice state-of-the-art models. The results are very promising on this current dataset. This paper also provides detailed analyses of the contents of the data. We report baseline f1-score and word error rate metrics respectively on NLP and ASR models trained on WolBanking77 dataset and also comparisons between models. We plan to share and conduct dataset maintenance, updates and to release open-source code.
>
---
#### [new 006] Multi-Hierarchical Feature Detection for Large Language Model Generated Text
- **分类: cs.CL; I.2.7; I.2.1**

- **简介: 该论文研究AI文本检测任务，探讨多特征融合是否优于单一模型。提出MHFD方法，结合语义、句法和统计特征，实验发现性能提升有限（0.4-2.6%），但计算开销大，表明现代神经模型可能已高效捕捉关键信号。**

- **链接: [http://arxiv.org/pdf/2509.18862v1](http://arxiv.org/pdf/2509.18862v1)**

> **作者:** Luyan Zhang; Xinyu Xie
>
> **备注:** 9 pages, 6 tables, empirical study on multi-feature AI text detection
>
> **摘要:** With the rapid advancement of large language model technology, there is growing interest in whether multi-feature approaches can significantly improve AI text detection beyond what single neural models achieve. While intuition suggests that combining semantic, syntactic, and statistical features should provide complementary signals, this assumption has not been rigorously tested with modern LLM-generated text. This paper provides a systematic empirical investigation of multi-hierarchical feature integration for AI text detection, specifically testing whether the computational overhead of combining multiple feature types is justified by performance gains. We implement MHFD (Multi-Hierarchical Feature Detection), integrating DeBERTa-based semantic analysis, syntactic parsing, and statistical probability features through adaptive fusion. Our investigation reveals important negative results: despite theoretical expectations, multi-feature integration provides minimal benefits (0.4-0.5% improvement) while incurring substantial computational costs (4.2x overhead), suggesting that modern neural language models may already capture most relevant detection signals efficiently. Experimental results on multiple benchmark datasets demonstrate that the MHFD method achieves 89.7% accuracy in in-domain detection and maintains 84.2% stable performance in cross-domain detection, showing modest improvements of 0.4-2.6% over existing methods.
>
---
#### [new 007] Are Smaller Open-Weight LLMs Closing the Gap to Proprietary Models for Biomedical Question Answering?
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文研究了小型开源大模型在生物医学问答任务中的表现，参与BioASQ挑战赛13B阶段。通过对比多种开源模型与GPT-4、Claude等闭源模型，并采用检索、上下文学习和集成方法，发现开源模型性能可媲美甚至超越闭源模型。**

- **链接: [http://arxiv.org/pdf/2509.18843v1](http://arxiv.org/pdf/2509.18843v1)**

> **作者:** Damian Stachura; Joanna Konieczna; Artur Nowak
>
> **备注:** CLEF 2025 Working Notes, 9-12 September 2025, Madrid, Spain
>
> **摘要:** Open-weight versions of large language models (LLMs) are rapidly advancing, with state-of-the-art models like DeepSeek-V3 now performing comparably to proprietary LLMs. This progression raises the question of whether small open-weight LLMs are capable of effectively replacing larger closed-source models. We are particularly interested in the context of biomedical question-answering, a domain we explored by participating in Task 13B Phase B of the BioASQ challenge. In this work, we compare several open-weight models against top-performing systems such as GPT-4o, GPT-4.1, Claude 3.5 Sonnet, and Claude 3.7 Sonnet. To enhance question answering capabilities, we use various techniques including retrieving the most relevant snippets based on embedding distance, in-context learning, and structured outputs. For certain submissions, we utilize ensemble approaches to leverage the diverse outputs generated by different models for exact-answer questions. Our results demonstrate that open-weight LLMs are comparable to proprietary ones. In some instances, open-weight LLMs even surpassed their closed counterparts, particularly when ensembling strategies were applied. All code is publicly available at https://github.com/evidenceprime/BioASQ-13b.
>
---
#### [new 008] SIRAG: Towards Stable and Interpretable RAG with A Process-Supervised Multi-Agent Framework
- **分类: cs.CL**

- **简介: 该论文针对RAG中检索与生成协调不足的问题，提出SIRAG框架。通过引入决策者和知识选择者两个轻量级代理，并结合过程监督和PPO训练，提升检索生成的稳定性和可解释性，适用于实际RAG应用。**

- **链接: [http://arxiv.org/pdf/2509.18167v1](http://arxiv.org/pdf/2509.18167v1)**

> **作者:** Junlin Wang; Zehao Wu; Shaowei Lu; Yanlan Li; Xinghao Huang
>
> **备注:** 5 pages,2 figures, IRAC under review
>
> **摘要:** Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to access external knowledge sources, but the effectiveness of RAG relies on the coordination between the retriever and the generator. Since these components are developed independently, their interaction is often suboptimal: the retriever may return irrelevant or redundant documents, while the generator may fail to fully leverage retrieved evidence. In this work, we propose a process-supervised multi-agent framework to bridge the gap between retriever and generator. The framework introduces two lightweight agents: a Decision Maker, which determines when to continue retrieval or stop for answer generation, and a Knowledge Selector, which filters retrieved documents to retain only the most useful evidence. To provide fine-grained supervision, we employ an LLM-as-a-Judge that evaluates each intermediate action with process-level rewards, ensuring more accurate credit assignment than relying solely on final answer correctness. We further adopt a tree-structured rollout strategy to explore diverse reasoning paths, and train both agents with Proximal Policy Optimization (PPO) in an end-to-end manner. Experiments on single-hop and multi-hop question answering benchmarks show that our approach achieves higher accuracy, more stable convergence, and produces more interpretable reasoning trajectories compared with standard RAG baselines. Importantly, the proposed framework is modular and plug-and-play, requiring no modification to the retriever or generator, making it practical for real-world RAG applications.
>
---
#### [new 009] Exploiting Tree Structure for Credit Assignment in RL Training of LLMs
- **分类: cs.CL**

- **简介: 该论文针对强化学习训练大语言模型时的信用分配问题，提出TEMPO算法。通过构建前缀树结构（P2T），在无需价值网络的情况下实现分支点的精确信用分配，优于PPO和GRPO，在数学与医疗QA任务中表现更优。**

- **链接: [http://arxiv.org/pdf/2509.18314v1](http://arxiv.org/pdf/2509.18314v1)**

> **作者:** Hieu Tran; Zonghai Yao; Hong Yu
>
> **备注:** 15 pages
>
> **摘要:** Reinforcement learning improves LLM reasoning, yet sparse delayed reward over long sequences makes token-level credit assignment the key bottleneck. We study the verifiable-reward setting, where the final answer is checkable and multiple responses can be drawn per prompt. Reasoning tasks in math and medical QA align with this setup, where only a few decision tokens significantly impact the outcome. PPO offers token-level advantages with a learned value model, but it is complex to train both the actor and critic models simultaneously, and it is not easily generalizable, as the token-level values from the critic model can make training prone to overfitting. GRPO is critic-free and supports verifiable rewards, but spreads a single sequence-level return across tokens and ignores branching. We introduce \textbf{Prefix-to-Tree (P2T)}, a simple procedure that converts a group of responses into a prefix tree and computes \emph{nonparametric} prefix values \(V(s)\) by aggregating descendant outcomes. Built on P2T, we propose \textbf{TEMPO} (\emph{\textbf{T}ree-\textbf{E}stimated \textbf{M}ean Prefix Value for \textbf{P}olicy \textbf{O}ptimization}), a critic-free algorithm that augments the group-relative outcome signal of GRPO with \emph{branch-gated} temporal-difference corrections derived from the tree. At non-branch tokens, the temporal-difference (TD) term is zero, so TEMPO reduces to GRPO; at branching tokens, it supplies precise token-level credit without a learned value network or extra judges/teachers. On Qwen3-1.7B/4B, TEMPO outperforms PPO and GRPO on in-distribution (MATH, MedQA) and out-of-distribution (GSM-HARD, AMC23, MedMCQA, MMLU-Medical) benchmarks, and reaches higher validation accuracy with roughly the same wall-clock time.
>
---
#### [new 010] CogniLoad: A Synthetic Natural Language Reasoning Benchmark With Tunable Length, Intrinsic Difficulty, and Distractor Density
- **分类: cs.CL; cs.AI; cs.LG; 68T50 (Primary) 68T07, 68T05, 68T20, 68T27 (Secondary); I.2.7; I.2.6; I.2.4; I.2.8**

- **简介: 该论文提出CogniLoad，一个基于认知负荷理论的合成自然语言推理基准，用于精确分析大模型在长上下文推理中的表现。通过可调参数（任务难度、干扰项密度和长度），揭示模型性能限制，指导模型优化。**

- **链接: [http://arxiv.org/pdf/2509.18458v1](http://arxiv.org/pdf/2509.18458v1)**

> **作者:** Daniel Kaiser; Arnoldo Frigessi; Ali Ramezani-Kebrya; Benjamin Ricaud
>
> **备注:** 29 pages (main: 12 + supplemental material: 17), 6 figures, 4 tables, Code: https://github.com/kaiserdan/cogniload, Data: https://huggingface.co/datasets/cogniloadteam/cogniload
>
> **摘要:** Current benchmarks for long-context reasoning in Large Language Models (LLMs) often blur critical factors like intrinsic task complexity, distractor interference, and task length. To enable more precise failure analysis, we introduce CogniLoad, a novel synthetic benchmark grounded in Cognitive Load Theory (CLT). CogniLoad generates natural-language logic puzzles with independently tunable parameters that reflect CLT's core dimensions: intrinsic difficulty ($d$) controls intrinsic load; distractor-to-signal ratio ($\rho$) regulates extraneous load; and task length ($N$) serves as an operational proxy for conditions demanding germane load. Evaluating 22 SotA reasoning LLMs, CogniLoad reveals distinct performance sensitivities, identifying task length as a dominant constraint and uncovering varied tolerances to intrinsic complexity and U-shaped responses to distractor ratios. By offering systematic, factorial control over these cognitive load dimensions, CogniLoad provides a reproducible, scalable, and diagnostically rich tool for dissecting LLM reasoning limitations and guiding future model development.
>
---
#### [new 011] Measuring AI "Slop" in Text
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决AI生成文本中“slop”（低质量内容）缺乏明确定义和衡量标准的问题。通过专家访谈构建分类体系，并提出可解释的评估维度，利用标注数据验证其与连贯性、相关性等特征的相关性。**

- **链接: [http://arxiv.org/pdf/2509.19163v1](http://arxiv.org/pdf/2509.19163v1)**

> **作者:** Chantal Shaib; Tuhin Chakrabarty; Diego Garcia-Olano; Byron C. Wallace
>
> **摘要:** AI "slop" is an increasingly popular term used to describe low-quality AI-generated text, but there is currently no agreed upon definition of this term nor a means to measure its occurrence. In this work, we develop a taxonomy of "slop" through interviews with experts in NLP, writing, and philosophy, and propose a set of interpretable dimensions for its assessment in text. Through span-level annotation, we find that binary "slop" judgments are (somewhat) subjective, but such determinations nonetheless correlate with latent dimensions such as coherence and relevance. Our framework can be used to evaluate AI-generated text in both detection and binary preference tasks, potentially offering new insights into the linguistic and stylistic factors that contribute to quality judgments.
>
---
#### [new 012] Charting a Decade of Computational Linguistics in Italy: The CLiC-it Corpus
- **分类: cs.CL; 68T50; I.2.7**

- **简介: 该论文属于研究趋势分析任务，旨在梳理过去十年意大利计算语言学与NLP的发展。通过构建CLiC-it会议的论文语料库，分析元数据和内容，揭示研究主题、作者背景等变化，为未来研究提供参考。**

- **链接: [http://arxiv.org/pdf/2509.19033v1](http://arxiv.org/pdf/2509.19033v1)**

> **作者:** Chiara Alzetta; Serena Auriemma; Alessandro Bondielli; Luca Dini; Chiara Fazzone; Alessio Miaschi; Martina Miliani; Marta Sartor
>
> **备注:** Submitted to IJCoL
>
> **摘要:** Over the past decade, Computational Linguistics (CL) and Natural Language Processing (NLP) have evolved rapidly, especially with the advent of Transformer-based Large Language Models (LLMs). This shift has transformed research goals and priorities, from Lexical and Semantic Resources to Language Modelling and Multimodality. In this study, we track the research trends of the Italian CL and NLP community through an analysis of the contributions to CLiC-it, arguably the leading Italian conference in the field. We compile the proceedings from the first 10 editions of the CLiC-it conference (from 2014 to 2024) into the CLiC-it Corpus, providing a comprehensive analysis of both its metadata, including author provenance, gender, affiliations, and more, as well as the content of the papers themselves, which address various topics. Our goal is to provide the Italian and international research communities with valuable insights into emerging trends and key developments over time, supporting informed decisions and future directions in the field.
>
---
#### [new 013] CompLLM: Compression for Long Context Q&A
- **分类: cs.CL**

- **简介: 该论文提出CompLLM，一种针对长上下文问答任务的高效软压缩方法。为解决LLM处理长文本时计算复杂度高、无法复用的问题，CompLLM将上下文分段独立压缩，实现线性扩展、模型泛化与缓存复用，显著提升推理效率并降低内存消耗。**

- **链接: [http://arxiv.org/pdf/2509.19228v1](http://arxiv.org/pdf/2509.19228v1)**

> **作者:** Gabriele Berton; Jayakrishnan Unnikrishnan; Son Tran; Mubarak Shah
>
> **摘要:** Large Language Models (LLMs) face significant computational challenges when processing long contexts due to the quadratic complexity of self-attention. While soft context compression methods, which map input text to smaller latent representations, have shown promise, their real-world adoption is limited. Existing techniques typically compress the context as a single unit, which leads to quadratic compression complexity and an inability to reuse computations across queries with overlapping contexts. In this work, we introduce CompLLM, a soft compression technique designed for practical deployment. Instead of processing the context holistically, CompLLM divides it into segments and compresses each one independently. This simple design choice yields three critical properties: efficiency, as the compression step scales linearly with the context length; scalability, enabling models trained on short sequences (e.g., 1k tokens) to generalize to contexts of 100k tokens; and reusability, allowing compressed segments to be cached and reused across different queries. Our experiments show that with a 2x compression rate, at high context lengths CompLLM speeds up Time To First Token (TTFT) by up to 4x and reduces the KV cache size by 50%. Furthermore, CompLLM achieves performance comparable to that obtained with the uncompressed context, and even surpasses it on very long sequences, demonstrating its effectiveness and practical utility.
>
---
#### [new 014] CCQA: Generating Question from Solution Can Improve Inference-Time Reasoning in SLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CCQA方法，用于提升小型语言模型（SLMs）在推理时的数学与常识推理能力。通过生成并评估问题-答案对的相似性，选择最优解。实验表明其优于现有方法，为高效推理提供新基线。**

- **链接: [http://arxiv.org/pdf/2509.18536v1](http://arxiv.org/pdf/2509.18536v1)**

> **作者:** Jin Young Kim; Ji Won Yoon
>
> **备注:** Published as a main conference paper at EMNLP 2025
>
> **摘要:** Recently, inference-time reasoning strategies have further improved the accuracy of large language models (LLMs), but their effectiveness on smaller models remains unclear. Based on the observation that conventional approaches often fail to improve performance in this context, we propose \textbf{C}ycle-\textbf{C}onsistency in \textbf{Q}uestion \textbf{A}nswering (CCQA), a novel reasoning method that can be effectively applied to SLMs. Inspired by cycle consistency, CCQA generates a question from each reasoning path and answer, evaluates each by its similarity to the original question, and then selects the candidate solution with the highest similarity score as the final response. Since conventional SLMs struggle to generate accurate questions from their own reasoning paths and answers, we employ a lightweight Flan-T5 model specialized for question generation to support this process efficiently. From the experimental results, it is verified that CCQA consistently outperforms existing state-of-the-art (SOTA) methods across eight models on mathematical and commonsense reasoning benchmarks. Furthermore, our method establishes a new practical baseline for efficient reasoning in SLMs. Source code can be found at https://github.com/scai-research/ccqa_official.
>
---
#### [new 015] ZERA: Zero-init Instruction Evolving Refinement Agent - From Zero Instructions to Structured Prompts via Principle-based Optimization
- **分类: cs.CL; cs.LG; I.2.7**

- **简介: 该论文提出ZERA框架，用于自动优化大语言模型的提示（Prompt），通过联合优化系统和用户提示，提升任务表现。它解决了传统方法依赖无结构反馈、样本量大和迭代周期长的问题，实现了高效、低成本的提示优化。**

- **链接: [http://arxiv.org/pdf/2509.18158v1](http://arxiv.org/pdf/2509.18158v1)**

> **作者:** Seungyoun Yi; Minsoo Khang; Sungrae Park
>
> **备注:** 9 pages, 4 figures. To appear in EMNLP 2025 Main Conference (Oral Presentation)
>
> **摘要:** Automatic Prompt Optimization (APO) improves large language model (LLM) performance by refining prompts for specific tasks. However, prior APO methods typically focus only on user prompts, rely on unstructured feedback, and require large sample sizes and long iteration cycles-making them costly and brittle. We propose ZERA (Zero-init Instruction Evolving Refinement Agent), a novel framework that jointly optimizes both system and user prompts through principled, low-overhead refinement. ZERA scores prompts using eight generalizable criteria with automatically inferred weights, and revises prompts based on these structured critiques. This enables fast convergence to high-quality prompts using minimal examples and short iteration cycles. We evaluate ZERA across five LLMs and nine diverse datasets spanning reasoning, summarization, and code generation tasks. Experimental results demonstrate consistent improvements over strong baselines. Further ablation studies highlight the contribution of each component to more effective prompt construction. Our implementation including all prompts is publicly available at https://github.com/younatics/zera-agent.
>
---
#### [new 016] GAUSS: Benchmarking Structured Mathematical Skills for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出GAUSS基准，用于评估大语言模型在数学领域的十二项核心技能。通过分类认知技能和设计针对性任务，构建模型的数学能力画像，揭示其优势与不足，推动基于技能的多维评估。**

- **链接: [http://arxiv.org/pdf/2509.18122v1](http://arxiv.org/pdf/2509.18122v1)**

> **作者:** Yue Zhang; Jiaxin Zhang; Qiuyu Ren; Tahsin Saffat; Xiaoxuan Liu; Zitong Yang; Banghua Zhu; Yi Ma
>
> **备注:** 120 pages (including appendix)
>
> **摘要:** We introduce \textbf{GAUSS} (\textbf{G}eneral \textbf{A}ssessment of \textbf{U}nderlying \textbf{S}tructured \textbf{S}kills in Mathematics), a benchmark that evaluates LLMs' mathematical abilities across twelve core skill dimensions, grouped into three domains: knowledge and understanding, problem solving and communication, and meta-skills and creativity. By categorizing problems according to cognitive skills and designing tasks that isolate specific abilities, GAUSS constructs comprehensive, fine-grained, and interpretable profiles of models' mathematical abilities. These profiles faithfully represent their underlying mathematical intelligence. To exemplify how to use the \textsc{GAUSS} benchmark, we have derived the skill profile of \textsc{GPT-5-thinking}, revealing its strengths and weaknesses as well as its differences relative to \textsc{o4-mini-high}, thereby underscoring the value of multidimensional, skill-based evaluation.
>
---
#### [new 017] Evaluating the Creativity of LLMs in Persian Literary Text Generation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估LLMs在波斯语文学文本生成中的创造力。研究构建了涵盖20个主题的数据集，从原创性、流畅性等四个维度评估模型表现，并分析其对四种修辞手法的运用能力，验证了自动评分方法的有效性。**

- **链接: [http://arxiv.org/pdf/2509.18401v1](http://arxiv.org/pdf/2509.18401v1)**

> **作者:** Armin Tourajmehr; Mohammad Reza Modarres; Yadollah Yaghoobzadeh
>
> **摘要:** Large language models (LLMs) have demonstrated notable creative abilities in generating literary texts, including poetry and short stories. However, prior research has primarily centered on English, with limited exploration of non-English literary traditions and without standardized methods for assessing creativity. In this paper, we evaluate the capacity of LLMs to generate Persian literary text enriched with culturally relevant expressions. We build a dataset of user-generated Persian literary spanning 20 diverse topics and assess model outputs along four creativity dimensions-originality, fluency, flexibility, and elaboration-by adapting the Torrance Tests of Creative Thinking. To reduce evaluation costs, we adopt an LLM as a judge for automated scoring and validate its reliability against human judgments using intraclass correlation coefficients, observing strong agreement. In addition, we analyze the models' ability to understand and employ four core literary devices: simile, metaphor, hyperbole, and antithesis. Our results highlight both the strengths and limitations of LLMs in Persian literary text generation, underscoring the need for further refinement.
>
---
#### [new 018] SloPalSpeech: A 2,8000-Hour Slovak Speech Corpus from Parliamentary Data
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文聚焦于低资源语言（斯洛伐克语）的自动语音识别任务，旨在解决训练数据不足的问题。作者构建了SloPalSpeech数据集（2806小时），并基于此微调Whisper模型，显著降低了词错误率，推动了低资源语音识别研究。**

- **链接: [http://arxiv.org/pdf/2509.19270v1](http://arxiv.org/pdf/2509.19270v1)**

> **作者:** Erik Božík; Marek Šuppa
>
> **摘要:** Automatic Speech Recognition (ASR) for low-resource languages like Slovak is hindered by the scarcity of training data. To address this, we introduce SloPalSpeech, a new, large-scale Slovak ASR dataset containing 2,806 hours of speech from parliamentary proceedings. We developed a robust processing pipeline to align and segment long-form recordings into clean, 30-second audio-transcript pairs suitable for model training. We use this dataset to fine-tune several OpenAI Whisper models (small, medium, large-v3, and large-v3-turbo), achieving significant Word Error Rate (WER) reductions on standard Slovak benchmarks like Common Voice and FLEURS. For instance, the fine-tuned Whisper-small model's WER dropped by up to 70\%, approaching the baseline performance of the much larger Whisper-large-v3 model. To foster future research in low-resource speech recognition, we publicly release the complete SloPalSpeech dataset, the fully segmented transcripts (60 million words), and all our fine-tuned models.
>
---
#### [new 019] Beyond the Leaderboard: Understanding Performance Disparities in Large Language Models via Model Diffing
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）的性能分析任务，旨在解决传统基准测试无法解释模型性能差异的问题。通过模型对比（model diffing），研究分析了Gemina-2-9b-it与SimPO增强版之间的能力差异，揭示了特定能力变化对性能的影响。**

- **链接: [http://arxiv.org/pdf/2509.18792v1](http://arxiv.org/pdf/2509.18792v1)**

> **作者:** Sabri Boughorbel; Fahim Dalvi; Nadir Durrani; Majd Hawasly
>
> **备注:** 12 pages, accepted to the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** As fine-tuning becomes the dominant paradigm for improving large language models (LLMs), understanding what changes during this process is increasingly important. Traditional benchmarking often fails to explain why one model outperforms another. In this work, we use model diffing, a mechanistic interpretability approach, to analyze the specific capability differences between Gemma-2-9b-it and a SimPO-enhanced variant. Using crosscoders, we identify and categorize latent representations that differentiate the two models. We find that SimPO acquired latent concepts predominantly enhance safety mechanisms (+32.8%), multilingual capabilities (+43.8%), and instruction-following (+151.7%), while its additional training also reduces emphasis on model self-reference (-44.1%) and hallucination management (-68.5%). Our analysis shows that model diffing can yield fine-grained insights beyond leaderboard metrics, attributing performance gaps to concrete mechanistic capabilities. This approach offers a transparent and targeted framework for comparing LLMs.
>
---
#### [new 020] Extractive Fact Decomposition for Interpretable Natural Language Inference in one Forward Pass
- **分类: cs.CL**

- **简介: 该论文针对自然语言推理（NLI）任务，旨在提升模型的可解释性与鲁棒性。提出JEDI方法，利用编码器架构，在一次前向传播中联合完成抽取式事实分解与推理，无需生成模型，通过合成数据训练，实验证明其在分布内外及对抗设置下表现优异。**

- **链接: [http://arxiv.org/pdf/2509.18901v1](http://arxiv.org/pdf/2509.18901v1)**

> **作者:** Nicholas Popovič; Michael Färber
>
> **备注:** EMNLP 2025
>
> **摘要:** Recent works in Natural Language Inference (NLI) and related tasks, such as automated fact-checking, employ atomic fact decomposition to enhance interpretability and robustness. For this, existing methods rely on resource-intensive generative large language models (LLMs) to perform decomposition. We propose JEDI, an encoder-only architecture that jointly performs extractive atomic fact decomposition and interpretable inference without requiring generative models during inference. To facilitate training, we produce a large corpus of synthetic rationales covering multiple NLI benchmarks. Experimental results demonstrate that JEDI achieves competitive accuracy in distribution and significantly improves robustness out of distribution and in adversarial settings over models based solely on extractive rationale supervision. Our findings show that interpretability and robust generalization in NLI can be realized using encoder-only architectures and synthetic rationales. Code and data available at https://jedi.nicpopovic.com
>
---
#### [new 021] ERFC: Happy Customers with Emotion Recognition and Forecasting in Conversation in Call Centers
- **分类: cs.CL**

- **简介: 该论文提出ERFC模型，用于对话中的情感识别与预测，旨在提升客服场景中客户体验。通过多模态、情感属性和上下文建模，实现未来情感的预测，帮助客服人员及时调整策略，解决客户不满问题。**

- **链接: [http://arxiv.org/pdf/2509.18175v1](http://arxiv.org/pdf/2509.18175v1)**

> **作者:** Aditi Debsharma; Bhushan Jagyasi; Surajit Sen; Priyanka Pandey; Devicharith Dovari; Yuvaraj V. C; Rosalin Parida; Gopali Contractor
>
> **备注:** 7 pages, 6 Figures, 4 Tables, 18 References
>
> **摘要:** Emotion Recognition in Conversation has been seen to be widely applicable in call center analytics, opinion mining, finance, retail, healthcare, and other industries. In a call center scenario, the role of the call center agent is not just confined to receiving calls but to also provide good customer experience by pacifying the frustration or anger of the customers. This can be achieved by maintaining neutral and positive emotion from the agent. As in any conversation, the emotion of one speaker is usually dependent on the emotion of other speaker. Hence the positive emotion of an agent, accompanied with the right resolution will help in enhancing customer experience. This can change an unhappy customer to a happy one. Imparting the right resolution at right time becomes easier if the agent has the insight of the emotion of future utterances. To predict the emotions of the future utterances we propose a novel architecture, Emotion Recognition and Forecasting in Conversation. Our proposed ERFC architecture considers multi modalities, different attributes of emotion, context and the interdependencies of the utterances of the speakers in the conversation. Our intensive experiments on the IEMOCAP dataset have shown the feasibility of the proposed ERFC. This approach can provide a tremendous business value for the applications like call center, where the happiness of customer is utmost important.
>
---
#### [new 022] Consistency-Aware Parameter-Preserving Knowledge Editing Framework for Multi-Hop Question Answering
- **分类: cs.CL**

- **简介: 该论文针对多跳问答任务中的参数保留知识编辑（PPKE）问题，提出CAPE-KG框架。旨在解决现有方法在知识一致性、更新稳定性及检索准确性方面的不足，通过确保KG构建、更新与检索的一致性，提升PPKE在多跳推理中的可靠性与性能。**

- **链接: [http://arxiv.org/pdf/2509.18655v1](http://arxiv.org/pdf/2509.18655v1)**

> **作者:** Lingwen Deng; Yifei Han; Long Zhang; Yue Du; Bin Li
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Parameter-Preserving Knowledge Editing (PPKE) enables updating models with new or corrected information without retraining or parameter adjustment. Recent PPKE approaches based on knowledge graphs (KG) to extend knowledge editing (KE) capabilities to multi-hop question answering (MHQA). However, these methods often lack consistency, leading to knowledge contamination, unstable updates, and retrieval behaviors that fail to reflect the intended edits. Such inconsistencies undermine the reliability of PPKE in multi- hop reasoning. We present CAPE-KG, Consistency-Aware Parameter-Preserving Editing with Knowledge Graphs, a novel consistency-aware framework for PPKE on MHQA. CAPE-KG ensures KG construction, update, and retrieval are always aligned with the requirements of the MHQA task, maintaining coherent reasoning over both unedited and edited knowledge. Extensive experiments on the MQuAKE benchmark show accuracy improvements in PPKE performance for MHQA, demonstrating the effectiveness of addressing consistency in PPKE.
>
---
#### [new 023] Reinforcement Learning on Pre-Training Data
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出RLPT方法，通过强化学习利用预训练数据提升大语言模型的推理能力。任务是优化LLM训练，解决高质量标注数据不足的问题。工作重点在于无需人工标注奖励信号，直接从预训练数据中生成奖励，促进模型在更广泛上下文中探索与学习。**

- **链接: [http://arxiv.org/pdf/2509.19249v1](http://arxiv.org/pdf/2509.19249v1)**

> **作者:** Siheng Li; Kejiao Li; Zenan Xu; Guanhua Huang; Evander Yang; Kun Li; Haoyuan Wu; Jiajia Wu; Zihao Zheng; Chenchen Zhang; Kun Shi; Kyrierl Deng; Qi Yi; Ruibin Xiong; Tingqiang Xu; Yuhao Jiang; Jianfeng Yan; Yuyuan Zeng; Guanghui Xu; Jinbao Xue; Zhijiang Xu; Zheng Fang; Shuai Li; Qibin Liu; Xiaoxue Li; Zhuoyu Li; Yangyu Tao; Fei Gao; Cheng Jiang; Bo Chao Wang; Kai Liu; Jianchen Zhu; Wai Lam; Wayyt Wang; Bo Zhou; Di Wang
>
> **备注:** Work in progress
>
> **摘要:** The growing disparity between the exponential scaling of computational resources and the finite growth of high-quality text data now constrains conventional scaling approaches for large language models (LLMs). To address this challenge, we introduce Reinforcement Learning on Pre-Training data (RLPT), a new training-time scaling paradigm for optimizing LLMs. In contrast to prior approaches that scale training primarily through supervised learning, RLPT enables the policy to autonomously explore meaningful trajectories to learn from pre-training data and improve its capability through reinforcement learning (RL). While existing RL strategies such as reinforcement learning from human feedback (RLHF) and reinforcement learning with verifiable rewards (RLVR) rely on human annotation for reward construction, RLPT eliminates this dependency by deriving reward signals directly from pre-training data. Specifically, it adopts a next-segment reasoning objective, rewarding the policy for accurately predicting subsequent text segments conditioned on the preceding context. This formulation allows RL to be scaled on pre-training data, encouraging the exploration of richer trajectories across broader contexts and thereby fostering more generalizable reasoning skills. Extensive experiments on both general-domain and mathematical reasoning benchmarks across multiple models validate the effectiveness of RLPT. For example, when applied to Qwen3-4B-Base, RLPT yields absolute improvements of $3.0$, $5.1$, $8.1$, $6.0$, $6.6$, and $5.3$ on MMLU, MMLU-Pro, GPQA-Diamond, KOR-Bench, AIME24, and AIME25, respectively. The results further demonstrate favorable scaling behavior, suggesting strong potential for continued gains with more compute. In addition, RLPT provides a solid foundation, extending the reasoning boundaries of LLMs and enhancing RLVR performance.
>
---
#### [new 024] Speculate Deep and Accurate: Lossless and Training-Free Acceleration for Offloaded LLMs via Substitute Speculative Decoding
- **分类: cs.CL**

- **简介: 该论文提出SubSpec方法，用于在有限显存下加速大语言模型的参数卸载。针对现有推测解码依赖训练且加速有限的问题，SubSpec无需训练，通过生成低比特替代层构建高度对齐的草案模型，实现无损加速，平均提速达12.5倍。**

- **链接: [http://arxiv.org/pdf/2509.18344v1](http://arxiv.org/pdf/2509.18344v1)**

> **作者:** Pei-Shuo Wang; Jian-Jia Chen; Chun-Che Yang; Chi-Chih Chang; Ning-Chi Huang; Mohamed S. Abdelfattah; Kai-Chiang Wu
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** The immense model sizes of large language models (LLMs) challenge deployment on memory-limited consumer GPUs. Although model compression and parameter offloading are common strategies to address memory limitations, compression can degrade quality, and offloading maintains quality but suffers from slow inference. Speculative decoding presents a promising avenue to accelerate parameter offloading, utilizing a fast draft model to propose multiple draft tokens, which are then verified by the target LLM in parallel with a single forward pass. This method reduces the time-consuming data transfers in forward passes that involve offloaded weight transfers. Existing methods often rely on pretrained weights of the same family, but require additional training to align with custom-trained models. Moreover, approaches that involve draft model training usually yield only modest speedups. This limitation arises from insufficient alignment with the target model, preventing higher token acceptance lengths. To address these challenges and achieve greater speedups, we propose SubSpec, a plug-and-play method to accelerate parameter offloading that is lossless and training-free. SubSpec constructs a highly aligned draft model by generating low-bit quantized substitute layers from offloaded target LLM portions. Additionally, our method shares the remaining GPU-resident layers and the KV-Cache, further reducing memory overhead and enhance alignment. SubSpec achieves a high average acceptance length, delivering 9.1x speedup for Qwen2.5 7B on MT-Bench (8GB VRAM limit) and an average of 12.5x speedup for Qwen2.5 32B on popular generation benchmarks (24GB VRAM limit).
>
---
#### [new 025] AECBench: A Hierarchical Benchmark for Knowledge Evaluation of Large Language Models in the AEC Field
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出了AECBench，一个针对建筑领域大语言模型的知识评估基准。通过定义23项任务和五级认知框架，评估LLMs在知识记忆、理解、推理等能力上的表现，揭示其在安全关键领域的可靠性问题，为后续研究提供基础。**

- **链接: [http://arxiv.org/pdf/2509.18776v1](http://arxiv.org/pdf/2509.18776v1)**

> **作者:** Chen Liang; Zhaoqi Huang; Haofen Wang; Fu Chai; Chunying Yu; Huanhuan Wei; Zhengjie Liu; Yanpeng Li; Hongjun Wang; Ruifeng Luo; Xianzhong Zhao
>
> **摘要:** Large language models (LLMs), as a novel information technology, are seeing increasing adoption in the Architecture, Engineering, and Construction (AEC) field. They have shown their potential to streamline processes throughout the building lifecycle. However, the robustness and reliability of LLMs in such a specialized and safety-critical domain remain to be evaluated. To address this challenge, this paper establishes AECBench, a comprehensive benchmark designed to quantify the strengths and limitations of current LLMs in the AEC domain. The benchmark defines 23 representative tasks within a five-level cognition-oriented evaluation framework encompassing Knowledge Memorization, Understanding, Reasoning, Calculation, and Application. These tasks were derived from authentic AEC practice, with scope ranging from codes retrieval to specialized documents generation. Subsequently, a 4,800-question dataset encompassing diverse formats, including open-ended questions, was crafted primarily by engineers and validated through a two-round expert review. Furthermore, an LLM-as-a-Judge approach was introduced to provide a scalable and consistent methodology for evaluating complex, long-form responses leveraging expert-derived rubrics. Through the evaluation of nine LLMs, a clear performance decline across five cognitive levels was revealed. Despite demonstrating proficiency in foundational tasks at the Knowledge Memorization and Understanding levels, the models showed significant performance deficits, particularly in interpreting knowledge from tables in building codes, executing complex reasoning and calculation, and generating domain-specific documents. Consequently, this study lays the groundwork for future research and development aimed at the robust and reliable integration of LLMs into safety-critical engineering practices.
>
---
#### [new 026] Speech Vecalign: an Embedding-based Method for Aligning Parallel Speech Documents
- **分类: cs.CL**

- **简介: 该论文提出Speech Vecalign，一种基于嵌入的并行语音文档对齐方法，无需文本转录。相比现有方法，其对齐更长且更稳健，减少了噪声。应用于3000小时英德语语音数据，提升了语音翻译模型性能。**

- **链接: [http://arxiv.org/pdf/2509.18360v1](http://arxiv.org/pdf/2509.18360v1)**

> **作者:** Chutong Meng; Philipp Koehn
>
> **备注:** Accepted by EMNLP 2025 (main)
>
> **摘要:** We present Speech Vecalign, a parallel speech document alignment method that monotonically aligns speech segment embeddings and does not depend on text transcriptions. Compared to the baseline method Global Mining, a variant of speech mining, Speech Vecalign produces longer speech-to-speech alignments. It also demonstrates greater robustness than Local Mining, another speech mining variant, as it produces less noise. We applied Speech Vecalign to 3,000 hours of unlabeled parallel English-German (En-De) speech documents from VoxPopuli, yielding about 1,000 hours of high-quality alignments. We then trained En-De speech-to-speech translation models on the aligned data. Speech Vecalign improves the En-to-De and De-to-En performance over Global Mining by 0.37 and 0.18 ASR-BLEU, respectively. Moreover, our models match or outperform SpeechMatrix model performance, despite using 8 times fewer raw speech documents.
>
---
#### [new 027] Pathways of Thoughts: Multi-Directional Thinking for Long-form Personalized Question Answering
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文聚焦个性化问答任务，旨在解决从长文本中推断用户偏好并生成符合其背景和期望的回答的问题。提出Pathways of Thoughts（PoT）方法，在推理阶段通过多路径探索与聚合，提升LLM的个性化回答能力，实验表明其在LaMP-QA基准上表现优异。**

- **链接: [http://arxiv.org/pdf/2509.19094v1](http://arxiv.org/pdf/2509.19094v1)**

> **作者:** Alireza Salemi; Cheng Li; Mingyang Zhang; Qiaozhu Mei; Zhuowan Li; Spurthi Amba Hombaiah; Weize Kong; Tao Chen; Hamed Zamani; Michael Bendersky
>
> **摘要:** Personalization is essential for adapting question answering (QA) systems to user-specific information needs, thereby improving both accuracy and user satisfaction. However, personalized QA remains relatively underexplored due to challenges such as inferring preferences from long, noisy, and implicit contexts, and generating responses that are simultaneously correct, contextually appropriate, and aligned with user expectations and background knowledge. To address these challenges, we propose Pathways of Thoughts (PoT), an inference-stage method that applies to any large language model (LLM) without requiring task-specific fine-tuning. The approach models the reasoning of an LLM as an iterative decision process, where the model dynamically selects among cognitive operations such as reasoning, revision, personalization, and clarification. This enables exploration of multiple reasoning trajectories, producing diverse candidate responses that capture different perspectives. PoT then aggregates and reweights these candidates according to inferred user preferences, yielding a final personalized response that benefits from the complementary strengths of diverse reasoning paths. Experiments on the LaMP-QA benchmark for personalized QA show that PoT consistently outperforms competitive baselines, achieving up to a 13.1% relative improvement. Human evaluation corroborates these results, with annotators preferring outputs from PoT in 66% of cases and reporting ties in only 15% of cases.
>
---
#### [new 028] Global-Recent Semantic Reasoning on Dynamic Text-Attributed Graphs with Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对动态文本属性图（DyTAGs）中的语义推理任务，提出DyGRASP方法。旨在解决现有模型忽视近期与全局时间语义的问题，结合LLMs和时序GNNs，通过隐式推理、滑动窗口及显式链式结构，高效捕捉动态语义，提升节点预测性能。**

- **链接: [http://arxiv.org/pdf/2509.18742v1](http://arxiv.org/pdf/2509.18742v1)**

> **作者:** Yunan Wang; Jianxin Li; Ziwei Zhang
>
> **摘要:** Dynamic Text-Attribute Graphs (DyTAGs), characterized by time-evolving graph interactions and associated text attributes, are prevalent in real-world applications. Existing methods, such as Graph Neural Networks (GNNs) and Large Language Models (LLMs), mostly focus on static TAGs. Extending these existing methods to DyTAGs is challenging as they largely neglect the recent-global temporal semantics: the recent semantic dependencies among interaction texts and the global semantic evolution of nodes over time. Furthermore, applying LLMs to the abundant and evolving text in DyTAGs faces efficiency issues. To tackle these challenges, we propose Dynamic Global-Recent Adaptive Semantic Processing (DyGRASP), a novel method that leverages LLMs and temporal GNNs to efficiently and effectively reason on DyTAGs. Specifically, we first design a node-centric implicit reasoning method together with a sliding window mechanism to efficiently capture recent temporal semantics. In addition, to capture global semantic dynamics of nodes, we leverage explicit reasoning with tailored prompts and an RNN-like chain structure to infer long-term semantics. Lastly, we intricately integrate the recent and global temporal semantics as well as the dynamic graph structural information using updating and merging layers. Extensive experiments on DyTAG benchmarks demonstrate DyGRASP's superiority, achieving up to 34% improvement in Hit@10 for destination node retrieval task. Besides, DyGRASP exhibits strong generalization across different temporal GNNs and LLMs.
>
---
#### [new 029] Steering Multimodal Large Language Models Decoding for Context-Aware Safety
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多模态大语言模型在安全决策中对上下文感知不足的问题，提出SafeCoDe框架。通过对比解码和全局感知的token调整策略，动态优化生成过程，提升模型在视觉上下文下的安全拒绝能力，同时保持实用性。**

- **链接: [http://arxiv.org/pdf/2509.19212v1](http://arxiv.org/pdf/2509.19212v1)**

> **作者:** Zheyuan Liu; Zhangchen Xu; Guangyao Dou; Xiangchi Yuan; Zhaoxuan Tan; Radha Poovendran; Meng Jiang
>
> **备注:** A lightweight and model-agnostic decoding framework that dynamically adjusts token generation based on multimodal context
>
> **摘要:** Multimodal Large Language Models (MLLMs) are increasingly deployed in real-world applications, yet their ability to make context-aware safety decisions remains limited. Existing methods often fail to balance oversensitivity (unjustified refusals of benign queries) and undersensitivity (missed detection of visually grounded risks), leaving a persistent gap in safety alignment. To address this issue, we introduce Safety-aware Contrastive Decoding (SafeCoDe), a lightweight and model-agnostic decoding framework that dynamically adjusts token generation based on multimodal context. SafeCoDe operates in two stages: (1) a contrastive decoding mechanism that highlights tokens sensitive to visual context by contrasting real and Gaussian-noised images, and (2) a global-aware token modulation strategy that integrates scene-level reasoning with token-level adjustment to adapt refusals according to the predicted safety verdict. Extensive experiments across diverse MLLM architectures and safety benchmarks, covering undersensitivity, oversensitivity, and general safety evaluations, show that SafeCoDe consistently improves context-sensitive refusal behaviors while preserving model helpfulness.
>
---
#### [new 030] Soft Tokens, Hard Truths
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型的推理过程（Chain-of-Thought, CoT），旨在解决连续token训练困难的问题。提出一种基于强化学习的可扩展方法，无需依赖离散token蒸馏，有效学习数百token长度的连续CoT，并在数学推理任务中取得更好表现。**

- **链接: [http://arxiv.org/pdf/2509.19170v1](http://arxiv.org/pdf/2509.19170v1)**

> **作者:** Natasha Butt; Ariel Kwiatkowski; Ismail Labiad; Julia Kempe; Yann Ollivier
>
> **摘要:** The use of continuous instead of discrete tokens during the Chain-of-Thought (CoT) phase of reasoning LLMs has garnered attention recently, based on the intuition that a continuous mixture of discrete tokens could simulate a superposition of several reasoning paths simultaneously. Theoretical results have formally proven that continuous tokens have much greater expressivity and can solve specific problems more efficiently. However, practical use of continuous tokens has been limited by strong training difficulties: previous works either just use continuous tokens at inference time on a pre-trained discrete-token model, or must distill the continuous CoT from ground-truth discrete CoTs and face computational costs that limit the CoT to very few tokens. This is the first work introducing a scalable method to learn continuous CoTs via reinforcement learning (RL), without distilling from reference discrete CoTs. We use "soft" tokens: mixtures of tokens together with noise on the input embedding to provide RL exploration. Computational overhead is minimal, enabling us to learn continuous CoTs with hundreds of tokens. On math reasoning benchmarks with Llama and Qwen models up to 8B, training with continuous CoTs match discrete-token CoTs for pass@1 and surpass them for pass@32, showing greater CoT diversity. In systematic comparisons, the best-performing scenario is to train with continuous CoT tokens then use discrete tokens for inference, meaning the "soft" models can be deployed in a standard way. Finally, we show continuous CoT RL training better preserves the predictions of the base model on out-of-domain tasks, thus providing a softer touch to the base model.
>
---
#### [new 031] Developing an AI framework to automatically detect shared decision-making in patient-doctor conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决自动衡量医患对话中共享决策（SDM）的问题。研究通过构建深度学习和BERT模型，利用会话对齐（CA）评分方法，开发了一种可扩展的自动化框架来评估SDM效果。**

- **链接: [http://arxiv.org/pdf/2509.18439v1](http://arxiv.org/pdf/2509.18439v1)**

> **作者:** Oscar J. Ponce-Ponte; David Toro-Tobon; Luis F. Figueroa; Michael Gionfriddo; Megan Branda; Victor M. Montori; Saturnino Luz; Juan P. Brito
>
> **备注:** 53 pages, 1 figure, 4 tables, 5 supplementary figures, 13 supplementary tables
>
> **摘要:** Shared decision-making (SDM) is necessary to achieve patient-centred care. Currently no methodology exists to automatically measure SDM at scale. This study aimed to develop an automated approach to measure SDM by using language modelling and the conversational alignment (CA) score. A total of 157 video-recorded patient-doctor conversations from a randomized multi-centre trial evaluating SDM decision aids for anticoagulation in atrial fibrillations were transcribed and segmented into 42,559 sentences. Context-response pairs and negative sampling were employed to train deep learning (DL) models and fine-tuned BERT models via the next sentence prediction (NSP) task. Each top-performing model was used to calculate four types of CA scores. A random-effects analysis by clinician, adjusting for age, sex, race, and trial arm, assessed the association between CA scores and SDM outcomes: the Decisional Conflict Scale (DCS) and the Observing Patient Involvement in Decision-Making 12 (OPTION12) scores. p-values were corrected for multiple comparisons with the Benjamini-Hochberg method. Among 157 patients (34% female, mean age 70 SD 10.8), clinicians on average spoke more words than patients (1911 vs 773). The DL model without the stylebook strategy achieved a recall@1 of 0.227, while the fine-tuned BERTbase (110M) achieved the highest recall@1 with 0.640. The AbsMax (18.36 SE7.74 p=0.025) and Max CA (21.02 SE7.63 p=0.012) scores generated with the DL without stylebook were associated with OPTION12. The Max CA score generated with the fine-tuned BERTbase (110M) was associated with the DCS score (-27.61 SE12.63 p=0.037). BERT model sizes did not have an impact the association between CA scores and SDM. This study introduces an automated, scalable methodology to measure SDM in patient-doctor conversations through explainable CA scores, with potential to evaluate SDM strategies at scale.
>
---
#### [new 032] Context-Aware Hierarchical Taxonomy Generation for Scientific Papers via LLM-Guided Multi-Aspect Clustering
- **分类: cs.CL**

- **简介: 该论文提出一种基于LLM引导的多方面聚类方法，用于生成科学论文的上下文感知层次分类体系。旨在解决现有方法在分类连贯性和粒度上的不足，构建了一个新框架和评估基准，实验表明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.19125v1](http://arxiv.org/pdf/2509.19125v1)**

> **作者:** Kun Zhu; Lizi Liao; Yuxuan Gu; Lei Huang; Xiaocheng Feng; Bing Qin
>
> **备注:** Accepted to EMNLP 2025 Main
>
> **摘要:** The rapid growth of scientific literature demands efficient methods to organize and synthesize research findings. Existing taxonomy construction methods, leveraging unsupervised clustering or direct prompting of large language models (LLMs), often lack coherence and granularity. We propose a novel context-aware hierarchical taxonomy generation framework that integrates LLM-guided multi-aspect encoding with dynamic clustering. Our method leverages LLMs to identify key aspects of each paper (e.g., methodology, dataset, evaluation) and generates aspect-specific paper summaries, which are then encoded and clustered along each aspect to form a coherent hierarchy. In addition, we introduce a new evaluation benchmark of 156 expert-crafted taxonomies encompassing 11.6k papers, providing the first naturally annotated dataset for this task. Experimental results demonstrate that our method significantly outperforms prior approaches, achieving state-of-the-art performance in taxonomy coherence, granularity, and interpretability.
>
---
#### [new 033] Diversity Boosts AI-Generated Text Detection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦于AI生成文本检测任务，旨在解决现有方法在高质量文本和可解释性上的不足。提出DivEye框架，利用基于意外度的可解释统计特征，捕捉文本不可预测性的变化，提升检测性能并提供解释性洞察。**

- **链接: [http://arxiv.org/pdf/2509.18880v1](http://arxiv.org/pdf/2509.18880v1)**

> **作者:** Advik Raj Basani; Pin-Yu Chen
>
> **备注:** Project Webpage: https://diveye.vercel.app/
>
> **摘要:** Detecting AI-generated text is an increasing necessity to combat misuse of LLMs in education, business compliance, journalism, and social media, where synthetic fluency can mask misinformation or deception. While prior detectors often rely on token-level likelihoods or opaque black-box classifiers, these approaches struggle against high-quality generations and offer little interpretability. In this work, we propose DivEye, a novel detection framework that captures how unpredictability fluctuates across a text using surprisal-based features. Motivated by the observation that human-authored text exhibits richer variability in lexical and structural unpredictability than LLM outputs, DivEye captures this signal through a set of interpretable statistical features. Our method outperforms existing zero-shot detectors by up to 33.2% and achieves competitive performance with fine-tuned baselines across multiple benchmarks. DivEye is robust to paraphrasing and adversarial attacks, generalizes well across domains and models, and improves the performance of existing detectors by up to 18.7% when used as an auxiliary signal. Beyond detection, DivEye provides interpretable insights into why a text is flagged, pointing to rhythmic unpredictability as a powerful and underexplored signal for LLM detection.
>
---
#### [new 034] MAPEX: A Multi-Agent Pipeline for Keyphrase Extraction
- **分类: cs.CL**

- **简介: 该论文提出MAPEX，一种用于关键词提取的多智能体框架。针对现有无监督方法在不同场景下适应性差的问题，设计了专家招募、知识增强等模块，并采用双路径策略适配文本长度。实验表明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.18813v1](http://arxiv.org/pdf/2509.18813v1)**

> **作者:** Liting Zhang; Shiwan Zhao; Aobo Kong; Qicheng Li
>
> **摘要:** Keyphrase extraction is a fundamental task in natural language processing. However, existing unsupervised prompt-based methods for Large Language Models (LLMs) often rely on single-stage inference pipelines with uniform prompting, regardless of document length or LLM backbone. Such one-size-fits-all designs hinder the full exploitation of LLMs' reasoning and generation capabilities, especially given the complexity of keyphrase extraction across diverse scenarios. To address these challenges, we propose MAPEX, the first framework that introduces multi-agent collaboration into keyphrase extraction. MAPEX coordinates LLM-based agents through modules for expert recruitment, candidate extraction, topic guidance, knowledge augmentation, and post-processing. A dual-path strategy dynamically adapts to document length: knowledge-driven extraction for short texts and topic-guided extraction for long texts. Extensive experiments on six benchmark datasets across three different LLMs demonstrate its strong generalization and universality, outperforming the state-of-the-art unsupervised method by 2.44\% and standard LLM baselines by 4.01\% in F1@5 on average. Code is available at https://github.com/NKU-LITI/MAPEX.
>
---
#### [new 035] Financial Risk Relation Identification through Dual-view Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于金融风险关系识别任务，旨在解决传统依赖人工分析的主观性和低效问题。工作提出一种基于Form 10-K文件的无监督方法，利用NLP技术提取企业间隐含风险关联，并量化风险关系得分，提升分析的透明度与可扩展性。**

- **链接: [http://arxiv.org/pdf/2509.18775v1](http://arxiv.org/pdf/2509.18775v1)**

> **作者:** Wei-Ning Chiu; Yu-Hsiang Wang; Andy Hsiao; Yu-Shiang Huang; Chuan-Ju Wang
>
> **备注:** 11 pages, 3 figures, EMNLP 2025 Main Conference
>
> **摘要:** A multitude of interconnected risk events -- ranging from regulatory changes to geopolitical tensions -- can trigger ripple effects across firms. Identifying inter-firm risk relations is thus crucial for applications like portfolio management and investment strategy. Traditionally, such assessments rely on expert judgment and manual analysis, which are, however, subjective, labor-intensive, and difficult to scale. To address this, we propose a systematic method for extracting inter-firm risk relations using Form 10-K filings -- authoritative, standardized financial documents -- as our data source. Leveraging recent advances in natural language processing, our approach captures implicit and abstract risk connections through unsupervised fine-tuning based on chronological and lexical patterns in the filings. This enables the development of a domain-specific financial encoder with a deeper contextual understanding and introduces a quantitative risk relation score for transparency, interpretable analysis. Extensive experiments demonstrate that our method outperforms strong baselines across multiple evaluation settings.
>
---
#### [new 036] Human-Annotated NER Dataset for the Kyrgyz Language
- **分类: cs.CL**

- **简介: 该论文提出了KyrgyzNER，首个针对吉尔吉斯语的命名实体识别数据集。数据集包含1,499篇新闻，涵盖27类实体。作者展示了标注方案、挑战及统计信息，并评估了多种NER模型。研究强调了多语言预训练模型在低资源语言中的潜力与挑战。**

- **链接: [http://arxiv.org/pdf/2509.19109v1](http://arxiv.org/pdf/2509.19109v1)**

> **作者:** Timur Turatali; Anton Alekseev; Gulira Jumalieva; Gulnara Kabaeva; Sergey Nikolenko
>
> **备注:** Accepted to TurkLang-2025 conference, DOI and copyright will be added upon confirmation of acceptance to publication in IEEE Xplore
>
> **摘要:** We introduce KyrgyzNER, the first manually annotated named entity recognition dataset for the Kyrgyz language. Comprising 1,499 news articles from the 24.KG news portal, the dataset contains 10,900 sentences and 39,075 entity mentions across 27 named entity classes. We show our annotation scheme, discuss the challenges encountered in the annotation process, and present the descriptive statistics. We also evaluate several named entity recognition models, including traditional sequence labeling approaches based on conditional random fields and state-of-the-art multilingual transformer-based models fine-tuned on our dataset. While all models show difficulties with rare entity categories, models such as the multilingual RoBERTa variant pretrained on a large corpus across many languages achieve a promising balance between precision and recall. These findings emphasize both the challenges and opportunities of using multilingual pretrained models for processing languages with limited resources. Although the multilingual RoBERTa model performed best, other multilingual models yielded comparable results. This suggests that future work exploring more granular annotation schemes may offer deeper insights for Kyrgyz language processing pipelines evaluation.
>
---
#### [new 037] Evaluating Large Language Models for Detecting Antisemitism
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理任务，旨在解决检测反犹太主义内容的问题。研究评估了八种开源大语言模型的检测能力，提出了一种新的Guided-CoT提示方法，并分析了模型性能与错误行为。**

- **链接: [http://arxiv.org/pdf/2509.18293v1](http://arxiv.org/pdf/2509.18293v1)**

> **作者:** Jay Patel; Hrudayangam Mehta; Jeremy Blackburn
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Detecting hateful content is a challenging and important problem. Automated tools, like machine-learning models, can help, but they require continuous training to adapt to the ever-changing landscape of social media. In this work, we evaluate eight open-source LLMs' capability to detect antisemitic content, specifically leveraging in-context definition as a policy guideline. We explore various prompting techniques and design a new CoT-like prompt, Guided-CoT. Guided-CoT handles the in-context policy well, increasing performance across all evaluated models, regardless of decoding configuration, model sizes, or reasoning capability. Notably, Llama 3.1 70B outperforms fine-tuned GPT-3.5. Additionally, we examine LLM errors and introduce metrics to quantify semantic divergence in model-generated rationales, revealing notable differences and paradoxical behaviors among LLMs. Our experiments highlight the differences observed across LLMs' utility, explainability, and reliability.
>
---
#### [new 038] False Friends Are Not Foes: Investigating Vocabulary Overlap in Multilingual Language Models
- **分类: cs.CL**

- **简介: 该论文研究多语言模型中词汇重叠对跨语言迁移的影响。通过控制实验，分析不同重叠设置下的双语模型，发现词汇重叠有助于捕捉跨语言语义关系并提升性能，证明共享词汇对多语言模型设计有益。属于自然语言处理任务。**

- **链接: [http://arxiv.org/pdf/2509.18750v1](http://arxiv.org/pdf/2509.18750v1)**

> **作者:** Julie Kallini; Dan Jurafsky; Christopher Potts; Martijn Bartelds
>
> **摘要:** Subword tokenizers trained on multilingual corpora naturally produce overlapping tokens across languages. Does token overlap facilitate cross-lingual transfer or instead introduce interference between languages? Prior work offers mixed evidence, partly due to varied setups and confounders, such as token frequency or subword segmentation granularity. To address this question, we devise a controlled experiment where we train bilingual autoregressive models on multiple language pairs under systematically varied vocabulary overlap settings. Crucially, we explore a new dimension to understanding how overlap affects transfer: the semantic similarity of tokens shared across languages. We first analyze our models' hidden representations and find that overlap of any kind creates embedding spaces that capture cross-lingual semantic relationships, while this effect is much weaker in models with disjoint vocabularies. On XNLI and XQuAD, we find that models with overlap outperform models with disjoint vocabularies, and that transfer performance generally improves as overlap increases. Overall, our findings highlight the advantages of token overlap in multilingual models and show that substantial shared vocabulary remains a beneficial design choice for multilingual tokenizers.
>
---
#### [new 039] Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction
- **分类: cs.CL**

- **简介: 该论文研究LLM作为评分者的不确定性分析，提出基于符合预测的区间评估框架，设计序数边界调整和中点评分方法，提升评价可靠性。任务为自然语言生成评估，解决评分不确定性问题。**

- **链接: [http://arxiv.org/pdf/2509.18658v1](http://arxiv.org/pdf/2509.18658v1)**

> **作者:** Huanxin Sheng; Xinyi Liu; Hangfeng He; Jieyu Zhao; Jian Kang
>
> **备注:** To appear in EMNLP 2025. Our code and data are available at \url{https://github.com/BruceSheng1202/Analyzing_Uncertainty_of_LLM-as-a-Judge
>
> **摘要:** LLM-as-a-judge has become a promising paradigm for using large language models (LLMs) to evaluate natural language generation (NLG), but the uncertainty of its evaluation remains underexplored. This lack of reliability may limit its deployment in many applications. This work presents the first framework to analyze the uncertainty by offering a prediction interval of LLM-based scoring via conformal prediction. Conformal prediction constructs continuous prediction intervals from a single evaluation run, and we design an ordinal boundary adjustment for discrete rating tasks. We also suggest a midpoint-based score within the interval as a low-bias alternative to raw model score and weighted average. We perform extensive experiments and analysis, which show that conformal prediction can provide valid prediction interval with coverage guarantees. We also explore the usefulness of interval midpoint and judge reprompting for better judgment.
>
---
#### [new 040] A Rhythm-Aware Phrase Insertion for Classical Arabic Poetry Composition
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种基于ByT5模型的阿拉伯诗歌节奏感知短语插入方法，旨在解决古典阿拉伯诗作中保持节奏与语义一致的问题。通过定制字节级节奏提取和条件去噪微调策略，实现高质量的节奏匹配与语义连贯性。**

- **链接: [http://arxiv.org/pdf/2509.18514v1](http://arxiv.org/pdf/2509.18514v1)**

> **作者:** Mohamad Elzohbi; Richard Zhao
>
> **备注:** Accepted for the Third Arabic Natural Language Processing Conference (ArabicNLP 2025)
>
> **摘要:** This paper presents a methodology for inserting phrases in Arabic poems to conform to a specific rhythm using ByT5, a byte-level multilingual transformer-based model. Our work discusses a rule-based grapheme-to-beat transformation tailored for extracting the rhythm from fully diacritized Arabic script. Our approach employs a conditional denoising objective to fine-tune ByT5, where the model reconstructs masked words to match a target rhythm. We adopt a curriculum learning strategy, pre-training on a general Arabic dataset before fine-tuning on poetic dataset, and explore cross-lingual transfer from English to Arabic. Experimental results demonstrate that our models achieve high rhythmic alignment while maintaining semantic coherence. The proposed model has the potential to be used in co-creative applications in the process of composing classical Arabic poems.
>
---
#### [new 041] Brittleness and Promise: Knowledge Graph Based Reward Modeling for Diagnostic Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨基于知识图谱的奖励建模用于医疗诊断推理。针对大模型缺乏可靠知识推理的问题，提出将LLM作为KG推理路径的奖励模型进行训练。系统评估了任务设定与训练方法，并测试其在下游诊断任务中的泛化能力，发现该方法在路径判断上有效但泛化性较弱。**

- **链接: [http://arxiv.org/pdf/2509.18316v1](http://arxiv.org/pdf/2509.18316v1)**

> **作者:** Saksham Khatwani; He Cheng; Majid Afshar; Dmitriy Dligach; Yanjun Gao
>
> **摘要:** Large language models (LLMs) show promise for diagnostic reasoning but often lack reliable, knowledge grounded inference. Knowledge graphs (KGs), such as the Unified Medical Language System (UMLS), offer structured biomedical knowledge that can support trustworthy reasoning. Prior approaches typically integrate KGs via retrieval augmented generation or fine tuning, inserting KG content into prompts rather than enabling structured reasoning. We explore an alternative paradigm: treating the LLM as a reward model of KG reasoning paths, where the model learns to judge whether a candidate path leads to correct diagnosis for a given patient input. This approach is inspired by recent work that leverages reward training to enhance model reasoning abilities, and grounded in computational theory, which suggests that verifying a solution is often easier than generating one from scratch. It also parallels physicians' diagnostic assessment, where they judge which sequences of findings and intermediate conditions most plausibly support a diagnosis. We first systematically evaluate five task formulation for knowledge path judging and eight training paradigm. Second, we test whether the path judging abilities generalize to downstream diagnostic tasks, including diagnosis summarization and medical question answering. Experiments with three open source instruct-tuned LLMs reveal both promise and brittleness: while specific reward optimization and distillation lead to strong path-judging performance, the transferability to downstream tasks remain weak. Our finding provides the first systematic assessment of "reward model style" reasoning over clinical KGs, offering insights into how structured, reward-based supervision influences diagnostic reasoning in GenAI systems for healthcare.
>
---
#### [new 042] TsqLoRA: Towards Sensitivity and Quality Low-Rank Adaptation for Efficient Fine-Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TsqLoRA，针对自然语言处理任务中微调大模型的效率问题。为解决全参数微调计算成本高和现有方法忽视层敏感性与数据质量的问题，TsqLoRA结合数据质量驱动选择与敏感度感知的低秩适应，提升微调效率并保持性能。**

- **链接: [http://arxiv.org/pdf/2509.18585v1](http://arxiv.org/pdf/2509.18585v1)**

> **作者:** Yu Chen; Yifei Han; Long Zhang; Yue Du; Bin Li
>
> **备注:** 5 pages, 4 figures, published to ICASSP2026
>
> **摘要:** Fine-tuning large pre-trained models for downstream tasks has become a fundamental approach in natural language processing. Fully fine-tuning all model parameters is computationally expensive and memory-intensive, especially in resource-constrained environments. Existing parameter-efficient fine-tuning methods reduce the number of trainable parameters but typically overlook the varying sensitivity of different model layers and the importance of training data. In this work, we propose TsqLoRA, a novel method that integrates data-quality-driven selection with sensitivity-aware low-rank adaptation, consisted of two main components: a quality-aware sampling mechanism for selecting the most informative training data, and a dynamic rank allocation module that adjusts the rank of each layer based on its sensitivity to parameter updates. The experimental results demonstrate that TsqLoRA improves fine-tuning efficiency while maintaining or even improving performance on a variety of NLP tasks. Our code will be available at https://github.com/Benjamin-Ricky/TsqLoRA.
>
---
#### [new 043] Interactive Real-Time Speaker Diarization Correction with Human Feedback
- **分类: cs.CL**

- **简介: 该论文研究的是**说话人日志（Speaker Diarization）修正任务**，旨在解决自动语音系统中**说话人归属错误的问题**。作者提出一个结合**大语言模型（LLM）和用户实时反馈**的系统，通过流式ASR与日志分析、多说话人片段分割及在线注册更新等技术，提升日志准确性，并在实验中显著降低了错误率。**

- **链接: [http://arxiv.org/pdf/2509.18377v1](http://arxiv.org/pdf/2509.18377v1)**

> **作者:** Xinlu He; Yiwen Guan; Badrivishal Paurana; Zilin Dai; Jacob Whitehill
>
> **摘要:** Most automatic speech processing systems operate in "open loop" mode without user feedback about who said what; yet, human-in-the-loop workflows can potentially enable higher accuracy. We propose an LLM-assisted speaker diarization correction system that lets users fix speaker attribution errors in real time. The pipeline performs streaming ASR and diarization, uses an LLM to deliver concise summaries to the users, and accepts brief verbal feedback that is immediately incorporated without disrupting interactions. Moreover, we develop techniques to make the workflow more effective: First, a split-when-merged (SWM) technique detects and splits multi-speaker segments that the ASR erroneously attributes to just a single speaker. Second, online speaker enrollments are collected based on users' diarization corrections, thus helping to prevent speaker diarization errors from occurring in the future. LLM-driven simulations on the AMI test set indicate that our system substantially reduces DER by 9.92% and speaker confusion error by 44.23%. We further analyze correction efficacy under different settings, including summary vs full transcript display, the number of online enrollments limitation, and correction frequency.
>
---
#### [new 044] Trace Is In Sentences: Unbiased Lightweight ChatGPT-Generated Text Detector
- **分类: cs.CL; eess.SP**

- **简介: 该论文提出一种轻量级AI生成文本检测方法，任务是识别原始及经简单修改的ChatGPT生成文本。针对现有方法易受词级变换影响、存在偏见等问题，通过建模句子结构关系并结合对比学习与因果图，提取稳定结构特征以提升检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.18535v1](http://arxiv.org/pdf/2509.18535v1)**

> **作者:** Mo Mu; Dianqiao Lei; Chang Li
>
> **摘要:** The widespread adoption of ChatGPT has raised concerns about its misuse, highlighting the need for robust detection of AI-generated text. Current word-level detectors are vulnerable to paraphrasing or simple prompts (PSP), suffer from biases induced by ChatGPT's word-level patterns (CWP) and training data content, degrade on modified text, and often require large models or online LLM interaction. To tackle these issues, we introduce a novel task to detect both original and PSP-modified AI-generated texts, and propose a lightweight framework that classifies texts based on their internal structure, which remains invariant under word-level changes. Our approach encodes sentence embeddings from pre-trained language models and models their relationships via attention. We employ contrastive learning to mitigate embedding biases from autoregressive generation and incorporate a causal graph with counterfactual methods to isolate structural features from topic-related biases. Experiments on two curated datasets, including abstract comparisons and revised life FAQs, validate the effectiveness of our method.
>
---
#### [new 045] Are most sentences unique? An empirical examination of Chomskyan claims
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在验证“多数句子是唯一的”这一语言学主张。作者使用NLTK解析不同语料库，统计完全重复的句子数量，发现唯一句子虽常见，但重复句在各类语料中也占一定比例。**

- **链接: [http://arxiv.org/pdf/2509.19108v1](http://arxiv.org/pdf/2509.19108v1)**

> **作者:** Hiram Ring
>
> **摘要:** A repeated claim in linguistics is that the majority of linguistic utterances are unique. For example, Pinker (1994: 10), summarizing an argument by Noam Chomsky, states that "virtually every sentence that a person utters or understands is a brand-new combination of words, appearing for the first time in the history of the universe." With the increased availability of large corpora, this is a claim that can be empirically investigated. The current paper addresses the question by using the NLTK Python library to parse corpora of different genres, providing counts of exact string matches in each. Results show that while completely unique sentences are often the majority of corpora, this is highly constrained by genre, and that duplicate sentences are not an insignificant part of any individual corpus.
>
---
#### [new 046] DRISHTIKON: A Multimodal Multilingual Benchmark for Testing Language Models' Understanding on Indian Culture
- **分类: cs.CL; cs.MM**

- **简介: 该论文提出DRISHTIKON，一个聚焦印度文化的多模态多语言基准，用于评估语言模型对印度文化理解的能力。针对现有基准缺乏文化深度和语言多样性的问题，构建了包含64,000对图文数据的测试集，并评估多种视觉-语言模型的表现，揭示其在低资源语言和传统文化上的不足。**

- **链接: [http://arxiv.org/pdf/2509.19274v1](http://arxiv.org/pdf/2509.19274v1)**

> **作者:** Arijit Maji; Raghvendra Kumar; Akash Ghosh; Anushka; Nemil Shah; Abhilekh Borah; Vanshika Shah; Nishant Mishra; Sriparna Saha
>
> **备注:** EMNLP MAINS 2025
>
> **摘要:** We introduce DRISHTIKON, a first-of-its-kind multimodal and multilingual benchmark centered exclusively on Indian culture, designed to evaluate the cultural understanding of generative AI systems. Unlike existing benchmarks with a generic or global scope, DRISHTIKON offers deep, fine-grained coverage across India's diverse regions, spanning 15 languages, covering all states and union territories, and incorporating over 64,000 aligned text-image pairs. The dataset captures rich cultural themes including festivals, attire, cuisines, art forms, and historical heritage amongst many more. We evaluate a wide range of vision-language models (VLMs), including open-source small and large models, proprietary systems, reasoning-specialized VLMs, and Indic-focused models, across zero-shot and chain-of-thought settings. Our results expose key limitations in current models' ability to reason over culturally grounded, multimodal inputs, particularly for low-resource languages and less-documented traditions. DRISHTIKON fills a vital gap in inclusive AI research, offering a robust testbed to advance culturally aware, multimodally competent language technologies.
>
---
#### [new 047] UniECG: Understanding and Generating ECG in One Unified Model
- **分类: cs.CL**

- **简介: 该论文提出UniECG，首个能同时进行ECG解读和文本生成ECG的统一模型。针对现有模型无法准确理解或生成ECG的问题，采用两阶段训练方法，扩展了ECG模型的能力边界。**

- **链接: [http://arxiv.org/pdf/2509.18588v1](http://arxiv.org/pdf/2509.18588v1)**

> **作者:** Jiarui Jin; Haoyu Wang; Xiang Lan; Jun Li; Gaofeng Cheng; Hongyan Li; Shenda Hong
>
> **摘要:** Recent unified models such as GPT-5 have achieved encouraging progress on vision-language tasks. However, these unified models typically fail to correctly understand ECG signals and provide accurate medical diagnoses, nor can they correctly generate ECG signals. To address these limitations, we propose UniECG, the first unified model for ECG capable of concurrently performing evidence-based ECG interpretation and text-conditioned ECG generation tasks. Through a decoupled two-stage training approach, the model first learns evidence-based interpretation skills (ECG-to-Text), and then injects ECG generation capabilities (Text-to-ECG) via latent space alignment. UniECG can autonomously choose to interpret or generate an ECG based on user input, significantly extending the capability boundaries of current ECG models. Our code and checkpoints will be made publicly available at https://github.com/PKUDigitalHealth/UniECG upon acceptance.
>
---
#### [new 048] Systematic Comparative Analysis of Large Pretrained Language Models on Contextualized Medication Event Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床自然语言处理任务，旨在解决从电子健康记录中提取药物事件及其上下文信息的问题。研究对比了多个预训练模型在CMED数据集上的表现，分析其在药物事件抽取和分类中的效果。**

- **链接: [http://arxiv.org/pdf/2509.19224v1](http://arxiv.org/pdf/2509.19224v1)**

> **作者:** Tariq Abdul-Quddoos; Xishuang Dong; Lijun Qian
>
> **摘要:** Attention-based models have become the leading approach in modeling medical language for Natural Language Processing (NLP) in clinical notes. These models outperform traditional techniques by effectively capturing contextual rep- resentations of language. In this research a comparative analysis is done amongst pre- trained attention based models namely Bert Base, BioBert, two variations of Bio+Clinical Bert, RoBerta, and Clinical Long- former on task related to Electronic Health Record (EHR) information extraction. The tasks from Track 1 of Harvard Medical School's 2022 National Clinical NLP Challenges (n2c2) are considered for this comparison, with the Contextualized Medication Event Dataset (CMED) given for these task. CMED is a dataset of unstructured EHRs and annotated notes that contain task relevant information about the EHRs. The goal of the challenge is to develop effective solutions for extracting contextual information related to patient medication events from EHRs using data driven methods. Each pre-trained model is fine-tuned and applied on CMED to perform medication extraction, medical event detection, and multi-dimensional medication event context classification. Pro- cessing methods are also detailed for breaking down EHRs for compatibility with the applied models. Performance analysis has been carried out using a script based on constructing medical terms from the evaluation portion of CMED with metrics including recall, precision, and F1-Score. The results demonstrate that models pre-trained on clinical data are more effective in detecting medication and medication events, but Bert Base, pre- trained on general domain data showed to be the most effective for classifying the context of events related to medications.
>
---
#### [new 049] DTW-Align: Bridging the Modality Gap in End-to-End Speech Translation with Dynamic Time Warping Alignment
- **分类: cs.CL**

- **简介: 该论文研究端到端语音翻译任务，旨在解决语音与文本模态差异问题。提出DTW-Align方法，利用动态时间规整对齐语音和文本嵌入，无需对齐工具，提升了对齐精度和效率，尤其在低资源场景表现更优。**

- **链接: [http://arxiv.org/pdf/2509.18987v1](http://arxiv.org/pdf/2509.18987v1)**

> **作者:** Abderrahmane Issam; Yusuf Can Semerci; Jan Scholtes; Gerasimos Spanakis
>
> **备注:** Accepted at WMT2025
>
> **摘要:** End-to-End Speech Translation (E2E-ST) is the task of translating source speech directly into target text bypassing the intermediate transcription step. The representation discrepancy between the speech and text modalities has motivated research on what is known as bridging the modality gap. State-of-the-art methods addressed this by aligning speech and text representations on the word or token level. Unfortunately, this requires an alignment tool that is not available for all languages. Although this issue has been addressed by aligning speech and text embeddings using nearest-neighbor similarity search, it does not lead to accurate alignments. In this work, we adapt Dynamic Time Warping (DTW) for aligning speech and text embeddings during training. Our experiments demonstrate the effectiveness of our method in bridging the modality gap in E2E-ST. Compared to previous work, our method produces more accurate alignments and achieves comparable E2E-ST results while being significantly faster. Furthermore, our method outperforms previous work in low resource settings on 5 out of 6 language directions.
>
---
#### [new 050] Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity
- **分类: cs.CL; 68T50; I.2.7**

- **简介: 该论文提出一种基于先验统计的噪声文本过滤方法，用于替代计算昂贵的困惑度过滤。通过词频统计快速筛选数据，在20个下游任务中表现最优，效率提升超1000倍，适用于代码、数学和多语言场景。**

- **链接: [http://arxiv.org/pdf/2509.18577v1](http://arxiv.org/pdf/2509.18577v1)**

> **作者:** Yeongbin Seo; Gayoung Kim; Jaehyung Kim; Jinyoung Yeo
>
> **摘要:** As large language models (LLMs) are pretrained on massive web corpora, careful selection of data becomes essential to ensure effective and efficient learning. While perplexity (PPL)-based filtering has shown strong performance, it suffers from drawbacks: substantial time costs and inherent unreliability of the model when handling noisy or out-of-distribution samples. In this work, we propose a simple yet powerful alternative: a prior-based data filtering method that estimates token priors using corpus-level term frequency statistics, inspired by linguistic insights on word roles and lexical density. Our approach filters documents based on the mean and standard deviation of token priors, serving as a fast proxy to PPL while requiring no model inference. Despite its simplicity, the prior-based filter achieves the highest average performance across 20 downstream benchmarks, while reducing time cost by over 1000x compared to PPL-based filtering. We further demonstrate its applicability to symbolic languages such as code and math, and its dynamic adaptability to multilingual corpora without supervision
>
---
#### [new 051] LOTUSDIS: A Thai far-field meeting corpus for robust conversational ASR
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出了LOTUSDIS，一个用于提升远场对话ASR鲁棒性的泰语会议语料库。针对远场语音识别性能下降的问题，构建了包含114小时真实对话的数据集，并通过基线实验验证了距离多样数据对提升识别效果的重要性。**

- **链接: [http://arxiv.org/pdf/2509.18722v1](http://arxiv.org/pdf/2509.18722v1)**

> **作者:** Pattara Tipaksorn; Sumonmas Thatphithakkul; Vataya Chunwijitra; Kwanchiva Thangthai
>
> **摘要:** We present LOTUSDIS, a publicly available Thai meeting corpus designed to advance far-field conversational ASR. The dataset comprises 114 hours of spontaneous, unscripted dialogue collected in 15-20 minute sessions with three participants, where overlapping speech is frequent and natural. Speech was recorded simultaneously by nine independent single-channel devices spanning six microphone types at distances from 0.12 m to 10 m, preserving the authentic effects of reverberation, noise, and device coloration without relying on microphone arrays. We provide standard train, dev, test splits and release a reproducible baseline system. We benchmarked several Whisper variants under zero-shot and fine-tuned conditions. Off-the-shelf models showed strong degradation with distance, confirming a mismatch between pre-training data and Thai far-field speech. Fine-tuning on LOTUSDIS dramatically improved robustness: a Thai Whisper baseline reduced overall WER from 64.3 to 38.3 and far-field WER from 81.6 to 49.5, with especially large gains on the most distant microphones. These results underscore the importance of distance-diverse training data for robust ASR. The corpus is available under CC-BY-SA 4.0. We also release training and evaluation scripts as a baseline system to promote reproducible research in this field.
>
---
#### [new 052] NormGenesis: Multicultural Dialogue Generation via Exemplar-Guided Social Norm Modeling and Violation Recovery
- **分类: cs.CL**

- **简介: 该论文提出NormGenesis，一个多文化对话生成框架，旨在解决对话系统在不同文化背景下生成符合社会规范的响应问题。通过引入Violation-to-Resolution对话类型和示例引导的迭代优化，提升了对话的自然性和文化适应性，并构建了一个高质量多语言对话数据集。**

- **链接: [http://arxiv.org/pdf/2509.18395v1](http://arxiv.org/pdf/2509.18395v1)**

> **作者:** Minki Hong; Jangho Choi; Jihie Kim
>
> **备注:** 39 pages, 17 figures, EMNLP 2025 Main Conference
>
> **摘要:** Social norms govern culturally appropriate behavior in communication, enabling dialogue systems to produce responses that are not only coherent but also socially acceptable. We present NormGenesis, a multicultural framework for generating and annotating socially grounded dialogues across English, Chinese, and Korean. To model the dynamics of social interaction beyond static norm classification, we propose a novel dialogue type, Violation-to-Resolution (V2R), which models the progression of conversations following norm violations through recognition and socially appropriate repair. To improve pragmatic consistency in underrepresented languages, we implement an exemplar-based iterative refinement early in the dialogue synthesis process. This design introduces alignment with linguistic, emotional, and sociocultural expectations before full dialogue generation begins. Using this framework, we construct a dataset of 10,800 multi-turn dialogues annotated at the turn level for norm adherence, speaker intent, and emotional response. Human and LLM-based evaluations demonstrate that NormGenesis significantly outperforms existing datasets in refinement quality, dialogue naturalness, and generalization performance. We show that models trained on our V2R-augmented data exhibit improved pragmatic competence in ethically sensitive contexts. Our work establishes a new benchmark for culturally adaptive dialogue modeling and provides a scalable methodology for norm-aware generation across linguistically and culturally diverse languages.
>
---
#### [new 053] LAWCAT: Efficient Distillation from Quadratic to Linear Attention with Convolution across Tokens for Long Context Modeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LAWSAT，一种高效的线性注意力框架，旨在解决长序列建模中Transformer二次复杂度的瓶颈问题。通过知识蒸馏和卷积增强局部依赖，实现高性能、低资源消耗的长上下文模型，适用于边缘部署。**

- **链接: [http://arxiv.org/pdf/2509.18467v1](http://arxiv.org/pdf/2509.18467v1)**

> **作者:** Zeyu Liu; Souvik Kundu; Lianghao Jiang; Anni Li; Srikanth Ronanki; Sravan Bodapati; Gourav Datta; Peter A. Beerel
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Although transformer architectures have achieved state-of-the-art performance across diverse domains, their quadratic computational complexity with respect to sequence length remains a significant bottleneck, particularly for latency-sensitive long-context applications. While recent linear-complexity alternatives are increasingly powerful, effectively training them from scratch is still resource-intensive. To overcome these limitations, we propose LAWCAT (Linear Attention with Convolution Across Time), a novel linearization framework designed to efficiently transfer the capabilities of pre-trained transformers into a performant linear attention architecture. LAWCAT integrates causal Conv1D layers to enhance local dependency modeling and employs normalized gated linear attention to improve generalization across varying context lengths. Our comprehensive evaluations demonstrate that, distilling Mistral-7B with only 1K-length sequences yields over 90\% passkey retrieval accuracy up to 22K tokens, significantly extending its effective context window. Similarly, Llama3.2-1B LAWCAT variant achieves competitive performance on S-NIAH 1\&2\&3 tasks (1K-8K context length) and BABILong benchmark (QA2\&QA3, 0K-16K context length), requiring less than 0.1\% pre-training tokens compared with pre-training models. Furthermore, LAWCAT exhibits faster prefill speeds than FlashAttention-2 for sequences exceeding 8K tokens. LAWCAT thus provides an efficient pathway to high-performance, long-context linear models suitable for edge deployment, reducing reliance on extensive long-sequence training data and computational resources.
>
---
#### [new 054] Dynamic Prompt Fusion for Multi-Task and Cross-Domain Adaptation in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出一种动态提示融合方法，用于提升大语言模型在多任务和跨领域场景下的泛化能力。针对固定提示模板的局限性，设计了包含提示池和任务感知调度机制的统一框架，有效缓解任务干扰并增强模型适应性与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.18113v1](http://arxiv.org/pdf/2509.18113v1)**

> **作者:** Xin Hu; Yue Kang; Guanzi Yao; Tianze Kang; Mengjie Wang; Heyao Liu
>
> **摘要:** This study addresses the generalization limitations commonly observed in large language models under multi-task and cross-domain settings. Unlike prior methods such as SPoT, which depends on fixed prompt templates, our study introduces a unified multi-task learning framework with dynamic prompt scheduling mechanism. By introducing a prompt pool and a task-aware scheduling strategy, the method dynamically combines and aligns prompts for different tasks. This enhances the model's ability to capture semantic differences across tasks. During prompt fusion, the model uses task embeddings and a gating mechanism to finely control the prompt signals. This ensures alignment between prompt content and task-specific demands. At the same time, it builds flexible sharing pathways across tasks. In addition, the proposed optimization objective centers on joint multi-task learning. It incorporates an automatic learning strategy for scheduling weights, which effectively mitigates task interference and negative transfer. To evaluate the effectiveness of the method, a series of sensitivity experiments were conducted. These experiments examined the impact of prompt temperature parameters and task number variation. The results confirm the advantages of the proposed mechanism in maintaining model stability and enhancing transferability. Experimental findings show that the prompt scheduling method significantly improves performance on a range of language understanding and knowledge reasoning tasks. These results fully demonstrate its applicability and effectiveness in unified multi-task modeling and cross-domain adaptation.
>
---
#### [new 055] Actions Speak Louder than Prompts: A Large-Scale Study of LLMs for Graph Inference
- **分类: cs.CL**

- **简介: 该论文研究LLM在图推理任务（如节点分类）中的应用，解决LLM与图数据交互能力理解不足的问题。通过大规模实验对比提示、工具使用和代码生成等方法，分析不同图结构和文本特征下的表现，发现代码生成在长文本和高异质性图中效果最佳，为LLM图推理设计提供指导。**

- **链接: [http://arxiv.org/pdf/2509.18487v1](http://arxiv.org/pdf/2509.18487v1)**

> **作者:** Ben Finkelshtein; Silviu Cucerzan; Sujay Kumar Jauhar; Ryen White
>
> **摘要:** Large language models (LLMs) are increasingly used for text-rich graph machine learning tasks such as node classification in high-impact domains like fraud detection and recommendation systems. Yet, despite a surge of interest, the field lacks a principled understanding of the capabilities of LLMs in their interaction with graph data. In this work, we conduct a large-scale, controlled evaluation across several key axes of variability to systematically assess the strengths and weaknesses of LLM-based graph reasoning methods in text-based applications. The axes include the LLM-graph interaction mode, comparing prompting, tool-use, and code generation; dataset domains, spanning citation, web-link, e-commerce, and social networks; structural regimes contrasting homophilic and heterophilic graphs; feature characteristics involving both short- and long-text node attributes; and model configurations with varying LLM sizes and reasoning capabilities. We further analyze dependencies by methodically truncating features, deleting edges, and removing labels to quantify reliance on input types. Our findings provide practical and actionable guidance. (1) LLMs as code generators achieve the strongest overall performance on graph data, with especially large gains on long-text or high-degree graphs where prompting quickly exceeds the token budget. (2) All interaction strategies remain effective on heterophilic graphs, challenging the assumption that LLM-based methods collapse under low homophily. (3) Code generation is able to flexibly adapt its reliance between structure, features, or labels to leverage the most informative input type. Together, these findings provide a comprehensive view of the strengths and limitations of current LLM-graph interaction modes and highlight key design principles for future approaches.
>
---
#### [new 056] MemOrb: A Plug-and-Play Verbal-Reinforcement Memory Layer for E-Commerce Customer Service
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对电商客服中LLM代理的可靠性问题，提出MemOrb——一种无需微调的轻量级记忆层。通过存储多轮对话的策略反思，提升任务成功率与稳定性，实验显示多轮成功率达63%提升。属于对话系统优化任务。**

- **链接: [http://arxiv.org/pdf/2509.18713v1](http://arxiv.org/pdf/2509.18713v1)**

> **作者:** Yizhe Huang; Yang Liu; Ruiyu Zhao; Xiaolong Zhong; Xingming Yue; Ling Jiang
>
> **摘要:** Large Language Model-based agents(LLM-based agents) are increasingly deployed in customer service, yet they often forget across sessions, repeat errors, and lack mechanisms for continual self-improvement. This makes them unreliable in dynamic settings where stability and consistency are critical. To better evaluate these properties, we emphasize two indicators: task success rate as a measure of overall effectiveness, and consistency metrics such as Pass$^k$ to capture reliability across multiple trials. To address the limitations of existing approaches, we propose MemOrb, a lightweight and plug-and-play verbal reinforcement memory layer that distills multi-turn interactions into compact strategy reflections. These reflections are stored in a shared memory bank and retrieved to guide decision-making, without requiring any fine-tuning. Experiments show that MemOrb significantly improves both success rate and stability, achieving up to a 63 percentage-point gain in multi-turn success rate and delivering more consistent performance across repeated trials. Our results demonstrate that structured reflection is a powerful mechanism for enhancing long-term reliability of frozen LLM agents in customer service scenarios.
>
---
#### [new 057] Investigating Test-Time Scaling with Reranking for Machine Translation
- **分类: cs.CL**

- **简介: 该论文研究了机器翻译中的测试时扩展（TTS）方法，通过生成多个候选译文并选择最优解来提升质量。作者在WMT24基准上系统评估了不同模型规模和计算预算下的效果，发现TTS在高资源语言中有效，且小模型在高N值下可媲美大模型。**

- **链接: [http://arxiv.org/pdf/2509.19020v1](http://arxiv.org/pdf/2509.19020v1)**

> **作者:** Shaomu Tan; Ryosuke Mitani; Ritvik Choudhary; Toshiyuki Sekiya
>
> **摘要:** Scaling model parameters has become the de facto strategy for improving NLP systems, but it comes with substantial computational costs. Test-Time Scaling (TTS) offers an alternative by allocating more computation at inference: generating multiple candidates and selecting the best. While effective in tasks such as mathematical reasoning, TTS has not been systematically explored for machine translation (MT). In this paper, we present the first systematic study of TTS for MT, investigating a simple but practical best-of-N framework on WMT24 benchmarks. Our experiments cover six high-resource and one low-resource language pairs, five model sizes (3B-72B), and various TTS compute budget (N up to 1024). Our results show that a) For high-resource languages, TTS generally improves translation quality according to multiple neural MT evaluation metrics, and our human evaluation confirms these gains; b) Augmenting smaller models with large $N$ can match or surpass larger models at $N{=}1$ with more compute cost; c) Under fixed compute budgets, larger models are typically more efficient, and TTS can degrade quality due to metric blind spots in low-resource cases.
>
---
#### [new 058] Extracting Conceptual Spaces from LLMs Using Prototype Embeddings
- **分类: cs.CL**

- **简介: 该论文属于知识表示学习任务，旨在从大语言模型（LLMs）中提取概念空间。传统方法难以构建具有认知意义的特征维度，而该文提出通过原型嵌入编码特征，并微调模型以对齐概念空间维度，从而有效提取可解释的概念空间。**

- **链接: [http://arxiv.org/pdf/2509.19269v1](http://arxiv.org/pdf/2509.19269v1)**

> **作者:** Nitesh Kumar; Usashi Chatterjee; Steven Schockaert
>
> **摘要:** Conceptual spaces represent entities and concepts using cognitively meaningful dimensions, typically referring to perceptual features. Such representations are widely used in cognitive science and have the potential to serve as a cornerstone for explainable AI. Unfortunately, they have proven notoriously difficult to learn, although recent LLMs appear to capture the required perceptual features to a remarkable extent. Nonetheless, practical methods for extracting the corresponding conceptual spaces are currently still lacking. While various methods exist for extracting embeddings from LLMs, extracting conceptual spaces also requires us to encode the underlying features. In this paper, we propose a strategy in which features (e.g. sweetness) are encoded by embedding the description of a corresponding prototype (e.g. a very sweet food). To improve this strategy, we fine-tune the LLM to align the prototype embeddings with the corresponding conceptual space dimensions. Our empirical analysis finds this approach to be highly effective.
>
---
#### [new 059] Thinking in a Crowd: How Auxiliary Information Shapes LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文研究了辅助信息对具备逐步推理能力的大语言模型（LLMs）的影响。提出了SciAux数据集，发现误导性信息会显著降低模型性能，强调模型需具备信息评估能力以提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.18163v1](http://arxiv.org/pdf/2509.18163v1)**

> **作者:** Haodong Zhao; Chenyan Zhao; Yansi Li; Zhuosheng Zhang; Gongshen Liu
>
> **备注:** Work in progress
>
> **摘要:** The capacity of Large Language Models (LLMs) to reason is fundamental to their application in complex, knowledge-intensive domains. In real-world scenarios, LLMs are often augmented with external information that can be helpful, irrelevant, or even misleading. This paper investigates the causal impact of such auxiliary information on the reasoning process of LLMs with explicit step-by-step thinking capabilities. We introduce SciAux, a new dataset derived from ScienceQA, to systematically test the robustness of the model against these types of information. Our findings reveal a critical vulnerability: the model's deliberative "thinking mode" is a double-edged sword. While helpful context improves accuracy, misleading information causes a catastrophic drop in performance, which is amplified by the thinking process. Instead of conferring robustness, thinking reinforces the degree of error when provided with misinformation. This highlights that the challenge is not merely to make models "think", but to endow them with the critical faculty to evaluate the information upon which their reasoning is based. The SciAux dataset is available at https://huggingface.co/datasets/billhdzhao/SciAux.
>
---
#### [new 060] Event Causality Identification with Synthetic Control
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究事件因果性识别（ECI）任务，旨在从文本中准确区分事件间的因果与相关关系。针对传统方法易产生错误因果的问题，提出基于合成控制的方法，通过生成虚拟对照个体来估算因果效应，从而提升因果识别的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.18156v1](http://arxiv.org/pdf/2509.18156v1)**

> **作者:** Haoyu Wang; Fengze Liu; Jiayao Zhang; Dan Roth; Kyle Richardson
>
> **摘要:** Event causality identification (ECI), a process that extracts causal relations between events from text, is crucial for distinguishing causation from correlation. Traditional approaches to ECI have primarily utilized linguistic patterns and multi-hop relational inference, risking false causality identification due to informal usage of causality and specious graphical inference. In this paper, we adopt the Rubin Causal Model to identify event causality: given two temporally ordered events, we see the first event as the treatment and the second one as the observed outcome. Determining their causality involves manipulating the treatment and estimating the resultant change in the likelihood of the outcome. Given that it is only possible to implement manipulation conceptually in the text domain, as a work-around, we try to find a twin for the protagonist from existing corpora. This twin should have identical life experiences with the protagonist before the treatment but undergoes an intervention of treatment. However, the practical difficulty of locating such a match limits its feasibility. Addressing this issue, we use the synthetic control method to generate such a twin' from relevant historical data, leveraging text embedding synthesis and inversion techniques. This approach allows us to identify causal relations more robustly than previous methods, including GPT-4, which is demonstrated on a causality benchmark, COPES-hard.
>
---
#### [new 061] HarmoniFuse: A Component-Selective and Prompt-Adaptive Framework for Multi-Task Speech Language Modeling
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出HarmoniFuse，一种多任务语音语言建模框架，旨在解决ASR和SER任务因信息需求不同导致的性能下降问题。通过组件选择与提示自适应机制，实现任务相关特征的融合与优化，提升模型在有限数据下的鲁棒性与效果。**

- **链接: [http://arxiv.org/pdf/2509.18570v1](http://arxiv.org/pdf/2509.18570v1)**

> **作者:** Yuke Si; Runyan Yang; Yingying Gao; Junlan Feng; Chao Deng; Shilei Zhang
>
> **备注:** 5 pages; submitted to ICASSP 2026
>
> **摘要:** Recent advances in large language models have facilitated the development of unified speech language models (SLMs) capable of supporting multiple speech tasks within a shared architecture. However, tasks such as automatic speech recognition (ASR) and speech emotion recognition (SER) rely on distinct types of information: ASR primarily depends on linguistic content, whereas SER requires the integration of both linguistic and paralinguistic cues. Existing multitask SLMs typically adopt naive parameter sharing or prompt-based conditioning without explicitly modeling the differences in information composition required by each task. Such designs risk task interference and performance degradation, especially under limited data conditions. To address these limitations, we propose HarmoniFuse, a component-selective and prompt-adaptive framework for multi-task speech language modeling. HarmoniFuse is designed to harmonize heterogeneous task demands by selecting and fusing task-relevant components of speech representations. Specifically, it integrates a gated speech encoder to extract task-specific acoustic features and a prompt-adaptive dynamic fusion module to aggregate transformer layers based on task characteristics. In addition, a batch-interleaved training strategy enables leveraging separate ASR and SER datasets without requiring joint annotation. Experimental results demonstrate that HarmoniFuse improves both ASR and SER performance, offering a scalable and robust solution for multitask speech understanding under realistic data constraints.
>
---
#### [new 062] Finding My Voice: Generative Reconstruction of Disordered Speech for Automated Clinical Evaluation
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文提出ChiReSSD，用于重建儿童语音障碍者的语音，在抑制发音错误的同时保留说话人身份。通过解耦风格的TTS方法，实现了语音质量与身份保持的提升，并在临床评估中表现出良好的自动评分相关性。**

- **链接: [http://arxiv.org/pdf/2509.19231v1](http://arxiv.org/pdf/2509.19231v1)**

> **作者:** Karen Rosero; Eunjung Yeo; David R. Mortensen; Cortney Van't Slot; Rami R. Hallac; Carlos Busso
>
> **摘要:** We present ChiReSSD, a speech reconstruction framework that preserves children speaker's identity while suppressing mispronunciations. Unlike prior approaches trained on healthy adult speech, ChiReSSD adapts to the voices of children with speech sound disorders (SSD), with particular emphasis on pitch and prosody. We evaluate our method on the STAR dataset and report substantial improvements in lexical accuracy and speaker identity preservation. Furthermore, we automatically predict the phonetic content in the original and reconstructed pairs, where the proportion of corrected consonants is comparable to the percentage of correct consonants (PCC), a clinical speech assessment metric. Our experiments show Pearson correlation of 0.63 between automatic and human expert annotations, highlighting the potential to reduce the manual transcription burden. In addition, experiments on the TORGO dataset demonstrate effective generalization for reconstructing adult dysarthric speech. Our results indicate that disentangled, style-based TTS reconstruction can provide identity-preserving speech across diverse clinical populations.
>
---
#### [new 063] Pay More Attention To Audio: Mitigating Imbalance of Cross-Modal Attention in Large Audio Language Models
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 该论文针对大音视语言模型（LALM）中音频-文本注意力不平衡问题，提出训练免费方法MATA。通过动态增强音频token的注意力权重，提升音频推理性能，在MMAU和MMAR基准上取得显著效果。**

- **链接: [http://arxiv.org/pdf/2509.18816v1](http://arxiv.org/pdf/2509.18816v1)**

> **作者:** Junyu Wang; Ziyang Ma; Zhengding Luo; Tianrui Wang; Meng Ge; Xiaobao Wang; Longbiao Wang
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Large Audio-Language Models (LALMs) often suffer from audio-textual attention imbalance, prioritizing text over acoustic information, particularly in the multi-modal fusion layers of the Transformer architecture. This bias hinders their ability to fully utilize acoustic cues, causing suboptimal performance on audio reasoning tasks. To mitigate this, we propose \textbf{MATA}, a novel training-free method that dynamically pushes LALMs to pay \textbf{M}ore \textbf{A}ttention \textbf{T}o \textbf{A}udio tokens within the self-attention mechanism. Specifically, MATA intervenes post raw attention scoring, targeting only the last token in intermediate layers without introducing additional parameters or computational overhead. Experiments on the MMAU and MMAR benchmarks confirm MATA's effectiveness, with consistent performance gains. Notably, on MMAR, MATA enables an open-source model to surpass the proprietary Gemini 2.0 Flash for the first time. Our work provides an efficient solution to mitigate attention bias and opens a new research direction for enhancing the audio-processing capabilities of multi-modal models.
>
---
#### [new 064] Teaching Audio Models to Reason: A Unified Framework for Source- and Layer-wise Distillation
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出一种统一的知识蒸馏框架，用于将文本模型的推理能力迁移至音频模型。针对音频与文本模态差异及缺乏结构监督的问题，引入源级和层级蒸馏策略，提升音频模型的推理性能。属于音频建模任务。**

- **链接: [http://arxiv.org/pdf/2509.18579v1](http://arxiv.org/pdf/2509.18579v1)**

> **作者:** Runyan Yang; Yuke Si; Yingying Gao; Junlan Feng; Chao Deng; Shilei Zhang
>
> **备注:** 5 pages; submitted to ICASSP 2026
>
> **摘要:** While large audio language models excel at tasks like ASR and emotion recognition, they still struggle with complex reasoning due to the modality gap between audio and text as well as the lack of structured intermediate supervision. To address this, we propose a unified knowledge distillation framework to transfer reasoning capabilities from a high-capacity textual teacher model to a student audio models while preserving its acoustic competence. Our method introduces two key dimensions: source-wise distillation, which leverages both textual and acoustic teachers to provide complementary modality-specific supervision; and layer-wise distillation, which aligns teacher signals with appropriate student layers to improve transfer efficiency. This dual-dimensional strategy enables fine-grained control over the distillation process, effectively bridging the gap between symbolic reasoning and speech representations. Experimental results show significant improvements in audio reasoning performance, demonstrating the effectiveness of our framework as a reasoning transfer solution for audio modeling.
>
---
#### [new 065] PiMoE: Token-Level Routing for Integrating High-Precision Computation and Reasoning
- **分类: cs.LG; cs.CE; cs.CL**

- **简介: 该论文提出PiMoE架构，旨在解决大语言模型无法高效集成高精度计算与推理的问题。通过在神经网络中内生整合计算能力，实现token级路由，提升准确性、响应速度和能效，适用于科学或工业智能系统任务。**

- **链接: [http://arxiv.org/pdf/2509.18169v1](http://arxiv.org/pdf/2509.18169v1)**

> **作者:** Hengbo Xiao; Jingyuan Fan; Xin Tong; Jingzhao Zhang; Chao Lu; Guannan He
>
> **摘要:** Complex systems typically rely on high-precision numerical computation to support decisions, but current large language models (LLMs) cannot yet incorporate such computations as an intrinsic and interpretable capability with existing architectures. Mainstream multi-agent approaches can leverage external experts, but inevitably introduce communication overhead and suffer from inefficient multimodal emergent capability and limited scalability. To this end, we propose PiMoE (Physically-isolated Mixture of Experts), a training and inference architecture for integrating computation and reasoning. Instead of the workflow paradigm of tool invocation, PiMoE endogenously integrates computational capabilities into neural networks after separately training experts, a text-to-computation module, and a router. At inference, the router directs computation and reasoning at the token level, thereby enabling iterative alternation within a single chain of thought. We evaluate PiMoE on two reasoning-computation tasks against LLM finetuning and the multi-agent system approaches. Results show that the PiMoE architecture achieves not only higher accuracy than directly finetuning LLMs but also significant improvements in response latency, token usage, and GPU energy consumption compared with mainstream multi-agent approaches. PiMoE offers an efficient, interpretable, and scalable paradigm for next-generation scientific or industrial intelligent systems.
>
---
#### [new 066] Citrus-V: Advancing Medical Foundation Models with Unified Medical Image Grounding for Clinical Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出Citrus-V，一种统一的医疗多模态基础模型，融合图像分析与文本推理，解决现有模型泛化能力差的问题。工作包括：集成检测、分割与多模态推理，设计新训练方法，并发布开源数据集，实现从视觉定位到临床诊断的全流程任务。**

- **链接: [http://arxiv.org/pdf/2509.19090v1](http://arxiv.org/pdf/2509.19090v1)**

> **作者:** Guoxin Wang; Jun Zhao; Xinyi Liu; Yanbo Liu; Xuyang Cao; Chao Li; Zhuoyun Liu; Qintian Sun; Fangru Zhou; Haoqiang Xing; Zhenhong Yang
>
> **摘要:** Medical imaging provides critical evidence for clinical diagnosis, treatment planning, and surgical decisions, yet most existing imaging models are narrowly focused and require multiple specialized networks, limiting their generalization. Although large-scale language and multimodal models exhibit strong reasoning and multi-task capabilities, real-world clinical applications demand precise visual grounding, multimodal integration, and chain-of-thought reasoning. We introduce Citrus-V, a multimodal medical foundation model that combines image analysis with textual reasoning. The model integrates detection, segmentation, and multimodal chain-of-thought reasoning, enabling pixel-level lesion localization, structured report generation, and physician-like diagnostic inference in a single framework. We propose a novel multimodal training approach and release a curated open-source data suite covering reasoning, detection, segmentation, and document understanding tasks. Evaluations demonstrate that Citrus-V outperforms existing open-source medical models and expert-level imaging systems across multiple benchmarks, delivering a unified pipeline from visual grounding to clinical reasoning and supporting precise lesion quantification, automated reporting, and reliable second opinions.
>
---
#### [new 067] Cross-Cultural Transfer of Commonsense Reasoning in LLMs: Evidence from the Arab World
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究大语言模型（LLMs）在阿拉伯世界的跨文化常识推理迁移问题。针对西方偏见限制模型在多元文化中的表现，作者使用轻量级对齐方法，在13个阿拉伯国家数据上验证了少量文化示例可提升跨文化性能，证明文化常识的可迁移性。**

- **链接: [http://arxiv.org/pdf/2509.19265v1](http://arxiv.org/pdf/2509.19265v1)**

> **作者:** Saeed Almheiri; Rania Hossam; Mena Attia; Chenxi Wang; Preslav Nakov; Timothy Baldwin; Fajri Koto
>
> **备注:** EMNLP 2025 - Findings
>
> **摘要:** Large language models (LLMs) often reflect Western-centric biases, limiting their effectiveness in diverse cultural contexts. Although some work has explored cultural alignment, the potential for cross-cultural transfer, using alignment in one culture to improve performance in others, remains underexplored. This paper investigates cross-cultural transfer of commonsense reasoning in the Arab world, where linguistic and historical similarities coexist with local cultural differences. Using a culturally grounded commonsense reasoning dataset covering 13 Arab countries, we evaluate lightweight alignment methods such as in-context learning and demonstration-based reinforcement (DITTO), alongside baselines like supervised fine-tuning and direct preference optimization. Our results show that merely 12 culture-specific examples from one country can improve performance in others by 10\% on average, within multilingual models. In addition, we demonstrate that out-of-culture demonstrations from Indonesia and US contexts can match or surpass in-culture alignment for MCQ reasoning, highlighting cultural commonsense transferability beyond the Arab world. These findings demonstrate that efficient cross-cultural alignment is possible and offer a promising approach to adapt LLMs to low-resource cultural settings.
>
---
#### [new 068] VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出了VIR-Bench，一个用于评估多模态大语言模型（MLLMs）在长距离旅行视频中地理时空理解能力的新基准。通过200个旅行视频的行程重建任务，揭示了当前MLLMs在处理大规模时空信息时的不足，并验证了改进后的旅行规划代理的有效性。**

- **链接: [http://arxiv.org/pdf/2509.19002v1](http://arxiv.org/pdf/2509.19002v1)**

> **作者:** Hao Wang; Eiki Murata; Lingfang Zhang; Ayako Sato; So Fukuda; Ziqi Yin; Wentao Hu; Keisuke Nakao; Yusuke Nakamura; Sebastian Zwirner; Yi-Chia Chen; Hiroyuki Otomo; Hiroki Ouchi; Daisuke Kawahara
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have significantly enhanced video understanding capabilities, opening new possibilities for practical applications. Yet current video benchmarks focus largely on indoor scenes or short-range outdoor activities, leaving the challenges associated with long-distance travel largely unexplored. Mastering extended geospatial-temporal trajectories is critical for next-generation MLLMs, underpinning real-world tasks such as embodied-AI planning and navigation. To bridge this gap, we present VIR-Bench, a novel benchmark consisting of 200 travel videos that frames itinerary reconstruction as a challenging task designed to evaluate and push forward MLLMs' geospatial-temporal intelligence. Experimental results reveal that state-of-the-art MLLMs, including proprietary ones, struggle to achieve high scores, underscoring the difficulty of handling videos that span extended spatial and temporal scales. Moreover, we conduct an in-depth case study in which we develop a prototype travel-planning agent that leverages the insights gained from VIR-Bench. The agent's markedly improved itinerary recommendations verify that our evaluation protocol not only benchmarks models effectively but also translates into concrete performance gains in user-facing applications.
>
---
#### [new 069] OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦于放射学报告生成任务，旨在解决数据和计算资源消耗大的问题。提出OraPO方法，结合FactScore奖励机制，实现单阶段、高效的强化学习训练，在少量数据下取得SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.18600v1](http://arxiv.org/pdf/2509.18600v1)**

> **作者:** Zhuoxiao Chen; Hongyang Yu; Ying Xu; Yadan Luo; Long Duong; Yuan-Fang Li
>
> **摘要:** Radiology report generation (RRG) aims to automatically produce clinically faithful reports from chest X-ray images. Prevailing work typically follows a scale-driven paradigm, by multi-stage training over large paired corpora and oversized backbones, making pipelines highly data- and compute-intensive. In this paper, we propose Oracle-educated GRPO {OraPO) with a FactScore-based reward (FactS) to tackle the RRG task under constrained budgets. OraPO enables single-stage, RL-only training by converting failed GRPO explorations on rare or difficult studies into direct preference supervision via a lightweight oracle step. FactS grounds learning in diagnostic evidence by extracting atomic clinical facts and checking entailment against ground-truth labels, yielding dense, interpretable sentence-level rewards. Together, OraPO and FactS create a compact and powerful framework that significantly improves learning efficiency on clinically challenging cases, setting the new SOTA performance on the CheXpert Plus dataset (0.341 in F1) with 2--3 orders of magnitude less training data using a small base VLM on modest hardware.
>
---
#### [new 070] Failure Makes the Agent Stronger: Enhancing Accuracy through Structured Reflection for Reliable Tool Interactions
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对工具增强型大语言模型在多轮交互中错误恢复能力差的问题，提出结构化反思方法。通过显式训练“诊断-修正”流程，提升工具调用的可靠性与准确性，构建了评估基准Tool-Reflection-Bench，实验显示显著提升了多轮任务成功率和错误修复能力。**

- **链接: [http://arxiv.org/pdf/2509.18847v1](http://arxiv.org/pdf/2509.18847v1)**

> **作者:** Junhao Su; Yuanliang Wan; Junwei Yang; Hengyu Shi; Tianyang Han; Junfeng Luo; Yurui Qiu
>
> **备注:** 9pages
>
> **摘要:** Tool-augmented large language models (LLMs) are usually trained with supervised imitation or coarse-grained reinforcement learning that optimizes single tool calls. Current self-reflection practices rely on heuristic prompts or one-way reasoning: the model is urged to 'think more' instead of learning error diagnosis and repair. This is fragile in multi-turn interactions; after a failure the model often repeats the same mistake. We propose structured reflection, which turns the path from error to repair into an explicit, controllable, and trainable action. The agent produces a short yet precise reflection: it diagnoses the failure using evidence from the previous step and then proposes a correct, executable follow-up call. For training we combine DAPO and GSPO objectives with a reward scheme tailored to tool use, optimizing the stepwise strategy Reflect, then Call, then Final. To evaluate, we introduce Tool-Reflection-Bench, a lightweight benchmark that programmatically checks structural validity, executability, parameter correctness, and result consistency. Tasks are built as mini trajectories of erroneous call, reflection, and corrected call, with disjoint train and test splits. Experiments on BFCL v3 and Tool-Reflection-Bench show large gains in multi-turn tool-call success and error recovery, and a reduction of redundant calls. These results indicate that making reflection explicit and optimizing it directly improves the reliability of tool interaction and offers a reproducible path for agents to learn from failure.
>
---
#### [new 071] Conversational Orientation Reasoning: Egocentric-to-Allocentric Navigation with Multimodal Chain-of-Thought
- **分类: cs.LG; cs.AI; cs.CL; cs.RO**

- **简介: 该论文聚焦于对话导航中的自体-他体方向推理任务，旨在解决中文对话中将自体方位（如“我的右边”）转换为绝对方向（N/E/S/W）的问题。提出了COR数据集和MCoT框架，结合语音识别与坐标信息，通过结构化三步推理实现高精度方向推断，适用于资源受限环境。**

- **链接: [http://arxiv.org/pdf/2509.18200v1](http://arxiv.org/pdf/2509.18200v1)**

> **作者:** Yu Ti Huang
>
> **摘要:** Conversational agents must translate egocentric utterances (e.g., "on my right") into allocentric orientations (N/E/S/W). This challenge is particularly critical in indoor or complex facilities where GPS signals are weak and detailed maps are unavailable. While chain-of-thought (CoT) prompting has advanced reasoning in language and vision tasks, its application to multimodal spatial orientation remains underexplored. We introduce Conversational Orientation Reasoning (COR), a new benchmark designed for Traditional Chinese conversational navigation projected from real-world environments, addressing egocentric-to-allocentric reasoning in non-English and ASR-transcribed scenarios. We propose a multimodal chain-of-thought (MCoT) framework, which integrates ASR-transcribed speech with landmark coordinates through a structured three-step reasoning process: (1) extracting spatial relations, (2) mapping coordinates to absolute directions, and (3) inferring user orientation. A curriculum learning strategy progressively builds these capabilities on Taiwan-LLM-13B-v2.0-Chat, a mid-sized model representative of resource-constrained settings. Experiments show that MCoT achieves 100% orientation accuracy on clean transcripts and 98.1% with ASR transcripts, substantially outperforming unimodal and non-structured baselines. Moreover, MCoT demonstrates robustness under noisy conversational conditions, including ASR recognition errors and multilingual code-switching. The model also maintains high accuracy in cross-domain evaluation and resilience to linguistic variation, domain shift, and referential ambiguity. These findings highlight the potential of structured MCoT spatial reasoning as a path toward interpretable and resource-efficient embodied navigation.
>
---
#### [new 072] Safe-SAIL: Towards a Fine-grained Safety Landscape of Large Language Models via Sparse Autoencoder Interpretation Framework
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Safe-SAIL框架，通过稀疏自编码器（SAE）解释大语言模型的安全相关特征，旨在系统识别并分析高风险行为，解决现有方法在安全概念层面解释不足的问题。**

- **链接: [http://arxiv.org/pdf/2509.18127v1](http://arxiv.org/pdf/2509.18127v1)**

> **作者:** Jiaqi Weng; Han Zheng; Hanyu Zhang; Qinqin He; Jialing Tao; Hui Xue; Zhixuan Chu; Xiting Wang
>
> **摘要:** Increasing deployment of large language models (LLMs) in real-world applications raises significant safety concerns. Most existing safety research focuses on evaluating LLM outputs or specific safety tasks, limiting their ability to ad- dress broader, undefined risks. Sparse Autoencoders (SAEs) facilitate interpretability research to clarify model behavior by explaining single-meaning atomic features decomposed from entangled signals. jHowever, prior applications on SAEs do not interpret features with fine-grained safety-related con- cepts, thus inadequately addressing safety-critical behaviors, such as generating toxic responses and violating safety regu- lations. For rigorous safety analysis, we must extract a rich and diverse set of safety-relevant features that effectively capture these high-risk behaviors, yet face two challenges: identifying SAEs with the greatest potential for generating safety concept-specific neurons, and the prohibitively high cost of detailed feature explanation. In this paper, we pro- pose Safe-SAIL, a framework for interpreting SAE features within LLMs to advance mechanistic understanding in safety domains. Our approach systematically identifies SAE with best concept-specific interpretability, explains safety-related neurons, and introduces efficient strategies to scale up the in- terpretation process. We will release a comprehensive toolkit including SAE checkpoints and human-readable neuron ex- planations, which supports empirical analysis of safety risks to promote research on LLM safety.
>
---
#### [new 073] Agentic AutoSurvey: Let LLMs Survey LLMs
- **分类: cs.IR; cs.CL; cs.HC**

- **简介: 该论文提出Agentic AutoSurvey，一个多智能体框架用于自动化生成文献综述。针对现有方法在快速发展的LLM领域中的不足，设计四个专业智能体协同工作，提升综述的综合质量与覆盖度，在六个主题上实验验证了其优越性。**

- **链接: [http://arxiv.org/pdf/2509.18661v1](http://arxiv.org/pdf/2509.18661v1)**

> **作者:** Yixin Liu; Yonghui Wu; Denghui Zhang; Lichao Sun
>
> **备注:** 29 pages, 7 figures
>
> **摘要:** The exponential growth of scientific literature poses unprecedented challenges for researchers attempting to synthesize knowledge across rapidly evolving fields. We present \textbf{Agentic AutoSurvey}, a multi-agent framework for automated survey generation that addresses fundamental limitations in existing approaches. Our system employs four specialized agents (Paper Search Specialist, Topic Mining \& Clustering, Academic Survey Writer, and Quality Evaluator) working in concert to generate comprehensive literature surveys with superior synthesis quality. Through experiments on six representative LLM research topics from COLM 2024 categories, we demonstrate that our multi-agent approach achieves significant improvements over existing baselines, scoring 8.18/10 compared to AutoSurvey's 4.77/10. The multi-agent architecture processes 75--443 papers per topic (847 total across six topics) while targeting high citation coverage (often $\geq$80\% on 75--100-paper sets; lower on very large sets such as RLHF) through specialized agent orchestration. Our 12-dimension evaluation captures organization, synthesis integration, and critical analysis beyond basic metrics. These findings demonstrate that multi-agent architectures represent a meaningful advancement for automated literature survey generation in rapidly evolving scientific domains.
>
---
#### [new 074] TurnBack: A Geospatial Route Cognition Benchmark for Large Language Models through Reverse Route
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出TurnBack，一个用于评估大语言模型（LLMs）地理空间路径认知能力的基准。针对现有研究缺乏量化指标和数据集的问题，构建了包含36000条全球路线的数据集，并设计PathBuilder工具及新评估框架，揭示LLMs在路径逆向任务中的局限性。**

- **链接: [http://arxiv.org/pdf/2509.18173v1](http://arxiv.org/pdf/2509.18173v1)**

> **作者:** Hongyi Luo; Qing Cheng; Daniel Matos; Hari Krishna Gadi; Yanfeng Zhang; Lu Liu; Yongliang Wang; Niclas Zeller; Daniel Cremers; Liqiu Meng
>
> **备注:** Accepted to EMNLP 2025 (Main). This is the camera-ready/author version
>
> **摘要:** Humans can interpret geospatial information through natural language, while the geospatial cognition capabilities of Large Language Models (LLMs) remain underexplored. Prior research in this domain has been constrained by non-quantifiable metrics, limited evaluation datasets and unclear research hierarchies. Therefore, we propose a large-scale benchmark and conduct a comprehensive evaluation of the geospatial route cognition of LLMs. We create a large-scale evaluation dataset comprised of 36000 routes from 12 metropolises worldwide. Then, we introduce PathBuilder, a novel tool for converting natural language instructions into navigation routes, and vice versa, bridging the gap between geospatial information and natural language. Finally, we propose a new evaluation framework and metrics to rigorously assess 11 state-of-the-art (SOTA) LLMs on the task of route reversal. The benchmark reveals that LLMs exhibit limitation to reverse routes: most reverse routes neither return to the starting point nor are similar to the optimal route. Additionally, LLMs face challenges such as low robustness in route generation and high confidence for their incorrect answers. Code\ \&\ Data available here: \href{https://github.com/bghjmn32/EMNLP2025_Turnback}{TurnBack.}
>
---
#### [new 075] Memory-QA: Answering Recall Questions Based on Multimodal Memories
- **分类: cs.AI; cs.CL; cs.DB**

- **简介: 该论文提出Memory-QA任务，旨在基于多模态记忆回答视觉内容的回忆问题。针对记忆构建、时空信息利用及多记忆融合等挑战，设计了Pensieve系统，并构建了多模态基准，实验显示其在问答准确率上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.18436v1](http://arxiv.org/pdf/2509.18436v1)**

> **作者:** Hongda Jiang; Xinyuan Zhang; Siddhant Garg; Rishab Arora; Shiun-Zu Kuo; Jiayang Xu; Christopher Brossman; Yue Liu; Aaron Colak; Ahmed Aly; Anuj Kumar; Xin Luna Dong
>
> **摘要:** We introduce Memory-QA, a novel real-world task that involves answering recall questions about visual content from previously stored multimodal memories. This task poses unique challenges, including the creation of task-oriented memories, the effective utilization of temporal and location information within memories, and the ability to draw upon multiple memories to answer a recall question. To address these challenges, we propose a comprehensive pipeline, Pensieve, integrating memory-specific augmentation, time- and location-aware multi-signal retrieval, and multi-memory QA fine-tuning. We created a multimodal benchmark to illustrate various real challenges in this task, and show the superior performance of Pensieve over state-of-the-art solutions (up to 14% on QA accuracy).
>
---
#### [new 076] No Verifiable Reward for Prosody: Toward Preference-Guided Prosody Learning in TTS
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文针对TTS任务中韵律自然性不足的问题，提出一种基于少量人类偏好数据的迭代DPO方法，在KoCC-TTS数据集上实现了优于GRPO和商业基线的自然语音生成效果。**

- **链接: [http://arxiv.org/pdf/2509.18531v1](http://arxiv.org/pdf/2509.18531v1)**

> **作者:** Seungyoun Shin; Dongha Ahn; Jiwoo Kim; Sungwook Jeon
>
> **备注:** submitted to ICASSP 2026
>
> **摘要:** Recent work reports gains in neural text-to-speech (TTS) with Group Relative Policy Optimization (GRPO). However, in the absence of a verifiable reward for \textit{prosody}, GRPO trained on transcription-oriented signals (CER/NLL) lowers error rates yet collapses prosody into monotone, unnatural speech; adding speaker-similarity further destabilizes training and degrades CER. We address this with an \textit{iterative Direct Preference Optimization (DPO)} scheme that uses only a few hundred human-labeled preference pairs per round to directly optimize prosodic naturalness while regularizing to the current model. On \textbf{KoCC-TTS}, a curated dataset of authentic Korean call center interactions capturing task-oriented dialogues, our method attains the highest human preference (ELO) with competitive CER, outperforming GRPO and strong commercial baselines. These results suggest that when prosody cannot be rewarded automatically, \textit{human preference optimization} offers a practical and data-efficient path to natural and robust TTS. The demo page is available at \href{https://tts.ch.dev}
>
---
#### [new 077] Baseer: A Vision-Language Model for Arabic Document-to-Markdown OCR
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Baseer，一个针对阿拉伯语文档OCR的视觉-语言模型。为解决阿拉伯语OCR因连写、字体多样和从右到左书写带来的挑战，作者采用特定领域微调策略，并构建高质量基准Misraj-DocOCR。实验表明Baseer在OCR准确率上达到新水平。**

- **链接: [http://arxiv.org/pdf/2509.18174v1](http://arxiv.org/pdf/2509.18174v1)**

> **作者:** Khalil Hennara; Muhammad Hreden; Mohamed Motasim Hamed; Ahmad Bastati; Zeina Aldallal; Sara Chrouf; Safwan AlModhayan
>
> **摘要:** Arabic document OCR remains a challenging task due to the language's cursive script, diverse fonts, diacritics, and right-to-left orientation. While modern Multimodal Large Language Models (MLLMs) have advanced document understanding for high-resource languages, their performance on Arabic remains limited. In this work, we introduce Baseer, a vision-language model fine- tuned specifically for Arabic document OCR. Leveraging a large-scale dataset combining synthetic and real-world documents, Baseer is trained using a decoder-only fine-tuning strategy to adapt a pre-trained MLLM while preserving general visual features. We also present Misraj-DocOCR, a high-quality, expert-verified benchmark designed for rigorous evaluation of Arabic OCR systems. Our experiments show that Baseer significantly outperforms existing open-source and commercial solutions, achieving a WER of 0.25 and establishing a new state-of-the-art in the domain of Arabic document OCR. Our results highlight the benefits of domain-specific adaptation of general-purpose MLLMs and establish a strong baseline for high-accuracy OCR on morphologically rich languages like Arabic.
>
---
#### [new 078] ColorBlindnessEval: Can Vision-Language Models Pass Color Blindness Tests?
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出ColorBlindnessEval，一个基于色盲测试的视觉-语言模型（VLM）鲁棒性评估基准。通过500张类似Ishihara的图像，测试VLM在复杂视觉模式下识别数字的能力，并与人类表现对比，揭示模型在对抗性场景中的局限性和幻觉问题，旨在提升VLM在现实应用中的可靠性。**

- **链接: [http://arxiv.org/pdf/2509.19070v1](http://arxiv.org/pdf/2509.19070v1)**

> **作者:** Zijian Ling; Han Zhang; Yazhuo Zhou; Jiahao Cui
>
> **备注:** Accepted at the Open Science for Foundation Models (SCI-FM) Workshop at ICLR 2025
>
> **摘要:** This paper presents ColorBlindnessEval, a novel benchmark designed to evaluate the robustness of Vision-Language Models (VLMs) in visually adversarial scenarios inspired by the Ishihara color blindness test. Our dataset comprises 500 Ishihara-like images featuring numbers from 0 to 99 with varying color combinations, challenging VLMs to accurately recognize numerical information embedded in complex visual patterns. We assess 9 VLMs using Yes/No and open-ended prompts and compare their performance with human participants. Our experiments reveal limitations in the models' ability to interpret numbers in adversarial contexts, highlighting prevalent hallucination issues. These findings underscore the need to improve the robustness of VLMs in complex visual environments. ColorBlindnessEval serves as a valuable tool for benchmarking and improving the reliability of VLMs in real-world applications where accuracy is critical.
>
---
#### [new 079] The Illusion of Readiness: Stress Testing Large Frontier Models on Multimodal Medical Benchmarks
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI在医疗领域的评估任务，旨在揭示当前主流大模型在医学基准测试中高分背后的脆弱性。作者通过压力测试发现模型依赖技巧而非真实理解，并指出基准测试存在误导性，强调需提升模型的鲁棒性和医学合理性。**

- **链接: [http://arxiv.org/pdf/2509.18234v1](http://arxiv.org/pdf/2509.18234v1)**

> **作者:** Yu Gu; Jingjing Fu; Xiaodong Liu; Jeya Maria Jose Valanarasu; Noel Codella; Reuben Tan; Qianchu Liu; Ying Jin; Sheng Zhang; Jinyu Wang; Rui Wang; Lei Song; Guanghui Qin; Naoto Usuyama; Cliff Wong; Cheng Hao; Hohin Lee; Praneeth Sanapathi; Sarah Hilado; Bian Jiang; Javier Alvarez-Valle; Mu Wei; Jianfeng Gao; Eric Horvitz; Matt Lungren; Hoifung Poon; Paul Vozila
>
> **备注:** 35 pages
>
> **摘要:** Large frontier models like GPT-5 now achieve top scores on medical benchmarks. But our stress tests tell a different story. Leading systems often guess correctly even when key inputs like images are removed, flip answers under trivial prompt changes, and fabricate convincing yet flawed reasoning. These aren't glitches; they expose how today's benchmarks reward test-taking tricks over medical understanding. We evaluate six flagship models across six widely used benchmarks and find that high leaderboard scores hide brittleness and shortcut learning. Through clinician-guided rubric evaluation, we show that benchmarks vary widely in what they truly measure yet are treated interchangeably, masking failure modes. We caution that medical benchmark scores do not directly reflect real-world readiness. If we want AI to earn trust in healthcare, we must demand more than leaderboard wins and must hold systems accountable for robustness, sound reasoning, and alignment with real medical demands.
>
---
## 更新

#### [replaced 001] Toxicity Red-Teaming: Benchmarking LLM Safety in Singapore's Low-Resource Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.15260v2](http://arxiv.org/pdf/2509.15260v2)**

> **作者:** Yujia Hu; Ming Shan Hee; Preslav Nakov; Roy Ka-Wei Lee
>
> **备注:** 9 pages, EMNLP 2025
>
> **摘要:** The advancement of Large Language Models (LLMs) has transformed natural language processing; however, their safety mechanisms remain under-explored in low-resource, multilingual settings. Here, we aim to bridge this gap. In particular, we introduce \textsf{SGToxicGuard}, a novel dataset and evaluation framework for benchmarking LLM safety in Singapore's diverse linguistic context, including Singlish, Chinese, Malay, and Tamil. SGToxicGuard adopts a red-teaming approach to systematically probe LLM vulnerabilities in three real-world scenarios: \textit{conversation}, \textit{question-answering}, and \textit{content composition}. We conduct extensive experiments with state-of-the-art multilingual LLMs, and the results uncover critical gaps in their safety guardrails. By offering actionable insights into cultural sensitivity and toxicity mitigation, we lay the foundation for safer and more inclusive AI systems in linguistically diverse environments.\footnote{Link to the dataset: https://github.com/Social-AI-Studio/SGToxicGuard.} \textcolor{red}{Disclaimer: This paper contains sensitive content that may be disturbing to some readers.}
>
---
#### [replaced 002] LaMP-Cap: Personalized Figure Caption Generation With Multimodal Figure Profiles
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.06561v4](http://arxiv.org/pdf/2506.06561v4)**

> **作者:** Ho Yin 'Sam' Ng; Ting-Yao Hsu; Aashish Anantha Ramakrishnan; Branislav Kveton; Nedim Lipka; Franck Dernoncourt; Dongwon Lee; Tong Yu; Sungchul Kim; Ryan A. Rossi; Ting-Hao 'Kenneth' Huang
>
> **备注:** Accepted to EMNLP 2025 Findings. The LaMP-CAP dataset is publicly available at: https://github.com/Crowd-AI-Lab/lamp-cap
>
> **摘要:** Figure captions are crucial for helping readers understand and remember a figure's key message. Many models have been developed to generate these captions, helping authors compose better quality captions more easily. Yet, authors almost always need to revise generic AI-generated captions to match their writing style and the domain's style, highlighting the need for personalization. Despite language models' personalization (LaMP) advances, these technologies often focus on text-only settings and rarely address scenarios where both inputs and profiles are multimodal. This paper introduces LaMP-Cap, a dataset for personalized figure caption generation with multimodal figure profiles. For each target figure, LaMP-Cap provides not only the needed inputs, such as figure images, but also up to three other figures from the same document--each with its image, caption, and figure-mentioning paragraphs--as a profile to characterize the context. Experiments with four LLMs show that using profile information consistently helps generate captions closer to the original author-written ones. Ablation studies reveal that images in the profile are more helpful than figure-mentioning paragraphs, highlighting the advantage of using multimodal profiles over text-only ones.
>
---
#### [replaced 003] Is Pre-training Truly Better Than Meta-Learning?
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2306.13841v2](http://arxiv.org/pdf/2306.13841v2)**

> **作者:** Brando Miranda; Patrick Yu; Saumya Goyal; Yu-Xiong Wang; Sanmi Koyejo
>
> **摘要:** In the context of few-shot learning, it is currently believed that a fixed pre-trained (PT) model, along with fine-tuning the final layer during evaluation, outperforms standard meta-learning algorithms. We re-evaluate these claims under an in-depth empirical examination of an extensive set of formally diverse datasets and compare PT to Model Agnostic Meta-Learning (MAML). Unlike previous work, we emphasize a fair comparison by using: the same architecture, the same optimizer, and all models trained to convergence. Crucially, we use a more rigorous statistical tool -- the effect size (Cohen's d) -- to determine the practical significance of the difference between a model trained with PT vs. a MAML. We then use a previously proposed metric -- the diversity coefficient -- to compute the average formal diversity of a dataset. Using this analysis, we demonstrate the following: 1. when the formal diversity of a data set is low, PT beats MAML on average and 2. when the formal diversity is high, MAML beats PT on average. The caveat is that the magnitude of the average difference between a PT vs. MAML using the effect size is low (according to classical statistical thresholds) -- less than 0.2. Nevertheless, this observation is contrary to the currently held belief that a pre-trained model is always better than a meta-learning model. Our extensive experiments consider 21 few-shot learning benchmarks, including the large-scale few-shot learning dataset Meta-Data set. We also show no significant difference between a MAML model vs. a PT model with GPT-2 on Openwebtext. We, therefore, conclude that a pre-trained model does not always beat a meta-learned model and that the formal diversity of a dataset is a driving factor.
>
---
#### [replaced 004] Language Models Can Predict Their Own Behavior
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13329v2](http://arxiv.org/pdf/2502.13329v2)**

> **作者:** Dhananjay Ashok; Jonathan May
>
> **备注:** Presented at the Thirty-Ninth Annual Conference on Neural Information Processing Systems (2025)
>
> **摘要:** The text produced by language models (LMs) can exhibit specific `behaviors,' such as a failure to follow alignment training, that we hope to detect and react to during deployment. Identifying these behaviors can often only be done post facto, i.e., after the entire text of the output has been generated. We provide evidence that there are times when we can predict how an LM will behave early in computation, before even a single token is generated. We show that probes trained on the internal representation of input tokens alone can predict a wide range of eventual behaviors over the entire output sequence. Using methods from conformal prediction, we provide provable bounds on the estimation error of our probes, creating precise early warning systems for these behaviors. The conformal probes can identify instances that will trigger alignment failures (jailbreaking) and instruction-following failures, without requiring a single token to be generated. An early warning system built on the probes reduces jailbreaking by 91%. Our probes also show promise in pre-emptively estimating how confident the model will be in its response, a behavior that cannot be detected using the output text alone. Conformal probes can preemptively estimate the final prediction of an LM that uses Chain-of-Thought (CoT) prompting, hence accelerating inference. When applied to an LM that uses CoT to perform text classification, the probes drastically reduce inference costs (65% on average across 27 datasets), with negligible accuracy loss. Encouragingly, probes generalize to unseen datasets and perform better on larger models, suggesting applicability to the largest of models in real-world settings.
>
---
#### [replaced 005] Seeing is Not Understanding: A Benchmark on Perception-Cognition Disparities in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.11101v3](http://arxiv.org/pdf/2509.11101v3)**

> **作者:** Haokun Li; Yazhou Zhang; Jizhi Ding; Qiuchi Li; Peng Zhang
>
> **备注:** I need to modify the content of the article
>
> **摘要:** With the rapid advancement of Multimodal Large Language Models (MLLMs), they have demonstrated exceptional capabilities across a variety of vision-language tasks. However, current evaluation benchmarks predominantly focus on objective visual question answering or captioning, inadequately assessing the models' ability to understand complex and subjective human emotions. To bridge this gap, we introduce EmoBench-Reddit, a novel, hierarchical benchmark for multimodal emotion understanding. The dataset comprises 350 meticulously curated samples from the social media platform Reddit, each containing an image, associated user-provided text, and an emotion category (sad, humor, sarcasm, happy) confirmed by user flairs. We designed a hierarchical task framework that progresses from basic perception to advanced cognition, with each data point featuring six multiple-choice questions and one open-ended question of increasing difficulty. Perception tasks evaluate the model's ability to identify basic visual elements (e.g., colors, objects), while cognition tasks require scene reasoning, intent understanding, and deep empathy integrating textual context. We ensured annotation quality through a combination of AI assistance (Claude 4) and manual verification.We conducted a comprehensive evaluation of nine leading MLLMs, including GPT-5, Gemini-2.5-pro, and GPT-4o, on EmoBench-Reddit.
>
---
#### [replaced 006] Can GRPO Boost Complex Multimodal Table Understanding?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.16889v2](http://arxiv.org/pdf/2509.16889v2)**

> **作者:** Xiaoqiang Kang; Shengen Wu; Zimu Wang; Yilin Liu; Xiaobo Jin; Kaizhu Huang; Wei Wang; Yutao Yue; Xiaowei Huang; Qiufeng Wang
>
> **备注:** EMNLP 2025
>
> **摘要:** Existing table understanding methods face challenges due to complex table structures and intricate logical reasoning. While supervised finetuning (SFT) dominates existing research, reinforcement learning (RL), such as Group Relative Policy Optimization (GRPO), has shown promise but struggled with low initial policy accuracy and coarse rewards in tabular contexts. In this paper, we introduce Table-R1, a three-stage RL framework that enhances multimodal table understanding through: (1) Warm-up that prompts initial perception and reasoning capabilities, (2) Perception Alignment GRPO (PA-GRPO), which employs continuous Tree-Edit-Distance Similarity (TEDS) rewards for recognizing table structures and contents, and (3) Hint-Completion GRPO (HC-GRPO), which utilizes fine-grained rewards of residual steps based on the hint-guided question. Extensive experiments demonstrate that Table-R1 can boost the model's table reasoning performance obviously on both held-in and held-out datasets, outperforming SFT and GRPO largely. Notably, Qwen2-VL-7B with Table-R1 surpasses larger specific table understanding models (e.g., Table-LLaVA 13B), even achieving comparable performance to the closed-source model GPT-4o on held-in datasets, demonstrating the efficacy of each stage of Table-R1 in overcoming initialization bottlenecks and reward sparsity, thereby advancing robust multimodal table understanding.
>
---
#### [replaced 007] Beyond Input Activations: Identifying Influential Latents by Gradient Sparse Autoencoders
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.08080v2](http://arxiv.org/pdf/2505.08080v2)**

> **作者:** Dong Shu; Xuansheng Wu; Haiyan Zhao; Mengnan Du; Ninghao Liu
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Sparse Autoencoders (SAEs) have recently emerged as powerful tools for interpreting and steering the internal representations of large language models (LLMs). However, conventional approaches to analyzing SAEs typically rely solely on input-side activations, without considering the causal influence between each latent feature and the model's output. This work is built on two key hypotheses: (1) activated latents do not contribute equally to the construction of the model's output, and (2) only latents with high causal influence are effective for model steering. To validate these hypotheses, we propose Gradient Sparse Autoencoder (GradSAE), a simple yet effective method that identifies the most influential latents by incorporating output-side gradient information.
>
---
#### [replaced 008] LiTEx: A Linguistic Taxonomy of Explanations for Understanding Within-Label Variation in Natural Language Inference
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22848v3](http://arxiv.org/pdf/2505.22848v3)**

> **作者:** Pingjun Hong; Beiduo Chen; Siyao Peng; Marie-Catherine de Marneffe; Barbara Plank
>
> **备注:** Accepted by EMNLP 2025 Main, 22 pages, 7 figures
>
> **摘要:** There is increasing evidence of Human Label Variation (HLV) in Natural Language Inference (NLI), where annotators assign different labels to the same premise-hypothesis pair. However, within-label variation--cases where annotators agree on the same label but provide divergent reasoning--poses an additional and mostly overlooked challenge. Several NLI datasets contain highlighted words in the NLI item as explanations, but the same spans on the NLI item can be highlighted for different reasons, as evidenced by free-text explanations, which offer a window into annotators' reasoning. To systematically understand this problem and gain insight into the rationales behind NLI labels, we introduce LITEX, a linguistically-informed taxonomy for categorizing free-text explanations. Using this taxonomy, we annotate a subset of the e-SNLI dataset, validate the taxonomy's reliability, and analyze how it aligns with NLI labels, highlights, and explanations. We further assess the taxonomy's usefulness in explanation generation, demonstrating that conditioning generation on LITEX yields explanations that are linguistically closer to human explanations than those generated using only labels or highlights. Our approach thus not only captures within-label variation but also shows how taxonomy-guided generation for reasoning can bridge the gap between human and model explanations more effectively than existing strategies.
>
---
#### [replaced 009] PDTrim: Targeted Pruning for Prefill-Decode Disaggregation in Inference
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.04467v3](http://arxiv.org/pdf/2509.04467v3)**

> **作者:** Hao Zhang; Mengsi Lyu; Zhuo Chen; Xingrun Xing; Yulong Ao; Yonghua Lin
>
> **备注:** 22 pages
>
> **摘要:** Large Language Models (LLMs) demonstrate exceptional capabilities across various tasks, but their deployment is constrained by high computational and memory costs. Model pruning provides an effective means to alleviate these demands. However, existing methods often ignore the characteristics of prefill-decode (PD) disaggregation in practice. In this paper, we propose a novel pruning method for PD disaggregation inference, enabling more precise and efficient block and KV Cache pruning. Our approach constructs pruning and distillation sets to perform iterative block removal independently for the prefill and decode stages, obtaining better pruning solutions. Moreover, we introduce a token-aware cache pruning mechanism that retains all KV Cache in the prefill stage but selectively reuses entries for the first and last token sequences in selected layers during decode, reducing communication costs with minimal overhead. Extensive experiments demonstrate that our approach consistently achieves strong performance in both PD disaggregation and PD unified settings without disaggregation. Under the same (default) settings, our method achieves improved performance and faster inference, along with a 4.95$\times$ reduction in data transmission bandwidth consumption.
>
---
#### [replaced 010] Breaking Token Into Concepts: Exploring Extreme Compression in Token Representation Via Compositional Shared Semantics
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.17737v2](http://arxiv.org/pdf/2509.17737v2)**

> **作者:** Kavin R V; Pawan Goyal
>
> **备注:** 5 pages, 1 figure Accepted at EMNLP 2025 Findings (Short)
>
> **摘要:** Standard language models employ unique, monolithic embeddings for each token, potentially limiting their ability to capture the multifaceted nature of word meanings. We investigate whether tokens can be more effectively represented through a compositional structure that accumulates diverse semantic facets. To explore this, we propose Aggregate Semantic Grouping (ASG), a novel approach leveraging Product Quantization (PQ). We apply ASG to standard transformer architectures (mBERT, XLM-R, mT5) and evaluate this representational scheme across diverse tasks (NLI, NER, QA), as well as a biomedical domain-specific benchmark (BC5CDR) using BioBERT. Our findings demonstrate that representing tokens compositionally via ASG achieves extreme compression in embedding parameters (0.4--0.5\%) while maintaining $>$95\% task performance relative to the base model, even in generative tasks and extends to both cross lingual transfer and domain-specific settings. These results validate the principle that tokens can be effectively modeled as combinations of shared semantic building blocks. ASG offers a simple yet concrete method for achieving this, showcasing how compositional representations can capture linguistic richness while enabling compact yet semantically rich models.
>
---
#### [replaced 011] A suite of allotaxonometric tools for the comparison of complex systems using rank-turbulence divergence
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21808v2](http://arxiv.org/pdf/2506.21808v2)**

> **作者:** Jonathan St-Onge; Ashley M. A. Fehr; Carter Ward; Calla G. Beauregard; Michael V. Arnold; Samuel F. Rosenblatt; Benjamin Cooley; Christopher M. Danforth; Peter Sheridan Dodds
>
> **备注:** 4 pages, 2 figures
>
> **摘要:** Describing and comparing complex systems requires principled, theoretically grounded tools. Built around the phenomenon of type turbulence, allotaxonographs provide map-and-list visual comparisons of pairs of heavy-tailed distributions. Allotaxonographs are designed to accommodate a wide range of instruments including rank- and probability-turbulence divergences, Jenson-Shannon divergence, and generalized entropy divergences. Here, we describe a suite of programmatic tools for rendering allotaxonographs for rank-turbulence divergence in Matlab, Javascript, and Python, all of which have different use cases.
>
---
#### [replaced 012] DivLogicEval: A Framework for Benchmarking Logical Reasoning Evaluation in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.15587v2](http://arxiv.org/pdf/2509.15587v2)**

> **作者:** Tsz Ting Chung; Lemao Liu; Mo Yu; Dit-Yan Yeung
>
> **备注:** Accepted by EMNLP 2025. Project Page: https://ttchungc.github.io/projects/divlogiceval/
>
> **摘要:** Logic reasoning in natural language has been recognized as an important measure of human intelligence for Large Language Models (LLMs). Popular benchmarks may entangle multiple reasoning skills and thus provide unfaithful evaluations on the logic reasoning skill. Meanwhile, existing logic reasoning benchmarks are limited in language diversity and their distributions are deviated from the distribution of an ideal logic reasoning benchmark, which may lead to biased evaluation results. This paper thereby proposes a new classical logic benchmark DivLogicEval, consisting of natural sentences composed of diverse statements in a counterintuitive way. To ensure a more reliable evaluation, we also introduce a new evaluation metric that mitigates the influence of bias and randomness inherent in LLMs. Through experiments, we demonstrate the extent to which logical reasoning is required to answer the questions in DivLogicEval and compare the performance of different popular LLMs in conducting logical reasoning.
>
---
#### [replaced 013] Unraveling Misinformation Propagation in LLM Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18555v2](http://arxiv.org/pdf/2505.18555v2)**

> **作者:** Yiyang Feng; Yichen Wang; Shaobo Cui; Boi Faltings; Mina Lee; Jiawei Zhou
>
> **备注:** Accepted to EMNLP 2025 (Findings)
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities in reasoning, positioning them as promising tools for supporting human problem-solving. However, what happens when their performance is affected by misinformation, i.e., incorrect inputs introduced by users due to oversights or gaps in knowledge? Such misinformation is prevalent in real-world interactions with LLMs, yet how it propagates within LLMs' reasoning process remains underexplored. Focusing on mathematical reasoning, we present a comprehensive analysis of how misinformation affects intermediate reasoning steps and final answers. We also examine how effectively LLMs can correct misinformation when explicitly instructed to do so. Even with explicit instructions, LLMs succeed less than half the time in rectifying misinformation, despite possessing correct internal knowledge, leading to significant accuracy drops (10.02% - 72.20%), and the degradation holds with thinking models (4.30% - 19.97%). Further analysis shows that applying factual corrections early in the reasoning process most effectively reduces misinformation propagation, and fine-tuning on synthesized data with early-stage corrections significantly improves reasoning factuality. Our work offers a practical approach to mitigating misinformation propagation.
>
---
#### [replaced 014] MiCRo: Mixture Modeling and Context-aware Routing for Personalized Preference Learning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24846v2](http://arxiv.org/pdf/2505.24846v2)**

> **作者:** Jingyan Shen; Jiarui Yao; Rui Yang; Yifan Sun; Feng Luo; Rui Pan; Tong Zhang; Han Zhao
>
> **摘要:** Reward modeling is a key step in building safe foundation models when applying reinforcement learning from human feedback (RLHF) to align Large Language Models (LLMs). However, reward modeling based on the Bradley-Terry (BT) model assumes a global reward function, failing to capture the inherently diverse and heterogeneous human preferences. Hence, such oversimplification limits LLMs from supporting personalization and pluralistic alignment. Theoretically, we show that when human preferences follow a mixture distribution of diverse subgroups, a single BT model has an irreducible error. While existing solutions, such as multi-objective learning with fine-grained annotations, help address this issue, they are costly and constrained by predefined attributes, failing to fully capture the richness of human values. In this work, we introduce MiCRo, a two-stage framework that enhances personalized preference learning by leveraging large-scale binary preference datasets without requiring explicit fine-grained annotations. In the first stage, MiCRo introduces context-aware mixture modeling approach to capture diverse human preferences. In the second stage, MiCRo integrates an online routing strategy that dynamically adapts mixture weights based on specific context to resolve ambiguity, allowing for efficient and scalable preference adaptation with minimal additional supervision. Experiments on multiple preference datasets demonstrate that MiCRo effectively captures diverse human preferences and significantly improves downstream personalization.
>
---
#### [replaced 015] Disambiguation in Conversational Question Answering in the Era of LLMs and Agents: A Survey
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12543v2](http://arxiv.org/pdf/2505.12543v2)**

> **作者:** Md Mehrab Tanjim; Yeonjun In; Xiang Chen; Victor S. Bursztyn; Ryan A. Rossi; Sungchul Kim; Guang-Jie Ren; Vaishnavi Muppala; Shun Jiang; Yongsung Kim; Chanyoung Park
>
> **备注:** 14 pages, 2 figures, Accepted at EMNLP 2025 Main Conference
>
> **摘要:** Ambiguity remains a fundamental challenge in Natural Language Processing (NLP) due to the inherent complexity and flexibility of human language. With the advent of Large Language Models (LLMs), addressing ambiguity has become even more critical due to their expanded capabilities and applications. In the context of Conversational Question Answering (CQA), this paper explores the definition, forms, and implications of ambiguity for language driven systems, particularly in the context of LLMs. We define key terms and concepts, categorize various disambiguation approaches enabled by LLMs, and provide a comparative analysis of their advantages and disadvantages. We also explore publicly available datasets for benchmarking ambiguity detection and resolution techniques and highlight their relevance for ongoing research. Finally, we identify open problems and future research directions, especially in agentic settings, proposing areas for further investigation. By offering a comprehensive review of current research on ambiguities and disambiguation with LLMs, we aim to contribute to the development of more robust and reliable LLM-based systems.
>
---
#### [replaced 016] QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning on Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.17428v2](http://arxiv.org/pdf/2509.17428v2)**

> **作者:** Hyesung Jeon; Seojune Lee; Beomseok Kang; Yulhwa Kim; Jae-Joon Kim
>
> **备注:** 25 pages, 9 figures, 14 tables
>
> **摘要:** The demand for efficient deployment of large language models (LLMs) has driven interest in quantization, which reduces inference cost, and parameter-efficient fine-tuning (PEFT), which lowers training overhead. This motivated the development of quantization-aware PEFT to produce accurate yet efficient quantized models. In this setting, reducing quantization error prior to fine-tuning is crucial for achieving high model accuracy. However, existing methods that rely on low-rank adaptation suffer from limited representational capacity. Recent Fourier-related transform (FT)-based adapters offer greater representational power than low-rank adapters, but their direct integration into quantized models often results in ineffective error reduction and increased computational overhead. To overcome these limitations, we propose QWHA, a method that integrates FT-based adapters into quantized models by employing the Walsh-Hadamard Transform (WHT) as the transform kernel, together with a novel adapter initialization scheme incorporating adaptive parameter selection and value refinement. We demonstrate that QWHA effectively mitigates quantization errors while facilitating fine-tuning, and that its design substantially reduces computational cost. Experimental results show that QWHA consistently outperforms baselines in low-bit quantization accuracy and achieves significant training speedups over existing FT-based adapters. The code is available at https://github.com/vantaa89/qwha.
>
---
#### [replaced 017] PruneCD: Contrasting Pruned Self Model to Improve Decoding Factuality
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.16598v2](http://arxiv.org/pdf/2509.16598v2)**

> **作者:** Byeongho Yu; Changhun Lee; Jungyu Jin; Eunhyeok Park
>
> **备注:** accepted at EMNLP 2025 Main Conference
>
> **摘要:** To mitigate the hallucination problem in large language models, DoLa exploits early exit logits from the same model as a contrastive prior. However, we found that these early exit logits tend to be flat, low in magnitude, and fail to reflect meaningful contrasts. To address this, we propose PruneCD, a novel contrastive decoding method that constructs the amateur model via layer pruning rather than early exit. This design leads to more informative and well-aligned logits, enabling more effective contrastive decoding. Through qualitative and quantitative analyses, we demonstrate that PruneCD consistently improves factuality with minimal inference overhead, offering a robust and practical approach to mitigating hallucinations in LLMs.
>
---
#### [replaced 018] Post-hoc Study of Climate Microtargeting on Social Media Ads with LLMs: Thematic Insights and Fairness Evaluation
- **分类: cs.CL; cs.AI; cs.CY; cs.SI**

- **链接: [http://arxiv.org/pdf/2410.05401v4](http://arxiv.org/pdf/2410.05401v4)**

> **作者:** Tunazzina Islam; Dan Goldwasser
>
> **备注:** Accepted at Findings of 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** Climate change communication on social media increasingly employs microtargeting strategies to effectively reach and influence specific demographic groups. This study presents a post-hoc analysis of microtargeting practices within climate campaigns by leveraging large language models (LLMs) to examine Meta (previously known as Facebook) advertisements. Our analysis focuses on two key aspects: demographic targeting and fairness. We evaluate the ability of LLMs to accurately predict the intended demographic targets, such as gender and age group. Furthermore, we instruct the LLMs to generate explanations for their classifications, providing transparent reasoning behind each decision. These explanations reveal the specific thematic elements used to engage different demographic segments, highlighting distinct strategies tailored to various audiences. Our findings show that young adults are primarily targeted through messages emphasizing activism and environmental consciousness, while women are engaged through themes related to caregiving roles and social advocacy. Additionally, we conduct a comprehensive fairness analysis to uncover biases in model predictions. We assess disparities in accuracy and error rates across demographic groups using established fairness metrics such as Demographic Parity, Equal Opportunity, and Predictive Equality. Our findings indicate that while LLMs perform well overall, certain biases exist, particularly in the classification of male audiences. The analysis of thematic explanations uncovers recurring patterns in messaging strategies tailored to various demographic groups, while the fairness analysis underscores the need for more inclusive targeting methods. This study provides a valuable framework for future research aimed at enhancing transparency, accountability, and inclusivity in social media-driven climate campaigns.
>
---
#### [replaced 019] Small LLMs with Expert Blocks Are Good Enough for Hyperparamter Tuning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.15561v2](http://arxiv.org/pdf/2509.15561v2)**

> **作者:** Om Naphade; Saksham Bansal; Parikshit Pareek
>
> **摘要:** Hyper-parameter Tuning (HPT) is a necessary step in machine learning (ML) pipelines but becomes computationally expensive and opaque with larger models. Recently, Large Language Models (LLMs) have been explored for HPT, yet most rely on models exceeding 100 billion parameters. We propose an Expert Block Framework for HPT using Small LLMs. At its core is the Trajectory Context Summarizer (TCS), a deterministic block that transforms raw training trajectories into structured context, enabling small LLMs to analyze optimization progress with reliability comparable to larger models. Using two locally-run LLMs (phi4:reasoning14B and qwen2.5-coder:32B) and a 10-trial budget, our TCS-enabled HPT pipeline achieves average performance within ~0.9 percentage points of GPT-4 across six diverse tasks.
>
---
#### [replaced 020] Triangulating LLM Progress through Benchmarks, Games, and Cognitive Tests
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14359v3](http://arxiv.org/pdf/2502.14359v3)**

> **作者:** Filippo Momentè; Alessandro Suglia; Mario Giulianelli; Ambra Ferrari; Alexander Koller; Oliver Lemon; David Schlangen; Raquel Fernández; Raffaella Bernardi
>
> **摘要:** We examine three evaluation paradigms: standard benchmarks (e.g., MMLU and BBH), interactive games (e.g., Signalling Games or Taboo), and cognitive tests (e.g., for working memory or theory of mind). First, we investigate which of the former two-benchmarks or games-is most effective at discriminating LLMs of varying quality. Then, inspired by human cognitive assessments, we compile a suite of targeted tests that measure cognitive abilities deemed essential for effective language use, and we investigate their correlation with model performance in benchmarks and games. Our analyses reveal that interactive games are superior to standard benchmarks in discriminating models. Causal and logical reasoning correlate with both static and interactive tests, while differences emerge regarding core executive functions and social/emotional skills, which correlate more with games. We advocate for the development of new interactive benchmarks and targeted cognitive tasks inspired by assessing human abilities but designed specifically for LLMs.
>
---
#### [replaced 021] K-DeCore: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning via Knowledge Decoupling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.16929v2](http://arxiv.org/pdf/2509.16929v2)**

> **作者:** Yongrui Chen; Yi Huang; Yunchang Liu; Shenyu Zhang; Junhao He; Tongtong Wu; Guilin Qi; Tianxing Wu
>
> **备注:** Accepted in Neurips 2025 (poster)
>
> **摘要:** Continual Structured Knowledge Reasoning (CSKR) focuses on training models to handle sequential tasks, where each task involves translating natural language questions into structured queries grounded in structured knowledge. Existing general continual learning approaches face significant challenges when applied to this task, including poor generalization to heterogeneous structured knowledge and inefficient reasoning due to parameter growth as tasks increase. To address these limitations, we propose a novel CSKR framework, \textsc{K-DeCore}, which operates with a fixed number of tunable parameters. Unlike prior methods, \textsc{K-DeCore} introduces a knowledge decoupling mechanism that disentangles the reasoning process into task-specific and task-agnostic stages, effectively bridging the gaps across diverse tasks. Building on this foundation, \textsc{K-DeCore} integrates a dual-perspective memory consolidation mechanism for distinct stages and introduces a structure-guided pseudo-data synthesis strategy to further enhance the model's generalization capabilities. Extensive experiments on four benchmark datasets demonstrate the superiority of \textsc{K-DeCore} over existing continual learning methods across multiple metrics, leveraging various backbone large language models.
>
---
#### [replaced 022] Memorization or Reasoning? Exploring the Idiom Understanding of LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16216v2](http://arxiv.org/pdf/2505.16216v2)**

> **作者:** Jisu Kim; Youngwoo Shin; Uiji Hwang; Jihun Choi; Richeng Xuan; Taeuk Kim
>
> **备注:** EMNLP 2025
>
> **摘要:** Idioms have long posed a challenge due to their unique linguistic properties, which set them apart from other common expressions. While recent studies have leveraged large language models (LLMs) to handle idioms across various tasks, e.g., idiom-containing sentence generation and idiomatic machine translation, little is known about the underlying mechanisms of idiom processing in LLMs, particularly in multilingual settings. To this end, we introduce MIDAS, a new large-scale dataset of idioms in six languages, each paired with its corresponding meaning. Leveraging this resource, we conduct a comprehensive evaluation of LLMs' idiom processing ability, identifying key factors that influence their performance. Our findings suggest that LLMs rely not only on memorization, but also adopt a hybrid approach that integrates contextual cues and reasoning, especially when processing compositional idioms. This implies that idiom understanding in LLMs emerges from an interplay between internal knowledge retrieval and reasoning-based inference.
>
---
#### [replaced 023] Exploring Model Kinship for Merging Large Language Models
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2410.12613v3](http://arxiv.org/pdf/2410.12613v3)**

> **作者:** Yedi Hu; Yunzhi Yao; Ningyu Zhang; Huajun Chen; Shumin Deng
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Model merging has emerged as a key technique for enhancing the capabilities and efficiency of Large Language Models (LLMs). The open-source community has driven model evolution by iteratively merging existing models, yet a principled understanding of the gains and underlying factors in model merging remains limited. In this work, we study model evolution through iterative merging, drawing an analogy to biological evolution, and introduce the concept of model kinship, the degree of similarity or relatedness between LLMs. Through comprehensive empirical analysis, we show that model kinship is closely linked to the performance improvements achieved by merging, providing a useful criterion for selecting candidate models. Building on this insight, we propose a new model merging strategy: Top-k Greedy Merging with Model Kinship, which can improve benchmark performance. Specifically, we discover that incorporating model kinship as a guiding criterion enables continuous merging while mitigating performance degradation caused by local optima, thereby facilitating more effective model evolution. Code is available at https://github.com/zjunlp/ModelKinship.
>
---
#### [replaced 024] Clip Your Sequences Fairly: Enforcing Length Fairness for Sequence-Level RL
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.09177v2](http://arxiv.org/pdf/2509.09177v2)**

> **作者:** Hanyi Mao; Quanjia Xiao; Lei Pang; Haixiao Liu
>
> **摘要:** We propose FSPO (Fair Sequence Policy Optimization), a sequence-level reinforcement learning method for LLMs that enforces length-fair clipping on the importance-sampling (IS) weight. We study RL methods with sequence-level IS and identify a mismatch when PPO/GRPO-style clipping is transplanted to sequences: a fixed clip range systematically reweights short vs.\ long responses, distorting the optimization direction. FSPO introduces a simple remedy: we clip the sequence log-IS ratio with a band that scales as $\sqrt{L}$. Theoretically, we formalize length fairness via a Length Reweighting Error (LRE) and prove that small LRE yields a cosine directional guarantee between the clipped and true updates. Empirically, FSPO flattens clip rates across length bins, stabilizes training, and outperforms all baselines across multiple evaluation datasets on Qwen3-8B-Base model.
>
---
#### [replaced 025] Can LLMs Explain Themselves Counterfactually?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.18156v2](http://arxiv.org/pdf/2502.18156v2)**

> **作者:** Zahra Dehghanighobadi; Asja Fischer; Muhammad Bilal Zafar
>
> **摘要:** Explanations are an important tool for gaining insights into the behavior of ML models, calibrating user trust and ensuring regulatory compliance. Past few years have seen a flurry of post-hoc methods for generating model explanations, many of which involve computing model gradients or solving specially designed optimization problems. However, owing to the remarkable reasoning abilities of Large Language Model (LLMs), self-explanation, that is, prompting the model to explain its outputs has recently emerged as a new paradigm. In this work, we study a specific type of self-explanations, self-generated counterfactual explanations (SCEs). We design tests for measuring the efficacy of LLMs in generating SCEs. Analysis over various LLM families, model sizes, temperature settings, and datasets reveals that LLMs sometimes struggle to generate SCEs. Even when they do, their prediction often does not agree with their own counterfactual reasoning.
>
---
#### [replaced 026] Reward-Shifted Speculative Sampling Is An Efficient Test-Time Weak-to-Strong Aligner
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.15044v3](http://arxiv.org/pdf/2508.15044v3)**

> **作者:** Bolian Li; Yanran Wu; Xinyu Luo; Ruqi Zhang
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** Aligning large language models (LLMs) with human preferences has become a critical step in their development. Recent research has increasingly focused on test-time alignment, where additional compute is allocated during inference to enhance LLM safety and reasoning capabilities. However, these test-time alignment techniques often incur substantial inference costs, limiting their practical application. We are inspired by the speculative sampling acceleration, which leverages a small draft model to efficiently predict future tokens, to address the efficiency bottleneck of test-time alignment. We introduce the reward-shifted speculative sampling (SSS) algorithm, in which the draft model is aligned with human preferences, while the target model remains unchanged. We theoretically demonstrate that the distributional shift between the aligned draft model and the unaligned target model can be exploited to recover the RLHF optimal solution without actually obtaining it, by modifying the acceptance criterion and bonus token distribution. Our algorithm achieves superior gold reward scores at a significantly reduced inference cost in test-time weak-to-strong alignment experiments, thereby validating both its effectiveness and efficiency.
>
---
#### [replaced 027] Gender and Political Bias in Large Language Models: A Demonstration Platform
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.16264v2](http://arxiv.org/pdf/2509.16264v2)**

> **作者:** Wenjie Lin; Hange Liu; Xutao Mao; Yingying Zhuang; Jingwei Shi; Xudong Han; Tianyu Shi; Jinrui Yang
>
> **备注:** online demo: https://euro-parl-vote-demo.vercel.app/; Video: https://www.youtube.com/@Jinrui-sf2jg
>
> **摘要:** We present ParlAI Vote, an interactive system for exploring European Parliament debates and votes, and for testing LLMs on vote prediction and bias analysis. This platform connects debate topics, speeches, and roll-call outcomes, and includes rich demographic data such as gender, age, country, and political group. Users can browse debates, inspect linked speeches, compare real voting outcomes with predictions from frontier LLMs, and view error breakdowns by demographic group. Visualizing the EuroParlVote benchmark and its core tasks of gender classification and vote prediction, ParlAI Vote highlights systematic performance bias in state-of-the-art LLMs. The system unifies data, models, and visual analytics in a single interface, lowering the barrier for reproducing findings, auditing behavior, and running counterfactual scenarios. It supports research, education, and public engagement with legislative decision-making, while making clear both the strengths and the limitations of current LLMs in political analysis.
>
---
#### [replaced 028] Anything Goes? A Crosslinguistic Study of (Im)possible Language Learning in LMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18795v3](http://arxiv.org/pdf/2502.18795v3)**

> **作者:** Xiulin Yang; Tatsuya Aoyama; Yuekun Yao; Ethan Wilcox
>
> **备注:** ACL 2025
>
> **摘要:** Do language models (LMs) offer insights into human language learning? A common argument against this idea is that because their architecture and training paradigm are so vastly different from humans, LMs can learn arbitrary inputs as easily as natural languages. We test this claim by training LMs to model impossible and typologically unattested languages. Unlike previous work, which has focused exclusively on English, we conduct experiments on 12 languages from 4 language families with two newly constructed parallel corpora. Our results show that while GPT-2 small can largely distinguish attested languages from their impossible counterparts, it does not achieve perfect separation between all the attested languages and all the impossible ones. We further test whether GPT-2 small distinguishes typologically attested from unattested languages with different NP orders by manipulating word order based on Greenberg's Universal 20. We find that the model's perplexity scores do not distinguish attested vs. unattested word orders, while its performance on the generalization test does. These findings suggest that LMs exhibit some human-like inductive biases, though these biases are weaker than those found in human learners.
>
---
#### [replaced 029] CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.21074v3](http://arxiv.org/pdf/2502.21074v3)**

> **作者:** Zhenyi Shen; Hanqi Yan; Linhai Zhang; Zhanghao Hu; Yali Du; Yulan He
>
> **备注:** 17 pages. Code available at https://github.com/zhenyi4/codi
>
> **摘要:** Chain-of-Thought (CoT) reasoning enhances Large Language Models (LLMs) by encouraging step-by-step reasoning in natural language. However, leveraging a latent continuous space for reasoning may offer benefits in terms of both efficiency and robustness. Prior implicit CoT methods attempt to bypass language completely by reasoning in continuous space but have consistently underperformed compared to the standard explicit CoT approach. We introduce CODI (Continuous Chain-of-Thought via Self-Distillation), a novel training framework that effectively compresses natural language CoT into continuous space. CODI jointly trains a teacher task (Explicit CoT) and a student task (Implicit CoT), distilling the reasoning ability from language into continuous space by aligning the hidden states of a designated token. Our experiments show that CODI is the first implicit CoT approach to match the performance of explicit CoT on GSM8k at the GPT-2 scale, achieving a 3.1x compression rate and outperforming the previous state-of-the-art by 28.2% in accuracy. CODI also demonstrates robustness, generalizable to complex datasets, and interpretability. These results validate that LLMs can reason effectively not only in natural language, but also in a latent continuous space. Code is available at https://github.com/zhenyi4/codi.
>
---
#### [replaced 030] Linguistic Neuron Overlap Patterns to Facilitate Cross-lingual Transfer on Low-resource Languages
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.17078v2](http://arxiv.org/pdf/2508.17078v2)**

> **作者:** Yuemei Xu; Kexin Xu; Jian Zhou; Ling Hu; Lin Gui
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** The current Large Language Models (LLMs) face significant challenges in improving their performance on low-resource languages and urgently need data-efficient methods without costly fine-tuning. From the perspective of language-bridge, we propose a simple yet effective method, namely BridgeX-ICL, to improve the zero-shot Cross-lingual In-Context Learning (X-ICL) for low-resource languages. Unlike existing works focusing on language-specific neurons, BridgeX-ICL explores whether sharing neurons can improve cross-lingual performance in LLMs. We construct neuron probe data from the ground-truth MUSE bilingual dictionaries, and define a subset of language overlap neurons accordingly to ensure full activation of these anchored neurons. Subsequently, we propose an HSIC-based metric to quantify LLMs' internal linguistic spectrum based on overlapping neurons, guiding optimal bridge selection. The experiments conducted on 4 cross-lingual tasks and 15 language pairs from 7 diverse families, covering both high-low and moderate-low pairs, validate the effectiveness of BridgeX-ICL and offer empirical insights into the underlying multilingual mechanisms of LLMs. The code is publicly available at https://github.com/xuyuemei/BridgeX-ICL.
>
---
#### [replaced 031] Training Language Model Agents to Find Vulnerabilities with CTF-Dojo
- **分类: cs.SE; cs.CL; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.18370v2](http://arxiv.org/pdf/2508.18370v2)**

> **作者:** Terry Yue Zhuo; Dingmin Wang; Hantian Ding; Varun Kumar; Zijian Wang
>
> **摘要:** Large language models (LLMs) have demonstrated exceptional capabilities when trained within executable runtime environments, notably excelling at software engineering tasks through verified feedback loops. Yet, scalable and generalizable execution-grounded environments remain scarce, limiting progress in training more capable ML agents. We introduce CTF-Dojo, the first large-scale executable runtime tailored for training LLMs with verifiable feedback, featuring 658 fully functional Capture-The-Flag (CTF)-style challenges containerized in Docker with guaranteed reproducibility. To enable rapid scaling without manual intervention, we develop CTF-Forge, an automated pipeline that transforms publicly available artifacts into ready-to-use execution environments in minutes, eliminating weeks of expert configuration traditionally required. We trained LLM-based agents on just 486 high-quality, execution-verified trajectories from CTF-Dojo, achieving up to 11.6% absolute gains over strong baselines across three competitive benchmarks: InterCode-CTF, NYU CTF Bench, and Cybench. Our best-performing 32B model reaches 31.9% Pass@1, establishing a new open-weight state-of-the-art that rivals frontier models like DeepSeek-V3-0324 and Gemini-2.5-Flash. By framing CTF-style tasks as a benchmark for executable-agent learning, CTF-Dojo demonstrates that execution-grounded training signals are not only effective but pivotal in advancing high-performance ML agents without dependence on costly proprietary systems.
>
---
#### [replaced 032] Think in Safety: Unveiling and Mitigating Safety Alignment Collapse in Multimodal Large Reasoning Model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.06538v3](http://arxiv.org/pdf/2505.06538v3)**

> **作者:** Xinyue Lou; You Li; Jinan Xu; Xiangyu Shi; Chi Chen; Kaiyu Huang
>
> **备注:** Accepted by EMNLP2025-main(Oral Presentation, SAC score: 10)
>
> **摘要:** The rapid development of Multimodal Large Reasoning Models (MLRMs) has demonstrated broad application potential, yet their safety and reliability remain critical concerns that require systematic exploration. To address this gap, we conduct a comprehensive and systematic safety evaluation of 11 MLRMs across 5 benchmarks and unveil prevalent safety degradation phenomena in most advanced models. Moreover, our analysis reveals distinct safety patterns across different benchmarks: significant safety degradation is observed across jailbreak robustness benchmarks, whereas safety-awareness benchmarks demonstrate less pronounced degradation. In particular, the long thought process in some scenarios even enhances safety performance. Therefore, it is a potential approach to address safety issues in MLRMs by leveraging the intrinsic reasoning capabilities of the model to detect unsafe intent. To operationalize this insight, we construct a multimodal tuning dataset that incorporates a safety-oriented thought process. Experimental results from fine-tuning existing MLRMs with this dataset effectively enhances the safety on both jailbreak robustness and safety-awareness benchmarks. This study provides a new perspective for developing safe MLRMs. Our dataset is available at https://github.com/xinyuelou/Think-in-Safety.
>
---
#### [replaced 033] Large Language Models Implicitly Learn to See and Hear Just By Reading
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.17091v2](http://arxiv.org/pdf/2505.17091v2)**

> **作者:** Prateek Verma; Mert Pilanci
>
> **备注:** 6 pages, 3 figures, 4 tables. Added BLIP reference
>
> **摘要:** This paper presents a fascinating find: By training an auto-regressive LLM model on text tokens, the text model inherently develops internally an ability to understand images and audio, thereby developing the ability to see and hear just by reading. Popular audio and visual LLM models fine-tune text LLM models to give text output conditioned on images and audio embeddings. On the other hand, our architecture takes in patches of images, audio waveforms or tokens as input. It gives us the embeddings or category labels typical of a classification pipeline. We show the generality of text weights in aiding audio classification for datasets FSD-50K and GTZAN. Further, we show this working for image classification on CIFAR-10 and Fashion-MNIST, as well on image patches. This pushes the notion of text-LLMs learning powerful internal circuits that can be utilized by activating necessary connections for various applications rather than training models from scratch every single time.
>
---
#### [replaced 034] Columbo: Expanding Abbreviated Column Names for Tabular Data Using Large Language Models
- **分类: cs.CL; cs.DB**

- **链接: [http://arxiv.org/pdf/2508.09403v3](http://arxiv.org/pdf/2508.09403v3)**

> **作者:** Ting Cai; Stephen Sheen; AnHai Doan
>
> **备注:** Accepted to Findings of EMNLP 2025; 19 pages, 14 figures
>
> **摘要:** Expanding the abbreviated column names of tables, such as "esal" to "employee salary", is critical for many downstream NLP tasks for tabular data, such as NL2SQL, table QA, and keyword search. This problem arises in enterprises, domain sciences, government agencies, and more. In this paper, we make three contributions that significantly advance the state of the art. First, we show that the synthetic public data used by prior work has major limitations, and we introduce four new datasets in enterprise/science domains, with real-world abbreviations. Second, we show that accuracy measures used by prior work seriously undercount correct expansions, and we propose new synonym-aware measures that capture accuracy much more accurately. Finally, we develop Columbo, a powerful LLM-based solution that exploits context, rules, chain-of-thought reasoning, and token-level analysis. Extensive experiments show that Columbo significantly outperforms NameGuess, the current most advanced solution, by 4-29%, over five datasets. Columbo has been used in production on EDI, a major data lake for environmental sciences.
>
---
#### [replaced 035] Improving Low-Resource Sequence Labeling with Knowledge Fusion and Contextual Label Explanations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.19093v3](http://arxiv.org/pdf/2501.19093v3)**

> **作者:** Peichao Lai; Jiaxin Gan; Feiyang Ye; Yilei Wang; Bin Cui
>
> **摘要:** Sequence labeling remains a significant challenge in low-resource, domain-specific scenarios, particularly for character-dense languages like Chinese. Existing methods primarily focus on enhancing model comprehension and improving data diversity to boost performance. However, these approaches still struggle with inadequate model applicability and semantic distribution biases in domain-specific contexts. To overcome these limitations, we propose a novel framework that combines an LLM-based knowledge enhancement workflow with a span-based Knowledge Fusion for Rich and Efficient Extraction (KnowFREE) model. Our workflow employs explanation prompts to generate precise contextual interpretations of target entities, effectively mitigating semantic biases and enriching the model's contextual understanding. The KnowFREE model further integrates extension label features, enabling efficient nested entity extraction without relying on external knowledge during inference. Experiments on multiple Chinese domain-specific sequence labeling datasets demonstrate that our approach achieves state-of-the-art performance, effectively addressing the challenges posed by low-resource settings.
>
---
#### [replaced 036] DOTA: Distributional Test-Time Adaptation of Vision-Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2409.19375v2](http://arxiv.org/pdf/2409.19375v2)**

> **作者:** Zongbo Han; Jialong Yang; Guangyu Wang; Junfan Li; Qianli Xu; Mike Zheng Shou; Changqing Zhang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Vision-language foundation models (VLMs), such as CLIP, exhibit remarkable performance across a wide range of tasks. However, deploying these models can be unreliable when significant distribution gaps exist between training and test data, while fine-tuning for diverse scenarios is often costly. Cache-based test-time adapters offer an efficient alternative by storing representative test samples to guide subsequent classifications. Yet, these methods typically employ naive cache management with limited capacity, leading to severe catastrophic forgetting when samples are inevitably dropped during updates. In this paper, we propose DOTA (DistributiOnal Test-time Adaptation), a simple yet effective method addressing this limitation. Crucially, instead of merely memorizing individual test samples, DOTA continuously estimates the underlying distribution of the test data stream. Test-time posterior probabilities are then computed using these dynamically estimated distributions via Bayes' theorem for adaptation. This distribution-centric approach enables the model to continually learn and adapt to the deployment environment. Extensive experiments validate that DOTA significantly mitigates forgetting and achieves state-of-the-art performance compared to existing methods.
>
---
#### [replaced 037] Athena: Enhancing Multimodal Reasoning with Data-efficient Process Reward Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09532v2](http://arxiv.org/pdf/2506.09532v2)**

> **作者:** Shuai Wang; Zhenhua Liu; Jiaheng Wei; Xuanwu Yin; Dong Li; Emad Barsoum
>
> **摘要:** We present Athena-PRM, a multimodal process reward model (PRM) designed to evaluate the reward score for each step in solving complex reasoning problems. Developing high-performance PRMs typically demands significant time and financial investment, primarily due to the necessity for step-level annotations of reasoning steps. Conventional automated labeling methods, such as Monte Carlo estimation, often produce noisy labels and incur substantial computational costs. To efficiently generate high-quality process-labeled data, we propose leveraging prediction consistency between weak and strong completers as a criterion for identifying reliable process labels. Remarkably, Athena-PRM demonstrates outstanding effectiveness across various scenarios and benchmarks with just 5,000 samples. Furthermore, we also develop two effective strategies to improve the performance of PRMs: ORM initialization and up-sampling for negative data. We validate our approach in three specific scenarios: verification for test time scaling, direct evaluation of reasoning step correctness, and reward ranked fine-tuning. Our Athena-PRM consistently achieves superior performance across multiple benchmarks and scenarios. Notably, when using Qwen2.5-VL-7B as the policy model, Athena-PRM enhances performance by 10.2 points on WeMath and 7.1 points on MathVista for test time scaling. Furthermore, Athena-PRM sets the state-of-the-art (SoTA) results in VisualProcessBench and outperforms the previous SoTA by 3.9 F1-score, showcasing its robust capability to accurately assess the correctness of the reasoning step. Additionally, utilizing Athena-PRM as the reward model, we develop Athena-7B with reward ranked fine-tuning and outperforms baseline with a significant margin on five benchmarks.
>
---
#### [replaced 038] NileChat: Towards Linguistically Diverse and Culturally Aware LLMs for Local Communities
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18383v3](http://arxiv.org/pdf/2505.18383v3)**

> **作者:** Abdellah El Mekki; Houdaifa Atou; Omer Nacar; Shady Shehata; Muhammad Abdul-Mageed
>
> **备注:** Accepted to EMNLP 2025 (Main Conference). Camera-ready version. Data & models: https://github.com/UBC-NLP/nilechat
>
> **摘要:** Enhancing the linguistic capabilities of Large Language Models (LLMs) to include low-resource languages is a critical research area. Current research directions predominantly rely on synthetic data generated by translating English corpora, which, while demonstrating promising linguistic understanding and translation abilities, often results in models aligned with source language culture. These models frequently fail to represent the cultural heritage and values of local communities. This work proposes a methodology to create both synthetic and retrieval-based pre-training data tailored to a specific community, considering its (i) language, (ii) cultural heritage, and (iii) cultural values. We demonstrate our methodology using Egyptian and Moroccan dialects as testbeds, chosen for their linguistic and cultural richness and current underrepresentation in LLMs. As a proof-of-concept, we develop NileChat, a 3B parameter Egyptian and Moroccan Arabic LLM adapted for Egyptian and Moroccan communities, incorporating their language, cultural heritage, and values. Our results on various understanding, translation, and cultural and values alignment benchmarks show that NileChat outperforms existing Arabic-aware LLMs of similar size and performs on par with larger models. This work addresses Arabic dialect in LLMs with a focus on cultural and values alignment via controlled synthetic data generation and retrieval-augmented pre-training for Moroccan Darija and Egyptian Arabic, including Arabizi variants, advancing Arabic NLP for low-resource communities. We share our methods, data, and models with the community to promote the inclusion and coverage of more diverse communities in cultural LLM development: https://github.com/UBC-NLP/nilechat .
>
---
#### [replaced 039] T-Detect: Tail-Aware Statistical Normalization for Robust Detection of Adversarial Machine-Generated Text
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.23577v2](http://arxiv.org/pdf/2507.23577v2)**

> **作者:** Alva West; Luodan Zhang; Liuliu Zhang; Minjun Zhu; Yixuan Weng; Yue Zhang
>
> **摘要:** Large language models (LLMs) have shown the capability to generate fluent and logical content, presenting significant challenges to machine-generated text detection, particularly text polished by adversarial perturbations such as paraphrasing. Current zero-shot detectors often employ Gaussian distributions as statistical measure for computing detection thresholds, which falters when confronted with the heavy-tailed statistical artifacts characteristic of adversarial or non-native English texts. In this paper, we introduce T-Detect, a novel detection method that fundamentally redesigns the curvature-based detectors. Our primary innovation is the replacement of standard Gaussian normalization with a heavy-tailed discrepancy score derived from the Student's t-distribution. This approach is theoretically grounded in the empirical observation that adversarial texts exhibit significant leptokurtosis, rendering traditional statistical assumptions inadequate. T-Detect computes a detection score by normalizing the log-likelihood of a passage against the expected moments of a t-distribution, providing superior resilience to statistical outliers. We validate our approach on the challenging RAID benchmark for adversarial text and the comprehensive HART dataset. Experiments show that T-Detect provides a consistent performance uplift over strong baselines, improving AUROC by up to 3.9\% in targeted domains. When integrated into a two-dimensional detection framework (CT), our method achieves state-of-the-art performance, with an AUROC of 0.926 on the Books domain of RAID. Our contributions are a new, theoretically-justified statistical foundation for text detection, an ablation-validated method that demonstrates superior robustness, and a comprehensive analysis of its performance under adversarial conditions. Ours code are released at https://github.com/ResearAI/t-detect.
>
---
#### [replaced 040] Union of Experts: Adapting Hierarchical Routing to Equivalently Decomposed Transformer
- **分类: cs.LG; cs.AI; cs.CL; 68T07; I.5.1; I.2.0**

- **链接: [http://arxiv.org/pdf/2503.02495v3](http://arxiv.org/pdf/2503.02495v3)**

> **作者:** Yujiao Yang; Jing Lian; Linhui Li
>
> **摘要:** Mixture-of-Experts (MoE) enhances model performance while maintaining computational efficiency, making it well-suited for large-scale applications. Conventional mixture-of-experts (MoE) architectures suffer from suboptimal coordination dynamics, where isolated expert operations expose the model to overfitting risks. Moreover, they have not been effectively extended to attention blocks, which limits further efficiency improvements. To tackle these issues, we propose Union-of-Experts (UoE), which decomposes the transformer model into an equivalent group of experts and applies a hierarchical routing mechanism to allocate input subspaces to specialized experts. Our approach advances MoE design with four key innovations: (1) Constructing expert groups by partitioning non-MoE models into functionally equivalent specialists (2) Developing a hierarchical routing paradigm that integrates patch-wise data selection and expert selection strategies. (3) Extending the MoE design to attention blocks. (4) Proposing a hardware-optimized parallelization scheme that exploits batched matrix multiplications for efficient expert computation. The experiments demonstrate that our UoE model surpasses Full Attention, state-of-the-art MoEs and efficient transformers in several tasks across image and natural language domains. In language modeling tasks, UoE achieves an average reduction of 2.38 in perplexity compared to the best-performing MoE method with only 76% of its FLOPs. In the Long Range Arena benchmark, it demonstrates an average score at least 0.68% higher than all comparison models, with only 50% of the FLOPs of the best MoE method. In image classification, it yields an average accuracy improvement of 1.75% over the best model while maintaining comparable FLOPs. The source codes are available at https://github.com/YujiaoYang-work/UoE.
>
---
#### [replaced 041] Identifying and Answering Questions with False Assumptions: An Interpretable Approach
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.15139v2](http://arxiv.org/pdf/2508.15139v2)**

> **作者:** Zijie Wang; Eduardo Blanco
>
> **备注:** To appear at EMNLP 2025 Main conference
>
> **摘要:** People often ask questions with false assumptions, a type of question that does not have regular answers. Answering such questions requires first identifying the false assumptions. Large Language Models (LLMs) often generate misleading answers to these questions because of hallucinations. In this paper, we focus on identifying and answering questions with false assumptions in several domains. We first investigate whether the problem reduces to fact verification. Then, we present an approach leveraging external evidence to mitigate hallucinations. Experiments with five LLMs demonstrate that (1) incorporating retrieved evidence is beneficial and (2) generating and validating atomic assumptions yields more improvements and provides an interpretable answer by pinpointing the false assumptions.
>
---
#### [replaced 042] Enhancing Unsupervised Sentence Embeddings via Knowledge-Driven Data Augmentation and Gaussian-Decayed Contrastive Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.12887v4](http://arxiv.org/pdf/2409.12887v4)**

> **作者:** Peichao Lai; Zhengfeng Zhang; Wentao Zhang; Fangcheng Fu; Bin Cui
>
> **摘要:** Recently, using large language models (LLMs) for data augmentation has led to considerable improvements in unsupervised sentence embedding models. However, existing methods encounter two primary challenges: limited data diversity and high data noise. Current approaches often neglect fine-grained knowledge, such as entities and quantities, leading to insufficient diversity. Besides, unsupervised data frequently lacks discriminative information, and the generated synthetic samples may introduce noise. In this paper, we propose a pipeline-based data augmentation method via LLMs and introduce the Gaussian-decayed gradient-assisted Contrastive Sentence Embedding (GCSE) model to enhance unsupervised sentence embeddings. To tackle the issue of low data diversity, our pipeline utilizes knowledge graphs (KGs) to extract entities and quantities, enabling LLMs to generate more diverse samples. To address high data noise, the GCSE model uses a Gaussian-decayed function to limit the impact of false hard negative samples, enhancing the model's discriminative capability. Experimental results show that our approach achieves state-of-the-art performance in semantic textual similarity (STS) tasks, using fewer data samples and smaller LLMs, demonstrating its efficiency and robustness across various models.
>
---
#### [replaced 043] PolBiX: Detecting LLMs' Political Bias in Fact-Checking through X-phemisms
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.15335v2](http://arxiv.org/pdf/2509.15335v2)**

> **作者:** Charlott Jakob; David Harbecke; Patrick Parschan; Pia Wenzel Neves; Vera Schmitt
>
> **备注:** Accepted at Findings of EMNLP 2025, camera-ready version
>
> **摘要:** Large Language Models are increasingly used in applications requiring objective assessment, which could be compromised by political bias. Many studies found preferences for left-leaning positions in LLMs, but downstream effects on tasks like fact-checking remain underexplored. In this study, we systematically investigate political bias through exchanging words with euphemisms or dysphemisms in German claims. We construct minimal pairs of factually equivalent claims that differ in political connotation, to assess the consistency of LLMs in classifying them as true or false. We evaluate six LLMs and find that, more than political leaning, the presence of judgmental words significantly influences truthfulness assessment. While a few models show tendencies of political bias, this is not mitigated by explicitly calling for objectivism in prompts. Warning: This paper contains content that may be offensive or upsetting.
>
---
#### [replaced 044] MediSyn: A Generalist Text-Guided Latent Diffusion Model For Diverse Medical Image Synthesis
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.09806v5](http://arxiv.org/pdf/2405.09806v5)**

> **作者:** Joseph Cho; Mrudang Mathur; Cyril Zakka; Dhamanpreet Kaur; Matthew Leipzig; Alex Dalal; Aravind Krishnan; Eubee Koo; Karen Wai; Cindy S. Zhao; Akshay Chaudhari; Matthew Duda; Ashley Choi; Ehsan Rahimy; Lyna Azzouz; Robyn Fong; Rohan Shad; William Hiesinger
>
> **摘要:** Deep learning algorithms require extensive data to achieve robust performance. However, data availability is often restricted in the medical domain due to patient privacy concerns. Synthetic data presents a possible solution to these challenges. Recently, image generative models have found increasing use for medical applications but are often designed for singular medical specialties and imaging modalities, thus limiting their broader utility. To address this, we introduce MediSyn: a text-guided, latent diffusion model capable of generating synthetic images from 6 medical specialties and 10 image types. Through extensive experimentation, we first demonstrate that MediSyn quantitatively matches or surpasses the performance of specialist models. Second, we show that our synthetic images are realistic and exhibit strong alignment with their corresponding text prompts, as validated by a team of expert physicians. Third, we provide empirical evidence that our synthetic images are visually distinct from their corresponding real patient images. Finally, we demonstrate that in data-limited settings, classifiers trained solely on synthetic data or real data supplemented with synthetic data can outperform those trained solely on real data. Our findings highlight the immense potential of generalist image generative models to accelerate algorithmic research and development in medicine.
>
---
#### [replaced 045] Automated Generation of Research Workflows from Academic Papers: A Full-text Mining Framework
- **分类: cs.CL; cs.DL; cs.IR**

- **链接: [http://arxiv.org/pdf/2509.12955v2](http://arxiv.org/pdf/2509.12955v2)**

> **作者:** Heng Zhang; Chengzhi Zhang
>
> **摘要:** The automated generation of research workflows is essential for improving the reproducibility of research and accelerating the paradigm of "AI for Science". However, existing methods typically extract merely fragmented procedural components and thus fail to capture complete research workflows. To address this gap, we propose an end-to-end framework that generates comprehensive, structured research workflows by mining full-text academic papers. As a case study in the Natural Language Processing (NLP) domain, our paragraph-centric approach first employs Positive-Unlabeled (PU) Learning with SciBERT to identify workflow-descriptive paragraphs, achieving an F1-score of 0.9772. Subsequently, we utilize Flan-T5 with prompt learning to generate workflow phrases from these paragraphs, yielding ROUGE-1, ROUGE-2, and ROUGE-L scores of 0.4543, 0.2877, and 0.4427, respectively. These phrases are then systematically categorized into data preparation, data processing, and data analysis stages using ChatGPT with few-shot learning, achieving a classification precision of 0.958. By mapping categorized phrases to their document locations in the documents, we finally generate readable visual flowcharts of the entire research workflows. This approach facilitates the analysis of workflows derived from an NLP corpus and reveals key methodological shifts over the past two decades, including the increasing emphasis on data analysis and the transition from feature engineering to ablation studies. Our work offers a validated technical framework for automated workflow generation, along with a novel, process-oriented perspective for the empirical investigation of evolving scientific paradigms. Source code and data are available at: https://github.com/ZH-heng/research_workflow.
>
---
#### [replaced 046] Fine-Tuning is Subgraph Search: A New Lens on Learning Dynamics
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.06106v3](http://arxiv.org/pdf/2502.06106v3)**

> **作者:** Yueyan Li; Wenhao Gao; Caixia Yuan; Xiaojie Wang
>
> **摘要:** The study of mechanistic interpretability aims to reverse-engineer a model to explain its behaviors. While recent studies have focused on the static mechanism of a certain behavior, the learning dynamics inside a model remain to be explored. In this work, we develop a fine-tuning method for analyzing the mechanism behind learning. Inspired by the concept of intrinsic dimension, we view a model as a computational graph with redundancy for a specific task, and treat the fine-tuning process as a search for and optimization of a subgraph within this graph. Based on this hypothesis, we propose circuit-tuning, an algorithm that iteratively builds the subgraph for a specific task and updates the relevant parameters in a heuristic way. We first validate our hypothesis through a carefully designed experiment and provide a detailed analysis of the learning dynamics during fine-tuning. Subsequently, we conduct experiments on more complex tasks, demonstrating that circuit-tuning could strike a balance between the performance on the target task and the general capabilities. Our work offers a new analytical method for the dynamics of fine-tuning, provides new findings on the mechanisms behind the training process, and inspires the design of superior algorithms for the training of neural networks.
>
---
#### [replaced 047] Automating Steering for Safe Multimodal Large Language Models
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2507.13255v3](http://arxiv.org/pdf/2507.13255v3)**

> **作者:** Lyucheng Wu; Mengru Wang; Ziwen Xu; Tri Cao; Nay Oo; Bryan Hooi; Shumin Deng
>
> **备注:** EMNLP 2025 Main Conference. 23 pages (8+ for main); 25 figures; 1 table
>
> **摘要:** Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems.
>
---
#### [replaced 048] Improving Image Captioning Descriptiveness by Ranking and LLM-based Fusion
- **分类: cs.CV; cs.AI; cs.CL; cs.DB; cs.LG**

- **链接: [http://arxiv.org/pdf/2306.11593v2](http://arxiv.org/pdf/2306.11593v2)**

> **作者:** Luigi Celona; Simone Bianco; Marco Donzella; Paolo Napoletano
>
> **备注:** This manuscript has been accepted for publication in Springer Neural Computing and Applications
>
> **摘要:** State-of-The-Art (SoTA) image captioning models are often trained on the MicroSoft Common Objects in Context (MS-COCO) dataset, which contains human-annotated captions with an average length of approximately ten tokens. Although effective for general scene understanding, these short captions often fail to capture complex scenes and convey detailed information. Moreover, captioning models tend to exhibit bias towards the ``average'' caption, which captures only the more general aspects, thus overlooking finer details. In this paper, we present a novel approach to generate richer and more informative image captions by combining the captions generated from different SoTA captioning models. Our proposed method requires no additional model training: given an image, it leverages pre-trained models from the literature to generate the initial captions, and then ranks them using a newly introduced image-text-based metric, which we name BLIPScore. Subsequently, the top two captions are fused using a Large Language Model (LLM) to produce the final, more detailed description. Experimental results on the MS-COCO and Flickr30k test sets demonstrate the effectiveness of our approach in terms of caption-image alignment and hallucination reduction according to the ALOHa, CAPTURE, and Polos metrics. A subjective study lends additional support to these results, suggesting that the captions produced by our model are generally perceived as more consistent with human judgment. By combining the strengths of diverse SoTA models, our method enhances the quality and appeal of image captions, bridging the gap between automated systems and the rich and informative nature of human-generated descriptions. This advance enables the generation of more suitable captions for the training of both vision-language and captioning models.
>
---
#### [replaced 049] GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.04183v4](http://arxiv.org/pdf/2409.04183v4)**

> **作者:** Ziyin Zhang; Hang Yu; Shijie Li; Peng Di; Jianguo Li; Rui Wang
>
> **备注:** ACL 2025
>
> **摘要:** Programming languages possess rich semantic information - such as data flow - that is represented by graphs and not available from the surface form of source code. Recent code language models have scaled to billions of parameters, but model source code solely as text tokens while ignoring any other structural information. Conversely, models that do encode structural information of code make modifications to the Transformer architecture, limiting their scale and compatibility with pretrained LLMs. In this work, we take the best of both worlds with GALLa - Graph Aligned Large Language Models. GALLa utilizes graph neural networks and cross-modal alignment technologies to inject the structural information of code into LLMs as an auxiliary task during finetuning. This framework is both model-agnostic and task-agnostic, as it can be applied to any code LLM for any code downstream task, and requires the structural graph data only at training time from a corpus unrelated to the finetuning data, while incurring no cost at inference time over the baseline LLM. Experiments on five code tasks with seven different baseline LLMs ranging in size from 350M to 14B validate the effectiveness of GALLa, demonstrating consistent improvement over the baseline, even for powerful models such as LLaMA3 and Qwen2.5-Coder.
>
---
#### [replaced 050] T2R-bench: A Benchmark for Generating Article-Level Reports from Real World Industrial Tables
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.19813v4](http://arxiv.org/pdf/2508.19813v4)**

> **作者:** Jie Zhang; Changzai Pan; Kaiwen Wei; Sishi Xiong; Yu Zhao; Xiangyu Li; Jiaxin Peng; Xiaoyan Gu; Jian Yang; Wenhan Chang; Zhenhe Wu; Jiang Zhong; Shuangyong Song; Yongxiang Li; Xuelong Li
>
> **摘要:** Extensive research has been conducted to explore the capabilities of large language models (LLMs) in table reasoning. However, the essential task of transforming tables information into reports remains a significant challenge for industrial applications. This task is plagued by two critical issues: 1) the complexity and diversity of tables lead to suboptimal reasoning outcomes; and 2) existing table benchmarks lack the capacity to adequately assess the practical application of this task. To fill this gap, we propose the table-to-report task and construct a bilingual benchmark named T2R-bench, where the key information flow from the tables to the reports for this task. The benchmark comprises 457 industrial tables, all derived from real-world scenarios and encompassing 19 industry domains as well as 4 types of industrial tables. Furthermore, we propose an evaluation criteria to fairly measure the quality of report generation. The experiments on 25 widely-used LLMs reveal that even state-of-the-art models like Deepseek-R1 only achieves performance with 62.71 overall score, indicating that LLMs still have room for improvement on T2R-bench.
>
---
#### [replaced 051] AI-Generated Text is Non-Stationary: Detection via Temporal Tomography
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.01754v2](http://arxiv.org/pdf/2508.01754v2)**

> **作者:** Alva West; Yixuan Weng; Minjun Zhu; Luodan Zhang; Zhen Lin; Guangsheng Bao; Yue Zhang
>
> **摘要:** The field of AI-generated text detection has evolved from supervised classification to zero-shot statistical analysis. However, current approaches share a fundamental limitation: they aggregate token-level measurements into scalar scores, discarding positional information about where anomalies occur. Our empirical analysis reveals that AI-generated text exhibits significant non-stationarity, statistical properties vary by 73.8\% more between text segments compared to human writing. This discovery explains why existing detectors fail against localized adversarial perturbations that exploit this overlooked characteristic. We introduce Temporal Discrepancy Tomography (TDT), a novel detection paradigm that preserves positional information by reformulating detection as a signal processing task. TDT treats token-level discrepancies as a time-series signal and applies Continuous Wavelet Transform to generate a two-dimensional time-scale representation, capturing both the location and linguistic scale of statistical anomalies. On the RAID benchmark, TDT achieves 0.855 AUROC (7.1\% improvement over the best baseline). More importantly, TDT demonstrates robust performance on adversarial tasks, with 14.1\% AUROC improvement on HART Level 2 paraphrasing attacks. Despite its sophisticated analysis, TDT maintains practical efficiency with only 13\% computational overhead. Our work establishes non-stationarity as a fundamental characteristic of AI-generated text and demonstrates that preserving temporal dynamics is essential for robust detection.
>
---
#### [replaced 052] JOLT-SQL: Joint Loss Tuning of Text-to-SQL with Confusion-aware Noisy Schema Sampling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14305v3](http://arxiv.org/pdf/2505.14305v3)**

> **作者:** Jinwang Song; Hongying Zan; Kunli Zhang; Lingling Mu; Yingjie Han; Haobo Hua; Min Peng
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Text-to-SQL, which maps natural language to SQL queries, has benefited greatly from recent advances in Large Language Models (LLMs). While LLMs offer various paradigms for this task, including prompting and supervised fine-tuning (SFT), SFT approaches still face challenges such as complex multi-stage pipelines and poor robustness to noisy schema information. To address these limitations, we present JOLT-SQL, a streamlined single-stage SFT framework that jointly optimizes schema linking and SQL generation via a unified loss. JOLT-SQL employs discriminative schema linking, enhanced by local bidirectional attention, alongside a confusion-aware noisy schema sampling strategy with selective attention to improve robustness under noisy schema conditions. Experiments on the Spider and BIRD benchmarks demonstrate that JOLT-SQL achieves state-of-the-art execution accuracy among comparable-size open-source models, while significantly improving both training and inference efficiency. Our code is available at https://github.com/Songjw133/JOLT-SQL.
>
---
#### [replaced 053] Large Language Models Do Multi-Label Classification Differently
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17510v2](http://arxiv.org/pdf/2505.17510v2)**

> **作者:** Marcus Ma; Georgios Chochlakis; Niyantha Maruthu Pandiyan; Jesse Thomason; Shrikanth Narayanan
>
> **备注:** To be published in the Main Conference Proceedings of EMNLP 2025, 24 pages, 16 figures, 7 tables
>
> **摘要:** Multi-label classification is prevalent in real-world settings, but the behavior of Large Language Models (LLMs) in this setting is understudied. We investigate how autoregressive LLMs perform multi-label classification, focusing on subjective tasks, by analyzing the output distributions of the models at each label generation step. We find that the initial probability distribution for the first label often does not reflect the eventual final output, even in terms of relative order and find LLMs tend to suppress all but one label at each generation step. We further observe that as model scale increases, their token distributions exhibit lower entropy and higher single-label confidence, but the internal relative ranking of the labels improves. Finetuning methods such as supervised finetuning and reinforcement learning amplify this phenomenon. We introduce the task of distribution alignment for multi-label settings: aligning LLM-derived label distributions with empirical distributions estimated from annotator responses in subjective tasks. We propose both zero-shot and supervised methods which improve both alignment and predictive performance over existing approaches. We find one method -- taking the max probability over all label generation distributions instead of just using the initial probability distribution -- improves both distribution alignment and overall F1 classification without adding any additional computation.
>
---
#### [replaced 054] Compositional Phoneme Approximation for L1-Grounded L2 Pronunciation Training
- **分类: cs.CL; cs.SD; eess.AS; H.5.5**

- **链接: [http://arxiv.org/pdf/2411.10927v4](http://arxiv.org/pdf/2411.10927v4)**

> **作者:** Jisang Park; Minu Kim; DaYoung Hong; Jongha Lee
>
> **摘要:** Learners of a second language (L2) often map non-native phonemes with similar native-language (L1) phonemes, making conventional L2-focused training slow and effortful. To address this, we propose an L1-grounded pronunciation training method based on compositional phoneme approximation (CPA), a feature-based representation technique that approximates L2 sounds with sequences of L1 phonemes. Evaluations with 20 Korean non-native English speakers show that CPA-based training achieves a 76% in-box formant rate in acoustic analysis, over 20% relative improvement in phoneme recognition accuracy, and over 80% of speech being rated as more native-like, with minimal training.
>
---
#### [replaced 055] OpenWHO: A Document-Level Parallel Corpus for Health Translation in Low-Resource Languages
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.16048v4](http://arxiv.org/pdf/2508.16048v4)**

> **作者:** Raphaël Merx; Hanna Suominen; Trevor Cohn; Ekaterina Vylomova
>
> **备注:** Accepted at WMT 2025
>
> **摘要:** In machine translation (MT), health is a high-stakes domain characterised by widespread deployment and domain-specific vocabulary. However, there is a lack of MT evaluation datasets for low-resource languages in this domain. To address this gap, we introduce OpenWHO, a document-level parallel corpus of 2,978 documents and 26,824 sentences from the World Health Organization's e-learning platform. Sourced from expert-authored, professionally translated materials shielded from web-crawling, OpenWHO spans a diverse range of over 20 languages, of which nine are low-resource. Leveraging this new resource, we evaluate modern large language models (LLMs) against traditional MT models. Our findings reveal that LLMs consistently outperform traditional MT models, with Gemini 2.5 Flash achieving a +4.79 ChrF point improvement over NLLB-54B on our low-resource test set. Further, we investigate how LLM context utilisation affects accuracy, finding that the benefits of document-level translation are most pronounced in specialised domains like health. We release the OpenWHO corpus to encourage further research into low-resource MT in the health domain.
>
---
#### [replaced 056] RAG+: Enhancing Retrieval-Augmented Generation with Application-Aware Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11555v4](http://arxiv.org/pdf/2506.11555v4)**

> **作者:** Yu Wang; Shiwan Zhao; Zhihu Wang; Ming Fan; Xicheng Zhang; Yubo Zhang; Zhengfan Wang; Heyuan Huang; Ting Liu
>
> **摘要:** The integration of external knowledge through Retrieval-Augmented Generation (RAG) has become foundational in enhancing large language models (LLMs) for knowledge-intensive tasks. However, existing RAG paradigms often overlook the cognitive step of applying knowledge, leaving a gap between retrieved facts and task-specific reasoning. In this work, we introduce RAG+, a principled and modular extension that explicitly incorporates application-aware reasoning into the RAG pipeline. RAG+ constructs a dual corpus consisting of knowledge and aligned application examples, created either manually or automatically, and retrieves both jointly during inference. This design enables LLMs not only to access relevant information but also to apply it within structured, goal-oriented reasoning processes. Experiments across mathematical, legal, and medical domains, conducted on multiple models, demonstrate that RAG+ consistently outperforms standard RAG variants, achieving average improvements of 3-5%, and peak gains up to 13.5% in complex scenarios. By bridging retrieval with actionable application, RAG+ advances a more cognitively grounded framework for knowledge integration, representing a step toward more interpretable and capable LLMs.
>
---
#### [replaced 057] Program Synthesis via Test-Time Transduction
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.17393v2](http://arxiv.org/pdf/2509.17393v2)**

> **作者:** Kang-il Lee; Jahyun Koo; Seunghyun Yoon; Minbeom Kim; Hyukhun Koh; Dongryeol Lee; Kyomin Jung
>
> **备注:** NeurIPS 2025
>
> **摘要:** We introduce transductive program synthesis, a new formulation of the program synthesis task that explicitly leverages test inputs during synthesis. While prior approaches to program synthesis--whether based on natural language descriptions or input-output examples--typically aim to generalize from training examples, they often struggle with robustness, especially in real-world settings where training examples are limited and test inputs involve various edge cases. To address this, we propose a novel framework that improves robustness by treating synthesis as an active learning over a finite hypothesis class defined by programs' outputs. We use an LLM to predict outputs for selected test inputs and eliminate inconsistent hypotheses, where the inputs are chosen via a greedy maximin algorithm to minimize the number of LLM queries required. We evaluate our approach on four benchmarks: Playgol, MBPP+, 1D-ARC, and programmatic world modeling on MiniGrid. We demonstrate that our method significantly improves program synthesis in both accuracy and efficiency. We release our code at https://github.com/klee972/SYNTRA.
>
---
#### [replaced 058] Abduct, Act, Predict: Scaffolding Causal Inference for Automated Failure Attribution in Multi-Agent Systems
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.10401v2](http://arxiv.org/pdf/2509.10401v2)**

> **作者:** Alva West; Yixuan Weng; Minjun Zhu; Zhen Lin; Zhiyuan Ning; Yue Zhang
>
> **摘要:** Failure attribution in multi-agent systems -- pinpointing the exact step where a decisive error occurs -- is a critical yet unsolved challenge. Current methods treat this as a pattern recognition task over long conversation logs, leading to critically low step-level accuracy (below 17\%), which renders them impractical for debugging complex systems. Their core weakness is a fundamental inability to perform robust counterfactual reasoning: to determine if correcting a single action would have actually averted the task failure. To bridge this \emph{counterfactual inference gap}, we introduce Abduct-Act-Predict (A2P) Scaffolding, a novel agent framework that transforms failure attribution from pattern recognition into a structured causal inference task. A2P explicitly guides a large language model through a formal three-step reasoning process within a single inference pass: (1) Abduction, to infer the hidden root causes behind an agent's actions; (2) Action, to define a minimal corrective intervention; and (3) Prediction, to simulate the subsequent trajectory and verify if the intervention resolves the failure. This structured approach leverages the holistic context of the entire conversation while imposing a rigorous causal logic on the model's analysis. Our extensive experiments on the Who\&When benchmark demonstrate its efficacy. On the Algorithm-Generated dataset, A2P achieves 47.46\% step-level accuracy, a 2.85$\times$ improvement over the 16.67\% of the baseline. On the more complex Hand-Crafted dataset, it achieves 29.31\% step accuracy, a 2.43$\times$ improvement over the baseline's 12.07\%. By reframing the problem through a causal lens, A2P Scaffolding provides a robust, verifiable, and significantly more accurate solution for automated failure attribution. Ours code are released at https://github.com/ResearAI/A2P.
>
---
#### [replaced 059] Retrieval Enhanced Feedback via In-context Neural Error-book
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.16313v4](http://arxiv.org/pdf/2508.16313v4)**

> **作者:** Jongyeop Hyun; Bumsoo Kim
>
> **备注:** Accepted at EMNLP 2025 main conference
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have significantly improved reasoning capabilities, with in-context learning (ICL) emerging as a key technique for adaptation without retraining. While previous works have focused on leveraging correct examples, recent research highlights the importance of learning from errors to enhance performance. However, existing methods lack a structured framework for analyzing and mitigating errors, particularly in Multimodal Large Language Models (MLLMs), where integrating visual and textual inputs adds complexity. To address this issue, we propose REFINE: Retrieval-Enhanced Feedback via In-context Neural Error-book, a teacher-student framework that systematically structures errors and provides targeted feedback. REFINE introduces three systematic queries to construct structured feedback -- Feed-Target, Feed-Check, and Feed-Path -- to enhance multimodal reasoning by prioritizing relevant visual information, diagnosing critical failure points, and formulating corrective actions. Unlike prior approaches that rely on redundant retrievals, REFINE optimizes structured feedback retrieval, improving inference efficiency, token usage, and scalability. Our results demonstrate substantial speedup, reduced computational costs, and successful generalization, highlighting REFINE's potential for enhancing multimodal reasoning.
>
---
#### [replaced 060] LogicGuard: Improving Embodied LLM agents through Temporal Logic based Critics
- **分类: cs.AI; cs.CL; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.03293v2](http://arxiv.org/pdf/2507.03293v2)**

> **作者:** Anand Gokhale; Vaibhav Srivastava; Francesco Bullo
>
> **备注:** Modified version of prior LTLCrit work with new robotics dataset
>
> **摘要:** Large language models (LLMs) have shown promise in zero-shot and single step reasoning and decision making problems, but in long horizon sequential planning tasks, their errors compound, often leading to unreliable or inefficient behavior. We introduce LogicGuard, a modular actor-critic architecture in which an LLM actor is guided by a trajectory level LLM critic that communicates through Linear Temporal Logic (LTL). Our setup combines the reasoning strengths of language models with the guarantees of formal logic. The actor selects high-level actions from natural language observations, while the critic analyzes full trajectories and proposes new LTL constraints that shield the actor from future unsafe or inefficient behavior. LogicGuard supports both fixed safety rules and adaptive, learned constraints, and is model-agnostic: any LLM-based planner can serve as the actor, with LogicGuard acting as a logic-generating wrapper. We formalize planning as graph traversal under symbolic constraints, allowing LogicGuard to analyze failed or suboptimal trajectories and generate new temporal logic rules that improve future behavior. To demonstrate generality, we evaluate LogicGuard across two distinct settings: short-horizon general tasks and long-horizon specialist tasks. On the Behavior benchmark of 100 household tasks, LogicGuard increases task completion rates by 25% over a baseline InnerMonologue planner. On the Minecraft diamond-mining task, which is long-horizon and requires multiple interdependent subgoals, LogicGuard improves both efficiency and safety compared to SayCan and InnerMonologue. These results show that enabling LLMs to supervise each other through temporal logic yields more reliable, efficient and safe decision-making for both embodied agents.
>
---
#### [replaced 061] Meta-Semantics Augmented Few-Shot Relational Learning
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05684v3](http://arxiv.org/pdf/2505.05684v3)**

> **作者:** Han Wu; Jie Yin
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Few-shot relational learning on knowledge graph (KGs) aims to perform reasoning over relations with only a few training examples. While current methods have focused primarily on leveraging specific relational information, rich semantics inherent in KGs have been largely overlooked. To bridge this gap, we propose PromptMeta, a novel prompted meta-learning framework that seamlessly integrates meta-semantics with relational information for few-shot relational learning. PromptMeta introduces two core innovations: (1) a Meta-Semantic Prompt (MSP) pool that learns and consolidates high-level meta-semantics shared across tasks, enabling effective knowledge transfer and adaptation to newly emerging relations; and (2) a learnable fusion mechanism that dynamically combines meta-semantics with task-specific relational information tailored to different few-shot tasks. Both components are optimized jointly with model parameters within a meta-learning framework. Extensive experiments and analyses on two real-world KG benchmarks validate the effectiveness of PromptMeta in adapting to new relations with limited supervision.
>
---
#### [replaced 062] Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.04802v2](http://arxiv.org/pdf/2509.04802v2)**

> **作者:** Ilham Wicaksono; Zekun Wu; Rahul Patel; Theo King; Adriano Koshiyama; Philip Treleaven
>
> **摘要:** As large language models transition to agentic systems, current safety evaluation frameworks face critical gaps in assessing deployment-specific risks. We introduce AgentSeer, an observability-based evaluation framework that decomposes agentic executions into granular action and component graphs, enabling systematic agentic-situational assessment. Through cross-model validation on GPT-OSS-20B and Gemini-2.0-flash using HarmBench single turn and iterative refinement attacks, we demonstrate fundamental differences between model-level and agentic-level vulnerability profiles. Model-level evaluation reveals baseline differences: GPT-OSS-20B (39.47% ASR) versus Gemini-2.0-flash (50.00% ASR), with both models showing susceptibility to social engineering while maintaining logic-based attack resistance. However, agentic-level assessment exposes agent-specific risks invisible to traditional evaluation. We discover "agentic-only" vulnerabilities that emerge exclusively in agentic contexts, with tool-calling showing 24-60% higher ASR across both models. Cross-model analysis reveals universal agentic patterns, agent transfer operations as highest-risk tools, semantic rather than syntactic vulnerability mechanisms, and context-dependent attack effectiveness, alongside model-specific security profiles in absolute ASR levels and optimal injection strategies. Direct attack transfer from model-level to agentic contexts shows degraded performance (GPT-OSS-20B: 57% human injection ASR; Gemini-2.0-flash: 28%), while context-aware iterative attacks successfully compromise objectives that failed at model-level, confirming systematic evaluation gaps. These findings establish the urgent need for agentic-situation evaluation paradigms, with AgentSeer providing the standardized methodology and empirical validation.
>
---
#### [replaced 063] LongLLaVA: Scaling Multi-modal LLMs to 1000 Images Efficiently via a Hybrid Architecture
- **分类: cs.CL; cs.AI; cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2409.02889v3](http://arxiv.org/pdf/2409.02889v3)**

> **作者:** Xidong Wang; Dingjie Song; Shunian Chen; Junyin Chen; Zhenyang Cai; Chen Zhang; Lichao Sun; Benyou Wang
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Expanding the long-context capabilities of Multi-modal Large Language Models~(MLLMs) is critical for advancing video understanding and high-resolution image analysis. Achieving this requires systematic improvements in model architecture, data construction, and training strategies, particularly to address challenges such as performance degradation with increasing image counts and high computational costs. In this paper, we propose a hybrid architecture that integrates Mamba and Transformer blocks, introduce data construction methods that capture both temporal and spatial dependencies, and employ a progressive training strategy. Our released model, LongLLaVA (\textbf{Long}-Context \textbf{L}arge \textbf{L}anguage \textbf{a}nd \textbf{V}ision \textbf{A}ssistant), demonstrates an effective balance between efficiency and performance. LongLLaVA achieves competitive results across various benchmarks while maintaining high throughput and low memory consumption. Notably, it can process nearly one thousand images on a single A100 80GB GPU, underscoring its potential for a wide range of multi-modal applications.
>
---
#### [replaced 064] Pandora: A Code-Driven Large Language Model Agent for Unified Reasoning Across Diverse Structured Knowledge
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.12734v2](http://arxiv.org/pdf/2504.12734v2)**

> **作者:** Yongrui Chen; Junhao He; Linbo Fu; Shenyu Zhang; Rihui Jin; Xinbang Dai; Jiaqi Li; Dehai Min; Nan Hu; Yuxin Zhang; Guilin Qi; Yi Huang; Tongtong Wu
>
> **备注:** New version is arXiv:2508.17905
>
> **摘要:** Unified Structured Knowledge Reasoning (USKR) aims to answer natural language questions (NLQs) by using structured sources such as tables, databases, and knowledge graphs in a unified way. Existing USKR methods either rely on employing task-specific strategies or custom-defined representations, which struggle to leverage the knowledge transfer between different SKR tasks or align with the prior of LLMs, thereby limiting their performance. This paper proposes a novel USKR framework named \textsc{Pandora}, which takes advantage of \textsc{Python}'s \textsc{Pandas} API to construct a unified knowledge representation for alignment with LLM pre-training. It employs an LLM to generate textual reasoning steps and executable Python code for each question. Demonstrations are drawn from a memory of training examples that cover various SKR tasks, facilitating knowledge transfer. Extensive experiments on four benchmarks involving three SKR tasks demonstrate that \textsc{Pandora} outperforms existing unified frameworks and competes effectively with task-specific methods.
>
---
#### [replaced 065] Large Vision-Language Model Alignment and Misalignment: A Survey Through the Lens of Explainability
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.01346v3](http://arxiv.org/pdf/2501.01346v3)**

> **作者:** Dong Shu; Haiyan Zhao; Jingyu Hu; Weiru Liu; Ali Payani; Lu Cheng; Mengnan Du
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities in processing both visual and textual information. However, the critical challenge of alignment between visual and textual representations is not fully understood. This survey presents a comprehensive examination of alignment and misalignment in LVLMs through an explainability lens. We first examine the fundamentals of alignment, exploring its representational and behavioral aspects, training methodologies, and theoretical foundations. We then analyze misalignment phenomena across three semantic levels: object, attribute, and relational misalignment. Our investigation reveals that misalignment emerges from challenges at multiple levels: the data level, the model level, and the inference level. We provide a comprehensive review of existing mitigation strategies, categorizing them into parameter-frozen and parameter-tuning approaches. Finally, we outline promising future research directions, emphasizing the need for standardized evaluation protocols and in-depth explainability studies.
>
---
#### [replaced 066] A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.05613v3](http://arxiv.org/pdf/2503.05613v3)**

> **作者:** Dong Shu; Xuansheng Wu; Haiyan Zhao; Daking Rai; Ziyu Yao; Ninghao Liu; Mengnan Du
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Large Language Models (LLMs) have transformed natural language processing, yet their internal mechanisms remain largely opaque. Recently, mechanistic interpretability has attracted significant attention from the research community as a means to understand the inner workings of LLMs. Among various mechanistic interpretability approaches, Sparse Autoencoders (SAEs) have emerged as a promising method due to their ability to disentangle the complex, superimposed features within LLMs into more interpretable components. This paper presents a comprehensive survey of SAEs for interpreting and understanding the internal workings of LLMs. Our major contributions include: (1) exploring the technical framework of SAEs, covering basic architecture, design improvements, and effective training strategies; (2) examining different approaches to explaining SAE features, categorized into input-based and output-based explanation methods; (3) discussing evaluation methods for assessing SAE performance, covering both structural and functional metrics; and (4) investigating real-world applications of SAEs in understanding and manipulating LLM behaviors.
>
---
#### [replaced 067] Are Vision-Language Models Safe in the Wild? A Meme-Based Benchmark Study
- **分类: cs.CL; cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15389v3](http://arxiv.org/pdf/2505.15389v3)**

> **作者:** DongGeon Lee; Joonwon Jang; Jihae Jeong; Hwanjo Yu
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Rapid deployment of vision-language models (VLMs) magnifies safety risks, yet most evaluations rely on artificial images. This study asks: How safe are current VLMs when confronted with meme images that ordinary users share? To investigate this question, we introduce MemeSafetyBench, a 50,430-instance benchmark pairing real meme images with both harmful and benign instructions. Using a comprehensive safety taxonomy and LLM-based instruction generation, we assess multiple VLMs across single and multi-turn interactions. We investigate how real-world memes influence harmful outputs, the mitigating effects of conversational context, and the relationship between model scale and safety metrics. Our findings demonstrate that VLMs are more vulnerable to meme-based harmful prompts than to synthetic or typographic images. Memes significantly increase harmful responses and decrease refusals compared to text-only inputs. Though multi-turn interactions provide partial mitigation, elevated vulnerability persists. These results highlight the need for ecologically valid evaluations and stronger safety mechanisms. MemeSafetyBench is publicly available at https://github.com/oneonlee/Meme-Safety-Bench.
>
---
#### [replaced 068] ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.15235v3](http://arxiv.org/pdf/2509.15235v3)**

> **作者:** Jialiang Kang; Han Shu; Wenshuo Li; Yingjie Zhai; Xinghao Chen
>
> **备注:** NeurIPS 2025
>
> **摘要:** Speculative decoding is a widely adopted technique for accelerating inference in large language models (LLMs), yet its application to vision-language models (VLMs) remains underexplored, with existing methods achieving only modest speedups (<1.5x). This gap is increasingly significant as multimodal capabilities become central to large-scale models. We hypothesize that large VLMs can effectively filter redundant image information layer by layer without compromising textual comprehension, whereas smaller draft models struggle to do so. To address this, we introduce Vision-Aware Speculative Decoding (ViSpec), a novel framework tailored for VLMs. ViSpec employs a lightweight vision adaptor module to compress image tokens into a compact representation, which is seamlessly integrated into the draft model's attention mechanism while preserving original image positional information. Additionally, we extract a global feature vector for each input image and augment all subsequent text tokens with this feature to enhance multimodal coherence. To overcome the scarcity of multimodal datasets with long assistant responses, we curate a specialized training dataset by repurposing existing datasets and generating extended outputs using the target VLM with modified prompts. Our training strategy mitigates the risk of the draft model exploiting direct access to the target model's hidden states, which could otherwise lead to shortcut learning when training solely on target model outputs. Extensive experiments validate ViSpec, achieving, to our knowledge, the first substantial speedup in VLM speculative decoding. Code is available at https://github.com/KangJialiang/ViSpec.
>
---
#### [replaced 069] DeepResonance: Enhancing Multimodal Music Understanding via Music-centric Multi-way Instruction Tuning
- **分类: cs.SD; cs.AI; cs.CL; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.12623v3](http://arxiv.org/pdf/2502.12623v3)**

> **作者:** Zhuoyuan Mao; Mengjie Zhao; Qiyu Wu; Hiromi Wakaki; Yuki Mitsufuji
>
> **备注:** Accepted to EMNLP 2025 main conference
>
> **摘要:** Recent advancements in music large language models (LLMs) have significantly improved music understanding tasks, which involve the model's ability to analyze and interpret various musical elements. These improvements primarily focused on integrating both music and text inputs. However, the potential of incorporating additional modalities such as images, videos and textual music features to enhance music understanding remains unexplored. To bridge this gap, we propose DeepResonance, a multimodal music understanding LLM fine-tuned via multi-way instruction tuning with multi-way aligned music, text, image, and video data. To this end, we construct Music4way-MI2T, Music4way-MV2T, and Music4way-Any2T, three 4-way training and evaluation datasets designed to enable DeepResonance to integrate both visual and textual music feature content. We also introduce multi-sampled ImageBind embeddings and a pre-LLM fusion Transformer to enhance modality fusion prior to input into text LLMs, tailoring for multi-way instruction tuning. Our model achieves state-of-the-art performances across six music understanding tasks, highlighting the benefits of the auxiliary modalities and the structural superiority of DeepResonance. We open-source the codes, models and datasets we constructed: github.com/sony/DeepResonance.
>
---
#### [replaced 070] Please Translate Again: Two Simple Experiments on Whether Human-Like Reasoning Helps Translation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04521v2](http://arxiv.org/pdf/2506.04521v2)**

> **作者:** Di Wu; Seth Aycock; Christof Monz
>
> **备注:** EMNLP Main 2025 17 pages, 15 figures
>
> **摘要:** Large Language Models (LLMs) demonstrate strong reasoning capabilities for many tasks, often by explicitly decomposing the task via Chain-of-Thought (CoT) reasoning. Recent work on LLM-based translation designs hand-crafted prompts to decompose translation, or trains models to incorporate intermediate steps. Translating Step-by-step (Briakou et al., 2024), for instance, introduces a multi-step prompt with decomposition and refinement of translation with LLMs, which achieved state-of-the-art results on WMT24 test data. In this work, we scrutinise this strategy's effectiveness. Empirically, we find no clear evidence that performance gains stem from explicitly decomposing the translation process via CoT, at least for the models on test; and we show prompting LLMs to 'translate again' and self-refine yields even better results than human-like step-by-step prompting. While the decomposition influences translation behaviour, faithfulness to the decomposition has both positive and negative effects on translation. Our analysis therefore suggests a divergence between the optimal translation strategies for humans and LLMs.
>
---
#### [replaced 071] Language Models as Causal Effect Generators
- **分类: cs.CL; cs.AI; cs.LG; stat.AP; stat.ME; stat.ML**

- **链接: [http://arxiv.org/pdf/2411.08019v2](http://arxiv.org/pdf/2411.08019v2)**

> **作者:** Lucius E. J. Bynum; Kyunghyun Cho
>
> **摘要:** In this work, we present sequence-driven structural causal models (SD-SCMs), a framework for specifying causal models with user-defined structure and language-model-defined mechanisms. We characterize how an SD-SCM enables sampling from observational, interventional, and counterfactual distributions according to the desired causal structure. We then leverage this procedure to propose a new type of benchmark for causal inference methods, generating individual-level counterfactual data to test treatment effect estimation. We create an example benchmark consisting of thousands of datasets, and test a suite of popular estimation methods for average, conditional average, and individual treatment effect estimation. We find under this benchmark that (1) causal methods outperform non-causal methods and that (2) even state-of-the-art methods struggle with individualized effect estimation, suggesting this benchmark captures some inherent difficulties in causal estimation. Apart from generating data, this same technique can underpin the auditing of language models for (un)desirable causal effects, such as misinformation or discrimination. We believe SD-SCMs can serve as a useful tool in any application that would benefit from sequential data with controllable causal structure.
>
---
#### [replaced 072] Combining Constrained and Unconstrained Decoding via Boosting: BoostCD and Its Application to Information Extraction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14901v2](http://arxiv.org/pdf/2506.14901v2)**

> **作者:** Marija Šakota; Robert West
>
> **摘要:** Many recent approaches to structured NLP tasks use an autoregressive language model $M$ to map unstructured input text $x$ to output text $y$ representing structured objects (such as tuples, lists, trees, code, etc.), where the desired output structure is enforced via constrained decoding. During training, these approaches do not require the model to be aware of the constraints, which are merely implicit in the training outputs $y$. This is advantageous as it allows for dynamic constraints without requiring retraining, but can lead to low-quality output during constrained decoding at test time. We overcome this problem with Boosted Constrained Decoding (BoostCD), which combines constrained and unconstrained decoding in two phases: Phase 1 decodes from the base model $M$ twice, in constrained and unconstrained mode, obtaining two weak predictions. In phase 2, a learned autoregressive boosted model combines the two weak predictions into one final prediction. The mistakes made by the base model with vs. without constraints tend to be complementary, which the boosted model learns to exploit for improved performance. We demonstrate the power of BoostCD by applying it to closed information extraction. Our model, BoostIE, outperforms prior approaches both in and out of distribution, addressing several common errors identified in those approaches.
>
---
#### [replaced 073] ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.19470v3](http://arxiv.org/pdf/2503.19470v3)**

> **作者:** Mingyang Chen; Linzhuang Sun; Tianpeng Li; Haoze Sun; Yijie Zhou; Chenzheng Zhu; Haofen Wang; Jeff Z. Pan; Wen Zhang; Huajun Chen; Fan Yang; Zenan Zhou; Weipeng Chen
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities in reasoning, exemplified by the success of OpenAI-o1 and DeepSeek-R1. However, integrating reasoning with external search processes remains challenging, especially for complex multi-hop questions requiring multiple retrieval steps. We propose ReSearch, a novel framework that trains LLMs to Reason with Search via reinforcement learning without using any supervised data on reasoning steps. Our approach treats search operations as integral components of the reasoning chain, where when and how to perform searches is guided by text-based thinking, and search results subsequently influence further reasoning. We train ReSearch on Qwen2.5-7B(-Instruct) and Qwen2.5-32B(-Instruct) models and conduct extensive experiments. Despite being trained on only one dataset, our models demonstrate strong generalizability across various benchmarks. Analysis reveals that ReSearch naturally elicits advanced reasoning capabilities such as reflection and self-correction during the reinforcement learning process.
>
---
#### [replaced 074] Generative Medical Event Models Improve with Scale
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.12104v2](http://arxiv.org/pdf/2508.12104v2)**

> **作者:** Shane Waxler; Paul Blazek; Davis White; Daniel Sneider; Kevin Chung; Mani Nagarathnam; Patrick Williams; Hank Voeller; Karen Wong; Matthew Swanhorst; Sheng Zhang; Naoto Usuyama; Cliff Wong; Tristan Naumann; Hoifung Poon; Andrew Loza; Daniella Meeker; Seth Hain; Rahul Shah
>
> **摘要:** Realizing personalized medicine at scale calls for methods that distill insights from longitudinal patient journeys, which can be viewed as a sequence of medical events. Foundation models pretrained on large-scale medical event data represent a promising direction for scaling real-world evidence generation and generalizing to diverse downstream tasks. Using Epic Cosmos, a dataset with medical events from de-identified longitudinal health records for 16.3 billion encounters over 300 million unique patient records from 310 health systems, we introduce the Comet models, a family of decoder-only transformer models pretrained on 118 million patients representing 115 billion discrete medical events (151 billion tokens). We present the largest scaling-law study of medical event data, establishing a methodology for pretraining and revealing power-law scaling relationships for compute, tokens, and model size. Consequently, we pretrained a series of compute-optimal models with up to 1 billion parameters. Conditioned on a patient's real-world history, Comet autoregressively predicts the next medical event to simulate patient health timelines. We studied 78 real-world tasks, including diagnosis prediction, disease prognosis, and healthcare operations. Remarkably for a foundation model with generic pretraining and simulation-based inference, Comet generally outperformed or matched task-specific supervised models on these tasks, without requiring task-specific fine-tuning or few-shot examples. Comet's predictive power consistently improves as the model and pretraining scale. Our results show that Comet, a generative medical event foundation model, can effectively capture complex clinical dynamics, providing an extensible and generalizable framework to support clinical decision-making, streamline healthcare operations, and improve patient outcomes.
>
---
#### [replaced 075] CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.16356v2](http://arxiv.org/pdf/2503.16356v2)**

> **作者:** Yunzhi Yao; Jizhan Fang; Jia-Chen Gu; Ningyu Zhang; Shumin Deng; Huajun Chen; Nanyun Peng
>
> **备注:** EMNLP 2025
>
> **摘要:** Knowledge Editing (KE) enables the modification of outdated or incorrect information in large language models (LLMs). While existing KE methods can update isolated facts, they often fail to generalize these updates to multi-hop reasoning tasks that rely on the modified knowledge. Through an analysis of reasoning circuits -- the neural pathways LLMs use for knowledge-based inference, we find that current layer-localized KE approaches (e.g., MEMIT, WISE), which edit only single or a few model layers, inadequately integrate updated knowledge into these reasoning pathways. To address this limitation, we present CaKE (Circuit-aware Knowledge Editing), a novel method that enhances the effective integration of updated knowledge in LLMs. By only leveraging a few curated data samples guided by our circuit-based analysis, CaKE stimulates the model to develop appropriate reasoning circuits for newly incorporated knowledge. Experiments show that CaKE enables more accurate and consistent use of edited knowledge across related reasoning tasks, achieving an average improvement of 20% in multi-hop reasoning accuracy on the MQuAKE dataset while requiring less memory than existing KE methods. We release the code and data in https://github.com/zjunlp/CaKE.
>
---
#### [replaced 076] When and How Long Did Therapy Happen? Soft-Supervising Temporal Localization Using Audio-Language Models
- **分类: eess.AS; cs.CL; cs.HC; 68T07; I.2.7; I.5.4; H.5.2**

- **链接: [http://arxiv.org/pdf/2506.09707v3](http://arxiv.org/pdf/2506.09707v3)**

> **作者:** Suhas BN; Andrew M. Sherrill; Jyoti Alaparthi; Dominik Mattioli; Rosa I. Arriaga; Chris W. Wiese; Saeed Abdullah
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Prolonged Exposure (PE) therapy is an effective treatment for post-traumatic stress disorder (PTSD), but evaluating therapist fidelity remains labor-intensive due to the need for manual review of session recordings. We present a method for the automatic temporal localization of key PE fidelity elements, identifying their start and stop times, directly from session audio and transcripts. Our approach fine-tunes a large pre-trained audio-language model, Qwen2-Audio, using Low-Rank Adaptation (LoRA) to process focused 30-second windows of audio-transcript input. Fidelity labels for three core protocol phases, therapist orientation (P1), imaginal exposure (P2), and post-imaginal processing (P3), are generated via LLM-based prompting and verified by trained raters. The model is trained to predict normalized boundary offsets using soft supervision guided by task-specific prompts. On a dataset of 308 real PE sessions, our best configuration (LoRA rank 8, 30s windows) achieves a mean absolute error (MAE) of 5.3s across tasks, within typical rater tolerance for timestamp review, enabling practical fidelity QC. We further analyze the effects of window size and LoRA rank, highlighting the importance of context granularity and model adaptation. This work introduces a privacy-preserving, scalable framework for fidelity tracking in PE therapy, with potential to support clinician training, supervision, and quality assurance.
>
---
#### [replaced 077] Fine-Grained Detection of AI-Generated Text Using Sentence-Level Segmentation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.17830v2](http://arxiv.org/pdf/2509.17830v2)**

> **作者:** Lekkala Sai Teja; Annepaka Yadagiri; Partha Pakray; Chukhu Chunka; Mangadoddi Srikar Vardhan
>
> **备注:** 14 pages, 14 figures
>
> **摘要:** Generation of Artificial Intelligence (AI) texts in important works has become a common practice that can be used to misuse and abuse AI at various levels. Traditional AI detectors often rely on document-level classification, which struggles to identify AI content in hybrid or slightly edited texts designed to avoid detection, leading to concerns about the model's efficiency, which makes it hard to distinguish between human-written and AI-generated texts. A sentence-level sequence labeling model proposed to detect transitions between human- and AI-generated text, leveraging nuanced linguistic signals overlooked by document-level classifiers. By this method, detecting and segmenting AI and human-written text within a single document at the token-level granularity is achieved. Our model combines the state-of-the-art pre-trained Transformer models, incorporating Neural Networks (NN) and Conditional Random Fields (CRFs). This approach extends the power of transformers to extract semantic and syntactic patterns, and the neural network component to capture enhanced sequence-level representations, thereby improving the boundary predictions by the CRF layer, which enhances sequence recognition and further identification of the partition between Human- and AI-generated texts. The evaluation is performed on two publicly available benchmark datasets containing collaborative human and AI-generated texts. Our experimental comparisons are with zero-shot detectors and the existing state-of-the-art models, along with rigorous ablation studies to justify that this approach, in particular, can accurately detect the spans of AI texts in a completely collaborative text. All our source code and the processed datasets are available in our GitHub repository.
>
---
#### [replaced 078] EMMA: End-to-End Multimodal Model for Autonomous Driving
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.23262v3](http://arxiv.org/pdf/2410.23262v3)**

> **作者:** Jyh-Jing Hwang; Runsheng Xu; Hubert Lin; Wei-Chih Hung; Jingwei Ji; Kristy Choi; Di Huang; Tong He; Paul Covington; Benjamin Sapp; Yin Zhou; James Guo; Dragomir Anguelov; Mingxing Tan
>
> **备注:** Accepted by TMLR. Blog post: https://waymo.com/blog/2024/10/introducing-emma/
>
> **摘要:** We introduce EMMA, an End-to-end Multimodal Model for Autonomous driving. Built upon a multi-modal large language model foundation like Gemini, EMMA directly maps raw camera sensor data into various driving-specific outputs, including planner trajectories, perception objects, and road graph elements. EMMA maximizes the utility of world knowledge from the pre-trained large language models, by representing all non-sensor inputs (e.g. navigation instructions and ego vehicle status) and outputs (e.g. trajectories and 3D locations) as natural language text. This approach allows EMMA to jointly process various driving tasks in a unified language space, and generate the outputs for each task using task-specific prompts. Empirically, we demonstrate EMMA's effectiveness by achieving state-of-the-art performance in motion planning on nuScenes as well as competitive results on the Waymo Open Motion Dataset (WOMD). EMMA also yields competitive results for camera-primary 3D object detection on the Waymo Open Dataset (WOD). We show that co-training EMMA with planner trajectories, object detection, and road graph tasks yields improvements across all three domains, highlighting EMMA's potential as a generalist model for autonomous driving applications. We hope that our results will inspire research to further evolve the state of the art in autonomous driving model architectures.
>
---
#### [replaced 079] Privacy-Aware In-Context Learning for Large Language Models
- **分类: cs.LG; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2509.13625v3](http://arxiv.org/pdf/2509.13625v3)**

> **作者:** Bishnu Bhusal; Manoj Acharya; Ramneet Kaur; Colin Samplawski; Anirban Roy; Adam D. Cobb; Rohit Chadha; Susmit Jha
>
> **摘要:** Large language models (LLMs) have significantly transformed natural language understanding and generation, but they raise privacy concerns due to potential exposure of sensitive information. Studies have highlighted the risk of information leakage, where adversaries can extract sensitive information embedded in the prompts. In this work, we introduce a novel private prediction framework for generating high-quality synthetic text with strong privacy guarantees. Our approach leverages the Differential Privacy (DP) framework to ensure worst-case theoretical bounds on information leakage without requiring any fine-tuning of the underlying models. The proposed method performs inference on private records and aggregates the resulting per-token output distributions. This enables the generation of longer and coherent synthetic text while maintaining privacy guarantees. Additionally, we propose a simple blending operation that combines private and public inference to further enhance utility. Empirical evaluations demonstrate that our approach outperforms previous state-of-the-art methods on in-context-learning (ICL) tasks, making it a promising direction for privacy-preserving text generation while maintaining high utility.
>
---
#### [replaced 080] SoK: Large Language Model Copyright Auditing via Fingerprinting
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.19843v2](http://arxiv.org/pdf/2508.19843v2)**

> **作者:** Shuo Shao; Yiming Li; Yu He; Hongwei Yao; Wenyuan Yang; Dacheng Tao; Zhan Qin
>
> **摘要:** The broad capabilities and substantial resources required to train Large Language Models (LLMs) make them valuable intellectual property, yet they remain vulnerable to copyright infringement, such as unauthorized use and model theft. LLM fingerprinting, a non-intrusive technique that extracts and compares the distinctive features from LLMs to identify infringements, offers a promising solution to copyright auditing. However, its reliability remains uncertain due to the prevalence of diverse model modifications and the lack of standardized evaluation. In this SoK, we present the first comprehensive study of LLM fingerprinting. We introduce a unified framework and formal taxonomy that categorizes existing methods into white-box and black-box approaches, providing a structured overview of the state of the art. We further propose LeaFBench, the first systematic benchmark for evaluating LLM fingerprinting under realistic deployment scenarios. Built upon mainstream foundation models and comprising 149 distinct model instances, LeaFBench integrates 13 representative post-development techniques, spanning both parameter-altering methods (e.g., fine-tuning, quantization) and parameter-independent mechanisms (e.g., system prompts, RAG). Extensive experiments on LeaFBench reveal the strengths and weaknesses of existing methods, thereby outlining future research directions and critical open problems in this emerging field. The code is available at https://github.com/shaoshuo-ss/LeaFBench.
>
---
#### [replaced 081] RPG: A Repository Planning Graph for Unified and Scalable Codebase Generation
- **分类: cs.CL; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2509.16198v2](http://arxiv.org/pdf/2509.16198v2)**

> **作者:** Jane Luo; Xin Zhang; Steven Liu; Jie Wu; Yiming Huang; Yangyu Huang; Chengyu Yin; Ying Xin; Jianfeng Liu; Yuefeng Zhan; Hao Sun; Qi Chen; Scarlett Li; Mao Yang
>
> **摘要:** Large language models excel at function- and file-level code generation, yet generating complete repositories from scratch remains a fundamental challenge. This process demands coherent and reliable planning across proposal- and implementation-level stages, while natural language, due to its ambiguity and verbosity, is ill-suited for faithfully representing complex software structures. To address this, we introduce the Repository Planning Graph (RPG), a persistent representation that unifies proposal- and implementation-level planning by encoding capabilities, file structures, data flows, and functions in one graph. RPG replaces ambiguous natural language with an explicit blueprint, enabling long-horizon planning and scalable repository generation. Building on RPG, we develop ZeroRepo, a graph-driven framework for repository generation from scratch. It operates in three stages: proposal-level planning and implementation-level refinement to construct the graph, followed by graph-guided code generation with test validation. To evaluate this setting, we construct RepoCraft, a benchmark of six real-world projects with 1,052 tasks. On RepoCraft, ZeroRepo generates repositories averaging 36K Code Lines, roughly 3.9$\times$ the strongest baseline (Claude Code) and about 64$\times$ other baselines. It attains 81.5% functional coverage and a 69.7% pass rate, exceeding Claude Code by 27.3 and 35.8 percentage points, respectively. Further analysis shows that RPG models complex dependencies, enables progressively more sophisticated planning through near-linear scaling, and enhances LLM understanding of repositories, thereby accelerating agent localization.
>
---
#### [replaced 082] Benchmarking Critical Questions Generation: A Challenging Reasoning Task for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11341v3](http://arxiv.org/pdf/2505.11341v3)**

> **作者:** Banca Calvo Figueras; Rodrigo Agerri
>
> **摘要:** The task of Critical Questions Generation (CQs-Gen) aims to foster critical thinking by enabling systems to generate questions that expose underlying assumptions and challenge the validity of argumentative reasoning structures. Despite growing interest in this area, progress has been hindered by the lack of suitable datasets and automatic evaluation standards. This paper presents a comprehensive approach to support the development and benchmarking of systems for this task. We construct the first large-scale dataset including ~5K manually annotated questions. We also investigate automatic evaluation methods and propose reference-based techniques as the strategy that best correlates with human judgments. Our zero-shot evaluation of 11 LLMs establishes a strong baseline while showcasing the difficulty of the task. Data and code plus a public leaderboard are provided to encourage further research, not only in terms of model performance, but also to explore the practical benefits of CQs-Gen for both automated reasoning and human critical thinking.
>
---
#### [replaced 083] Promote, Suppress, Iterate: How Language Models Answer One-to-Many Factual Queries
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.20475v3](http://arxiv.org/pdf/2502.20475v3)**

> **作者:** Tianyi Lorena Yan; Robin Jia
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** To answer one-to-many factual queries (e.g., listing cities of a country), a language model (LM) must simultaneously recall knowledge and avoid repeating previous answers. How are these two subtasks implemented and integrated internally? Across multiple datasets, models, and prompt templates, we identify a promote-then-suppress mechanism: the model first recalls all answers, and then suppresses previously generated ones. Specifically, LMs use both the subject and previous answer tokens to perform knowledge recall, with attention propagating subject information and MLPs promoting the answers. Then, attention attends to and suppresses previous answer tokens, while MLPs amplify the suppression signal. Our mechanism is corroborated by extensive experimental evidence: in addition to using early decoding and causal tracing, we analyze how components use different tokens by introducing both Token Lens, which decodes aggregated attention updates from specified tokens, and a knockout method that analyzes changes in MLP outputs after removing attention to specified tokens. Overall, we provide new insights into how LMs' internal components interact with different input tokens to support complex factual recall. Code is available at https://github.com/Lorenayannnnn/how-lms-answer-one-to-many-factual-queries.
>
---
#### [replaced 084] LookAhead Tuning: Safer Language Models via Partial Answer Previews
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.19041v2](http://arxiv.org/pdf/2503.19041v2)**

> **作者:** Kangwei Liu; Mengru Wang; Yujie Luo; Yuan Lin; Mengshu Sun; Lei Liang; Zhiqiang Zhang; Jun Zhou; Bryan Hooi; Shumin Deng
>
> **备注:** Work in progress
>
> **摘要:** Fine-tuning enables large language models (LLMs) to adapt to specific domains, but often compromises their previously established safety alignment. To mitigate the degradation of model safety during fine-tuning, we introduce LookAhead Tuning, a lightweight and effective data-driven approach that preserves safety during fine-tuning. The method introduces two simple strategies that modify training data by previewing partial answer prefixes, thereby minimizing perturbations to the model's initial token distributions and maintaining its built-in safety mechanisms. Comprehensive experiments demonstrate that LookAhead Tuning effectively maintains model safety without sacrificing robust performance on downstream tasks. Our findings position LookAhead Tuning as a reliable and efficient solution for the safe and effective adaptation of LLMs.
>
---
#### [replaced 085] LightThinker: Thinking Step-by-Step Compression
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2502.15589v2](http://arxiv.org/pdf/2502.15589v2)**

> **作者:** Jintian Zhang; Yuqi Zhu; Mengshu Sun; Yujie Luo; Shuofei Qiao; Lun Du; Da Zheng; Huajun Chen; Ningyu Zhang
>
> **备注:** EMNLP 2025 (oral)
>
> **摘要:** Large language models (LLMs) have shown remarkable performance in complex reasoning tasks, but their efficiency is hindered by the substantial memory and computational costs associated with generating lengthy tokens. In this paper, we propose LightThinker, a novel method that enables LLMs to dynamically compress intermediate thoughts during reasoning. Inspired by human cognitive processes, LightThinker compresses verbose thought steps into compact representations and discards the original reasoning chains, thereby significantly reducing the number of tokens stored in the context window. This is achieved by training the model on when and how to perform compression through data construction, mapping hidden states to condensed gist tokens, and creating specialized attention masks. Additionally, we introduce the Dependency (Dep) metric to quantify the degree of compression by measuring the reliance on historical tokens during generation. Extensive experiments on four datasets and two models show that LightThinker reduces peak memory usage and inference time, while maintaining competitive accuracy. Our work provides a new direction for improving the efficiency of LLMs in complex reasoning tasks without sacrificing performance. Code is released at https://github.com/zjunlp/LightThinker.
>
---
#### [replaced 086] VLDBench Evaluating Multimodal Disinformation with Regulatory Alignment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11361v4](http://arxiv.org/pdf/2502.11361v4)**

> **作者:** Shaina Raza; Ashmal Vayani; Aditya Jain; Aravind Narayanan; Vahid Reza Khazaie; Syed Raza Bashir; Elham Dolatabadi; Gias Uddin; Christos Emmanouilidis; Rizwan Qureshi; Mubarak Shah
>
> **备注:** under review
>
> **摘要:** Detecting disinformation that blends manipulated text and images has become increasingly challenging, as AI tools make synthetic content easy to generate and disseminate. While most existing AI safety benchmarks focus on single modality misinformation (i.e., false content shared without intent to deceive), intentional multimodal disinformation, such as propaganda or conspiracy theories that imitate credible news, remains largely unaddressed. We introduce the Vision-Language Disinformation Detection Benchmark (VLDBench), the first large-scale resource supporting both unimodal (text-only) and multimodal (text + image) disinformation detection. VLDBench comprises approximately 62,000 labeled text-image pairs across 13 categories, curated from 58 news outlets. Using a semi-automated pipeline followed by expert review, 22 domain experts invested over 500 hours to produce high-quality annotations with substantial inter-annotator agreement. Evaluations of state-of-the-art Large Language Models (LLMs) and Vision-Language Models (VLMs) on VLDBench show that incorporating visual cues improves detection accuracy by 5 to 35 percentage points over text-only models. VLDBench provides data and code for evaluation, fine-tuning, and robustness testing to support disinformation analysis. Developed in alignment with AI governance frameworks (e.g., the MIT AI Risk Repository), VLDBench offers a principled foundation for advancing trustworthy disinformation detection in multimodal media. Project: https://vectorinstitute.github.io/VLDBench/ Dataset: https://huggingface.co/datasets/vector-institute/VLDBench Code: https://github.com/VectorInstitute/VLDBench
>
---
