# 自然语言处理 cs.CL

- **最新发布 104 篇**

- **更新 51 篇**

## 最新发布

#### [new 001] OpenReward: Learning to Reward Long-form Agentic Tasks via Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文针对长文本生成任务中奖励模型难以评估外部证据依赖的准确性问题，提出OpenRM，一种通过调用外部工具获取证据来评判响应质量的强化学习奖励模型。通过合成数据与GRPO训练，提升对复杂任务的判别能力，并在推理与训练阶段应用，显著改善大模型对齐效果。**

- **链接: [http://arxiv.org/pdf/2510.24636v1](http://arxiv.org/pdf/2510.24636v1)**

> **作者:** Ziyou Hu; Zhengliang Shi; Minghang Zhu; Haitao Li; Teng Sun; Pengjie Ren; Suzan Verberne; Zhaochun Ren
>
> **摘要:** Reward models (RMs) have become essential for aligning large language models (LLMs), serving as scalable proxies for human evaluation in both training and inference. However, existing RMs struggle on knowledge-intensive and long-form tasks, where evaluating correctness requires grounding beyond the model's internal knowledge. This limitation hinders them from reliably discriminating subtle quality differences, especially when external evidence is necessary. To address this, we introduce OpenRM, a tool-augmented long-form reward model that systematically judges open-ended responses by invoking external tools to gather relevant evidence. We train OpenRM with Group Relative Policy Optimization (GRPO) on over 27K synthesized pairwise examples generated through a controllable data synthesis framework. The training objective jointly supervises intermediate tool usage and final outcome accuracy, incentivizing our reward model to learn effective evidence-based judgment strategies. Extensive experiments on three newly-collected datasets and two widely-used benchmarks demonstrate that OpenRM substantially outperforms existing reward modeling approaches. As a further step, we integrate OpenRM into both inference-time response selection and training-time data selection. This yields consistent gains in downstream LLM alignment tasks, highlighting the potential of tool-augmented reward models for scaling reliable long-form evaluation.
>
---
#### [new 002] Language Models for Longitudinal Clinical Prediction
- **分类: cs.CL**

- **简介: 该论文针对纵向临床预测任务，提出一种轻量级框架，利用冻结的大语言模型整合患者病史与上下文信息，实现无需微调的精准预测。解决了小样本下阿尔茨海默病早期监测的难题，显著提升预测准确性与可靠性。**

- **链接: [http://arxiv.org/pdf/2510.23884v1](http://arxiv.org/pdf/2510.23884v1)**

> **作者:** Tananun Songdechakraiwut; Michael Lutz
>
> **摘要:** We explore a lightweight framework that adapts frozen large language models to analyze longitudinal clinical data. The approach integrates patient history and context within the language model space to generate accurate forecasts without model fine-tuning. Applied to neuropsychological assessments, it achieves accurate and reliable performance even with minimal training data, showing promise for early-stage Alzheimer's monitoring.
>
---
#### [new 003] Beyond Line-Level Filtering for the Pretraining Corpora of LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型预训练语料库中的线级过滤问题，提出两种改进方法：模式感知的线级去重（PLD）和模式感知的句末标点过滤（PTF）。通过考虑文本在文档中的序列分布，保留结构重要内容，提升模型性能。实验表明，该方法在多选题和问答任务上均显著提升效果。**

- **链接: [http://arxiv.org/pdf/2510.24139v1](http://arxiv.org/pdf/2510.24139v1)**

> **作者:** Chanwoo Park; Suyoung Park; Yelim Ahn; Jongmin Kim; Jongyeon Park; Jaejin Lee
>
> **备注:** submitted to ACL ARR Rolling Review
>
> **摘要:** While traditional line-level filtering techniques, such as line-level deduplication and trailing-punctuation filters, are commonly used, these basic methods can sometimes discard valuable content, negatively affecting downstream performance. In this paper, we introduce two methods-pattern-aware line-level deduplication (PLD) and pattern-aware trailing punctuation filtering (PTF)-by enhancing the conventional filtering techniques. Our approach not only considers line-level signals but also takes into account their sequential distribution across documents, enabling us to retain structurally important content that might otherwise be removed. We evaluate these proposed methods by training small language models (1 B parameters) in both English and Korean. The results demonstrate that our methods consistently improve performance on multiple-choice benchmarks and significantly enhance generative question-answering accuracy on both SQuAD v1 and KorQuAD v1.
>
---
#### [new 004] AfriMTEB and AfriE5: Benchmarking and Adapting Text Embedding Models for African Languages
- **分类: cs.CL**

- **简介: 该论文针对非洲语言在文本嵌入领域的缺失问题，提出AfriMTEB基准测试集，涵盖59种非洲语言的14项任务。同时，通过跨语言对比蒸馏改进mE5模型，推出AfriE5，实现对非洲语言的高效文本嵌入，显著优于现有模型。**

- **链接: [http://arxiv.org/pdf/2510.23896v1](http://arxiv.org/pdf/2510.23896v1)**

> **作者:** Kosei Uemura; Miaoran Zhang; David Ifeoluwa Adelani
>
> **摘要:** Text embeddings are an essential building component of several NLP tasks such as retrieval-augmented generation which is crucial for preventing hallucinations in LLMs. Despite the recent release of massively multilingual MTEB (MMTEB), African languages remain underrepresented, with existing tasks often repurposed from translation benchmarks such as FLORES clustering or SIB-200. In this paper, we introduce AfriMTEB -- a regional expansion of MMTEB covering 59 languages, 14 tasks, and 38 datasets, including six newly added datasets. Unlike many MMTEB datasets that include fewer than five languages, the new additions span 14 to 56 African languages and introduce entirely new tasks, such as hate speech detection, intent detection, and emotion classification, which were not previously covered. Complementing this, we present AfriE5, an adaptation of the instruction-tuned mE5 model to African languages through cross-lingual contrastive distillation. Our evaluation shows that AfriE5 achieves state-of-the-art performance, outperforming strong baselines such as Gemini-Embeddings and mE5.
>
---
#### [new 005] ParallelMuse: Agentic Parallel Thinking for Deep Information Seeking
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对信息搜索任务中深度探索的效率与推理整合难题，提出ParallelMuse框架。通过功能分区的局部回放与不确定性引导的路径复用提升探索效率，并利用推理冗余进行无损压缩与答案聚合，显著提升性能并减少计算开销。**

- **链接: [http://arxiv.org/pdf/2510.24698v1](http://arxiv.org/pdf/2510.24698v1)**

> **作者:** Baixuan Li; Dingchu Zhang; Jialong Wu; Wenbiao Yin; Zhengwei Tao; Yida Zhao; Liwen Zhang; Haiyang Shen; Runnan Fang; Pengjun Xie; Jingren Zhou; Yong Jiang
>
> **摘要:** Parallel thinking expands exploration breadth, complementing the deep exploration of information-seeking (IS) agents to further enhance problem-solving capability. However, conventional parallel thinking faces two key challenges in this setting: inefficiency from repeatedly rolling out from scratch, and difficulty in integrating long-horizon reasoning trajectories during answer generation, as limited context capacity prevents full consideration of the reasoning process. To address these issues, we propose ParallelMuse, a two-stage paradigm designed for deep IS agents. The first stage, Functionality-Specified Partial Rollout, partitions generated sequences into functional regions and performs uncertainty-guided path reuse and branching to enhance exploration efficiency. The second stage, Compressed Reasoning Aggregation, exploits reasoning redundancy to losslessly compress information relevant to answer derivation and synthesize a coherent final answer. Experiments across multiple open-source agents and benchmarks demonstrate up to 62% performance improvement with a 10--30% reduction in exploratory token consumption.
>
---
#### [new 006] Breaking the Benchmark: Revealing LLM Bias via Minimal Contextual Augmentation
- **分类: cs.CL; cs.LG**

- **简介: 该论文聚焦于大语言模型（LLM）公平性评估任务，针对现有基准测试中模型偏见被低估的问题，提出一种基于最小上下文增强的通用框架。通过微小输入扰动，揭示了主流LLM在多样群体上的隐蔽偏见，表明模型对弱势群体更易产生刻板行为，强调需扩展研究至更多元化社区。**

- **链接: [http://arxiv.org/pdf/2510.23921v1](http://arxiv.org/pdf/2510.23921v1)**

> **作者:** Kaveh Eskandari Miandoab; Mahammed Kamruzzaman; Arshia Gharooni; Gene Louis Kim; Vasanth Sarathy; Ninareh Mehrabi
>
> **备注:** 9 pages, 3 figures, 3 tables
>
> **摘要:** Large Language Models have been shown to demonstrate stereotypical biases in their representations and behavior due to the discriminative nature of the data that they have been trained on. Despite significant progress in the development of methods and models that refrain from using stereotypical information in their decision-making, recent work has shown that approaches used for bias alignment are brittle. In this work, we introduce a novel and general augmentation framework that involves three plug-and-play steps and is applicable to a number of fairness evaluation benchmarks. Through application of augmentation to a fairness evaluation dataset (Bias Benchmark for Question Answering (BBQ)), we find that Large Language Models (LLMs), including state-of-the-art open and closed weight models, are susceptible to perturbations to their inputs, showcasing a higher likelihood to behave stereotypically. Furthermore, we find that such models are more likely to have biased behavior in cases where the target demographic belongs to a community less studied by the literature, underlining the need to expand the fairness and safety research to include more diverse communities.
>
---
#### [new 007] Reinforcement Learning for Long-Horizon Multi-Turn Search Agents
- **分类: cs.CL**

- **简介: 该论文研究长周期多轮搜索智能体，旨在提升大语言模型在复杂任务中的表现。针对传统提示方法局限，提出基于强化学习的训练方法，使模型通过经验优化决策，在法律文档检索任务中显著超越现有模型，且更长的对话轮次有助于性能提升。**

- **链接: [http://arxiv.org/pdf/2510.24126v1](http://arxiv.org/pdf/2510.24126v1)**

> **作者:** Vivek Kalyan; Martin Andrews
>
> **备注:** 4 pages plus references and appendices. Accepted into the First Workshop on Multi-Turn Interactions in Large Language Models at NeurIPS 2025
>
> **摘要:** Large Language Model (LLM) agents can leverage multiple turns and tools to solve complex tasks, with prompt-based approaches achieving strong performance. This work demonstrates that Reinforcement Learning (RL) can push capabilities significantly further by learning from experience. Through experiments on a legal document search benchmark, we show that our RL-trained 14 Billion parameter model outperforms frontier class models (85% vs 78% accuracy). In addition, we explore turn-restricted regimes, during training and at test-time, that show these agents achieve better results if allowed to operate over longer multi-turn horizons.
>
---
#### [new 008] Ko-MuSR: A Multistep Soft Reasoning Benchmark for LLMs Capable of Understanding Korean
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Ko-MuSR，首个面向韩语长篇叙事的多步软推理基准，旨在评估大模型在韩语环境下的复杂推理能力。解决韩语长文本推理评估缺失问题，通过构建高质量韩语数据集与提示策略，验证了跨语言推理泛化性，并推动韩语NLP发展。**

- **链接: [http://arxiv.org/pdf/2510.24150v1](http://arxiv.org/pdf/2510.24150v1)**

> **作者:** Chanwoo Park; Suyoung Park; JiA Kang; Jongyeon Park; Sangho Kim; Hyunji M. Park; Sumin Bae; Mingyu Kang; Jaejin Lee
>
> **备注:** submitted to ACL ARR Rolling Review
>
> **摘要:** We present Ko-MuSR, the first benchmark to comprehensively evaluate multistep, soft reasoning in long Korean narratives while minimizing data contamination. Built following MuSR, Ko-MuSR features fully Korean narratives, reasoning chains, and multiple-choice questions verified by human annotators for logical consistency and answerability. Evaluations of four large language models -- two multilingual and two Korean-specialized -- show that multilingual models outperform Korean-focused ones even in Korean reasoning tasks, indicating cross-lingual generalization of reasoning ability. Carefully designed prompting strategies, which combine few-shot examples, reasoning traces, and task-specific hints, further boost accuracy, approaching human-level performance. Ko-MuSR offers a solid foundation for advancing Korean NLP by enabling systematic evaluation of long-context reasoning and prompting strategies.
>
---
#### [new 009] Leveraging LLMs for Early Alzheimer's Prediction
- **分类: cs.CL**

- **简介: 该论文提出一种基于预训练语言模型（LLM）的阿尔茨海默病早期预测方法，将动态fMRI脑连接数据转为时间序列，通过归一化与特征映射，输入冻结的LLM进行临床预测。任务为早期阿尔茨海默病识别，解决了传统方法对复杂脑动态模式建模不足的问题，显著提升预测敏感性与准确性。**

- **链接: [http://arxiv.org/pdf/2510.23946v1](http://arxiv.org/pdf/2510.23946v1)**

> **作者:** Tananun Songdechakraiwut
>
> **摘要:** We present a connectome-informed LLM framework that encodes dynamic fMRI connectivity as temporal sequences, applies robust normalization, and maps these data into a representation suitable for a frozen pre-trained LLM for clinical prediction. Applied to early Alzheimer's detection, our method achieves sensitive prediction with error rates well below clinically recognized margins, with implications for timely Alzheimer's intervention.
>
---
#### [new 010] Quantifying the Effects of Word Length, Frequency, and Predictability on Dyslexia
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文研究阅读障碍者在自然阅读中因词长、频率和可预测性导致的阅读时间成本。通过眼动追踪分析，发现所有因素均影响阅读时间，且阅读障碍者对可预测性更敏感。干预性模拟显示，调整这些因素可缩小与正常读者的差距约三分之一，支持语言工作记忆和语音编码理论，为干预和模型提供依据。**

- **链接: [http://arxiv.org/pdf/2510.24647v1](http://arxiv.org/pdf/2510.24647v1)**

> **作者:** Hugo Rydel-Johnston; Alex Kafkas
>
> **摘要:** We ask where, and under what conditions, dyslexic reading costs arise in a large-scale naturalistic reading dataset. Using eye-tracking aligned to word-level features (word length, frequency, and predictability), we model how each feature influences dyslexic time costs. We find that all three features robustly change reading times in both typical and dyslexic readers, and that dyslexic readers show stronger sensitivities to each, especially predictability. Counterfactual manipulations of these features substantially narrow the dyslexic-control gap by about one third, with predictability showing the strongest effect, followed by length and frequency. These patterns align with dyslexia theories that posit heightened demands on linguistic working memory and phonological encoding, and they motivate further work on lexical complexity and parafoveal preview benefits to explain the remaining gap. In short, we quantify when extra dyslexic costs arise, how large they are, and offer actionable guidance for interventions and computational models for dyslexics.
>
---
#### [new 011] BitSkip: An Empirical Analysis of Quantization and Early Exit Composition
- **分类: cs.CL; 68T05; I.2.6; I.2.7**

- **简介: 该论文研究高效大语言模型的量化与早退出组合策略。针对量化与动态路由协同效应不明确的问题，提出BitSkip框架。发现8位量化无哈达玛变换模型性能最优，且具优异早退出能力，实现32.5%加速仅损失4%质量。**

- **链接: [http://arxiv.org/pdf/2510.23766v1](http://arxiv.org/pdf/2510.23766v1)**

> **作者:** Ramshankar Bhuvaneswaran; Handan Liu
>
> **备注:** Submitted to JMLR
>
> **摘要:** The pursuit of efficient Large Language Models (LLMs) has led to increasingly complex techniques like extreme quantization and dynamic routing. While individual benefits of these methods are well-documented, their compositional effects remain poorly understood. This paper introduces BitSkip, a hybrid architectural framework for systematically explor- ing these interactions. Counter-intuitively, our findings reveal that a simple 8-bit quantized model without Hadamard transform (BitSkip-V1) not only outperforms its more complex 4-bit and Hadamard-enhanced counterparts but also competes the full-precision baseline in quality (perplexity of 1.13 vs 1.19) . The introduction of Hadamard transforms, even at 8- bit precision, catastrophically degraded performance by over 37,000%, tracing fundamental training instability. Our BitSkip-V1 recipe demonstrates superior early-exit characteristics, with layer 18 providing optimal 32.5% speed gain for minimal 4% quality loss.
>
---
#### [new 012] Can LLMs Translate Human Instructions into a Reinforcement Learning Agent's Internal Emergent Symbolic Representation?
- **分类: cs.CL; cs.RO**

- **简介: 该论文研究大语言模型（LLMs）将人类自然语言指令翻译为强化学习智能体内部涌现符号表征的能力。针对符号表征与语言之间的对齐问题，作者在蚁迷宫和蚁坠落环境中评估GPT、Claude等模型的翻译性能，发现其效果受划分粒度和任务复杂度影响显著，揭示了当前LLMs在表示对齐上的局限性。**

- **链接: [http://arxiv.org/pdf/2510.24259v1](http://arxiv.org/pdf/2510.24259v1)**

> **作者:** Ziqi Ma; Sao Mai Nguyen; Philippe Xu
>
> **摘要:** Emergent symbolic representations are critical for enabling developmental learning agents to plan and generalize across tasks. In this work, we investigate whether large language models (LLMs) can translate human natural language instructions into the internal symbolic representations that emerge during hierarchical reinforcement learning. We apply a structured evaluation framework to measure the translation performance of commonly seen LLMs -- GPT, Claude, Deepseek and Grok -- across different internal symbolic partitions generated by a hierarchical reinforcement learning algorithm in the Ant Maze and Ant Fall environments. Our findings reveal that although LLMs demonstrate some ability to translate natural language into a symbolic representation of the environment dynamics, their performance is highly sensitive to partition granularity and task complexity. The results expose limitations in current LLMs capacity for representation alignment, highlighting the need for further research on robust alignment between language and internal agent representations.
>
---
#### [new 013] SpecKD: Speculative Decoding for Effective Knowledge Distillation of LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型知识蒸馏中因统一应用损失导致噪声干扰的问题，提出SpecKD框架。通过引入类推测解码的动态门控机制，仅对教师自信的token进行蒸馏，提升学生模型性能与训练稳定性，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24021v1](http://arxiv.org/pdf/2510.24021v1)**

> **作者:** Haiduo Huang; Jiangcheng Song; Yadong Zhang; Pengju Ren
>
> **摘要:** Knowledge Distillation (KD) has become a cornerstone technique for compressing Large Language Models (LLMs) into smaller, more efficient student models. However, conventional KD approaches typically apply the distillation loss uniformly across all tokens, regardless of the teacher's confidence. This indiscriminate mimicry can introduce noise, as the student is forced to learn from the teacher's uncertain or high-entropy predictions, which may ultimately harm student performance-especially when the teacher is much larger and more powerful. To address this, we propose Speculative Knowledge Distillation (SpecKD), a novel, plug-and-play framework that introduces a dynamic, token-level gating mechanism inspired by the "propose-and-verify" paradigm of speculative decoding. At each step, the student's token proposal is verified against the teacher's distribution; the distillation loss is selectively applied only to "accepted" tokens, while "rejected" tokens are masked out. Extensive experiments on diverse text generation tasks show that SpecKD consistently and significantly outperforms strong KD baselines, leading to more stable training and more capable student models, and achieving state-of-the-art results.
>
---
#### [new 014] SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models
- **分类: cs.CL**

- **简介: 该论文提出SynthWorlds框架，用于分离语言模型的推理与知识能力。针对现有评估中事实记忆干扰推理判断的问题，构建真实与合成映射的平行世界，设计镜像任务，实现对推理能力的可控评估，揭示知识优势差距，推动模型改进。**

- **链接: [http://arxiv.org/pdf/2510.24427v1](http://arxiv.org/pdf/2510.24427v1)**

> **作者:** Ken Gu; Advait Bhat; Mike A Merrill; Robert West; Xin Liu; Daniel McDuff; Tim Althoff
>
> **摘要:** Evaluating the reasoning ability of language models (LMs) is complicated by their extensive parametric world knowledge, where benchmark performance often reflects factual recall rather than genuine reasoning. Existing datasets and approaches (e.g., temporal filtering, paraphrasing, adversarial substitution) cannot cleanly separate the two. We present SynthWorlds, a framework that disentangles task reasoning complexity from factual knowledge. In SynthWorlds, we construct parallel corpora representing two worlds with identical interconnected structure: a real-mapped world, where models may exploit parametric knowledge, and a synthetic-mapped world, where such knowledge is meaningless. On top of these corpora, we design two mirrored tasks as case studies: multi-hop question answering and page navigation, which maintain equal reasoning difficulty across worlds. Experiments in parametric-only (e.g., closed-book QA) and knowledge-augmented (e.g., retrieval-augmented) LM settings reveal a persistent knowledge advantage gap, defined as the performance boost models gain from memorized parametric world knowledge. Knowledge acquisition and integration mechanisms reduce but do not eliminate this gap, highlighting opportunities for system improvements. Fully automatic and scalable, SynthWorlds provides a controlled environment for evaluating LMs in ways that were previously challenging, enabling precise and testable comparisons of reasoning and memorization.
>
---
#### [new 015] Squrve: A Unified and Modular Framework for Complex Real-World Text-to-SQL Tasks
- **分类: cs.CL**

- **简介: 该论文提出Squrve框架，旨在解决复杂真实场景下文本到SQL转换的集成难题。通过统一执行范式与七种原子协作组件，实现多方法协同，显著提升复杂查询处理能力，推动研究向实际应用落地。**

- **链接: [http://arxiv.org/pdf/2510.24102v1](http://arxiv.org/pdf/2510.24102v1)**

> **作者:** Yihan Wang; Peiyu Liu; Runyu Chen; Jiaxing Pu; Wei Xu
>
> **摘要:** Text-to-SQL technology has evolved rapidly, with diverse academic methods achieving impressive results. However, deploying these techniques in real-world systems remains challenging due to limited integration tools. Despite these advances, we introduce Squrve, a unified, modular, and extensive Text-to-SQL framework designed to bring together research advances and real-world applications. Squrve first establishes a universal execution paradigm that standardizes invocation interfaces, then proposes a multi-actor collaboration mechanism based on seven abstracted effective atomic actor components. Experiments on widely adopted benchmarks demonstrate that the collaborative workflows consistently outperform the original individual methods, thereby opening up a new effective avenue for tackling complex real-world queries. The codes are available at https://github.com/Satissss/Squrve.
>
---
#### [new 016] Repurposing Synthetic Data for Fine-grained Search Agent Supervision
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型搜索代理训练中因忽略实体信息导致的“近似正确”样本被误判问题，提出E-GRPO框架。通过引入基于实体匹配的密集奖励机制，使模型能从推理正确但答案错误的样本中学习，显著提升准确率与推理效率。**

- **链接: [http://arxiv.org/pdf/2510.24694v1](http://arxiv.org/pdf/2510.24694v1)**

> **作者:** Yida Zhao; Kuan Li; Xixi Wu; Liwen Zhang; Dingchu Zhang; Baixuan Li; Maojia Song; Zhuo Chen; Chenxi Wang; Xinyu Wang; Kewei Tu; Pengjun Xie; Jingren Zhou; Yong Jiang
>
> **摘要:** LLM-based search agents are increasingly trained on entity-centric synthetic data to solve complex, knowledge-intensive tasks. However, prevailing training methods like Group Relative Policy Optimization (GRPO) discard this rich entity information, relying instead on sparse, outcome-based rewards. This critical limitation renders them unable to distinguish informative "near-miss" samples-those with substantially correct reasoning but a flawed final answer-from complete failures, thus discarding valuable learning signals. We address this by leveraging the very entities discarded during training. Our empirical analysis reveals a strong positive correlation between the number of ground-truth entities identified during an agent's reasoning process and final answer accuracy. Building on this insight, we introduce Entity-aware Group Relative Policy Optimization (E-GRPO), a novel framework that formulates a dense entity-aware reward function. E-GRPO assigns partial rewards to incorrect samples proportional to their entity match rate, enabling the model to effectively learn from these "near-misses". Experiments on diverse question-answering (QA) and deep research benchmarks show that E-GRPO consistently and significantly outperforms the GRPO baseline. Furthermore, our analysis reveals that E-GRPO not only achieves superior accuracy but also induces more efficient reasoning policies that require fewer tool calls, demonstrating a more effective and sample-efficient approach to aligning search agents.
>
---
#### [new 017] Talk2Ref: A Dataset for Reference Prediction from Scientific Talks
- **分类: cs.CL**

- **简介: 该论文提出参考文献预测任务（RPT），旨在从科学演讲中自动识别相关文献。为支持该任务，构建了首个大规模数据集Talk2Ref，包含6,279个演讲及43,429篇引用文献。研究设计了双编码器模型并探索长文本处理与领域适应策略，实验表明微调可显著提升预测性能。**

- **链接: [http://arxiv.org/pdf/2510.24478v1](http://arxiv.org/pdf/2510.24478v1)**

> **作者:** Frederik Broy; Maike Züfle; Jan Niehues
>
> **摘要:** Scientific talks are a growing medium for disseminating research, and automatically identifying relevant literature that grounds or enriches a talk would be highly valuable for researchers and students alike. We introduce Reference Prediction from Talks (RPT), a new task that maps long, and unstructured scientific presentations to relevant papers. To support research on RPT, we present Talk2Ref, the first large-scale dataset of its kind, containing 6,279 talks and 43,429 cited papers (26 per talk on average), where relevance is approximated by the papers cited in the talk's corresponding source publication. We establish strong baselines by evaluating state-of-the-art text embedding models in zero-shot retrieval scenarios, and propose a dual-encoder architecture trained on Talk2Ref. We further explore strategies for handling long transcripts, as well as training for domain adaptation. Our results show that fine-tuning on Talk2Ref significantly improves citation prediction performance, demonstrating both the challenges of the task and the effectiveness of our dataset for learning semantic representations from spoken scientific content. The dataset and trained models are released under an open license to foster future research on integrating spoken scientific communication into citation recommendation systems.
>
---
#### [new 018] From Memorization to Reasoning in the Spectrum of Loss Curvature
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究神经网络中的记忆现象，针对模型过度记忆训练数据的问题，提出基于损失曲率的权重分解方法，可有效抑制非目标记忆内容。通过分析低曲率权重对任务性能的影响，发现事实检索与算术任务依赖特定权重方向，验证了记忆机制的专一性，为去记忆提供新思路。**

- **链接: [http://arxiv.org/pdf/2510.24256v1](http://arxiv.org/pdf/2510.24256v1)**

> **作者:** Jack Merullo; Srihita Vatsavaya; Lucius Bushnaq; Owen Lewis
>
> **摘要:** We characterize how memorization is represented in transformer models and show that it can be disentangled in the weights of both language models (LMs) and vision transformers (ViTs) using a decomposition based on the loss landscape curvature. This insight is based on prior theoretical and empirical work showing that the curvature for memorized training points is much sharper than non memorized, meaning ordering weight components from high to low curvature can reveal a distinction without explicit labels. This motivates a weight editing procedure that suppresses far more recitation of untargeted memorized data more effectively than a recent unlearning method (BalancedSubnet), while maintaining lower perplexity. Since the basis of curvature has a natural interpretation for shared structure in model weights, we analyze the editing procedure extensively on its effect on downstream tasks in LMs, and find that fact retrieval and arithmetic are specifically and consistently negatively affected, even though open book fact retrieval and general logical reasoning is conserved. We posit these tasks rely heavily on specialized directions in weight space rather than general purpose mechanisms, regardless of whether those individual datapoints are memorized. We support this by showing a correspondence between task data's activation strength with low curvature components that we edit out, and the drop in task performance after the edit. Our work enhances the understanding of memorization in neural networks with practical applications towards removing it, and provides evidence for idiosyncratic, narrowly-used structures involved in solving tasks like math and fact retrieval.
>
---
#### [new 019] Can LLMs Narrate Tabular Data? An Evaluation Framework for Natural Language Representations of Text-to-SQL System Outputs
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦文本到SQL系统输出的自然语言表示（NLR）任务，针对大语言模型生成NLR时的信息丢失与错误问题，提出Combo-Eval评估框架与NLR-BIRD数据集，显著提升评估精度并减少61%的LLM调用。**

- **链接: [http://arxiv.org/pdf/2510.23854v1](http://arxiv.org/pdf/2510.23854v1)**

> **作者:** Jyotika Singh; Weiyi Sun; Amit Agarwal; Viji Krishnamurthy; Yassine Benajiba; Sujith Ravi; Dan Roth
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** In modern industry systems like multi-turn chat agents, Text-to-SQL technology bridges natural language (NL) questions and database (DB) querying. The conversion of tabular DB results into NL representations (NLRs) enables the chat-based interaction. Currently, NLR generation is typically handled by large language models (LLMs), but information loss or errors in presenting tabular results in NL remains largely unexplored. This paper introduces a novel evaluation method - Combo-Eval - for judgment of LLM-generated NLRs that combines the benefits of multiple existing methods, optimizing evaluation fidelity and achieving a significant reduction in LLM calls by 25-61%. Accompanying our method is NLR-BIRD, the first dedicated dataset for NLR benchmarking. Through human evaluations, we demonstrate the superior alignment of Combo-Eval with human judgments, applicable across scenarios with and without ground truth references.
>
---
#### [new 020] PICOs-RAG: PICO-supported Query Rewriting for Retrieval-Augmented Generation in Evidence-Based Medicine
- **分类: cs.CL**

- **简介: 该论文针对临床医学中复杂查询导致检索不准确的问题，提出PICOs-RAG方法。通过将用户查询转换为PICO格式，提升检索相关性与效率，显著改善大模型在循证医学中的表现。**

- **链接: [http://arxiv.org/pdf/2510.23998v1](http://arxiv.org/pdf/2510.23998v1)**

> **作者:** Mengzhou Sun; Sendong Zhao; Jianyu Chen; Bin Qin
>
> **摘要:** Evidence-based medicine (EBM) research has always been of paramount importance. It is important to find appropriate medical theoretical support for the needs from physicians or patients to reduce the occurrence of medical accidents. This process is often carried out by human querying relevant literature databases, which lacks objectivity and efficiency. Therefore, researchers utilize retrieval-augmented generation (RAG) to search for evidence and generate responses automatically. However, current RAG methods struggle to handle complex queries in real-world clinical scenarios. For example, when queries lack certain information or use imprecise language, the model may retrieve irrelevant evidence and generate unhelpful answers. To address this issue, we present the PICOs-RAG to expand the user queries into a better format. Our method can expand and normalize the queries into professional ones and use the PICO format, a search strategy tool present in EBM, to extract the most important information used for retrieval. This approach significantly enhances retrieval efficiency and relevance, resulting in up to an 8.8\% improvement compared to the baseline evaluated by our method. Thereby the PICOs-RAG improves the performance of the large language models into a helpful and reliable medical assistant in EBM.
>
---
#### [new 021] Challenging Multilingual LLMs: A New Taxonomy and Benchmark for Unraveling Hallucination in Translation
- **分类: cs.CL**

- **简介: 该论文针对多语言大模型翻译中的幻觉问题，提出新分类体系与诊断框架，构建了多语言、人工验证的基准HalloMTBench。通过前沿模型生成与多轮评估，筛选出5,435个高质量实例，揭示了模型规模、源文长度、语言偏见及强化学习导致的语言混杂等幻觉触发机制。**

- **链接: [http://arxiv.org/pdf/2510.24073v1](http://arxiv.org/pdf/2510.24073v1)**

> **作者:** Xinwei Wu; Heng Liu; Jiang Zhou; Xiaohu Zhao; Linlong Xu; Longyue Wang; Weihua Luo; Kaifu Zhang
>
> **摘要:** Large Language Models (LLMs) have advanced machine translation but remain vulnerable to hallucinations. Unfortunately, existing MT benchmarks are not capable of exposing failures in multilingual LLMs. To disclose hallucination in multilingual LLMs, we introduce a diagnostic framework with a taxonomy that separates Instruction Detachment from Source Detachment. Guided by this taxonomy, we create HalloMTBench, a multilingual, human-verified benchmark across 11 English-to-X directions. We employed 4 frontier LLMs to generate candidates and scrutinize these candidates with an ensemble of LLM judges, and expert validation. In this way, we curate 5,435 high-quality instances. We have evaluated 17 LLMs on HalloMTBench. Results reveal distinct ``hallucination triggers'' -- unique failure patterns reflecting model scale, source length sensitivity, linguistic biases, and Reinforcement-Learning (RL) amplified language mixing. HalloMTBench offers a forward-looking testbed for diagnosing LLM translation failures. HalloMTBench is available in https://huggingface.co/collections/AIDC-AI/marco-mt.
>
---
#### [new 022] Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型智能体微调中数据分散、格式异构的问题，提出轻量级统一协议ADP，实现多源异构数据的标准化表示。通过将13个数据集转换为ADP格式，支持多种代理框架训练，显著提升性能，推动可复现、可扩展的智能体训练。**

- **链接: [http://arxiv.org/pdf/2510.24702v1](http://arxiv.org/pdf/2510.24702v1)**

> **作者:** Yueqi Song; Ketan Ramaneti; Zaid Sheikh; Ziru Chen; Boyu Gou; Tianbao Xie; Yiheng Xu; Danyang Zhang; Apurva Gandhi; Fan Yang; Joseph Liu; Tianyue Ou; Zhihao Yuan; Frank Xu; Shuyan Zhou; Xingyao Wang; Xiang Yue; Tao Yu; Huan Sun; Yu Su; Graham Neubig
>
> **摘要:** Public research results on large-scale supervised finetuning of AI agents remain relatively rare, since the collection of agent training data presents unique challenges. In this work, we argue that the bottleneck is not a lack of underlying data sources, but that a large variety of data is fragmented across heterogeneous formats, tools, and interfaces. To this end, we introduce the agent data protocol (ADP), a light-weight representation language that serves as an "interlingua" between agent datasets in diverse formats and unified agent training pipelines downstream. The design of ADP is expressive enough to capture a large variety of tasks, including API/tool use, browsing, coding, software engineering, and general agentic workflows, while remaining simple to parse and train on without engineering at a per-dataset level. In experiments, we unified a broad collection of 13 existing agent training datasets into ADP format, and converted the standardized ADP data into training-ready formats for multiple agent frameworks. We performed SFT on these data, and demonstrated an average performance gain of ~20% over corresponding base models, and delivers state-of-the-art or near-SOTA performance on standard coding, browsing, tool use, and research benchmarks, without domain-specific tuning. All code and data are released publicly, in the hope that ADP could help lower the barrier to standardized, scalable, and reproducible agent training.
>
---
#### [new 023] Auto prompting without training labels: An LLM cascade for product quality assessment in e-commerce catalogs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种无需训练标签的LLM级联自动提示框架，用于电商商品质量评估。针对领域知识缺失与标注成本高的问题，通过自动生成并优化提示，显著提升评估精度与效率，实现99%专家工作量降低，并跨语言、多任务保持高性能。**

- **链接: [http://arxiv.org/pdf/2510.23941v1](http://arxiv.org/pdf/2510.23941v1)**

> **作者:** Soham Satyadharma; Fatemeh Sheikholeslami; Swati Kaul; Aziz Umit Batur; Suleiman A. Khan
>
> **摘要:** We introduce a novel, training free cascade for auto-prompting Large Language Models (LLMs) to assess product quality in e-commerce. Our system requires no training labels or model fine-tuning, instead automatically generating and refining prompts for evaluating attribute quality across tens of thousands of product category-attribute pairs. Starting from a seed of human-crafted prompts, the cascade progressively optimizes instructions to meet catalog-specific requirements. This approach bridges the gap between general language understanding and domain-specific knowledge at scale in complex industrial catalogs. Our extensive empirical evaluations shows the auto-prompt cascade improves precision and recall by $8-10\%$ over traditional chain-of-thought prompting. Notably, it achieves these gains while reducing domain expert effort from 5.1 hours to 3 minutes per attribute - a $99\%$ reduction. Additionally, the cascade generalizes effectively across five languages and multiple quality assessment tasks, consistently maintaining performance gains.
>
---
#### [new 024] "Mm, Wat?" Detecting Other-initiated Repair Requests in Dialogue
- **分类: cs.CL**

- **简介: 该论文聚焦于对话中“他人发起的修复请求”（OIR）检测任务，旨在提升对话系统对用户理解困难信号的识别能力。通过融合语言与韵律特征，构建多模态模型，显著提升检测效果，为改善人机对话的连贯性提供支持。**

- **链接: [http://arxiv.org/pdf/2510.24628v1](http://arxiv.org/pdf/2510.24628v1)**

> **作者:** Anh Ngo; Nicolas Rollet; Catherine Pelachaud; Chloe Clavel
>
> **备注:** 9 pages
>
> **摘要:** Maintaining mutual understanding is a key component in human-human conversation to avoid conversation breakdowns, in which repair, particularly Other-Initiated Repair (OIR, when one speaker signals trouble and prompts the other to resolve), plays a vital role. However, Conversational Agents (CAs) still fail to recognize user repair initiation, leading to breakdowns or disengagement. This work proposes a multimodal model to automatically detect repair initiation in Dutch dialogues by integrating linguistic and prosodic features grounded in Conversation Analysis. The results show that prosodic cues complement linguistic features and significantly improve the results of pretrained text and audio embeddings, offering insights into how different features interact. Future directions include incorporating visual cues, exploring multilingual and cross-context corpora to assess the robustness and generalizability.
>
---
#### [new 025] ReForm: Reflective Autoformalization with Prospective Bounded Sequence Optimization
- **分类: cs.CL**

- **简介: 该论文针对自然语言数学问题的自动形式化任务，解决大模型生成形式化语句时语义失真的问题。提出ReForm方法，通过引入反思机制与前景有界序列优化（PBSO），实现语义一致性评估与迭代修正，显著提升形式化准确率。**

- **链接: [http://arxiv.org/pdf/2510.24592v1](http://arxiv.org/pdf/2510.24592v1)**

> **作者:** Guoxin Chen; Jing Wu; Xinjie Chen; Wayne Xin Zhao; Ruihua Song; Chengxi Li; Kai Fan; Dayiheng Liu; Minpeng Liao
>
> **备注:** Ongoing Work
>
> **摘要:** Autoformalization, which translates natural language mathematics into machine-verifiable formal statements, is critical for using formal mathematical reasoning to solve math problems stated in natural language. While Large Language Models can generate syntactically correct formal statements, they often fail to preserve the original problem's semantic intent. This limitation arises from the LLM approaches' treating autoformalization as a simplistic translation task which lacks mechanisms for self-reflection and iterative refinement that human experts naturally employ. To address these issues, we propose ReForm, a Reflective Autoformalization method that tightly integrates semantic consistency evaluation into the autoformalization process. This enables the model to iteratively generate formal statements, assess its semantic fidelity, and self-correct identified errors through progressive refinement. To effectively train this reflective model, we introduce Prospective Bounded Sequence Optimization (PBSO), which employs different rewards at different sequence positions to ensure that the model develops both accurate autoformalization and correct semantic validations, preventing superficial critiques that would undermine the purpose of reflection. Extensive experiments across four autoformalization benchmarks demonstrate that ReForm achieves an average improvement of 17.2 percentage points over the strongest baselines. To further ensure evaluation reliability, we introduce ConsistencyCheck, a benchmark of 859 expert-annotated items that not only validates LLMs as judges but also reveals that autoformalization is inherently difficult: even human experts produce semantic errors in up to 38.5% of cases.
>
---
#### [new 026] BEST-RQ-Based Self-Supervised Learning for Whisper Domain Adaptation
- **分类: cs.CL**

- **简介: 该论文针对语音识别中低资源、跨领域场景下标注数据稀缺的问题，提出BEARD框架，通过自监督学习结合BEST-RQ目标与知识蒸馏，实现Whisper编码器的域适应。在航空管制语音数据上，仅用5000小时无标签语音显著提升识别性能，相对基线提升12%。**

- **链接: [http://arxiv.org/pdf/2510.24570v1](http://arxiv.org/pdf/2510.24570v1)**

> **作者:** Raphaël Bagat; Irina Illina; Emmanuel Vincent
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Automatic Speech Recognition (ASR) systems, despite large multilingual training, struggle in out-of-domain and low-resource scenarios where labeled data is scarce. We propose BEARD (BEST-RQ Encoder Adaptation with Re-training and Distillation), a novel framework designed to adapt Whisper's encoder using unlabeled data. Unlike traditional self-supervised learning methods, BEARD uniquely combines a BEST-RQ objective with knowledge distillation from a frozen teacher encoder, ensuring the encoder's complementarity with the pre-trained decoder. Our experiments focus on the ATCO2 corpus from the challenging Air Traffic Control (ATC) communications domain, characterized by non-native speech, noise, and specialized phraseology. Using about 5,000 hours of untranscribed speech for BEARD and 2 hours of transcribed speech for fine-tuning, the proposed approach significantly outperforms previous baseline and fine-tuned model, achieving a relative improvement of 12% compared to the fine-tuned model. To the best of our knowledge, this is the first work to use a self-supervised learning objective for domain adaptation of Whisper.
>
---
#### [new 027] Charting the European LLM Benchmarking Landscape: A New Taxonomy and a Set of Best Practices
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦多语言LLM评估任务，针对欧洲语言在基准测试中研究不足的问题，提出面向非英语场景的新分类体系与最佳实践，强调提升评估方法的语言文化敏感性，推动欧洲语言LLM评测的标准化与协同进展。**

- **链接: [http://arxiv.org/pdf/2510.24450v1](http://arxiv.org/pdf/2510.24450v1)**

> **作者:** Špela Vintar; Taja Kuzman Pungeršek; Mojca Brglez; Nikola Ljubešić
>
> **备注:** 12 pages, 1 figure. Submitted to the LREC 2026 conference
>
> **摘要:** While new benchmarks for large language models (LLMs) are being developed continuously to catch up with the growing capabilities of new models and AI in general, using and evaluating LLMs in non-English languages remains a little-charted landscape. We give a concise overview of recent developments in LLM benchmarking, and then propose a new taxonomy for the categorization of benchmarks that is tailored to multilingual or non-English use scenarios. We further propose a set of best practices and quality standards that could lead to a more coordinated development of benchmarks for European languages. Among other recommendations, we advocate for a higher language and culture sensitivity of evaluation methods.
>
---
#### [new 028] Tongyi DeepResearch Technical Report
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; cs.MA**

- **简介: 该论文提出Tongyi DeepResearch，一种用于长周期深度信息检索任务的智能体大模型。针对复杂研究任务中自主推理与信息获取难题，通过端到端训练框架与自动化数据合成，实现高效、稳定的深度研究能力。模型在多个基准上达SOTA，并开源全部资源。**

- **链接: [http://arxiv.org/pdf/2510.24701v1](http://arxiv.org/pdf/2510.24701v1)**

> **作者:** Tongyi DeepResearch Team; Baixuan Li; Bo Zhang; Dingchu Zhang; Fei Huang; Guangyu Li; Guoxin Chen; Huifeng Yin; Jialong Wu; Jingren Zhou; Kuan Li; Liangcai Su; Litu Ou; Liwen Zhang; Pengjun Xie; Rui Ye; Wenbiao Yin; Xinmiao Yu; Xinyu Wang; Xixi Wu; Xuanzhong Chen; Yida Zhao; Zhen Zhang; Zhengwei Tao; Zhongwang Zhang; Zile Qiao; Chenxi Wang; Donglei Yu; Gang Fu; Haiyang Shen; Jiayin Yang; Jun Lin; Junkai Zhang; Kui Zeng; Li Yang; Hailong Yin; Maojia Song; Ming Yan; Peng Xia; Qian Xiao; Rui Min; Ruixue Ding; Runnan Fang; Shaowei Chen; Shen Huang; Shihang Wang; Shihao Cai; Weizhou Shen; Xiaobin Wang; Xin Guan; Xinyu Geng; Yingcheng Shi; Yuning Wu; Zhuo Chen; Zijian Li; Yong Jiang
>
> **备注:** https://tongyi-agent.github.io/blog
>
> **摘要:** We present Tongyi DeepResearch, an agentic large language model, which is specifically designed for long-horizon, deep information-seeking research tasks. To incentivize autonomous deep research agency, Tongyi DeepResearch is developed through an end-to-end training framework that combines agentic mid-training and agentic post-training, enabling scalable reasoning and information seeking across complex tasks. We design a highly scalable data synthesis pipeline that is fully automatic, without relying on costly human annotation, and empowers all training stages. By constructing customized environments for each stage, our system enables stable and consistent interactions throughout. Tongyi DeepResearch, featuring 30.5 billion total parameters, with only 3.3 billion activated per token, achieves state-of-the-art performance across a range of agentic deep research benchmarks, including Humanity's Last Exam, BrowseComp, BrowseComp-ZH, WebWalkerQA, xbench-DeepSearch, FRAMES and xbench-DeepSearch-2510. We open-source the model, framework, and complete solutions to empower the community.
>
---
#### [new 029] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs
- **分类: cs.CL**

- **简介: 该论文针对长文本建模中注意力机制计算成本高的问题，提出动态分层稀疏注意力（DHSA）框架。通过在线动态分割序列、长度归一化聚合与重要性评分，实现高效低内存的长上下文推理，在保持精度的同时显著降低延迟与内存占用。**

- **链接: [http://arxiv.org/pdf/2510.24606v1](http://arxiv.org/pdf/2510.24606v1)**

> **作者:** Siheng Xiong; Joe Zou; Faramarz Fekri; Yae Jee Cho
>
> **备注:** Accepted to NeurIPS 2025 Workshop on Efficient Reasoning
>
> **摘要:** The quadratic cost of attention hinders the scalability of long-context LLMs, especially in resource-constrained settings. Existing static sparse methods such as sliding windows or global tokens utilizes the sparsity of attention to reduce the cost of attention, but poorly adapts to the content-dependent variations in attention due to their staticity. While previous work has proposed several dynamic approaches to improve flexibility, they still depend on predefined templates or heuristic mechanisms. Such strategies reduce generality and prune tokens that remain contextually important, limiting their accuracy across diverse tasks. To tackle these bottlenecks of existing methods for long-context modeling, we introduce Dynamic Hierarchical Sparse Attention (DHSA), a data-driven framework that dynamically predicts attention sparsity online without retraining. Our proposed DHSA adaptively segments sequences into variable-length chunks, then computes chunk representations by aggregating the token embeddings within each chunk. To avoid the bias introduced by varying chunk lengths, we apply length-normalized aggregation that scales the averaged embeddings by the square root of the chunk size. Finally, DHSA upsamples the chunk-level similarity scores to token level similarities to calculate importance scores that determine which token-level interactions should be preserved. Our experiments on Gemma2 with Needle-in-a-Haystack Test and LongBench show that DHSA matches dense attention in accuracy, while reducing prefill latency by 20-60% and peak memory usage by 35%. Compared to other representative baselines such as block sparse attention, DHSA achieves consistently higher accuracy (6-18% relative gains) with comparable or lower cost, offering an efficient and adaptable solution for long-context on-device LLMs.
>
---
#### [new 030] Levée d'ambiguïtés par grammaires locales
- **分类: cs.CL**

- **简介: 该论文针对自然语言处理中的词性标注歧义问题，提出一种适用于零漏检率目标的局部语法消歧方法。研究聚焦于INTEX系统，强调在多解情况下需验证语法交互效应，指出孤立分析规则不可靠，需详尽规范语法行为以确保正确标签不被遗漏。**

- **链接: [http://arxiv.org/pdf/2510.24530v1](http://arxiv.org/pdf/2510.24530v1)**

> **作者:** Eric G. C. Laporte
>
> **备注:** in French language
>
> **摘要:** Many words are ambiguous in terms of their part of speech (POS). However, when a word appears in a text, this ambiguity is generally much reduced. Disambiguating POS involves using context to reduce the number of POS associated with words, and is one of the main challenges of lexical tagging. The problem of labeling words by POS frequently arises in natural language processing, for example for spelling correction, grammar or style checking, expression recognition, text-to-speech conversion, text corpus analysis, etc. Lexical tagging systems are thus useful as an initial component of many natural language processing systems. A number of recent lexical tagging systems produce multiple solutions when the text is lexically ambiguous or the uniquely correct solution cannot be found. These contributions aim to guarantee a zero silence rate: the correct tag(s) for a word must never be discarded. This objective is unrealistic for systems that tag each word uniquely. This article concerns a lexical disambiguation method adapted to the objective of a zero silence rate and implemented in Silberztein's INTEX system (1993). We present here a formal description of this method. We show that to verify a local disambiguation grammar in this framework, it is not sufficient to consider the transducer paths separately: one needs to verify their interactions. Similarly, if a combination of multiple transducers is used, the result cannot be predicted by considering them in isolation. Furthermore, when examining the initial labeling of a text as produced by INTEX, ideas for disambiguation rules come spontaneously, but grammatical intuitions may turn out to be inaccurate, often due to an unforeseen construction or ambiguity. If a zero silence rate is targeted, local grammars must be carefully tested. This is where a detailed specification of what a grammar will do once applied to texts would be necessary.
>
---
#### [new 031] ComboBench: Can LLMs Manipulate Physical Devices to Play Virtual Reality Games?
- **分类: cs.CL; cs.AI; cs.HC; cs.SE**

- **简介: 该论文提出ComboBench基准，评估大语言模型（LLMs）将语义指令转化为虚拟现实（VR）设备操作序列的能力。针对LLMs在物理设备操控中缺乏空间与过程推理的问题，研究通过262个场景测试7个模型，发现其在复杂交互中表现受限，但少样本学习可显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.24706v1](http://arxiv.org/pdf/2510.24706v1)**

> **作者:** Shuqing Li; Jiayi Yan; Chenyu Niu; Jen-tse Huang; Yun Peng; Wenxuan Wang; Yepang Liu; Michael R. Lyu
>
> **摘要:** Virtual Reality (VR) games require players to translate high-level semantic actions into precise device manipulations using controllers and head-mounted displays (HMDs). While humans intuitively perform this translation based on common sense and embodied understanding, whether Large Language Models (LLMs) can effectively replicate this ability remains underexplored. This paper introduces a benchmark, ComboBench, evaluating LLMs' capability to translate semantic actions into VR device manipulation sequences across 262 scenarios from four popular VR games: Half-Life: Alyx, Into the Radius, Moss: Book II, and Vivecraft. We evaluate seven LLMs, including GPT-3.5, GPT-4, GPT-4o, Gemini-1.5-Pro, LLaMA-3-8B, Mixtral-8x7B, and GLM-4-Flash, compared against annotated ground truth and human performance. Our results reveal that while top-performing models like Gemini-1.5-Pro demonstrate strong task decomposition capabilities, they still struggle with procedural reasoning and spatial understanding compared to humans. Performance varies significantly across games, suggesting sensitivity to interaction complexity. Few-shot examples substantially improve performance, indicating potential for targeted enhancement of LLMs' VR manipulation capabilities. We release all materials at https://sites.google.com/view/combobench.
>
---
#### [new 032] Open Korean Historical Corpus: A Millennia-Scale Diachronic Collection of Public Domain Texts
- **分类: cs.CL**

- **简介: 该论文提出开放韩语历史语料库，解决韩语历史文本数据稀缺问题。针对语言演变中书写系统变迁与词汇分化，构建涵盖1300年、6种语言及多种书写形式的1800万文档、50亿词元的语料库，支持历时语言分析与大模型预训练。**

- **链接: [http://arxiv.org/pdf/2510.24541v1](http://arxiv.org/pdf/2510.24541v1)**

> **作者:** Seyoung Song; Nawon Kim; Songeun Chae; Kiwoong Park; Jiho Jin; Haneul Yoo; Kyunghyun Cho; Alice Oh
>
> **备注:** Dataset and code available at https://github.com/seyoungsong/OKHC
>
> **摘要:** The history of the Korean language is characterized by a discrepancy between its spoken and written forms and a pivotal shift from Chinese characters to the Hangul alphabet. However, this linguistic evolution has remained largely unexplored in NLP due to a lack of accessible historical corpora. To address this gap, we introduce the Open Korean Historical Corpus, a large-scale, openly licensed dataset spanning 1,300 years and 6 languages, as well as under-represented writing systems like Korean-style Sinitic (Idu) and Hanja-Hangul mixed script. This corpus contains 18 million documents and 5 billion tokens from 19 sources, ranging from the 7th century to 2025. We leverage this resource to quantitatively analyze major linguistic shifts: (1) Idu usage peaked in the 1860s before declining sharply; (2) the transition from Hanja to Hangul was a rapid transformation starting around 1890; and (3) North Korea's lexical divergence causes modern tokenizers to produce up to 51 times higher out-of-vocabulary rates. This work provides a foundational resource for quantitative diachronic analysis by capturing the history of the Korean language. Moreover, it can serve as a pre-training corpus for large language models, potentially improving their understanding of Sino-Korean vocabulary in modern Hangul as well as archaic writing systems.
>
---
#### [new 033] CRADLE Bench: A Clinician-Annotated Benchmark for Multi-Faceted Mental Health Crisis and Safety Risk Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CRADLE Bench，一个用于多维度心理危机检测的临床标注基准，涵盖7类危机事件并引入时间标签。针对语言模型在识别自杀意念、虐待等危机时的不足，构建了600个标注测试集与420个开发集，并利用多模型共识自动标注约4K训练数据，同时训练多种基于不同一致标准的检测模型。**

- **链接: [http://arxiv.org/pdf/2510.23845v1](http://arxiv.org/pdf/2510.23845v1)**

> **作者:** Grace Byun; Rebecca Lipschutz; Sean T. Minton; Abigail Lott; Jinho D. Choi
>
> **摘要:** Detecting mental health crisis situations such as suicide ideation, rape, domestic violence, child abuse, and sexual harassment is a critical yet underexplored challenge for language models. When such situations arise during user--model interactions, models must reliably flag them, as failure to do so can have serious consequences. In this work, we introduce CRADLE BENCH, a benchmark for multi-faceted crisis detection. Unlike previous efforts that focus on a limited set of crisis types, our benchmark covers seven types defined in line with clinical standards and is the first to incorporate temporal labels. Our benchmark provides 600 clinician-annotated evaluation examples and 420 development examples, together with a training corpus of around 4K examples automatically labeled using a majority-vote ensemble of multiple language models, which significantly outperforms single-model annotation. We further fine-tune six crisis detection models on subsets defined by consensus and unanimous ensemble agreement, providing complementary models trained under different agreement criteria.
>
---
#### [new 034] Relative Scaling Laws for LLMs
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的相对缩放规律，旨在揭示不同子群体间性能差距随规模变化的动态。针对传统缩放定律忽略分布异质性的问题，作者在相同计算预算下训练255个模型，发现各领域表现轨迹各异：学术能力趋同，方言差异依人口而变，AI风险行为分化。研究推动了对模型鲁棒性的更深入理解。**

- **链接: [http://arxiv.org/pdf/2510.24626v1](http://arxiv.org/pdf/2510.24626v1)**

> **作者:** William Held; David Hall; Percy Liang; Diyi Yang
>
> **摘要:** Scaling laws describe how language models improve with additional data, parameters, and compute. While widely used, they are typically measured on aggregate test sets. Aggregate evaluations yield clean trends but average over heterogeneous subpopulations, obscuring performance disparities. We introduce relative scaling laws, which track how performance gaps between test distributions evolve with scale rather than focusing solely on absolute error. Using 255 decoder-only Transformers trained under matched-compute (IsoFLOP) budgets from $10^{18}$--$10^{20}$ FLOPs on standard pretraining datasets, we find diverse trajectories: academic domains on MMLU converge toward parity; regional English dialects shift depending on population size; and clusters of AI risk behaviours split, with capability- and influence-related risks increasing during pretraining while adversarial risks do not. These results show that although scaling improves overall performance, it is not a universal equalizer. To support further study, we release all model checkpoints from this work to enable practitioners to measure relative alongside traditional scaling laws, in order to better prioritize robustness challenges in light of the bitter lesson.
>
---
#### [new 035] Diffusion LLM with Native Variable Generation Lengths: Let [EOS] Lead the Way
- **分类: cs.CL**

- **简介: 该论文针对扩散语言模型（dLLM）生成长度固定导致效率低的问题，提出dLLM-Var模型，通过训练模型精准预测[EOS]标记，实现原生可变长度生成。该方法支持块扩散推理，保持全局双向注意力与高并行性，在保证精度的同时，相比传统dLLM提升30.1倍速度，较自回归模型快2.4倍。**

- **链接: [http://arxiv.org/pdf/2510.24605v1](http://arxiv.org/pdf/2510.24605v1)**

> **作者:** Yicun Yang; Cong Wang; Shaobo Wang; Zichen Wen; Biqing Qi; Hanlin Xu; Linfeng Zhang
>
> **摘要:** Diffusion-based large language models (dLLMs) have exhibited substantial potential for parallel text generation, which may enable more efficient generation compared to autoregressive models. However, current dLLMs suffer from fixed generation lengths, which indicates the generation lengths of dLLMs have to be determined before decoding as a hyper-parameter, leading to issues in efficiency and flexibility. To solve these problems, in this work, we propose to train a diffusion LLM with native variable generation lengths, abbreviated as dLLM-Var. Concretely, we aim to train a model to accurately predict the [EOS] token in the generated text, which makes a dLLM be able to natively infer in a block diffusion manner, while still maintaining the ability of global bi-directional (full) attention and high parallelism. Experiments on standard benchmarks demonstrate that our method achieves a 30.1x speedup over traditional dLLM inference paradigms and a 2.4x speedup relative to autoregressive models such as Qwen and Llama. Our method achieves higher accuracy and faster inference, elevating dLLMs beyond mere academic novelty and supporting their practical use in real-world applications. Codes and models have been released.
>
---
#### [new 036] TEXT2DB: Integration-Aware Information Extraction with Large Language Model Agents
- **分类: cs.CL**

- **简介: 论文提出TEXT2DB任务，旨在将文本信息精准整合到目标数据库中。针对传统信息抽取与应用需求不匹配的问题，设计OPAL框架，通过观察、规划、分析三阶段实现动态适应数据库结构的抽取与更新，支持数据填充、行插入、列添加等操作。**

- **链接: [http://arxiv.org/pdf/2510.24014v1](http://arxiv.org/pdf/2510.24014v1)**

> **作者:** Yizhu Jiao; Sha Li; Sizhe Zhou; Heng Ji; Jiawei Han
>
> **备注:** ACL 2025. Source code: https://github.com/yzjiao/Text2DB
>
> **摘要:** The task of information extraction (IE) is to extract structured knowledge from text. However, it is often not straightforward to utilize IE output due to the mismatch between the IE ontology and the downstream application needs. We propose a new formulation of IE TEXT2DB that emphasizes the integration of IE output and the target database (or knowledge base). Given a user instruction, a document set, and a database, our task requires the model to update the database with values from the document set to satisfy the user instruction. This task requires understanding user instructions for what to extract and adapting to the given DB/KB schema for how to extract on the fly. To evaluate this new task, we introduce a new benchmark featuring common demands such as data infilling, row population, and column addition. In addition, we propose an LLM agent framework OPAL (Observe-PlanAnalyze LLM) which includes an Observer component that interacts with the database, the Planner component that generates a code-based plan with calls to IE models, and the Analyzer component that provides feedback regarding code quality before execution. Experiments show that OPAL can successfully adapt to diverse database schemas by generating different code plans and calling the required IE models. We also highlight difficult cases such as dealing with large databases with complex dependencies and extraction hallucination, which we believe deserve further investigation. Source code: https://github.com/yzjiao/Text2DB
>
---
#### [new 037] Teaching LLMs to Abstain via Fine-Grained Semantic Confidence Reward
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型幻觉问题，提出基于细粒度语义置信度奖励的强化学习框架，通过样本级置信度引导模型精准识别知识边界，实现更可靠的拒绝回答。解决了现有方法依赖粗粒度信号导致的误判问题，并引入新评估指标。**

- **链接: [http://arxiv.org/pdf/2510.24020v1](http://arxiv.org/pdf/2510.24020v1)**

> **作者:** Hao An; Yang Xu
>
> **备注:** 23pages, 4figures
>
> **摘要:** Mitigating hallucinations in Large Language Models (LLMs) is critical for their reliable deployment. Existing methods typically fine-tune LLMs to abstain from answering questions beyond their knowledge scope. However, these methods often rely on coarse-grained signals to guide LLMs to abstain, such as overall confidence or uncertainty scores on multiple sampled answers, which may result in an imprecise awareness of the model's own knowledge boundaries. To this end, we propose a novel reinforcement learning framework built on $\textbf{\underline{Fi}ne-grained \underline{S}emantic \underline{Co}nfidence \underline{Re}ward (\Ours)}$, which guides LLMs to abstain via sample-specific confidence. Specifically, our method operates by sampling multiple candidate answers and conducting semantic clustering, then training the LLM to retain answers within high-confidence clusters and discard those within low-confidence ones, thereby promoting accurate post-hoc abstention. Additionally, we propose a new metric for evaluating the reliability of abstention fine-tuning tasks more comprehensively. Our method significantly enhances reliability in both in-domain and out-of-distribution benchmarks.
>
---
#### [new 038] MuSaG: A Multimodal German Sarcasm Dataset with Full-Modal Annotations
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MuSaG，首个德语多模态讽刺数据集，包含33分钟电视节目中的文本、音频、视频三模态标注数据。旨在解决多模态讽刺检测难题，通过基准测试多种模型，揭示当前模型在融合多模态信息上的不足，推动更贴近真实场景的讽刺理解研究。**

- **链接: [http://arxiv.org/pdf/2510.24178v1](http://arxiv.org/pdf/2510.24178v1)**

> **作者:** Aaron Scott; Maike Züfle; Jan Niehues
>
> **摘要:** Sarcasm is a complex form of figurative language in which the intended meaning contradicts the literal one. Its prevalence in social media and popular culture poses persistent challenges for natural language understanding, sentiment analysis, and content moderation. With the emergence of multimodal large language models, sarcasm detection extends beyond text and requires integrating cues from audio and vision. We present MuSaG, the first German multimodal sarcasm detection dataset, consisting of 33 minutes of manually selected and human-annotated statements from German television shows. Each instance provides aligned text, audio, and video modalities, annotated separately by humans, enabling evaluation in unimodal and multimodal settings. We benchmark nine open-source and commercial models, spanning text, audio, vision, and multimodal architectures, and compare their performance to human annotations. Our results show that while humans rely heavily on audio in conversational settings, models perform best on text. This highlights a gap in current multimodal models and motivates the use of MuSaG for developing models better suited to realistic scenarios. We release MuSaG publicly to support future research on multimodal sarcasm detection and human-model alignment.
>
---
#### [new 039] A word association network methodology for evaluating implicit biases in LLMs compared to humans
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种基于词关联网络的隐式偏见评估方法，通过模拟语义启动探测大模型与人类的隐性认知结构。旨在解决大模型隐性社会偏见难以量化评估的问题，实现了对性别、宗教等多维度偏见的跨主体比较，为语言模型的公平性研究提供可扩展的分析框架。**

- **链接: [http://arxiv.org/pdf/2510.24488v1](http://arxiv.org/pdf/2510.24488v1)**

> **作者:** Katherine Abramski; Giulio Rossetti; Massimo Stella
>
> **备注:** 24 pages, 13 figures, 3 tables
>
> **摘要:** As Large language models (LLMs) become increasingly integrated into our lives, their inherent social biases remain a pressing concern. Detecting and evaluating these biases can be challenging because they are often implicit rather than explicit in nature, so developing evaluation methods that assess the implicit knowledge representations of LLMs is essential. We present a novel word association network methodology for evaluating implicit biases in LLMs based on simulating semantic priming within LLM-generated word association networks. Our prompt-based approach taps into the implicit relational structures encoded in LLMs, providing both quantitative and qualitative assessments of bias. Unlike most prompt-based evaluation methods, our method enables direct comparisons between various LLMs and humans, providing a valuable point of reference and offering new insights into the alignment of LLMs with human cognition. To demonstrate the utility of our methodology, we apply it to both humans and several widely used LLMs to investigate social biases related to gender, religion, ethnicity, sexual orientation, and political party. Our results reveal both convergences and divergences between LLM and human biases, providing new perspectives on the potential risks of using LLMs. Our methodology contributes to a systematic, scalable, and generalizable framework for evaluating and comparing biases across multiple LLMs and humans, advancing the goal of transparent and socially responsible language technologies.
>
---
#### [new 040] M-Eval: A Heterogeneity-Based Framework for Multi-evidence Validation in Medical RAG Systems
- **分类: cs.CL**

- **简介: 该论文针对医疗RAG系统中存在的幻觉和知识误用问题，提出M-Eval框架。基于循证医学的异质性分析，通过多源证据验证响应事实性与证据可靠性，提升LLM生成结果的准确性与可信度。**

- **链接: [http://arxiv.org/pdf/2510.23995v1](http://arxiv.org/pdf/2510.23995v1)**

> **作者:** Mengzhou Sun; Sendong Zhao; Jianyu Chen; Haochun Wang; Bin Qin
>
> **摘要:** Retrieval-augmented Generation (RAG) has demonstrated potential in enhancing medical question-answering systems through the integration of large language models (LLMs) with external medical literature. LLMs can retrieve relevant medical articles to generate more professional responses efficiently. However, current RAG applications still face problems. They generate incorrect information, such as hallucinations, and they fail to use external knowledge correctly. To solve these issues, we propose a new method named M-Eval. This method is inspired by the heterogeneity analysis approach used in Evidence-Based Medicine (EBM). Our approach can check for factual errors in RAG responses using evidence from multiple sources. First, we extract additional medical literature from external knowledge bases. Then, we retrieve the evidence documents generated by the RAG system. We use heterogeneity analysis to check whether the evidence supports different viewpoints in the response. In addition to verifying the accuracy of the response, we also assess the reliability of the evidence provided by the RAG system. Our method shows an improvement of up to 23.31% accuracy across various LLMs. This work can help detect errors in current RAG-based medical systems. It also makes the applications of LLMs more reliable and reduces diagnostic errors.
>
---
#### [new 041] Beyond Understanding: Evaluating the Pragmatic Gap in LLMs' Cultural Processing of Figurative Language
- **分类: cs.CL**

- **简介: 该论文聚焦于大语言模型在跨文化语用理解上的差距，针对阿拉伯语和英语的习语、谚语，评估模型在理解、语用使用与情感内涵方面的表现。研究发现模型在处理本地化文化表达时存在显著短板，尤其在语用层面。为此，作者构建并发布了首个用于埃及方言习语评估的数据集Kinayat。**

- **链接: [http://arxiv.org/pdf/2510.23828v1](http://arxiv.org/pdf/2510.23828v1)**

> **作者:** Mena Attia; Aashiq Muhamed; Mai Alkhamissi; Thamar Solorio; Mona Diab
>
> **摘要:** We present a comprehensive evaluation of the ability of large language models (LLMs) to process culturally grounded language, specifically to understand and pragmatically use figurative expressions that encode local knowledge and cultural nuance. Using figurative language as a proxy for cultural nuance and local knowledge, we design evaluation tasks for contextual understanding, pragmatic use, and connotation interpretation in Arabic and English. We evaluate 22 open- and closed-source LLMs on Egyptian Arabic idioms, multidialectal Arabic proverbs, and English proverbs. Our results show a consistent hierarchy: the average accuracy for Arabic proverbs is 4.29% lower than for English proverbs, and performance for Egyptian idioms is 10.28% lower than for Arabic proverbs. For the pragmatic use task, accuracy drops by 14.07% relative to understanding, though providing contextual idiomatic sentences improves accuracy by 10.66%. Models also struggle with connotative meaning, reaching at most 85.58% agreement with human annotators on idioms with 100% inter-annotator agreement. These findings demonstrate that figurative language serves as an effective diagnostic for cultural reasoning: while LLMs can often interpret figurative meaning, they face challenges in using it appropriately. To support future research, we release Kinayat, the first dataset of Egyptian Arabic idioms designed for both figurative understanding and pragmatic use evaluation.
>
---
#### [new 042] Zero-Shot Cross-Lingual Transfer using Prefix-Based Adaptation
- **分类: cs.CL; cs.AI; cs.LG; I.2.7**

- **简介: 该论文研究零样本跨语言迁移任务，针对解码器型大模型在多语言新任务上的适应难题。提出三种前缀调优方法，对比LoRA，在35+语言上实现更优性能，尤其在低资源场景下表现突出，验证了前缀方法在参数效率与跨语言泛化上的优势。**

- **链接: [http://arxiv.org/pdf/2510.24619v1](http://arxiv.org/pdf/2510.24619v1)**

> **作者:** Snegha A; Sayambhu Sen; Piyush Singh Pasi; Abhishek Singhania; Preethi Jyothi
>
> **备注:** 12 Pages
>
> **摘要:** With the release of new large language models (LLMs) like Llama and Mistral, zero-shot cross-lingual transfer has become increasingly feasible due to their multilingual pretraining and strong generalization capabilities. However, adapting these decoder-only LLMs to new tasks across languages remains challenging. While parameter-efficient fine-tuning (PeFT) techniques like Low-Rank Adaptation (LoRA) are widely used, prefix-based techniques such as soft prompt tuning, prefix tuning, and Llama Adapter are less explored, especially for zero-shot transfer in decoder-only models. We present a comprehensive study of three prefix-based methods for zero-shot cross-lingual transfer from English to 35+ high- and low-resource languages. Our analysis further explores transfer across linguistic families and scripts, as well as the impact of scaling model sizes from 1B to 24B. With Llama 3.1 8B, prefix methods outperform LoRA-baselines by up to 6% on the Belebele benchmark. Similar improvements were observed with Mistral v0.3 7B as well. Despite using only 1.23M learning parameters with prefix tuning, we achieve consistent improvements across diverse benchmarks. These findings highlight the potential of prefix-based techniques as an effective and scalable alternative to LoRA, particularly in low-resource multilingual settings.
>
---
#### [new 043] MERGE: Minimal Expression-Replacement GEneralization Test for Natural Language Inference
- **分类: cs.CL**

- **简介: 该论文针对自然语言推理（NLI）模型泛化能力弱的问题，提出MERGE方法，通过最小化表达替换生成高质量测试变体，评估模型在保留推理逻辑下的表现。实验表明模型性能下降4-20%，揭示其泛化缺陷，并分析了词类、词频与合理性的影响。**

- **链接: [http://arxiv.org/pdf/2510.24295v1](http://arxiv.org/pdf/2510.24295v1)**

> **作者:** Mădălina Zgreabăn; Tejaswini Deoskar; Lasha Abzianidze
>
> **备注:** Pre-print
>
> **摘要:** In recent years, many generalization benchmarks have shown language models' lack of robustness in natural language inference (NLI). However, manually creating new benchmarks is costly, while automatically generating high-quality ones, even by modifying existing benchmarks, is extremely difficult. In this paper, we propose a methodology for automatically generating high-quality variants of original NLI problems by replacing open-class words, while crucially preserving their underlying reasoning. We dub our generalization test as MERGE (Minimal Expression-Replacements GEneralization), which evaluates the correctness of models' predictions across reasoning-preserving variants of the original problem. Our results show that NLI models' perform 4-20% worse on variants, suggesting low generalizability even on such minimally altered problems. We also analyse how word class of the replacements, word probability, and plausibility influence NLI models' performance.
>
---
#### [new 044] InteractComp: Evaluating Search Agents With Ambiguous Queries
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出InteractComp基准，用于评估搜索代理识别并交互解决模糊查询的能力。针对现有模型缺乏互动机制、无法处理不完整查询的问题，构建了210个真实歧义问题，揭示模型普遍高估自身能力，且互动能力长期停滞。实验表明强制交互显著提升性能，凸显训练与评估中互动能力的重要性。**

- **链接: [http://arxiv.org/pdf/2510.24668v1](http://arxiv.org/pdf/2510.24668v1)**

> **作者:** Mingyi Deng; Lijun Huang; Yani Fan; Jiayi Zhang; Fashen Ren; Jinyi Bai; Fuzhen Yang; Dayi Miao; Zhaoyang Yu; Yifan Wu; Yanfei Zhang; Fengwei Teng; Yingjia Wan; Song Hu; Yude Li; Xin Jin; Conghao Hu; Haoyu Li; Qirui Fu; Tai Zhong; Xinyu Wang; Xiangru Tang; Nan Tang; Chenglin Wu; Yuyu Luo
>
> **摘要:** Language agents have demonstrated remarkable potential in web search and information retrieval. However, these search agents assume user queries are complete and unambiguous, an assumption that diverges from reality where users begin with incomplete queries requiring clarification through interaction. Yet most agents lack interactive mechanisms during the search process, and existing benchmarks cannot assess this capability. To address this gap, we introduce InteractComp, a benchmark designed to evaluate whether search agents can recognize query ambiguity and actively interact to resolve it during search. Following the principle of easy to verify, interact to disambiguate, we construct 210 expert-curated questions across 9 domains through a target-distractor methodology that creates genuine ambiguity resolvable only through interaction. Evaluation of 17 models reveals striking failure: the best model achieves only 13.73% accuracy despite 71.50% with complete context, exposing systematic overconfidence rather than reasoning deficits. Forced interaction produces dramatic gains, demonstrating latent capability current strategies fail to engage. Longitudinal analysis shows interaction capabilities stagnated over 15 months while search performance improved seven-fold, revealing a critical blind spot. This stagnation, coupled with the immediate feedback inherent to search tasks, makes InteractComp a valuable resource for both evaluating and training interaction capabilities in search agents. The code is available at https://github.com/FoundationAgents/InteractComp.
>
---
#### [new 045] Global PIQA: Evaluating Physical Commonsense Reasoning Across 100+ Languages and Cultures
- **分类: cs.CL**

- **简介: 该论文提出Global PIQA，一个覆盖100+语言和文化的物理常识推理基准，由全球335名研究者协作构建。旨在解决大语言模型在多语言、跨文化场景下评估缺失的问题，揭示模型在低资源语言中表现较差，凸显日常文化知识的不足。**

- **链接: [http://arxiv.org/pdf/2510.24081v1](http://arxiv.org/pdf/2510.24081v1)**

> **作者:** Tyler A. Chang; Catherine Arnett; Abdelrahman Eldesokey; Abdelrahman Sadallah; Abeer Kashar; Abolade Daud; Abosede Grace Olanihun; Adamu Labaran Mohammed; Adeyemi Praise; Adhikarinayum Meerajita Sharma; Aditi Gupta; Afitab Iyigun; Afonso Simplício; Ahmed Essouaied; Aicha Chorana; Akhil Eppa; Akintunde Oladipo; Akshay Ramesh; Aleksei Dorkin; Alfred Malengo Kondoro; Alham Fikri Aji; Ali Eren Çetintaş; Allan Hanbury; Alou Dembele; Alp Niksarli; Álvaro Arroyo; Amin Bajand; Amol Khanna; Ana Chkhaidze; Ana Condez; Andiswa Mkhonto; Andrew Hoblitzell; Andrew Tran; Angelos Poulis; Anirban Majumder; Anna Vacalopoulou; Annette Kuuipolani Kanahele Wong; Annika Simonsen; Anton Kovalev; Ashvanth. S; Ayodeji Joseph Lana; Barkin Kinay; Bashar Alhafni; Benedict Cibalinda Busole; Bernard Ghanem; Bharti Nathani; Biljana Stojanovska Đurić; Bola Agbonile; Bragi Bergsson; Bruce Torres Fischer; Burak Tutar; Burcu Alakuş Çınar; Cade J. Kanoniakapueo Kane; Can Udomcharoenchaikit; Catherine Arnett; Chadi Helwe; Chaithra Reddy Nerella; Chen Cecilia Liu; Chiamaka Glory Nwokolo; Cristina España-Bonet; Cynthia Amol; DaeYeop Lee; Dana Arad; Daniil Dzenhaliou; Daria Pugacheva; Dasol Choi; Daud Abolade; David Liu; David Semedo; Deborah Popoola; Deividas Mataciunas; Delphine Nyaboke; Dhyuthy Krishna Kumar; Diogo Glória-Silva; Diogo Tavares; Divyanshu Goyal; DongGeon Lee; Ebele Nwamaka Anajemba; Egonu Ngozi Grace; Elena Mickel; Elena Tutubalina; Elias Herranen; Emile Anand; Emmanuel Habumuremyi; Emuobonuvie Maria Ajiboye; Eryawan Presma Yulianrifat; Esther Adenuga; Ewa Rudnicka; Faith Olabisi Itiola; Faran Taimoor Butt; Fathima Thekkekara; Fatima Haouari; Filbert Aurelian Tjiaranata; Firas Laakom; Francesca Grasso; Francesco Orabona; Francesco Periti; Gbenga Kayode Solomon; Gia Nghia Ngo; Gloria Udhehdhe-oze; Gonçalo Martins; Gopi Naga Sai Ram Challagolla; Guijin Son; Gulnaz Abdykadyrova; Hafsteinn Einarsson; Hai Hu; Hamidreza Saffari; Hamza Zaidi; Haopeng Zhang; Harethah Abu Shairah; Harry Vuong; Hele-Andra Kuulmets; Houda Bouamor; Hwanjo Yu; Iben Nyholm Debess; İbrahim Ethem Deveci; Ikhlasul Akmal Hanif; Ikhyun Cho; Inês Calvo; Inês Vieira; Isaac Manzi; Ismail Daud; Itay Itzhak; Iuliia; Alekseenko; Ivan Belashkin; Ivan Spada; Ivan Zhelyazkov; Jacob Brinton; Jafar Isbarov; Jaka Čibej; Jan Čuhel; Jan Kocoń; Jauza Akbar Krito; Jebish Purbey; Jennifer Mickel; Jennifer Za; Jenny Kunz; Jihae Jeong; Jimena Tena Dávalos; Jinu Lee; João Magalhães; John Yi; Jongin Kim; Joseph Chataignon; Joseph Marvin Imperial; Jubeerathan Thevakumar; Judith Land; Junchen Jiang; Jungwhan Kim; Kairit Sirts; Kamesh R; Kamesh V; Kanda Patrick Tshinu; Kätriin Kukk; Kaustubh Ponkshe; Kavsar Huseynova; Ke He; Kelly Buchanan; Kengatharaiyer Sarveswaran; Kerem Zaman; Khalil Mrini; Kian Kyars; Krister Kruusmaa; Kusum Chouhan; Lainitha Krishnakumar; Laura Castro Sánchez; Laura Porrino Moscoso; Leshem Choshen; Levent Sencan; Lilja Øvrelid; Lisa Alazraki; Lovina Ehimen-Ugbede; Luheerathan Thevakumar; Luxshan Thavarasa; Mahnoor Malik; Mamadou K. Keita; Mansi Jangid; Marco De Santis; Marcos García; Marek Suppa; Mariam D'Ciofalo; Marii Ojastu; Maryam Sikander; Mausami Narayan; Maximos Skandalis; Mehak Mehak; Mehmet İlteriş Bozkurt; Melaku Bayu Workie; Menan Velayuthan; Michael Leventhal; Michał Marcińczuk; Mirna Potočnjak; Mohammadamin Shafiei; Mridul Sharma; Mrityunjaya Indoria; Muhammad Ravi Shulthan Habibi; Murat Kolić; Nada Galant; Naphat Permpredanun; Narada Maugin; Nicholas Kluge Corrêa; Nikola Ljubešić; Nirmal Thomas; Nisansa de Silva; Nisheeth Joshi; Nitish Ponkshe; Nizar Habash; Nneoma C. Udeze; Noel Thomas; Noémi Ligeti-Nagy; Nouhoum Coulibaly; Nsengiyumva Faustin; Odunayo Kareemat Buliaminu; Odunayo Ogundepo; Oghojafor Godswill Fejiro; Ogundipe Blessing Funmilola; Okechukwu God'spraise; Olanrewaju Samuel; Olaoye Deborah Oluwaseun; Olasoji Akindejoye; Olga Popova; Olga Snissarenko; Onyinye Anulika Chiemezie; Orkun Kinay; Osman Tursun; Owoeye Tobiloba Moses; Oyelade Oluwafemi Joshua; Oyesanmi Fiyinfoluwa; Pablo Gamallo; Pablo Rodríguez Fernández; Palak Arora; Pedro Valente; Peter Rupnik; Philip Oghenesuowho Ekiugbo; Pramit Sahoo; Prokopis Prokopidis; Pua Niau-Puhipau; Quadri Yahya; Rachele Mignone; Raghav Singhal; Ram Mohan Rao Kadiyala; Raphael Merx; Rapheal Afolayan; Ratnavel Rajalakshmi; Rishav Ghosh; Romina Oji; Ron Kekeha Solis; Rui Guerra; Rushikesh Zawar; Sa'ad Nasir Bashir; Saeed Alzaabi; Sahil Sandeep; Sai Pavan Batchu; SaiSandeep Kantareddy; Salsabila Zahirah Pranida; Sam Buchanan; Samuel Rutunda; Sander Land; Sarah Sulollari; Sardar Ali; Saroj Sapkota; Saulius Tautvaisas; Sayambhu Sen; Sayantani Banerjee; Sebastien Diarra; SenthilNathan. M; Sewoong Lee; Shaan Shah; Shankar Venkitachalam; Sharifa Djurabaeva; Sharon Ibejih; Shivanya Shomir Dutta; Siddhant Gupta; Silvia Paniagua Suárez; Sina Ahmadi; Sivasuthan Sukumar; Siyuan Song; Snegha A.; Sokratis Sofianopoulos; Sona Elza Simon; Sonja Benčina; Sophie Gvasalia; Sphurti Kirit More; Spyros Dragazis; Stephan P. Kaufhold; Suba. S; Sultan AlRashed; Surangika Ranathunga; Taiga Someya; Taja Kuzman Pungeršek; Tal Haklay; Tasi'u Jibril; Tatsuya Aoyama; Tea Abashidze; Terenz Jomar Dela Cruz; Terra Blevins; Themistoklis Nikas; Theresa Dora Idoko; Thu Mai Do; Tilek Chubakov; Tommaso Gargiani; Uma Rathore; Uni Johannesen; Uwuma Doris Ugwu; Vallerie Alexandra Putra; Vanya Bannihatti Kumar; Varsha Jeyarajalingam; Varvara Arzt; Vasudevan Nedumpozhimana; Viktoria Ondrejova; Viktoryia Horbik; Vishnu Vardhan Reddy Kummitha; Vuk Dinić; Walelign Tewabe Sewunetie; Winston Wu; Xiaojing Zhao; Yacouba Diarra; Yaniv Nikankin; Yash Mathur; Yixi Chen; Yiyuan Li; Yolanda Xavier; Yonatan Belinkov; Yusuf Ismail Abayomi; Zaid Alyafeai; Zhengyang Shan; Zhi Rui Tam; Zilu Tang; Zuzana Nadova; Baber Abbasi; Stella Biderman; David Stap; Duygu Ataman; Fabian Schmidt; Hila Gonen; Jiayi Wang; David Ifeoluwa Adelani
>
> **备注:** Preprint
>
> **摘要:** To date, there exist almost no culturally-specific evaluation benchmarks for large language models (LLMs) that cover a large number of languages and cultures. In this paper, we present Global PIQA, a participatory commonsense reasoning benchmark for over 100 languages, constructed by hand by 335 researchers from 65 countries around the world. The 116 language varieties in Global PIQA cover five continents, 14 language families, and 23 writing systems. In the non-parallel split of Global PIQA, over 50% of examples reference local foods, customs, traditions, or other culturally-specific elements. We find that state-of-the-art LLMs perform well on Global PIQA in aggregate, but they exhibit weaker performance in lower-resource languages (up to a 37% accuracy gap, despite random chance at 50%). Open models generally perform worse than proprietary models. Global PIQA highlights that in many languages and cultures, everyday knowledge remains an area for improvement, alongside more widely-discussed capabilities such as complex reasoning and expert knowledge. Beyond its uses for LLM evaluation, we hope that Global PIQA provides a glimpse into the wide diversity of cultures in which human language is embedded.
>
---
#### [new 046] CritiCal: Can Critique Help LLM Uncertainty or Confidence Calibration?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）的置信度校准问题，旨在提升高风险场景下的可信度。针对传统方法依赖精确标签的局限，提出基于自然语言批判的校准方法CritiCal，通过分析应批判不确定还是置信度，以及自批判与训练校准两种方式，显著提升模型在复杂推理和分布外任务中的置信度准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.24505v1](http://arxiv.org/pdf/2510.24505v1)**

> **作者:** Qing Zong; Jiayu Liu; Tianshi Zheng; Chunyang Li; Baixuan Xu; Haochen Shi; Weiqi Wang; Zhaowei Wang; Chunkit Chan; Yangqiu Song
>
> **摘要:** Accurate confidence calibration in Large Language Models (LLMs) is critical for safe use in high-stakes domains, where clear verbalized confidence enhances user trust. Traditional methods that mimic reference confidence expressions often fail to capture the reasoning needed for accurate confidence assessment. We propose natural language critiques as a solution, ideally suited for confidence calibration, as precise gold confidence labels are hard to obtain and often require multiple generations. This paper studies how natural language critiques can enhance verbalized confidence, addressing: (1) What to critique: uncertainty (question-focused) or confidence (answer-specific)? Analysis shows confidence suits multiple-choice tasks, while uncertainty excels in open-ended scenarios. (2) How to critique: self-critique or critique calibration training? We propose Self-Critique, enabling LLMs to critique and optimize their confidence beyond mere accuracy, and CritiCal, a novel Critique Calibration training method that leverages natural language critiques to improve confidence calibration, moving beyond direct numerical optimization. Experiments show that CritiCal significantly outperforms Self-Critique and other competitive baselines, even surpassing its teacher model, GPT-4o, in complex reasoning tasks. CritiCal also shows robust generalization in out-of-distribution settings, advancing LLM's reliability.
>
---
#### [new 047] Uncovering the Potential Risks in Unlearning: Danger of English-only Unlearning in Multilingual LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言大模型的去学习任务，针对仅用英文数据去学习导致的语言混淆问题。提出N-Mix评分量化语言混淆，揭示参考基准评估指标在高混淆时产生误判，并主张采用语义评估指标以更准确衡量去学习效果。**

- **链接: [http://arxiv.org/pdf/2510.23949v1](http://arxiv.org/pdf/2510.23949v1)**

> **作者:** Kyomin Hwang; Hyeonjin Kim; Seungyeon Kim; Sunghyun Wee; Nojun Kwak
>
> **摘要:** There have been a couple of studies showing that attempting to erase multilingual knowledge using only English data is insufficient for multilingual LLMs. However, their analyses remain highly performance-oriented. In this paper, we switch the point of view to evaluation, and address an additional blind spot which reveals itself when the multilingual LLM is fully finetuned with parallel multilingual dataset before unlearning. Here, language confusion occurs whereby a model responds in language different from that of the input prompt. Language confusion is a problematic phenomenon in unlearning, causing the standard reference-based metrics to fail. We tackle this phenomenon in three steps: (1) introduce N-gram-based Language-Mix (N-Mix) score to quantitatively show the language confusion is pervasive and consistent in multilingual LLMs, (2) demonstrate that reference-based metrics result in false negatives when N-Mix score is high, and(3) suggest the need of new type of unlearning evaluation that can directly assess the content of the generated sentences. We call this type of metrics as semantic-based metric.
>
---
#### [new 048] LongWeave: A Long-Form Generation Benchmark Bridging Real-World Relevance and Verifiability
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LongWeave，一个兼顾真实性和可验证性的长文本生成评估基准。针对现有基准或缺乏真实性或难验证的问题，引入约束-验证评估（CoV-Eval），构建基于真实场景的可验证任务，支持长输入输出，评估23个LLM在复杂约束下的长文本生成能力。**

- **链接: [http://arxiv.org/pdf/2510.24345v1](http://arxiv.org/pdf/2510.24345v1)**

> **作者:** Zikai Xiao; Fei Huang; Jianhong Tu; Jianhui Wei; Wen Ma; Yuxuan Zhou; Jian Wu; Bowen Yu; Zuozhu Liu; Junyang Lin
>
> **备注:** EMNLP Findings 2025
>
> **摘要:** Generating long, informative, and factual outputs remains a major challenge for Large Language Models (LLMs). Existing benchmarks for long-form generation typically assess real-world queries with hard-to-verify metrics or use synthetic setups that ease evaluation but overlook real-world intricacies. In this paper, we introduce \textbf{LongWeave}, which balances real-world and verifiable assessment with Constraint-Verifier Evaluation (CoV-Eval). CoV-Eval constructs tasks by first defining verifiable targets within real-world scenarios, then systematically generating corresponding queries, textual materials, and constraints based on these targets. This ensures that tasks are both realistic and objectively assessable, enabling rigorous assessment of model capabilities in meeting complex real-world constraints. LongWeave supports customizable input/output lengths (up to 64K/8K tokens) across seven distinct tasks. Evaluation on 23 LLMs shows that even state-of-the-art models encounter significant challenges in long-form generation as real-world complexity and output length increase.
>
---
#### [new 049] Evaluating LLMs on Generating Age-Appropriate Child-Like Conversations
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在评估大模型生成符合5岁和9岁儿童语言水平的挪威语对话的能力。研究对比了五种模型，通过教育专家盲评发现多数模型生成语言偏成人化，凸显低资源语言中儿童语料稀缺的问题。**

- **链接: [http://arxiv.org/pdf/2510.24250v1](http://arxiv.org/pdf/2510.24250v1)**

> **作者:** Syed Zohaib Hassan; Pål Halvorsen; Miriam S. Johnson; Pierre Lison
>
> **备注:** 11 pages excluding references and appendix. 3 figures and 6 tables
>
> **摘要:** Large Language Models (LLMs), predominantly trained on adult conversational data, face significant challenges when generating authentic, child-like dialogue for specialized applications. We present a comparative study evaluating five different LLMs (GPT-4, RUTER-LLAMA-2-13b, GPTSW, NorMistral-7b, and NorBloom-7b) to generate age-appropriate Norwegian conversations for children aged 5 and 9 years. Through a blind evaluation by eleven education professionals using both real child interview data and LLM-generated text samples, we assessed authenticity and developmental appropriateness. Our results show that evaluators achieved strong inter-rater reliability (ICC=0.75) and demonstrated higher accuracy in age prediction for younger children (5-year-olds) compared to older children (9-year-olds). While GPT-4 and NorBloom-7b performed relatively well, most models generated language perceived as more linguistically advanced than the target age groups. These findings highlight critical data-related challenges in developing LLM systems for specialized applications involving children, particularly in low-resource languages where comprehensive age-appropriate lexical resources are scarce.
>
---
#### [new 050] How Pragmatics Shape Articulation: A Computational Case Study in STEM ASL Discourse
- **分类: cs.CL**

- **简介: 该论文研究自然对话中语用如何影响手语的表达方式，属于手语计算建模任务。针对现有模型依赖孤立词汇数据的问题，构建了ASL STEM对话动作捕捉数据集，分析互动对话与独白中手势的时空差异，发现对话中手势更短且具同步性，提出评估嵌入模型对语用适应性的方法。**

- **链接: [http://arxiv.org/pdf/2510.23842v1](http://arxiv.org/pdf/2510.23842v1)**

> **作者:** Saki Imai; Lee Kezar; Laurel Aichler; Mert Inan; Erin Walker; Alicia Wooten; Lorna Quandt; Malihe Alikhani
>
> **摘要:** Most state-of-the-art sign language models are trained on interpreter or isolated vocabulary data, which overlooks the variability that characterizes natural dialogue. However, human communication dynamically adapts to contexts and interlocutors through spatiotemporal changes and articulation style. This specifically manifests itself in educational settings, where novel vocabularies are used by teachers, and students. To address this gap, we collect a motion capture dataset of American Sign Language (ASL) STEM (Science, Technology, Engineering, and Mathematics) dialogue that enables quantitative comparison between dyadic interactive signing, solo signed lecture, and interpreted articles. Using continuous kinematic features, we disentangle dialogue-specific entrainment from individual effort reduction and show spatiotemporal changes across repeated mentions of STEM terms. On average, dialogue signs are 24.6%-44.6% shorter in duration than the isolated signs, and show significant reductions absent in monologue contexts. Finally, we evaluate sign embedding models on their ability to recognize STEM signs and approximate how entrained the participants become over time. Our study bridges linguistic analysis and computational modeling to understand how pragmatics shape sign articulation and its representation in sign language technologies.
>
---
#### [new 051] Exploring the Influence of Relevant Knowledge for Natural Language Generation Interpretability
- **分类: cs.CL**

- **简介: 该论文研究外部知识对自然语言生成可解释性的影响，聚焦常识生成任务。通过构建KITGI基准，对比全知识与过滤知识下的生成效果，发现相关知识显著提升语句合理性和概念覆盖度，验证了知识增强对NLG可解释性的关键作用。**

- **链接: [http://arxiv.org/pdf/2510.24179v1](http://arxiv.org/pdf/2510.24179v1)**

> **作者:** Iván Martínez-Murillo; Paloma Moreda; Elena Lloret
>
> **摘要:** This paper explores the influence of external knowledge integration in Natural Language Generation (NLG), focusing on a commonsense generation task. We extend the CommonGen dataset by creating KITGI, a benchmark that pairs input concept sets with retrieved semantic relations from ConceptNet and includes manually annotated outputs. Using the T5-Large model, we compare sentence generation under two conditions: with full external knowledge and with filtered knowledge where highly relevant relations were deliberately removed. Our interpretability benchmark follows a three-stage method: (1) identifying and removing key knowledge, (2) regenerating sentences, and (3) manually assessing outputs for commonsense plausibility and concept coverage. Results show that sentences generated with full knowledge achieved 91\% correctness across both criteria, while filtering reduced performance drastically to 6\%. These findings demonstrate that relevant external knowledge is critical for maintaining both coherence and concept coverage in NLG. This work highlights the importance of designing interpretable, knowledge-enhanced NLG systems and calls for evaluation frameworks that capture the underlying reasoning beyond surface-level metrics.
>
---
#### [new 052] SPARTA: Evaluating Reasoning Segmentation Robustness through Black-Box Adversarial Paraphrasing in Text Autoencoder Latent Space
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对视觉语言任务中的推理分割，研究文本语义不变但能降低模型性能的对抗性改写问题。提出SPARTA方法，在文本自编码器隐空间中通过强化学习生成高保真对抗性改写，有效提升攻击成功率，揭示当前模型对语义等价文本扰动的脆弱性。**

- **链接: [http://arxiv.org/pdf/2510.24446v1](http://arxiv.org/pdf/2510.24446v1)**

> **作者:** Viktoriia Zinkovich; Anton Antonov; Andrei Spiridonov; Denis Shepelev; Andrey Moskalenko; Daria Pugacheva; Elena Tutubalina; Andrey Kuznetsov; Vlad Shakhuro
>
> **摘要:** Multimodal large language models (MLLMs) have shown impressive capabilities in vision-language tasks such as reasoning segmentation, where models generate segmentation masks based on textual queries. While prior work has primarily focused on perturbing image inputs, semantically equivalent textual paraphrases-crucial in real-world applications where users express the same intent in varied ways-remain underexplored. To address this gap, we introduce a novel adversarial paraphrasing task: generating grammatically correct paraphrases that preserve the original query meaning while degrading segmentation performance. To evaluate the quality of adversarial paraphrases, we develop a comprehensive automatic evaluation protocol validated with human studies. Furthermore, we introduce SPARTA-a black-box, sentence-level optimization method that operates in the low-dimensional semantic latent space of a text autoencoder, guided by reinforcement learning. SPARTA achieves significantly higher success rates, outperforming prior methods by up to 2x on both the ReasonSeg and LLMSeg-40k datasets. We use SPARTA and competitive baselines to assess the robustness of advanced reasoning segmentation models. We reveal that they remain vulnerable to adversarial paraphrasing-even under strict semantic and grammatical constraints. All code and data will be released publicly upon acceptance.
>
---
#### [new 053] Lookahead Tree-Based Rollouts for Enhanced Trajectory-Level Exploration in Reinforcement Learning with Verifiable Rewards
- **分类: cs.CL**

- **简介: 该论文针对强化学习中基于可验证奖励的推理任务，解决轨迹多样性不足问题。提出前瞻树状滚动生成（LATR）策略，通过高不确定性处分支、前瞻模拟与相似性剪枝，提升轨迹多样性，显著加速政策学习并提升性能。**

- **链接: [http://arxiv.org/pdf/2510.24302v1](http://arxiv.org/pdf/2510.24302v1)**

> **作者:** Shangyu Xing; Siyuan Wang; Chenyuan Yang; Xinyu Dai; Xiang Ren
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR), particularly with algorithms like Group Relative Policy Optimization (GRPO), has proven highly effective in enhancing the reasoning capabilities of large language models. However, a critical bottleneck in current pipelines lies in the limited diversity of sampled trajectories during group rollouts. Homogeneous trajectories and their associated rewards would diminish the return signals for policy updates, thereby hindering effective policy learning. This lack of diversity stems primarily from token-level stochastic sampling, where local variations are likely to collapse into near-identical reasoning paths. To address this limitation, we propose Lookahead Tree-Based Rollouts (LATR), a novel rollout strategy designed to explicitly promotes trajectory-level diversity by enforcing branching into different candidate tokens likely to yield distinct continuations. Specifically, LATR iteratively operates in three stages: (1) branching at high-uncertainty generation steps, (2) performing lookahead simulation for each new branch, and (3) pruning branches that exhibits prolonged similarity during simulation. Compared with stochastic Sampling, LATR accelerates policy learning by 131% on average and improves final pass@1 performance by 4.2% on both GRPO and Dynamic sAmpling Policy Optimization (DAPO) algorithms across different reasoning tasks. Our code and data are publicly available at https://github.com/starreeze/latr.
>
---
#### [new 054] META-RAG: Meta-Analysis-Inspired Evidence-Re-Ranking Method for Retrieval-Augmented Generation in Evidence-Based Medicine
- **分类: cs.CL**

- **简介: 该论文针对证据医学中检索增强生成（RAG）难以筛选高质量证据的问题，提出META-RAG方法。受系统评价启发，融合可靠性、异质性与外推分析，对医学文献进行重排序与过滤，显著提升诊断准确率（最高+11.4%），增强LLM输出的可靠性。**

- **链接: [http://arxiv.org/pdf/2510.24003v1](http://arxiv.org/pdf/2510.24003v1)**

> **作者:** Mengzhou Sun; Sendong Zhao; Jianyu Chen; Haochun Wang; Bin Qin
>
> **摘要:** Evidence-based medicine (EBM) holds a crucial role in clinical application. Given suitable medical articles, doctors effectively reduce the incidence of misdiagnoses. Researchers find it efficient to use large language models (LLMs) techniques like RAG for EBM tasks. However, the EBM maintains stringent requirements for evidence, and RAG applications in EBM struggle to efficiently distinguish high-quality evidence. Therefore, inspired by the meta-analysis used in EBM, we provide a new method to re-rank and filter the medical evidence. This method presents multiple principles to filter the best evidence for LLMs to diagnose. We employ a combination of several EBM methods to emulate the meta-analysis, which includes reliability analysis, heterogeneity analysis, and extrapolation analysis. These processes allow the users to retrieve the best medical evidence for the LLMs. Ultimately, we evaluate these high-quality articles and show an accuracy improvement of up to 11.4% in our experiments and results. Our method successfully enables RAG to extract higher-quality and more reliable evidence from the PubMed dataset. This work can reduce the infusion of incorrect knowledge into responses and help users receive more effective replies.
>
---
#### [new 055] Text Simplification with Sentence Embeddings
- **分类: cs.CL**

- **简介: 该论文研究文本简化任务，旨在通过句子嵌入空间中的变换实现复杂文本到简单文本的映射。作者提出用小型前馈神经网络学习高、低复杂度文本嵌入间的转换，无需大规模模型，即可在多个语言和数据集上取得良好效果，验证了嵌入空间变换的有效性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.24365v1](http://arxiv.org/pdf/2510.24365v1)**

> **作者:** Matthew Shardlow
>
> **摘要:** Sentence embeddings can be decoded to give approximations of the original texts used to create them. We explore this effect in the context of text simplification, demonstrating that reconstructed text embeddings preserve complexity levels. We experiment with a small feed forward neural network to effectively learn a transformation between sentence embeddings representing high-complexity and low-complexity texts. We provide comparison to a Seq2Seq and LLM-based approach, showing encouraging results in our much smaller learning setting. Finally, we demonstrate the applicability of our transformation to an unseen simplification dataset (MedEASI), as well as datasets from languages outside the training data (ES,DE). We conclude that learning transformations in sentence embedding space is a promising direction for future research and has potential to unlock the ability to develop small, but powerful models for text simplification and other natural language generation tasks.
>
---
#### [new 056] Beyond Neural Incompatibility: Easing Cross-Scale Knowledge Transfer in Large Language Models through Latent Semantic Alignment
- **分类: cs.CL; cs.LG**

- **简介: 该论文聚焦于大语言模型间的细粒度知识迁移任务，针对不同规模模型因神经结构差异导致的参数不兼容问题，提出基于潜在语义对齐的跨尺度知识迁移方法。通过利用层间激活实现语义对齐，提升知识迁移效率与效果。**

- **链接: [http://arxiv.org/pdf/2510.24208v1](http://arxiv.org/pdf/2510.24208v1)**

> **作者:** Jian Gu; Aldeida Aleti; Chunyang Chen; Hongyu Zhang
>
> **备注:** an early-stage version
>
> **摘要:** Large Language Models (LLMs) encode vast amounts of knowledge in their massive parameters, which is accessible to locate, trace, and analyze. Despite advances in neural interpretability, it is still not clear how to transfer knowledge in a fine-grained manner, namely parametric knowledge transfer (PKT). A key problem is enabling effective and efficient knowledge transfer across LLMs of different scales, which is essential for achieving greater flexibility and broader applicability in transferring knowledge between LLMs. Due to neural incompatibility, referring to the architectural and parametric differences between LLMs of varying scales, existing methods that directly reuse layer parameters are severely limited. In this paper, we identify the semantic alignment in latent space as the fundamental prerequisite for LLM cross-scale knowledge transfer. Instead of directly using the layer parameters, our approach takes activations as the medium of layer-wise knowledge transfer. Leveraging the semantics in latent space, our approach is simple and outperforms prior work, better aligning model behaviors across varying scales. Evaluations on four benchmarks demonstrate the efficacy of our method. Further analysis reveals the key factors easing cross-scale knowledge transfer and provides insights into the nature of latent semantic alignment.
>
---
#### [new 057] Optimizing Retrieval for RAG via Reinforced Contrastive Learning
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对RAG中检索器难以定义与标注相关性的挑战，提出R3框架，通过强化对比学习实现检索器在RAG环境中的自适应优化。无需标注数据，利用交互反馈自动生成对比信号，动态提升检索效果，显著优于基线模型且训练高效。**

- **链接: [http://arxiv.org/pdf/2510.24652v1](http://arxiv.org/pdf/2510.24652v1)**

> **作者:** Jiawei Zhou; Lei Chen
>
> **摘要:** As retrieval-augmented generation (RAG) becomes increasingly widespread, the role of information retrieval (IR) is shifting from retrieving information for human users to retrieving contextual knowledge for artificial intelligence (AI) systems, where relevance becomes difficult to define or annotate beforehand. To address this challenge, we propose R3, a Retrieval framework optimized for RAG through trialand-feedback Reinforced contrastive learning. Unlike prior approaches that rely on annotated or synthetic data for supervised fine-tuning, R3 enables the retriever to dynamically explore and optimize relevance within the RAG environment. During training, the retrieved results interact with the environment to produce contrastive signals that automatically guide the retriever's self-improvement. Extensive experiments across diverse tasks demonstrate that R3 improves RAG performance by 5.2% over the original retriever and surpasses state-of-the-art retrievers by 4.9%, while achieving comparable results to LLM-augmented retrieval and RAG systems built on post-trained or instruction-tuned LLMs. It is both efficient and practical, requiring only 4 GPUs and completing training within a single day.
>
---
#### [new 058] Iterative Critique-Refine Framework for Enhancing LLM Personalization
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文针对个性化文本生成中风格、语气与主题漂移的问题，提出PerFine框架。通过迭代式批评-修正机制，利用用户画像提供结构化反馈，实现无需训练的个性化增强。实验表明，该方法在多个数据集上显著提升生成质量，且具高效性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2510.24469v1](http://arxiv.org/pdf/2510.24469v1)**

> **作者:** Durga Prasad Maram; Dhruvin Gandhi; Zonghai Yao; Gayathri Akkinapalli; Franck Dernoncourt; Yu Wang; Ryan A. Rossi; Nesreen K. Ahmed
>
> **摘要:** Personalized text generation requires models not only to produce coherent text but also to align with a target user's style, tone, and topical focus. Existing retrieval-augmented approaches such as LaMP and PGraphRAG enrich profiles with user and neighbor histories, but they stop at generation and often yield outputs that drift in tone, topic, or style. We present PerFine, a unified, training-free critique-refine framework that enhances personalization through iterative, profile-grounded feedback. In each iteration, an LLM generator produces a draft conditioned on the retrieved profile, and a critic LLM - also conditioned on the same profile - provides structured feedback on tone, vocabulary, sentence structure, and topicality. The generator then revises, while a novel knockout strategy retains the stronger draft across iterations. We further study additional inference-time strategies such as Best-of-N and Topic Extraction to balance quality and efficiency. Across Yelp, Goodreads, and Amazon datasets, PerFine consistently improves personalization over PGraphRAG, with GEval gains of +7-13%, steady improvements over 3-5 refinement iterations, and scalability with increasing critic size. These results highlight that post-hoc, profile-aware feedback offers a powerful paradigm for personalized LLM generation that is both training-free and model-agnostic.
>
---
#### [new 059] HACK: Hallucinations Along Certainty and Knowledge Axes
- **分类: cs.CL; I.2.7**

- **简介: 该论文针对大模型幻觉问题，提出基于“知识”与“ certainty”双轴的分类框架。通过构建模型特定数据集，区分因缺知与有知却仍幻觉的类型，并验证其差异；发现部分方法在高置信度幻觉上失效，强调需针对性缓解策略。属于自然语言处理中的幻觉分析与治理任务。**

- **链接: [http://arxiv.org/pdf/2510.24222v1](http://arxiv.org/pdf/2510.24222v1)**

> **作者:** Adi Simhi; Jonathan Herzig; Itay Itzhak; Dana Arad; Zorik Gekhman; Roi Reichart; Fazl Barez; Gabriel Stanovsky; Idan Szpektor; Yonatan Belinkov
>
> **备注:** The code is available at https://github.com/technion-cs-nlp/HACK_Hallucinations_Along_Certainty_and_Knowledge_axes
>
> **摘要:** Hallucinations in LLMs present a critical barrier to their reliable usage. Existing research usually categorizes hallucination by their external properties rather than by the LLMs' underlying internal properties. This external focus overlooks that hallucinations may require tailored mitigation strategies based on their underlying mechanism. We propose a framework for categorizing hallucinations along two axes: knowledge and certainty. Since parametric knowledge and certainty may vary across models, our categorization method involves a model-specific dataset construction process that differentiates between those types of hallucinations. Along the knowledge axis, we distinguish between hallucinations caused by a lack of knowledge and those occurring despite the model having the knowledge of the correct response. To validate our framework along the knowledge axis, we apply steering mitigation, which relies on the existence of parametric knowledge to manipulate model activations. This addresses the lack of existing methods to validate knowledge categorization by showing a significant difference between the two hallucination types. We further analyze the distinct knowledge and hallucination patterns between models, showing that different hallucinations do occur despite shared parametric knowledge. Turning to the certainty axis, we identify a particularly concerning subset of hallucinations where models hallucinate with certainty despite having the correct knowledge internally. We introduce a new evaluation metric to measure the effectiveness of mitigation methods on this subset, revealing that while some methods perform well on average, they fail disproportionately on these critical cases. Our findings highlight the importance of considering both knowledge and certainty in hallucination analysis and call for targeted mitigation approaches that consider the hallucination underlying factors.
>
---
#### [new 060] Pie: A Programmable Serving System for Emerging LLM Applications
- **分类: cs.CL**

- **简介: 该论文提出Pie，一个面向新兴大语言模型应用的可编程推理服务系统。针对现有系统在复杂推理与智能体工作流下灵活性不足的问题，Pie将生成流程拆分为细粒度服务，通过WebAssembly运行用户自定义的inferlets，实现缓存策略、生成逻辑等的灵活定制，显著提升智能体任务的性能。**

- **链接: [http://arxiv.org/pdf/2510.24051v1](http://arxiv.org/pdf/2510.24051v1)**

> **作者:** In Gim; Zhiyao Ma; Seung-seob Lee; Lin Zhong
>
> **备注:** SOSP 2025. Source code available at https://github.com/pie-project/pie
>
> **摘要:** Emerging large language model (LLM) applications involve diverse reasoning strategies and agentic workflows, straining the capabilities of existing serving systems built on a monolithic token generation loop. This paper introduces Pie, a programmable LLM serving system designed for flexibility and efficiency. Pie decomposes the traditional generation loop into fine-grained service handlers exposed via an API and delegates control of the generation process to user-provided programs, called inferlets. This enables applications to implement new KV cache strategies, bespoke generation logic, and seamlessly integrate computation and I/O-entirely within the application, without requiring modifications to the serving system. Pie executes inferlets using WebAssembly, benefiting from its lightweight sandboxing. Our evaluation shows Pie matches state-of-the-art performance on standard tasks (3-12% latency overhead) while significantly improving latency and throughput (1.3x-3.4x higher) on agentic workflows by enabling application-specific optimizations.
>
---
#### [new 061] MetricX-25 and GemSpanEval: Google Translate Submissions to the WMT25 Evaluation Shared Task
- **分类: cs.CL**

- **简介: 该论文参与WMT25翻译评估共享任务，针对质量评分预测与错误片段检测两个子任务。提出MetricX-25（基于Gemma 3的编码器模型）和GemSpanEval（解码器模型），分别实现高精度质量评分预测与带上下文的错误片段生成式检测，显著优于前代方法。**

- **链接: [http://arxiv.org/pdf/2510.24707v1](http://arxiv.org/pdf/2510.24707v1)**

> **作者:** Juraj Juraska; Tobias Domhan; Mara Finkelstein; Tetsuji Nakagawa; Geza Kovacs; Daniel Deutsch; Pidong Wang; Markus Freitag
>
> **备注:** Accepted to WMT25
>
> **摘要:** In this paper, we present our submissions to the unified WMT25 Translation Evaluation Shared Task. For the Quality Score Prediction subtask, we create a new generation of MetricX with improvements in the input format and the training protocol, while for the Error Span Detection subtask we develop a new model, GemSpanEval, trained to predict error spans along with their severities and categories. Both systems are based on the state-of-the-art multilingual open-weights model Gemma 3, fine-tuned on publicly available WMT data. We demonstrate that MetricX-25, adapting Gemma 3 to an encoder-only architecture with a regression head on top, can be trained to effectively predict both MQM and ESA quality scores, and significantly outperforms its predecessor. Our decoder-only GemSpanEval model, on the other hand, we show to be competitive in error span detection with xCOMET, a strong encoder-only sequence-tagging baseline. With error span detection formulated as a generative task, we instruct the model to also output the context for each predicted error span, thus ensuring that error spans are identified unambiguously.
>
---
#### [new 062] AgentFold: Long-Horizon Web Agents with Proactive Context Management
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对长周期网页任务中上下文管理难题，提出AgentFold框架。通过模拟人类回顾性整合认知机制，实现多尺度主动上下文折叠，兼顾细节保留与全局抽象。实验表明，其在主流基准上表现优异，超越多个更大规模开源及领先闭源模型。**

- **链接: [http://arxiv.org/pdf/2510.24699v1](http://arxiv.org/pdf/2510.24699v1)**

> **作者:** Rui Ye; Zhongwang Zhang; Kuan Li; Huifeng Yin; Zhengwei Tao; Yida Zhao; Liangcai Su; Liwen Zhang; Zile Qiao; Xinyu Wang; Pengjun Xie; Fei Huang; Siheng Chen; Jingren Zhou; Yong Jiang
>
> **备注:** 26 pages, 9 figures
>
> **摘要:** LLM-based web agents show immense promise for information seeking, yet their effectiveness on long-horizon tasks is hindered by a fundamental trade-off in context management. Prevailing ReAct-based agents suffer from context saturation as they accumulate noisy, raw histories, while methods that fixedly summarize the full history at each step risk the irreversible loss of critical details. Addressing these, we introduce AgentFold, a novel agent paradigm centered on proactive context management, inspired by the human cognitive process of retrospective consolidation. AgentFold treats its context as a dynamic cognitive workspace to be actively sculpted, rather than a passive log to be filled. At each step, it learns to execute a `folding' operation, which manages its historical trajectory at multiple scales: it can perform granular condensations to preserve vital, fine-grained details, or deep consolidations to abstract away entire multi-step sub-tasks. The results on prominent benchmarks are striking: with simple supervised fine-tuning (without continual pre-training or RL), our AgentFold-30B-A3B agent achieves 36.2% on BrowseComp and 47.3% on BrowseComp-ZH. Notably, this performance not only surpasses or matches open-source models of a dramatically larger scale, such as the DeepSeek-V3.1-671B-A37B, but also surpasses leading proprietary agents like OpenAI's o4-mini.
>
---
#### [new 063] WebLeaper: Empowering Efficiency and Efficacy in WebAgent via Enabling Info-Rich Seeking
- **分类: cs.CL**

- **简介: 该论文针对大语言模型网页代理中的信息搜索效率低问题，提出WebLeaper框架。通过构建高覆盖任务与高效解题轨迹，将信息搜索建模为树状推理，利用维基百科表格生成三类任务，筛选精准高效路径，显著提升搜索效率与效果，在多个基准上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24697v1](http://arxiv.org/pdf/2510.24697v1)**

> **作者:** Zhengwei Tao; Haiyang Shen; Baixuan Li; Wenbiao Yin; Jialong Wu; Kuan Li; Zhongwang Zhang; Huifeng Yin; Rui Ye; Liwen Zhang; Xinyu Wang; Pengjun Xie; Jingren Zhou; Yong Jiang
>
> **摘要:** Large Language Model (LLM)-based agents have emerged as a transformative approach for open-ended problem solving, with information seeking (IS) being a core capability that enables autonomous reasoning and decision-making. While prior research has largely focused on improving retrieval depth, we observe that current IS agents often suffer from low search efficiency, which in turn constrains overall performance. A key factor underlying this inefficiency is the sparsity of target entities in training tasks, which limits opportunities for agents to learn and generalize efficient search behaviors. To address these challenges, we propose WebLeaper, a framework for constructing high-coverage IS tasks and generating efficient solution trajectories. We formulate IS as a tree-structured reasoning problem, enabling a substantially larger set of target entities to be embedded within a constrained context. Leveraging curated Wikipedia tables, we propose three variants for synthesizing IS tasks, Basic, Union, and Reverse-Union, to systematically increase both IS efficiency and efficacy. Finally, we curate training trajectories by retaining only those that are simultaneously accurate and efficient, ensuring that the model is optimized for both correctness and search performance. Extensive experiments on both basic and comprehensive settings, conducted on five IS benchmarks, BrowserComp, GAIA, xbench-DeepSearch, WideSearch, and Seal-0, demonstrate that our method consistently achieves improvements in both effectiveness and efficiency over strong baselines.
>
---
#### [new 064] RegSpeech12: A Regional Corpus of Bengali Spontaneous Speech Across Dialects
- **分类: cs.CL**

- **简介: 该论文聚焦于孟加拉语方言的自发语音建模，旨在解决方言多样性导致的自动语音识别（ASR）系统性能下降问题。研究构建了跨方言的区域语料库RegSpeech12，涵盖五种主要方言及孟加拉国多个地区，分析其语音与形态特征，推动面向方言的包容性语音技术发展。**

- **链接: [http://arxiv.org/pdf/2510.24096v1](http://arxiv.org/pdf/2510.24096v1)**

> **作者:** Md. Rezuwan Hassan; Azmol Hossain; Kanij Fatema; Rubayet Sabbir Faruque; Tanmoy Shome; Ruwad Naswan; Trina Chakraborty; Md. Foriduzzaman Zihad; Tawsif Tashwar Dipto; Nazia Tasnim; Nazmuddoha Ansary; Md. Mehedi Hasan Shawon; Ahmed Imtiaz Humayun; Md. Golam Rabiul Alam; Farig Sadeque; Asif Sushmit
>
> **备注:** 26 pages
>
> **摘要:** The Bengali language, spoken extensively across South Asia and among diasporic communities, exhibits considerable dialectal diversity shaped by geography, culture, and history. Phonological and pronunciation-based classifications broadly identify five principal dialect groups: Eastern Bengali, Manbhumi, Rangpuri, Varendri, and Rarhi. Within Bangladesh, further distinctions emerge through variation in vocabulary, syntax, and morphology, as observed in regions such as Chittagong, Sylhet, Rangpur, Rajshahi, Noakhali, and Barishal. Despite this linguistic richness, systematic research on the computational processing of Bengali dialects remains limited. This study seeks to document and analyze the phonetic and morphological properties of these dialects while exploring the feasibility of building computational models particularly Automatic Speech Recognition (ASR) systems tailored to regional varieties. Such efforts hold potential for applications in virtual assistants and broader language technologies, contributing to both the preservation of dialectal diversity and the advancement of inclusive digital tools for Bengali-speaking communities. The dataset created for this study is released for public use.
>
---
#### [new 065] Comprehensive and Efficient Distillation for Lightweight Sentiment Analysis Models
- **分类: cs.CL**

- **简介: 该论文针对轻量级情感分析模型的训练效率与知识覆盖问题，提出COMPEFFDIST框架。通过自动构建多样化指令和基于难度的数据筛选，提升知识蒸馏的全面性与效率，使3B模型性能媲美20倍大的教师模型，且仅需10%数据即可达到相同效果。**

- **链接: [http://arxiv.org/pdf/2510.24425v1](http://arxiv.org/pdf/2510.24425v1)**

> **作者:** Guangyu Xie; Yice Zhang; Jianzhu Bao; Qianlong Wang; Yang Sun; Bingbing Wang; Ruifeng Xu
>
> **备注:** Accepted by EMNLP 2025. 22 pages, 9 figures. The first two authors contribute equally
>
> **摘要:** Recent efforts leverage knowledge distillation techniques to develop lightweight and practical sentiment analysis models. These methods are grounded in human-written instructions and large-scale user texts. Despite the promising results, two key challenges remain: (1) manually written instructions are limited in diversity and quantity, making them insufficient to ensure comprehensive coverage of distilled knowledge; (2) large-scale user texts incur high computational cost, hindering the practicality of these methods. To this end, we introduce COMPEFFDIST, a comprehensive and efficient distillation framework for sentiment analysis. Our framework consists of two key modules: attribute-based automatic instruction construction and difficulty-based data filtering, which correspondingly tackle the aforementioned challenges. Applying our method across multiple model series (Llama-3, Qwen-3, and Gemma-3), we enable 3B student models to match the performance of 20x larger teacher models on most tasks. In addition, our approach greatly outperforms baseline methods in data efficiency, attaining the same performance level with only 10% of the data.
>
---
#### [new 066] AgentFrontier: Expanding the Capability Frontier of LLM Agents with ZPD-Guided Data Synthesis
- **分类: cs.CL**

- **简介: 该论文提出基于最近发展区（ZPD）理论的数据合成方法，构建AgentFrontier引擎，生成位于大模型能力边界上的高质量训练数据。通过持续预训练与定向微调，提升模型推理能力，并设计动态评估基准ZPD Exam。实验表明，所提模型在前沿任务上达到先进水平。**

- **链接: [http://arxiv.org/pdf/2510.24695v1](http://arxiv.org/pdf/2510.24695v1)**

> **作者:** Xuanzhong Chen; Zile Qiao; Guoxin Chen; Liangcai Su; Zhen Zhang; Xinyu Wang; Pengjun Xie; Fei Huang; Jingren Zhou; Yong Jiang
>
> **备注:** https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/
>
> **摘要:** Training large language model agents on tasks at the frontier of their capabilities is key to unlocking advanced reasoning. We introduce a data synthesis approach inspired by the educational theory of the Zone of Proximal Development (ZPD), which defines this frontier as tasks an LLM cannot solve alone but can master with guidance. To operationalize this, we present the AgentFrontier Engine, an automated pipeline that synthesizes high-quality, multidisciplinary data situated precisely within the LLM's ZPD. This engine supports both continued pre-training with knowledge-intensive data and targeted post-training on complex reasoning tasks. From the same framework, we derive the ZPD Exam, a dynamic and automated benchmark designed to evaluate agent capabilities on these frontier tasks. We train AgentFrontier-30B-A3B model on our synthesized data, which achieves state-of-the-art results on demanding benchmarks like Humanity's Last Exam, even surpassing some leading proprietary agents. Our work demonstrates that a ZPD-guided approach to data synthesis offers a scalable and effective path toward building more capable LLM agents.
>
---
#### [new 067] Evaluating Long-Term Memory for Long-Context Question Answering
- **分类: cs.CL**

- **简介: 该论文聚焦长上下文问答任务，旨在评估不同记忆机制对大模型长期对话能力的影响。通过构建LoCoMo基准，系统比较了全上下文提示、检索增强生成、代理记忆、情景记忆和过程记忆等方法，发现记忆增强可显著降低90%以上令牌消耗，且记忆架构应随模型能力调整。**

- **链接: [http://arxiv.org/pdf/2510.23730v1](http://arxiv.org/pdf/2510.23730v1)**

> **作者:** Alessandra Terranova; Björn Ross; Alexandra Birch
>
> **备注:** 14 pages including appendix, 3 figures. Submitted to October ARR and to Metacognition in Generative AI EurIPS workshop (under review for both)
>
> **摘要:** In order for large language models to achieve true conversational continuity and benefit from experiential learning, they need memory. While research has focused on the development of complex memory systems, it remains unclear which types of memory are most effective for long-context conversational tasks. We present a systematic evaluation of memory-augmented methods using LoCoMo, a benchmark of synthetic long-context dialogues annotated for question-answering tasks that require diverse reasoning strategies. We analyse full-context prompting, semantic memory through retrieval-augmented generation and agentic memory, episodic memory through in-context learning, and procedural memory through prompt optimization. Our findings show that memory-augmented approaches reduce token usage by over 90% while maintaining competitive accuracy. Memory architecture complexity should scale with model capability, with small foundation models benefitting most from RAG, and strong instruction-tuned reasoning model gaining from episodic learning through reflections and more complex agentic semantic memory. In particular, episodic memory can help LLMs recognise the limits of their own knowledge.
>
---
#### [new 068] Dark & Stormy: Modeling Humor in the Worst Sentences Ever Written
- **分类: cs.CL**

- **简介: 该论文聚焦于“糟糕幽默”文本的建模，属于自然语言处理中的幽默理解任务。针对现有模型在识别故意低质量幽默上表现差的问题，研究构建了来自巴尔沃-利顿小说竞赛的语料库，分析其融合比喻、元小说等修辞手法的特点，并发现大模型虽能模仿形式，却过度使用特定修辞且生成更多新颖词组。**

- **链接: [http://arxiv.org/pdf/2510.24538v1](http://arxiv.org/pdf/2510.24538v1)**

> **作者:** Venkata S Govindarajan; Laura Biester
>
> **摘要:** Textual humor is enormously diverse and computational studies need to account for this range, including intentionally bad humor. In this paper, we curate and analyze a novel corpus of sentences from the Bulwer-Lytton Fiction Contest to better understand "bad" humor in English. Standard humor detection models perform poorly on our corpus, and an analysis of literary devices finds that these sentences combine features common in existing humor datasets (e.g., puns, irony) with metaphor, metafiction and simile. LLMs prompted to synthesize contest-style sentences imitate the form but exaggerate the effect by over-using certain literary devices, and including far more novel adjective-noun bigrams than human writers. Data, code and analysis are available at https://github.com/venkatasg/bulwer-lytton
>
---
#### [new 069] Dissecting Role Cognition in Medical LLMs via Neuronal Ablation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究医疗大模型在角色扮演中的认知机制，旨在解决角色提示是否引发真实认知差异的问题。通过构建RPNA框架，结合神经元消融与表征分析，在三个医学问答数据集上验证发现：角色提示仅影响语言风格，未改变核心推理路径，表明当前角色扮演无法模拟真实医疗认知。**

- **链接: [http://arxiv.org/pdf/2510.24677v1](http://arxiv.org/pdf/2510.24677v1)**

> **作者:** Xun Liang; Huayi Lai; Hanyu Wang; Wentao Zhang; Linfeng Zhang; Yanfang Chen; Feiyu Xiong; Zhiyu Li
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Large language models (LLMs) have gained significant traction in medical decision support systems, particularly in the context of medical question answering and role-playing simulations. A common practice, Prompt-Based Role Playing (PBRP), instructs models to adopt different clinical roles (e.g., medical students, residents, attending physicians) to simulate varied professional behaviors. However, the impact of such role prompts on model reasoning capabilities remains unclear. This study introduces the RP-Neuron-Activated Evaluation Framework(RPNA) to evaluate whether role prompts induce distinct, role-specific cognitive processes in LLMs or merely modify linguistic style. We test this framework on three medical QA datasets, employing neuron ablation and representation analysis techniques to assess changes in reasoning pathways. Our results demonstrate that role prompts do not significantly enhance the medical reasoning abilities of LLMs. Instead, they primarily affect surface-level linguistic features, with no evidence of distinct reasoning pathways or cognitive differentiation across clinical roles. Despite superficial stylistic changes, the core decision-making mechanisms of LLMs remain uniform across roles, indicating that current PBRP methods fail to replicate the cognitive complexity found in real-world medical practice. This highlights the limitations of role-playing in medical AI and emphasizes the need for models that simulate genuine cognitive processes rather than linguistic imitation.We have released the related code in the following repository:https: //github.com/IAAR-Shanghai/RolePlay_LLMDoctor
>
---
#### [new 070] OraPlan-SQL: A Planning-Centric Framework for Complex Bilingual NL2SQL Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对复杂双语NL2SQL任务，提出OraPlan-SQL框架。通过单规划器结合反馈引导的元提示策略提升规划质量，引入实体链接与计划多样化机制，显著提高跨语言推理准确率与SQL有效性，获2025年Archer挑战赛第一名。**

- **链接: [http://arxiv.org/pdf/2510.23870v1](http://arxiv.org/pdf/2510.23870v1)**

> **作者:** Marianne Menglin Liu; Sai Ashish Somayajula; Syed Fahad Allam Shah; Sujith Ravi; Dan Roth
>
> **摘要:** We present OraPlan-SQL, our system for the Archer NL2SQL Evaluation Challenge 2025, a bilingual benchmark requiring complex reasoning such as arithmetic, commonsense, and hypothetical inference. OraPlan-SQL ranked first, exceeding the second-best system by more than 6% in execution accuracy (EX), with 55.0% in English and 56.7% in Chinese, while maintaining over 99% SQL validity (VA). Our system follows an agentic framework with two components: Planner agent that generates stepwise natural language plans, and SQL agent that converts these plans into executable SQL. Since SQL agent reliably adheres to the plan, our refinements focus on the planner. Unlike prior methods that rely on multiple sub-agents for planning and suffer from orchestration overhead, we introduce a feedback-guided meta-prompting strategy to refine a single planner. Failure cases from a held-out set are clustered with human input, and an LLM distills them into corrective guidelines that are integrated into the planner's system prompt, improving generalization without added complexity. For the multilingual scenario, to address transliteration and entity mismatch issues, we incorporate entity-linking guidelines that generate alternative surface forms for entities and explicitly include them in the plan. Finally, we enhance reliability through plan diversification: multiple candidate plans are generated for each query, with the SQL agent producing a query for each plan, and final output selected via majority voting over their executions.
>
---
#### [new 071] Success and Cost Elicit Convention Formation for Efficient Communication
- **分类: cs.CL**

- **简介: 该论文研究多智能体对话中的临时语言惯例形成任务，旨在提升模型间高效沟通能力。通过模拟参考游戏，训练大模型在无额外人类数据下自发形成简洁、高效的通信惯例，同时优化成功率与成本，实现比单一目标训练更优的沟通效率。**

- **链接: [http://arxiv.org/pdf/2510.24023v1](http://arxiv.org/pdf/2510.24023v1)**

> **作者:** Saujas Vaduguru; Yilun Hua; Yoav Artzi; Daniel Fried
>
> **摘要:** Humans leverage shared conversational context to become increasingly successful and efficient at communicating over time. One manifestation of this is the formation of ad hoc linguistic conventions, which allow people to coordinate on short, less costly utterances that are understood using shared conversational context. We present a method to train large multimodal models to form conventions, enabling efficient communication. Our approach uses simulated reference games between models, and requires no additional human-produced data. In repeated reference games involving photographs and tangram images, our method enables models to communicate efficiently with people: reducing the message length by up to 41% while increasing success by 15% over the course of the interaction. Human listeners respond faster when interacting with our model that forms conventions. We also show that training based on success or cost alone is insufficient - both are necessary to elicit convention formation.
>
---
#### [new 072] Mitigating Hallucination in Large Language Models (LLMs): An Application-Oriented Survey on RAG, Reasoning, and Agentic Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于大语言模型幻觉问题，属于自然语言处理中的可信生成任务。针对幻觉抑制难题，提出基于知识与逻辑的分类框架，系统分析RAG、推理增强及智能体系统在真实应用中协同缓解幻觉的机制，构建统一评估体系。**

- **链接: [http://arxiv.org/pdf/2510.24476v1](http://arxiv.org/pdf/2510.24476v1)**

> **作者:** Yihan Li; Xiyuan Fu; Ghanshyam Verma; Paul Buitelaar; Mingming Liu
>
> **备注:** 25 pages, 7 figures, 3 tables
>
> **摘要:** Hallucination remains one of the key obstacles to the reliable deployment of large language models (LLMs), particularly in real-world applications. Among various mitigation strategies, Retrieval-Augmented Generation (RAG) and reasoning enhancement have emerged as two of the most effective and widely adopted approaches, marking a shift from merely suppressing hallucinations to balancing creativity and reliability. However, their synergistic potential and underlying mechanisms for hallucination mitigation have not yet been systematically examined. This survey adopts an application-oriented perspective of capability enhancement to analyze how RAG, reasoning enhancement, and their integration in Agentic Systems mitigate hallucinations. We propose a taxonomy distinguishing knowledge-based and logic-based hallucinations, systematically examine how RAG and reasoning address each, and present a unified framework supported by real-world applications, evaluations, and benchmarks.
>
---
#### [new 073] Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content
- **分类: cs.CL; cs.AI; cs.CY; cs.MA**

- **简介: 该论文评估大语言模型生成伊斯兰内容的准确性，针对其在引文、教义一致性等方面的可靠性问题。采用双代理框架，结合定量与定性分析，对比GPT-4o、Ansari AI和Fanar在真实博客提问下的表现，发现模型仍存不足，呼吁建立以穆斯林视角为核心的社区基准。**

- **链接: [http://arxiv.org/pdf/2510.24438v1](http://arxiv.org/pdf/2510.24438v1)**

> **作者:** Abdullah Mushtaq; Rafay Naeem; Ezieddin Elmahjub; Ibrahim Ghaznavi; Shawqi Al-Maliki; Mohamed Abdallah; Ala Al-Fuqaha; Junaid Qadir
>
> **备注:** Accepted at 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: 5th Muslims in Machine Learning (MusIML) Workshop
>
> **摘要:** Large language models are increasingly used for Islamic guidance, but risk misquoting texts, misapplying jurisprudence, or producing culturally inconsistent responses. We pilot an evaluation of GPT-4o, Ansari AI, and Fanar on prompts from authentic Islamic blogs. Our dual-agent framework uses a quantitative agent for citation verification and six-dimensional scoring (e.g., Structure, Islamic Consistency, Citations) and a qualitative agent for five-dimensional side-by-side comparison (e.g., Tone, Depth, Originality). GPT-4o scored highest in Islamic Accuracy (3.93) and Citation (3.38), Ansari AI followed (3.68, 3.32), and Fanar lagged (2.76, 1.82). Despite relatively strong performance, models still fall short in reliably producing accurate Islamic content and citations -- a paramount requirement in faith-sensitive writing. GPT-4o had the highest mean quantitative score (3.90/5), while Ansari AI led qualitative pairwise wins (116/200). Fanar, though trailing, introduces innovations for Islamic and Arabic contexts. This study underscores the need for community-driven benchmarks centering Muslim perspectives, offering an early step toward more reliable AI in Islamic knowledge and other high-stakes domains such as medicine, law, and journalism.
>
---
#### [new 074] Abjad AI at NADI 2025: CATT-Whisper: Multimodal Diacritic Restoration Using Text and Speech Representations
- **分类: cs.CL**

- **简介: 该论文针对阿拉伯语方言的重音符号恢复任务，提出CATT-Whisper模型，融合自研文本编码器CATT与Whisper语音编码器。通过早期融合与交叉注意力机制整合文本与语音信息，并引入随机屏蔽语音增强鲁棒性。实验表明模型在测试集上达到WER 0.55、CER 0.13，有效提升恢复精度。**

- **链接: [http://arxiv.org/pdf/2510.24247v1](http://arxiv.org/pdf/2510.24247v1)**

> **作者:** Ahmad Ghannam; Naif Alharthi; Faris Alasmary; Kholood Al Tabash; Shouq Sadah; Lahouari Ghouti
>
> **摘要:** In this work, we tackle the Diacritic Restoration (DR) task for Arabic dialectal sentences using a multimodal approach that combines both textual and speech information. We propose a model that represents the text modality using an encoder extracted from our own pre-trained model named CATT. The speech component is handled by the encoder module of the OpenAI Whisper base model. Our solution is designed following two integration strategies. The former consists of fusing the speech tokens with the input at an early stage, where the 1500 frames of the audio segment are averaged over 10 consecutive frames, resulting in 150 speech tokens. To ensure embedding compatibility, these averaged tokens are processed through a linear projection layer prior to merging them with the text tokens. Contextual encoding is guaranteed by the CATT encoder module. The latter strategy relies on cross-attention, where text and speech embeddings are fused. The cross-attention output is then fed to the CATT classification head for token-level diacritic prediction. To further improve model robustness, we randomly deactivate the speech input during training, allowing the model to perform well with or without speech. Our experiments show that the proposed approach achieves a word error rate (WER) of 0.25 and a character error rate (CER) of 0.9 on the development set. On the test set, our model achieved WER and CER scores of 0.55 and 0.13, respectively.
>
---
#### [new 075] Temporal Blindness in Multi-Turn LLM Agents: Misaligned Tool Use vs. Human Time Perception
- **分类: cs.CL**

- **简介: 该论文研究多轮对话中大语言模型（LLM）的“时间盲区”问题，即模型因缺乏时间感知导致工具调用决策失误。为解决此问题，作者构建了TicToc-v1测试集，引入显式时间戳并基于人类偏好标注数据，评估模型在不同时间间隔下的工具调用准确性。结果表明，现有模型表现有限，需专门对齐训练以匹配人类时间认知。**

- **链接: [http://arxiv.org/pdf/2510.23853v1](http://arxiv.org/pdf/2510.23853v1)**

> **作者:** Yize Cheng; Arshia Soltani Moakhar; Chenrui Fan; Kazem Faghih; Parsa Hosseini; Wenxiao Wang; Soheil Feizi
>
> **备注:** preliminary work in progress
>
> **摘要:** Large language model agents are increasingly used in multi-turn conversational settings to interact with and execute tasks in dynamic environments. However, a key limitation is their temporal blindness: they, by default, operate with a stationary context, failing to account for the real-world time elapsed between messages. This becomes a critical liability when an agent must decide whether to invoke a tool based on how much time has passed since the last observation. Without temporal awareness, agents often either over-rely on previous context (skipping necessary tool calls), or under-rely on it (unnecessarily repeating tool calls). To study this challenge, we introduce TicToc-v1, a test set of multi-turn user-agent trajectories across 34 scenarios with varying time sensitivity. Each trajectory ends with a user question, where the need for a tool call depends on the amount of time elapsed since the last message. To give LLMs temporal context, we augment dialogue messages with explicit timestamps, bridging the gap between static dialogue and evolving environments. We then collected human preferences for these samples, creating two subsets: one where humans preferred relying on the previous observation (prefer-noTool), and another where they preferred a new tool call (prefer-Tool). We evaluated how well LLM tool-calling decisions align with human preferences under varying time intervals on TicToc-v1. Our analysis show that without time information, most models perform only slightly better than random, with the top alignment rate being just over 60%. While adding timestamps leads to a slight improvement, particularly for larger models, the improvement is modest, peaking at around 65%. We also show that naive, prompt-based alignment have limited effectiveness. Our findings highlight the need for specific post-training alignment to align multi-turn LLM tool use with human temporal perception.
>
---
#### [new 076] MQM Re-Annotation: A Technique for Collaborative Evaluation of Machine Translation
- **分类: cs.CL**

- **简介: 该论文针对机器翻译评估中评价质量随模型提升而被噪声稀释的问题，提出MQM重标注技术。通过让标注员复审并修正已有标注，提升标注准确性，主要发现遗漏错误被有效纠正，从而实现更高质量的协同评估。**

- **链接: [http://arxiv.org/pdf/2510.24664v1](http://arxiv.org/pdf/2510.24664v1)**

> **作者:** Parker Riley; Daniel Deutsch; Mara Finkelstein; Colten DiIanni; Juraj Juraska; Markus Freitag
>
> **摘要:** Human evaluation of machine translation is in an arms race with translation model quality: as our models get better, our evaluation methods need to be improved to ensure that quality gains are not lost in evaluation noise. To this end, we experiment with a two-stage version of the current state-of-the-art translation evaluation paradigm (MQM), which we call MQM re-annotation. In this setup, an MQM annotator reviews and edits a set of pre-existing MQM annotations, that may have come from themselves, another human annotator, or an automatic MQM annotation system. We demonstrate that rater behavior in re-annotation aligns with our goals, and that re-annotation results in higher-quality annotations, mostly due to finding errors that were missed during the first pass.
>
---
#### [new 077] Towards Transparent Reasoning: What Drives Faithfulness in Large Language Models?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型解释的可信性问题，聚焦医疗等敏感领域中模型推理与训练策略对解释忠实性的影响。通过控制少样本示例、提示设计和微调方式，发现示例质量与数量、提示设计及指令微调显著影响模型解释的可信度，为提升模型可解释性提供实证依据。**

- **链接: [http://arxiv.org/pdf/2510.24236v1](http://arxiv.org/pdf/2510.24236v1)**

> **作者:** Teague McMillan; Gabriele Dominici; Martin Gjoreski; Marc Langheinrich
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
>
> **摘要:** Large Language Models (LLMs) often produce explanations that do not faithfully reflect the factors driving their predictions. In healthcare settings, such unfaithfulness is especially problematic: explanations that omit salient clinical cues or mask spurious shortcuts can undermine clinician trust and lead to unsafe decision support. We study how inference and training-time choices shape explanation faithfulness, focusing on factors practitioners can control at deployment. We evaluate three LLMs (GPT-4.1-mini, LLaMA 70B, LLaMA 8B) on two datasets-BBQ (social bias) and MedQA (medical licensing questions), and manipulate the number and type of few-shot examples, prompting strategies, and training procedure. Our results show: (i) both the quantity and quality of few-shot examples significantly impact model faithfulness; (ii) faithfulness is sensitive to prompting design; (iii) the instruction-tuning phase improves measured faithfulness on MedQA. These findings offer insights into strategies for enhancing the interpretability and trustworthiness of LLMs in sensitive domains.
>
---
#### [new 078] ReplicationBench: Can AI Agents Replicate Astrophysics Research Papers?
- **分类: cs.CL; astro-ph.IM**

- **简介: 该论文提出ReplicationBench，用于评估AI agents复制天体物理学论文的能力。针对当前AI在科学研究中可靠性不足的问题，构建了由专家验证的、覆盖论文全流程的任务框架，测试其方法忠实性与结果正确性。实验表明顶尖模型表现仍低于20%，揭示了多种失败模式，为衡量AI在数据驱动科学中的可信度提供了可扩展基准。**

- **链接: [http://arxiv.org/pdf/2510.24591v1](http://arxiv.org/pdf/2510.24591v1)**

> **作者:** Christine Ye; Sihan Yuan; Suchetha Cooray; Steven Dillmann; Ian L. V. Roque; Dalya Baron; Philipp Frank; Sergio Martin-Alvarez; Nolan Koblischke; Frank J Qu; Diyi Yang; Risa Wechsler; Ioana Ciuca
>
> **摘要:** Frontier AI agents show increasing promise as scientific research assistants, and may eventually be useful for extended, open-ended research workflows. However, in order to use agents for novel research, we must first assess the underlying faithfulness and correctness of their work. To evaluate agents as research assistants, we introduce ReplicationBench, an evaluation framework that tests whether agents can replicate entire research papers drawn from the astrophysics literature. Astrophysics, where research relies heavily on archival data and computational study while requiring little real-world experimentation, is a particularly useful testbed for AI agents in scientific research. We split each paper into tasks which require agents to replicate the paper's core contributions, including the experimental setup, derivations, data analysis, and codebase. Each task is co-developed with the original paper authors and targets a key scientific result, enabling objective evaluation of both faithfulness (adherence to original methods) and correctness (technical accuracy of results). ReplicationBench is extremely challenging for current frontier language models: even the best-performing language models score under 20%. We analyze ReplicationBench trajectories in collaboration with domain experts and find a rich, diverse set of failure modes for agents in scientific research. ReplicationBench establishes the first benchmark of paper-scale, expert-validated astrophysics research tasks, reveals insights about agent performance generalizable to other domains of data-driven science, and provides a scalable framework for measuring AI agents' reliability in scientific research.
>
---
#### [new 079] Critique-RL: Training Language Models for Critiquing through Two-Stage Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对语言模型评述任务，提出基于两阶段强化学习的Critique-RL方法，解决无强监督下批判性反馈模型训练难题。通过先提升评述者判别力，再优化帮助性并保持判别力，显著提升模型在跨域任务上的表现。**

- **链接: [http://arxiv.org/pdf/2510.24320v1](http://arxiv.org/pdf/2510.24320v1)**

> **作者:** Zhiheng Xi; Jixuan Huang; Xin Guo; Boyang Hong; Dingwen Yang; Xiaoran Fan; Shuo Li; Zehui Chen; Junjie Ye; Siyu Yuan; Zhengyin Du; Xuesong Yao; Yufei Xu; Jiecao Chen; Rui Zheng; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **备注:** Preprint, 25 pages, 9 figures. Code: https://github.com/WooooDyy/Critique-RL
>
> **摘要:** Training critiquing language models to assess and provide feedback on model outputs is a promising way to improve LLMs for complex reasoning tasks. However, existing approaches typically rely on stronger supervisors for annotating critique data. To address this, we propose Critique-RL, an online RL approach for developing critiquing language models without stronger supervision. Our approach operates on a two-player paradigm: the actor generates a response, the critic provides feedback, and the actor refines the response accordingly. We first reveal that relying solely on indirect reward signals from the actor's outputs for RL optimization often leads to unsatisfactory critics: while their helpfulness (i.e., providing constructive feedback) improves, the discriminability (i.e., determining whether a response is high-quality or not) remains poor, resulting in marginal performance gains. To overcome this, Critique-RL adopts a two-stage optimization strategy. In stage I, it reinforces the discriminability of the critic with direct rule-based reward signals; in stage II, it introduces indirect rewards based on actor refinement to improve the critic's helpfulness, while maintaining its discriminability via appropriate regularization. Extensive experiments across various tasks and models show that Critique-RL delivers substantial performance improvements. For example, it achieves a 9.02% gain on in-domain tasks and a 5.70% gain on out-of-domain tasks for Qwen2.5-7B, highlighting its potential.
>
---
#### [new 080] Agent-based Automated Claim Matching with Instruction-following LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对自动化声明匹配任务，提出基于代理的两步流程：先用LLM生成提示，再用另一LLM进行二分类匹配。研究表明，LLM生成的提示优于人工提示，小模型在生成中表现良好，且分步使用不同LLM更高效，揭示了LLM对任务的理解。**

- **链接: [http://arxiv.org/pdf/2510.23924v1](http://arxiv.org/pdf/2510.23924v1)**

> **作者:** Dina Pisarevskaya; Arkaitz Zubiaga
>
> **备注:** Accepted for the International Joint Conference on Natural Language Processing & Asia-Pacific Chapter of the Association for Computational Linguistics (2025) Findings
>
> **摘要:** We present a novel agent-based approach for the automated claim matching task with instruction-following LLMs. We propose a two-step pipeline that first generates prompts with LLMs, to then perform claim matching as a binary classification task with LLMs. We demonstrate that LLM-generated prompts can outperform SOTA with human-generated prompts, and that smaller LLMs can do as well as larger ones in the generation process, allowing to save computational resources. We also demonstrate the effectiveness of using different LLMs for each step of the pipeline, i.e. using an LLM for prompt generation, and another for claim matching. Our investigation into the prompt generation process in turn reveals insights into the LLMs' understanding of claim matching.
>
---
#### [new 081] Beyond MCQ: An Open-Ended Arabic Cultural QA Benchmark with Dialect Variants
- **分类: cs.CL; cs.AI; 68T50; F.2.2; I.2.7**

- **简介: 该论文针对阿拉伯语文化常识问答中方言差异与开放题型的评估难题，提出将标准阿拉伯语多选题翻译为多种方言并转为开放题，构建首个跨语言变体对齐的基准数据集。通过零样本与微调实验，揭示模型在方言与开放回答上的性能短板，并引入思维链提升推理能力。**

- **链接: [http://arxiv.org/pdf/2510.24328v1](http://arxiv.org/pdf/2510.24328v1)**

> **作者:** Hunzalah Hassan Bhatti; Firoj Alam
>
> **备注:** Cultural Knowledge, Everyday Knowledge, Open-Ended Question, Chain-of-Thought, Large Language Models, Native, Multilingual, Language Diversity
>
> **摘要:** Large Language Models (LLMs) are increasingly used to answer everyday questions, yet their performance on culturally grounded and dialectal content remains uneven across languages. We propose a comprehensive method that (i) translates Modern Standard Arabic (MSA) multiple-choice questions (MCQs) into English and several Arabic dialects, (ii) converts them into open-ended questions (OEQs), (iii) benchmarks a range of zero-shot and fine-tuned LLMs under both MCQ and OEQ settings, and (iv) generates chain-of-thought (CoT) rationales to fine-tune models for step-by-step reasoning. Using this method, we extend an existing dataset in which QAs are parallelly aligned across multiple language varieties, making it, to our knowledge, the first of its kind. We conduct extensive experiments with both open and closed models. Our findings show that (i) models underperform on Arabic dialects, revealing persistent gaps in culturally grounded and dialect-specific knowledge; (ii) Arabic-centric models perform well on MCQs but struggle with OEQs; and (iii) CoT improves judged correctness while yielding mixed n-gram-based metrics. The developed dataset will be publicly released to support further research on culturally and linguistically inclusive evaluation.
>
---
#### [new 082] SPICE: Self-Play In Corpus Environments Improves Reasoning
- **分类: cs.CL**

- **简介: 该论文提出SPICE框架，用于提升模型的推理能力。针对现有自洽学习缺乏持续挑战性任务的问题，通过文档驱动的对抗性自对弈机制，让模型在真实语料环境中自动生成渐进式难题并求解，实现持续自我优化，在数学与通用推理任务上显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.24684v1](http://arxiv.org/pdf/2510.24684v1)**

> **作者:** Bo Liu; Chuanyang Jin; Seungone Kim; Weizhe Yuan; Wenting Zhao; Ilia Kulikov; Xian Li; Sainbayar Sukhbaatar; Jack Lanchantin; Jason Weston
>
> **摘要:** Self-improving systems require environmental interaction for continuous adaptation. We introduce SPICE (Self-Play In Corpus Environments), a reinforcement learning framework where a single model acts in two roles: a Challenger that mines documents from a large corpus to generate diverse reasoning tasks, and a Reasoner that solves them. Through adversarial dynamics, the Challenger creates an automatic curriculum at the frontier of the Reasoner's capability, while corpus grounding provides the rich, near-inexhaustible external signal necessary for sustained improvement. Unlike existing ungrounded self-play methods that offer more limited benefits, SPICE achieves consistent gains across mathematical (+8.9%) and general reasoning (+9.8%) benchmarks on multiple model families. Our analysis reveals how document grounding is a key ingredient in SPICE to continuously generate its own increasingly challenging goals and achieve them, enabling sustained self-improvement.
>
---
#### [new 083] Evolving Diagnostic Agents in a Virtual Clinical Environment
- **分类: cs.CL**

- **简介: 该论文提出一种基于强化学习的诊断智能体框架，旨在训练大模型在虚拟临床环境中进行多轮交互式诊断。针对传统模型依赖静态病例、缺乏动态决策能力的问题，构建了DiagGym虚拟环境与DiagBench评估基准，通过端到端强化学习提升诊断准确率与检查推荐效果，显著优于现有主流模型。**

- **链接: [http://arxiv.org/pdf/2510.24654v1](http://arxiv.org/pdf/2510.24654v1)**

> **作者:** Pengcheng Qiu; Chaoyi Wu; Junwei Liu; Qiaoyu Zheng; Yusheng Liao; Haowen Wang; Yun Yue; Qianrui Fan; Shuai Zhen; Jian Wang; Jinjie Gu; Yanfeng Wang; Ya Zhang; Weidi Xie
>
> **摘要:** In this paper, we present a framework for training large language models (LLMs) as diagnostic agents with reinforcement learning, enabling them to manage multi-turn diagnostic processes, adaptively select examinations, and commit to final diagnoses. Unlike instruction-tuned models trained on static case summaries, our method acquires diagnostic strategies through interactive exploration and outcome-based feedback. Our contributions are fourfold: (i) We present DiagGym, a diagnostics world model trained with electronic health records that emits examination outcomes conditioned on patient history and recommended examination, serving as a virtual clinical environment for realistic diagnosis training and evaluation; (ii) We train DiagAgent via end-to-end, multi-turn reinforcement learning to learn diagnostic policies that optimize both information yield and diagnostic accuracy; (iii) We introduce DiagBench, a diagnostic benchmark comprising 750 cases with physician-validated examination recommendations and 99 cases annotated with 973 physician-written rubrics on diagnosis process; (iv) we demonstrate superior performance across diverse diagnostic settings. DiagAgent significantly outperforms 10 state-of-the-art LLMs, including DeepSeek-v3 and GPT-4o, as well as two prompt-engineered agents. In single-turn settings, DiagAgent achieves 9.34% higher diagnostic accuracy and 44.03% improvement in examination recommendation hit ratio. In end-to-end settings, it delivers 15.12% increase in diagnostic accuracy and 23.09% boost in examination recommendation F1 score. In rubric-based evaluation, it surpasses the next-best model, Claude-sonnet-4, by 7.1% in weighted rubric score. These findings indicate that learning policies in interactive clinical environments confers dynamic and clinically meaningful diagnostic management abilities unattainable through passive training alone.
>
---
#### [new 084] LuxIT: A Luxembourgish Instruction Tuning Dataset from Monolingual Seed Data
- **分类: cs.CL**

- **简介: 该论文针对低资源语言 Luxembourgish 缺乏高质量指令微调数据的问题，提出 LuxIT 数据集。基于母语文本，利用 DeepSeek-R1 生成并经 LLM 评判筛选，构建了单语指令数据。实验表明其在不同模型上效果不一，验证了方法可行性与优化空间。**

- **链接: [http://arxiv.org/pdf/2510.24434v1](http://arxiv.org/pdf/2510.24434v1)**

> **作者:** Julian Valline; Cedric Lothritz; Jordi Cabot
>
> **摘要:** The effectiveness of instruction-tuned Large Language Models (LLMs) is often limited in low-resource linguistic settings due to a lack of high-quality training data. We introduce LuxIT, a novel, monolingual instruction tuning dataset for Luxembourgish developed to mitigate this challenge. We synthesize the dataset from a corpus of native Luxembourgish texts, utilizing DeepSeek-R1-0528, chosen for its shown proficiency in Luxembourgish. Following generation, we apply a quality assurance process, employing an LLM-as-a-judge approach. To investigate the practical utility of the dataset, we fine-tune several smaller-scale LLMs on LuxIT. Subsequent benchmarking against their base models on Luxembourgish language proficiency examinations, however, yields mixed results, with performance varying significantly across different models. LuxIT represents a critical contribution to Luxembourgish natural language processing and offers a replicable monolingual methodology, though our findings highlight the need for further research to optimize its application.
>
---
#### [new 085] Automatically Benchmarking LLM Code Agents through Agent-Driven Annotation and Evaluation
- **分类: cs.SE; cs.CL**

- **简介: 该论文针对代码智能体评估中标注成本高、依赖单元测试等痛点，提出基于代理驱动的标注与评估框架。构建了包含50个真实项目、20个领域的PRDBench基准，引入“代理作为裁判”机制，支持多样化评估指标，实现高效、灵活、可扩展的项目级代码智能体评测。**

- **链接: [http://arxiv.org/pdf/2510.24358v1](http://arxiv.org/pdf/2510.24358v1)**

> **作者:** Lingyue Fu; Bolun Zhang; Hao Guan; Yaoming Zhu; Lin Qiu; Weiwen Liu; Xuezhi Cao; Xunliang Cai; Weinan Zhang; Yong Yu
>
> **摘要:** Recent advances in code agents have enabled automated software development at the project level, supported by large language models (LLMs) and widely adopted tools. However, existing benchmarks for code agent evaluation face two major limitations: high annotation cost and expertise requirements, and rigid evaluation metrics that rely primarily on unit tests. To address these challenges, we propose an agent-driven benchmark construction pipeline that leverages human supervision to efficiently generate diverse and challenging project-level tasks. Based on this approach, we introduce PRDBench, a novel benchmark comprising 50 real-world Python projects across 20 domains, each with structured Product Requirement Document (PRD) requirements, comprehensive evaluation criteria, and reference implementations. PRDBench features rich data sources, high task complexity, and flexible metrics. We further employ an Agent-as-a-Judge paradigm to score agent outputs, enabling the evaluation of various test types beyond unit tests. Extensive experiments on PRDBench demonstrate its effectiveness in assessing the capabilities of both code agents and evaluation agents, providing a scalable and robust framework for annotation and evaluation.
>
---
#### [new 086] Latent Sketchpad: Sketching Visual Thoughts to Elicit Multimodal Reasoning in MLLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Latent Sketchpad框架，解决MLLMs在复杂视觉规划与想象任务中表现不足的问题。通过引入内嵌视觉思维的生成机制，使模型在推理过程中交替进行文本与视觉潜表示生成，提升多模态推理能力，并实现可解释的草图输出。**

- **链接: [http://arxiv.org/pdf/2510.24514v1](http://arxiv.org/pdf/2510.24514v1)**

> **作者:** Huanyu Zhang; Wenshan Wu; Chengzu Li; Ning Shang; Yan Xia; Yangyu Huang; Yifan Zhang; Li Dong; Zhang Zhang; Liang Wang; Tieniu Tan; Furu Wei
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at visual understanding, they often struggle in complex scenarios that require visual planning and imagination. Inspired by how humans use sketching as a form of visual thinking to develop and communicate ideas, we introduce Latent Sketchpad, a framework that equips MLLMs with an internal visual scratchpad. The internal visual representations of MLLMs have traditionally been confined to perceptual understanding. We repurpose them to support generative visual thought without compromising reasoning ability. Building on frontier MLLMs, our approach integrates visual generation directly into their native autoregressive reasoning process. It allows the model to interleave textual reasoning with the generation of visual latents. These latents guide the internal thought process and can be translated into sketch images for interpretability. To realize this, we introduce two components: a Context-Aware Vision Head autoregressively produces visual representations, and a pretrained Sketch Decoder renders these into human-interpretable images. We evaluate the framework on our new dataset MazePlanning. Experiments across various MLLMs show that Latent Sketchpad delivers comparable or even superior reasoning performance to their backbone. It further generalizes across distinct frontier MLLMs, including Gemma3 and Qwen2.5-VL. By extending model's textual reasoning to visual thinking, our framework opens new opportunities for richer human-computer interaction and broader applications. More details and resources are available on our project page: https://latent-sketchpad.github.io/.
>
---
#### [new 087] GraphNet: A Large-Scale Computational Graph Dataset for Tensor Compiler Research
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出GraphNet，一个包含2.7K个真实深度学习计算图的大规模数据集，用于张量编译器研究。针对编译器优化效果评估难题，提出Speedup Score S(t)和Error-aware Speedup Score ES(t)指标，结合运行效率与正确性，有效衡量编译器性能，并在CV与NLP任务上验证了其实用性。**

- **链接: [http://arxiv.org/pdf/2510.24035v1](http://arxiv.org/pdf/2510.24035v1)**

> **作者:** Xinqi Li; Yiqun Liu; Shan Jiang; Enrong Zheng; Huaijin Zheng; Wenhao Dai; Haodong Deng; Dianhai Yu; Yanjun Ma
>
> **摘要:** We introduce GraphNet, a dataset of 2.7K real-world deep learning computational graphs with rich metadata, spanning six major task categories across multiple deep learning frameworks. To evaluate tensor compiler performance on these samples, we propose the benchmark metric Speedup Score S(t), which jointly considers runtime speedup and execution correctness under tunable tolerance levels, offering a reliable measure of general optimization capability. Furthermore, we extend S(t) to the Error-aware Speedup Score ES(t), which incorporates error information and helps compiler developers identify key performance bottlenecks. In this report, we benchmark the default tensor compilers, CINN for PaddlePaddle and TorchInductor for PyTorch, on computer vision (CV) and natural language processing (NLP) samples to demonstrate the practicality of GraphNet. The full construction pipeline with graph extraction and compiler evaluation tools is available at https://github.com/PaddlePaddle/GraphNet .
>
---
#### [new 088] emg2speech: synthesizing speech from electromyography using self-supervised speech models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出emg2speech，一种基于自监督语音模型的肌电到语音合成方法。针对无语言能力者无法发声的问题，利用面部肌肉肌电信号（EMG）直接生成语音。通过发现自监督语音特征与肌电功率的强线性关系，实现端到端的EMG到语音转换，无需显式发音模型或声码器训练。**

- **链接: [http://arxiv.org/pdf/2510.23969v1](http://arxiv.org/pdf/2510.23969v1)**

> **作者:** Harshavardhana T. Gowda; Lee M. Miller
>
> **摘要:** We present a neuromuscular speech interface that translates electromyographic (EMG) signals collected from orofacial muscles during speech articulation directly into audio. We show that self-supervised speech (SS) representations exhibit a strong linear relationship with the electrical power of muscle action potentials: SS features can be linearly mapped to EMG power with a correlation of $r = 0.85$. Moreover, EMG power vectors corresponding to different articulatory gestures form structured and separable clusters in feature space. This relationship: $\text{SS features}$ $\xrightarrow{\texttt{linear mapping}}$ $\text{EMG power}$ $\xrightarrow{\texttt{gesture-specific clustering}}$ $\text{articulatory movements}$, highlights that SS models implicitly encode articulatory mechanisms. Leveraging this property, we directly map EMG signals to SS feature space and synthesize speech, enabling end-to-end EMG-to-speech generation without explicit articulatory models and vocoder training.
>
---
#### [new 089] GIFT: Group-relative Implicit Fine Tuning Integrates GRPO with DPO and UNA
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出GIFT框架，用于大语言模型对齐任务。针对传统强化学习方法难以有效利用隐式奖励的问题，GIFT通过联合归一化隐式与显式奖励，将非凸优化转化为可微的均方误差损失，实现高效、稳定且低过拟合的在线对齐训练，显著提升推理与对齐性能。**

- **链接: [http://arxiv.org/pdf/2510.23868v1](http://arxiv.org/pdf/2510.23868v1)**

> **作者:** Zhichao Wang
>
> **摘要:** I propose \textbf{G}roup-relative \textbf{I}mplicit \textbf{F}ine \textbf{T}uning (GIFT), a novel reinforcement learning framework for aligning LLMs. Instead of directly maximizing cumulative rewards like PPO or GRPO, GIFT minimizes the discrepancy between implicit and explicit reward models. It combines three key ideas: (1) the online multi-response generation and normalization of GRPO, (2) the implicit reward formulation of DPO, and (3) the implicit-explicit reward alignment principle of UNA. By jointly normalizing the implicit and explicit rewards, GIFT eliminates an otherwise intractable term that prevents effective use of implicit rewards. This normalization transforms the complex reward maximization objective into a simple mean squared error (MSE) loss between the normalized reward functions, converting a non-convex optimization problem into a convex, stable, and analytically differentiable formulation. Unlike offline methods such as DPO and UNA, GIFT remains on-policy and thus retains exploration capability. Compared to GRPO, it requires fewer hyperparameters, converges faster, and generalizes better with significantly reduced training overfitting. Empirically, GIFT achieves superior reasoning and alignment performance on mathematical benchmarks while remaining computationally efficient.
>
---
#### [new 090] An Enhanced Dual Transformer Contrastive Network for Multimodal Sentiment Analysis
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文聚焦多模态情感分析任务，旨在融合文本与图像信息以更准确识别情感。针对现有方法跨模态交互不足的问题，提出Bert-ViT-EF模型，通过早期融合增强交互，并引入双变换器对比网络（DTCN），结合深层文本建模与对比学习，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.23617v1](http://arxiv.org/pdf/2510.23617v1)**

> **作者:** Phuong Q. Dao; Mark Roantree; Vuong M. Ngo
>
> **备注:** The paper has been accepted for presentation at the MEDES 2025 conference
>
> **摘要:** Multimodal Sentiment Analysis (MSA) seeks to understand human emotions by jointly analyzing data from multiple modalities typically text and images offering a richer and more accurate interpretation than unimodal approaches. In this paper, we first propose BERT-ViT-EF, a novel model that combines powerful Transformer-based encoders BERT for textual input and ViT for visual input through an early fusion strategy. This approach facilitates deeper cross-modal interactions and more effective joint representation learning. To further enhance the model's capability, we propose an extension called the Dual Transformer Contrastive Network (DTCN), which builds upon BERT-ViT-EF. DTCN incorporates an additional Transformer encoder layer after BERT to refine textual context (before fusion) and employs contrastive learning to align text and image representations, fostering robust multimodal feature learning. Empirical results on two widely used MSA benchmarks MVSA-Single and TumEmo demonstrate the effectiveness of our approach. DTCN achieves best accuracy (78.4%) and F1-score (78.3%) on TumEmo, and delivers competitive performance on MVSA-Single, with 76.6% accuracy and 75.9% F1-score. These improvements highlight the benefits of early fusion and deeper contextual modeling in Transformer-based multimodal sentiment analysis.
>
---
#### [new 091] Latent Chain-of-Thought for Visual Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对视觉推理任务，解决大视觉语言模型在链式思维推理中泛化性差、依赖有偏奖励模型的问题。提出基于后验推断的隐式链式思维框架，采用变分推断与稀疏奖励机制，提升推理多样性与可靠性，并通过贝叶斯推理缩放策略高效生成最优推理路径。**

- **链接: [http://arxiv.org/pdf/2510.23925v1](http://arxiv.org/pdf/2510.23925v1)**

> **作者:** Guohao Sun; Hang Hua; Jian Wang; Jiebo Luo; Sohail Dianat; Majid Rabbani; Raghuveer Rao; Zhiqiang Tao
>
> **备注:** NeurIPS 2025
>
> **摘要:** Chain-of-thought (CoT) reasoning is critical for improving the interpretability and reliability of Large Vision-Language Models (LVLMs). However, existing training algorithms such as SFT, PPO, and GRPO may not generalize well across unseen reasoning tasks and heavily rely on a biased reward model. To address this challenge, we reformulate reasoning in LVLMs as posterior inference and propose a scalable training algorithm based on amortized variational inference. By leveraging diversity-seeking reinforcement learning algorithms, we introduce a novel sparse reward function for token-level learning signals that encourage diverse, high-likelihood latent CoT, overcoming deterministic sampling limitations and avoiding reward hacking. Additionally, we implement a Bayesian inference-scaling strategy that replaces costly Best-of-N and Beam Search with a marginal likelihood to efficiently rank optimal rationales and answers. We empirically demonstrate that the proposed method enhances the state-of-the-art LVLMs on seven reasoning benchmarks, in terms of effectiveness, generalization, and interpretability.
>
---
#### [new 092] RoboOmni: Proactive Robot Manipulation in Omni-modal Context
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出RoboOmni框架，面向多模态情境下的主动机器人操作任务。针对现实交互中用户不常下达明确指令的问题，通过融合语音、视觉与环境声音，实现意图的主动推断与协作。构建了包含140k场景的OmniAction数据集，推动了无需显式命令的机器人智能发展。**

- **链接: [http://arxiv.org/pdf/2510.23763v1](http://arxiv.org/pdf/2510.23763v1)**

> **作者:** Siyin Wang; Jinlan Fu; Feihong Liu; Xinzhe He; Huangxuan Wu; Junhao Shi; Kexin Huang; Zhaoye Fei; Jingjing Gong; Zuxuan Wu; Yugang Jiang; See-Kiong Ng; Tat-Seng Chua; Xipeng Qiu
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have driven rapid progress in Vision-Language-Action (VLA) models for robotic manipulation. Although effective in many scenarios, current approaches largely rely on explicit instructions, whereas in real-world interactions, humans rarely issue instructions directly. Effective collaboration requires robots to infer user intentions proactively. In this work, we introduce cross-modal contextual instructions, a new setting where intent is derived from spoken dialogue, environmental sounds, and visual cues rather than explicit commands. To address this new setting, we present RoboOmni, a Perceiver-Thinker-Talker-Executor framework based on end-to-end omni-modal LLMs that unifies intention recognition, interaction confirmation, and action execution. RoboOmni fuses auditory and visual signals spatiotemporally for robust intention recognition, while supporting direct speech interaction. To address the absence of training data for proactive intention recognition in robotic manipulation, we build OmniAction, comprising 140k episodes, 5k+ speakers, 2.4k event sounds, 640 backgrounds, and six contextual instruction types. Experiments in simulation and real-world settings show that RoboOmni surpasses text- and ASR-based baselines in success rate, inference speed, intention recognition, and proactive assistance.
>
---
#### [new 093] VisCoder2: Building Multi-Language Visualization Coding Agents
- **分类: cs.SE; cs.AI; cs.CL; cs.PL**

- **简介: 该论文聚焦于多语言可视化代码生成任务，针对现有模型语言覆盖窄、执行不可靠、缺乏迭代纠错的问题，构建了大规模多语言数据集VisCode-Multi-679K、可执行评估基准VisPlotBench，并提出VisCoder2多语言模型。实验表明，其在多轮自调试下达82.4%执行通过率，显著优于开源基线并逼近GPT-4.1性能。**

- **链接: [http://arxiv.org/pdf/2510.23642v1](http://arxiv.org/pdf/2510.23642v1)**

> **作者:** Yuansheng Ni; Songcheng Cai; Xiangchao Chen; Jiarong Liang; Zhiheng Lyu; Jiaqi Deng; Kai Zou; Ping Nie; Fei Yuan; Xiang Yue; Wenhu Chen
>
> **摘要:** Large language models (LLMs) have recently enabled coding agents capable of generating, executing, and revising visualization code. However, existing models often fail in practical workflows due to limited language coverage, unreliable execution, and lack of iterative correction mechanisms. Progress has been constrained by narrow datasets and benchmarks that emphasize single-round generation and single-language tasks. To address these challenges, we introduce three complementary resources for advancing visualization coding agents. VisCode-Multi-679K is a large-scale, supervised dataset containing 679K validated and executable visualization samples with multi-turn correction dialogues across 12 programming languages. VisPlotBench is a benchmark for systematic evaluation, featuring executable tasks, rendered outputs, and protocols for both initial generation and multi-round self-debug. Finally, we present VisCoder2, a family of multi-language visualization models trained on VisCode-Multi-679K. Experiments show that VisCoder2 significantly outperforms strong open-source baselines and approaches the performance of proprietary models like GPT-4.1, with further gains from iterative self-debug, reaching 82.4% overall execution pass rate at the 32B scale, particularly in symbolic or compiler-dependent languages.
>
---
#### [new 094] OS-Sentinel: Towards Safety-Enhanced Mobile GUI Agents via Hybrid Validation in Realistic Workflows
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文针对移动GUI代理的安全性问题，提出OS-Sentinel框架，融合形式化验证与视觉语言模型的上下文判断，实现对系统级违规和上下文风险的混合检测。基于动态沙箱与真实操作轨迹构建基准，显著提升安全检测效果。**

- **链接: [http://arxiv.org/pdf/2510.24411v1](http://arxiv.org/pdf/2510.24411v1)**

> **作者:** Qiushi Sun; Mukai Li; Zhoumianze Liu; Zhihui Xie; Fangzhi Xu; Zhangyue Yin; Kanzhi Cheng; Zehao Li; Zichen Ding; Qi Liu; Zhiyong Wu; Zhuosheng Zhang; Ben Kao; Lingpeng Kong
>
> **备注:** work in progress
>
> **摘要:** Computer-using agents powered by Vision-Language Models (VLMs) have demonstrated human-like capabilities in operating digital environments like mobile platforms. While these agents hold great promise for advancing digital automation, their potential for unsafe operations, such as system compromise and privacy leakage, is raising significant concerns. Detecting these safety concerns across the vast and complex operational space of mobile environments presents a formidable challenge that remains critically underexplored. To establish a foundation for mobile agent safety research, we introduce MobileRisk-Live, a dynamic sandbox environment accompanied by a safety detection benchmark comprising realistic trajectories with fine-grained annotations. Built upon this, we propose OS-Sentinel, a novel hybrid safety detection framework that synergistically combines a Formal Verifier for detecting explicit system-level violations with a VLM-based Contextual Judge for assessing contextual risks and agent actions. Experiments show that OS-Sentinel achieves 10%-30% improvements over existing approaches across multiple metrics. Further analysis provides critical insights that foster the development of safer and more reliable autonomous mobile agents.
>
---
#### [new 095] STAR-Bench: Probing Deep Spatio-Temporal Reasoning as Audio 4D Intelligence
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出STAR-Bench，用于评估音频4D智能——对时间与三维空间中声音动态的细粒度推理能力。针对现有基准依赖文本描述、忽略感知推理的问题，构建包含基础感知与综合时空推理的评测体系，结合物理模拟与人工标注数据，揭示模型在感知与推理上的显著差距，推动更鲁棒的物理世界理解模型发展。**

- **链接: [http://arxiv.org/pdf/2510.24693v1](http://arxiv.org/pdf/2510.24693v1)**

> **作者:** Zihan Liu; Zhikang Niu; Qiuyang Xiao; Zhisheng Zheng; Ruoqi Yuan; Yuhang Zang; Yuhang Cao; Xiaoyi Dong; Jianze Liang; Xie Chen; Leilei Sun; Dahua Lin; Jiaqi Wang
>
> **备注:** Homepage: https://internlm.github.io/StarBench/
>
> **摘要:** Despite rapid progress in Multi-modal Large Language Models and Large Audio-Language Models, existing audio benchmarks largely test semantics that can be recovered from text captions, masking deficits in fine-grained perceptual reasoning. We formalize audio 4D intelligence that is defined as reasoning over sound dynamics in time and 3D space, and introduce STAR-Bench to measure it. STAR-Bench combines a Foundational Acoustic Perception setting (six attributes under absolute and relative regimes) with a Holistic Spatio-Temporal Reasoning setting that includes segment reordering for continuous and discrete processes and spatial tasks spanning static localization, multi-source relations, and dynamic trajectories. Our data curation pipeline uses two methods to ensure high-quality samples. For foundational tasks, we use procedurally synthesized and physics-simulated audio. For holistic data, we follow a four-stage process that includes human annotation and final selection based on human performance. Unlike prior benchmarks where caption-only answering reduces accuracy slightly, STAR-Bench induces far larger drops (-31.5\% temporal, -35.2\% spatial), evidencing its focus on linguistically hard-to-describe cues. Evaluating 19 models reveals substantial gaps compared with humans and a capability hierarchy: closed-source models are bottlenecked by fine-grained perception, while open-source models lag across perception, knowledge, and reasoning. Our STAR-Bench provides critical insights and a clear path forward for developing future models with a more robust understanding of the physical world.
>
---
#### [new 096] VC4VG: Optimizing Video Captions for Text-to-Video Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对文本到视频生成（T2V）中视频描述质量影响模型性能的问题，提出VC4VG框架，通过多维度分析与优化视频标题，构建专用评估基准VC4VG-Bench，实验证明高质量标题显著提升生成效果，推动T2V训练数据优化。**

- **链接: [http://arxiv.org/pdf/2510.24134v1](http://arxiv.org/pdf/2510.24134v1)**

> **作者:** Yang Du; Zhuoran Lin; Kaiqiang Song; Biao Wang; Zhicheng Zheng; Tiezheng Ge; Bo Zheng; Qin Jin
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Recent advances in text-to-video (T2V) generation highlight the critical role of high-quality video-text pairs in training models capable of producing coherent and instruction-aligned videos. However, strategies for optimizing video captions specifically for T2V training remain underexplored. In this paper, we introduce VC4VG (Video Captioning for Video Generation), a comprehensive caption optimization framework tailored to the needs of T2V models.We begin by analyzing caption content from a T2V perspective, decomposing the essential elements required for video reconstruction into multiple dimensions, and proposing a principled caption design methodology. To support evaluation, we construct VC4VG-Bench, a new benchmark featuring fine-grained, multi-dimensional, and necessity-graded metrics aligned with T2V-specific requirements.Extensive T2V fine-tuning experiments demonstrate a strong correlation between improved caption quality and video generation performance, validating the effectiveness of our approach. We release all benchmark tools and code at https://github.com/qyr0403/VC4VG to support further research.
>
---
#### [new 097] NUM2EVENT: Interpretable Event Reasoning from Numerical time-series
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出NUM2EVENT任务，旨在从数值时间序列中推理可解释的结构化事件。针对现有模型对数值信号理解不足的问题，提出融合代理引导提取器、合成生成器与两阶段微调的框架，实现对数值变化的因果推理与中间解释生成，显著提升事件识别精度。**

- **链接: [http://arxiv.org/pdf/2510.23630v1](http://arxiv.org/pdf/2510.23630v1)**

> **作者:** Ninghui Feng; Yiyan Qi
>
> **摘要:** Large language models (LLMs) have recently demonstrated impressive multimodal reasoning capabilities, yet their understanding of purely numerical time-series signals remains limited. Existing approaches mainly focus on forecasting or trend description, without uncovering the latent events that drive numerical changes or explaining the reasoning process behind them. In this work, we introduce the task of number-to-event reasoning and decoding, which aims to infer interpretable structured events from numerical inputs, even when current text is unavailable. To address the data scarcity and semantic alignment challenges, we propose a reasoning-aware framework that integrates an agent-guided event extractor (AGE), a marked multivariate Hawkes-based synthetic generator (EveDTS), and a two-stage fine-tuning pipeline combining a time-series encoder with a structured decoder. Our model explicitly reasons over numerical changes, generates intermediate explanations, and outputs structured event hypotheses. Experiments on multi-domain datasets show that our method substantially outperforms strong LLM baselines in event-level precision and recall. These results suggest a new direction for bridging quantitative reasoning and semantic understanding, enabling LLMs to explain and predict events directly from numerical dynamics.
>
---
#### [new 098] MUStReason: A Benchmark for Diagnosing Pragmatic Reasoning in Video-LMs for Multimodal Sarcasm Detection
- **分类: cs.LG; cs.CL**

- **简介: 该论文聚焦多模态讽刺检测任务，针对视频语言模型在识别讽刺意图时难以融合多模态线索并进行语用推理的问题。提出MUStReason基准，包含模态相关线索与推理步骤标注，并引入PragCoT框架，引导模型关注隐含意图而非字面意义，实现对感知与推理过程的诊断与提升。**

- **链接: [http://arxiv.org/pdf/2510.23727v1](http://arxiv.org/pdf/2510.23727v1)**

> **作者:** Anisha Saha; Varsha Suresh; Timothy Hospedales; Vera Demberg
>
> **摘要:** Sarcasm is a specific type of irony which involves discerning what is said from what is meant. Detecting sarcasm depends not only on the literal content of an utterance but also on non-verbal cues such as speaker's tonality, facial expressions and conversational context. However, current multimodal models struggle with complex tasks like sarcasm detection, which require identifying relevant cues across modalities and pragmatically reasoning over them to infer the speaker's intention. To explore these limitations in VideoLMs, we introduce MUStReason, a diagnostic benchmark enriched with annotations of modality-specific relevant cues and underlying reasoning steps to identify sarcastic intent. In addition to benchmarking sarcasm classification performance in VideoLMs, using MUStReason we quantitatively and qualitatively evaluate the generated reasoning by disentangling the problem into perception and reasoning, we propose PragCoT, a framework that steers VideoLMs to focus on implied intentions over literal meaning, a property core to detecting sarcasm.
>
---
#### [new 099] Flight Delay Prediction via Cross-Modality Adaptation of Large Language Models and Aircraft Trajectory Representation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于飞行延误预测任务，旨在提升预测精度与实时性。通过将飞机轨迹数据跨模态适配至语言模型，融合航路信息、天气和机场通告等文本数据，实现对终端区航班延误的亚分钟级精准预测，增强了对延误源的上下文理解，具备实际应用与扩展潜力。**

- **链接: [http://arxiv.org/pdf/2510.23636v1](http://arxiv.org/pdf/2510.23636v1)**

> **作者:** Thaweerath Phisannupawong; Joshua Julian Damanik; Han-Lim Choi
>
> **备注:** Preprint submitted to Aerospace Science and Technology (Elsevier) for possible publication
>
> **摘要:** Flight delay prediction has become a key focus in air traffic management, as delays highlight inefficiencies that impact overall network performance. This paper presents a lightweight large language model-based multimodal flight delay prediction, formulated from the perspective of air traffic controllers monitoring aircraft delay after entering the terminal area. The approach integrates trajectory representations with textual aeronautical information, including flight information, weather reports, and aerodrome notices, by adapting trajectory data into the language modality to capture airspace conditions. Experimental results show that the model consistently achieves sub-minute prediction error by effectively leveraging contextual information related to the sources of delay. The framework demonstrates that linguistic understanding, when combined with cross-modality adaptation of trajectory information, enhances delay prediction. Moreover, the approach shows practicality and scalability for real-world operations, supporting real-time updates that refine predictions upon receiving new operational information.
>
---
#### [new 100] Combining Textual and Structural Information for Premise Selection in Lean
- **分类: cs.LG; cs.AI; cs.CL; cs.LO**

- **简介: 该论文针对形式化证明中的前提选择任务，解决现有方法忽略前提间依赖关系的问题。提出结合文本嵌入与图神经网络的图增强模型，利用异构依赖图捕捉状态-前提和前提-前提关系，在LeanDojo基准上显著优于基线方法。**

- **链接: [http://arxiv.org/pdf/2510.23637v1](http://arxiv.org/pdf/2510.23637v1)**

> **作者:** Job Petrovčič; David Eliecer Narvaez Denis; Ljupčo Todorovski
>
> **摘要:** Premise selection is a key bottleneck for scaling theorem proving in large formal libraries. Yet existing language-based methods often treat premises in isolation, ignoring the web of dependencies that connects them. We present a graph-augmented approach that combines dense text embeddings of Lean formalizations with graph neural networks over a heterogeneous dependency graph capturing both state--premise and premise--premise relations. On the LeanDojo Benchmark, our method outperforms the ReProver language-based baseline by over 25% across standard retrieval metrics. These results demonstrate the power of relational information for more effective premise selection.
>
---
#### [new 101] A Neural Model for Contextual Biasing Score Learning and Filtering
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文针对语音识别中的上下文偏置任务，提出一种基于注意力机制的评分模型，通过判别性目标提升真实短语得分并抑制干扰项。方法可过滤候选短语，并用于浅层融合，显著提升识别准确率，且兼容任意ASR系统。**

- **链接: [http://arxiv.org/pdf/2510.23849v1](http://arxiv.org/pdf/2510.23849v1)**

> **作者:** Wanting Huang; Weiran Wang
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** Contextual biasing improves automatic speech recognition (ASR) by integrating external knowledge, such as user-specific phrases or entities, during decoding. In this work, we use an attention-based biasing decoder to produce scores for candidate phrases based on acoustic information extracted by an ASR encoder, which can be used to filter out unlikely phrases and to calculate bonus for shallow-fusion biasing. We introduce a per-token discriminative objective that encourages higher scores for ground-truth phrases while suppressing distractors. Experiments on the Librispeech biasing benchmark show that our method effectively filters out majority of the candidate phrases, and significantly improves recognition accuracy under different biasing conditions when the scores are used in shallow fusion biasing. Our approach is modular and can be used with any ASR system, and the filtering mechanism can potentially boost performance of other biasing methods.
>
---
#### [new 102] ViPER: Empowering the Self-Evolution of Visual Perception Abilities in Vision-Language Model
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对视觉语言模型（VLM）在细粒度视觉感知能力上的局限性，提出ViPER框架。通过自举式双阶段强化学习，实现图像与实例级重建的闭环训练，促进模型自我批判与进化，显著提升感知能力，同时保持通用性。**

- **链接: [http://arxiv.org/pdf/2510.24285v1](http://arxiv.org/pdf/2510.24285v1)**

> **作者:** Juntian Zhang; Song Jin; Chuanqi Cheng; Yuhan Liu; Yankai Lin; Xun Zhang; Yufei Zhang; Fei Jiang; Guojun Yin; Wei Lin; Rui Yan
>
> **摘要:** The limited capacity for fine-grained visual perception presents a critical bottleneck for Vision-Language Models (VLMs) in real-world applications. Addressing this is challenging due to the scarcity of high-quality data and the limitations of existing methods: supervised fine-tuning (SFT) often compromises general capabilities, while reinforcement fine-tuning (RFT) prioritizes textual reasoning over visual perception. To bridge this gap, we propose a novel two-stage task that structures visual perception learning as a coarse-to-fine progressive process. Based on this task formulation, we develop ViPER, a self-bootstrapping framework specifically designed to enable iterative evolution through self-critiquing and self-prediction. By synergistically integrating image-level and instance-level reconstruction with a two-stage reinforcement learning strategy, ViPER establishes a closed-loop training paradigm, where internally synthesized data directly fuel the enhancement of perceptual ability. Applied to the Qwen2.5-VL family, ViPER produces the Qwen-Viper series. With an average gain of 1.7% on seven comprehensive benchmarks spanning various tasks and up to 6.0% on fine-grained perception, Qwen-Viper consistently demonstrates superior performance across different vision-language scenarios while maintaining generalizability. Beyond enabling self-improvement in perceptual capabilities, ViPER provides concrete evidence for the reciprocal relationship between generation and understanding, a breakthrough to developing more autonomous and capable VLMs.
>
---
#### [new 103] From Detection to Discovery: A Closed-Loop Approach for Simultaneous and Continuous Medical Knowledge Expansion and Depression Detection on Social Media
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出一种闭环大语言模型-知识图谱框架，用于社交媒体上抑郁症状的持续检测与医学知识扩展。解决传统方法仅预测不更新知识的问题，通过迭代学习实现预测与知识共演化，提升检测准确率并发现临床新发现。**

- **链接: [http://arxiv.org/pdf/2510.23626v1](http://arxiv.org/pdf/2510.23626v1)**

> **作者:** Shuang Geng; Wenli Zhang; Jiaheng Xie; Rui Wang; Sudha Ram
>
> **备注:** Presented at SWAIB2025 and HICSS2026
>
> **摘要:** Social media user-generated content (UGC) provides real-time, self-reported indicators of mental health conditions such as depression, offering a valuable source for predictive analytics. While prior studies integrate medical knowledge to improve prediction accuracy, they overlook the opportunity to simultaneously expand such knowledge through predictive processes. We develop a Closed-Loop Large Language Model (LLM)-Knowledge Graph framework that integrates prediction and knowledge expansion in an iterative learning cycle. In the knowledge-aware depression detection phase, the LLM jointly performs depression detection and entity extraction, while the knowledge graph represents and weights these entities to refine prediction performance. In the knowledge refinement and expansion phase, new entities, relationships, and entity types extracted by the LLM are incorporated into the knowledge graph under expert supervision, enabling continual knowledge evolution. Using large-scale UGC, the framework enhances both predictive accuracy and medical understanding. Expert evaluations confirmed the discovery of clinically meaningful symptoms, comorbidities, and social triggers complementary to existing literature. We conceptualize and operationalize prediction-through-learning and learning-through-prediction as mutually reinforcing processes, advancing both methodological and theoretical understanding in predictive analytics. The framework demonstrates the co-evolution of computational models and domain knowledge, offering a foundation for adaptive, data-driven knowledge systems applicable to other dynamic risk monitoring contexts.
>
---
#### [new 104] Law in Silico: Simulating Legal Society with LLM-Based Agents
- **分类: cs.AI; cs.CL; cs.CY; cs.MA**

- **简介: 该论文提出基于大语言模型的法律社会仿真框架Law in Silico，旨在模拟立法、司法与执法等制度运行。针对真实法律实验难实施的问题，利用LLM代理实现个体决策与制度互动，验证法律理论并评估系统对弱势群体权利的保护效果。**

- **链接: [http://arxiv.org/pdf/2510.24442v1](http://arxiv.org/pdf/2510.24442v1)**

> **作者:** Yiding Wang; Yuxuan Chen; Fanxu Meng; Xifan Chen; Xiaolei Yang; Muhan Zhang
>
> **摘要:** Since real-world legal experiments are often costly or infeasible, simulating legal societies with Artificial Intelligence (AI) systems provides an effective alternative for verifying and developing legal theory, as well as supporting legal administration. Large Language Models (LLMs), with their world knowledge and role-playing capabilities, are strong candidates to serve as the foundation for legal society simulation. However, the application of LLMs to simulate legal systems remains underexplored. In this work, we introduce Law in Silico, an LLM-based agent framework for simulating legal scenarios with individual decision-making and institutional mechanisms of legislation, adjudication, and enforcement. Our experiments, which compare simulated crime rates with real-world data, demonstrate that LLM-based agents can largely reproduce macro-level crime trends and provide insights that align with real-world observations. At the same time, micro-level simulations reveal that a well-functioning, transparent, and adaptive legal system offers better protection of the rights of vulnerable individuals.
>
---
## 更新

#### [replaced 001] TokenTiming: A Dynamic Alignment Method for Universal Speculative Decoding Model Pairs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.15545v2](http://arxiv.org/pdf/2510.15545v2)**

> **作者:** Sibo Xiao; Jinyuan Fu; Zhongle Xie; Lidan Shou
>
> **摘要:** Accelerating the inference of large language models (LLMs) has been a critical challenge in generative AI. Speculative decoding (SD) substantially improves LLM inference efficiency. However, its utility is limited by a fundamental constraint: the draft and target models must share the same vocabulary, thus limiting the herd of available draft models and often necessitating the training of a new model from scratch. Inspired by Dynamic Time Warping (DTW), a classic algorithm for aligning time series, we propose the algorithm TokenTiming for universal speculative decoding. It operates by re-encoding the draft token sequence to get a new target token sequence, and then uses DTW to build a mapping to transfer the probability distributions for speculative sampling. Benefiting from this, our method accommodates mismatched vocabularies and works with any off-the-shelf models without retraining and modification. We conduct comprehensive experiments on various tasks, demonstrating 1.57x speedup. This work enables a universal approach for draft model selection, making SD a more versatile and practical tool for LLM acceleration.
>
---
#### [replaced 002] BrowseConf: Confidence-Guided Test-Time Scaling for Web Agents
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.23458v2](http://arxiv.org/pdf/2510.23458v2)**

> **作者:** Litu Ou; Kuan Li; Huifeng Yin; Liwen Zhang; Zhongwang Zhang; Xixi Wu; Rui Ye; Zile Qiao; Pengjun Xie; Jingren Zhou; Yong Jiang
>
> **备注:** 25 pages
>
> **摘要:** Confidence in LLMs is a useful indicator of model uncertainty and answer reliability. Existing work mainly focused on single-turn scenarios, while research on confidence in complex multi-turn interactions is limited. In this paper, we investigate whether LLM-based search agents have the ability to communicate their own confidence through verbalized confidence scores after long sequences of actions, a significantly more challenging task compared to outputting confidence in a single interaction. Experimenting on open-source agentic models, we first find that models exhibit much higher task accuracy at high confidence while having near-zero accuracy when confidence is low. Based on this observation, we propose Test-Time Scaling (TTS) methods that use confidence scores to determine answer quality, encourage the model to try again until reaching a satisfactory confidence level. Results show that our proposed methods significantly reduce token consumption while demonstrating competitive performance compared to baseline fixed budget TTS methods.
>
---
#### [replaced 003] TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.20445v5](http://arxiv.org/pdf/2410.20445v5)**

> **作者:** Yuwei Du; Jie Feng; Jie Zhao; Yong Li
>
> **备注:** Accepted by NeurIPS 2025, https://github.com/tsinghua-fib-lab/TrajAgent
>
> **摘要:** Trajectory modeling, which includes research on trajectory data pattern mining and future prediction, has widespread applications in areas such as life services, urban transportation, and public administration. Numerous methods have been proposed to address specific problems within trajectory modeling. However, the heterogeneity of data and the diversity of trajectory tasks make effective and reliable trajectory modeling an important yet highly challenging endeavor, even for domain experts. In this paper, we propose TrajAgent, an agent framework powered by large language models, designed to facilitate robust and efficient trajectory modeling through automation modeling. This framework leverages and optimizes diverse specialized models to address various trajectory modeling tasks across different datasets effectively. In TrajAgent, we first develop UniEnv, an execution environment with a unified data and model interface, to support the execution and training of various models. Building on UniEnv, we introduce an agentic workflow designed for automatic trajectory modeling across various trajectory tasks and data. Furthermore, we introduce collaborative learning schema between LLM-based agents and small speciallized models, to enhance the performance of the whole framework effectively. Extensive experiments on five tasks using four real-world datasets demonstrate the effectiveness of TrajAgent in automated trajectory modeling, achieving a performance improvement of 2.38%-69.91% over baseline methods. The codes and data can be accessed via https://github.com/tsinghua-fib-lab/TrajAgent.
>
---
#### [replaced 004] SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.15355v2](http://arxiv.org/pdf/2506.15355v2)**

> **作者:** Arijit Maji; Raghvendra Kumar; Akash Ghosh; Anushka; Sriparna Saha
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Language Models (LMs) are indispensable tools shaping modern workflows, but their global effectiveness depends on understanding local socio-cultural contexts. To address this, we introduce SANSKRITI, a benchmark designed to evaluate language models' comprehension of India's rich cultural diversity. Comprising 21,853 meticulously curated question-answer pairs spanning 28 states and 8 union territories, SANSKRITI is the largest dataset for testing Indian cultural knowledge. It covers sixteen key attributes of Indian culture: rituals and ceremonies, history, tourism, cuisine, dance and music, costume, language, art, festivals, religion, medicine, transport, sports, nightlife, and personalities, providing a comprehensive representation of India's cultural tapestry. We evaluate SANSKRITI on leading Large Language Models (LLMs), Indic Language Models (ILMs), and Small Language Models (SLMs), revealing significant disparities in their ability to handle culturally nuanced queries, with many models struggling in region-specific contexts. By offering an extensive, culturally rich, and diverse dataset, SANSKRITI sets a new standard for assessing and improving the cultural understanding of LMs.
>
---
#### [replaced 005] From Language to Action: A Review of Large Language Models as Autonomous Agents and Tool Users
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.17281v2](http://arxiv.org/pdf/2508.17281v2)**

> **作者:** Sadia Sultana Chowa; Riasad Alvi; Subhey Sadi Rahman; Md Abdur Rahman; Mohaimenul Azam Khan Raiaan; Md Rafiqul Islam; Mukhtar Hussain; Sami Azam
>
> **备注:** Submitted to Artificial Intelligence Review for peer review
>
> **摘要:** The pursuit of human-level artificial intelligence (AI) has significantly advanced the development of autonomous agents and Large Language Models (LLMs). LLMs are now widely utilized as decision-making agents for their ability to interpret instructions, manage sequential tasks, and adapt through feedback. This review examines recent developments in employing LLMs as autonomous agents and tool users and comprises seven research questions. We only used the papers published between 2023 and 2025 in conferences of the A* and A rank and Q1 journals. A structured analysis of the LLM agents' architectural design principles, dividing their applications into single-agent and multi-agent systems, and strategies for integrating external tools is presented. In addition, the cognitive mechanisms of LLM, including reasoning, planning, and memory, and the impact of prompting methods and fine-tuning procedures on agent performance are also investigated. Furthermore, we evaluated current benchmarks and assessment protocols and have provided an analysis of 68 publicly available datasets to assess the performance of LLM-based agents in various tasks. In conducting this review, we have identified critical findings on verifiable reasoning of LLMs, the capacity for self-improvement, and the personalization of LLM-based agents. Finally, we have discussed ten future research directions to overcome these gaps.
>
---
#### [replaced 006] Navigation with VLM framework: Towards Going to Any Language
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.02787v2](http://arxiv.org/pdf/2410.02787v2)**

> **作者:** Zecheng Yin; Chonghao Cheng; and Yao Guo; Zhen Li
>
> **备注:** under review
>
> **摘要:** Navigating towards fully open language goals and exploring open scenes in an intelligent way have always raised significant challenges. Recently, Vision Language Models (VLMs) have demonstrated remarkable capabilities to reason with both language and visual data. Although many works have focused on leveraging VLMs for navigation in open scenes, they often require high computational cost, rely on object-centric approaches, or depend on environmental priors in detailed human instructions. We introduce Navigation with VLM (NavVLM), a training-free framework that harnesses open-source VLMs to enable robots to navigate effectively, even for human-friendly language goal such as abstract places, actions, or specific objects in open scenes. NavVLM leverages the VLM as its cognitive core to perceive environmental information and constantly provides exploration guidance achieving intelligent navigation with only a neat target rather than a detailed instruction with environment prior. We evaluated and validated NavVLM in both simulation and real-world experiments. In simulation, our framework achieves state-of-the-art performance in Success weighted by Path Length (SPL) on object-specifc tasks in richly detailed environments from Matterport 3D (MP3D), Habitat Matterport 3D (HM3D) and Gibson. With navigation episode reported, NavVLM demonstrates the capabilities to navigate towards any open-set languages. In real-world validation, we validated our framework's effectiveness in real-world robot at indoor scene.
>
---
#### [replaced 007] A Comprehensive Survey on Reinforcement Learning-based Agentic Search: Foundations, Roles, Optimizations, Evaluations, and Applications
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.16724v2](http://arxiv.org/pdf/2510.16724v2)**

> **作者:** Minhua Lin; Zongyu Wu; Zhichao Xu; Hui Liu; Xianfeng Tang; Qi He; Charu Aggarwal; Hui Liu; Xiang Zhang; Suhang Wang
>
> **备注:** 38 pages, 4 figures, 7 tables
>
> **摘要:** The advent of large language models (LLMs) has transformed information access and reasoning through open-ended natural language interaction. However, LLMs remain limited by static knowledge, factual hallucinations, and the inability to retrieve real-time or domain-specific information. Retrieval-Augmented Generation (RAG) mitigates these issues by grounding model outputs in external evidence, but traditional RAG pipelines are often single turn and heuristic, lacking adaptive control over retrieval and reasoning. Recent advances in agentic search address these limitations by enabling LLMs to plan, retrieve, and reflect through multi-step interaction with search environments. Within this paradigm, reinforcement learning (RL) offers a powerful mechanism for adaptive and self-improving search behavior. This survey provides the first comprehensive overview of \emph{RL-based agentic search}, organizing the emerging field along three complementary dimensions: (i) What RL is for (functional roles), (ii) How RL is used (optimization strategies), and (iii) Where RL is applied (scope of optimization). We summarize representative methods, evaluation protocols, and applications, and discuss open challenges and future directions toward building reliable and scalable RL driven agentic search systems. We hope this survey will inspire future research on the integration of RL and agentic search. Our repository is available at https://github.com/ventr1c/Awesome-RL-based-Agentic-Search-Papers.
>
---
#### [replaced 008] ReCode: Unify Plan and Action for Universal Granularity Control
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.23564v2](http://arxiv.org/pdf/2510.23564v2)**

> **作者:** Zhaoyang Yu; Jiayi Zhang; Huixue Su; Yufan Zhao; Yifan Wu; Mingyi Deng; Jinyu Xiang; Yizhang Lin; Lingxiao Tang; Yingchao Li; Yuyu Luo; Bang Liu; Chenglin Wu
>
> **摘要:** Real-world tasks require decisions at varying granularities, and humans excel at this by leveraging a unified cognitive representation where planning is fundamentally understood as a high-level form of action. However, current Large Language Model (LLM)-based agents lack this crucial capability to operate fluidly across decision granularities. This limitation stems from existing paradigms that enforce a rigid separation between high-level planning and low-level action, which impairs dynamic adaptability and limits generalization. We propose ReCode (Recursive Code Generation), a novel paradigm that addresses this limitation by unifying planning and action within a single code representation. In this representation, ReCode treats high-level plans as abstract placeholder functions, which the agent then recursively decomposes into finer-grained sub-functions until reaching primitive actions. This recursive approach dissolves the rigid boundary between plan and action, enabling the agent to dynamically control its decision granularity. Furthermore, the recursive structure inherently generates rich, multi-granularity training data, enabling models to learn hierarchical decision-making processes. Extensive experiments show ReCode significantly surpasses advanced baselines in inference performance and demonstrates exceptional data efficiency in training, validating our core insight that unifying planning and action through recursive code generation is a powerful and effective approach to achieving universal granularity control. The code is available at https://github.com/FoundationAgents/ReCode.
>
---
#### [replaced 009] Evaluation of Geographical Distortions in Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.17401v2](http://arxiv.org/pdf/2404.17401v2)**

> **作者:** Rémy Decoupes; Roberto Interdonato; Mathieu Roche; Maguelonne Teisseire; Sarah Valentin
>
> **备注:** Accepted version. Published in Machine Learning (Springer) 114:263 (2025). Open access under a CC BY-NC-ND 4.0 license. DOI: 10.1007/s10994-025-06916-9
>
> **摘要:** Language models now constitute essential tools for improving efficiency for many professional tasks such as writing, coding, or learning. For this reason, it is imperative to identify inherent biases. In the field of Natural Language Processing, five sources of bias are well-identified: data, annotation, representation, models, and research design. This study focuses on biases related to geographical knowledge. We explore the connection between geography and language models by highlighting their tendency to misrepresent spatial information, thus leading to distortions in the representation of geographical distances. This study introduces four indicators to assess these distortions, by comparing geographical and semantic distances. Experiments are conducted from these four indicators with ten widely used language models. Results underscore the critical necessity of inspecting and rectifying spatial biases in language models to ensure accurate and equitable representations.
>
---
#### [replaced 010] Seeing Symbols, Missing Cultures: Probing Vision-Language Models' Reasoning on Fire Imagery and Cultural Meaning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.23311v2](http://arxiv.org/pdf/2509.23311v2)**

> **作者:** Haorui Yu; Yang Zhao; Yijia Chu; Qiufeng Yi
>
> **备注:** 8 pages, 5 figures, 4 tables. Submitted to WiNLP 2025 Workshop at COLING 2025
>
> **摘要:** Vision-Language Models (VLMs) often appear culturally competent but rely on superficial pattern matching rather than genuine cultural understanding. We introduce a diagnostic framework to probe VLM reasoning on fire-themed cultural imagery through both classification and explanation analysis. Testing multiple models on Western festivals, non-Western traditions, and emergency scenes reveals systematic biases: models correctly identify prominent Western festivals but struggle with underrepresented cultural events, frequently offering vague labels or dangerously misclassifying emergencies as celebrations. These failures expose the risks of symbolic shortcuts and highlight the need for cultural evaluation beyond accuracy metrics to ensure interpretable and fair multimodal systems.
>
---
#### [replaced 011] NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.06560v4](http://arxiv.org/pdf/2504.06560v4)**

> **作者:** Lanrui Wang; Mingyu Zheng; Hongyin Tang; Zheng Lin; Yanan Cao; Jingang Wang; Xunliang Cai; Weiping Wang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Processing structured tabular data, particularly large and lengthy tables, constitutes a fundamental yet challenging task for large language models (LLMs). However, existing long-context benchmarks like Needle-in-a-Haystack primarily focus on unstructured text, neglecting the challenge of diverse structured tables. Meanwhile, previous tabular benchmarks mainly consider downstream tasks that require high-level reasoning abilities, and overlook models' underlying fine-grained perception of individual table cells, which is crucial for practical and robust LLM-based table applications. To address this gap, we introduce \textsc{NeedleInATable} (NIAT), a new long-context tabular benchmark that treats each table cell as a ``needle'' and requires models to extract the target cell based on cell locations or lookup questions. Our comprehensive evaluation of various LLMs and multimodal LLMs reveals a substantial performance gap between popular downstream tabular tasks and the simpler NIAT task, suggesting that they may rely on dataset-specific correlations or shortcuts to obtain better benchmark results but lack truly robust long-context understanding towards structured tables. Furthermore, we demonstrate that using synthesized NIAT training data can effectively improve performance on both NIAT task and downstream tabular tasks, which validates the importance of NIAT capability for LLMs' genuine table understanding ability.
>
---
#### [replaced 012] FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01068v4](http://arxiv.org/pdf/2502.01068v4)**

> **作者:** Dongwon Jo; Jiwon Song; Yulhwa Kim; Jae-Joon Kim
>
> **摘要:** While large language models (LLMs) excel at handling long-context sequences, they require substantial prefill computation and key-value (KV) cache, which can heavily burden computational efficiency and memory usage in both prefill and decoding stages. Recent works that compress KV caches with prefill acceleration reduce this cost but inadvertently tie the prefill compute reduction to the decoding KV budget. This coupling arises from overlooking the layer-dependent variation of critical context, often leading to accuracy degradation. To address this issue, we introduce FastKV, a KV cache compression framework designed to reduce latency in both prefill and decoding by leveraging the stabilization of token importance in later layers. FastKV performs full-context computation until a Token-Selective Propagation (TSP) layer, which forwards only the most informative tokens to subsequent layers. From these propagated tokens, FastKV independently selects salient KV entries for caching, thereby decoupling KV budget from the prefill compute reduction based on the TSP decision. This independent control of the TSP rate and KV retention rate enables flexible optimization of efficiency and accuracy. Experimental results show that FastKV achieves speedups of up to 1.82$\times$ in prefill and 2.87$\times$ in decoding compared to the full-context baseline, while matching the accuracy of the baselines that only accelerate the decoding stage. Our code is available at https://github.com/dongwonjo/FastKV.
>
---
#### [replaced 013] Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views
- **分类: cs.CV; cs.CL; cs.RO; I.2.10; I.2.9; I.2.7; H.5.2**

- **链接: [http://arxiv.org/pdf/2510.22672v2](http://arxiv.org/pdf/2510.22672v2)**

> **作者:** Anna Deichler; Jonas Beskow
>
> **备注:** 10 pages, 6 figures, 2 tables. Accepted to the NeurIPS 2025 Workshop on SPACE in Vision, Language, and Embodied AI (SpaVLE). Dataset: https://huggingface.co/datasets/annadeichler/KTH-ARIA-referential
>
> **摘要:** We introduce Look and Tell, a multimodal dataset for studying referential communication across egocentric and exocentric perspectives. Using Meta Project Aria smart glasses and stationary cameras, we recorded synchronized gaze, speech, and video as 25 participants instructed a partner to identify ingredients in a kitchen. Combined with 3D scene reconstructions, this setup provides a benchmark for evaluating how different spatial representations (2D vs. 3D; ego vs. exo) affect multimodal grounding. The dataset contains 3.67 hours of recordings, including 2,707 richly annotated referential expressions, and is designed to advance the development of embodied agents that can understand and engage in situated dialogue.
>
---
#### [replaced 014] DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.14205v2](http://arxiv.org/pdf/2510.14205v2)**

> **作者:** Bingsheng Yao; Bo Sun; Yuanzhe Dong; Yuxuan Lu; Dakuo Wang
>
> **备注:** In Submission
>
> **摘要:** The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF).DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these divergences.We evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie reviews.DPRF can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and scenarios.Our work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.
>
---
#### [replaced 015] Learned, Lagged, LLM-splained: LLM Responses to End User Security Questions
- **分类: cs.CR; cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2411.14571v2](http://arxiv.org/pdf/2411.14571v2)**

> **作者:** Vijay Prakash; Kevin Lee; Arkaprabha Bhattacharya; Danny Yuxing Huang; Jessica Staddon
>
> **备注:** 17 pages, 7 tables
>
> **摘要:** Answering end user security questions is challenging. While large language models (LLMs) like GPT, LLAMA, and Gemini are far from error-free, they have shown promise in answering a variety of questions outside of security. We studied LLM performance in the area of end user security by qualitatively evaluating 3 popular LLMs on 900 systematically collected end user security questions. While LLMs demonstrate broad generalist ``knowledge'' of end user security information, there are patterns of errors and limitations across LLMs consisting of stale and inaccurate answers, and indirect or unresponsive communication styles, all of which impacts the quality of information received. Based on these patterns, we suggest directions for model improvement and recommend user strategies for interacting with LLMs when seeking assistance with security.
>
---
#### [replaced 016] DrVoice: Parallel Speech-Text Voice Conversation Model via Dual-Resolution Speech Representations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09349v3](http://arxiv.org/pdf/2506.09349v3)**

> **作者:** Chao-Hong Tan; Qian Chen; Wen Wang; Chong Deng; Qinglin Zhang; Luyao Cheng; Hai Yu; Xin Zhang; Xiang Lv; Tianyu Zhao; Chong Zhang; Yukun Ma; Yafeng Chen; Hui Wang; Jiaqing Liu; Xiangang Li; Jieping Ye
>
> **备注:** Work in progress
>
> **摘要:** Recent studies on end-to-end (E2E) speech generation with large language models (LLMs) have attracted significant community attention, with multiple works extending text-based LLMs to generate discrete speech tokens. Existing E2E approaches primarily fall into two categories: (1) Methods that generate discrete speech tokens independently without incorporating them into the LLM's autoregressive process, resulting in text generation being unaware of concurrent speech synthesis. (2) Models that generate interleaved or parallel speech- text tokens through joint autoregressive modeling, enabling mutual modality awareness during generation. This paper presents DrVoice, a parallel speech- text voice conversation model based on joint autoregressive modeling, featuring dual-resolution speech representations. Notably, while current methods utilize mainly 12.5Hz input audio representation, our proposed dual-resolution mechanism reduces the input frequency for the LLM to 5Hz, significantly reducing computational cost and alleviating the frequency discrepancy between speech and text tokens and in turn better exploiting LLMs' capabilities. Experimental results demonstrate that DRVOICE-7B establishes new state-of-the-art (SOTA) on OpenAudioBench and Big Bench Audio benchmarks, while achieving performance comparable to the SOTA on VoiceBench and UltraEval-Audio benchmarks, making it a leading open-source speech foundation model in ~7B models.
>
---
#### [replaced 017] Mano Technical Report
- **分类: cs.MM; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.17336v2](http://arxiv.org/pdf/2509.17336v2)**

> **作者:** Tianyu Fu; Anyang Su; Chenxu Zhao; Hanning Wang; Minghui Wu; Zhe Yu; Fei Hu; Mingjia Shi; Wei Dong; Jiayao Wang; Yuyang Chen; Ruiyang Yu; Siran Peng; Menglin Li; Nan Huang; Haitian Wei; Jiawei Yu; Yi Xin; Xilin Zhao; Kai Gu; Ping Jiang; Sifan Zhou; Shuo Wang
>
> **摘要:** Graphical user interfaces (GUIs) are the primary medium for human-computer interaction, yet automating GUI interactions remains challenging due to the complexity of visual elements, dynamic environments, and the need for multi-step reasoning. Existing methods based on vision-language models (VLMs) often suffer from limited resolution, domain mismatch, and insufficient sequential decisionmaking capability. To address these issues, we propose Mano, a robust GUI agent built upon a multi-modal foundation model pre-trained on extensive web and computer system data. Our approach integrates a novel simulated environment for high-fidelity data generation, a three-stage training pipeline (supervised fine-tuning, offline reinforcement learning, and online reinforcement learning), and a verification module for error recovery. Mano demonstrates state-of-the-art performance on multiple GUI benchmarks, including Mind2Web and OSWorld, achieving significant improvements in success rate and operational accuracy. Our work provides new insights into the effective integration of reinforcement learning with VLMs for practical GUI agent deployment, highlighting the importance of domain-specific data, iterative training, and holistic reward design.
>
---
#### [replaced 018] MATCH: Task-Driven Code Evaluation through Contrastive Learning
- **分类: cs.CL; cs.SE**

- **链接: [http://arxiv.org/pdf/2510.23169v2](http://arxiv.org/pdf/2510.23169v2)**

> **作者:** Marah Ghoummaid; Vladimir Tchuiev; Ofek Glick; Michal Moshkovitz; Dotan Di Castro
>
> **摘要:** AI-based code generation is increasingly prevalent, with GitHub Copilot estimated to generate 46% of the code on GitHub. Accurately evaluating how well generated code aligns with developer intent remains a critical challenge. Traditional evaluation methods, such as unit tests, are often unscalable and costly. Syntactic similarity metrics (e.g., BLEU, ROUGE) fail to capture code functionality, and metrics like CodeBERTScore require reference code, which is not always available. To address the gap in reference-free evaluation, with few alternatives such as ICE-Score, this paper introduces MATCH, a novel reference-free metric. MATCH uses Contrastive Learning to generate meaningful embeddings for code and natural language task descriptions, enabling similarity scoring that reflects how well generated code implements the task. We show that MATCH achieves stronger correlations with functional correctness and human preference than existing metrics across multiple programming languages.
>
---
#### [replaced 019] The Dialogue That Heals: A Comprehensive Evaluation of Doctor Agents' Inquiry Capability
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.24958v2](http://arxiv.org/pdf/2509.24958v2)**

> **作者:** Linlu Gong; Ante Wang; Yunghwei Lai; Weizhi Ma; Yang Liu
>
> **摘要:** An effective physician should possess a combination of empathy, expertise, patience, and clear communication when treating a patient. Recent advances have successfully endowed AI doctors with expert diagnostic skills, particularly the ability to actively seek information through inquiry. However, other essential qualities of a good doctor remain overlooked. To bridge this gap, we present MAQuE(Medical Agent Questioning Evaluation), the largest-ever benchmark for the automatic and comprehensive evaluation of medical multi-turn questioning. It features 3,000 realistically simulated patient agents that exhibit diverse linguistic patterns, cognitive limitations, emotional responses, and tendencies for passive disclosure. We also introduce a multi-faceted evaluation framework, covering task success, inquiry proficiency, dialogue competence, inquiry efficiency, and patient experience. Experiments on different LLMs reveal substantial challenges across the evaluation aspects. Even state-of-the-art models show significant room for improvement in their inquiry capabilities. These models are highly sensitive to variations in realistic patient behavior, which considerably impacts diagnostic accuracy. Furthermore, our fine-grained metrics expose trade-offs between different evaluation perspectives, highlighting the challenge of balancing performance and practicality in real-world clinical settings.
>
---
#### [replaced 020] Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05316v3](http://arxiv.org/pdf/2506.05316v3)**

> **作者:** Yifan Sun; Jingyan Shen; Yibin Wang; Tianyu Chen; Zhendong Wang; Mingyuan Zhou; Huan Zhang
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Reinforcement learning (RL) has become an effective approach for fine-tuning large language models (LLMs), particularly to enhance their reasoning capabilities. However, RL fine-tuning remains highly resource-intensive, and existing work has largely overlooked the problem of data efficiency. In this paper, we propose two techniques to improve data efficiency in LLM RL fine-tuning: difficulty-targeted online data selection and rollout replay. We introduce the notion of adaptive difficulty to guide online data selection, prioritizing questions of moderate difficulty that are more likely to yield informative learning signals. To estimate adaptive difficulty efficiently, we develop an attention-based framework that requires rollouts for only a small reference set of questions. The adaptive difficulty of the remaining questions is then estimated based on their similarity to this set. To further reduce rollout cost, we introduce a rollout replay mechanism inspired by experience replay in traditional RL. This technique reuses recent rollouts, lowering per-step computation while maintaining stable updates. Experiments across 6 LLM-dataset combinations show that our method reduces RL fine-tuning time by 23% to 62% while reaching the same level of performance as the original GRPO algorithm. Our code is available at https://github.com/ASTRAL-Group/data-efficient-llm-rl.
>
---
#### [replaced 021] Context-level Language Modeling by Learning Predictive Context Embeddings
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.20280v2](http://arxiv.org/pdf/2510.20280v2)**

> **作者:** Beiya Dai; Yuliang Liu; Daozheng Xue; Qipeng Guo; Kai Chen; Xinbing Wang; Bowen Zhou; Zhouhan Lin
>
> **备注:** 16pages,6 figures
>
> **摘要:** Next-token prediction (NTP) is the cornerstone of modern large language models (LLMs) pretraining, driving their unprecedented capabilities in text generation, reasoning, and instruction following. However, the token-level prediction limits the model's capacity to capture higher-level semantic structures and long-range contextual relationships. To overcome this limitation, we introduce \textbf{ContextLM}, a framework that augments standard pretraining with an inherent \textbf{next-context prediction} objective. This mechanism trains the model to learn predictive representations of multi-token contexts, leveraging error signals derived from future token chunks. Crucially, ContextLM achieves this enhancement while remaining fully compatible with the standard autoregressive, token-by-token evaluation paradigm (e.g., perplexity). Extensive experiments on the GPT2 and Pythia model families, scaled up to $1.5$B parameters, show that ContextLM delivers consistent improvements in both perplexity and downstream task performance. Our analysis indicates that next-context prediction provides a scalable and efficient pathway to stronger language modeling, yielding better long-range coherence and more effective attention allocation with minimal computational overhead.
>
---
#### [replaced 022] LittleBit: Ultra Low-Bit Quantization via Latent Factorization
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.13771v2](http://arxiv.org/pdf/2506.13771v2)**

> **作者:** Banseok Lee; Dongkyu Kim; Youngcheon You; Youngmin Kim
>
> **备注:** Accepted to NeurIPS 2025. Banseok Lee and Dongkyu Kim contributed equally
>
> **摘要:** Deploying large language models (LLMs) often faces challenges from substantial memory and computational costs. Quantization offers a solution, yet performance degradation in the sub-1-bit regime remains particularly difficult. This paper introduces LittleBit, a novel method for extreme LLM compression. It targets levels like 0.1 bits per weight (BPW), achieving nearly 31$\times$ memory reduction, e.g., Llama2-13B to under 0.9 GB. LittleBit represents weights in a low-rank form using latent matrix factorization, subsequently binarizing these factors. To counteract information loss from this extreme precision, it integrates a multi-scale compensation mechanism. This includes row, column, and an additional latent dimension that learns per-rank importance. Two key contributions enable effective training: Dual Sign-Value-Independent Decomposition (Dual-SVID) for quantization-aware training (QAT) initialization, and integrated Residual Compensation to mitigate errors. Extensive experiments confirm LittleBit's superiority in sub-1-bit quantization: e.g., its 0.1 BPW performance on Llama2-7B surpasses the leading method's 0.7 BPW. LittleBit establishes a new, viable size-performance trade-off--unlocking a potential 11.6$\times$ speedup over FP16 at the kernel level--and makes powerful LLMs practical for resource-constrained environments.
>
---
#### [replaced 023] Zero-Shot Tokenizer Transfer
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2405.07883v2](http://arxiv.org/pdf/2405.07883v2)**

> **作者:** Benjamin Minixhofer; Edoardo Maria Ponti; Ivan Vulić
>
> **备注:** NeurIPS 2024
>
> **摘要:** Language models (LMs) are bound to their tokenizer, which maps raw text to a sequence of vocabulary items (tokens). This restricts their flexibility: for example, LMs trained primarily on English may still perform well in other natural and programming languages, but have vastly decreased efficiency due to their English-centric tokenizer. To mitigate this, we should be able to swap the original LM tokenizer with an arbitrary one, on the fly, without degrading performance. Hence, in this work we define a new problem: Zero-Shot Tokenizer Transfer (ZeTT). The challenge at the core of ZeTT is finding embeddings for the tokens in the vocabulary of the new tokenizer. Since prior heuristics for initializing embeddings often perform at chance level in a ZeTT setting, we propose a new solution: we train a hypernetwork taking a tokenizer as input and predicting the corresponding embeddings. We empirically demonstrate that the hypernetwork generalizes to new tokenizers both with encoder (e.g., XLM-R) and decoder LLMs (e.g., Mistral-7B). Our method comes close to the original models' performance in cross-lingual and coding tasks while markedly reducing the length of the tokenized sequence. We also find that the remaining gap can be quickly closed by continued training on less than 1B tokens. Finally, we show that a ZeTT hypernetwork trained for a base (L)LM can also be applied to fine-tuned variants without extra training. Overall, our results make substantial strides toward detaching LMs from their tokenizer.
>
---
#### [replaced 024] Retrieval-Augmented Generation-based Relation Extraction
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2404.13397v2](http://arxiv.org/pdf/2404.13397v2)**

> **作者:** Sefika Efeoglu; Adrian Paschke
>
> **备注:** published at the Semantic Web journal. The last version is available: https://doi.org/10.1177/22104968251385519
>
> **摘要:** Information Extraction (IE) is a transformative process that converts unstructured text data into a structured format by employing entity and relation extraction (RE) methodologies. The identification of the relation between a pair of entities plays a crucial role within this framework. Despite the existence of various techniques for relation extraction, their efficacy heavily relies on access to labeled data and substantial computational resources. In addressing these challenges, Large Language Models (LLMs) emerge as promising solutions; however, they might return hallucinating responses due to their own training data. To overcome these limitations, Retrieved-Augmented Generation-based Relation Extraction (RAG4RE) in this work is proposed, offering a pathway to enhance the performance of relation extraction tasks. This work evaluated the effectiveness of our RAG4RE approach utilizing different LLMs. Through the utilization of established benchmarks, such as TACRED, TACREV, Re-TACRED, and SemEval RE datasets, our aim is to comprehensively evaluate the efficacy of our RAG4RE approach. In particularly, we leverage prominent LLMs including Flan T5, Llama2, and Mistral in our investigation. The results of our study demonstrate that our RAG4RE approach surpasses performance of traditional RE approaches based solely on LLMs, particularly evident in the TACRED dataset and its variations. Furthermore, our approach exhibits remarkable performance compared to previous RE methodologies across both TACRED and TACREV datasets, underscoring its efficacy and potential for advancing RE tasks in natural language processing.
>
---
#### [replaced 025] PVP: An Image Dataset for Personalized Visual Persuasion with Persuasion Strategies, Viewer Characteristics, and Persuasiveness Ratings
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.00481v2](http://arxiv.org/pdf/2506.00481v2)**

> **作者:** Junseo Kim; Jongwook Han; Dongmin Choi; Jongwook Yoon; Eun-Ju Lee; Yohan Jo
>
> **备注:** ACL 2025 Main. Code and dataset are released at: https://github.com/holi-lab/PVP_Personalized_Visual_Persuasion
>
> **摘要:** Visual persuasion, which uses visual elements to influence cognition and behaviors, is crucial in fields such as advertising and political communication. With recent advancements in artificial intelligence, there is growing potential to develop persuasive systems that automatically generate persuasive images tailored to individuals. However, a significant bottleneck in this area is the lack of comprehensive datasets that connect the persuasiveness of images with the personal information about those who evaluated the images. To address this gap and facilitate technological advancements in personalized visual persuasion, we release the Personalized Visual Persuasion (PVP) dataset, comprising 28,454 persuasive images across 596 messages and 9 persuasion strategies. Importantly, the PVP dataset provides persuasiveness scores of images evaluated by 2,521 human annotators, along with their demographic and psychological characteristics (personality traits and values). We demonstrate the utility of our dataset by developing a persuasive image generator and an automated evaluator, and establish benchmark baselines. Our experiments reveal that incorporating psychological characteristics enhances the generation and evaluation of persuasive images, providing valuable insights for personalized visual persuasion.
>
---
#### [replaced 026] AdaRewriter: Unleashing the Power of Prompting-based Conversational Query Reformulation via Test-Time Adaptation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01381v2](http://arxiv.org/pdf/2506.01381v2)**

> **作者:** Yilong Lai; Jialong Wu; Zhenglin Wang; Deyu Zhou
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Prompting-based conversational query reformulation has emerged as a powerful approach for conversational search, refining ambiguous user queries into standalone search queries. Best-of-N reformulation over the generated candidates via prompting shows impressive potential scaling capability. However, both the previous tuning methods (training time) and adaptation approaches (test time) can not fully unleash their benefits. In this paper, we propose AdaRewriter, a novel framework for query reformulation using an outcome-supervised reward model via test-time adaptation. By training a lightweight reward model with contrastive ranking loss, AdaRewriter selects the most promising reformulation during inference. Notably, it can operate effectively in black-box systems, including commercial LLM APIs. Experiments on five conversational search datasets show that AdaRewriter significantly outperforms the existing methods across most settings, demonstrating the potential of test-time adaptation for conversational query reformulation.
>
---
#### [replaced 027] LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.10114v3](http://arxiv.org/pdf/2510.10114v3)**

> **作者:** Luyao Zhuang; Shengyuan Chen; Yilin Xiao; Huachi Zhou; Yujing Zhang; Hao Chen; Qinggang Zhang; Xiao Huang
>
> **摘要:** Retrieval-Augmented Generation (RAG) is widely used to mitigate hallucinations of Large Language Models (LLMs) by leveraging external knowledge. While effective for simple queries, traditional RAG systems struggle with large-scale, unstructured corpora where information is fragmented. Recent advances incorporate knowledge graphs to capture relational structures, enabling more comprehensive retrieval for complex, multi-hop reasoning tasks. However, existing graph-based RAG (GraphRAG) methods rely on unstable and costly relation extraction for graph construction, often producing noisy graphs with incorrect or inconsistent relations that degrade retrieval quality. In this paper, we revisit the pipeline of existing GraphRAG systems and propose LinearRAG (Linear Graph-based Retrieval-Augmented Generation), an efficient framework that enables reliable graph construction and precise passage retrieval. Specifically, LinearRAG constructs a relation-free hierarchical graph, termed Tri-Graph, using only lightweight entity extraction and semantic linking, avoiding unstable relation modeling. This new paradigm of graph construction scales linearly with corpus size and incurs no extra token consumption, providing an economical and reliable indexing of the original passages. For retrieval, LinearRAG adopts a two-stage strategy: (i) relevant entity activation via local semantic bridging, followed by (ii) passage retrieval through global importance aggregation. Extensive experiments on four datasets demonstrate that LinearRAG significantly outperforms baseline models. Our code and datasets are available at https://github.com/DEEP-PolyU/LinearRAG.
>
---
#### [replaced 028] OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.15870v2](http://arxiv.org/pdf/2510.15870v2)**

> **作者:** Hanrong Ye; Chao-Han Huck Yang; Arushi Goel; Wei Huang; Ligeng Zhu; Yuanhang Su; Sean Lin; An-Chieh Cheng; Zhen Wan; Jinchuan Tian; Yuming Lou; Dong Yang; Zhijian Liu; Yukang Chen; Ambrish Dantrey; Ehsan Jahangiri; Sreyan Ghosh; Daguang Xu; Ehsan Hosseini-Asl; Danial Mohseni Taheri; Vidya Murali; Sifei Liu; Yao Lu; Oluwatobi Olabiyi; Yu-Chiang Frank Wang; Rafael Valle; Bryan Catanzaro; Andrew Tao; Song Han; Jan Kautz; Hongxu Yin; Pavlo Molchanov
>
> **备注:** Technical Report. Code: https://github.com/NVlabs/OmniVinci
>
> **摘要:** Advancing machine intelligence requires developing the ability to perceive across multiple modalities, much as humans sense the world. We introduce OmniVinci, an initiative to build a strong, open-source, omni-modal LLM. We carefully study the design choices across model architecture and data curation. For model architecture, we present three key innovations: (i) OmniAlignNet for strengthening alignment between vision and audio embeddings in a shared omni-modal latent space; (ii) Temporal Embedding Grouping for capturing relative temporal alignment between vision and audio signals; and (iii) Constrained Rotary Time Embedding for encoding absolute temporal information in omni-modal embeddings. We introduce a curation and synthesis pipeline that generates 24M single-modal and omni-modal conversations. We find that modalities reinforce one another in both perception and reasoning. Our model, OmniVinci, outperforms Qwen2.5-Omni with +19.05 on DailyOmni (cross-modal understanding), +1.7 on MMAR (audio), and +3.9 on Video-MME (vision), while using just 0.2T training tokens - a 6 times reduction compared to Qwen2.5-Omni's 1.2T. We finally demonstrate omni-modal advantages in downstream applications spanning robotics, medical AI, and smart factory.
>
---
#### [replaced 029] Offline Learning and Forgetting for Reasoning with Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.11364v4](http://arxiv.org/pdf/2504.11364v4)**

> **作者:** Tianwei Ni; Allen Nie; Sapana Chaudhary; Yao Liu; Huzefa Rangwala; Rasool Fakoor
>
> **备注:** Published in Transactions on Machine Learning Research (TMLR), 2025. Code: https://github.com/twni2016/llm-reasoning-uft
>
> **摘要:** Leveraging inference-time search in large language models has proven effective in further enhancing a trained model's capability to solve complex mathematical and reasoning problems. However, this approach significantly increases computational costs and inference time, as the model must generate and evaluate multiple candidate solutions to identify a viable reasoning path. To address this, we propose an effective approach that integrates search capabilities directly into the model by fine-tuning it on unpaired successful (learning) and failed reasoning paths (forgetting) derived from diverse search methods. A key challenge we identify is that naive fine-tuning can degrade the model's search capability; we show this can be mitigated with a smaller learning rate. Extensive experiments on the challenging Game-of-24 and Countdown arithmetic puzzles show that, replacing CoT-generated data with search-generated data for offline fine-tuning improves success rates by around 23% over inference-time search baselines, while reducing inference time by 180$\times$. On top of this, our learning and forgetting objective consistently outperforms both supervised fine-tuning and preference-based methods.
>
---
#### [replaced 030] BRIDGE: Benchmarking Large Language Models for Understanding Real-world Clinical Practice Text
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.19467v3](http://arxiv.org/pdf/2504.19467v3)**

> **作者:** Jiageng Wu; Bowen Gu; Ren Zhou; Kevin Xie; Doug Snyder; Yixing Jiang; Valentina Carducci; Richard Wyss; Rishi J Desai; Emily Alsentzer; Leo Anthony Celi; Adam Rodman; Sebastian Schneeweiss; Jonathan H. Chen; Santiago Romero-Brufau; Kueiyu Joshua Lin; Jie Yang
>
> **摘要:** Large language models (LLMs) hold great promise for medical applications and are evolving rapidly, with new models being released at an accelerated pace. However, benchmarking on large-scale real-world data such as electronic health records (EHRs) is critical, as clinical decisions are directly informed by these sources, yet current evaluations remain limited. Most existing benchmarks rely on medical exam-style questions or PubMed-derived text, failing to capture the complexity of real-world clinical data. Others focus narrowly on specific application scenarios, limiting their generalizability across broader clinical use. To address this gap, we present BRIDGE, a comprehensive multilingual benchmark comprising 87 tasks sourced from real-world clinical data sources across nine languages. It covers eight major task types spanning the entire continuum of patient care across six clinical stages and 20 representative applications, including triage and referral, consultation, information extraction, diagnosis, prognosis, and billing coding, and involves 14 clinical specialties. We systematically evaluated 95 LLMs (including DeepSeek-R1, GPT-4o, Gemini series, and Qwen3 series) under various inference strategies. Our results reveal substantial performance variation across model sizes, languages, natural language processing tasks, and clinical specialties. Notably, we demonstrate that open-source LLMs can achieve performance comparable to proprietary models, while medically fine-tuned LLMs based on older architectures often underperform versus updated general-purpose models. The BRIDGE and its corresponding leaderboard serve as a foundational resource and a unique reference for the development and evaluation of new LLMs in real-world clinical text understanding. The BRIDGE leaderboard: https://huggingface.co/spaces/YLab-Open/BRIDGE-Medical-Leaderboard
>
---
#### [replaced 031] MENTOR: A Reinforcement Learning Framework for Enabling Tool Use in Small Models via Teacher-Optimized Rewards
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.18383v2](http://arxiv.org/pdf/2510.18383v2)**

> **作者:** ChangSu Choi; Hoyun Song; Dongyeon Kim; WooHyeon Jung; Minkyung Cho; Sunjin Park; NohHyeob Bae; Seona Yu; KyungTae Lim
>
> **摘要:** Distilling the tool-using capabilities of large language models (LLMs) into smaller, more efficient small language models (SLMs) is a key challenge for their practical application. The predominant approach, supervised fine-tuning (SFT), suffers from poor generalization as it trains models to imitate a static set of teacher trajectories rather than learn a robust methodology. While reinforcement learning (RL) offers an alternative, the standard RL using sparse rewards fails to effectively guide SLMs, causing them to struggle with inefficient exploration and adopt suboptimal strategies. To address these distinct challenges, we propose MENTOR, a framework that synergistically combines RL with teacher-guided distillation. Instead of simple imitation, MENTOR employs an RL-based process to learn a more generalizable policy through exploration. In addition, to solve the problem of reward sparsity, it uses a teacher's reference trajectory to construct a dense, composite teacher-guided reward that provides fine-grained guidance. Extensive experiments demonstrate that MENTOR significantly improves the cross-domain generalization and strategic competence of SLMs compared to both SFT and standard sparse-reward RL baselines.
>
---
#### [replaced 032] Provable Scaling Laws for the Test-Time Compute of Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19477v5](http://arxiv.org/pdf/2411.19477v5)**

> **作者:** Yanxi Chen; Xuchen Pan; Yaliang Li; Bolin Ding; Jingren Zhou
>
> **备注:** NeurIPS 2025 camera-ready version
>
> **摘要:** We propose two simple, principled and practical algorithms that enjoy provable scaling laws for the test-time compute of large language models (LLMs). The first one is a two-stage knockout-style algorithm: given an input problem, it first generates multiple candidate solutions, and then aggregate them via a knockout tournament for the final output. Assuming that the LLM can generate a correct solution with non-zero probability and do better than a random guess in comparing a pair of correct and incorrect solutions, we prove theoretically that the failure probability of this algorithm decays to zero exponentially or by a power law (depending on the specific way of scaling) as its test-time compute grows. The second one is a two-stage league-style algorithm, where each candidate is evaluated by its average win rate against multiple opponents, rather than eliminated upon loss to a single opponent. Under analogous but more robust assumptions, we prove that its failure probability also decays to zero exponentially with more test-time compute. Both algorithms require a black-box LLM and nothing else (e.g., no verifier or reward model) for a minimalistic implementation, which makes them appealing for practical applications and easy to adapt for different tasks. Through extensive experiments with diverse models and datasets, we validate the proposed theories and demonstrate the outstanding scaling properties of both algorithms.
>
---
#### [replaced 033] The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2505.14617v3](http://arxiv.org/pdf/2505.14617v3)**

> **作者:** Sahar Abdelnabi; Ahmed Salem
>
> **备注:** NeurIPS 2025 (Spotlight). Code is available at: https://github.com/microsoft/Test_Awareness_Steering
>
> **摘要:** Reasoning-focused LLMs sometimes alter their behavior when they detect that they are being evaluated, which can lead them to optimize for test-passing performance or to comply more readily with harmful prompts if real-world consequences appear absent. We present the first quantitative study of how such "test awareness" impacts model behavior, particularly its performance on safety-related tasks. We introduce a white-box probing framework that (i) linearly identifies awareness-related activations and (ii) steers models toward or away from test awareness while monitoring downstream performance. We apply our method to different state-of-the-art open-weight reasoning LLMs across both realistic and hypothetical tasks (denoting tests or simulations). Our results demonstrate that test awareness significantly impacts safety alignment (such as compliance with harmful requests and conforming to stereotypes) with effects varying in both magnitude and direction across models. By providing control over this latent effect, our work aims to provide a stress-test mechanism and increase trust in how we perform safety evaluations.
>
---
#### [replaced 034] Exploration of Summarization by Generative Language Models for Automated Scoring of Long Essays
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.22830v2](http://arxiv.org/pdf/2510.22830v2)**

> **作者:** Haowei Hua; Hong Jiao; Xinyi Wang
>
> **备注:** 19 pages, 5 Tables 7 Figures, Presentation at Artificial Intelligence in Measurement and Education Conference (AIME-Con)
>
> **摘要:** BERT and its variants are extensively explored for automated scoring. However, a limit of 512 tokens for these encoder-based models showed the deficiency in automated scoring of long essays. Thus, this research explores generative language models for automated scoring of long essays via summarization and prompting. The results revealed great improvement of scoring accuracy with QWK increased from 0.822 to 0.8878 for the Learning Agency Lab Automated Essay Scoring 2.0 dataset.
>
---
#### [replaced 035] TableTime: Reformulating Time Series Classification as Training-Free Table Understanding with Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.15737v4](http://arxiv.org/pdf/2411.15737v4)**

> **作者:** Jiahao Wang; Mingyue Cheng; Qingyang Mao; Yitong Zhou; Daoyu Wang; Qi Liu; Feiyang Xu; Xin Li
>
> **摘要:** Large language models (LLMs) have demonstrated their effectiveness in multivariate time series classification (MTSC). Effective adaptation of LLMs for MTSC necessitates informative data representations. Existing LLM-based methods directly encode embeddings for time series within the latent space of LLMs from scratch to align with semantic space of LLMs. Despite their effectiveness, we reveal that these methods conceal three inherent bottlenecks: (1) they struggle to encode temporal and channel-specific information in a lossless manner, both of which are critical components of multivariate time series; (2) it is much difficult to align the learned representation space with the semantic space of the LLMs; (3) they require task-specific retraining, which is both computationally expensive and labor-intensive. To bridge these gaps, we propose TableTime, which reformulates MTSC as a table understanding task. Specifically, TableTime introduces the following strategies: (1) convert multivariate time series into a tabular form, thus minimizing information loss to the greatest extent; (2) represent tabular time series in text format to achieve natural alignment with the semantic space of LLMs; (3) design a reasoning framework that integrates contextual text information, neighborhood assistance, multi-path inference and problem decomposition to enhance the reasoning ability of LLMs and realize zero-shot classification. Extensive experiments performed on 10 publicly representative datasets from UEA archive verify the superiorities of the TableTime.
>
---
#### [replaced 036] Says Who? Effective Zero-Shot Annotation of Focalization
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.11390v3](http://arxiv.org/pdf/2409.11390v3)**

> **作者:** Rebecca M. M. Hicke; Yuri Bizzoni; Pascale Feldkamp; Ross Deans Kristensen-McLachlan
>
> **备注:** Accepted at CHR 2025
>
> **摘要:** Focalization describes the way in which access to narrative information is restricted or controlled based on the knowledge available to knowledge of the narrator. It is encoded via a wide range of lexico-grammatical features and is subject to reader interpretation. Even trained annotators frequently disagree on correct labels, suggesting this task is both qualitatively and computationally challenging. In this work, we test how well five contemporary large language model (LLM) families and two baselines perform when annotating short literary excerpts for focalization. Despite the challenging nature of the task, we find that LLMs show comparable performance to trained human annotators, with GPT-4o achieving an average F1 of 84.79%. Further, we demonstrate that the log probabilities output by GPT-family models frequently reflect the difficulty of annotating particular excerpts. Finally, we provide a case study analyzing sixteen Stephen King novels, demonstrating the usefulness of this approach for computational literary studies and the insights gleaned from examining focalization at scale.
>
---
#### [replaced 037] AutoJudge: Judge Decoding Without Manual Annotation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.20039v3](http://arxiv.org/pdf/2504.20039v3)**

> **作者:** Roman Garipov; Fedor Velikonivtsev; Ivan Ermakov; Ruslan Svirschevski; Vage Egiazarian; Max Ryabinin
>
> **摘要:** We introduce AutoJudge, a method that accelerates large language model (LLM) inference with task-specific lossy speculative decoding. Instead of matching the original model output distribution token-by-token, we identify which of the generated tokens affect the downstream quality of the response, relaxing the distribution match guarantee so that the "unimportant" tokens can be generated faster. Our approach relies on a semi-greedy search algorithm to test which of the mismatches between target and draft models should be corrected to preserve quality and which ones may be skipped. We then train a lightweight classifier based on existing LLM embeddings to predict, at inference time, which mismatching tokens can be safely accepted without compromising the final answer quality. We evaluate the effectiveness of AutoJudge with multiple draft/target model pairs on mathematical reasoning and programming benchmarks, achieving significant speedups at the cost of a minor accuracy reduction. Notably, on GSM8k with the Llama 3.1 70B target model, our approach achieves up to $\approx2\times$ speedup over speculative decoding at the cost of $\le 1\%$ drop in accuracy. When applied to the LiveCodeBench benchmark, AutoJudge automatically detects programming-specific important tokens, accepting $\ge 25$ tokens per speculation cycle at $2\%$ drop in Pass@1. Our approach requires no human annotation and is easy to integrate with modern LLM inference frameworks.
>
---
#### [replaced 038] Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17100v2](http://arxiv.org/pdf/2505.17100v2)**

> **作者:** Haoyan Yang; Runxue Bao; Cao Xiao; Jun Ma; Parminder Bhatia; Shangqian Gao; Taha Kass-Hout
>
> **备注:** Accepted at NeurIPS 2025 (Camera-Ready Version)
>
> **摘要:** LLM-as-a-Judge has emerged as a promising tool for automatically evaluating generated outputs, but its reliability is often undermined by potential biases in judgment. Existing efforts to mitigate these biases face key limitations: in-context learning-based methods fail to address rooted biases due to the evaluator's limited capacity for self-reflection, whereas fine-tuning is not applicable to all evaluator types, especially closed-source models. To address this challenge, we introduce the Reasoning-based Bias Detector (RBD), which is a plug-in module that identifies biased evaluations and generates structured reasoning to guide evaluator self-correction. Rather than modifying the evaluator itself, RBD operates externally and engages in an iterative process of bias detection and feedback-driven revision. To support its development, we design a complete pipeline consisting of biased dataset construction, supervision collection, distilled reasoning-based fine-tuning of RBD, and integration with LLM evaluators. We fine-tune four sizes of RBD models, ranging from 1.5B to 14B, and observe consistent performance improvements across all scales. Experimental results on 4 bias types--verbosity, position, bandwagon, and sentiment--evaluated using 8 LLM evaluators demonstrate RBD's strong effectiveness. For example, the RBD-8B model improves evaluation accuracy by an average of 18.5% and consistency by 10.9%, and surpasses prompting-based baselines and fine-tuned judges by 12.8% and 17.2%, respectively. These results highlight RBD's effectiveness and scalability. Additional experiments further demonstrate its strong generalization across biases and domains, as well as its efficiency.
>
---
#### [replaced 039] AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2510.01268v3](http://arxiv.org/pdf/2510.01268v3)**

> **作者:** Hongyi Zhou; Jin Zhu; Pingfan Su; Kai Ye; Ying Yang; Shakeel A O B Gavioli-Akilagun; Chengchun Shi
>
> **备注:** Accepted by NeurIPS2025
>
> **摘要:** We study the problem of determining whether a piece of text has been authored by a human or by a large language model (LLM). Existing state of the art logits-based detectors make use of statistics derived from the log-probability of the observed text evaluated using the distribution function of a given source LLM. However, relying solely on log probabilities can be sub-optimal. In response, we introduce AdaDetectGPT -- a novel classifier that adaptively learns a witness function from training data to enhance the performance of logits-based detectors. We provide statistical guarantees on its true positive rate, false positive rate, true negative rate and false negative rate. Extensive numerical studies show AdaDetectGPT nearly uniformly improves the state-of-the-art method in various combination of datasets and LLMs, and the improvement can reach up to 37\%. A python implementation of our method is available at https://github.com/Mamba413/AdaDetectGPT.
>
---
#### [replaced 040] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11842v3](http://arxiv.org/pdf/2505.11842v3)**

> **作者:** Xuannan Liu; Zekun Li; Zheqi He; Peipei Li; Shuhan Xia; Xing Cui; Huaibo Huang; Xi Yang; Ran He
>
> **备注:** Accepted by NeurIPS 2025 Dataset and Benchmark Track, Project page: https://liuxuannan.github.io/Video-SafetyBench.github.io/
>
> **摘要:** The increasing deployment of Large Vision-Language Models (LVLMs) raises safety concerns under potential malicious inputs. However, existing multimodal safety evaluations primarily focus on model vulnerabilities exposed by static image inputs, ignoring the temporal dynamics of video that may induce distinct safety risks. To bridge this gap, we introduce Video-SafetyBench, the first comprehensive benchmark designed to evaluate the safety of LVLMs under video-text attacks. It comprises 2,264 video-text pairs spanning 48 fine-grained unsafe categories, each pairing a synthesized video with either a harmful query, which contains explicit malice, or a benign query, which appears harmless but triggers harmful behavior when interpreted alongside the video. To generate semantically accurate videos for safety evaluation, we design a controllable pipeline that decomposes video semantics into subject images (what is shown) and motion text (how it moves), which jointly guide the synthesis of query-relevant videos. To effectively evaluate uncertain or borderline harmful outputs, we propose RJScore, a novel LLM-based metric that incorporates the confidence of judge models and human-aligned decision threshold calibration. Extensive experiments show that benign-query video composition achieves average attack success rates of 67.2%, revealing consistent vulnerabilities to video-induced attacks. We believe Video-SafetyBench will catalyze future research into video-based safety evaluation and defense strategies.
>
---
#### [replaced 041] Offline RL by Reward-Weighted Fine-Tuning for Conversation Optimization
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06964v2](http://arxiv.org/pdf/2506.06964v2)**

> **作者:** Subhojyoti Mukherjee; Viet Dac Lai; Raghavendra Addanki; Ryan Rossi; Seunghyun Yoon; Trung Bui; Anup Rao; Jayakumar Subramanian; Branislav Kveton
>
> **备注:** Accepted at NeurIPS 2025 (main conference)
>
> **摘要:** Offline reinforcement learning (RL) is a variant of RL where the policy is learned from a previously collected dataset of trajectories and rewards. In our work, we propose a practical approach to offline RL with large language models (LLMs). We recast the problem as reward-weighted fine-tuning, which can be solved using similar techniques to supervised fine-tuning (SFT). To showcase the value of our approach, we apply it to learning short-horizon question-answering policies of a fixed length, where the agent reasons about potential answers or asks clarifying questions. Our work stands in a stark contrast to state-of-the-art methods in this domain, based on SFT and direct preference optimization, which have additional hyper-parameters and do not directly optimize for rewards. We compare to them empirically, and report major gains in both optimized rewards and language quality.
>
---
#### [replaced 042] Semantic Agreement Enables Efficient Open-Ended LLM Cascades
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21837v3](http://arxiv.org/pdf/2509.21837v3)**

> **作者:** Duncan Soiffer; Steven Kolawole; Virginia Smith
>
> **备注:** 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP) Industry Track
>
> **摘要:** Cascade systems route computational requests to smaller models when possible and defer to larger models only when necessary, offering a promising approach to balance cost and quality in LLM deployment. However, they face a fundamental challenge in open-ended text generation: determining output reliability when generation quality lies on a continuous spectrum, often with multiple valid responses. To address this, we propose semantic agreement -- meaning-level consensus between ensemble outputs -- as a training-free signal for reliable deferral. We show that when diverse model outputs agree semantically, their consensus is a stronger reliability signal than token-level confidence. Evaluated from 500M to 70B-parameter models, we find that semantic cascades match or surpass target-model quality at 40% of the cost and reduce latency by up to 60%. Our method requires no model internals, works across black-box APIs, and remains robust to model updates, making it a practical baseline for real-world LLM deployment.
>
---
#### [replaced 043] Surface Reading LLMs: Synthetic Text and its Styles
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.22162v2](http://arxiv.org/pdf/2510.22162v2)**

> **作者:** Hannes Bajohr
>
> **备注:** 12 pages, 1 figure
>
> **摘要:** Despite a potential plateau in ML advancement, the societal impact of large language models lies not in approaching superintelligence but in generating text surfaces indistinguishable from human writing. While Critical AI Studies provides essential material and socio-technical critique, it risks overlooking how LLMs phenomenologically reshape meaning-making. This paper proposes a semiotics of "surface integrity" as attending to the immediate plane where LLMs inscribe themselves into human communication. I distinguish three knowledge interests in ML research (epistemology, epist\=em\=e, and epistemics) and argue for integrating surface-level stylistic analysis alongside depth-oriented critique. Through two case studies examining stylistic markers of synthetic text, I argue how attending to style as a semiotic phenomenon reveals LLMs as cultural actors that transform the conditions of meaning emergence and circulation in contemporary discourse, independent of questions about machine consciousness.
>
---
#### [replaced 044] Detecting Latin in Historical Books with Large Language Models: A Multimodal Benchmark
- **分类: cs.CL; cs.AI; cs.CV; cs.DL**

- **链接: [http://arxiv.org/pdf/2510.19585v2](http://arxiv.org/pdf/2510.19585v2)**

> **作者:** Yu Wu; Ke Shu; Jonas Fischer; Lidia Pivovarova; David Rosson; Eetu Mäkelä; Mikko Tolonen
>
> **备注:** Under review. Both the dataset and code will be published
>
> **摘要:** This paper presents a novel task of extracting Latin fragments from mixed-language historical documents with varied layouts. We benchmark and evaluate the performance of large foundation models against a multimodal dataset of 724 annotated pages. The results demonstrate that reliable Latin detection with contemporary models is achievable. Our study provides the first comprehensive analysis of these models' capabilities and limits for this task.
>
---
#### [replaced 045] RARE: Retrieval-Aware Robustness Evaluation for Retrieval-Augmented Generation Systems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00789v3](http://arxiv.org/pdf/2506.00789v3)**

> **作者:** Yixiao Zeng; Tianyu Cao; Danqing Wang; Xinran Zhao; Zimeng Qiu; Morteza Ziyadi; Tongshuang Wu; Lei Li
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances recency and factuality in answers. However, existing evaluations rarely test how well these systems cope with real-world noise, conflicting between internal and external retrieved contexts, or fast-changing facts. We introduce Retrieval-Aware Robustness Evaluation (RARE), a unified framework and large-scale benchmark that jointly stress-tests query and document perturbations over dynamic, time-sensitive corpora. One of the central features of RARE is a knowledge-graph-driven synthesis pipeline (RARE-Get) that automatically extracts single and multi-hop relations from the customized corpus and generates multi-level question sets without manual intervention. Leveraging this pipeline, we construct a dataset (RARE-Set) spanning 527 expert-level time-sensitive finance, economics, and policy documents and 48295 questions whose distribution evolves as the underlying sources change. To quantify resilience, we formalize retrieval-conditioned robustness metrics (RARE-Met) that capture a model's ability to remain correct or recover when queries, documents, or real-world retrieval results are systematically altered. Our findings reveal that RAG systems are unexpectedly sensitive to perturbations. Moreover, they consistently demonstrate lower robustness on multi-hop queries compared to single-hop queries across all domains.
>
---
#### [replaced 046] Are you sure? Measuring models bias in content moderation through uncertainty
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.22699v2](http://arxiv.org/pdf/2509.22699v2)**

> **作者:** Alessandra Urbinati; Mirko Lai; Simona Frenda; Marco Antonio Stranisci
>
> **备注:** accepted at Findings of ACL: EMNLP 2025
>
> **摘要:** Automatic content moderation is crucial to ensuring safety in social media. Language Model-based classifiers are being increasingly adopted for this task, but it has been shown that they perpetuate racial and social biases. Even if several resources and benchmark corpora have been developed to challenge this issue, measuring the fairness of models in content moderation remains an open issue. In this work, we present an unsupervised approach that benchmarks models on the basis of their uncertainty in classifying messages annotated by people belonging to vulnerable groups. We use uncertainty, computed by means of the conformal prediction technique, as a proxy to analyze the bias of 11 models against women and non-white annotators and observe to what extent it diverges from metrics based on performance, such as the $F_1$ score. The results show that some pre-trained models predict with high accuracy the labels coming from minority groups, even if the confidence in their prediction is low. Therefore, by measuring the confidence of models, we are able to see which groups of annotators are better represented in pre-trained models and lead the debiasing process of these models before their effective use.
>
---
#### [replaced 047] Discourse Features Enhance Detection of Document-Level Machine-Generated Content
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.12679v2](http://arxiv.org/pdf/2412.12679v2)**

> **作者:** Yupei Li; Manuel Milling; Lucia Specia; Björn W. Schuller
>
> **备注:** Accepted by IJCNN 2025
>
> **摘要:** The availability of high-quality APIs for Large Language Models (LLMs) has facilitated the widespread creation of Machine-Generated Content (MGC), posing challenges such as academic plagiarism and the spread of misinformation. Existing MGC detectors often focus solely on surface-level information, overlooking implicit and structural features. This makes them susceptible to deception by surface-level sentence patterns, particularly for longer texts and in texts that have been subsequently paraphrased. To overcome these challenges, we introduce novel methodologies and datasets. Besides the publicly available dataset Plagbench, we developed the paraphrased Long-Form Question and Answer (paraLFQA) and paraphrased Writing Prompts (paraWP) datasets using GPT and DIPPER, a discourse paraphrasing tool, by extending artifacts from their original versions. To better capture the structure of longer texts at document level, we propose DTransformer, a model that integrates discourse analysis through PDTB preprocessing to encode structural features. It results in substantial performance gains across both datasets - 15.5% absolute improvement on paraLFQA, 4% absolute improvement on paraWP, and 1.5% absolute improvemene on M4 compared to SOTA approaches. The data and code are available at: https://github.com/myxp-lyp/Discourse-Features-Enhance-Detection-of-Document-Level-Machine-Generated-Content.git.
>
---
#### [replaced 048] Face the Facts! Evaluating RAG-based Fact-checking Pipelines in Realistic Settings
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2412.15189v2](http://arxiv.org/pdf/2412.15189v2)**

> **作者:** Daniel Russo; Stefano Menini; Jacopo Staiano; Marco Guerini
>
> **备注:** Code and data at https://github.com/drusso98/face-the-facts - Accepted for publication at INLG 2025
>
> **摘要:** Natural Language Processing and Generation systems have recently shown the potential to complement and streamline the costly and time-consuming job of professional fact-checkers. In this work, we lift several constraints of current state-of-the-art pipelines for automated fact-checking based on the Retrieval-Augmented Generation (RAG) paradigm. Our goal is to benchmark, under more realistic scenarios, RAG-based methods for the generation of verdicts - i.e., short texts discussing the veracity of a claim - evaluating them on stylistically complex claims and heterogeneous, yet reliable, knowledge bases. Our findings show a complex landscape, where, for example, LLM-based retrievers outperform other retrieval techniques, though they still struggle with heterogeneous knowledge bases; larger models excel in verdict faithfulness, while smaller models provide better context adherence, with human evaluations favouring zero-shot and one-shot approaches for informativeness, and fine-tuned models for emotional alignment.
>
---
#### [replaced 049] GRPO-MA: Multi-Answer Generation in GRPO for Stable and Efficient Chain-of-Thought Training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.24494v2](http://arxiv.org/pdf/2509.24494v2)**

> **作者:** Hongcheng Wang; Yinuo Huang; Sukai Wang; Guanghui Ren; Hao Dong
>
> **备注:** Under review
>
> **摘要:** Recent progress, such as DeepSeek-R1, has shown that the GRPO algorithm, a Reinforcement Learning (RL) approach, can effectively train Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) and Vision-Language Models (VLMs). In this paper, we analyze three challenges of GRPO: gradient coupling between thoughts and answers, sparse reward signals caused by limited parallel sampling, and unstable advantage estimation. To mitigate these challenges, we propose GRPO-MA, a simple yet theoretically grounded method that leverages multi-answer generation from each thought process, enabling more robust and efficient optimization. Theoretically, we show that the variance of thought advantage decreases as the number of answers per thought increases. Empirically, our gradient analysis confirms this effect, showing that GRPO-MA reduces gradient spikes compared to GRPO. Experiments on math, code, and diverse multimodal tasks demonstrate that GRPO-MA substantially improves performance and training efficiency. Our ablation studies further reveal that increasing the number of answers per thought consistently enhances model performance.
>
---
#### [replaced 050] SEER: The Span-based Emotion Evidence Retrieval Benchmark
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.03490v2](http://arxiv.org/pdf/2510.03490v2)**

> **作者:** Aneesha Sampath; Oya Aran; Emily Mower Provost
>
> **摘要:** We introduce the SEER (Span-based Emotion Evidence Retrieval) Benchmark to test Large Language Models' (LLMs) ability to identify the specific spans of text that express emotion. Unlike traditional emotion recognition tasks that assign a single label to an entire sentence, SEER targets the underexplored task of emotion evidence detection: pinpointing which exact phrases convey emotion. This span-level approach is crucial for applications like empathetic dialogue and clinical support, which need to know how emotion is expressed, not just what the emotion is. SEER includes two tasks: identifying emotion evidence within a single sentence, and identifying evidence across a short passage of five consecutive sentences. It contains new annotations for both emotion and emotion evidence on 1200 real-world sentences. We evaluate 14 open-source LLMs and find that, while some models approach average human performance on single-sentence inputs, their accuracy degrades in longer passages. Our error analysis reveals key failure modes, including overreliance on emotion keywords and false positives in neutral text.
>
---
#### [replaced 051] MINED: Probing and Updating with Multimodal Time-Sensitive Knowledge for Large Multimodal Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.19457v2](http://arxiv.org/pdf/2510.19457v2)**

> **作者:** Kailin Jiang; Ning Jiang; Yuntao Du; Yuchen Ren; Yuchen Li; Yifan Gao; Jinhe Bi; Yunpu Ma; Qingqing Liu; Xianhao Wang; Yifan Jia; Hongbo Jiang; Yaocong Hu; Bin Li; Lei Liu
>
> **备注:** project page:https://mined-lmm.github.io/
>
> **摘要:** Large Multimodal Models (LMMs) encode rich factual knowledge via cross-modal pre-training, yet their static representations struggle to maintain an accurate understanding of time-sensitive factual knowledge. Existing benchmarks remain constrained by static designs, inadequately evaluating LMMs' ability to understand time-sensitive knowledge. To address this gap, we propose MINED, a comprehensive benchmark that evaluates temporal awareness along 6 key dimensions and 11 challenging tasks: cognition, awareness, trustworthiness, understanding, reasoning, and robustness. MINED is constructed from Wikipedia by two professional annotators, containing 2,104 time-sensitive knowledge samples spanning six knowledge types. Evaluating 15 widely used LMMs on MINED shows that Gemini-2.5-Pro achieves the highest average CEM score of 63.07, while most open-source LMMs still lack time understanding ability. Meanwhile, LMMs perform best on organization knowledge, whereas their performance is weakest on sport. To address these challenges, we investigate the feasibility of updating time-sensitive knowledge in LMMs through knowledge editing methods and observe that LMMs can effectively update knowledge via knowledge editing methods in single editing scenarios.
>
---
