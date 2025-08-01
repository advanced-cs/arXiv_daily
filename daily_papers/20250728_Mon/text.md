# 自然语言处理 cs.CL

- **最新发布 52 篇**

- **更新 46 篇**

## 最新发布

#### [new 001] AutoPCR: Automated Phenotype Concept Recognition by Prompting
- **分类: cs.CL**

- **简介: 论文提出AutoPCR，用于生物医学文本中的表型概念识别（CR）任务，旨在解决现有方法依赖本体训练、泛化能力差的问题。AutoPCR通过结合规则与神经网络提取实体，使用SapBERT检索候选，并通过提示大语言模型进行实体链接，无需特定本体训练，提升了跨文本类型和新本体的识别性能。**

- **链接: [http://arxiv.org/pdf/2507.19315v1](http://arxiv.org/pdf/2507.19315v1)**

> **作者:** Yicheng Tao; Yuanhao Huang; Jie Liu
>
> **摘要:** Phenotype concept recognition (CR) is a fundamental task in biomedical text mining, enabling applications such as clinical diagnostics and knowledge graph construction. However, existing methods often require ontology-specific training and struggle to generalize across diverse text types and evolving biomedical terminology. We present AutoPCR, a prompt-based phenotype CR method that does not require ontology-specific training. AutoPCR performs CR in three stages: entity extraction using a hybrid of rule-based and neural tagging strategies, candidate retrieval via SapBERT, and entity linking through prompting a large language model. Experiments on four benchmark datasets show that AutoPCR achieves the best average and most robust performance across both mention-level and document-level evaluations, surpassing prior state-of-the-art methods. Further ablation and transfer studies demonstrate its inductive capability and generalizability to new ontologies.
>
---
#### [new 002] MindFlow+: A Self-Evolving Agent for E-Commerce Customer Service
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决电商客服中动态多轮对话质量低的问题。论文提出MindFlow+模型，结合大语言模型与模仿学习、离线强化学习，通过工具增强示范构建和奖励条件数据建模提升对话效果。实验表明其在上下文相关性、灵活性和任务准确性上优于基线模型。**

- **链接: [http://arxiv.org/pdf/2507.18884v1](http://arxiv.org/pdf/2507.18884v1)**

> **作者:** Ming Gong; Xucheng Huang; Ziheng Xu; Vijayan K. Asari
>
> **摘要:** High-quality dialogue is crucial for e-commerce customer service, yet traditional intent-based systems struggle with dynamic, multi-turn interactions. We present MindFlow+, a self-evolving dialogue agent that learns domain-specific behavior by combining large language models (LLMs) with imitation learning and offline reinforcement learning (RL). MindFlow+ introduces two data-centric mechanisms to guide learning: tool-augmented demonstration construction, which exposes the model to knowledge-enhanced and agentic (ReAct-style) interactions for effective tool use; and reward-conditioned data modeling, which aligns responses with task-specific goals using reward signals. To evaluate the model's role in response generation, we introduce the AI Contribution Ratio, a novel metric quantifying AI involvement in dialogue. Experiments on real-world e-commerce conversations show that MindFlow+ outperforms strong baselines in contextual relevance, flexibility, and task accuracy. These results demonstrate the potential of combining LLMs tool reasoning, and reward-guided learning to build domain-specialized, context-aware dialogue systems.
>
---
#### [new 003] Large language models provide unsafe answers to patient-posed medical questions
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于医疗安全评估任务，旨在解决大型语言模型（LLM）聊天机器人在医疗建议中的安全性问题。研究通过分析四个主流LLM在医疗问题上的回答，评估其潜在风险，发现不同模型存在显著差异，部分回答可能对患者造成伤害，强调需进一步提升LLM在医疗领域的安全性。**

- **链接: [http://arxiv.org/pdf/2507.18905v1](http://arxiv.org/pdf/2507.18905v1)**

> **作者:** Rachel L. Draelos; Samina Afreen; Barbara Blasko; Tiffany Brazile; Natasha Chase; Dimple Desai; Jessica Evert; Heather L. Gardner; Lauren Herrmann; Aswathy Vaikom House; Stephanie Kass; Marianne Kavan; Kirshma Khemani; Amanda Koire; Lauren M. McDonald; Zahraa Rabeeah; Amy Shah
>
> **备注:** 20 pages
>
> **摘要:** Millions of patients are already using large language model (LLM) chatbots for medical advice on a regular basis, raising patient safety concerns. This physician-led red-teaming study compares the safety of four publicly available chatbots--Claude by Anthropic, Gemini by Google, GPT-4o by OpenAI, and Llama3-70B by Meta--on a new dataset, HealthAdvice, using an evaluation framework that enables quantitative and qualitative analysis. In total, 888 chatbot responses are evaluated for 222 patient-posed advice-seeking medical questions on primary care topics spanning internal medicine, women's health, and pediatrics. We find statistically significant differences between chatbots. The rate of problematic responses varies from 21.6 percent (Claude) to 43.2 percent (Llama), with unsafe responses varying from 5 percent (Claude) to 13 percent (GPT-4o, Llama). Qualitative results reveal chatbot responses with the potential to lead to serious patient harm. This study suggests that millions of patients could be receiving unsafe medical advice from publicly available chatbots, and further work is needed to improve the clinical safety of these powerful tools.
>
---
#### [new 004] Smooth Reading: Bridging the Gap of Recurrent LLM to Self-Attention LLM on Long-Context Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决循环大语言模型（Recurrent LLMs）在长上下文任务中性能不如自注意力模型的问题。作者提出“Smooth Reading”方法，通过分块处理和迭代总结上下文，降低内存需求。实验表明该方法显著缩小了两种模型间的性能差距，同时保持了循环模型的效率优势。**

- **链接: [http://arxiv.org/pdf/2507.19353v1](http://arxiv.org/pdf/2507.19353v1)**

> **作者:** Kai Liu; Zhan Su; Peijie Dong; Fengran Mo; Jianfei Gao; ShaoTing Zhang; Kai Chen
>
> **摘要:** Recently, recurrent large language models (Recurrent LLMs) with linear computational complexity have re-emerged as efficient alternatives to self-attention-based LLMs (Self-Attention LLMs), which have quadratic complexity. However, Recurrent LLMs often underperform on long-context tasks due to their limited fixed-size memory. Previous research has primarily focused on enhancing the memory capacity of Recurrent LLMs through architectural innovations, but these approaches have not yet enabled Recurrent LLMs to match the performance of Self-Attention LLMs on long-context tasks. We argue that this limitation arises because processing the entire context at once is not well-suited for Recurrent LLMs. In this paper, we propose Smooth Reading, a chunk-wise inference method inspired by human reading strategies. Smooth Reading processes context in chunks and iteratively summarizes the contextual information, thereby reducing memory demands and making the approach more compatible with Recurrent LLMs. Our experimental results show that this method substantially narrows the performance gap between Recurrent and Self-Attention LLMs on long-context tasks, while preserving the efficiency advantages of Recurrent LLMs. Our Smooth Reading boosts SWA-3B-4k (a Recurrent LLM) from 5.68% lower to 3.61% higher performance than Self-Attention LLMs on LongBench. Besides, our method maintains the high efficiency, training 3x faster and inferring 2x faster at 64k context compared to Self-Attention LLMs. To our knowledge, this is the first work to achieve comparable performance using Recurrent LLMs compared with Self-Attention LLMs on long-context tasks. We hope our method will inspire future research in this area. To facilitate further progress, we will release code and dataset.
>
---
#### [new 005] SpeechIQ: Speech Intelligence Quotient Across Cognitive Levels in Voice Understanding Large Language Models
- **分类: cs.CL; cs.AI; cs.SC; cs.SD; eess.AS**

- **简介: 该论文提出了SpeechIQ（SIQ），一种基于认知层次的语音理解评估框架，用于评估语音大模型（LLM Voice）在语音理解方面的能力。它结合了Bloom分类法的三个认知层次：记忆、理解和应用，旨在超越传统的WER指标，提供更全面的模型评估与比较，并检测标注错误和幻觉问题。**

- **链接: [http://arxiv.org/pdf/2507.19361v1](http://arxiv.org/pdf/2507.19361v1)**

> **作者:** Zhen Wan; Chao-Han Huck Yang; Yahan Yu; Jinchuan Tian; Sheng Li; Ke Hu; Zhehuai Chen; Shinji Watanabe; Fei Cheng; Chenhui Chu; Sadao Kurohashi
>
> **备注:** Our Speech-IQ leaderboard will be hosted at huggingface.co/spaces/nvidia/Speech-IQ-leaderboard. ACL 2025 main
>
> **摘要:** We introduce Speech-based Intelligence Quotient (SIQ) as a new form of human cognition-inspired evaluation pipeline for voice understanding large language models, LLM Voice, designed to assess their voice understanding ability. Moving beyond popular voice understanding metrics such as word error rate (WER), SIQ examines LLM Voice across three cognitive levels motivated by Bloom's Taxonomy: (1) Remembering (i.e., WER for verbatim accuracy); (2) Understanding (i.e., similarity of LLM's interpretations); and (3) Application (i.e., QA accuracy for simulating downstream tasks). We demonstrate that SIQ not only quantifies voice understanding abilities but also provides unified comparisons between cascaded methods (e.g., ASR LLM) and end-to-end models, identifies annotation errors in existing benchmarks, and detects hallucinations in LLM Voice. Our framework represents a first-of-its-kind intelligence examination that bridges cognitive principles with voice-oriented benchmarks, while exposing overlooked challenges in multi-modal training.
>
---
#### [new 006] Arg-LLaDA: Argument Summarization via Large Language Diffusion Models and Sufficiency-Aware Refinement
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的论点摘要任务，旨在解决现有方法在生成阶段缺乏迭代优化的问题。作者提出了Arg-LLaDA框架，通过扩散模型和充分性感知机制，迭代改进摘要，提升其准确性与简洁性。**

- **链接: [http://arxiv.org/pdf/2507.19081v1](http://arxiv.org/pdf/2507.19081v1)**

> **作者:** Hao Li; Yizheng Sun; Viktor Schlegel; Kailai Yang; Riza Batista-Navarro; Goran Nenadic
>
> **备注:** Preprint
>
> **摘要:** Argument summarization aims to generate concise, structured representations of complex, multi-perspective debates. While recent work has advanced the identification and clustering of argumentative components, the generation stage remains underexplored. Existing approaches typically rely on single-pass generation, offering limited support for factual correction or structural refinement. To address this gap, we introduce Arg-LLaDA, a novel large language diffusion framework that iteratively improves summaries via sufficiency-guided remasking and regeneration. Our method combines a flexible masking controller with a sufficiency-checking module to identify and revise unsupported, redundant, or incomplete spans, yielding more faithful, concise, and coherent outputs. Empirical results on two benchmark datasets demonstrate that Arg-LLaDA surpasses state-of-the-art baselines in 7 out of 10 automatic evaluation metrics. In addition, human evaluations reveal substantial improvements across core dimensions, coverage, faithfulness, and conciseness, validating the effectiveness of our iterative, sufficiency-aware generation strategy.
>
---
#### [new 007] A Similarity Measure for Comparing Conversational Dynamics
- **分类: cs.CL**

- **简介: 该论文旨在解决如何通过自动化方法全面比较对话的整体互动动态这一问题。他们提出了一种用于衡量对话相似性的新方法，设计了验证框架以测试其效果，并利用该方法分析了在线社区中的对话动态，揭示了情境权力的作用。任务属于对话分析与自然语言处理。**

- **链接: [http://arxiv.org/pdf/2507.18956v1](http://arxiv.org/pdf/2507.18956v1)**

> **作者:** Sang Min Jung; Kaixiang Zhang; Cristian Danescu-Niculescu-Mizil
>
> **备注:** Code and demos available in ConvoKit (https://convokit.cornell.edu/)
>
> **摘要:** The quality of a conversation goes beyond the individual quality of each reply, and instead emerges from how these combine into interactional patterns that give the conversation its distinctive overall "shape". However, there is no robust automated method for comparing conversations in terms of their overall interactional dynamics. Such methods could enhance the analysis of conversational data and help evaluate conversational agents more holistically. In this work, we introduce a similarity measure for comparing conversations with respect to their dynamics. We design a validation framework for testing the robustness of the metric in capturing differences in conversation dynamics and for assessing its sensitivity to the topic of the conversations. Finally, to illustrate the measure's utility, we use it to analyze conversational dynamics in a large online community, bringing new insights into the role of situational power in conversations.
>
---
#### [new 008] Can Small-Scale Data Poisoning Exacerbate Dialect-Linked Biases in Large Language Models?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究小规模数据投毒是否会加剧大语言模型中的方言相关偏见。任务是分析非裔美国白话英语（AAVE）与标准美式英语（SAE）在毒性输出上的差异。作者使用LLaMA模型和GPT-4o评估投毒数据对模型输出的影响，发现AAVE更易受投毒影响，且模型规模越大，偏见越明显。提出了需改进方言感知评估和训练策略。**

- **链接: [http://arxiv.org/pdf/2507.19195v1](http://arxiv.org/pdf/2507.19195v1)**

> **作者:** Chaymaa Abbas; Mariette Awad; Razane Tajeddine
>
> **摘要:** Despite the ongoing improvements in the design of large language models (LLMs) to foster inclusion and balanced responses, these systems remain susceptible to encoding and amplifying social biases. This study examines how dialectal variation, specifically African American Vernacular English (AAVE) versus Standard American English (SAE), interacts with data poisoning to influence toxicity in outputs. Using both small- and medium-scale LLaMA models, we show that even minimal exposure to poisoned data significantly increases toxicity for AAVE inputs, while it remains comparatively unaffected for SAE. Larger models exhibit a more significant amplification effect which suggests heightened susceptibility with scale. To further assess these disparities, we employed GPT-4o as a fairness auditor, which identified harmful stereotypical patterns disproportionately tied to AAVE inputs, including portrayals of aggression, criminality, and intellectual inferiority. These findings underscore the compounding impact of data poisoning and dialectal bias and emphasize the need for dialect-aware evaluation, targeted debiasing interventions, and socially responsible training protocols during development.
>
---
#### [new 009] Specification Self-Correction: Mitigating In-Context Reward Hacking Through Test-Time Refinement
- **分类: cs.CL; cs.AI**

- **简介: 论文提出“规格自修正”（SSC）方法，解决语言模型在上下文中利用错误规格进行奖励欺骗的问题。该方法在推理时通过多步过程让模型自我审查并修正规格，提升对齐效果，无需修改模型权重。实验显示其将漏洞利用减少超90%。**

- **链接: [http://arxiv.org/pdf/2507.18742v1](http://arxiv.org/pdf/2507.18742v1)**

> **作者:** Víctor Gallego
>
> **备注:** Accepted to SCALR Workshop @ COLM 2025
>
> **摘要:** Language models (LMs) are susceptible to in-context reward hacking, where they exploit flaws in tainted or faulty written specifications or rubrics to achieve high scores without fulfilling the user's true intent. We introduce Specification Self-Correction (SSC), a novel, test-time framework that enables an LM to identify and correct flaws within its own guiding specification. SSC employs a multi-step inference process where the model first generates a response based on a potentially tainted specification, critiques its output, and then revises the specification itself to remove the exploitable loophole. A final, more robust response is then generated using this self-corrected specification. Across experiments spanning creative writing and agentic coding tasks with several LMs, we demonstrate that while models initially game tainted specifications in 50-70\% of cases, the SSC process reduces this vulnerability by over 90\%. This dynamic repair occurs at inference time, requires no weight modification, and leads to more robustly aligned model behavior. Code at https://github.com/vicgalle/specification-self-correction .
>
---
#### [new 010] Jailbreaking Large Language Diffusion Models: Revealing Hidden Safety Flaws in Diffusion-Based Text Generation
- **分类: cs.CL**

- **简介: 该论文研究大型语言扩散模型（LLDMs）的安全漏洞，旨在解决其潜在有害生成内容的问题。作者提出了一种针对LLDMs的并行解码越狱攻击方法（PAD），通过多点注意力攻击实现高效攻击，揭示了LLDMs在安全性方面的重大隐患，并为扩散模型的安全部署提供了关键见解。**

- **链接: [http://arxiv.org/pdf/2507.19227v1](http://arxiv.org/pdf/2507.19227v1)**

> **作者:** Yuanhe Zhang; Fangzhou Xie; Zhenhong Zhou; Zherui Li; Hao Chen; Kun Wang; Yufei Guo
>
> **摘要:** Large Language Diffusion Models (LLDMs) exhibit comparable performance to LLMs while offering distinct advantages in inference speed and mathematical reasoning tasks.The precise and rapid generation capabilities of LLDMs amplify concerns of harmful generations, while existing jailbreak methodologies designed for Large Language Models (LLMs) prove limited effectiveness against LLDMs and fail to expose safety vulnerabilities.Successful defense cannot definitively resolve harmful generation concerns, as it remains unclear whether LLDMs possess safety robustness or existing attacks are incompatible with diffusion-based architectures.To address this, we first reveal the vulnerability of LLDMs to jailbreak and demonstrate that attack failure in LLDMs stems from fundamental architectural differences.We present a PArallel Decoding jailbreak (PAD) for diffusion-based language models. PAD introduces Multi-Point Attention Attack, which guides parallel generative processes toward harmful outputs that inspired by affirmative response patterns in LLMs. Experimental evaluations across four LLDMs demonstrate that PAD achieves jailbreak attack success rates by 97%, revealing significant safety vulnerabilities. Furthermore, compared to autoregressive LLMs of the same size, LLDMs increase the harmful generation speed by 2x, significantly highlighting risks of uncontrolled misuse.Through comprehensive analysis, we provide an investigation into LLDM architecture, offering critical insights for the secure deployment of diffusion-based language models.
>
---
#### [new 011] TokenSmith: Streamlining Data Editing, Search, and Inspection for Large-Scale Language Model Training and Interpretability
- **分类: cs.CL**

- **简介: 论文提出TokenSmith，一个用于大规模语言模型预训练数据交互编辑、搜索和分析的开源工具库。它解决了预训练数据处理流程繁琐、分散的问题，支持多种数据操作，无需修改训练代码，适用于主流框架，提升数据集调试、验证和实验效率。**

- **链接: [http://arxiv.org/pdf/2507.19419v1](http://arxiv.org/pdf/2507.19419v1)**

> **作者:** Mohammad Aflah Khan; Ameya Godbole; Johnny Tian-Zheng Wei; Ryan Wang; James Flemings; Krishna Gummadi; Willie Neiswanger; Robin Jia
>
> **摘要:** Understanding the relationship between training data and model behavior during pretraining is crucial, but existing workflows make this process cumbersome, fragmented, and often inaccessible to researchers. We present TokenSmith, an open-source library for interactive editing, inspection, and analysis of datasets used in Megatron-style pretraining frameworks such as GPT-NeoX, Megatron, and NVIDIA NeMo. TokenSmith supports a wide range of operations including searching, viewing, ingesting, exporting, inspecting, and sampling data, all accessible through a simple user interface and a modular backend. It also enables structured editing of pretraining data without requiring changes to training code, simplifying dataset debugging, validation, and experimentation. TokenSmith is designed as a plug and play addition to existing large language model pretraining workflows, thereby democratizing access to production-grade dataset tooling. TokenSmith is hosted on GitHub1, with accompanying documentation and tutorials. A demonstration video is also available on YouTube.
>
---
#### [new 012] A Toolbox, Not a Hammer -- Multi-TAG: Scaling Math Reasoning with Multi-Tool Aggregation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于数学推理任务，旨在解决复杂数学问题求解中单工具推理不准确的问题。论文提出Multi-TAG框架，在推理每一步调用多个工具并聚合其输出，提升解题准确性和鲁棒性。方法无需微调，适用于各类大模型。**

- **链接: [http://arxiv.org/pdf/2507.18973v1](http://arxiv.org/pdf/2507.18973v1)**

> **作者:** Bohan Yao; Vikas Yadav
>
> **备注:** 21 pages, 3 figures
>
> **摘要:** Augmenting large language models (LLMs) with external tools is a promising avenue for developing high-performance mathematical reasoning systems. Prior tool-augmented approaches typically finetune an LLM to select and invoke a single tool at each reasoning step and show promising results on simpler math reasoning benchmarks such as GSM8K. However, these approaches struggle with more complex math problems that require precise reasoning over multiple steps. To address this limitation, in this work, we propose Multi-TAG, a Multi-Tool AGgregation-based framework. Instead of relying on a single tool, Multi-TAG guides an LLM to concurrently invoke multiple tools at each reasoning step. It then aggregates their diverse outputs to verify and refine the reasoning process, enhancing solution robustness and accuracy. Notably, Multi-TAG is a finetuning-free, inference-only framework, making it readily applicable to any LLM backbone, including large open-weight models which are computationally expensive to finetune and proprietary frontier models which cannot be finetuned with custom recipes. We evaluate Multi-TAG on four challenging benchmarks: MATH500, AIME, AMC, and OlympiadBench. Across both open-weight and closed-source LLM backbones, Multi-TAG consistently and substantially outperforms state-of-the-art baselines, achieving average improvements of 6.0% to 7.5% over state-of-the-art baselines.
>
---
#### [new 013] The Role of Orthographic Consistency in Multilingual Embedding Models for Text Classification in Arabic-Script Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的多语言文本分类任务。它旨在解决现有通用多语言模型（如mBERT和XLM-RoBERTa）在阿拉伯语系语言中表现不佳的问题。作者构建了针对阿拉伯语、库尔德语、波斯语和乌尔都语的AS-RoBERTa模型，通过语言特异性的预训练提升分类性能。实验表明，新模型在多个任务上优于通用模型，并通过消融实验验证了语言特异性预训练的重要性。**

- **链接: [http://arxiv.org/pdf/2507.18762v1](http://arxiv.org/pdf/2507.18762v1)**

> **作者:** Abdulhady Abas Abdullah; Amir H. Gandomi; Tarik A Rashid; Seyedali Mirjalili; Laith Abualigah; Milena Živković; Hadi Veisi
>
> **摘要:** In natural language processing, multilingual models like mBERT and XLM-RoBERTa promise broad coverage but often struggle with languages that share a script yet differ in orthographic norms and cultural context. This issue is especially notable in Arabic-script languages such as Kurdish Sorani, Arabic, Persian, and Urdu. We introduce the Arabic Script RoBERTa (AS-RoBERTa) family: four RoBERTa-based models, each pre-trained on a large corpus tailored to its specific language. By focusing pre-training on language-specific script features and statistics, our models capture patterns overlooked by general-purpose models. When fine-tuned on classification tasks, AS-RoBERTa variants outperform mBERT and XLM-RoBERTa by 2 to 5 percentage points. An ablation study confirms that script-focused pre-training is central to these gains. Error analysis using confusion matrices shows how shared script traits and domain-specific content affect performance. Our results highlight the value of script-aware specialization for languages using the Arabic script and support further work on pre-training strategies rooted in script and language specificity.
>
---
#### [new 014] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?
- **分类: cs.CL**

- **简介: 该论文属于自动化评估任务，旨在解决社会科学研究可重复性评估耗时费力的问题。作者构建了REPRO-Bench基准，包含112个任务实例，用于测试AI代理基于论文PDF和复现包评估可重复性的能力。实验表明现有AI代理表现不佳，但改进后的REPRO-Agent将准确率提升了71%。**

- **链接: [http://arxiv.org/pdf/2507.18901v1](http://arxiv.org/pdf/2507.18901v1)**

> **作者:** Chuxuan Hu; Liyun Zhang; Yeji Lim; Aum Wadhwani; Austin Peters; Daniel Kang
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Assessing the reproducibility of social science papers is essential for promoting rigor in research processes, but manual assessment is costly. With recent advances in agentic AI systems (i.e., AI agents), we seek to evaluate their capability to automate this process. However, existing benchmarks for reproducing research papers (1) focus solely on reproducing results using provided code and data without assessing their consistency with the paper, (2) oversimplify real-world scenarios, and (3) lack necessary diversity in data formats and programming languages. To address these issues, we introduce REPRO-Bench, a collection of 112 task instances, each representing a social science paper with a publicly available reproduction report. The agents are tasked with assessing the reproducibility of the paper based on the original paper PDF and the corresponding reproduction package. REPRO-Bench features end-to-end evaluation tasks on the reproducibility of social science papers with complexity comparable to real-world assessments. We evaluate three representative AI agents on REPRO-Bench, with the best-performing agent achieving an accuracy of only 21.4%. Building on our empirical analysis, we develop REPRO-Agent, which improves the highest accuracy achieved by existing agents by 71%. We conclude that more advanced AI agents should be developed to automate real-world reproducibility assessment. REPRO-Bench is publicly available at https://github.com/uiuc-kang-lab/REPRO-Bench.
>
---
#### [new 015] A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在系统回顾检索增强生成（RAG）系统的发展，解决大语言模型在事实准确性、知识更新和上下文相关性方面的不足。论文分析了RAG的核心组件、演进历程、应用挑战及未来方向。**

- **链接: [http://arxiv.org/pdf/2507.18910v1](http://arxiv.org/pdf/2507.18910v1)**

> **作者:** Agada Joseph Oche; Ademola Glory Folashade; Tirthankar Ghosal; Arpan Biswas
>
> **备注:** 33 pages, 2 figures
>
> **摘要:** Retrieval-Augmented Generation (RAG) represents a major advancement in natural language processing (NLP), combining large language models (LLMs) with information retrieval systems to enhance factual grounding, accuracy, and contextual relevance. This paper presents a comprehensive systematic review of RAG, tracing its evolution from early developments in open domain question answering to recent state-of-the-art implementations across diverse applications. The review begins by outlining the motivations behind RAG, particularly its ability to mitigate hallucinations and outdated knowledge in parametric models. Core technical components-retrieval mechanisms, sequence-to-sequence generation models, and fusion strategies are examined in detail. A year-by-year analysis highlights key milestones and research trends, providing insight into RAG's rapid growth. The paper further explores the deployment of RAG in enterprise systems, addressing practical challenges related to retrieval of proprietary data, security, and scalability. A comparative evaluation of RAG implementations is conducted, benchmarking performance on retrieval accuracy, generation fluency, latency, and computational efficiency. Persistent challenges such as retrieval quality, privacy concerns, and integration overhead are critically assessed. Finally, the review highlights emerging solutions, including hybrid retrieval approaches, privacy-preserving techniques, optimized fusion strategies, and agentic RAG architectures. These innovations point toward a future of more reliable, efficient, and context-aware knowledge-intensive NLP systems.
>
---
#### [new 016] Objectifying the Subjective: Cognitive Biases in Topic Interpretations
- **分类: cs.CL**

- **简介: 该论文研究用户如何主观解读话题，旨在改进话题质量评估。通过用户实验与主题分析，发现用户依赖启发式思维而非概率判断来解释话题，提出基于锚定调整理论的解读模型，强调需考虑认知偏差设计更合理的话题评估框架。**

- **链接: [http://arxiv.org/pdf/2507.19117v1](http://arxiv.org/pdf/2507.19117v1)**

> **作者:** Swapnil Hingmire; Ze Shi Li; Shiyu; Zeng; Ahmed Musa Awon; Luiz Franciscatto Guerra; Neil Ernst
>
> **备注:** Accepted for publication at the Transactions of ACL (TACL) (pre-MIT Press publication version)
>
> **摘要:** Interpretation of topics is crucial for their downstream applications. State-of-the-art evaluation measures of topic quality such as coherence and word intrusion do not measure how much a topic facilitates the exploration of a corpus. To design evaluation measures grounded on a task, and a population of users, we do user studies to understand how users interpret topics. We propose constructs of topic quality and ask users to assess them in the context of a topic and provide rationale behind evaluations. We use reflexive thematic analysis to identify themes of topic interpretations from rationales. Users interpret topics based on availability and representativeness heuristics rather than probability. We propose a theory of topic interpretation based on the anchoring-and-adjustment heuristic: users anchor on salient words and make semantic adjustments to arrive at an interpretation. Topic interpretation can be viewed as making a judgment under uncertainty by an ecologically rational user, and hence cognitive biases aware user models and evaluation frameworks are needed.
>
---
#### [new 017] Conversations Gone Awry, But Then? Evaluating Conversational Forecasting Models
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于对话预测任务，旨在解决预测对话是否会失控的问题。作者构建了首个统一评估框架，引入新指标，用于比较不同模型在预测对话走向失控方面的能力，并提供当前研究进展的全面分析。**

- **链接: [http://arxiv.org/pdf/2507.19470v1](http://arxiv.org/pdf/2507.19470v1)**

> **作者:** Son Quoc Tran; Tushaar Gangavarapu; Nicholas Chernogor; Jonathan P. Chang; Cristian Danescu-Niculescu-Mizil
>
> **备注:** Code and data available as part of ConvoKit: https://convokit.cornell.edu
>
> **摘要:** We often rely on our intuition to anticipate the direction of a conversation. Endowing automated systems with similar foresight can enable them to assist human-human interactions. Recent work on developing models with this predictive capacity has focused on the Conversations Gone Awry (CGA) task: forecasting whether an ongoing conversation will derail. In this work, we revisit this task and introduce the first uniform evaluation framework, creating a benchmark that enables direct and reliable comparisons between different architectures. This allows us to present an up-to-date overview of the current progress in CGA models, in light of recent advancements in language modeling. Our framework also introduces a novel metric that captures a model's ability to revise its forecast as the conversation progresses.
>
---
#### [new 018] LLaVA-NeuMT: Selective Layer-Neuron Modulation for Efficient Multilingual Multimodal Translation
- **分类: cs.CL; cs.MM**

- **简介: 该论文属于多语言多模态翻译任务，旨在解决多语言环境下跨语言干扰和参数共享策略无效的问题。论文提出了LLaVA-NeuMT框架，通过选择性地调节模型层和神经元，优化语言特定与通用表示，实现高效翻译。实验表明其性能优越，且仅需微调40%参数。**

- **链接: [http://arxiv.org/pdf/2507.18940v1](http://arxiv.org/pdf/2507.18940v1)**

> **作者:** Jingxuan Wei; Caijun Jia; Qi Chen; Yujun Cai; Linzhuang Sun; Xiangxiang Zhang; Gaowei Wu; Bihui Yu
>
> **摘要:** Multimodal Machine Translation (MMT) enhances translation quality by incorporating visual context, helping to resolve textual ambiguities. While existing MMT methods perform well in bilingual settings, extending them to multilingual translation remains challenging due to cross-lingual interference and ineffective parameter-sharing strategies. To address this, we propose LLaVA-NeuMT, a novel multimodal multilingual translation framework that explicitly models language-specific and language-agnostic representations to mitigate multilingual interference. Our approach consists of a layer selection mechanism that identifies the most informative layers for different language pairs and a neuron-level adaptation strategy that dynamically selects language-specific and agnostic neurons to improve translation quality while reducing redundancy. We conduct extensive experiments on the M3-Multi30K and M3-AmbigCaps datasets, demonstrating that LLaVA-NeuMT, while fine-tuning only 40\% of the model parameters, surpasses full fine-tuning approaches and ultimately achieves SOTA results on both datasets. Our analysis further provides insights into the importance of selected layers and neurons in multimodal multilingual adaptation, offering an efficient and scalable solution to cross-lingual adaptation in multimodal translation.
>
---
#### [new 019] PrismRAG: Boosting RAG Factuality with Distractor Resilience and Strategized Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出PrismRAG，一种提升检索增强生成（RAG）事实性的框架。该框架通过引入干扰项感知的问答对训练模型，并培养以推理为中心的习惯，使其在面对误导性信息时具备更强的鲁棒性与推理能力，从而提升RAG在多领域问答任务中的表现。**

- **链接: [http://arxiv.org/pdf/2507.18857v1](http://arxiv.org/pdf/2507.18857v1)**

> **作者:** Mohammad Kachuee; Teja Gollapudi; Minseok Kim; Yin Huang; Kai Sun; Xiao Yang; Jiaqi Wang; Nirav Shah; Yue Liu; Aaron Colak; Anuj Kumar; Wen-tau Yih; Xin Luna Dong
>
> **摘要:** Retrieval-augmented generation (RAG) often falls short when retrieved context includes confusing semi-relevant passages, or when answering questions require deep contextual understanding and reasoning. We propose an efficient fine-tuning framework, called PrismRAG, that (i) trains the model with distractor-aware QA pairs mixing gold evidence with subtle distractor passages, and (ii) instills reasoning-centric habits that make the LLM plan, rationalize, and synthesize without relying on extensive human engineered instructions. Evaluated across 12 open-book RAG QA benchmarks spanning diverse application domains and scenarios, PrismRAG improves average factuality by 5.4%, outperforming state-of-the-art solutions.
>
---
#### [new 020] Evaluating Code-Mixing in LLMs Across 18 Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型在多语种混用场景下的表现。现有基准测试覆盖有限，论文提出新方法生成合成数据，评估18种语言混用情况，发现模型表现欠佳，并探讨提升方法。**

- **链接: [http://arxiv.org/pdf/2507.18791v1](http://arxiv.org/pdf/2507.18791v1)**

> **作者:** Yilun Yang; Yekun Chai
>
> **摘要:** Code-mixing, the practice of switching between languages within a conversation, presents unique challenges for traditional natural language processing. Existing benchmarks, such as LinCE and GLUECoS, are limited by narrow language pairings and tasks, failing to adequately evaluate the code-mixing capabilities of large language models (LLMs). Despite the significance of code-mixing for multilingual users, research on LLMs in this context remains limited. Additionally, current methods for generating code-mixed data are underdeveloped. In this paper, we conduct a comprehensive evaluation of LLMs' performance on code-mixed data across 18 languages from seven language families. We also propose a novel approach for generating synthetic code-mixed texts by combining word substitution with GPT-4 prompting. Our analysis reveals consistent underperformance of LLMs on code-mixed datasets involving multiple language families. We suggest that improvements in training data size, model scale, and few-shot learning could enhance their performance.
>
---
#### [new 021] How Much Do Large Language Model Cheat on Evaluation? Benchmarking Overestimation under the One-Time-Pad-Based Framework
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于自然语言处理任务，旨在解决大型语言模型在评估中存在高估问题。通过提出ArxivRoll框架，包括生成私有测试用例的SCP和度量污染与偏差的Rugged Scores，实现对模型性能的动态、公正评估，并量化当前模型的高估程度。**

- **链接: [http://arxiv.org/pdf/2507.19219v1](http://arxiv.org/pdf/2507.19219v1)**

> **作者:** Zi Liang; Liantong Yu; Shiyu Zhang; Qingqing Ye; Haibo Hu
>
> **备注:** Source code: https://github.com/liangzid/ArxivRoll/ Website: https://arxivroll.moreoverai.com/
>
> **摘要:** Overestimation in evaluating large language models (LLMs) has become an increasing concern. Due to the contamination of public benchmarks or imbalanced model training, LLMs may achieve unreal evaluation results on public benchmarks, either intentionally or unintentionally, which leads to unfair comparisons among LLMs and undermines their realistic capability assessments. Existing benchmarks attempt to address these issues by keeping test cases permanently secret, mitigating contamination through human evaluation, or repeatedly collecting and constructing new samples. However, these approaches fail to ensure reproducibility, transparency, and high efficiency simultaneously. Moreover, the extent of overestimation in current LLMs remains unquantified. To address these issues, we propose ArxivRoll, a dynamic evaluation framework inspired by one-time pad encryption in cryptography. ArxivRoll comprises two key components: \emph{i) SCP (Sequencing, Cloze, and Prediction)}, an automated generator for private test cases, and \emph{ii) Rugged Scores (RS)}, metrics that measure the proportion of public benchmark contamination and training bias. Leveraging SCP, ArxivRoll constructs a new benchmark every six months using recent articles from ArXiv and employs them for one-time evaluations of LLM performance. Extensive experiments demonstrate the high quality of our benchmark, and we provide a systematic evaluation of current LLMs. The source code is available at https://github.com/liangzid/ArxivRoll/.
>
---
#### [new 022] Uncovering Cross-Linguistic Disparities in LLMs using Sparse Autoencoders
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言大模型分析与优化任务，旨在解决中低资源语言在模型表现上的不足。通过稀疏自编码器分析激活模式，发现中低资源语言激活值较低，采用基于LoRA的激活感知微调方法显著提升激活值，并带来性能改善。**

- **链接: [http://arxiv.org/pdf/2507.18918v1](http://arxiv.org/pdf/2507.18918v1)**

> **作者:** Richmond Sin Jing Xuan; Jalil Huseynov; Yang Zhang
>
> **摘要:** Multilingual large language models (LLMs) exhibit strong cross-linguistic generalization, yet medium to low resource languages underperform on common benchmarks such as ARC-Challenge, MMLU, and HellaSwag. We analyze activation patterns in Gemma-2-2B across all 26 residual layers and 10 languages: Chinese (zh), Russian (ru), Spanish (es), Italian (it), medium to low resource languages including Indonesian (id), Catalan (ca), Marathi (mr), Malayalam (ml), and Hindi (hi), with English (en) as the reference. Using Sparse Autoencoders (SAEs), we reveal systematic disparities in activation patterns. Medium to low resource languages receive up to 26.27 percent lower activations in early layers, with a persistent gap of 19.89 percent in deeper layers. To address this, we apply activation-aware fine-tuning via Low-Rank Adaptation (LoRA), leading to substantial activation gains, such as 87.69 percent for Malayalam and 86.32 percent for Hindi, while maintaining English retention at approximately 91 percent. After fine-tuning, benchmark results show modest but consistent improvements, highlighting activation alignment as a key factor in enhancing multilingual LLM performance.
>
---
#### [new 023] Enhancing Speech Emotion Recognition Leveraging Aligning Timestamps of ASR Transcripts and Speaker Diarization
- **分类: cs.CL; I.2.7; I.5.1**

- **简介: 该论文属于语音情感识别任务，旨在解决自动语音识别与说话人分割在时间戳上的不一致问题。通过引入时间戳对齐的多模态方法，结合文本和音频特征，提升情感识别准确率。实验表明该方法在IEMOCAP数据集上优于基线模型。**

- **链接: [http://arxiv.org/pdf/2507.19356v1](http://arxiv.org/pdf/2507.19356v1)**

> **作者:** Hsuan-Yu Wang; Pei-Ying Lee; Berlin Chen
>
> **备注:** 6 pages, 3 figures, to appear in the Proceedings of the 2025 International Conference on Asian Language Processing (IALP)
>
> **摘要:** In this paper, we investigate the impact of incorporating timestamp-based alignment between Automatic Speech Recognition (ASR) transcripts and Speaker Diarization (SD) outputs on Speech Emotion Recognition (SER) accuracy. Misalignment between these two modalities often reduces the reliability of multimodal emotion recognition systems, particularly in conversational contexts. To address this issue, we introduce an alignment pipeline utilizing pre-trained ASR and speaker diarization models, systematically synchronizing timestamps to generate accurately labeled speaker segments. Our multimodal approach combines textual embeddings extracted via RoBERTa with audio embeddings from Wav2Vec, leveraging cross-attention fusion enhanced by a gating mechanism. Experimental evaluations on the IEMOCAP benchmark dataset demonstrate that precise timestamp alignment improves SER accuracy, outperforming baseline methods that lack synchronization. The results highlight the critical importance of temporal alignment, demonstrating its effectiveness in enhancing overall emotion recognition accuracy and providing a foundation for robust multimodal emotion analysis.
>
---
#### [new 024] CueBuddy: helping non-native English speakers navigate English-centric STEM education
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出了CueBuddy，旨在帮助非英语母语学生应对英语为主的STEM教育。其任务是通过实时识别技术关键词并提供多语言术语查询，辅助学生理解复杂术语，从而提升学习效果。论文解决了大规模实时翻译模型成本高和术语处理难的问题，提出了一个轻量级的替代方案。**

- **链接: [http://arxiv.org/pdf/2507.18827v1](http://arxiv.org/pdf/2507.18827v1)**

> **作者:** Pranav Gupta
>
> **摘要:** Students across the world in STEM classes, especially in the Global South, fall behind their peers who are more fluent in English, despite being at par with them in terms of scientific prerequisites. While many of them are able to follow everyday English at ease, key terms in English stay challenging. In most cases, such students have had most of their course prerequisites in a lower resource language. Live speech translation to lower resource languages is a promising area of research, however, models for speech translation can be too expensive on a large scale and often struggle with technical content. In this paper, we describe CueBuddy, which aims to remediate these issues by providing real-time "lexical cues" through technical keyword spotting along real-time multilingual glossary lookup to help students stay up to speed with complex English jargon without disrupting their concentration on the lecture. We also describe the limitations and future extensions of our approach.
>
---
#### [new 025] ylmmcl at Multilingual Text Detoxification 2025: Lexicon-Guided Detoxification and Classifier-Gated Rewriting
- **分类: cs.CL; cs.LG**

- **简介: 论文属于多语言文本解毒任务，旨在解决多语言文本中毒性内容的识别与改写问题。作者提出了一个结合词典引导标注、微调序列到序列模型和分类器门控机制的多语言解毒框架。该方法在多个语言中表现出优于基线模型的效果，尤其在英语、俄语和法语等高资源语言中泛化能力强，最终在PAN-2025比赛中获得第九名。**

- **链接: [http://arxiv.org/pdf/2507.18769v1](http://arxiv.org/pdf/2507.18769v1)**

> **作者:** Nicole Lai-Lopez; Lusha Wang; Su Yuan; Liza Zhang
>
> **备注:** 16 pages, 5 figures, 3 tables,
>
> **摘要:** In this work, we introduce our solution for the Multilingual Text Detoxification Task in the PAN-2025 competition for the ylmmcl team: a robust multilingual text detoxification pipeline that integrates lexicon-guided tagging, a fine-tuned sequence-to-sequence model (s-nlp/mt0-xl-detox-orpo) and an iterative classifier-based gatekeeping mechanism. Our approach departs from prior unsupervised or monolingual pipelines by leveraging explicit toxic word annotation via the multilingual_toxic_lexicon to guide detoxification with greater precision and cross-lingual generalization. Our final model achieves the highest STA (0.922) from our previous attempts, and an average official J score of 0.612 for toxic inputs in both the development and test sets. It also achieved xCOMET scores of 0.793 (dev) and 0.787 (test). This performance outperforms baseline and backtranslation methods across multiple languages, and shows strong generalization in high-resource settings (English, Russian, French). Despite some trade-offs in SIM, the model demonstrates consistent improvements in detoxification strength. In the competition, our team achieved ninth place with a score of 0.612.
>
---
#### [new 026] Mining Contextualized Visual Associations from Images for Creativity Understanding
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言理解任务，旨在解决现有模型缺乏抽象关联表达的问题。作者提出一种从图像中挖掘上下文视觉关联的方法，并构建了包含170万创意描述的MSCOCO数据集。通过生成具抽象性的描述并提升创意领域零样本检索效果，推动创造力理解。**

- **链接: [http://arxiv.org/pdf/2507.18915v1](http://arxiv.org/pdf/2507.18915v1)**

> **作者:** Ananya Sahu; Amith Ananthram; Kathleen McKeown
>
> **摘要:** Understanding another person's creative output requires a shared language of association. However, when training vision-language models such as CLIP, we rely on web-scraped datasets containing short, predominantly literal, alt-text. In this work, we introduce a method for mining contextualized associations for salient visual elements in an image that can scale to any unlabeled dataset. Given an image, we can use these mined associations to generate high quality creative captions at increasing degrees of abstraction. With our method, we produce a new dataset of visual associations and 1.7m creative captions for the images in MSCOCO. Human evaluation confirms that these captions remain visually grounded while exhibiting recognizably increasing abstraction. Moreover, fine-tuning a visual encoder on this dataset yields meaningful improvements in zero-shot image-text retrieval in two creative domains: poetry and metaphor visualization. We release our dataset, our generation code and our models for use by the broader community.
>
---
#### [new 027] NUTMEG: Separating Signal From Noise in Annotator Disagreement
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决多标注者数据中的噪声与真实分歧区分问题。作者提出了贝叶斯模型NUTMEG，利用标注者背景信息过滤噪声，保留系统性分歧，并验证其在合成数据和下游任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.18890v1](http://arxiv.org/pdf/2507.18890v1)**

> **作者:** Jonathan Ivey; Susan Gauch; David Jurgens
>
> **摘要:** NLP models often rely on human-labeled data for training and evaluation. Many approaches crowdsource this data from a large number of annotators with varying skills, backgrounds, and motivations, resulting in conflicting annotations. These conflicts have traditionally been resolved by aggregation methods that assume disagreements are errors. Recent work has argued that for many tasks annotators may have genuine disagreements and that variation should be treated as signal rather than noise. However, few models separate signal and noise in annotator disagreement. In this work, we introduce NUTMEG, a new Bayesian model that incorporates information about annotator backgrounds to remove noisy annotations from human-labeled training data while preserving systematic disagreements. Using synthetic data, we show that NUTMEG is more effective at recovering ground-truth from annotations with systematic disagreement than traditional aggregation methods. We provide further analysis characterizing how differences in subpopulation sizes, rates of disagreement, and rates of spam affect the performance of our model. Finally, we demonstrate that downstream models trained on NUTMEG-aggregated data significantly outperform models trained on data from traditionally aggregation methods. Our results highlight the importance of accounting for both annotator competence and systematic disagreements when training on human-labeled data.
>
---
#### [new 028] Identifying Fine-grained Forms of Populism in Political Discourse: A Case Study on Donald Trump's Presidential Campaigns
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与政治学交叉任务，旨在解决识别政治话语中民粹主义细粒度形式的问题。论文构建了新数据集，评估了多种语言模型的民粹主义识别能力，并提出基于RoBERTa的分类器。分析了特朗普竞选演讲中的民粹主义策略，检验模型在欧洲政治语境中的迁移能力。**

- **链接: [http://arxiv.org/pdf/2507.19303v1](http://arxiv.org/pdf/2507.19303v1)**

> **作者:** Ilias Chalkidis; Stephanie Brandl; Paris Aslanidis
>
> **备注:** Pre-print
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of instruction-following tasks, yet their grasp of nuanced social science concepts remains underexplored. This paper examines whether LLMs can identify and classify fine-grained forms of populism, a complex and contested concept in both academic and media debates. To this end, we curate and release novel datasets specifically designed to capture populist discourse. We evaluate a range of pre-trained (large) language models, both open-weight and proprietary, across multiple prompting paradigms. Our analysis reveals notable variation in performance, highlighting the limitations of LLMs in detecting populist discourse. We find that a fine-tuned RoBERTa classifier vastly outperforms all new-era instruction-tuned LLMs, unless fine-tuned. Additionally, we apply our best-performing model to analyze campaign speeches by Donald Trump, extracting valuable insights into his strategic use of populist rhetoric. Finally, we assess the generalizability of these models by benchmarking them on campaign speeches by European politicians, offering a lens into cross-context transferability in political discourse analysis. In this setting, we find that instruction-tuned LLMs exhibit greater robustness on out-of-domain data.
>
---
#### [new 029] Data Augmentation for Spoken Grammatical Error Correction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于口语语法纠错任务，旨在解决高质量标注口语语法纠错数据不足的问题。论文提出了一种自动化生成带有语法错误和不流利表达的语音-文本对的方法，并设计了评估生成数据的客观指标。实验验证了生成数据在口语和书面语法纠错中的有效性，使用了首个公开的带语法错误标注的口语数据集S&I Corpus进行测试。**

- **链接: [http://arxiv.org/pdf/2507.19374v1](http://arxiv.org/pdf/2507.19374v1)**

> **作者:** Penny Karanasou; Mengjie Qian; Stefano Bannò; Mark J. F. Gales; Kate M. Knill
>
> **备注:** This work has been accepted by ISCA SLaTE 2025
>
> **摘要:** While there exist strong benchmark datasets for grammatical error correction (GEC), high-quality annotated spoken datasets for Spoken GEC (SGEC) are still under-resourced. In this paper, we propose a fully automated method to generate audio-text pairs with grammatical errors and disfluencies. Moreover, we propose a series of objective metrics that can be used to evaluate the generated data and choose the more suitable dataset for SGEC. The goal is to generate an augmented dataset that maintains the textual and acoustic characteristics of the original data while providing new types of errors. This augmented dataset should augment and enrich the original corpus without altering the language assessment scores of the second language (L2) learners. We evaluate the use of the augmented corpus both for written GEC (the text part) and for SGEC (the audio-text pairs). Our experiments are conducted on the S\&I Corpus, the first publicly available speech dataset with grammar error annotations.
>
---
#### [new 030] SLoW: Select Low-frequency Words! Automatic Dictionary Selection for Translation on Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大型语言模型翻译时字典选择效率低的问题。作者提出SLoW方法，通过自动选择低频词字典，在不访问训练数据的情况下优化翻译性能并节省token使用，适用于多语言翻译场景。**

- **链接: [http://arxiv.org/pdf/2507.18902v1](http://arxiv.org/pdf/2507.18902v1)**

> **作者:** Hongyuan Lu; Zixuan Li; Zefan Zhang; Wai Lam
>
> **摘要:** There are more than 7,000 languages around the world, and current Large Language Models (LLMs) only support hundreds of languages. Dictionary-based prompting methods can enhance translation on them, but most methods use all the available dictionaries, which could be expensive. Instead, it will be flexible to have a trade-off between token consumption and translation performance. This paper proposes a novel task called \textbf{A}utomatic \textbf{D}ictionary \textbf{S}election (\textbf{ADS}). The goal of the task is to automatically select which dictionary to use to enhance translation. We propose a novel and effective method which we call \textbf{S}elect \textbf{Lo}w-frequency \textbf{W}ords! (\textbf{SLoW}) which selects those dictionaries that have a lower frequency. Our methods have unique advantages. First, there is no need for access to the training data for frequency estimation (which is usually unavailable). Second, it inherits the advantage of dictionary-based methods, where no additional tuning is required on LLMs. Experimental results on 100 languages from FLORES indicate that SLoW surpasses strong baselines, and it can obviously save token usage, with many languages even surpassing the translation performance of the full dictionary baseline.\footnote{A shocking fact is that there is no need to use the actual training data (often unobtainable) for frequency estimation, and an estimation frequency obtained using public resources is still apparently effective in improving translation with ChatGPT and Llama, and DeepSeek.}\footnote{Code and data available upon publication.}
>
---
#### [new 031] GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG; cs.SE; I.2.7; I.2.6; I.2.4; I.2.8**

- **简介: 论文提出GEPA，一种基于自然语言反思的提示优化方法，用于提升大语言模型在下游任务中的表现。它通过反思试错过程中的轨迹，自动诊断问题并优化提示。相比强化学习方法GRPO，GEPA在更少尝试次数下取得了更高性能，适用于提示优化和代码优化等任务。**

- **链接: [http://arxiv.org/pdf/2507.19457v1](http://arxiv.org/pdf/2507.19457v1)**

> **作者:** Lakshya A Agrawal; Shangyin Tan; Dilara Soylu; Noah Ziems; Rishi Khare; Krista Opsahl-Ong; Arnav Singhvi; Herumb Shandilya; Michael J Ryan; Meng Jiang; Christopher Potts; Koushik Sen; Alexandros G. Dimakis; Ion Stoica; Dan Klein; Matei Zaharia; Omar Khattab
>
> **摘要:** Large language models (LLMs) are increasingly adapted to downstream tasks via reinforcement learning (RL) methods like Group Relative Policy Optimization (GRPO), which often require thousands of rollouts to learn new tasks. We argue that the interpretable nature of language can often provide a much richer learning medium for LLMs, compared with policy gradients derived from sparse, scalar rewards. To test this, we introduce GEPA (Genetic-Pareto), a prompt optimizer that thoroughly incorporates natural language reflection to learn high-level rules from trial and error. Given any AI system containing one or more LLM prompts, GEPA samples system-level trajectories (e.g., reasoning, tool calls, and tool outputs) and reflects on them in natural language to diagnose problems, propose and test prompt updates, and combine complementary lessons from the Pareto frontier of its own attempts. As a result of GEPA's design, it can often turn even just a few rollouts into a large quality gain. Across four tasks, GEPA outperforms GRPO by 10% on average and by up to 20%, while using up to 35x fewer rollouts. GEPA also outperforms the leading prompt optimizer, MIPROv2, by over 10% across two LLMs, and demonstrates promising results as an inference-time search strategy for code optimization.
>
---
#### [new 032] An Empirical Investigation of Gender Stereotype Representation in Large Language Models: The Italian Case
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文研究大型语言模型（LLMs）在意大利语中生成文本时是否体现性别职业刻板印象。任务是分析LLMs对无性别提示的响应是否存在偏见。工作包括设计实验，使用不同职业组合的提示，测试ChatGPT和Gemini模型的响应，发现模型显著关联“她”代词与助理角色，揭示AI生成文本的伦理问题。**

- **链接: [http://arxiv.org/pdf/2507.19156v1](http://arxiv.org/pdf/2507.19156v1)**

> **作者:** Gioele Giachino; Marco Rondina; Antonio Vetrò; Riccardo Coppola; Juan Carlos De Martin
>
> **备注:** 16 pages, European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD 2025) - 5th Workshop on Bias and Fairness in AI (BIAS25)
>
> **摘要:** The increasing use of Large Language Models (LLMs) in a large variety of domains has sparked worries about how easily they can perpetuate stereotypes and contribute to the generation of biased content. With a focus on gender and professional bias, this work examines in which manner LLMs shape responses to ungendered prompts, contributing to biased outputs. This analysis uses a structured experimental method, giving different prompts involving three different professional job combinations, which are also characterized by a hierarchical relationship. This study uses Italian, a language with extensive grammatical gender differences, to highlight potential limitations in current LLMs' ability to generate objective text in non-English languages. Two popular LLM-based chatbots are examined, namely OpenAI ChatGPT (gpt-4o-mini) and Google Gemini (gemini-1.5-flash). Through APIs, we collected a range of 3600 responses. The results highlight how content generated by LLMs can perpetuate stereotypes. For example, Gemini associated 100% (ChatGPT 97%) of 'she' pronouns to the 'assistant' rather than the 'manager'. The presence of bias in AI-generated text can have significant implications in many fields, such as in the workplaces or in job selections, raising ethical concerns about its use. Understanding these risks is pivotal to developing mitigation strategies and assuring that AI-based systems do not increase social inequalities, but rather contribute to more equitable outcomes. Future research directions include expanding the study to additional chatbots or languages, refining prompt engineering methods or further exploiting a larger experimental base.
>
---
#### [new 033] Towards Domain Specification of Embedding Models in Medicine
- **分类: cs.CL**

- **简介: 该论文属于医学文本嵌入模型研究任务，旨在解决现有模型训练数据窄、方法落后及评估不全面的问题。作者提出了MEDTE模型，通过多源数据对比学习生成高质量医学文本嵌入，并构建了包含51个任务的综合基准测试集，显著提升了模型评估与实际应用效果。**

- **链接: [http://arxiv.org/pdf/2507.19407v1](http://arxiv.org/pdf/2507.19407v1)**

> **作者:** Mohammad Khodadad; Ali Shiraee; Mahdi Astaraki; Hamidreza Mahyar
>
> **摘要:** Medical text embedding models are foundational to a wide array of healthcare applications, ranging from clinical decision support and biomedical information retrieval to medical question answering, yet they remain hampered by two critical shortcomings. First, most models are trained on a narrow slice of medical and biological data, beside not being up to date in terms of methodology, making them ill suited to capture the diversity of terminology and semantics encountered in practice. Second, existing evaluations are often inadequate: even widely used benchmarks fail to generalize across the full spectrum of real world medical tasks. To address these gaps, we leverage MEDTE, a GTE model extensively fine-tuned on diverse medical corpora through self-supervised contrastive learning across multiple data sources, to deliver robust medical text embeddings. Alongside this model, we propose a comprehensive benchmark suite of 51 tasks spanning classification, clustering, pair classification, and retrieval modeled on the Massive Text Embedding Benchmark (MTEB) but tailored to the nuances of medical text. Our results demonstrate that this combined approach not only establishes a robust evaluation framework but also yields embeddings that consistently outperform state of the art alternatives in different tasks.
>
---
#### [new 034] Detection of Adverse Drug Events in Dutch clinical free text documents using Transformer Models: benchmark study
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的医学信息抽取任务，旨在解决荷兰语临床文本中不良药物事件（ADE）的检测问题。作者使用多种Transformer模型和Bi-LSTM模型，基于命名实体识别和关系分类技术，评估其在不同临床场景下的ADE检测效果，建立了ADE检测的基准。**

- **链接: [http://arxiv.org/pdf/2507.19396v1](http://arxiv.org/pdf/2507.19396v1)**

> **作者:** Rachel M. Murphy; Nishant Mishra; Nicolette F. de Keizer; Dave A. Dongelmans; Kitty J. Jager; Ameen Abu-Hanna; Joanna E. Klopotowska; Iacer Calixto
>
> **备注:** 30 Pages, 5 Figures (Main Paper), 19 Pages, 2 Figures(Supplements). Rachel M. Murphy and Nishant Mishra are shared first authors. Joanna E. Klopotowska and Iacer Calixto are shared last authors
>
> **摘要:** In this study, we set a benchmark for adverse drug event (ADE) detection in Dutch clinical free text documents using several transformer models, clinical scenarios and fit-for-purpose performance measures. We trained a Bidirectional Long Short-Term Memory (Bi-LSTM) model and four transformer-based Dutch and/or multilingual encoder models (BERTje, RobBERT, MedRoBERTa.nl, and NuNER) for the tasks of named entity recognition (NER) and relation classification (RC) using 102 richly annotated Dutch ICU clinical progress notes. Anonymized free text clinical progress notes of patients admitted to intensive care unit (ICU) of one academic hospital and discharge letters of patients admitted to Internal Medicine wards of two non-academic hospitals were reused. We evaluated our ADE RC models internally using gold standard (two-step task) and predicted entities (end-to-end task). In addition, all models were externally validated on detecting ADEs at the document level. We report both micro- and macro-averaged F1 scores, given the imbalance of ADEs in the datasets. Although differences for the ADE RC task between the models were small, MedRoBERTa.nl was the best performing model with macro-averaged F1 score of 0.63 using gold standard and 0.62 using predicted entities. The MedRoBERTa.nl models also performed the best in our external validation and achieved recall of between 0.67 to 0.74 using predicted entities, meaning between 67 to 74% of discharge letters with ADEs were detected. Our benchmark study presents a robust and clinically meaningful approach for evaluating language models for ADE detection in clinical free text documents. Our study highlights the need to use appropriate performance measures fit for the task of ADE detection in clinical free-text documents and envisioned future clinical use.
>
---
#### [new 035] Legal Document Summarization: Enhancing Judicial Efficiency through Automation Detection
- **分类: cs.CL**

- **简介: 该论文属于法律文本摘要任务，旨在通过自动化提取关键信息提升司法效率。论文提出一种基于自然语言处理和机器学习的方法，从法律文本中识别模式并生成精准摘要，以减轻法律工作者负担并减少遗漏风险。实验表明该方法在保持内容完整性的同时显著提升处理速度。**

- **链接: [http://arxiv.org/pdf/2507.18952v1](http://arxiv.org/pdf/2507.18952v1)**

> **作者:** Yongjie Li; Ruilin Nong; Jianan Liu; Lucas Evans
>
> **摘要:** Legal document summarization represents a significant advancement towards improving judicial efficiency through the automation of key information detection. Our approach leverages state-of-the-art natural language processing techniques to meticulously identify and extract essential data from extensive legal texts, which facilitates a more efficient review process. By employing advanced machine learning algorithms, the framework recognizes underlying patterns within judicial documents to create precise summaries that encapsulate the crucial elements. This automation alleviates the burden on legal professionals, concurrently reducing the likelihood of overlooking vital information that could lead to errors. Through comprehensive experiments conducted with actual legal datasets, we demonstrate the capability of our method to generate high-quality summaries while preserving the integrity of the original content and enhancing processing times considerably. The results reveal marked improvements in operational efficiency, allowing legal practitioners to direct their efforts toward critical analytical and decision-making activities instead of manual reviews. This research highlights promising technology-driven strategies that can significantly alter workflow dynamics within the legal sector, emphasizing the role of automation in refining judicial processes.
>
---
#### [new 036] Debating Truth: Debate-driven Claim Verification with Multiple Large Language Model Agents
- **分类: cs.CL**

- **简介: 该论文属于信息验证任务，旨在解决复杂主张验证问题。现有方法难以处理多方面证据的复杂主张验证。论文提出DebateCV框架，通过多个LLM代理进行辩论式主张验证，包含正反方辩手和评估论点的调解者。通过合成辩论数据增强训练，提升了验证性能。实验表明该方法在多种证据质量下均优于现有技术。**

- **链接: [http://arxiv.org/pdf/2507.19090v1](http://arxiv.org/pdf/2507.19090v1)**

> **作者:** Haorui He; Yupeng Li; Dacheng Wen; Reynold Cheng; Francis C. M. Lau
>
> **摘要:** Claim verification is critical for enhancing digital literacy. However, the state-of-the-art single-LLM methods struggle with complex claim verification that involves multi-faceted evidences. Inspired by real-world fact-checking practices, we propose DebateCV, the first claim verification framework that adopts a debate-driven methodology using multiple LLM agents. In our framework, two Debaters take opposing stances on a claim and engage in multi-round argumentation, while a Moderator evaluates the arguments and renders a verdict with justifications. To further improve the performance of the Moderator, we introduce a novel post-training strategy that leverages synthetic debate data generated by the zero-shot DebateCV, effectively addressing the scarcity of real-world debate-driven claim verification data. Experimental results show that our method outperforms existing claim verification methods under varying levels of evidence quality. Our code and dataset are publicly available at https://anonymous.4open.science/r/DebateCV-6781.
>
---
#### [new 037] Adaptive Learning Systems: Personalized Curriculum Design Using LLM-Powered Analytics
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于教育技术任务，旨在解决传统教育难以个性化的问题。作者提出一种基于大语言模型的自适应学习系统框架，通过实时数据分析，动态调整学习路径并推荐资源。实验表明该方法提升了学习参与度与知识保留率，适用于多种教育环境。**

- **链接: [http://arxiv.org/pdf/2507.18949v1](http://arxiv.org/pdf/2507.18949v1)**

> **作者:** Yongjie Li; Ruilin Nong; Jianan Liu; Lucas Evans
>
> **摘要:** Large language models (LLMs) are revolutionizing the field of education by enabling personalized learning experiences tailored to individual student needs. In this paper, we introduce a framework for Adaptive Learning Systems that leverages LLM-powered analytics for personalized curriculum design. This innovative approach uses advanced machine learning to analyze real-time data, allowing the system to adapt learning pathways and recommend resources that align with each learner's progress. By continuously assessing students, our framework enhances instructional strategies, ensuring that the materials presented are relevant and engaging. Experimental results indicate a marked improvement in both learner engagement and knowledge retention when using a customized curriculum. Evaluations conducted across varied educational environments demonstrate the framework's flexibility and positive influence on learning outcomes, potentially reshaping conventional educational practices into a more adaptive and student-centered model.
>
---
#### [new 038] A Markov Categorical Framework for Language Modeling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言建模任务，旨在理解为何负对数似然（NLL）目标能产生强大表示。作者引入马尔可夫范畴框架，形式化自回归生成过程，并分析NLL优化的隐含机制，揭示其与谱对比学习的联系，以及信息流动和几何结构学习的关系。**

- **链接: [http://arxiv.org/pdf/2507.19247v1](http://arxiv.org/pdf/2507.19247v1)**

> **作者:** Yifan Zhang
>
> **备注:** Project Page: https://github.com/asiresearch/lm-theory
>
> **摘要:** Auto-regressive language models factorize sequence probabilities and are trained by minimizing the negative log-likelihood (NLL) objective. While empirically powerful, a deep theoretical understanding of why this simple objective yields such versatile representations remains elusive. This work introduces a unifying analytical framework using Markov Categories (MCs) to deconstruct the AR generation process and the NLL objective. We model the single-step generation map as a composition of Markov kernels in the category Stoch. This compositional view, when enriched with statistical divergences, allows us to dissect information flow and learned geometry. Our framework makes three main contributions. First, we provide a formal, information-theoretic rationale for the success of modern speculative decoding methods like EAGLE, quantifying the information surplus in hidden states that these methods exploit. Second, we formalize how NLL minimization forces the model to learn not just the next token, but the data's intrinsic conditional stochasticity, a process we analyze using categorical entropy. Third, and most centrally, we prove that NLL training acts as an implicit form of spectral contrastive learning. By analyzing the information geometry of the model's prediction head, we show that NLL implicitly forces the learned representation space to align with the eigenspectrum of a predictive similarity operator, thereby learning a geometrically structured space without explicit contrastive pairs. This compositional and information-geometric perspective reveals the deep structural principles underlying the effectiveness of modern LMs. Project Page: https://github.com/asiresearch/lm-theory
>
---
#### [new 039] Closing the Modality Gap for Mixed Modality Search
- **分类: cs.CV; cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于混合模态搜索任务，旨在解决对比视觉-语言模型（如CLIP）在跨模态检索中存在的模态差距问题。作者提出GR-CLIP方法，有效缩小模态差距，显著提升检索性能。**

- **链接: [http://arxiv.org/pdf/2507.19054v1](http://arxiv.org/pdf/2507.19054v1)**

> **作者:** Binxu Li; Yuhui Zhang; Xiaohan Wang; Weixin Liang; Ludwig Schmidt; Serena Yeung-Levy
>
> **备注:** Project page: https://yuhui-zh15.github.io/MixedModalitySearch/
>
> **摘要:** Mixed modality search -- retrieving information across a heterogeneous corpus composed of images, texts, and multimodal documents -- is an important yet underexplored real-world application. In this work, we investigate how contrastive vision-language models, such as CLIP, perform on the mixed modality search task. Our analysis reveals a critical limitation: these models exhibit a pronounced modality gap in the embedding space, where image and text embeddings form distinct clusters, leading to intra-modal ranking bias and inter-modal fusion failure. To address this issue, we propose GR-CLIP, a lightweight post-hoc calibration method that removes the modality gap in CLIP's embedding space. Evaluated on MixBench -- the first benchmark specifically designed for mixed modality search -- GR-CLIP improves NDCG@10 by up to 26 percentage points over CLIP, surpasses recent vision-language generative embedding models by 4 percentage points, while using 75x less compute.
>
---
#### [new 040] FD-Bench: A Full-Duplex Benchmarking Pipeline Designed for Full Duplex Spoken Dialogue Systems
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决全双工语音对话系统（FDSDS）缺乏有效评估指标的问题。作者设计了FD-Bench基准测试流程，结合大语言模型、语音合成与识别技术，评估系统在用户打断、延迟等复杂场景下的表现，并对多个开源系统进行了测试，揭示了其在频繁干扰和噪声条件下的不足。**

- **链接: [http://arxiv.org/pdf/2507.19040v1](http://arxiv.org/pdf/2507.19040v1)**

> **作者:** Yizhou Peng; Yi-Wen Chao; Dianwen Ng; Yukun Ma; Chongjia Ni; Bin Ma; Eng Siong Chng
>
> **备注:** Accepted to Interspeech 2025. 5 pages
>
> **摘要:** Full-duplex spoken dialogue systems (FDSDS) enable more natural human-machine interactions by allowing real-time user interruptions and backchanneling, compared to traditional SDS that rely on turn-taking. However, existing benchmarks lack metrics for FD scenes, e.g., evaluating model performance during user interruptions. In this paper, we present a comprehensive FD benchmarking pipeline utilizing LLMs, TTS, and ASR to address this gap. It assesses FDSDS's ability to handle user interruptions, manage delays, and maintain robustness in challenging scenarios with diverse novel metrics. We applied our benchmark to three open-source FDSDS (Moshi, Freeze-omni, and VITA-1.5) using over 40 hours of generated speech, with 293 simulated conversations and 1,200 interruptions. The results show that all models continue to face challenges, such as failing to respond to user interruptions, under frequent disruptions and noisy conditions. Demonstrations, data, and code will be released.
>
---
#### [new 041] Distilling a Small Utility-Based Passage Selector to Enhance Retrieval-Augmented Generation
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于检索增强生成（RAG）任务，旨在解决大型语言模型（LLMs）在实用判断中的高计算成本问题。论文提出一种将LLMs的实用判断能力蒸馏到小型模型的方法，通过基于实用性的选择而非排名，动态选择有用段落。实验表明，该方法在降低计算成本的同时提升了复杂问题的回答质量。**

- **链接: [http://arxiv.org/pdf/2507.19102v1](http://arxiv.org/pdf/2507.19102v1)**

> **作者:** Hengran Zhang; Keping Bi; Jiafeng Guo; Jiaming Zhang; Shuaiqiang Wang; Dawei Yin; Xueqi Cheng
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating retrieved information. Standard retrieval process prioritized relevance, focusing on topical alignment between queries and passages. In contrast, in RAG, the emphasis has shifted to utility, which considers the usefulness of passages for generating accurate answers. Despite empirical evidence showing the benefits of utility-based retrieval in RAG, the high computational cost of using LLMs for utility judgments limits the number of passages evaluated. This restriction is problematic for complex queries requiring extensive information. To address this, we propose a method to distill the utility judgment capabilities of LLMs into smaller, more efficient models. Our approach focuses on utility-based selection rather than ranking, enabling dynamic passage selection tailored to specific queries without the need for fixed thresholds. We train student models to learn pseudo-answer generation and utility judgments from teacher LLMs, using a sliding window method that dynamically selects useful passages. Our experiments demonstrate that utility-based selection provides a flexible and cost-effective solution for RAG, significantly reducing computational costs while improving answer quality. We present the distillation results using Qwen3-32B as the teacher model for both relevance ranking and utility-based selection, distilled into RankQwen1.7B and UtilityQwen1.7B. Our findings indicate that for complex questions, utility-based selection is more effective than relevance ranking in enhancing answer generation performance. We will release the relevance ranking and utility-based selection annotations for the MS MARCO dataset, supporting further research in this area.
>
---
#### [new 042] Advancing Event Forecasting through Massive Training of Large Language Models: Challenges, Solutions, and Broader Impacts
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于事件预测任务，旨在提升大语言模型（LLMs）在事件预测中的性能。论文分析了大规模训练LLMs进行事件预测面临的挑战，如数据稀疏、知识截止和奖励结构单一等问题，并提出相应解决方案，包括贝叶斯网络、反事实事件利用和辅助奖励信号等方法，推动LLMs向“超级预测者”水平发展。**

- **链接: [http://arxiv.org/pdf/2507.19477v1](http://arxiv.org/pdf/2507.19477v1)**

> **作者:** Sang-Woo Lee; Sohee Yang; Donghyun Kwak; Noah Y. Siegel
>
> **摘要:** Many recent papers have studied the development of superforecaster-level event forecasting LLMs. While methodological problems with early studies cast doubt on the use of LLMs for event forecasting, recent studies with improved evaluation methods have shown that state-of-the-art LLMs are gradually reaching superforecaster-level performance, and reinforcement learning has also been reported to improve future forecasting. Additionally, the unprecedented success of recent reasoning models and Deep Research-style models suggests that technology capable of greatly improving forecasting performance has been developed. Therefore, based on these positive recent trends, we argue that the time is ripe for research on large-scale training of superforecaster-level event forecasting LLMs. We discuss two key research directions: training methods and data acquisition. For training, we first introduce three difficulties of LLM-based event forecasting training: noisiness-sparsity, knowledge cut-off, and simple reward structure problems. Then, we present related ideas to mitigate these problems: hypothetical event Bayesian networks, utilizing poorly-recalled and counterfactual events, and auxiliary reward signals. For data, we propose aggressive use of market, public, and crawling datasets to enable large-scale training and evaluation. Finally, we explain how these technical advances could enable AI to provide predictive intelligence to society in broader areas. This position paper presents promising specific paths and considerations for getting closer to superforecaster-level AI technology, aiming to call for researchers' interest in these directions.
>
---
#### [new 043] MLLM-based Speech Recognition: When and How is Multimodality Beneficial?
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于多模态语音识别任务，旨在研究多模态大语言模型（MLLMs）在噪声环境中提升语音识别准确率的条件与方法。通过实验分析不同模态的互补性、模态同步性、视觉编码质量等因素对ASR的影响，提供优化策略并加深对多模态语音识别的理解。**

- **链接: [http://arxiv.org/pdf/2507.19037v1](http://arxiv.org/pdf/2507.19037v1)**

> **作者:** Yiwen Guan; Viet Anh Trinh; Vivek Voleti; Jacob Whitehill
>
> **摘要:** Recent advances in multi-modal large language models (MLLMs) have opened new possibilities for unified modeling of speech, text, images, and other modalities. Building on our prior work, this paper examines the conditions and model architectures under which multiple input modalities can improve automatic speech recognition (ASR) accuracy in noisy environments. Through experiments on synthetic and real-world data, we find that (1) harnessing more modalities usually improves ASR accuracy, as each modality provides complementary information, but the improvement depends on the amount of auditory noise. (2) Synchronized modalities (e.g., lip movements) are more useful at high noise levels whereas unsynchronized modalities (e.g., image context) are most helpful at moderate noise levels. (3) Higher-quality visual representations consistently improve ASR accuracy, highlighting the importance of developing more powerful visual encoders. (4) Mamba exhibits similar trends regarding the benefits of multimodality as do Transformers. (5) The input order of modalities as well as their weights in the loss function can significantly impact accuracy. These findings both offer practical insights and help to deepen our understanding of multi-modal speech recognition under challenging conditions.
>
---
#### [new 044] OS-MAP: How Far Can Computer-Using Agents Go in Breadth and Depth?
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文属于计算机使用代理任务，旨在解决现有基准未能充分评估代理在多样化任务中的自动化水平与泛化能力的问题。作者构建了OS-MAP基准，包含416个真实任务，从自动化层级和需求层次两个维度评估代理能力，以推动代理研究与实际应用的发展。**

- **链接: [http://arxiv.org/pdf/2507.19132v1](http://arxiv.org/pdf/2507.19132v1)**

> **作者:** Xuetian Chen; Yinghao Chen; Xinfeng Yuan; Zhuo Peng; Lu Chen; Yuekeng Li; Zhoujia Zhang; Yingqian Huang; Leyan Huang; Jiaqing Liang; Tianbao Xie; Zhiyong Wu; Qiushi Sun; Biqing Qi; Bowen Zhou
>
> **备注:** Work in progress
>
> **摘要:** Computer-using agents have shown strong potential to boost human productivity and enable new application forms across platforms. While recent advances have led to usable applications, existing benchmarks fail to account for the internal task heterogeneity and the corresponding agent capabilities, as well as their alignment with actual user demands-hindering both targeted capability development and the reliable transition of research progress into practical deployment. To bridge the gap, we present OS-MAP, a benchmark for daily computer-using automation that organizes its 416 realistic tasks across 15 applications along two key dimensions: a five-level taxonomy of automation and a generalization scope derived from a real-world user demand hierarchy. To enable fine-grained analysis of required capabilities and alignment with real-world scenarios, OS-MAP evaluates agents along two dimensions: automation level across a five-level taxonomy, and generalization scope across a demand hierarchy. This design captures varying levels of required agent autonomy and generalization, forming a performance-generalization evaluation matrix for structured and comprehensive assessment. Experiments show that even State-of-the-Art agents with VLM backbones struggle with higher-level tasks involving perception, reasoning, and coordination-highlighting the need for a deeper understanding of current strengths and limitations to drive the future progress in computer-using agents research and deployment. All code, environments, baselines, and data are publicly available at https://github.com/OS-Copilot/OS-Map.
>
---
#### [new 045] PurpCode: Reasoning for Safer Code Generation
- **分类: cs.CR; cs.CL; cs.LG; cs.SE**

- **简介: 该论文提出PurpCode，旨在提升代码生成模型的安全性。通过两阶段训练方法（规则学习与强化学习），使模型生成无漏洞代码并防止支持恶意行为。论文属于代码生成与网络安全任务，解决代码安全与模型滥用问题。**

- **链接: [http://arxiv.org/pdf/2507.19060v1](http://arxiv.org/pdf/2507.19060v1)**

> **作者:** Jiawei Liu; Nirav Diwan; Zhe Wang; Haoyu Zhai; Xiaona Zhou; Kiet A. Nguyen; Tianjiao Yu; Muntasir Wahed; Yinlin Deng; Hadjer Benkraouda; Yuxiang Wei; Lingming Zhang; Ismini Lourentzou; Gang Wang
>
> **摘要:** We introduce PurpCode, the first post-training recipe for training safe code reasoning models towards generating secure code and defending against malicious cyberactivities. PurpCode trains a reasoning model in two stages: (i) Rule Learning, which explicitly teaches the model to reference cybersafety rules to generate vulnerability-free code and to avoid facilitating malicious cyberactivities; and (ii) Reinforcement Learning, which optimizes model safety and preserves model utility through diverse, multi-objective reward mechanisms. To empower the training pipelines with comprehensive cybersafety data, we conduct internal red-teaming to synthesize comprehensive and high-coverage prompts based on real-world tasks for inducing unsafe cyberactivities in the model. Based on PurpCode, we develop a reasoning-based coding model, namely PurpCode-32B, which demonstrates state-of-the-art cybersafety, outperforming various frontier models. Meanwhile, our alignment method decreases the model overrefusal rates in both general and cybersafety-specific scenarios, while preserving model utility in both code generation and common security knowledge.
>
---
#### [new 046] People Are Highly Cooperative with Large Language Models, Especially When Communication Is Possible or Following Human Interaction
- **分类: cs.HC; cs.CL; cs.CY; econ.GN; q-fin.EC; I.2.7; H.5.2; H.5.3; K.4.3**

- **简介: 该论文研究人类与大语言模型（LLM）在合作行为上的差异，属于人机交互与行为经济学任务。它旨在解决在商业环境中LLM能否有效替代人类进行合作的问题。通过囚徒困境实验，比较人类与LLM互动时的合作率，发现LLM虽合作率略低，但在可沟通或与人类互动后合作显著提升。**

- **链接: [http://arxiv.org/pdf/2507.18639v1](http://arxiv.org/pdf/2507.18639v1)**

> **作者:** Paweł Niszczota; Tomasz Grzegorczyk; Alexander Pastukhov
>
> **摘要:** Machines driven by large language models (LLMs) have the potential to augment humans across various tasks, a development with profound implications for business settings where effective communication, collaboration, and stakeholder trust are paramount. To explore how interacting with an LLM instead of a human might shift cooperative behavior in such settings, we used the Prisoner's Dilemma game -- a surrogate of several real-world managerial and economic scenarios. In Experiment 1 (N=100), participants engaged in a thirty-round repeated game against a human, a classic bot, and an LLM (GPT, in real-time). In Experiment 2 (N=192), participants played a one-shot game against a human or an LLM, with half of them allowed to communicate with their opponent, enabling LLMs to leverage a key advantage over older-generation machines. Cooperation rates with LLMs -- while lower by approximately 10-15 percentage points compared to interactions with human opponents -- were nonetheless high. This finding was particularly notable in Experiment 2, where the psychological cost of selfish behavior was reduced. Although allowing communication about cooperation did not close the human-machine behavioral gap, it increased the likelihood of cooperation with both humans and LLMs equally (by 88%), which is particularly surprising for LLMs given their non-human nature and the assumption that people might be less receptive to cooperating with machines compared to human counterparts. Additionally, cooperation with LLMs was higher following prior interaction with humans, suggesting a spillover effect in cooperative behavior. Our findings validate the (careful) use of LLMs by businesses in settings that have a cooperative component.
>
---
#### [new 047] Towards Multimodal Social Conversations with Robots: Using Vision-Language Models
- **分类: cs.RO; cs.CL; cs.HC**

- **简介: 该论文探讨如何使社交机器人通过视觉-语言模型实现多模态社交对话。任务是提升机器人在开放域对话中的社交能力，解决其缺乏多模态交互能力的问题。工作包括分析多模态系统需求、提出使用视觉-语言模型的方案、讨论技术挑战与评估方法。**

- **链接: [http://arxiv.org/pdf/2507.19196v1](http://arxiv.org/pdf/2507.19196v1)**

> **作者:** Ruben Janssens; Tony Belpaeme
>
> **备注:** Submitted to the workshop "Human - Foundation Models Interaction: A Focus On Multimodal Information" (FoMo-HRI) at IEEE RO-MAN 2025
>
> **摘要:** Large language models have given social robots the ability to autonomously engage in open-domain conversations. However, they are still missing a fundamental social skill: making use of the multiple modalities that carry social interactions. While previous work has focused on task-oriented interactions that require referencing the environment or specific phenomena in social interactions such as dialogue breakdowns, we outline the overall needs of a multimodal system for social conversations with robots. We then argue that vision-language models are able to process this wide range of visual information in a sufficiently general manner for autonomous social robots. We describe how to adapt them to this setting, which technical challenges remain, and briefly discuss evaluation practices.
>
---
#### [new 048] MMBench-GUI: Hierarchical Multi-Platform Evaluation Framework for GUI Agents
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于GUI自动化评估任务，旨在解决跨平台GUI代理评估与效率问题。作者提出了MMBench-GUI基准框架，包含四个评估层级，并设计了EQA指标衡量效率。研究发现视觉定位与模块化框架对任务成功至关重要，强调长期推理与高效策略的必要性。**

- **链接: [http://arxiv.org/pdf/2507.19478v1](http://arxiv.org/pdf/2507.19478v1)**

> **作者:** Xuehui Wang; Zhenyu Wu; JingJing Xie; Zichen Ding; Bowen Yang; Zehao Li; Zhaoyang Liu; Qingyun Li; Xuan Dong; Zhe Chen; Weiyun Wang; Xiangyu Zhao; Jixuan Chen; Haodong Duan; Tianbao Xie; Chenyu Yang; Shiqian Su; Yue Yu; Yuan Huang; Yiqian Liu; Xiao Zhang; Yanting Zhang; Xiangyu Yue; Weijie Su; Xizhou Zhu; Wei Shen; Jifeng Dai; Wenhai Wang
>
> **备注:** in progress
>
> **摘要:** We introduce MMBench-GUI, a hierarchical benchmark for evaluating GUI automation agents across Windows, macOS, Linux, iOS, Android, and Web platforms. It comprises four levels: GUI Content Understanding, Element Grounding, Task Automation, and Task Collaboration, covering essential skills for GUI agents. In addition, we propose a novel Efficiency-Quality Area (EQA) metric to assess GUI agent execution efficiency in online automation scenarios. Through MMBench-GUI, we identify accurate visual grounding as a critical determinant of overall task success, emphasizing the substantial benefits of modular frameworks that integrate specialized grounding modules. Furthermore, to achieve reliable GUI automation, an agent requires strong task planning and cross-platform generalization abilities, with long-context memory, a broad action space, and long-term reasoning playing a critical role. More important, task efficiency remains a critically underexplored dimension, and all models suffer from substantial inefficiencies, with excessive redundant steps even when tasks are ultimately completed. The integration of precise localization, effective planning, and early stopping strategies is indispensable to enable truly efficient and scalable GUI automation. Our benchmark code, evaluation data, and running environment will be publicly available at https://github.com/open-compass/MMBench-GUI.
>
---
#### [new 049] TreeReader: A Hierarchical Academic Paper Reader Powered by Language Models
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于信息可视化与学术阅读辅助工具任务，旨在解决传统线性格式阅读学术论文时的信息定位困难与认知负担问题。作者设计了TreeReader，利用语言模型将论文结构化为可交互的树状摘要，支持按需展开细节，提升阅读效率与理解效果。**

- **链接: [http://arxiv.org/pdf/2507.18945v1](http://arxiv.org/pdf/2507.18945v1)**

> **作者:** Zijian Zhang; Pan Chen; Fangshi Du; Runlong Ye; Oliver Huang; Michael Liut; Alán Aspuru-Guzik
>
> **摘要:** Efficiently navigating and understanding academic papers is crucial for scientific progress. Traditional linear formats like PDF and HTML can cause cognitive overload and obscure a paper's hierarchical structure, making it difficult to locate key information. While LLM-based chatbots offer summarization, they often lack nuanced understanding of specific sections, may produce unreliable information, and typically discard the document's navigational structure. Drawing insights from a formative study on academic reading practices, we introduce TreeReader, a novel language model-augmented paper reader. TreeReader decomposes papers into an interactive tree structure where each section is initially represented by an LLM-generated concise summary, with underlying details accessible on demand. This design allows users to quickly grasp core ideas, selectively explore sections of interest, and verify summaries against the source text. A user study was conducted to evaluate TreeReader's impact on reading efficiency and comprehension. TreeReader provides a more focused and efficient way to navigate and understand complex academic literature by bridging hierarchical summarization with interactive exploration.
>
---
#### [new 050] LOTUS: A Leaderboard for Detailed Image Captioning from Quality to Societal Bias and User Preferences
- **分类: cs.CV; cs.AI; cs.CL; cs.CY; cs.LG**

- **简介: 该论文属于图像描述生成任务，旨在解决当前评估方法缺乏统一标准、社会偏见考量及用户偏好整合的问题。作者构建了LOTUS排行榜，全面评估描述质量、风险及社会偏见，并引入用户偏好导向的评价标准。**

- **链接: [http://arxiv.org/pdf/2507.19362v1](http://arxiv.org/pdf/2507.19362v1)**

> **作者:** Yusuke Hirota; Boyi Li; Ryo Hachiuma; Yueh-Hua Wu; Boris Ivanovic; Yuta Nakashima; Marco Pavone; Yejin Choi; Yu-Chiang Frank Wang; Chao-Han Huck Yang
>
> **备注:** Accepted to ACL 2025. Leaderboard: huggingface.co/spaces/nvidia/lotus-vlm-bias-leaderboard
>
> **摘要:** Large Vision-Language Models (LVLMs) have transformed image captioning, shifting from concise captions to detailed descriptions. We introduce LOTUS, a leaderboard for evaluating detailed captions, addressing three main gaps in existing evaluations: lack of standardized criteria, bias-aware assessments, and user preference considerations. LOTUS comprehensively evaluates various aspects, including caption quality (e.g., alignment, descriptiveness), risks (\eg, hallucination), and societal biases (e.g., gender bias) while enabling preference-oriented evaluations by tailoring criteria to diverse user preferences. Our analysis of recent LVLMs reveals no single model excels across all criteria, while correlations emerge between caption detail and bias risks. Preference-oriented evaluations demonstrate that optimal model selection depends on user priorities.
>
---
#### [new 051] Should Top-Down Clustering Affect Boundaries in Unsupervised Word Discovery?
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于无监督词发现任务，旨在将未标注语音分割为类词单元并聚类生成词典。论文比较了自下而上和自上而下的聚类方法对边界选择的影响，发现两者在多语言数据上表现相当，但自下而上方法更快。分析表明，聚类步骤是性能瓶颈，建议未来研究更优聚类技术和表征学习。**

- **链接: [http://arxiv.org/pdf/2507.19204v1](http://arxiv.org/pdf/2507.19204v1)**

> **作者:** Simon Malan; Benjamin van Niekerk; Herman Kamper
>
> **备注:** 5 figures, 5 tables
>
> **摘要:** We investigate the problem of segmenting unlabeled speech into word-like units and clustering these to create a lexicon. Prior work can be categorized into two frameworks. Bottom-up methods first determine boundaries and then cluster the fixed segmented words into a lexicon. In contrast, top-down methods incorporate information from the clustered words to inform boundary selection. However, it is unclear whether top-down information is necessary to improve segmentation. To explore this, we look at two similar approaches that differ in whether top-down clustering informs boundary selection. Our simple bottom-up strategy predicts word boundaries using the dissimilarity between adjacent self-supervised features, then clusters the resulting segments to construct a lexicon. Our top-down system is an updated version of the ES-KMeans dynamic programming method that iteratively uses K-means to update its boundaries. On the five-language ZeroSpeech benchmarks, both approaches achieve comparable state-of-the-art results, with the bottom-up system being nearly five times faster. Through detailed analyses, we show that the top-down influence of ES-KMeans can be beneficial (depending on factors like the candidate boundaries), but in many cases the simple bottom-up method performs just as well. For both methods, we show that the clustering step is a limiting factor. Therefore, we recommend that future work focus on improved clustering techniques and learning more discriminative word-like representations. Project code repository: https://github.com/s-malan/prom-seg-clus.
>
---
#### [new 052] Injecting External Knowledge into the Reasoning Process Enhances Retrieval-Augmented Generation
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于知识密集型任务中的检索增强生成（RAG）任务，旨在解决RAG系统因检索到低质量内容导致效果下降的问题。作者提出Passage Injection方法，将检索内容显式融入大语言模型的推理过程，以增强其识别和抵御噪声内容的能力，从而提升RAG系统的鲁棒性与性能。**

- **链接: [http://arxiv.org/pdf/2507.19333v1](http://arxiv.org/pdf/2507.19333v1)**

> **作者:** Minghao Tang; Shiyu Ni; Jiafeng Guo; Keping Bi
>
> **摘要:** Retrieval-augmented generation (RAG) has been widely adopted to augment large language models (LLMs) with external knowledge for knowledge-intensive tasks. However, its effectiveness is often undermined by the presence of noisy (i.e., low-quality) retrieved passages. Enhancing LLMs' robustness to such noise is critical for improving the reliability of RAG systems. Recent advances have equipped LLMs with strong reasoning and self-reflection capabilities, allowing them to identify and correct errors in their reasoning process. Inspired by this ability, we propose Passage Injection-a simple yet effective method that explicitly incorporates retrieved passages into LLMs' reasoning process, aiming to enhance the model's ability to recognize and resist noisy passages. We validate Passage Injection under general RAG settings using BM25 as the retriever. Experiments on four reasoning-enhanced LLMs across four factual QA datasets demonstrate that Passage Injection significantly improves overall RAG performance. Further analysis on two noisy retrieval settings-random noise, where the model is provided irrelevant passages, and counterfactual noise, where it is given misleading passages-shows that Passage Injection consistently improves robustness. Controlled experiments confirm that Passage Injection can also effectively leverage helpful passages. These findings suggest that incorporating passages in LLMs' reasoning process is a promising direction for building more robust RAG systems. The code can be found \href{here}{https://github.com/mh-tang/Passage-Injection}.
>
---
## 更新

#### [replaced 001] GOAT-SLM: A Spoken Language Model with Paralinguistic and Speaker Characteristic Awareness
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.18119v2](http://arxiv.org/pdf/2507.18119v2)**

> **作者:** Hongjie Chen; Zehan Li; Yaodong Song; Wenming Deng; Yitong Yao; Yuxin Zhang; Hang Lv; Xuechao Zhu; Jian Kang; Jie Lian; Jie Li; Chao Wang; Shuangyong Song; Yongxiang Li; Zhongjiang He; Xuelong Li
>
> **摘要:** Recent advances in end-to-end spoken language models (SLMs) have significantly improved the ability of AI systems to engage in natural spoken interactions. However, most existing models treat speech merely as a vehicle for linguistic content, often overlooking the rich paralinguistic and speaker characteristic cues embedded in human speech, such as dialect, age, emotion, and non-speech vocalizations. In this work, we introduce GOAT-SLM, a novel spoken language model with paralinguistic and speaker characteristic awareness, designed to extend spoken language modeling beyond text semantics. GOAT-SLM adopts a dual-modality head architecture that decouples linguistic modeling from acoustic realization, enabling robust language understanding while supporting expressive and adaptive speech generation. To enhance model efficiency and versatility, we propose a modular, staged training strategy that progressively aligns linguistic, paralinguistic, and speaker characteristic information using large-scale speech-text corpora. Experimental results on TELEVAL, a multi-dimensional evaluation benchmark, demonstrate that GOAT-SLM achieves well-balanced performance across both semantic and non-semantic tasks, and outperforms existing open-source models in handling emotion, dialectal variation, and age-sensitive interactions. This work highlights the importance of modeling beyond linguistic content and advances the development of more natural, adaptive, and socially aware spoken language systems.
>
---
#### [replaced 002] Re:Form -- Reducing Human Priors in Scalable Formal Software Verification with RL in LLMs: A Preliminary Study on Dafny
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.16331v2](http://arxiv.org/pdf/2507.16331v2)**

> **作者:** Chuanhao Yan; Fengdi Che; Xuhan Huang; Xu Xu; Xin Li; Yizhi Li; Xingwei Qu; Jingzhe Shi; Zhuangzhuang He; Chenghua Lin; Yaodong Yang; Binhang Yuan; Hang Zhao; Yu Qiao; Bowen Zhou; Jie Fu
>
> **摘要:** Existing informal language-based (e.g., human language) Large Language Models (LLMs) trained with Reinforcement Learning (RL) face a significant challenge: their verification processes, which provide crucial training signals, are neither reliable nor scalable. In fact, the prevalent large proprietary models could hardly generate verifiable programs. A promising yet largely uncharted alternative is formal language-based reasoning. Grounding LLMs in rigorous formal systems where generative models operate in formal language spaces (e.g., Dafny) enables the automatic and mathematically provable verification of their reasoning processes and outcomes. This capability is pivotal for achieving large-scale, reliable formal software verification. It is a common practice to employ human-annotated chain-of-thought and other human priors to induce the reasoning and coding capabilities of LLMs. Unfortunately, it becomes unacceptably all-consuming to provide such priors for supervising complex programming tasks. In this work, we systematically explore ways to reduce human priors with the formal language, Dafny, as the main environment for our pilot study. Our pipeline mainly relies on introducing an automatic and scalable data curation pipeline, and careful RL designs integrated with feedback from the formal language verifier. We introduce DafnyComp, a benchmark of compositional formal programs with auto-formalized specifications for specification reasoning. Our supervised fine-tuning (SFT) stage enables even small models (e.g., 0.5B) to generate syntactically valid and verifiable Dafny code, surpassing proprietary models. RL with regularization further improves performance, achieving stronger generalization to out-of-domain tasks and outperforming all strong baselines on the challenging DafnyComp benchmark.
>
---
#### [replaced 003] Promptomatix: An Automatic Prompt Optimization Framework for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.14241v3](http://arxiv.org/pdf/2507.14241v3)**

> **作者:** Rithesh Murthy; Ming Zhu; Liangwei Yang; Jielin Qiu; Juntao Tan; Shelby Heinecke; Caiming Xiong; Silvio Savarese; Huan Wang
>
> **摘要:** Large Language Models (LLMs) perform best with well-crafted prompts, yet prompt engineering remains manual, inconsistent, and inaccessible to non-experts. We introduce Promptomatix, an automatic prompt optimization framework that transforms natural language task descriptions into high-quality prompts without requiring manual tuning or domain expertise. Promptomatix supports both a lightweight meta-prompt-based optimizer and a DSPy-powered compiler, with modular design enabling future extension to more advanced frameworks. The system analyzes user intent, generates synthetic training data, selects prompting strategies, and refines prompts using cost-aware objectives. Evaluated across 5 task categories, Promptomatix achieves competitive or superior performance compared to existing libraries, while reducing prompt length and computational overhead making prompt optimization scalable and efficient.
>
---
#### [replaced 004] Acoustically Precise Hesitation Tagging Is Essential for End-to-End Verbatim Transcription Systems
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.04076v2](http://arxiv.org/pdf/2506.04076v2)**

> **作者:** Jhen-Ke Lin; Hao-Chien Lu; Chung-Chun Wang; Hong-Yun Lin; Berlin Chen
>
> **备注:** accepted to the ISCA SLaTE-2025 Workshop
>
> **摘要:** Verbatim transcription for automatic speaking assessment demands accurate capture of disfluencies, crucial for downstream tasks like error analysis and feedback. However, many ASR systems discard or generalize hesitations, losing important acoustic details. We fine-tune Whisper models on the Speak & Improve 2025 corpus using low-rank adaptation (LoRA), without recourse to external audio training data. We compare three annotation schemes: removing hesitations (Pure), generic tags (Rich), and acoustically precise fillers inferred by Gemini 2.0 Flash from existing audio-transcript pairs (Extra). Our challenge system achieved 6.47% WER (Pure) and 5.81% WER (Extra). Post-challenge experiments reveal that fine-tuning Whisper Large V3 Turbo with the "Extra" scheme yielded a 5.5% WER, an 11.3% relative improvement over the "Pure" scheme (6.2% WER). This demonstrates that explicit, realistic filled-pause labeling significantly enhances ASR accuracy for verbatim L2 speech transcription.
>
---
#### [replaced 005] JCAPT: A Joint Modeling Approach for CAPT
- **分类: cs.CL; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.19315v2](http://arxiv.org/pdf/2506.19315v2)**

> **作者:** Tzu-Hsuan Yang; Yue-Yang He; Berlin Chen
>
> **备注:** Accepted to the ISCA SLaTE-2025 Workshop
>
> **摘要:** Effective pronunciation feedback is critical in second language (L2) learning, for which computer-assisted pronunciation training (CAPT) systems often encompass two key tasks: automatic pronunciation assessment (APA) and mispronunciation detection and diagnosis (MDD). Recent work has shown that joint modeling of these two tasks can yield mutual benefits. Our unified framework leverages Mamba, a selective state space model (SSM), while integrating phonological features and think token strategies to jointly enhance interpretability and fine-grained temporal reasoning in APA and MDD. To our knowledge, this is the first study to combine phonological attribution, SSM-based modeling, and prompting in CAPT. A series of experiments conducted on the speechocean762 benchmark demonstrate that our model consistently outperforms prior methods, particularly on the MDD task.
>
---
#### [replaced 006] Ensemble Debiasing Across Class and Sample Levels for Fairer Prompting Accuracy
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.05157v4](http://arxiv.org/pdf/2503.05157v4)**

> **作者:** Ruixi Lin; Ziqiao Wang; Yang You
>
> **备注:** Published as a conference paper at COLM 2025
>
> **摘要:** Language models are strong few-shot learners and achieve good overall accuracy in text classification tasks, masking the fact that their results suffer from great class accuracy imbalance. We believe that the pursuit of overall accuracy should not come from enriching the strong classes, but from raising up the weak ones. To address the imbalance, we propose a Heaviside step function based ensemble debiasing method, which enables flexible rectifications of in-context learned class probabilities at both class and sample levels. Evaluations with Llama-2-13B on seven text classification benchmarks show that our approach achieves state-of-the-art overall accuracy gains with balanced class accuracies. More importantly, we perform analyses on the resulted probability correction scheme, showing that sample-level corrections are necessary to elevate weak classes. Due to effectively correcting weak classes, our method also brings significant performance gains to a larger model variant, Llama-2-70B, especially on a biomedical domain task, further demonstrating the necessity of ensemble debiasing at both levels. Our source code is available at https://github.com/NUS-HPC-AI-Lab/DCS.
>
---
#### [replaced 007] Technical Report of TeleChat2, TeleChat2.5 and T1
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2507.18013v2](http://arxiv.org/pdf/2507.18013v2)**

> **作者:** Zihan Wang; Xinzhang Liu; Yitong Yao; Chao Wang; Yu Zhao; Zhihao Yang; Wenmin Deng; Kaipeng Jia; Jiaxin Peng; Yuyao Huang; Sishi Xiong; Zhuo Jiang; Kaidong Yu; Xiaohui Hu; Fubei Yao; Ruiyu Fang; Zhuoru Jiang; Ruiting Song; Qiyi Xie; Rui Xue; Xuewei He; Yanlei Xue; Zhu Yuan; Zhaoxi Zhang; Zilu Huang; Shiquan Wang; Xin Wang; Hanming Wu; Mingyuan Wang; Xufeng Zhan; Yuhan Sun; Zhaohu Xing; Yuhao Jiang; Bingkai Yang; Shuangyong Song; Yongxiang Li; Zhongjiang He; Xuelong Li
>
> **备注:** 32 pages, 5 figures
>
> **摘要:** We introduce the latest series of TeleChat models: \textbf{TeleChat2}, \textbf{TeleChat2.5}, and \textbf{T1}, offering a significant upgrade over their predecessor, TeleChat. Despite minimal changes to the model architecture, the new series achieves substantial performance gains through enhanced training strategies in both pre-training and post-training stages. The series begins with \textbf{TeleChat2}, which undergoes pretraining on 10 trillion high-quality and diverse tokens. This is followed by Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to further enhance its capabilities. \textbf{TeleChat2.5} and \textbf{T1} expand the pipeline by incorporating a continual pretraining phase with domain-specific datasets, combined with reinforcement learning (RL) to improve performance in code generation and mathematical reasoning tasks. The \textbf{T1} variant is designed for complex reasoning, supporting long Chain-of-Thought (CoT) reasoning and demonstrating substantial improvements in mathematics and coding. In contrast, \textbf{TeleChat2.5} prioritizes speed, delivering rapid inference. Both flagship models of \textbf{T1} and \textbf{TeleChat2.5} are dense Transformer-based architectures with 115B parameters, showcasing significant advancements in reasoning and general task performance compared to the original TeleChat. Notably, \textbf{T1-115B} outperform proprietary models such as OpenAI's o1-mini and GPT-4o. We publicly release \textbf{TeleChat2}, \textbf{TeleChat2.5} and \textbf{T1}, including post-trained versions with 35B and 115B parameters, to empower developers and researchers with state-of-the-art language models tailored for diverse applications.
>
---
#### [replaced 008] Noise Contrastive Estimation-based Matching Framework for Low-Resource Security Attack Pattern Recognition
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2401.10337v4](http://arxiv.org/pdf/2401.10337v4)**

> **作者:** Tu Nguyen; Nedim Šrndić; Alexander Neth
>
> **备注:** accepted at EACL 2024, in ARR October 2023
>
> **摘要:** Tactics, Techniques and Procedures (TTPs) represent sophisticated attack patterns in the cybersecurity domain, described encyclopedically in textual knowledge bases. Identifying TTPs in cybersecurity writing, often called TTP mapping, is an important and challenging task. Conventional learning approaches often target the problem in the classical multi-class or multilabel classification setting. This setting hinders the learning ability of the model due to a large number of classes (i.e., TTPs), the inevitable skewness of the label distribution and the complex hierarchical structure of the label space. We formulate the problem in a different learning paradigm, where the assignment of a text to a TTP label is decided by the direct semantic similarity between the two, thus reducing the complexity of competing solely over the large labeling space. To that end, we propose a neural matching architecture with an effective sampling-based learn-to-compare mechanism, facilitating the learning process of the matching model despite constrained resources.
>
---
#### [replaced 009] References Matter: Investigating the Impact of Reference Set Variation on Summarization Evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14335v2](http://arxiv.org/pdf/2506.14335v2)**

> **作者:** Silvia Casola; Yang Janet Liu; Siyao Peng; Oliver Kraus; Albert Gatt; Barbara Plank
>
> **摘要:** Human language production exhibits remarkable richness and variation, reflecting diverse communication styles and intents. However, this variation is often overlooked in summarization evaluation. While having multiple reference summaries is known to improve correlation with human judgments, the impact of the reference set on reference-based metrics has not been systematically investigated. This work examines the sensitivity of widely used reference-based metrics in relation to the choice of reference sets, analyzing three diverse multi-reference summarization datasets: SummEval, GUMSum, and DUC2004. We demonstrate that many popular metrics exhibit significant instability. This instability is particularly concerning for n-gram-based metrics like ROUGE, where model rankings vary depending on the reference sets, undermining the reliability of model comparisons. We also collect human judgments on LLM outputs for genre-diverse data and examine their correlation with metrics to supplement existing findings beyond newswire summaries, finding weak-to-no correlation. Taken together, we recommend incorporating reference set variation into summarization evaluation to enhance consistency alongside correlation with human judgments, especially when evaluating LLMs.
>
---
#### [replaced 010] Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.00151v2](http://arxiv.org/pdf/2503.00151v2)**

> **作者:** Fakhraddin Alwajih; Abdellah El Mekki; Samar Mohamed Magdy; Abdelrahim A. Elmadany; Omer Nacar; El Moatez Billah Nagoudi; Reem Abdel-Salam; Hanin Atwany; Youssef Nafea; Abdulfattah Mohammed Yahya; Rahaf Alhamouri; Hamzah A. Alsayadi; Hiba Zayed; Sara Shatnawi; Serry Sibaee; Yasir Ech-Chammakhy; Walid Al-Dhabyani; Marwa Mohamed Ali; Imen Jarraya; Ahmed Oumar El-Shangiti; Aisha Alraeesi; Mohammed Anwar Al-Ghrawi; Abdulrahman S. Al-Batati; Elgizouli Mohamed; Noha Taha Elgindi; Muhammed Saeed; Houdaifa Atou; Issam Ait Yahia; Abdelhak Bouayad; Mohammed Machrouh; Amal Makouar; Dania Alkawi; Mukhtar Mohamed; Safaa Taher Abdelfadil; Amine Ziad Ounnoughene; Rouabhia Anfel; Rwaa Assi; Ahmed Sorkatti; Mohamedou Cheikh Tourad; Anis Koubaa; Ismail Berrada; Mustafa Jarrar; Shady Shehata; Muhammad Abdul-Mageed
>
> **备注:** More information about our dataset is available at our project page: https://github.com/UBC-NLP/palm
>
> **摘要:** As large language models (LLMs) become increasingly integrated into daily life, ensuring their cultural sensitivity and inclusivity is paramount. We introduce our dataset, a year-long community-driven project covering all 22 Arab countries. The dataset includes instructions (input, response pairs) in both Modern Standard Arabic (MSA) and dialectal Arabic (DA), spanning 20 diverse topics. Built by a team of 44 researchers across the Arab world, all of whom are authors of this paper, our dataset offers a broad, inclusive perspective. We use our dataset to evaluate the cultural and dialectal capabilities of several frontier LLMs, revealing notable limitations. For instance, while closed-source LLMs generally exhibit strong performance, they are not without flaws, and smaller open-source models face greater challenges. Moreover, certain countries (e.g., Egypt, the UAE) appear better represented than others (e.g., Iraq, Mauritania, Yemen). Our annotation guidelines, code, and data for reproducibility are publicly available.
>
---
#### [replaced 011] Accelerating Multimodal Large Language Models via Dynamic Visual-Token Exit and the Empirical Findings
- **分类: cs.CV; cs.CL; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2411.19628v2](http://arxiv.org/pdf/2411.19628v2)**

> **作者:** Qiong Wu; Wenhao Lin; Yiyi Zhou; Weihao Ye; Zhanpeng Zen; Xiaoshuai Sun; Rongrong Ji
>
> **摘要:** The excessive use of visual tokens in existing Multimoal Large Language Models (MLLMs) often exhibits obvious redundancy and brings in prohibitively expensive computation. To gain insights into this problem, we first conduct extensive empirical studies on the attention behaviors of MLLMs, and summarize three main inference stages in MLLMs: (i) Early fusion between tokens is first accomplished quickly. (ii) Intra-modality modeling then comes to play. (iii) Multimodal reasoning} resumes and lasts until the end of inference. In particular, we reveal that visual tokens will stop contributing to reasoning when the text tokens receive enough image information, yielding obvious visual redundancy. Based on these generalized observations, we propose a simple yet effective method to improve the efficiency of MLLMs, termed dynamic visual-token exit (DyVTE). DyVTE uses lightweight hyper-networks to perceive the text token status and decide the removal of all visual tokens after a certain layer, thereby addressing the observed visual redundancy. To validate VTE, we apply it to a set of MLLMs, including LLaVA, VILA, Eagle and InternVL, and conduct extensive experiments on a bunch of benchmarks. The experiment results not only show the effectiveness of our VTE in improving MLLMs' efficiency, but also yield the general modeling patterns of MLLMs, well facilitating the in-depth understanding of MLLMs. Our code is released at https://github.com/DoubtedSteam/DyVTE.
>
---
#### [replaced 012] MedicalBERT: enhancing biomedical natural language processing using pretrained BERT-based model
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.08013v2](http://arxiv.org/pdf/2507.08013v2)**

> **作者:** K. Sahit Reddy; N. Ragavenderan; Vasanth K.; Ganesh N. Naik; Vishalakshi Prabhu; Nagaraja G. S
>
> **摘要:** Recent advances in natural language processing (NLP) have been driven bypretrained language models like BERT, RoBERTa, T5, and GPT. Thesemodels excel at understanding complex texts, but biomedical literature, withits domain-specific terminology, poses challenges that models likeWord2Vec and bidirectional long short-term memory (Bi-LSTM) can't fullyaddress. GPT and T5, despite capturing context, fall short in tasks needingbidirectional understanding, unlike BERT. Addressing this, we proposedMedicalBERT, a pretrained BERT model trained on a large biomedicaldataset and equipped with domain-specific vocabulary that enhances thecomprehension of biomedical terminology. MedicalBERT model is furtheroptimized and fine-tuned to address diverse tasks, including named entityrecognition, relation extraction, question answering, sentence similarity, anddocument classification. Performance metrics such as the F1-score,accuracy, and Pearson correlation are employed to showcase the efficiencyof our model in comparison to other BERT-based models such as BioBERT,SciBERT, and ClinicalBERT. MedicalBERT outperforms these models onmost of the benchmarks, and surpasses the general-purpose BERT model by5.67% on average across all the tasks evaluated respectively. This work alsounderscores the potential of leveraging pretrained BERT models for medicalNLP tasks, demonstrating the effectiveness of transfer learning techniques incapturing domain-specific information. (PDF) MedicalBERT: enhancing biomedical natural language processing using pretrained BERT-based model. Available from: https://www.researchgate.net/publication/392489050_MedicalBERT_enhancing_biomedical_natural_language_processing_using_pretrained_BERT-based_model [accessed Jul 06 2025].
>
---
#### [replaced 013] Relation Extraction with Instance-Adapted Predicate Descriptions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.17799v2](http://arxiv.org/pdf/2503.17799v2)**

> **作者:** Yuhang Jiang; Ramakanth Kavuluru
>
> **备注:** This paper has been accepted to appear in the proceedings of AMIA 2025
>
> **摘要:** Relation extraction (RE) is a standard information extraction task playing a major role in downstream applications such as knowledge discovery and question answering. Although decoder-only large language models are excelling in generative tasks, smaller encoder models are still the go to architecture for RE. In this paper, we revisit fine-tuning such smaller models using a novel dual-encoder architecture with a joint contrastive and cross-entropy loss. Unlike previous methods that employ a fixed linear layer for predicate representations, our approach uses a second encoder to compute instance-specific predicate representations by infusing them with real entity spans from corresponding input instances. We conducted experiments on two biomedical RE datasets and two general domain datasets. Our approach achieved F1 score improvements ranging from 1% to 2% over state-of-the-art methods with a simple but elegant formulation. Ablation studies justify the importance of various components built into the proposed architecture.
>
---
#### [replaced 014] Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.00842v2](http://arxiv.org/pdf/2506.00842v2)**

> **作者:** Jiawei Gu; Ziting Xian; Yuanzhen Xie; Ye Liu; Enjie Liu; Ruichao Zhong; Mochi Gao; Yunzhi Tan; Bo Hu; Zang Li
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Large language models (LLMs) achieve strong performance on plain text tasks but underperform on structured data like tables and databases. Potential challenges arise from their underexposure during pre-training and rigid text-to-structure transfer mechanisms. Unlike humans who seamlessly apply learned patterns across data modalities, LLMs struggle to infer implicit relationships embedded in tabular formats, especially in the absence of explicit structural guidance. To bridge this cognitive gap, we introduce Contrastive Retrieval-Augmented Generation on Experience (CoRE), a framework that builds experience memory representations and enhances generalization through contrastive In-Context Learning (ICL) to simulate human-like knowledge transfer. Experiments on Text-to-SQL and TableQA show CoRE significantly improves performance, achieving average gains of 3.44% and 4.24%, with up to 17.2% on challenging tasks. Our Monte Carlo Tree Search (MCTS)-generated Experience Memory expands training data 8-9x, enhancing diversity and domain coverage. This training-free and continual method propels LLMs toward structured knowledge expertise.
>
---
#### [replaced 015] A Fisher's exact test justification of the TF-IDF term-weighting scheme
- **分类: cs.CL; cs.IR; math.ST; stat.TH**

- **链接: [http://arxiv.org/pdf/2507.15742v2](http://arxiv.org/pdf/2507.15742v2)**

> **作者:** Paul Sheridan; Zeyad Ahmed; Aitazaz A. Farooque
>
> **备注:** 23 pages, 4 tables, accepted in The American Statistician 2025
>
> **摘要:** Term frequency-inverse document frequency, or TF-IDF for short, is arguably the most celebrated mathematical expression in the history of information retrieval. Conceived as a simple heuristic quantifying the extent to which a given term's occurrences are concentrated in any one given document out of many, TF-IDF and its many variants are routinely used as term-weighting schemes in diverse text analysis applications. There is a growing body of scholarship dedicated to placing TF-IDF on a sound theoretical foundation. Building on that tradition, this paper justifies the use of TF-IDF to the statistics community by demonstrating how the famed expression can be understood from a significance testing perspective. We show that the common TF-IDF variant TF-ICF is, under mild regularity conditions, closely related to the negative logarithm of the $p$-value from a one-tailed version of Fisher's exact test of statistical significance. As a corollary, we establish a connection between TF-IDF and the said negative log-transformed $p$-value under certain idealized assumptions. We further demonstrate, as a limiting case, that this same quantity converges to TF-IDF in the limit of an infinitely large document collection. The Fisher's exact test justification of TF-IDF equips the working statistician with a ready explanation of the term-weighting scheme's long-established effectiveness.
>
---
#### [replaced 016] Long-Form Answers to Visual Questions from Blind and Low Vision People
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2408.06303v2](http://arxiv.org/pdf/2408.06303v2)**

> **作者:** Mina Huh; Fangyuan Xu; Yi-Hao Peng; Chongyan Chen; Hansika Murugu; Danna Gurari; Eunsol Choi; Amy Pavel
>
> **备注:** COLM 2024 Oral Spotlight
>
> **摘要:** Vision language models can now generate long-form answers to questions about images - long-form visual question answers (LFVQA). We contribute VizWiz-LF, a dataset of long-form answers to visual questions posed by blind and low vision (BLV) users. VizWiz-LF contains 4.2k long-form answers to 600 visual questions, collected from human expert describers and six VQA models. We develop and annotate functional roles of sentences of LFVQA and demonstrate that long-form answers contain information beyond the question answer such as explanations and suggestions. We further conduct automatic and human evaluations with BLV and sighted people to evaluate long-form answers. BLV people perceive both human-written and generated long-form answers to be plausible, but generated answers often hallucinate incorrect visual details, especially for unanswerable visual questions (e.g., blurry or irrelevant images). To reduce hallucinations, we evaluate the ability of VQA models to abstain from answering unanswerable questions across multiple prompting strategies.
>
---
#### [replaced 017] DoctorAgent-RL: A Multi-Agent Collaborative Reinforcement Learning System for Multi-Turn Clinical Dialogue
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19630v2](http://arxiv.org/pdf/2505.19630v2)**

> **作者:** Yichun Feng; Jiawei Wang; Lu Zhou; Zhen Lei; Yixue Li
>
> **摘要:** Large language models (LLMs) have demonstrated excellent capabilities in the field of biomedical question answering, but their application in real-world clinical consultations still faces core challenges. Single-round consultation systems require patients to describe all symptoms upfront, leading to vague diagnosis with unclear complaints. Traditional multi-turn dialogue models, constrained by static supervised learning, lack flexibility and fail to intelligently extract key clinical information. To address these limitations, we propose \Ours{}, a reinforcement learning (RL)-based multi-agent collaborative framework that models medical consultations as a dynamic decision-making process under uncertainty. The doctor agent continuously optimizes its questioning strategy within the RL framework through multi-turn interactions with the patient agent, dynamically adjusting its information-gathering path based on comprehensive rewards from the Consultation Evaluator. This RL fine-tuning mechanism enables LLMs to autonomously develop interaction strategies aligned with clinical reasoning logic, rather than superficially imitating patterns in existing dialogue data. Notably, we constructed MTMedDialog, the first English multi-turn medical consultation dataset capable of simulating patient interactions. Experiments demonstrate that \Ours{} outperforms existing models in both multi-turn reasoning capability and final diagnostic performance. This approach shows immense practical value by reducing misdiagnosis risks in time-pressured settings, freeing clinicians for complex cases, and pioneering a strategy to optimize medical resource allocation and alleviate workforce shortages. Code and data are available at https://github.com/JarvisUSTC/DoctorAgent-RL
>
---
#### [replaced 018] Analyze Feature Flow to Enhance Interpretation and Steering in Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.03032v3](http://arxiv.org/pdf/2502.03032v3)**

> **作者:** Daniil Laptev; Nikita Balagansky; Yaroslav Aksenov; Daniil Gavrilov
>
> **摘要:** We introduce a new approach to systematically map features discovered by sparse autoencoder across consecutive layers of large language models, extending earlier work that examined inter-layer feature links. By using a data-free cosine similarity technique, we trace how specific features persist, transform, or first appear at each stage. This method yields granular flow graphs of feature evolution, enabling fine-grained interpretability and mechanistic insights into model computations. Crucially, we demonstrate how these cross-layer feature maps facilitate direct steering of model behavior by amplifying or suppressing chosen features, achieving targeted thematic control in text generation. Together, our findings highlight the utility of a causal, cross-layer interpretability framework that not only clarifies how features develop through forward passes but also provides new means for transparent manipulation of large language models.
>
---
#### [replaced 019] 3LM: Bridging Arabic, STEM, and Code through Benchmarking
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.15850v3](http://arxiv.org/pdf/2507.15850v3)**

> **作者:** Basma El Amel Boussaha; Leen AlQadi; Mugariya Farooq; Shaikha Alsuwaidi; Giulia Campesan; Ahmed Alzubaidi; Mohammed Alyafeai; Hakim Hacid
>
> **摘要:** Arabic is one of the most widely spoken languages in the world, yet efforts to develop and evaluate Large Language Models (LLMs) for Arabic remain relatively limited. Most existing Arabic benchmarks focus on linguistic, cultural, or religious content, leaving a significant gap in domains like STEM and code which are increasingly relevant for real-world LLM applications. To help bridge this gap, we present 3LM, a suite of three benchmarks designed specifically for Arabic. The first is a set of STEM-related question-answer pairs, naturally sourced from Arabic textbooks and educational worksheets. The second consists of synthetically generated STEM questions, created using the same sources. The third benchmark focuses on code generation, built through a careful translation of two widely used code benchmarks, incorporating a human-in-the-loop process with several rounds of review to ensure high-quality and faithful translations. We release all three benchmarks publicly to support the growth of Arabic LLM research in these essential but underrepresented areas.
>
---
#### [replaced 020] Toward Super Agent System with Hybrid AI Routers
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2504.10519v2](http://arxiv.org/pdf/2504.10519v2)**

> **作者:** Yuhang Yao; Haixin Wang; Yibo Chen; Jiawen Wang; Min Chang Jordan Ren; Bosheng Ding; Salman Avestimehr; Chaoyang He
>
> **摘要:** AI Agents powered by Large Language Models are transforming the world through enormous applications. A super agent has the potential to fulfill diverse user needs, such as summarization, coding, and research, by accurately understanding user intent and leveraging the appropriate tools to solve tasks. However, to make such an agent viable for real-world deployment and accessible at scale, significant optimizations are required to ensure high efficiency and low cost. This position paper presents a design of the Super Agent System powered by the hybrid AI routers. Upon receiving a user prompt, the system first detects the intent of the user, then routes the request to specialized task agents with the necessary tools or automatically generates agentic workflows. In practice, most applications directly serve as AI assistants on edge devices such as phones and robots. As different language models vary in capability and cloud-based models often entail high computational costs, latency, and privacy concerns, we then explore the hybrid mode where the router dynamically selects between local and cloud models based on task complexity. Finally, we introduce the blueprint of an on-device super agent enhanced with cloud. With advances in multi-modality models and edge hardware, we envision that most computations can be handled locally, with cloud collaboration only as needed. Such architecture paves the way for super agents to be seamlessly integrated into everyday life in the near future.
>
---
#### [replaced 021] Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.13618v3](http://arxiv.org/pdf/2507.13618v3)**

> **作者:** Shanbo Cheng; Yu Bao; Qian Cao; Luyang Huang; Liyan Kang; Zhicheng Liu; Yu Lu; Wenhao Zhu; Jingwen Chen; Zhichao Huang; Tao Li; Yifu Li; Huiying Lin; Sitong Liu; Ningxin Peng; Shuaijie She; Lu Xu; Nuo Xu; Sen Yang; Runsheng Yu; Yiming Yu; Liehao Zou; Hang Li; Lu Lu; Yuxuan Wang; Yonghui Wu
>
> **摘要:** Multilingual translation stands as a challenging task for large language models (LLMs) to handle intricate language patterns and stilted translations that arise in automated translations. In this paper, we introduce Seed-X, a family of open-source LLMs comprising instruct and reasoning models, pushing the limits of translation capability with 7B parameter size. The base model is pre-trained on a diverse, high-quality dataset encompassing both monolingual and bilingual content across 28 languages, harnessing the full potential of multilingual data. The instruct model is then finetuned to translate by Chain-of-Thought (CoT) reasoning and further enhanced through reinforcement learning (RL) to achieve better generalization across diverse language pairs. Seed-X achieves performance comparable to leading closed-source models, including Gemini-2.5 and GPT-4o, across 28 languages, and significantly outperforms larger open-source models in both automatic metrics and human evaluations. We share the best practices through our optimization process, and make the parameter public available for advancing translation research and applications.
>
---
#### [replaced 022] AI Flow: Perspectives, Scenarios, and Approaches
- **分类: cs.AI; cs.CL; cs.CV; cs.DC; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.12479v3](http://arxiv.org/pdf/2506.12479v3)**

> **作者:** Hongjun An; Wenhan Hu; Sida Huang; Siqi Huang; Ruanjun Li; Yuanzhi Liang; Jiawei Shao; Yiliang Song; Zihan Wang; Cheng Yuan; Chi Zhang; Hongyuan Zhang; Wenhao Zhuang; Xuelong Li
>
> **备注:** Authors are with Institute of Artificial Intelligence (TeleAI), China Telecom, China. Author names are listed alphabetically by surname. This work was conducted at TeleAI, facilitated by Dr. Jiawei Shao (e-mail: shaojw2@chinatelecom.cn) under the leadership of Prof. Xuelong Li. The corresponding author is Prof. Xuelong Li (e-mail: xuelong li@ieee.org), the CTO and Chief Scientist of China Telecom
>
> **摘要:** Pioneered by the foundational information theory by Claude Shannon and the visionary framework of machine intelligence by Alan Turing, the convergent evolution of information and communication technologies (IT/CT) has created an unbroken wave of connectivity and computation. This synergy has sparked a technological revolution, now reaching its peak with large artificial intelligence (AI) models that are reshaping industries and redefining human-machine collaboration. However, the realization of ubiquitous intelligence faces considerable challenges due to substantial resource consumption in large models and high communication bandwidth demands. To address these challenges, AI Flow has been introduced as a multidisciplinary framework that integrates cutting-edge IT and CT advancements, with a particular emphasis on the following three key points. First, device-edge-cloud framework serves as the foundation, which integrates end devices, edge servers, and cloud clusters to optimize scalability and efficiency for low-latency model inference. Second, we introduce the concept of familial models, which refers to a series of different-sized models with aligned hidden features, enabling effective collaboration and the flexibility to adapt to varying resource constraints and dynamic scenarios. Third, connectivity- and interaction-based intelligence emergence is a novel paradigm of AI Flow. By leveraging communication networks to enhance connectivity, the collaboration among AI models across heterogeneous nodes achieves emergent intelligence that surpasses the capability of any single model. The innovations of AI Flow provide enhanced intelligence, timely responsiveness, and ubiquitous accessibility to AI services, paving the way for the tighter fusion of AI techniques and communication systems.
>
---
#### [replaced 023] An Efficient Sparse Fine-Tuning with Low Quantization Error via Neural Network Pruning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.11439v2](http://arxiv.org/pdf/2502.11439v2)**

> **作者:** Cen-Jhih Li; Aditya Bhaskara
>
> **摘要:** Fine-tuning is an important step in adapting foundation models such as large language models to downstream tasks. To make this step more accessible to users with limited computational budgets, it is crucial to develop fine-tuning methods that are memory and computationally efficient. Sparse Fine-tuning (SpFT) and Low-rank adaptation (LoRA) are two frameworks that have emerged for addressing this problem and have been adopted widely in practice. In this work, we develop a new SpFT framework, based on ideas from neural network pruning. At a high level, we first identify ``important'' neurons/nodes using feature importance metrics from network pruning (specifically, we use the structural pruning method), and then perform fine-tuning by restricting to weights involving these neurons. Experiments on common language tasks show our method improves SpFT's memory efficiency by 20-50\% while matching the accuracy of state-of-the-art methods like LoRA's variants.
>
---
#### [replaced 024] XAI4LLM. Let Machine Learning Models and LLMs Collaborate for Enhanced In-Context Learning in Healthcare
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2405.06270v4](http://arxiv.org/pdf/2405.06270v4)**

> **作者:** Fatemeh Nazary; Yashar Deldjoo; Tommaso Di Noia; Eugenio di Sciascio
>
> **摘要:** Clinical decision support systems require models that are not only highly accurate but also equitable and sensitive to the implications of missed diagnoses. In this study, we introduce a knowledge-guided in-context learning (ICL) framework designed to enable large language models (LLMs) to effectively process structured clinical data. Our approach integrates domain-specific feature groupings, carefully balanced few-shot examples, and task-specific prompting strategies. We systematically evaluate this method across seventy distinct ICL designs by various prompt variations and two different communication styles-natural-language narrative and numeric conversational-and compare its performance to robust classical machine learning (ML) benchmarks on tasks involving heart disease and diabetes prediction. Our findings indicate that while traditional ML models maintain superior performance in balanced precision-recall scenarios, LLMs employing narrative prompts with integrated domain knowledge achieve higher recall and significantly reduce gender bias, effectively narrowing fairness disparities by an order of magnitude. Despite the current limitation of increased inference latency, LLMs provide notable advantages, including the capacity for zero-shot deployment and enhanced equity. This research offers the first comprehensive analysis of ICL design considerations for applying LLMs to tabular clinical tasks and highlights distillation and multimodal extensions as promising directions for future research.
>
---
#### [replaced 025] Natural Language Processing for Tigrinya: Current State and Future Directions
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2507.17974v2](http://arxiv.org/pdf/2507.17974v2)**

> **作者:** Fitsum Gaim; Jong C. Park
>
> **摘要:** Despite being spoken by millions of people, Tigrinya remains severely underrepresented in Natural Language Processing (NLP) research. This work presents a comprehensive survey of NLP research for Tigrinya, analyzing over 40 studies spanning more than a decade of work from 2011 to 2025. We systematically review the current state of computational resources, models, and applications across ten distinct downstream tasks, including morphological processing, machine translation, speech recognition, and question-answering. Our analysis reveals a clear trajectory from foundational, rule-based systems to modern neural architectures, with progress consistently unlocked by resource creation milestones. We identify key challenges rooted in Tigrinya's morphological complexity and resource scarcity, while highlighting promising research directions, including morphology-aware modeling, cross-lingual transfer, and community-centered resource development. This work serves as both a comprehensive reference for researchers and a roadmap for advancing Tigrinya NLP. A curated metadata of the surveyed studies and resources is made publicly available.
>
---
#### [replaced 026] Can LLMs Predict Citation Intent? An Experimental Analysis of In-context Learning and Fine-tuning on Open LLMs
- **分类: cs.CL; cs.DL**

- **链接: [http://arxiv.org/pdf/2502.14561v3](http://arxiv.org/pdf/2502.14561v3)**

> **作者:** Paris Koloveas; Serafeim Chatzopoulos; Thanasis Vergoulis; Christos Tryfonopoulos
>
> **备注:** Accepted for publication on TPDL 2025
>
> **摘要:** This work investigates the ability of open Large Language Models (LLMs) to predict citation intent through in-context learning and fine-tuning. Unlike traditional approaches relying on domain-specific pre-trained models like SciBERT, we demonstrate that general-purpose LLMs can be adapted to this task with minimal task-specific data. We evaluate twelve model variations across five prominent open LLM families using zero-, one-, few-, and many-shot prompting. Our experimental study identifies the top-performing model and prompting parameters through extensive in-context learning experiments. We then demonstrate the significant impact of task-specific adaptation by fine-tuning this model, achieving a relative F1-score improvement of 8% on the SciCite dataset and 4.3% on the ACL-ARC dataset compared to the instruction-tuned baseline. These findings provide valuable insights for model selection and prompt engineering. Additionally, we make our end-to-end evaluation framework and models openly available for future use.
>
---
#### [replaced 027] Comparison of pipeline, sequence-to-sequence, and GPT models for end-to-end relation extraction: experiments with the rare disease use-case
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2311.13729v3](http://arxiv.org/pdf/2311.13729v3)**

> **作者:** Shashank Gupta; Xuguang Ai; Ramakanth Kavuluru
>
> **备注:** An updated version of this paper has appeared in the proceedings of NLDB 2025 with a different title. The corresonding DOI is in the metadata provided below
>
> **摘要:** End-to-end relation extraction (E2ERE) is an important and realistic application of natural language processing (NLP) in biomedicine. In this paper, we aim to compare three prevailing paradigms for E2ERE using a complex dataset focused on rare diseases involving discontinuous and nested entities. We use the RareDis information extraction dataset to evaluate three competing approaches (for E2ERE): NER $\rightarrow$ RE pipelines, joint sequence to sequence models, and generative pre-trained transformer (GPT) models. We use comparable state-of-the-art models and best practices for each of these approaches and conduct error analyses to assess their failure modes. Our findings reveal that pipeline models are still the best, while sequence-to-sequence models are not far behind; GPT models with eight times as many parameters are worse than even sequence-to-sequence models and lose to pipeline models by over 10 F1 points. Partial matches and discontinuous entities caused many NER errors contributing to lower overall E2E performances. We also verify these findings on a second E2ERE dataset for chemical-protein interactions. Although generative LM-based methods are more suitable for zero-shot settings, when training data is available, our results show that it is better to work with more conventional models trained and tailored for E2ERE. More innovative methods are needed to marry the best of the both worlds from smaller encoder-decoder pipeline models and the larger GPT models to improve E2ERE. As of now, we see that well designed pipeline models offer substantial performance gains at a lower cost and carbon footprint for E2ERE. Our contribution is also the first to conduct E2ERE for the RareDis dataset.
>
---
#### [replaced 028] T2ISafety: Benchmark for Assessing Fairness, Toxicity, and Privacy in Image Generation
- **分类: cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2501.12612v3](http://arxiv.org/pdf/2501.12612v3)**

> **作者:** Lijun Li; Zhelun Shi; Xuhao Hu; Bowen Dong; Yiran Qin; Xihui Liu; Lu Sheng; Jing Shao
>
> **备注:** Accepted at CVPR 2025
>
> **摘要:** Text-to-image (T2I) models have rapidly advanced, enabling the generation of high-quality images from text prompts across various domains. However, these models present notable safety concerns, including the risk of generating harmful, biased, or private content. Current research on assessing T2I safety remains in its early stages. While some efforts have been made to evaluate models on specific safety dimensions, many critical risks remain unexplored. To address this gap, we introduce T2ISafety, a safety benchmark that evaluates T2I models across three key domains: toxicity, fairness, and bias. We build a detailed hierarchy of 12 tasks and 44 categories based on these three domains, and meticulously collect 70K corresponding prompts. Based on this taxonomy and prompt set, we build a large-scale T2I dataset with 68K manually annotated images and train an evaluator capable of detecting critical risks that previous work has failed to identify, including risks that even ultra-large proprietary models like GPTs cannot correctly detect. We evaluate 12 prominent diffusion models on T2ISafety and reveal several concerns including persistent issues with racial fairness, a tendency to generate toxic content, and significant variation in privacy protection across the models, even with defense methods like concept erasing. Data and evaluator are released under https://github.com/adwardlee/t2i_safety.
>
---
#### [replaced 029] How Important is Domain Specificity in Language Models and Instruction Finetuning for Biomedical Relation Extraction?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2402.13470v2](http://arxiv.org/pdf/2402.13470v2)**

> **作者:** Aviv Brokman; Ramakanth Kavuluru
>
> **备注:** A version of this paper has appeared in the proceedings of NLDB 2025 with a slightly different title. The corresponding DOI is also listed below in the metadata
>
> **摘要:** Cutting edge techniques developed in the general NLP domain are often subsequently applied to the high-value, data-rich biomedical domain. The past few years have seen generative language models (LMs), instruction finetuning, and few-shot learning become foci of NLP research. As such, generative LMs pretrained on biomedical corpora have proliferated and biomedical instruction finetuning has been attempted as well, all with the hope that domain specificity improves performance on downstream tasks. Given the nontrivial effort in training such models, we investigate what, if any, benefits they have in the key biomedical NLP task of relation extraction. Specifically, we address two questions: (1) Do LMs trained on biomedical corpora outperform those trained on general domain corpora? (2) Do models instruction finetuned on biomedical datasets outperform those finetuned on assorted datasets or those simply pretrained? We tackle these questions using existing LMs, testing across four datasets. In a surprising result, general-domain models typically outperformed biomedical-domain models. However, biomedical instruction finetuning improved performance to a similar degree as general instruction finetuning, despite having orders of magnitude fewer instructions. Our findings suggest it may be more fruitful to focus research effort on larger-scale biomedical instruction finetuning of general LMs over building domain-specific biomedical LMs
>
---
#### [replaced 030] Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.IT; cs.LG; cs.NE; math.IT**

- **链接: [http://arxiv.org/pdf/2506.19037v3](http://arxiv.org/pdf/2506.19037v3)**

> **作者:** Omer Luxembourg; Haim Permuter; Eliya Nachmani
>
> **摘要:** Masked diffusion language models (MDLMs) promise fast, non-autoregressive text generation, yet existing samplers, which pick tokens to unmask based on model confidence, ignore interactions when unmasking multiple positions in parallel and effectively reduce to slow, autoregressive behavior. We propose the Dilated Unmasking Scheduler (DUS), an inference-only, planner-model-free method that partitions sequence positions into non-adjacent dilated groups and unmasked them in parallel so as to minimize an upper bound on joint entropy gain at each denoising step. By explicitly trading off the number of network calls against generation quality, DUS recovers most of the performance lost under traditional parallel unmasking strategies. Across math (GSM8K, MATH500), code (HumanEval, MBPP) and general-knowledge benchmarks (BBH, MMLU-Pro), DUS outperforms confidence-based planners, without modifying the underlying denoiser, and reveals the true speed-quality frontier of MDLMs.
>
---
#### [replaced 031] Distillation Scaling Laws
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2502.08606v2](http://arxiv.org/pdf/2502.08606v2)**

> **作者:** Dan Busbridge; Amitis Shidani; Floris Weers; Jason Ramapuram; Etai Littwin; Russ Webb
>
> **备注:** Version accepted to ICML 2025. 69 pages, 54 figures, 13 tables
>
> **摘要:** We propose a distillation scaling law that estimates distilled model performance based on a compute budget and its allocation between the student and teacher. Our findings mitigate the risks associated with large-scale distillation by enabling compute-optimal allocation for both the teacher and student to maximize student performance. We provide compute-optimal distillation recipes for two key scenarios: when a teacher already exists, and when a teacher needs training. In settings involving many students or an existing teacher, distillation outperforms supervised learning up to a compute level that scales predictably with student size. Conversely, if only one student is to be distilled and a teacher also requires training, supervised learning is generally preferable. Additionally, our large-scale study of distillation increases our understanding of the process and helps inform experimental design.
>
---
#### [replaced 032] Scalpel vs. Hammer: GRPO Amplifies Existing Capabilities, SFT Replaces Them
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.10616v2](http://arxiv.org/pdf/2507.10616v2)**

> **作者:** Neel Rajani; Aryo Pradipta Gema; Seraphina Goldfarb-Tarrant; Ivan Titov
>
> **摘要:** Training large language models (LLMs) for reasoning via maths and code datasets has become a major new focus in LLM post-training. Two particularly popular approaches are reinforcement learning (RL) and supervised fine-tuning (SFT), but their training dynamics are poorly understood. We present a comparative analysis of RL and SFT on the same maths problems with the same model and similar hyperparameters. We find that RL yields minor in-domain gains on maths and slight degradation on knowledge-intensive benchmarks like MMLU, while both trends are more pronounced in SFT. We also analyse model parameters across checkpoints, observing that both algorithms modify query and key weights the most. Meanwhile, SFT exhibits greater updates and also affects mid-layer MLPs more, leading us to hypothesise that this may have caused the out-of-domain degradation. We therefore investigate whether freezing parts of the model during training can mitigate the reduced performance on knowledge-intensive benchmarks. However, our results are inconclusive, with benefits on GPQA:Diamond and degradation on other benchmarks. Taken together, our observations provide a preliminary indication for why RL amplifies existing capabilities, while SFT replaces old skills with new ones.
>
---
#### [replaced 033] LLMs are Also Effective Embedding Models: An In-depth Overview
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.12591v2](http://arxiv.org/pdf/2412.12591v2)**

> **作者:** Chongyang Tao; Tao Shen; Shen Gao; Junshuo Zhang; Zhen Li; Kai Hua; Wenpeng Hu; Zhengwei Tao; Shuai Ma
>
> **备注:** 38 pages
>
> **摘要:** Large language models (LLMs) have revolutionized natural language processing by achieving state-of-the-art performance across various tasks. Recently, their effectiveness as embedding models has gained attention, marking a paradigm shift from traditional encoder-only models like ELMo and BERT to decoder-only, large-scale LLMs such as GPT, LLaMA, and Mistral. This survey provides an in-depth overview of this transition, beginning with foundational techniques before the LLM era, followed by LLM-based embedding models through two main strategies to derive embeddings from LLMs. 1) Direct prompting: We mainly discuss the prompt designs and the underlying rationale for deriving competitive embeddings. 2) Data-centric tuning: We cover extensive aspects that affect tuning an embedding model, including model architecture, training objectives, data constructions, etc. Upon the above, we also cover advanced methods for producing embeddings from longer texts, multilingual, code, cross-modal data, as well as reasoning-aware and other domain-specific scenarios. Furthermore, we discuss factors affecting choices of embedding models, such as performance/efficiency comparisons, dense vs sparse embeddings, pooling strategies, and scaling law. Lastly, the survey highlights the limitations and challenges in adapting LLMs for embeddings, including cross-task embedding quality, trade-offs between efficiency and accuracy, low-resource, long-context, data bias, robustness, etc. This survey serves as a valuable resource for researchers and practitioners by synthesizing current advancements, highlighting key challenges, and offering a comprehensive framework for future work aimed at enhancing the effectiveness and efficiency of LLMs as embedding models.
>
---
#### [replaced 034] Advancing biomolecular understanding and design following human instructions
- **分类: cs.CL; q-bio.BM**

- **链接: [http://arxiv.org/pdf/2410.07919v2](http://arxiv.org/pdf/2410.07919v2)**

> **作者:** Xiang Zhuang; Keyan Ding; Tianwen Lyu; Yinuo Jiang; Xiaotong Li; Zhuoyi Xiang; Zeyuan Wang; Ming Qin; Kehua Feng; Jike Wang; Qiang Zhang; Huajun Chen
>
> **摘要:** Understanding and designing biomolecules, such as proteins and small molecules, is central to advancing drug discovery, synthetic biology and enzyme engineering. Recent breakthroughs in artificial intelligence have revolutionized biomolecular research, achieving remarkable accuracy in biomolecular prediction and design. However, a critical gap remains between artificial intelligence's computational capabilities and researchers' intuitive goals, particularly in using natural language to bridge complex tasks with human intentions. Large language models have shown potential to interpret human intentions, yet their application to biomolecular research remains nascent due to challenges including specialized knowledge requirements, multimodal data integration, and semantic alignment between natural language and biomolecules. To address these limitations, we present InstructBioMol, a large language model designed to bridge natural language and biomolecules through a comprehensive any-to-any alignment of natural language, molecules and proteins. This model can integrate multimodal biomolecules as the input, and enable researchers to articulate design goals in natural language, providing biomolecular outputs that meet precise biological needs. Experimental results demonstrate that InstructBioMol can understand and design biomolecules following human instructions. In particular, it can generate drug molecules with a 10% improvement in binding affinity and design enzymes that achieve an enzyme-substrate pair prediction score of 70.4. This highlights its potential to transform real-world biomolecular research. The code is available at https://github.com/HICAI-ZJU/InstructBioMol.
>
---
#### [replaced 035] ToolACE: Winning the Points of LLM Function Calling
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.00920v2](http://arxiv.org/pdf/2409.00920v2)**

> **作者:** Weiwen Liu; Xu Huang; Xingshan Zeng; Xinlong Hao; Shuai Yu; Dexun Li; Shuai Wang; Weinan Gan; Zhengying Liu; Yuanqing Yu; Zezhong Wang; Yuxian Wang; Wu Ning; Yutai Hou; Bin Wang; Chuhan Wu; Xinzhi Wang; Yong Liu; Yasheng Wang; Duyu Tang; Dandan Tu; Lifeng Shang; Xin Jiang; Ruiming Tang; Defu Lian; Qun Liu; Enhong Chen
>
> **备注:** 21 pages, 22 figures
>
> **摘要:** Function calling significantly extends the application boundary of large language models, where high-quality and diverse training data is critical for unlocking this capability. However, real function-calling data is quite challenging to collect and annotate, while synthetic data generated by existing pipelines tends to lack coverage and accuracy. In this paper, we present ToolACE, an automatic agentic pipeline designed to generate accurate, complex, and diverse tool-learning data. ToolACE leverages a novel self-evolution synthesis process to curate a comprehensive API pool of 26,507 diverse APIs. Dialogs are further generated through the interplay among multiple agents, guided by a formalized thinking process. To ensure data accuracy, we implement a dual-layer verification system combining rule-based and model-based checks. We demonstrate that models trained on our synthesized data, even with only 8B parameters, achieve state-of-the-art performance on the Berkeley Function-Calling Leaderboard, rivaling the latest GPT-4 models. Our model and a subset of the data are publicly available at https://huggingface.co/Team-ACE.
>
---
#### [replaced 036] A Comprehensive Evaluation of Semantic Relation Knowledge of Pretrained Language Models and Humans
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.01131v4](http://arxiv.org/pdf/2412.01131v4)**

> **作者:** Zhihan Cao; Hiroaki Yamada; Simone Teufel; Takenobu Tokunaga
>
> **摘要:** Recently, much work has concerned itself with the enigma of what exactly pretrained language models~(PLMs) learn about different aspects of language, and how they learn it. One stream of this type of research investigates the knowledge that PLMs have about semantic relations. However, many aspects of semantic relations were left unexplored. Generally, only one relation has been considered, namely hypernymy. Furthermore, previous work did not measure humans' performance on the same task as that performed by the PLMs. This means that at this point in time, there is only an incomplete view of the extent of these models' semantic relation knowledge. To address this gap, we introduce a comprehensive evaluation framework covering five relations beyond hypernymy, namely hyponymy, holonymy, meronymy, antonymy, and synonymy. We use five metrics (two newly introduced here) for recently untreated aspects of semantic relation knowledge, namely soundness, completeness, symmetry, prototypicality, and distinguishability. Using these, we can fairly compare humans and models on the same task. Our extensive experiments involve six PLMs, four masked and two causal language models. The results reveal a significant knowledge gap between humans and models for all semantic relations. In general, causal language models, despite their wide use, do not always perform significantly better than masked language models. Antonymy is the outlier relation where all models perform reasonably well.
>
---
#### [replaced 037] Distilling the Implicit Multi-Branch Structure in LLMs' Reasoning via Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16142v3](http://arxiv.org/pdf/2505.16142v3)**

> **作者:** Shicheng Xu; Liang Pang; Yunchang Zhu; Jia Gu; Zihao Wei; Jingcheng Deng; Feiyang Pan; Huawei Shen; Xueqi Cheng
>
> **备注:** 15 pages
>
> **摘要:** Distilling reasoning paths from teacher to student models via supervised fine-tuning (SFT) provides a shortcut for improving the reasoning ability of smaller Large Language Models (LLMs). However, the reasoning paths generated by teacher models often reflect only surface-level traces of their underlying authentic reasoning. Insights from cognitive neuroscience suggest that authentic reasoning involves a complex interweaving between meta-reasoning (which selects appropriate sub-problems from multiple candidates) and solving (which addresses the sub-problem). This implies authentic reasoning has an implicit multi-branch structure. Supervised fine-tuning collapses this rich structure into a flat sequence of token prediction in the teacher's reasoning path, preventing effective distillation of this structure to students. To address this limitation, we propose RLKD, a reinforcement learning (RL)-based distillation framework guided by a novel Generative Structure Reward Model (GSRM). Our GSRM converts reasoning paths into multiple meta-reasoning-solving steps and computes rewards to measure structural alignment between student and teacher reasoning. RLKD combines this reward with RL, enabling student LLMs to internalize the teacher's implicit multi-branch reasoning structure rather than merely mimicking fixed output paths. Experiments show RLKD surpasses standard SFT-RL pipelines even when trained on 0.1% of data under an RL-only regime, unlocking greater student reasoning potential than SFT-based distillation.
>
---
#### [replaced 038] HIVMedQA: Benchmarking large language models for HIV medical decision support
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.18143v2](http://arxiv.org/pdf/2507.18143v2)**

> **作者:** Gonzalo Cardenal-Antolin; Jacques Fellay; Bashkim Jaha; Roger Kouyos; Niko Beerenwinkel; Diane Duroux
>
> **摘要:** Large language models (LLMs) are emerging as valuable tools to support clinicians in routine decision-making. HIV management is a compelling use case due to its complexity, including diverse treatment options, comorbidities, and adherence challenges. However, integrating LLMs into clinical practice raises concerns about accuracy, potential harm, and clinician acceptance. Despite their promise, AI applications in HIV care remain underexplored, and LLM benchmarking studies are scarce. This study evaluates the current capabilities of LLMs in HIV management, highlighting their strengths and limitations. We introduce HIVMedQA, a benchmark designed to assess open-ended medical question answering in HIV care. The dataset consists of curated, clinically relevant questions developed with input from an infectious disease physician. We evaluated seven general-purpose and three medically specialized LLMs, applying prompt engineering to enhance performance. Our evaluation framework incorporates both lexical similarity and an LLM-as-a-judge approach, extended to better reflect clinical relevance. We assessed performance across key dimensions: question comprehension, reasoning, knowledge recall, bias, potential harm, and factual accuracy. Results show that Gemini 2.5 Pro consistently outperformed other models across most dimensions. Notably, two of the top three models were proprietary. Performance declined as question complexity increased. Medically fine-tuned models did not always outperform general-purpose ones, and larger model size was not a reliable predictor of performance. Reasoning and comprehension were more challenging than factual recall, and cognitive biases such as recency and status quo were observed. These findings underscore the need for targeted development and evaluation to ensure safe, effective LLM integration in clinical care.
>
---
#### [replaced 039] Verbalized Representation Learning for Interpretable Few-Shot Generalization
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.18651v2](http://arxiv.org/pdf/2411.18651v2)**

> **作者:** Cheng-Fu Yang; Da Yin; Wenbo Hu; Nanyun Peng; Bolei Zhou; Kai-Wei Chang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Humans recognize objects after observing only a few examples, a remarkable capability enabled by their inherent language understanding of the real-world environment. Developing verbalized and interpretable representation can significantly improve model generalization in low-data settings. In this work, we propose Verbalized Representation Learning (VRL), a novel approach for automatically extracting human-interpretable features for object recognition using few-shot data. Our method uniquely captures inter-class differences and intra-class commonalities in the form of natural language by employing a Vision-Language Model (VLM) to identify key discriminative features between different classes and shared characteristics within the same class. These verbalized features are then mapped to numeric vectors through the VLM. The resulting feature vectors can be further utilized to train and infer with downstream classifiers. Experimental results show that, at the same model scale, VRL achieves a 24% absolute improvement over prior state-of-the-art methods while using 95% less data and a smaller mode. Furthermore, compared to human-labeled attributes, the features learned by VRL exhibit a 20% absolute gain when used for downstream classification tasks. Code is available at: https://github.com/joeyy5588/VRL/tree/main.
>
---
#### [replaced 040] Kill two birds with one stone: generalized and robust AI-generated text detection via dynamic perturbations
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.21019v2](http://arxiv.org/pdf/2504.21019v2)**

> **作者:** Yinghan Zhou; Juan Wen; Wanli Peng; Yiming Xue; Ziwei Zhang; Zhengxian Wu
>
> **备注:** Accepted by NAACL 2025 main conference
>
> **摘要:** The growing popularity of large language models has raised concerns regarding the potential to misuse AI-generated text (AIGT). It becomes increasingly critical to establish an excellent AIGT detection method with high generalization and robustness. However, existing methods either focus on model generalization or concentrate on robustness. The unified mechanism, to simultaneously address the challenges of generalization and robustness, is less explored. In this paper, we argue that robustness can be view as a specific form of domain shift, and empirically reveal an intrinsic mechanism for model generalization of AIGT detection task. Then, we proposed a novel AIGT detection method (DP-Net) via dynamic perturbations introduced by a reinforcement learning with elaborated reward and action. Experimentally, extensive results show that the proposed DP-Net significantly outperforms some state-of-the-art AIGT detection methods for generalization capacity in three cross-domain scenarios. Meanwhile, the DP-Net achieves best robustness under two text adversarial attacks. The code is publicly available at https://github.com/CAU-ISS-Lab/AIGT-Detection-Evade-Detection/tree/main/DP-Net.
>
---
#### [replaced 041] MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.12549v2](http://arxiv.org/pdf/2406.12549v2)**

> **作者:** Dominik Macko; Jakub Kopal; Robert Moro; Ivan Srba
>
> **备注:** ACL 2025 main
>
> **摘要:** Recent LLMs are able to generate high-quality multilingual texts, indistinguishable for humans from authentic human-written ones. Research in machine-generated text detection is however mostly focused on the English language and longer texts, such as news articles, scientific papers or student essays. Social-media texts are usually much shorter and often feature informal language, grammatical errors, or distinct linguistic items (e.g., emoticons, hashtags). There is a gap in studying the ability of existing methods in detection of such texts, reflected also in the lack of existing multilingual benchmark datasets. To fill this gap we propose the first multilingual (22 languages) and multi-platform (5 social media platforms) dataset for benchmarking machine-generated text detection in the social-media domain, called MultiSocial. It contains 472,097 texts, of which about 58k are human-written and approximately the same amount is generated by each of 7 multilingual LLMs. We use this benchmark to compare existing detection methods in zero-shot as well as fine-tuned form. Our results indicate that the fine-tuned detectors have no problem to be trained on social-media texts and that the platform selection for training matters.
>
---
#### [replaced 042] An Investigation of Prompt Variations for Zero-shot LLM-based Rankers
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.14117v4](http://arxiv.org/pdf/2406.14117v4)**

> **作者:** Shuoqi Sun; Shengyao Zhuang; Shuai Wang; Guido Zuccon
>
> **备注:** Accepted for publication at the 47th European Conference on Information Retrieval (ECIR 2025)
>
> **摘要:** We provide a systematic understanding of the impact of specific components and wordings used in prompts on the effectiveness of rankers based on zero-shot Large Language Models (LLMs). Several zero-shot ranking methods based on LLMs have recently been proposed. Among many aspects, methods differ across (1) the ranking algorithm they implement, e.g., pointwise vs. listwise, (2) the backbone LLMs used, e.g., GPT3.5 vs. FLAN-T5, (3) the components and wording used in prompts, e.g., the use or not of role-definition (role-playing) and the actual words used to express this. It is currently unclear whether performance differences are due to the underlying ranking algorithm, or because of spurious factors such as better choice of words used in prompts. This confusion risks to undermine future research. Through our large-scale experimentation and analysis, we find that ranking algorithms do contribute to differences between methods for zero-shot LLM ranking. However, so do the LLM backbones -- but even more importantly, the choice of prompt components and wordings affect the ranking. In fact, in our experiments, we find that, at times, these latter elements have more impact on the ranker's effectiveness than the actual ranking algorithms, and that differences among ranking methods become more blurred when prompt variations are considered.
>
---
#### [replaced 043] RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale
- **分类: cs.CL; cs.AI; cs.LG; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.03005v3](http://arxiv.org/pdf/2505.03005v3)**

> **作者:** Daniel Goldstein; Eric Alcaide; Janna Lu; Eugene Cheah
>
> **摘要:** We present Rapid Attention Distillation to Linear Attention Decoders at Scale (RADLADS), a protocol for rapidly converting softmax attention transformers into linear attention decoder models, along with two new RWKV-variant architectures, and models converted from popular Qwen2.5 open source models in 7B, 32B, and 72B sizes. Our conversion process requires only 350-700M tokens, less than 0.005% of the token count used to train the original teacher models. Converting to our 72B linear attention model costs less than \$2,000 USD at today's prices, yet quality at inference remains close to the original transformer. These models achieve state-of-the-art downstream performance across a set of standard benchmarks for linear attention models of their size. We release all our models on HuggingFace under the Apache 2.0 license, with the exception of our 72B models which are also governed by the Qwen License Agreement. Models at https://huggingface.co/collections/recursal/radlads-6818ee69e99e729ba8a87102 Training Code at https://github.com/recursal/RADLADS-paper
>
---
#### [replaced 044] Spike No More: Stabilizing the Pre-training of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2312.16903v4](http://arxiv.org/pdf/2312.16903v4)**

> **作者:** Sho Takase; Shun Kiyono; Sosuke Kobayashi; Jun Suzuki
>
> **备注:** COLM 2025
>
> **摘要:** Loss spikes often occur during pre-training of large language models. The spikes degrade the performance of large language models and sometimes ruin the pre-training. Since the pre-training needs a vast computational budget, we should avoid such spikes. Based on the assumption that the loss spike is caused by the sudden growth of the gradient norm, we explore factors to keep the gradient norm small through an analysis of the spectral norms of the Jacobian matrices for the sub-layers. Our findings suggest that stabilizing the pre-training process requires two conditions: small sub-layers and large shortcut. We conduct various experiments to empirically verify our theoretical analyses. Experimental results demonstrate that methods satisfying the conditions effectively prevent loss spikes during pre-training.
>
---
#### [replaced 045] Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2412.13666v2](http://arxiv.org/pdf/2412.13666v2)**

> **作者:** Aneta Zugecova; Dominik Macko; Ivan Srba; Robert Moro; Jakub Kopal; Katarina Marcincinova; Matus Mesarcik
>
> **备注:** ACL 2025 main
>
> **摘要:** The capabilities of recent large language models (LLMs) to generate high-quality content indistinguishable by humans from human-written texts raises many concerns regarding their misuse. Previous research has shown that LLMs can be effectively misused for generating disinformation news articles following predefined narratives. Their capabilities to generate personalized (in various aspects) content have also been evaluated and mostly found usable. However, a combination of personalization and disinformation abilities of LLMs has not been comprehensively studied yet. Such a dangerous combination should trigger integrated safety filters of the LLMs, if there are some. This study fills this gap by evaluating vulnerabilities of recent open and closed LLMs, and their willingness to generate personalized disinformation news articles in English. We further explore whether the LLMs can reliably meta-evaluate the personalization quality and whether the personalization affects the generated-texts detectability. Our results demonstrate the need for stronger safety-filters and disclaimers, as those are not properly functioning in most of the evaluated LLMs. Additionally, our study revealed that the personalization actually reduces the safety-filter activations; thus effectively functioning as a jailbreak. Such behavior must be urgently addressed by LLM developers and service providers.
>
---
#### [replaced 046] SALM-Duplex: Efficient and Direct Duplex Modeling for Speech-to-Speech Language Model
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.15670v4](http://arxiv.org/pdf/2505.15670v4)**

> **作者:** Ke Hu; Ehsan Hosseini-Asl; Chen Chen; Edresson Casanova; Subhankar Ghosh; Piotr Żelasko; Zhehuai Chen; Jason Li; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Spoken dialogue is an intuitive form of human-computer interaction, yet current speech language models often remain constrained to turn-based exchanges, lacking real-time adaptability such as user barge-in. We propose a novel duplex speech to speech (S2S) architecture featuring continuous user inputs and codec agent outputs with channel fusion that directly models simultaneous user and agent streams. Using a pretrained streaming encoder for user input enables the first duplex S2S model without requiring speech pretrain. Separate architectures for agent and user modeling facilitate codec fine-tuning for better agent voices and halve the bitrate (0.6 kbps) compared to previous works. Experimental results show that the proposed model outperforms previous duplex models in reasoning, turn-taking, and barge-in abilities. The model requires significantly less speech data, as speech pretrain is skipped, which markedly simplifies the process of building a duplex S2S model from any LLMs. Finally, it is the first openly available duplex S2S model with training and inference code to foster reproducibility.
>
---
