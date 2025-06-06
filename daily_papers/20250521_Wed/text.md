# 自然语言处理 cs.CL

- **最新发布 196 篇**

- **更新 126 篇**

## 最新发布

#### [new 001] Source framing triggers systematic evaluation bias in Large Language Models
- **分类: cs.CL; cs.CY**

- **简介: 该研究通过让4种LLM评估不同来源（LLM或特定国籍人类）的文本，发现标注"中国来源"会显著降低模型评价一致性，揭示来源框架对LLM评估的系统性偏见，挑战其公正性。任务为检测LLM评价偏见，解决其受来源影响的公平性问题。**

- **链接: [http://arxiv.org/pdf/2505.13488v1](http://arxiv.org/pdf/2505.13488v1)**

> **作者:** Federico Germani; Giovanni Spitale
>
> **摘要:** Large Language Models (LLMs) are increasingly used not only to generate text but also to evaluate it, raising urgent questions about whether their judgments are consistent, unbiased, and robust to framing effects. In this study, we systematically examine inter- and intra-model agreement across four state-of-the-art LLMs (OpenAI o3-mini, Deepseek Reasoner, xAI Grok 2, and Mistral) tasked with evaluating 4,800 narrative statements on 24 different topics of social, political, and public health relevance, for a total of 192,000 assessments. We manipulate the disclosed source of each statement to assess how attribution to either another LLM or a human author of specified nationality affects evaluation outcomes. We find that, in the blind condition, different LLMs display a remarkably high degree of inter- and intra-model agreement across topics. However, this alignment breaks down when source framing is introduced. Here we show that attributing statements to Chinese individuals systematically lowers agreement scores across all models, and in particular for Deepseek Reasoner. Our findings reveal that framing effects can deeply affect text evaluation, with significant implications for the integrity, neutrality, and fairness of LLM-mediated information systems.
>
---
#### [new 002] Induction Head Toxicity Mechanistically Explains Repetition Curse in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型重复诅咒的机制，发现诱导头（执行上下文学习的注意力头）的"毒性"——其在生成时过度主导输出logits、抑制其他头——是主因，提出注意力头正则化方法以减少重复，促进多样化输出。**

- **链接: [http://arxiv.org/pdf/2505.13514v1](http://arxiv.org/pdf/2505.13514v1)**

> **作者:** Shuxun Wang; Qingyu Yin; Chak Tou Leong; Qiang Zhang; Linyi Yang
>
> **摘要:** Repetition curse is a phenomenon where Large Language Models (LLMs) generate repetitive sequences of tokens or cyclic sequences. While the repetition curse has been widely observed, its underlying mechanisms remain poorly understood. In this work, we investigate the role of induction heads--a specific type of attention head known for their ability to perform in-context learning--in driving this repetitive behavior. Specifically, we focus on the "toxicity" of induction heads, which we define as their tendency to dominate the model's output logits during repetition, effectively excluding other attention heads from contributing to the generation process. Our findings have important implications for the design and training of LLMs. By identifying induction heads as a key driver of the repetition curse, we provide a mechanistic explanation for this phenomenon and suggest potential avenues for mitigation. We also propose a technique with attention head regularization that could be employed to reduce the dominance of induction heads during generation, thereby promoting more diverse and coherent outputs.
>
---
#### [new 003] Gender Trouble in Language Models: An Empirical Audit Guided by Gender Performativity Theory
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文属于AI伦理研究，旨在解决语言模型中性别偏见的深层问题。通过性别表观理论，测试16种模型发现其将性别固化为二元生物概念，导致跨性别者身份被边缘化。研究呼吁重新定义性别偏见的评估框架。**

- **链接: [http://arxiv.org/pdf/2505.14080v1](http://arxiv.org/pdf/2505.14080v1)**

> **作者:** Franziska Sofia Hafner; Ana Valdivia; Luc Rocher
>
> **摘要:** Language models encode and subsequently perpetuate harmful gendered stereotypes. Research has succeeded in mitigating some of these harms, e.g. by dissociating non-gendered terms such as occupations from gendered terms such as 'woman' and 'man'. This approach, however, remains superficial given that associations are only one form of prejudice through which gendered harms arise. Critical scholarship on gender, such as gender performativity theory, emphasizes how harms often arise from the construction of gender itself, such as conflating gender with biological sex. In language models, these issues could lead to the erasure of transgender and gender diverse identities and cause harms in downstream applications, from misgendering users to misdiagnosing patients based on wrong assumptions about their anatomy. For FAccT research on gendered harms to go beyond superficial linguistic associations, we advocate for a broader definition of 'gender bias' in language models. We operationalize insights on the construction of gender through language from gender studies literature and then empirically test how 16 language models of different architectures, training datasets, and model sizes encode gender. We find that language models tend to encode gender as a binary category tied to biological sex, and that gendered terms that do not neatly fall into one of these binary categories are erased and pathologized. Finally, we show that larger models, which achieve better results on performance benchmarks, learn stronger associations between gender and sex, further reinforcing a narrow understanding of gender. Our findings lead us to call for a re-evaluation of how gendered harms in language models are defined and addressed.
>
---
#### [new 004] Temporal Alignment of Time Sensitive Facts with Activation Engineering
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究时间敏感事实对齐任务，解决LLMs因训练数据时间冲突导致回答不准确的问题。提出用激活工程技术将LLaMA 2版本定位到特定时间点，通过调整注入层和提示策略，实验显示相对/显式提示分别提升44%/16%，效果媲美微调但更高效，无需额外数据。**

- **链接: [http://arxiv.org/pdf/2505.14158v1](http://arxiv.org/pdf/2505.14158v1)**

> **作者:** Sanjay Govindan; Maurice Pagnucco; Yang Song
>
> **摘要:** Large Language Models (LLMs) are trained on diverse and often conflicting knowledge spanning multiple domains and time periods. Some of this knowledge is only valid within specific temporal contexts, such as answering the question, "Who is the President of the United States in 2022?" Ensuring LLMs generate time appropriate responses is crucial for maintaining relevance and accuracy. In this work we explore activation engineering as a method for temporally aligning LLMs to improve factual recall without any training or dataset creation. In this research we explore an activation engineering technique to ground three versions of LLaMA 2 to specific points in time and examine the effects of varying injection layers and prompting strategies. Our experiments demonstrate up to a 44% and 16% improvement in relative and explicit prompting respectively, achieving comparable performance to the fine-tuning method proposed by Zhao et al. (2024) . Notably, our approach achieves similar results to the fine-tuning baseline while being significantly more computationally efficient and requiring no pre-aligned datasets.
>
---
#### [new 005] Can Pruning Improve Reasoning? Revisiting Long-CoT Compression with Capability in Mind for Better Reasoning
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决长推理链（Long-CoT）难以有效蒸馏至小模型的问题。提出Prune-on-Logic框架，将Long-CoT转为逻辑图并选择性修剪低效推理步骤，发现保留核心推理同时修剪验证步骤可提升小模型推理精度并降低成本，揭示语义精简而非缩短推理链更重要。**

- **链接: [http://arxiv.org/pdf/2505.14582v1](http://arxiv.org/pdf/2505.14582v1)**

> **作者:** Shangziqi Zhao; Jiahao Yuan; Guisong Yang; Usman Naseem
>
> **备注:** 17 pages,4 figures
>
> **摘要:** Long chain-of-thought (Long-CoT) reasoning improves accuracy in LLMs, yet its verbose, self-reflective style often hinders effective distillation into small language models (SLMs). We revisit Long-CoT compression through the lens of capability alignment and ask: Can pruning improve reasoning? We propose Prune-on-Logic, a structure-aware framework that transforms Long-CoT into logic graphs and selectively prunes low-utility reasoning steps under self-verification constraints. Through systematic analysis across three pruning strategies -- targeting entire chains, core reasoning, and verification -- we find that pruning verification steps yields consistent accuracy gains while reducing inference cost, outperforming token-level baselines and uncompressed fine-tuning. In contrast, pruning reasoning or all-chain steps degrades performance, revealing that small models benefit not from shorter CoTs, but from semantically leaner ones. Our findings highlight pruning as a structural optimization strategy for aligning CoT reasoning with SLM capacity.
>
---
#### [new 006] FAID: Fine-grained AI-generated Text Detection using Multi-task Auxiliary and Multi-level Contrastive Learning
- **分类: cs.CL**

- **简介: 该论文提出FAID框架，用于细粒度检测人类、AI生成及人机协作文本，并识别AI模型家族。任务为文本分类与模型溯源；解决现有二分类无法区分三类文本及模型家族的问题。工作包括构建多语言多领域数据集FAIDSet，结合多任务辅助分类与多级对比学习捕捉风格特征，增强跨领域泛化及分布外适应能力。**

- **链接: [http://arxiv.org/pdf/2505.14271v1](http://arxiv.org/pdf/2505.14271v1)**

> **作者:** Minh Ngoc Ta; Dong Cao Van; Duc-Anh Hoang; Minh Le-Anh; Truong Nguyen; My Anh Tran Nguyen; Yuxia Wang; Preslav Nakov; Sang Dinh
>
> **摘要:** The growing collaboration between humans and AI models in generative tasks has introduced new challenges in distinguishing between human-written, AI-generated, and human-AI collaborative texts. In this work, we collect a multilingual, multi-domain, multi-generator dataset FAIDSet. We further introduce a fine-grained detection framework FAID to classify text into these three categories, meanwhile identifying the underlying AI model family. Unlike existing binary classifiers, FAID is built to capture both authorship and model-specific characteristics. Our method combines multi-level contrastive learning with multi-task auxiliary classification to learn subtle stylistic cues. By modeling AI families as distinct stylistic entities, FAID offers improved interpretability. We incorporate an adaptation to address distributional shifts without retraining for unseen data. Experimental results demonstrate that FAID outperforms several baseline approaches, particularly enhancing the generalization accuracy on unseen domains and new AI models. It provide a potential solution for improving transparency and accountability in AI-assisted writing.
>
---
#### [new 007] SQLForge: Synthesizing Reliable and Diverse Data to Enhance Text-to-SQL Reasoning in LLMs
- **分类: cs.CL**

- **简介: 该论文属于文本到SQL推理任务，旨在缩小开源LLM与闭源模型的性能差距。提出SQLForge方法，通过SQL语法约束、反向翻译确保数据可靠性，结合模板扩展和领域探索提升多样性，微调开源模型形成SQLForge-LM，在Spider和BIRD基准中达85.7%和59.8%的准确率，性能最优。**

- **链接: [http://arxiv.org/pdf/2505.13725v1](http://arxiv.org/pdf/2505.13725v1)**

> **作者:** Yu Guo; Dong Jin; Shenghao Ye; Shuangwu Chen; Jian Yang; Xiaobin Tan
>
> **备注:** 12 pages, 7 figures, accepted to ACL Findings 2025
>
> **摘要:** Large Language models (LLMs) have demonstrated significant potential in text-to-SQL reasoning tasks, yet a substantial performance gap persists between existing open-source models and their closed-source counterparts. In this paper, we introduce SQLForge, a novel approach for synthesizing reliable and diverse data to enhance text-to-SQL reasoning in LLMs. We improve data reliability through SQL syntax constraints and SQL-to-question reverse translation, ensuring data logic at both structural and semantic levels. We also propose an SQL template enrichment and iterative data domain exploration mechanism to boost data diversity. Building on the augmented data, we fine-tune a variety of open-source models with different architectures and parameter sizes, resulting in a family of models termed SQLForge-LM. SQLForge-LM achieves the state-of-the-art performance on the widely recognized Spider and BIRD benchmarks among the open-source models. Specifically, SQLForge-LM achieves EX accuracy of 85.7% on Spider Dev and 59.8% on BIRD Dev, significantly narrowing the performance gap with closed-source methods.
>
---
#### [new 008] Success is in the Details: Evaluate and Enhance Details Sensitivity of Code LLMs through Counterfactuals
- **分类: cs.CL**

- **简介: 该论文属于代码LLMs评估与优化任务，旨在解决模型对问题描述细节变化敏感度不足的问题。提出CTF-Code基准（基于反事实扰动最小化输入变化以最大化输出差异），并开发CTF-Instruct微调框架，在现有数据中增加敏感度维度，实验显示显著提升模型性能。**

- **链接: [http://arxiv.org/pdf/2505.14597v1](http://arxiv.org/pdf/2505.14597v1)**

> **作者:** Xianzhen Luo; Qingfu Zhu; Zhiming Zhang; Mingzheng Xu; Tianhao Cheng; Yixuan Wang; Zheng Chu; Shijie Xuyang; Zhiyuan Ma; YuanTao Fan; Wanxiang Che
>
> **备注:** Code & Model is https://github.com/Luowaterbi/CTF-Instruct
>
> **摘要:** Code Sensitivity refers to the ability of Code LLMs to recognize and respond to details changes in problem descriptions. While current code benchmarks and instruction data focus on difficulty and diversity, sensitivity is overlooked. We first introduce the CTF-Code benchmark, constructed using counterfactual perturbations, minimizing input changes while maximizing output changes. The evaluation shows that many LLMs have a more than 10\% performance drop compared to the original problems. To fully utilize sensitivity, CTF-Instruct, an incremental instruction fine-tuning framework, extends on existing data and uses a selection mechanism to meet the three dimensions of difficulty, diversity, and sensitivity. Experiments show that LLMs fine-tuned with CTF-Instruct data achieve over a 2\% improvement on CTF-Code, and more than a 10\% performance boost on LiveCodeBench, validating the feasibility of enhancing LLMs' sensitivity to improve performance.
>
---
#### [new 009] Editing Across Languages: A Survey of Multilingual Knowledge Editing
- **分类: cs.CL**

- **简介: 该论文属于多语言知识编辑（MKE）任务，旨在解决跨语言事实性编辑可靠性问题。系统梳理了参数、记忆、微调和超网络等方法，总结评估基准与效果，分析跨语言传播挑战，并指出语言异质性、评估覆盖等开放问题，为可编辑语言模型研究提供基础。**

- **链接: [http://arxiv.org/pdf/2505.14393v1](http://arxiv.org/pdf/2505.14393v1)**

> **作者:** Nadir Durrani; Basel Mousi; Fahim Dalvi
>
> **摘要:** While Knowledge Editing has been extensively studied in monolingual settings, it remains underexplored in multilingual contexts. This survey systematizes recent research on Multilingual Knowledge Editing (MKE), a growing subdomain of model editing focused on ensuring factual edits generalize reliably across languages. We present a comprehensive taxonomy of MKE methods, covering parameter-based, memory-based, fine-tuning, and hypernetwork approaches. We survey available benchmarks,summarize key findings on method effectiveness and transfer patterns, identify challenges in cross-lingual propagation, and highlight open problems related to language anisotropy, evaluation coverage, and edit scalability. Our analysis consolidates a rapidly evolving area and lays the groundwork for future progress in editable language-aware LLMs.
>
---
#### [new 010] "Haet Bhasha aur Diskrimineshun": Phonetic Perturbations in Code-Mixed Hinglish to Red-Team LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出利用拼写扰动的印地-英语混合语（Hinglish）提示攻击多语言多模态模型，解决现有红队测试多语言覆盖不足的问题。通过混合语言与语音扰动策略，实现文本生成99%、图像生成78%的攻击成功率，揭示语音扰动干扰分词机制，推动多模态模型安全对齐研究。**

- **链接: [http://arxiv.org/pdf/2505.14226v1](http://arxiv.org/pdf/2505.14226v1)**

> **作者:** Darpan Aswal; Siddharth D Jaiswal
>
> **摘要:** Large Language Models (LLMs) have become increasingly powerful, with multilingual and multimodal capabilities improving by the day. These models are being evaluated through audits, alignment studies and red-teaming efforts to expose model vulnerabilities towards generating harmful, biased and unfair content. Existing red-teaming efforts have previously focused on the English language, using fixed template-based attacks; thus, models continue to be susceptible to multilingual jailbreaking strategies, especially in the multimodal context. In this study, we introduce a novel strategy that leverages code-mixing and phonetic perturbations to jailbreak LLMs for both text and image generation tasks. We also introduce two new jailbreak strategies that show higher effectiveness than baseline strategies. Our work presents a method to effectively bypass safety filters in LLMs while maintaining interpretability by applying phonetic misspellings to sensitive words in code-mixed prompts. Our novel prompts achieve a 99% Attack Success Rate for text generation and 78% for image generation, with Attack Relevance Rate of 100% for text generation and 95% for image generation when using the phonetically perturbed code-mixed prompts. Our interpretability experiments reveal that phonetic perturbations impact word tokenization, leading to jailbreak success. Our study motivates increasing the focus towards more generalizable safety alignment for multilingual multimodal models, especially in real-world settings wherein prompts can have misspelt words.
>
---
#### [new 011] Hidden Ghost Hand: Unveiling Backdoor Vulnerabilities in MLLM-Powered Mobile GUI Agents
- **分类: cs.CL**

- **简介: 该论文属于AI安全领域，研究MLLM驱动的移动GUI代理的后门漏洞。针对开放模型供应链威胁，提出AgentGhost框架，利用交互级触发器（如历史步骤、环境状态）结合目标级触发器，通过Min-Max优化和监督对比学习注入隐蔽后门，实现99.7%攻击成功率且仅1%功能降级，并设计防御方法将其准确率降至22.1%。**

- **链接: [http://arxiv.org/pdf/2505.14418v1](http://arxiv.org/pdf/2505.14418v1)**

> **作者:** Pengzhou Cheng; Haowen Hu; Zheng Wu; Zongru Wu; Tianjie Ju; Daizong Ding; Zhuosheng Zhang; Gongshen Liu
>
> **备注:** 25 pages, 10 figures, 12 Tables
>
> **摘要:** Graphical user interface (GUI) agents powered by multimodal large language models (MLLMs) have shown greater promise for human-interaction. However, due to the high fine-tuning cost, users often rely on open-source GUI agents or APIs offered by AI providers, which introduces a critical but underexplored supply chain threat: backdoor attacks. In this work, we first unveil that MLLM-powered GUI agents naturally expose multiple interaction-level triggers, such as historical steps, environment states, and task progress. Based on this observation, we introduce AgentGhost, an effective and stealthy framework for red-teaming backdoor attacks. Specifically, we first construct composite triggers by combining goal and interaction levels, allowing GUI agents to unintentionally activate backdoors while ensuring task utility. Then, we formulate backdoor injection as a Min-Max optimization problem that uses supervised contrastive learning to maximize the feature difference across sample classes at the representation space, improving flexibility of the backdoor. Meanwhile, it adopts supervised fine-tuning to minimize the discrepancy between backdoor and clean behavior generation, enhancing effectiveness and utility. Extensive evaluations of various agent models in two established mobile benchmarks show that AgentGhost is effective and generic, with attack accuracy that reaches 99.7\% on three attack objectives, and shows stealthiness with only 1\% utility degradation. Furthermore, we tailor a defense method against AgentGhost that reduces the attack accuracy to 22.1\%. Our code is available at \texttt{anonymous}.
>
---
#### [new 012] Logic Jailbreak: Efficiently Unlocking LLM Safety Restrictions Through Formal Logical Expression
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI安全任务，旨在解决大语言模型（LLM）安全机制易受jailbreak攻击的问题。提出LogiBreak方法，通过将恶意自然语言提示转化为形式逻辑表达，利用训练数据与逻辑输入的分布差异绕过安全限制；在多语言数据集验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.13527v1](http://arxiv.org/pdf/2505.13527v1)**

> **作者:** Jingyu Peng; Maolin Wang; Nan Wang; Xiangyu Zhao; Jiatong Li; Kai Zhang; Qi Liu
>
> **摘要:** Despite substantial advancements in aligning large language models (LLMs) with human values, current safety mechanisms remain susceptible to jailbreak attacks. We hypothesize that this vulnerability stems from distributional discrepancies between alignment-oriented prompts and malicious prompts. To investigate this, we introduce LogiBreak, a novel and universal black-box jailbreak method that leverages logical expression translation to circumvent LLM safety systems. By converting harmful natural language prompts into formal logical expressions, LogiBreak exploits the distributional gap between alignment data and logic-based inputs, preserving the underlying semantic intent and readability while evading safety constraints. We evaluate LogiBreak on a multilingual jailbreak dataset spanning three languages, demonstrating its effectiveness across various evaluation settings and linguistic contexts.
>
---
#### [new 013] Social Sycophancy: A Broader Understanding of LLM Sycophancy
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究LLM逢迎行为，扩展其定义至模糊情境（如建议、支持场景），提出"社会逢迎"理论，将其视为过度维护用户面子。开发ELEPHANT框架，通过五种行为指标在两个数据集评估八模型，发现LLM比人类更倾向逢迎（OEQ高47%，AITA错误支持42%），揭示其难以缓解且受偏好数据奖励。**

- **链接: [http://arxiv.org/pdf/2505.13995v1](http://arxiv.org/pdf/2505.13995v1)**

> **作者:** Myra Cheng; Sunny Yu; Cinoo Lee; Pranav Khadpe; Lujain Ibrahim; Dan Jurafsky
>
> **摘要:** A serious risk to the safety and utility of LLMs is sycophancy, i.e., excessive agreement with and flattery of the user. Yet existing work focuses on only one aspect of sycophancy: agreement with users' explicitly stated beliefs that can be compared to a ground truth. This overlooks forms of sycophancy that arise in ambiguous contexts such as advice and support-seeking, where there is no clear ground truth, yet sycophancy can reinforce harmful implicit assumptions, beliefs, or actions. To address this gap, we introduce a richer theory of social sycophancy in LLMs, characterizing sycophancy as the excessive preservation of a user's face (the positive self-image a person seeks to maintain in an interaction). We present ELEPHANT, a framework for evaluating social sycophancy across five face-preserving behaviors (emotional validation, moral endorsement, indirect language, indirect action, and accepting framing) on two datasets: open-ended questions (OEQ) and Reddit's r/AmITheAsshole (AITA). Across eight models, we show that LLMs consistently exhibit high rates of social sycophancy: on OEQ, they preserve face 47% more than humans, and on AITA, they affirm behavior deemed inappropriate by crowdsourced human judgments in 42% of cases. We further show that social sycophancy is rewarded in preference datasets and is not easily mitigated. Our work provides theoretical grounding and empirical tools (datasets and code) for understanding and addressing this under-recognized but consequential issue.
>
---
#### [new 014] MCIP: Protecting MCP Safety via Model Contextual Integrity Protocol
- **分类: cs.CL**

- **简介: 该论文属模型安全任务，针对MCP协议分散架构引发的安全风险，提出MCIP协议。通过分析MCP安全漏洞，建立细粒度风险分类与基准数据，改进LLMs在MCP交互中的安全检测能力，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.14590v1](http://arxiv.org/pdf/2505.14590v1)**

> **作者:** Huihao Jing; Haoran Li; Wenbin Hu; Qi Hu; Heli Xu; Tianshu Chu; Peizhao Hu; Yangqiu Song
>
> **备注:** 17 pages
>
> **摘要:** As Model Context Protocol (MCP) introduces an easy-to-use ecosystem for users and developers, it also brings underexplored safety risks. Its decentralized architecture, which separates clients and servers, poses unique challenges for systematic safety analysis. This paper proposes a novel framework to enhance MCP safety. Guided by the MAESTRO framework, we first analyze the missing safety mechanisms in MCP, and based on this analysis, we propose the Model Contextual Integrity Protocol (MCIP), a refined version of MCP that addresses these gaps.Next, we develop a fine-grained taxonomy that captures a diverse range of unsafe behaviors observed in MCP scenarios. Building on this taxonomy, we develop benchmark and training data that support the evaluation and improvement of LLMs' capabilities in identifying safety risks within MCP interactions. Leveraging the proposed benchmark and training data, we conduct extensive experiments on state-of-the-art LLMs. The results highlight LLMs' vulnerabilities in MCP interactions and demonstrate that our approach substantially improves their safety performance.
>
---
#### [new 015] Activation-Guided Consensus Merging for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于Large Language Models（LLMs）的模型合并任务。针对传统方法忽视神经层功能差异的问题，提出Activation-Guided Consensus Merging（ACM），通过计算预训练与微调模型激活间的互信息，动态分配层级合并系数，无需梯度计算即可保留任务能力。实验显示其在减少响应长度（55.3%）和提升推理精度（+1.3）上优于基线。**

- **链接: [http://arxiv.org/pdf/2505.14009v1](http://arxiv.org/pdf/2505.14009v1)**

> **作者:** Yuxuan Yao; Shuqi Liu; Zehua Liu; Qintong Li; Mingyang Liu; Xiongwei Han; Zhijiang Guo; Han Wu; Linqi Song
>
> **摘要:** Recent research has increasingly focused on reconciling the reasoning capabilities of System 2 with the efficiency of System 1. While existing training-based and prompt-based approaches face significant challenges in terms of efficiency and stability, model merging emerges as a promising strategy to integrate the diverse capabilities of different Large Language Models (LLMs) into a unified model. However, conventional model merging methods often assume uniform importance across layers, overlooking the functional heterogeneity inherent in neural components. To address this limitation, we propose \textbf{A}ctivation-Guided \textbf{C}onsensus \textbf{M}erging (\textbf{ACM}), a plug-and-play merging framework that determines layer-specific merging coefficients based on mutual information between activations of pre-trained and fine-tuned models. ACM effectively preserves task-specific capabilities without requiring gradient computations or additional training. Extensive experiments on Long-to-Short (L2S) and general merging tasks demonstrate that ACM consistently outperforms all baseline methods. For instance, in the case of Qwen-7B models, TIES-Merging equipped with ACM achieves a \textbf{55.3\%} reduction in response length while simultaneously improving reasoning accuracy by \textbf{1.3} points. We submit the code with the paper for reproducibility, and it will be publicly available.
>
---
#### [new 016] Evaluating Reasoning LLMs for Suicide Screening with the Columbia-Suicide Severity Rating Scale
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文评估LLMs（如Claude、GPT）使用C-SSRS量表进行自杀风险自动评估的可行性，解决AI替代人类筛查的潜在问题。通过零样本测试六模型在7级风险分类的表现，发现Claude/GPT接近人工标注，Mistral误差最低，误判多为相邻等级，强调需人类监督与伦理考量。**

- **链接: [http://arxiv.org/pdf/2505.13480v1](http://arxiv.org/pdf/2505.13480v1)**

> **作者:** Avinash Patil; Siru Tao; Amardeep Gedhu
>
> **备注:** 8 Pages, 6 Figures, 1 Table
>
> **摘要:** Suicide prevention remains a critical public health challenge. While online platforms such as Reddit's r/SuicideWatch have historically provided spaces for individuals to express suicidal thoughts and seek community support, the advent of large language models (LLMs) introduces a new paradigm-where individuals may begin disclosing ideation to AI systems instead of humans. This study evaluates the capability of LLMs to perform automated suicide risk assessment using the Columbia-Suicide Severity Rating Scale (C-SSRS). We assess the zero-shot performance of six models-including Claude, GPT, Mistral, and LLaMA-in classifying posts across a 7-point severity scale (Levels 0-6). Results indicate that Claude and GPT closely align with human annotations, while Mistral achieves the lowest ordinal prediction error. Most models exhibit ordinal sensitivity, with misclassifications typically occurring between adjacent severity levels. We further analyze confusion patterns, misclassification sources, and ethical considerations, underscoring the importance of human oversight, transparency, and cautious deployment. Full code and supplementary materials are available at https://github.com/av9ash/llm_cssrs_code.
>
---
#### [new 017] ABBA: Highly Expressive Hadamard Product Adaptation for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于参数高效微调（PEFT）任务，旨在解决现有方法（如LoRA、HiRA）表达能力受限的问题。提出ABBA方法，通过两个独立可学习的低秩矩阵的Hadamard积完全解耦更新，提升表达能力，在推理任务中表现最优。**

- **链接: [http://arxiv.org/pdf/2505.14238v1](http://arxiv.org/pdf/2505.14238v1)**

> **作者:** Raghav Singhal; Kaustubh Ponkshe; Rohit Vartak; Praneeth Vepakomma
>
> **备注:** Raghav Singhal, Kaustubh Ponkshe, and Rohit Vartak contributed equally to this work
>
> **摘要:** Large Language Models have demonstrated strong performance across a wide range of tasks, but adapting them efficiently to new domains remains a key challenge. Parameter-Efficient Fine-Tuning (PEFT) methods address this by introducing lightweight, trainable modules while keeping most pre-trained weights fixed. The prevailing approach, LoRA, models updates using a low-rank decomposition, but its expressivity is inherently constrained by the rank. Recent methods like HiRA aim to increase expressivity by incorporating a Hadamard product with the frozen weights, but still rely on the structure of the pre-trained model. We introduce ABBA, a new PEFT architecture that reparameterizes the update as a Hadamard product of two independently learnable low-rank matrices. In contrast to prior work, ABBA fully decouples the update from the pre-trained weights, enabling both components to be optimized freely. This leads to significantly higher expressivity under the same parameter budget. We formally analyze ABBA's expressive capacity and validate its advantages through matrix reconstruction experiments. Empirically, ABBA achieves state-of-the-art results on arithmetic and commonsense reasoning benchmarks, consistently outperforming existing PEFT methods by a significant margin across multiple models. Our code is publicly available at: https://github.com/CERT-Lab/abba.
>
---
#### [new 018] Unraveling Interwoven Roles of Large Language Models in Authorship Privacy: Obfuscation, Mimicking, and Verification
- **分类: cs.CL**

- **简介: 该论文聚焦作者身份隐私，研究LLMs在文本生成中隐式泄露隐私的问题。提出首个统一框架，分析AO（混淆）、AM（模仿）、AV（验证）三任务的动态关系及时效性，并评估人口元数据对其性能及隐私风险的影响。**

- **链接: [http://arxiv.org/pdf/2505.14195v1](http://arxiv.org/pdf/2505.14195v1)**

> **作者:** Tuc Nguyen; Yifan Hu; Thai Le
>
> **备注:** 17 pages, 3 figures
>
> **摘要:** Recent advancements in large language models (LLMs) have been fueled by large scale training corpora drawn from diverse sources such as websites, news articles, and books. These datasets often contain explicit user information, such as person names and addresses, that LLMs may unintentionally reproduce in their generated outputs. Beyond such explicit content, LLMs can also leak identity revealing cues through implicit signals such as distinctive writing styles, raising significant concerns about authorship privacy. There are three major automated tasks in authorship privacy, namely authorship obfuscation (AO), authorship mimicking (AM), and authorship verification (AV). Prior research has studied AO, AM, and AV independently. However, their interplays remain under explored, which leaves a major research gap, especially in the era of LLMs, where they are profoundly shaping how we curate and share user generated content, and the distinction between machine generated and human authored text is also increasingly blurred. This work then presents the first unified framework for analyzing the dynamic relationships among LLM enabled AO, AM, and AV in the context of authorship privacy. We quantify how they interact with each other to transform human authored text, examining effects at a single point in time and iteratively over time. We also examine the role of demographic metadata, such as gender, academic background, in modulating their performances, inter-task dynamics, and privacy risks. All source code will be publicly available.
>
---
#### [new 019] WirelessMathBench: A Mathematical Modeling Benchmark for LLMs in Wireless Communications
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出WirelessMathBench，一个针对LLMs在无线通信数学建模能力的基准测试。旨在解决LLMs在领域复杂数学推理中的不足，通过587个精选任务（含方程补全等）评估模型，发现现有模型在复杂任务表现差（最佳模型DeepSeek-R1方程补全仅7.83%），并公开工具以推动领域专用LLM发展。**

- **链接: [http://arxiv.org/pdf/2505.14354v1](http://arxiv.org/pdf/2505.14354v1)**

> **作者:** Xin Li; Mengbing Liu; Li Wei; Jiancheng An; Mérouane Debbah; Chau Yuen
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Large Language Models (LLMs) have achieved impressive results across a broad array of tasks, yet their capacity for complex, domain-specific mathematical reasoning-particularly in wireless communications-remains underexplored. In this work, we introduce WirelessMathBench, a novel benchmark specifically designed to evaluate LLMs on mathematical modeling challenges to wireless communications engineering. Our benchmark consists of 587 meticulously curated questions sourced from 40 state-of-the-art research papers, encompassing a diverse spectrum of tasks ranging from basic multiple-choice questions to complex equation completion tasks, including both partial and full completions, all of which rigorously adhere to physical and dimensional constraints. Through extensive experimentation with leading LLMs, we observe that while many models excel in basic recall tasks, their performance degrades significantly when reconstructing partially or fully obscured equations, exposing fundamental limitations in current LLMs. Even DeepSeek-R1, the best performer on our benchmark, achieves an average accuracy of only 38.05%, with a mere 7.83% success rate in full equation completion. By publicly releasing WirelessMathBench along with the evaluation toolkit, we aim to advance the development of more robust, domain-aware LLMs for wireless system analysis and broader engineering applications.
>
---
#### [new 020] FlashThink: An Early Exit Method For Efficient Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出FlashThink方法，针对大语言模型（LLMs）在推理任务中生成冗长内容导致计算效率低的问题。通过构建验证模型识别推理过程中的正确退出时机，实现早期终止推理，减少77%的推理内容且不降低准确率。任务为高效推理，解决计算冗余问题，工作包括提出早退机制及验证实验。**

- **链接: [http://arxiv.org/pdf/2505.13949v1](http://arxiv.org/pdf/2505.13949v1)**

> **作者:** Guochao Jiang; Guofeng Quan; Zepeng Ding; Ziqin Luo; Dixuan Wang; Zheng Hu
>
> **摘要:** Large Language Models (LLMs) have shown impressive performance in reasoning tasks. However, LLMs tend to generate excessively long reasoning content, leading to significant computational overhead. Our observations indicate that even on simple problems, LLMs tend to produce unnecessarily lengthy reasoning content, which is against intuitive expectations. Preliminary experiments show that at a certain point during the generation process, the model is already capable of producing the correct solution without completing the full reasoning content. Therefore, we consider that the reasoning process of the model can be exited early to achieve the purpose of efficient reasoning. We introduce a verification model that identifies the exact moment when the model can stop reasoning and still provide the correct answer. Comprehensive experiments on four different benchmarks demonstrate that our proposed method, FlashThink, effectively shortens the reasoning content while preserving the model accuracy. For the Deepseek-R1 and QwQ-32B models, we reduced the length of reasoning content by 77.04% and 77.47%, respectively, without reducing the accuracy.
>
---
#### [new 021] TRATES: Trait-Specific Rubric-Assisted Cross-Prompt Essay Scoring
- **分类: cs.CL**

- **简介: 该论文属于自动作文评分任务，旨在解决现有方法缺乏针对作文个体特征的精准评估问题。提出TRATES框架，利用大语言模型根据评分标准生成特征问题，结合通用写作质量和题目特定特征训练回归模型，实现跨题目的特质评分，在基准数据集上达新最优效果。**

- **链接: [http://arxiv.org/pdf/2505.14577v1](http://arxiv.org/pdf/2505.14577v1)**

> **作者:** Sohaila Eltanbouly; Salam Albatarni; Tamer Elsayed
>
> **备注:** Accepted at ACL 2025 Findings
>
> **摘要:** Research on holistic Automated Essay Scoring (AES) is long-dated; yet, there is a notable lack of attention for assessing essays according to individual traits. In this work, we propose TRATES, a novel trait-specific and rubric-based cross-prompt AES framework that is generic yet specific to the underlying trait. The framework leverages a Large Language Model (LLM) that utilizes the trait grading rubrics to generate trait-specific features (represented by assessment questions), then assesses those features given an essay. The trait-specific features are eventually combined with generic writing-quality and prompt-specific features to train a simple classical regression model that predicts trait scores of essays from an unseen prompt. Experiments show that TRATES achieves a new state-of-the-art performance across all traits on a widely-used dataset, with the generated LLM-based features being the most significant.
>
---
#### [new 022] Toward Effective Reinforcement Learning Fine-Tuning for Medical VQA in Vision-Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文聚焦医疗视觉问答任务，研究强化学习微调在医学多模态模型中的应用挑战。针对直接应用RL效果不佳的问题，探讨了基础模型初始化、医学语义对齐、奖励机制及偏差四大维度的影响，通过大量实验验证GRPO方法在准确性和推理质量上优于传统监督微调。**

- **链接: [http://arxiv.org/pdf/2505.13973v1](http://arxiv.org/pdf/2505.13973v1)**

> **作者:** Wenhui Zhu; Xuanzhao Dong; Xin Li; Peijie Qiu; Xiwen Chen; Abolfazl Razi; Aris Sotiras; Yi Su; Yalin Wang
>
> **摘要:** Recently, reinforcement learning (RL)-based tuning has shifted the trajectory of Multimodal Large Language Models (MLLMs), particularly following the introduction of Group Relative Policy Optimization (GRPO). However, directly applying it to medical tasks remains challenging for achieving clinically grounded model behavior. Motivated by the need to align model response with clinical expectations, we investigate four critical dimensions that affect the effectiveness of RL-based tuning in medical visual question answering (VQA): base model initialization strategy, the role of medical semantic alignment, the impact of length-based rewards on long-chain reasoning, and the influence of bias. We conduct extensive experiments to analyze these factors for medical MLLMs, providing new insights into how models are domain-specifically fine-tuned. Additionally, our results also demonstrate that GRPO-based RL tuning consistently outperforms standard supervised fine-tuning (SFT) in both accuracy and reasoning quality.
>
---
#### [new 023] A MIND for Reasoning: Meta-learning for In-context Deduction
- **分类: cs.CL**

- **简介: 该论文属于知识库前提选择任务，旨在解决大语言模型（LLMs）在泛化到新问题时演绎推理能力不足的问题。提出MIND方法，通过元学习微调，使小模型（1.5B-7B参数）能系统性识别支持假设的前提，显著提升泛化性能，尤其在低数据场景超越GPT-4o等大模型。**

- **链接: [http://arxiv.org/pdf/2505.14313v1](http://arxiv.org/pdf/2505.14313v1)**

> **作者:** Leonardo Bertolazzi; Manuel Vargas Guzmán; Raffaella Bernardi; Maciej Malicki; Jakub Szymanik
>
> **摘要:** Large language models (LLMs) are increasingly evaluated on formal tasks, where strong reasoning abilities define the state of the art. However, their ability to generalize to out-of-distribution problems remains limited. In this paper, we investigate how LLMs can achieve a systematic understanding of deductive rules. Our focus is on the task of identifying the appropriate subset of premises within a knowledge base needed to derive a given hypothesis. To tackle this challenge, we propose Meta-learning for In-context Deduction (MIND), a novel few-shot meta-learning fine-tuning approach. The goal of MIND is to enable models to generalize more effectively to unseen knowledge bases and to systematically apply inference rules. Our results show that MIND significantly improves generalization in small LMs ranging from 1.5B to 7B parameters. The benefits are especially pronounced in smaller models and low-data settings. Remarkably, small models fine-tuned with MIND outperform state-of-the-art LLMs, such as GPT-4o and o3-mini, on this task.
>
---
#### [new 024] TransBench: Benchmarking Machine Translation for Industrial-Scale Applications
- **分类: cs.CL**

- **简介: 该论文提出TransBench，针对工业级机器翻译（MT）的基准测试，解决通用MT模型在领域术语、文化差异等工业场景中的局限性。通过构建三层次评估框架（基础语言能力、领域专业性、文化适配），并创建含17,000句电商场景数据的多语言基准，结合传统指标与领域模型Marco-MOS，填补学术与工业评估的鸿沟，提供开源工具助力系统优化。**

- **链接: [http://arxiv.org/pdf/2505.14244v1](http://arxiv.org/pdf/2505.14244v1)**

> **作者:** Haijun Li; Tianqi Shi; Zifu Shang; Yuxuan Han; Xueyu Zhao; Hao Wang; Yu Qian; Zhiqiang Qian; Linlong Xu; Minghao Wu; Chenyang Lyu; Longyue Wang; Gongbo Tang; Weihua Luo; Zhao Xu; Kaifu Zhang
>
> **摘要:** Machine translation (MT) has become indispensable for cross-border communication in globalized industries like e-commerce, finance, and legal services, with recent advancements in large language models (LLMs) significantly enhancing translation quality. However, applying general-purpose MT models to industrial scenarios reveals critical limitations due to domain-specific terminology, cultural nuances, and stylistic conventions absent in generic benchmarks. Existing evaluation frameworks inadequately assess performance in specialized contexts, creating a gap between academic benchmarks and real-world efficacy. To address this, we propose a three-level translation capability framework: (1) Basic Linguistic Competence, (2) Domain-Specific Proficiency, and (3) Cultural Adaptation, emphasizing the need for holistic evaluation across these dimensions. We introduce TransBench, a benchmark tailored for industrial MT, initially targeting international e-commerce with 17,000 professionally translated sentences spanning 4 main scenarios and 33 language pairs. TransBench integrates traditional metrics (BLEU, TER) with Marco-MOS, a domain-specific evaluation model, and provides guidelines for reproducible benchmark construction. Our contributions include: (1) a structured framework for industrial MT evaluation, (2) the first publicly available benchmark for e-commerce translation, (3) novel metrics probing multi-level translation quality, and (4) open-sourced evaluation tools. This work bridges the evaluation gap, enabling researchers and practitioners to systematically assess and enhance MT systems for industry-specific needs.
>
---
#### [new 025] BAR: A Backward Reasoning based Agent for Complex Minecraft Tasks
- **分类: cs.CL**

- **简介: 该论文提出BAR代理，针对Minecraft复杂任务中正向推理失效问题（因初始状态与目标差距大），通过逆向推理从目标状态规划，设计递归目标分解、状态一致性维护和阶段记忆模块，实验验证其优势。**

- **链接: [http://arxiv.org/pdf/2505.14079v1](http://arxiv.org/pdf/2505.14079v1)**

> **作者:** Weihong Du; Wenrui Liao; Binyu Yan; Hongru Liang; Anthony G. Cohn; Wenqiang Lei
>
> **摘要:** Large language model (LLM) based agents have shown great potential in following human instructions and automatically completing various tasks. To complete a task, the agent needs to decompose it into easily executed steps by planning. Existing studies mainly conduct the planning by inferring what steps should be executed next starting from the agent's initial state. However, this forward reasoning paradigm doesn't work well for complex tasks. We propose to study this issue in Minecraft, a virtual environment that simulates complex tasks based on real-world scenarios. We believe that the failure of forward reasoning is caused by the big perception gap between the agent's initial state and task goal. To this end, we leverage backward reasoning and make the planning starting from the terminal state, which can directly achieve the task goal in one step. Specifically, we design a BAckward Reasoning based agent (BAR). It is equipped with a recursive goal decomposition module, a state consistency maintaining module and a stage memory module to make robust, consistent, and efficient planning starting from the terminal state. Experimental results demonstrate the superiority of BAR over existing methods and the effectiveness of proposed modules.
>
---
#### [new 026] Are Large Language Models Good at Detecting Propaganda?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLM）在检测新闻中宣传技巧的任务表现，旨在评估其识别操纵性内容的能力。通过对比GPT-4、GPT-3.5、Claude 3与RoBERTa-CRF及MGN基线模型，发现GPT-4虽优于前代LLM（F1=0.16），但不及RoBERTa-CRF（F1=0.67）；三LLM仅在name-calling等部分技巧检测上超越MGN。**

- **链接: [http://arxiv.org/pdf/2505.13706v1](http://arxiv.org/pdf/2505.13706v1)**

> **作者:** Julia Jose; Rachel Greenstadt
>
> **摘要:** Propagandists use rhetorical devices that rely on logical fallacies and emotional appeals to advance their agendas. Recognizing these techniques is key to making informed decisions. Recent advances in Natural Language Processing (NLP) have enabled the development of systems capable of detecting manipulative content. In this study, we look at several Large Language Models and their performance in detecting propaganda techniques in news articles. We compare the performance of these LLMs with transformer-based models. We find that, while GPT-4 demonstrates superior F1 scores (F1=0.16) compared to GPT-3.5 and Claude 3 Opus, it does not outperform a RoBERTa-CRF baseline (F1=0.67). Additionally, we find that all three LLMs outperform a MultiGranularity Network (MGN) baseline in detecting instances of one out of six propaganda techniques (name-calling), with GPT-3.5 and GPT-4 also outperforming the MGN baseline in detecting instances of appeal to fear and flag-waving.
>
---
#### [new 027] Data-Efficient Hate Speech Detection via Cross-Lingual Nearest Neighbor Retrieval with Limited Labeled Data
- **分类: cs.CL; cs.CY; cs.MM**

- **简介: 该论文研究跨语言仇恨言论检测任务，旨在解决低资源语言标注数据稀缺问题。提出通过跨语言最近邻检索，用少量目标语言标注数据从多语料池中选取相关样本增强训练，提升检测效果。实验显示其方法在八种语言上优于基线模型，仅需200个样本即表现优异，并采用最大边际相关性减少冗余，兼具高效与可扩展性。**

- **链接: [http://arxiv.org/pdf/2505.14272v1](http://arxiv.org/pdf/2505.14272v1)**

> **作者:** Faeze Ghorbanpour; Daryna Dementieva; Alexander Fraser
>
> **摘要:** Considering the importance of detecting hateful language, labeled hate speech data is expensive and time-consuming to collect, particularly for low-resource languages. Prior work has demonstrated the effectiveness of cross-lingual transfer learning and data augmentation in improving performance on tasks with limited labeled data. To develop an efficient and scalable cross-lingual transfer learning approach, we leverage nearest-neighbor retrieval to augment minimal labeled data in the target language, thereby enhancing detection performance. Specifically, we assume access to a small set of labeled training instances in the target language and use these to retrieve the most relevant labeled examples from a large multilingual hate speech detection pool. We evaluate our approach on eight languages and demonstrate that it consistently outperforms models trained solely on the target language data. Furthermore, in most cases, our method surpasses the current state-of-the-art. Notably, our approach is highly data-efficient, retrieving as small as 200 instances in some cases while maintaining superior performance. Moreover, it is scalable, as the retrieval pool can be easily expanded, and the method can be readily adapted to new languages and tasks. We also apply maximum marginal relevance to mitigate redundancy and filter out highly similar retrieved instances, resulting in improvements in some languages.
>
---
#### [new 028] The Strawberry Problem: Emergence of Character-level Understanding in Tokenized Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型优化任务，旨在解决分词机制导致的字符级理解缺陷（如字母计数失败）。通过设计19个合成任务，发现字符级推理能力缓慢且突发性涌现，提出渗流模型解释该现象，并引入轻量架构改进字符级推理，平衡子词模型优势。**

- **链接: [http://arxiv.org/pdf/2505.14172v1](http://arxiv.org/pdf/2505.14172v1)**

> **作者:** Adrian Cosma; Stefan Ruseti; Emilian Radoi; Mihai Dascalu
>
> **备注:** 1 Table, 8 Figures
>
> **摘要:** Despite their remarkable progress across diverse domains, Large Language Models (LLMs) consistently fail at simple character-level tasks, such as counting letters in words, due to a fundamental limitation: tokenization. In this work, we frame this limitation as a problem of low mutual information and analyze it in terms of concept emergence. Using a suite of 19 synthetic tasks that isolate character-level reasoning in a controlled setting, we show that such capabilities emerge slowly, suddenly, and only late in training. We further show that percolation-based models of concept emergence explain these patterns, suggesting that learning character composition is not fundamentally different from learning commonsense knowledge. To address this bottleneck, we propose a lightweight architectural modification that significantly improves character-level reasoning while preserving the inductive advantages of subword models. Together, our results bridge low-level perceptual gaps in tokenized LMs and provide a principled framework for understanding and mitigating their structural blind spots. We make our code publicly available.
>
---
#### [new 029] Adapting Pretrained Language Models for Citation Classification via Self-Supervised Contrastive Learning
- **分类: cs.CL**

- **简介: 该论文属于学术引文分类任务，旨在解决标注数据稀缺、上下文噪声和关键词虚假关联问题。提出框架Citss，通过自监督对比学习（含句子裁剪和关键词扰动策略）优化预训练语言模型，同时兼容编码器和解码器模型，提升分类效果。**

- **链接: [http://arxiv.org/pdf/2505.14471v1](http://arxiv.org/pdf/2505.14471v1)**

> **作者:** Tong Li; Jiachuan Wang; Yongqi Zhang; Shuangyin Li; Lei Chen
>
> **备注:** Manuscripts, accepted to KDD 2025
>
> **摘要:** Citation classification, which identifies the intention behind academic citations, is pivotal for scholarly analysis. Previous works suggest fine-tuning pretrained language models (PLMs) on citation classification datasets, reaping the reward of the linguistic knowledge they gained during pretraining. However, directly fine-tuning for citation classification is challenging due to labeled data scarcity, contextual noise, and spurious keyphrase correlations. In this paper, we present a novel framework, Citss, that adapts the PLMs to overcome these challenges. Citss introduces self-supervised contrastive learning to alleviate data scarcity, and is equipped with two specialized strategies to obtain the contrastive pairs: sentence-level cropping, which enhances focus on target citations within long contexts, and keyphrase perturbation, which mitigates reliance on specific keyphrases. Compared with previous works that are only designed for encoder-based PLMs, Citss is carefully developed to be compatible with both encoder-based PLMs and decoder-based LLMs, to embrace the benefits of enlarged pretraining. Experiments with three benchmark datasets with both encoder-based PLMs and decoder-based LLMs demonstrate our superiority compared to the previous state of the art. Our code is available at: github.com/LITONG99/Citss
>
---
#### [new 030] Domain Gating Ensemble Networks for AI-Generated Text Detection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出DoGEN方法，用于AI生成文本检测。针对现有模型难以适应新领域的问题，通过集成领域专家模型并结合领域分类器权重，提升跨领域检测能力。实验显示其在领域内检测达最优，且优于更大模型的跨领域表现，并开源代码与模型。（99字）**

- **链接: [http://arxiv.org/pdf/2505.13855v1](http://arxiv.org/pdf/2505.13855v1)**

> **作者:** Arihant Tripathi; Liam Dugan; Charis Gao; Maggie Huan; Emma Jin; Peter Zhang; David Zhang; Julia Zhao; Chris Callison-Burch
>
> **备注:** Submitted to EMNLP 2025
>
> **摘要:** As state-of-the-art language models continue to improve, the need for robust detection of machine-generated text becomes increasingly critical. However, current state-of-the-art machine text detectors struggle to adapt to new unseen domains and generative models. In this paper we present DoGEN (Domain Gating Ensemble Networks), a technique that allows detectors to adapt to unseen domains by ensembling a set of domain expert detector models using weights from a domain classifier. We test DoGEN on a wide variety of domains from leading benchmarks and find that it achieves state-of-the-art performance on in-domain detection while outperforming models twice its size on out-of-domain detection. We release our code and trained models to assist in future research in domain-adaptive AI detection.
>
---
#### [new 031] Automatic Dataset Generation for Knowledge Intensive Question Answering Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识密集型问答任务，针对现有系统在复杂推理及多源信息整合中的不足，提出通过LLMs自动生成QA对以减少人工标注并提升模型推理能力，实验显示其逻辑连贯性和事实准确性优于人类标注数据。**

- **链接: [http://arxiv.org/pdf/2505.14212v1](http://arxiv.org/pdf/2505.14212v1)**

> **作者:** Sizhe Yuen; Ting Su; Ziyang Wang; Yali Du; Adam J. Sobey
>
> **摘要:** A question-answering (QA) system is to search suitable answers within a knowledge base. Current QA systems struggle with queries requiring complex reasoning or real-time knowledge integration. They are often supplemented with retrieval techniques on a data source such as Retrieval-Augmented Generation (RAG). However, RAG continues to face challenges in handling complex reasoning and logical connections between multiple sources of information. A novel approach for enhancing Large Language Models (LLMs) in knowledge-intensive QA tasks is presented through the automated generation of context-based QA pairs. This methodology leverages LLMs to create fine-tuning data, reducing reliance on human labelling and improving model comprehension and reasoning capabilities. The proposed system includes an automated QA generator and a model fine-tuner, evaluated using perplexity, ROUGE, BLEU, and BERTScore. Comprehensive experiments demonstrate improvements in logical coherence and factual accuracy, with implications for developing adaptable Artificial Intelligence (AI) systems. Mistral-7b-v0.3 outperforms Llama-3-8b with BERT F1, BLEU, and ROUGE scores 0.858, 0.172, and 0.260 of for the LLM generated QA pairs compared to scores of 0.836, 0.083, and 0.139 for the human annotated QA pairs.
>
---
#### [new 032] Noise Injection Systemically Degrades Large Language Model Safety Guardrails
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型安全机制的鲁棒性，探究噪声对安全护栏的影响。通过向多模型注入高斯噪声，发现噪声显著提升有害输出率（最高27%），深层安全微调无额外防护，但推理能力未受损，揭示现有安全技术漏洞，建议采用强化学习等方法提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.13500v1](http://arxiv.org/pdf/2505.13500v1)**

> **作者:** Prithviraj Singh Shahani; Matthias Scheutz
>
> **备注:** 9 pages,3 figures
>
> **摘要:** Safety guardrails in large language models (LLMs) are a critical component in preventing harmful outputs. Yet, their resilience under perturbation remains poorly understood. In this paper, we investigate the robustness of safety fine-tuning in LLMs by systematically injecting Gaussian noise into model activations. We show across multiple open-weight models that (1) Gaussian noise raises harmful-output rates (p < 0.001) by up to 27%, (2) that deeper safety fine-tuning affords no extra protection, and (3) that chain-of-thought reasoning remains largely intact. The findings reveal critical vulnerabilities in current safety alignment techniques and highlight the potential of reasoning-based and reinforcement learning approaches as promising direction for developing more robust AI safety systems. These results have important implications for real-world deployment of LLMs in safety-critical applications as these results imply that widely-deployed safety tuning methods can fail even without adversarial prompts.
>
---
#### [new 033] Exploring Graph Representations of Logical Forms for Language Modeling
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文提出基于逻辑形式的图结构语言模型（LFLMs），旨在提升数据效率。通过构建GFoLDS原型，证明其利用逻辑形式内置的语言知识可高效学习复杂模式，在下游任务中显著优于文本模型，且性能可随规模扩展，为低数据需求的语言建模提供新方向。**

- **链接: [http://arxiv.org/pdf/2505.14523v1](http://arxiv.org/pdf/2505.14523v1)**

> **作者:** Michael Sullivan
>
> **备注:** To be published in ACL 2025 Findings
>
> **摘要:** We make the case for language models over logical forms (LFLMs), arguing that such models are more data-efficient than their textual counterparts. To that end, we introduce the Graph-based Formal-Logical Distributional Semantics (GFoLDS) prototype, a pretrained LM over graph representations of logical forms, as a proof-of-concept of LFLMs. Using GFoLDS, we present strong experimental evidence that LFLMs can leverage the built-in, basic linguistic knowledge inherent in such models to immediately begin learning more complex patterns. On downstream tasks, we show that GFoLDS vastly outperforms textual, transformer LMs pretrained on similar amounts of data, indicating that LFLMs can learn with substantially less data than models over plain text. Furthermore, we show that the performance of this model is likely to scale with additional parameters and pretraining data, suggesting the viability of LFLMs in real-world applications.
>
---
#### [new 034] Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识蒸馏任务，旨在解决推理痕迹与最终答案准确性脱节的问题。通过规则分解问题为子任务（如分类、检索），生成可评估的结构化痕迹，实验发现正确痕迹不确保正确答案，揭示中间过程与结果关联性低，挑战了基于痕迹优化小模型的假设。**

- **链接: [http://arxiv.org/pdf/2505.13792v1](http://arxiv.org/pdf/2505.13792v1)**

> **作者:** Siddhant Bhambri; Upasana Biswas; Subbarao Kambhampati
>
> **备注:** 10 pages
>
> **摘要:** Question Answering (QA) poses a challenging and critical problem, particularly in today's age of interactive dialogue systems such as ChatGPT, Perplexity, Microsoft Copilot, etc. where users demand both accuracy and transparency in the model's outputs. Since smaller language models (SLMs) are computationally more efficient but often under-perform compared to larger models, Knowledge Distillation (KD) methods allow for finetuning these smaller models to improve their final performance. Lately, the intermediate tokens or the so called `reasoning' traces produced by Chain-of-Thought (CoT) or by reasoning models such as DeepSeek R1 are used as a training signal for KD. However, these reasoning traces are often verbose and difficult to interpret or evaluate. In this work, we aim to address the challenge of evaluating the faithfulness of these reasoning traces and their correlation with the final performance. To this end, we employ a KD method leveraging rule-based problem decomposition. This approach allows us to break down complex queries into structured sub-problems, generating interpretable traces whose correctness can be readily evaluated, even at inference time. Specifically, we demonstrate this approach on Open Book QA, decomposing the problem into a Classification step and an Information Retrieval step, thereby simplifying trace evaluation. Our SFT experiments with correct and incorrect traces on the CoTemp QA, Microsoft Machine Reading Comprehension QA, and Facebook bAbI QA datasets reveal the striking finding that correct traces do not necessarily imply that the model outputs the correct final solution. Similarly, we find a low correlation between correct final solutions and intermediate trace correctness. These results challenge the implicit assumption behind utilizing reasoning traces for improving SLMs' final performance via KD.
>
---
#### [new 035] From Templates to Natural Language: Generalization Challenges in Instruction-Tuned LLMs for Spatial Reasoning
- **分类: cs.CL**

- **简介: 该论文研究指令调优LLMs在空间推理任务中的泛化挑战，聚焦模型将合成指令推广到人类自然语言的难题。通过在2.5D网格物体排列任务中，用合成指令微调模型并测试其在合成/人类指令上的表现，揭示模型在复杂任务中的性能下降，并分析泛化差距。**

- **链接: [http://arxiv.org/pdf/2505.14425v1](http://arxiv.org/pdf/2505.14425v1)**

> **作者:** Chalamalasetti Kranti; Sherzod Hakimov; David Schlangen
>
> **备注:** 4 pages
>
> **摘要:** Instruction-tuned large language models (LLMs) have shown strong performance on a variety of tasks; however, generalizing from synthetic to human-authored instructions in grounded environments remains a challenge for them. In this work, we study generalization challenges in spatial grounding tasks where models interpret and translate instructions for building object arrangements on a $2.5$D grid. We fine-tune LLMs using only synthetic instructions and evaluate their performance on a benchmark dataset containing both synthetic and human-written instructions. Our results reveal that while models generalize well on simple tasks, their performance degrades significantly on more complex tasks. We present a detailed error analysis of the gaps in instruction generalization.
>
---
#### [new 036] IRLBench: A Multi-modal, Culturally Grounded, Parallel Irish-English Benchmark for Open-Ended LLM Reasoning Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出IRLBench，一个多模态、文化相关的爱尔兰-英语双语基准，用于评估LLM在低资源语言（如濒危爱尔兰语）的推理能力。针对现有基准的文本单一性、文化偏见及低资源支持不足问题，其基于爱尔兰高考设计12个学科的长文本生成任务，通过官方评分方案评估模型正确率与语言准确性，揭示模型在爱尔兰语表现显著低于英语（55.8% vs 76.2%），并开源数据与工具促进多语言AI研究。**

- **链接: [http://arxiv.org/pdf/2505.13498v1](http://arxiv.org/pdf/2505.13498v1)**

> **作者:** Khanh-Tung Tran; Barry O'Sullivan; Hoang D. Nguyen
>
> **摘要:** Recent advances in Large Language Models (LLMs) have demonstrated promising knowledge and reasoning abilities, yet their performance in multilingual and low-resource settings remains underexplored. Existing benchmarks often exhibit cultural bias, restrict evaluation to text-only, rely on multiple-choice formats, and, more importantly, are limited for extremely low-resource languages. To address these gaps, we introduce IRLBench, presented in parallel English and Irish, which is considered definitely endangered by UNESCO. Our benchmark consists of 12 representative subjects developed from the 2024 Irish Leaving Certificate exams, enabling fine-grained analysis of model capabilities across domains. By framing the task as long-form generation and leveraging the official marking scheme, it does not only support a comprehensive evaluation of correctness but also language fidelity. Our extensive experiments of leading closed-source and open-source LLMs reveal a persistent performance gap between English and Irish, in which models produce valid Irish responses less than 80\% of the time, and answer correctly 55.8\% of the time compared to 76.2\% in English for the best-performing model. We release IRLBench (https://huggingface.co/datasets/ReliableAI/IRLBench) and an accompanying evaluation codebase (https://github.com/ReML-AI/IRLBench) to enable future research on robust, culturally aware multilingual AI development.
>
---
#### [new 037] AutoRev: Automatic Peer Review System for Academic Research Papers
- **分类: cs.CL**

- **简介: 该论文提出AutoRev，属于学术论文自动评审任务，解决长输入导致的计算与性能问题。其通过图结构表示文档并提取关键段落，生成评审，效果超现有方法58.72%。**

- **链接: [http://arxiv.org/pdf/2505.14376v1](http://arxiv.org/pdf/2505.14376v1)**

> **作者:** Maitreya Prafulla Chitale; Ketaki Mangesh Shetye; Harshit Gupta; Manav Chaudhary; Vasudeva Varma
>
> **摘要:** Generating a review for an academic research paper is a complex task that requires a deep understanding of the document's content and the interdependencies between its sections. It demands not only insight into technical details but also an appreciation of the paper's overall coherence and structure. Recent methods have predominantly focused on fine-tuning large language models (LLMs) to address this challenge. However, they often overlook the computational and performance limitations imposed by long input token lengths. To address this, we introduce AutoRev, an Automatic Peer Review System for Academic Research Papers. Our novel framework represents an academic document as a graph, enabling the extraction of the most critical passages that contribute significantly to the review. This graph-based approach demonstrates effectiveness for review generation and is potentially adaptable to various downstream tasks, such as question answering, summarization, and document representation. When applied to review generation, our method outperforms SOTA baselines by an average of 58.72% across all evaluation metrics. We hope that our work will stimulate further research in applying graph-based extraction techniques to other downstream tasks in NLP. We plan to make our code public upon acceptance.
>
---
#### [new 038] Legal Rule Induction: Towards Generalizable Principle Discovery from Analogous Judicial Precedents
- **分类: cs.CL**

- **简介: 该论文提出法律规则归纳（LRI）任务，旨在从类案中提取可泛化的法律规则，解决模型在规则推理和符号推理上的不足。构建首个含5,121案例集的基准数据集，实验表明训练后模型能更好捕捉规则模式，但大模型仍存在过度泛化和幻觉问题。**

- **链接: [http://arxiv.org/pdf/2505.14104v1](http://arxiv.org/pdf/2505.14104v1)**

> **作者:** Wei Fan; Tianshi Zheng; Yiran Hu; Zheye Deng; Weiqi Wang; Baixuan Xu; Chunyang Li; Haoran Li; Weixing Shen; Yangqiu Song
>
> **备注:** Under Review
>
> **摘要:** Legal rules encompass not only codified statutes but also implicit adjudicatory principles derived from precedents that contain discretionary norms, social morality, and policy. While computational legal research has advanced in applying established rules to cases, inducing legal rules from judicial decisions remains understudied, constrained by limitations in model inference efficacy and symbolic reasoning capability. The advent of Large Language Models (LLMs) offers unprecedented opportunities for automating the extraction of such latent principles, yet progress is stymied by the absence of formal task definitions, benchmark datasets, and methodologies. To address this gap, we formalize Legal Rule Induction (LRI) as the task of deriving concise, generalizable doctrinal rules from sets of analogous precedents, distilling their shared preconditions, normative behaviors, and legal consequences. We introduce the first LRI benchmark, comprising 5,121 case sets (38,088 Chinese cases in total) for model tuning and 216 expert-annotated gold test sets. Experimental results reveal that: 1) State-of-the-art LLMs struggle with over-generalization and hallucination; 2) Training on our dataset markedly enhances LLMs capabilities in capturing nuanced rule patterns across similar cases.
>
---
#### [new 039] Technical Report on classification of literature related to children speech disorder
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文提出基于NLP的文献分类方法，解决儿童语言障碍领域系统化文献分类难题。通过检索4804篇PubMed文献，运用LDA和BERTopic模型结合定制停用词列表，识别出14个临床相关主题，评估显示模型具有良好的主题连贯性和分类效果，为自动文献综述提供基础。**

- **链接: [http://arxiv.org/pdf/2505.14242v1](http://arxiv.org/pdf/2505.14242v1)**

> **作者:** Ziang Wang; Amir Aryani
>
> **摘要:** This technical report presents a natural language processing (NLP)-based approach for systematically classifying scientific literature on childhood speech disorders. We retrieved and filtered 4,804 relevant articles published after 2015 from the PubMed database using domain-specific keywords. After cleaning and pre-processing the abstracts, we applied two topic modeling techniques - Latent Dirichlet Allocation (LDA) and BERTopic - to identify latent thematic structures in the corpus. Our models uncovered 14 clinically meaningful clusters, such as infantile hyperactivity and abnormal epileptic behavior. To improve relevance and precision, we incorporated a custom stop word list tailored to speech pathology. Evaluation results showed that the LDA model achieved a coherence score of 0.42 and a perplexity of -7.5, indicating strong topic coherence and predictive performance. The BERTopic model exhibited a low proportion of outlier topics (less than 20%), demonstrating its capacity to classify heterogeneous literature effectively. These results provide a foundation for automating literature reviews in speech-language pathology.
>
---
#### [new 040] Breaking Bad Tokens: Detoxification of LLMs Using Sparse Autoencoders
- **分类: cs.CL**

- **简介: 该论文提出基于稀疏自编码器（SAE）的因果干预方法，解决LLM生成有毒内容（如脏话、贬损言论）且易被绕过的缺陷。通过识别模型残差流中的毒性相关方向并定向调整激活，在GPT-2和Gemma-2中测试三档干预强度，最高降低20%毒性但可能影响流畅度，同时保持模型基准能力。指出宽SAE的特征分割会削弱安全性，强调解纠缠学习的重要性。**

- **链接: [http://arxiv.org/pdf/2505.14536v1](http://arxiv.org/pdf/2505.14536v1)**

> **作者:** Agam Goyal; Vedant Rathi; William Yeh; Yian Wang; Yuen Chen; Hari Sundaram
>
> **备注:** Preprint: 19 pages, 7 figures, 1 table
>
> **摘要:** Large language models (LLMs) are now ubiquitous in user-facing applications, yet they still generate undesirable toxic outputs, including profanity, vulgarity, and derogatory remarks. Although numerous detoxification methods exist, most apply broad, surface-level fixes and can therefore easily be circumvented by jailbreak attacks. In this paper we leverage sparse autoencoders (SAEs) to identify toxicity-related directions in the residual stream of models and perform targeted activation steering using the corresponding decoder vectors. We introduce three tiers of steering aggressiveness and evaluate them on GPT-2 Small and Gemma-2-2B, revealing trade-offs between toxicity reduction and language fluency. At stronger steering strengths, these causal interventions surpass competitive baselines in reducing toxicity by up to 20%, though fluency can degrade noticeably on GPT-2 Small depending on the aggressiveness. Crucially, standard NLP benchmark scores upon steering remain stable, indicating that the model's knowledge and general abilities are preserved. We further show that feature-splitting in wider SAEs hampers safety interventions, underscoring the importance of disentangled feature learning. Our findings highlight both the promise and the current limitations of SAE-based causal interventions for LLM detoxification, further suggesting practical guidelines for safer language-model deployment.
>
---
#### [new 041] AUTOLAW: Enhancing Legal Compliance in Large Language Models via Case Law Generation and Jury-Inspired Deliberation
- **分类: cs.CL**

- **简介: 该论文属于法律合规检测任务，旨在解决现有方法无法灵活适应区域法律差异的问题。提出AutoLaw框架，通过生成对抗案例法数据并模拟陪审团审议过程，动态提升LLM的法律合规性检测精度，经多基准测试验证有效。**

- **链接: [http://arxiv.org/pdf/2505.14015v1](http://arxiv.org/pdf/2505.14015v1)**

> **作者:** Tai D. Nguyen; Long H. Pham; Jun Sun
>
> **摘要:** The rapid advancement of domain-specific large language models (LLMs) in fields like law necessitates frameworks that account for nuanced regional legal distinctions, which are critical for ensuring compliance and trustworthiness. Existing legal evaluation benchmarks often lack adaptability and fail to address diverse local contexts, limiting their utility in dynamically evolving regulatory landscapes. To address these gaps, we propose AutoLaw, a novel violation detection framework that combines adversarial data generation with a jury-inspired deliberation process to enhance legal compliance of LLMs. Unlike static approaches, AutoLaw dynamically synthesizes case law to reflect local regulations and employs a pool of LLM-based "jurors" to simulate judicial decision-making. Jurors are ranked and selected based on synthesized legal expertise, enabling a deliberation process that minimizes bias and improves detection accuracy. Evaluations across three benchmarks: Law-SG, Case-SG (legality), and Unfair-TOS (policy), demonstrate AutoLaw's effectiveness: adversarial data generation improves LLM discrimination, while the jury-based voting strategy significantly boosts violation detection rates. Our results highlight the framework's ability to adaptively probe legal misalignments and deliver reliable, context-aware judgments, offering a scalable solution for evaluating and enhancing LLMs in legally sensitive applications.
>
---
#### [new 042] Towards Rehearsal-Free Continual Relation Extraction: Capturing Within-Task Variance with Adaptive Prompting
- **分类: cs.CL**

- **简介: 该论文聚焦持续关系抽取（CRE），旨在解决无回溯 continual learning 中任务识别不准确、遗忘严重及处理任务内/间变异的挑战。提出WAVE++方法，采用任务专用提示池、标签描述增强及生成模型巩固知识，无需存储数据，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.13944v1](http://arxiv.org/pdf/2505.13944v1)**

> **作者:** Bao-Ngoc Dao; Quang Nguyen; Luyen Ngo Dinh; Minh Le; Nam Le; Linh Ngo Van
>
> **摘要:** Memory-based approaches have shown strong performance in Continual Relation Extraction (CRE). However, storing examples from previous tasks increases memory usage and raises privacy concerns. Recently, prompt-based methods have emerged as a promising alternative, as they do not rely on storing past samples. Despite this progress, current prompt-based techniques face several core challenges in CRE, particularly in accurately identifying task identities and mitigating catastrophic forgetting. Existing prompt selection strategies often suffer from inaccuracies, lack robust mechanisms to prevent forgetting in shared parameters, and struggle to handle both cross-task and within-task variations. In this paper, we propose WAVE++, a novel approach inspired by the connection between prefix-tuning and mixture of experts. Specifically, we introduce task-specific prompt pools that enhance flexibility and adaptability across diverse tasks while avoiding boundary-spanning risks; this design more effectively captures variations within each task and across tasks. To further refine relation classification, we incorporate label descriptions that provide richer, more global context, enabling the model to better distinguish among different relations. We also propose a training-free mechanism to improve task prediction during inference. Moreover, we integrate a generative model to consolidate prior knowledge within the shared parameters, thereby removing the need for explicit data storage. Extensive experiments demonstrate that WAVE++ outperforms state-of-the-art prompt-based and rehearsal-based methods, offering a more robust solution for continual relation extraction. Our code is publicly available at https://github.com/PiDinosauR2804/WAVE-CRE-PLUS-PLUS.
>
---
#### [new 043] UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）终身编辑任务，解决大规模持续知识更新中的效率、资源消耗与性能保持问题。提出UltraEdit方法，通过轻量线性运算实现无需训练、主题限制及额外内存的快速参数调整，并采用终身归一化策略适应数据分布变化，在24GB显存GPU上实现7倍速编辑和百万级高精度更新。**

- **链接: [http://arxiv.org/pdf/2505.14679v1](http://arxiv.org/pdf/2505.14679v1)**

> **作者:** Xiaojie Gu; Guangxu Chen; Jungang Li; Jia-Chen Gu; Xuming Hu; Kai Zhang
>
> **摘要:** Lifelong learning enables large language models (LLMs) to adapt to evolving information by continually updating their internal knowledge. An ideal system should support efficient, wide-ranging updates while preserving existing capabilities and ensuring reliable deployment. Model editing stands out as a promising solution for this goal, offering a focused and efficient way to revise a model's internal knowledge. Although recent paradigms have made notable progress, they often struggle to meet the demands of practical lifelong adaptation at scale. To bridge this gap, we propose ULTRAEDIT-a fundamentally new editing solution that is training-, subject- and memory-free, making it particularly well-suited for ultra-scalable, real-world lifelong model editing. ULTRAEDIT performs editing through a self-contained process that relies solely on lightweight linear algebra operations to compute parameter shifts, enabling fast and consistent parameter modifications with minimal overhead. To improve scalability in lifelong settings, ULTRAEDIT employs a lifelong normalization strategy that continuously updates feature statistics across turns, allowing it to adapt to distributional shifts and maintain consistency over time. ULTRAEDIT achieves editing speeds over 7x faster than the previous state-of-the-art method-which was also the fastest known approach-while consuming less than 1/3 the VRAM, making it the only method currently capable of editing a 7B LLM on a 24GB consumer-grade GPU. Furthermore, we construct ULTRAEDITBENCH-the largest dataset in the field to date, with over 2M editing pairs-and demonstrate that our method supports up to 1M edits while maintaining high accuracy. Comprehensive experiments on four datasets and six models show that ULTRAEDIT consistently achieves superior performance across diverse model editing scenarios. Our code is available at: https://github.com/XiaojieGu/UltraEdit.
>
---
#### [new 044] Improved Methods for Model Pruning and Knowledge Distillation
- **分类: cs.CL; cs.CE**

- **简介: 该论文属于模型优化任务，针对现有剪枝方法导致性能下降或需大量重训的问题，提出MAMA剪枝方法，结合权重/偏置固定和GRPO奖励指标，实现高效模型压缩同时保持性能，实验显示其在多任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14052v1](http://arxiv.org/pdf/2505.14052v1)**

> **作者:** Wei Jiang; Anying Fu; Youling Zhang
>
> **摘要:** Model pruning is a performance optimization technique for large language models like R1 or o3-mini. However, existing pruning methods often lead to significant performance degradation or require extensive retraining and fine-tuning. This technique aims to identify and remove neurons, connections unlikely leading to the contribution during the human-computer interaction phase. Our goal is to obtain a much smaller and faster knowledge distilled model that can quickly generate content almost as good as those of the unpruned ones. We propose MAMA Pruning, short for Movement and Magnitude Analysis, an improved pruning method that effectively reduces model size and computational complexity while maintaining performance comparable to the original unpruned model even at extreme pruned levels. The improved method is based on weights, bias fixed in the pre-training phase and GRPO rewards verified during the post-training phase as our novel pruning indicators. Preliminary experimental results show that our method outperforms and be comparable to state-of-the-art methods across various pruning levels and different downstream computational linguistics tasks.
>
---
#### [new 045] Invisible Entropy: Towards Safe and Efficient Low-Entropy LLM Watermarking
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于LLM水印任务，解决低熵场景下传统水印方法依赖原模型导致的高计算成本、延迟及模型泄露风险。提出Invisible Entropy（IE），通过轻量特征提取器与熵标签预测下一词熵值，结合自适应阈值调整，在低熵时减少水印干扰，实现参数减少99%且性能领先。**

- **链接: [http://arxiv.org/pdf/2505.14112v1](http://arxiv.org/pdf/2505.14112v1)**

> **作者:** Tianle Gu; Zongqi Wang; Kexin Huang; Yuanqi Yao; Xiangliang Zhang; Yujiu Yang; Xiuying Chen
>
> **摘要:** Logit-based LLM watermarking traces and verifies AI-generated content by maintaining green and red token lists and increasing the likelihood of green tokens during generation. However, it fails in low-entropy scenarios, where predictable outputs make green token selection difficult without disrupting natural text flow. Existing approaches address this by assuming access to the original LLM to calculate entropy and selectively watermark high-entropy tokens. However, these methods face two major challenges: (1) high computational costs and detection delays due to reliance on the original LLM, and (2) potential risks of model leakage. To address these limitations, we propose Invisible Entropy (IE), a watermarking paradigm designed to enhance both safety and efficiency. Instead of relying on the original LLM, IE introduces a lightweight feature extractor and an entropy tagger to predict whether the entropy of the next token is high or low. Furthermore, based on theoretical analysis, we develop a threshold navigator that adaptively sets entropy thresholds. It identifies a threshold where the watermark ratio decreases as the green token count increases, enhancing the naturalness of the watermarked text and improving detection robustness. Experiments on HumanEval and MBPP datasets demonstrate that IE reduces parameter size by 99\% while achieving performance on par with state-of-the-art methods. Our work introduces a safe and efficient paradigm for low-entropy watermarking. https://github.com/Carol-gutianle/IE https://huggingface.co/datasets/Carol0110/IE-Tagger
>
---
#### [new 046] MultiHal: Multilingual Dataset for Knowledge-Graph Grounded Evaluation of LLM Hallucinations
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）幻觉评估任务，旨在解决现有基准依赖非结构化数据且缺乏多语言支持的问题。提出MultiHal——基于知识图谱的多语言多跳基准，通过筛选14万条KG路径得到2.59万高质量数据，验证了KG整合可提升模型事实性，推动图谱辅助的幻觉缓解研究。**

- **链接: [http://arxiv.org/pdf/2505.14101v1](http://arxiv.org/pdf/2505.14101v1)**

> **作者:** Ernests Lavrinovics; Russa Biswas; Katja Hose; Johannes Bjerva
>
> **摘要:** Large Language Models (LLMs) have inherent limitations of faithfulness and factuality, commonly referred to as hallucinations. Several benchmarks have been developed that provide a test bed for factuality evaluation within the context of English-centric datasets, while relying on supplementary informative context like web links or text passages but ignoring the available structured factual resources. To this end, Knowledge Graphs (KGs) have been identified as a useful aid for hallucination mitigation, as they provide a structured way to represent the facts about entities and their relations with minimal linguistic overhead. We bridge the lack of KG paths and multilinguality for factual language modeling within the existing hallucination evaluation benchmarks and propose a KG-based multilingual, multihop benchmark called \textbf{MultiHal} framed for generative text evaluation. As part of our data collection pipeline, we mined 140k KG-paths from open-domain KGs, from which we pruned noisy KG-paths, curating a high-quality subset of 25.9k. Our baseline evaluation shows an absolute scale increase by approximately 0.12 to 0.36 points for the semantic similarity score in KG-RAG over vanilla QA across multiple languages and multiple models, demonstrating the potential of KG integration. We anticipate MultiHal will foster future research towards several graph-based hallucination mitigation and fact-checking tasks.
>
---
#### [new 047] CtrlDiff: Boosting Large Diffusion Language Models with Dynamic Block Prediction and Controllable Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CtrlDiff框架，针对扩散语言模型固定长度输出和可控性差的问题，通过动态块划分（强化学习自适应调整块大小）与分类器引导控制机制，结合自回归与扩散模型优势，在保持并行生成的同时实现灵活长度及可控文本生成，缩小与自回归模型的性能差距。**

- **链接: [http://arxiv.org/pdf/2505.14455v1](http://arxiv.org/pdf/2505.14455v1)**

> **作者:** Chihan Huang; Hao Tang
>
> **摘要:** Although autoregressive models have dominated language modeling in recent years, there has been a growing interest in exploring alternative paradigms to the conventional next-token prediction framework. Diffusion-based language models have emerged as a compelling alternative due to their powerful parallel generation capabilities and inherent editability. However, these models are often constrained by fixed-length generation. A promising direction is to combine the strengths of both paradigms, segmenting sequences into blocks, modeling autoregressive dependencies across blocks while leveraging discrete diffusion to estimate the conditional distribution within each block given the preceding context. Nevertheless, their practical application is often hindered by two key limitations: rigid fixed-length outputs and a lack of flexible control mechanisms. In this work, we address the critical limitations of fixed granularity and weak controllability in current large diffusion language models. We propose CtrlDiff, a dynamic and controllable semi-autoregressive framework that adaptively determines the size of each generation block based on local semantics using reinforcement learning. Furthermore, we introduce a classifier-guided control mechanism tailored to discrete diffusion, which significantly reduces computational overhead while facilitating efficient post-hoc conditioning without retraining. Extensive experiments demonstrate that CtrlDiff sets a new standard among hybrid diffusion models, narrows the performance gap to state-of-the-art autoregressive approaches, and enables effective conditional text generation across diverse tasks.
>
---
#### [new 048] DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models
- **分类: cs.CL**

- **简介: 该论文属于大模型推理效率优化任务，旨在解决大型推理模型(LRM)因冗长推理链导致的低效问题。提出DRP框架，通过教师模型进行技能感知的步骤分解与内容剪枝，并蒸馏精简路径至学生模型，提升推理效率与精度。实验显示其在数学任务中显著减少token使用且提升准确率。**

- **链接: [http://arxiv.org/pdf/2505.13975v1](http://arxiv.org/pdf/2505.13975v1)**

> **作者:** Yuxuan Jiang; Dawei Li; Frank Ferraro
>
> **摘要:** While Large Reasoning Models (LRMs) have demonstrated success in complex reasoning tasks through long chain-of-thought (CoT) reasoning, their inference often involves excessively verbose reasoning traces, resulting in substantial inefficiency. To address this, we propose Distilled Reasoning Pruning (DRP), a hybrid framework that combines inference-time pruning with tuning-based distillation, two widely used strategies for efficient reasoning. DRP uses a teacher model to perform skill-aware step decomposition and content pruning, and then distills the pruned reasoning paths into a student model, enabling it to reason both efficiently and accurately. Across several challenging mathematical reasoning datasets, we find that models trained with DRP achieve substantial improvements in token efficiency without sacrificing accuracy. Specifically, DRP reduces average token usage on GSM8K from 917 to 328 while improving accuracy from 91.7% to 94.1%, and achieves a 43% token reduction on AIME with no performance drop. Further analysis shows that aligning the reasoning structure of training CoTs with the student's reasoning capacity is critical for effective knowledge transfer and performance gains.
>
---
#### [new 049] Language Models Optimized to Fool Detectors Still Have a Distinct Style (And How to Change It)
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究机器生成文本检测的鲁棒性，针对优化以欺骗检测器的语言模型，发现其保留独特风格特征，提出基于风格的检测方法仍有效；进一步设计新攻击策略缩小人机风格差异，但多样本下仍可区分，并开发AURA指标量化分布差异，强调不应依赖检测技术。**

- **链接: [http://arxiv.org/pdf/2505.14608v1](http://arxiv.org/pdf/2505.14608v1)**

> **作者:** Rafael Rivera Soto; Barry Chen; Nicholas Andrews
>
> **摘要:** Despite considerable progress in the development of machine-text detectors, it has been suggested that the problem is inherently hard, and therefore, that stakeholders should proceed under the assumption that machine-generated text cannot be reliably detected as such. We examine a recent such claim by Nicks et al. (2024) regarding the ease with which language models can be optimized to degrade the performance of machine-text detectors, including detectors not specifically optimized against. We identify a feature space$\unicode{x2013}$the stylistic feature space$\unicode{x2013}$that is robust to such optimization, and show that it may be used to reliably detect samples from language models optimized to prevent detection. Furthermore, we show that even when models are explicitly optimized against stylistic detectors, detection performance remains surprisingly unaffected. We then seek to understand if stylistic detectors are inherently more robust. To study this question, we explore a new paraphrasing approach that simultaneously aims to close the gap between human writing and machine writing in stylistic feature space while avoiding detection using traditional features. We show that when only a single sample is available for detection, this attack is universally effective across all detectors considered, including those that use writing style. However, as the number of samples available for detection grows, the human and machine distributions become distinguishable. This observation encourages us to introduce AURA, a metric that estimates the overlap between human and machine-generated distributions by analyzing how detector performance improves as more samples become available. Overall, our findings underscore previous recommendations to avoid reliance on machine-text detection.
>
---
#### [new 050] Think-J: Learning to Think for Generative LLM-as-a-Judge
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于生成式LLM-as-a-Judge优化任务，旨在提升大语言模型对生成回复的自动评判能力。针对现有方法效果不足的问题，提出Think-J框架：先用少量标注数据训练初始判断模型，再通过离线/在线强化学习优化评判思维路径，最终提升评估性能且无需额外标注。**

- **链接: [http://arxiv.org/pdf/2505.14268v1](http://arxiv.org/pdf/2505.14268v1)**

> **作者:** Hui Huang; Yancheng He; Hongli Zhou; Rui Zhang; Wei Liu; Weixun Wang; Wenbo Su; Bo Zheng; Jiaheng Liu
>
> **备注:** 16 pages, 14 figures
>
> **摘要:** LLM-as-a-Judge refers to the automatic modeling of preferences for responses generated by Large Language Models (LLMs), which is of significant importance for both LLM evaluation and reward modeling. Although generative LLMs have made substantial progress in various tasks, their performance as LLM-Judge still falls short of expectations. In this work, we propose Think-J, which improves generative LLM-as-a-Judge by learning how to think. We first utilized a small amount of curated data to develop the model with initial judgment thinking capabilities. Subsequently, we optimize the judgment thinking traces based on reinforcement learning (RL). We propose two methods for judgment thinking optimization, based on offline and online RL, respectively. The offline RL requires training a critic model to construct positive and negative examples for learning. The online method defines rule-based reward as feedback for optimization. Experimental results showed that our approach can significantly enhance the evaluation capability of generative LLM-Judge, surpassing both generative and classifier-based LLM-Judge without requiring extra human annotations.
>
---
#### [new 051] Mechanistic Fine-tuning for In-context Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于ICL（In-context Learning）机制性微调任务，旨在解决传统端到端微调计算成本高的问题。提出ABFT方法，通过优化注意力分数聚焦正确标签、抑制错误标签，提升模型性能与效率，仅需0.01%数据成本。实验显示其在多模型/数据集上表现更优，揭示ICL数据隐式促进诱导头形成。**

- **链接: [http://arxiv.org/pdf/2505.14233v1](http://arxiv.org/pdf/2505.14233v1)**

> **作者:** Hakaze Cho; Peng Luo; Mariko Kato; Rin Kaenbyou; Naoya Inoue
>
> **备注:** 28 pages, 31 figures, 6 tables
>
> **摘要:** In-context Learning (ICL) utilizes structured demonstration-query inputs to induce few-shot learning on Language Models (LMs), which are not originally pre-trained on ICL-style data. To bridge the gap between ICL and pre-training, some approaches fine-tune LMs on large ICL-style datasets by an end-to-end paradigm with massive computational costs. To reduce such costs, in this paper, we propose Attention Behavior Fine-Tuning (ABFT), utilizing the previous findings on the inner mechanism of ICL, building training objectives on the attention scores instead of the final outputs, to force the attention scores to focus on the correct label tokens presented in the context and mitigate attention scores from the wrong label tokens. Our experiments on 9 modern LMs and 8 datasets empirically find that ABFT outperforms in performance, robustness, unbiasedness, and efficiency, with only around 0.01% data cost compared to the previous methods. Moreover, our subsequent analysis finds that the end-to-end training objective contains the ABFT objective, suggesting the implicit bias of ICL-style data to the emergence of induction heads. Our work demonstrates the possibility of controlling specific module sequences within LMs to improve their behavior, opening up the future application of mechanistic interpretability.
>
---
#### [new 052] Through a Compressed Lens: Investigating the Impact of Quantization on LLM Explainability and Interpretability
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究量化对LLM可解释性与可理解性的影响。实验三种量化方法结合两种解释方法及两种分析技术，并通过用户研究，发现量化效果因配置不同而异，可能提升或降低透明性，强调部署透明关键应用需谨慎。**

- **链接: [http://arxiv.org/pdf/2505.13963v1](http://arxiv.org/pdf/2505.13963v1)**

> **作者:** Qianli Wang; Mingyang Wang; Nils Feldhus; Simon Ostermann; Yuan Cao; Hinrich Schütze; Sebastian Möller; Vera Schmitt
>
> **备注:** In submission
>
> **摘要:** Quantization methods are widely used to accelerate inference and streamline the deployment of large language models (LLMs). While prior research has extensively investigated the degradation of various LLM capabilities due to quantization, its effects on model explainability and interpretability, which are crucial for understanding decision-making processes, remain unexplored. To address this gap, we conduct comprehensive experiments using three common quantization techniques at distinct bit widths, in conjunction with two explainability methods, counterfactual examples and natural language explanations, as well as two interpretability approaches, knowledge memorization analysis and latent multi-hop reasoning analysis. We complement our analysis with a thorough user study, evaluating selected explainability methods. Our findings reveal that, depending on the configuration, quantization can significantly impact model explainability and interpretability. Notably, the direction of this effect is not consistent, as it strongly depends on (1) the quantization method, (2) the explainability or interpretability approach, and (3) the evaluation protocol. In some settings, human evaluation shows that quantization degrades explainability, while in others, it even leads to improvements. Our work serves as a cautionary tale, demonstrating that quantization can unpredictably affect model transparency. This insight has important implications for deploying LLMs in applications where transparency is a critical requirement.
>
---
#### [new 053] InfiGFusion: Graph-on-Logits Distillation via Efficient Gromov-Wasserstein for Model Fusion
- **分类: cs.CL**

- **简介: 该论文属于模型融合任务，旨在解决现有logit-based方法忽略词汇维度间语义依赖导致融合效果不佳的问题。提出InfiGFusion框架，通过构建全局协同激活图（GLD损失）显式建模跨维度交互，并设计高效Gromov-Wasserstein近似算法（O(n log n)），提升融合质量和稳定性，在多任务基准测试中表现最优。**

- **链接: [http://arxiv.org/pdf/2505.13893v1](http://arxiv.org/pdf/2505.13893v1)**

> **作者:** Yuanyi Wang; Zhaoyi Yan; Yiming Zhang; Qi Zhou; Yanggan Gu; Fei Wu; Hongxia Yang
>
> **摘要:** Recent advances in large language models (LLMs) have intensified efforts to fuse heterogeneous open-source models into a unified system that inherits their complementary strengths. Existing logit-based fusion methods maintain inference efficiency but treat vocabulary dimensions independently, overlooking semantic dependencies encoded by cross-dimension interactions. These dependencies reflect how token types interact under a model's internal reasoning and are essential for aligning models with diverse generation behaviors. To explicitly model these dependencies, we propose \textbf{InfiGFusion}, the first structure-aware fusion framework with a novel \textit{Graph-on-Logits Distillation} (GLD) loss. Specifically, we retain the top-$k$ logits per output and aggregate their outer products across sequence positions to form a global co-activation graph, where nodes represent vocabulary channels and edges quantify their joint activations. To ensure scalability and efficiency, we design a sorting-based closed-form approximation that reduces the original $O(n^4)$ cost of Gromov-Wasserstein distance to $O(n \log n)$, with provable approximation guarantees. Experiments across multiple fusion settings show that GLD consistently improves fusion quality and stability. InfiGFusion outperforms SOTA models and fusion baselines across 11 benchmarks spanning reasoning, coding, and mathematics. It shows particular strength in complex reasoning tasks, with +35.6 improvement on Multistep Arithmetic and +37.06 on Causal Judgement over SFT, demonstrating superior multi-step and relational inference.
>
---
#### [new 054] ThinkSwitcher: When to Think Hard, When to Think Fast
- **分类: cs.CL**

- **简介: 该论文属于优化大型推理模型（LRMs）计算效率的任务，旨在解决其在简单任务中过度思考导致的资源浪费问题。提出ThinkSwitcher框架，通过轻量级切换模块根据任务复杂度动态选择短/长链推理模式，实验证明可降低20-30%计算成本同时保持高精度。**

- **链接: [http://arxiv.org/pdf/2505.14183v1](http://arxiv.org/pdf/2505.14183v1)**

> **作者:** Guosheng Liang; Longguang Zhong; Ziyi Yang; Xiaojun Quan
>
> **摘要:** Large reasoning models (LRMs) excel at solving complex tasks by leveraging long chain-of-thought (CoT) reasoning. However, this often leads to overthinking on simple tasks, resulting in unnecessary computational overhead. We observe that LRMs inherently possess the capability for efficient short CoT reasoning, which can be reliably elicited through prompt design. To leverage this capability, we propose ThinkSwitcher, a framework that enables a single LRM to dynamically switch between short and long CoT modes based on task complexity. ThinkSwitcher introduces a lightweight switching module trained with supervision signals derived from the relative performance of each reasoning mode across tasks. Experiments on multiple reasoning benchmarks show that ThinkSwitcher reduces computational cost by 20-30% while maintaining high accuracy on complex tasks. This demonstrates the effectiveness of ThinkSwitcher as a scalable and efficient solution for unified LRM deployment.
>
---
#### [new 055] Void in Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言模型推理中层激活问题，提出L2自适应计算(LAC)检测未激活层（Void）。通过分析提示处理和生成阶段的激活差异，实验显示跳过Void层可提升性能（如Qwen2.5在MMLU准确率提升且仅用30%层），表明选择性跳过层能优化模型效率。**

- **链接: [http://arxiv.org/pdf/2505.14467v1](http://arxiv.org/pdf/2505.14467v1)**

> **作者:** Mani Shemiranifar
>
> **摘要:** Despite advances in transformer-based language models (LMs), a fundamental question remains largely unanswered: Are all layers activated during inference? We investigate this question by detecting unactivated layers (which we refer to as Voids) using a non-trainable and parameter-free adaptive computation method called L2 Adaptive Computation (LAC). We adapt LAC from its original efficiency-focused application to trace activated layers during inference. This method monitors changes in the L2-norm of activations to identify voids. We analyze layer activation in instruction-tuned LMs across two phases: Prompt Processing (PP), where we trace activated layers for each token in the input prompts, and Response Generation (RG), where we trace activated layers for each generated token. We further demonstrate that distinct layers are activated during these two phases. To show the effectiveness of our method, we evaluated three distinct instruction-tuned LMs from the Llama, Mistral, and Qwen families on three benchmarks: MMLU, GPQA Diamond, and BoolQ. For example, on MMLU with a zero-shot setting, skipping voids in Qwen2.5-7B-Instruct resulted in an improvement from 69.24 to 71.29 while the model uses only 30% of the layers. Similarly, Mistral-7B-Instruct-v0.3 on GPQA Diamond improved from 13.88 to 18.36 when using 70% of the layers during both the PP and RG phases. These results show that not all layers contribute equally during inference, and that selectively skipping most of them can improve the performance of models on certain tasks.
>
---
#### [new 056] DecIF: Improving Instruction-Following through Meta-Decomposition
- **分类: cs.CL**

- **简介: 该论文属于大型语言模型（LLM）指令遵循任务，旨在解决现有方法依赖外部资源导致的灵活性不足问题。提出DecIF框架，通过元分解技术自主生成结构化指令数据：分解指令为元信息并约束响应，检测不一致，拆分评估标准验证响应，提升数据质量。实验验证其有效性与普适性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.13990v1](http://arxiv.org/pdf/2505.13990v1)**

> **作者:** Tingfeng Hui; Pengyu Zhu; Bowen Ping; Ling Tang; Yaqi Zhang; Sen Su
>
> **备注:** Work in progress
>
> **摘要:** Instruction-following has emerged as a crucial capability for large language models (LLMs). However, existing approaches often rely on pre-existing documents or external resources to synthesize instruction-following data, which limits their flexibility and generalizability. In this paper, we introduce DecIF, a fully autonomous, meta-decomposition guided framework that generates diverse and high-quality instruction-following data using only LLMs. DecIF is grounded in the principle of decomposition. For instruction generation, we guide LLMs to iteratively produce various types of meta-information, which are then combined with response constraints to form well-structured and semantically rich instructions. We further utilize LLMs to detect and resolve potential inconsistencies within the generated instructions. Regarding response generation, we decompose each instruction into atomic-level evaluation criteria, enabling rigorous validation and the elimination of inaccurate instruction-response pairs. Extensive experiments across a wide range of scenarios and settings demonstrate DecIF's superior performance on instruction-following tasks. Further analysis highlights its strong flexibility, scalability, and generalizability in automatically synthesizing high-quality instruction data.
>
---
#### [new 057] MoMoE: Mixture of Moderation Experts Framework for AI-Assisted Online Governance
- **分类: cs.CL**

- **简介: 该论文属于AI辅助在线内容治理任务，解决现有内容审核需为每个社区单独建模且决策不透明的问题。提出MoMoE框架，通过整合社区专用专家与规范违规检测模块，结合解释功能实现跨社区可扩展、透明的审核，无需微调即达高性能，验证了轻量级可解释模型在人机协同治理中的潜力。**

- **链接: [http://arxiv.org/pdf/2505.14483v1](http://arxiv.org/pdf/2505.14483v1)**

> **作者:** Agam Goyal; Xianyang Zhan; Yilun Chen; Koustuv Saha; Eshwar Chandrasekharan
>
> **备注:** Preprint: 15 pages, 4 figures, 2 tables
>
> **摘要:** Large language models (LLMs) have shown great potential in flagging harmful content in online communities. Yet, existing approaches for moderation require a separate model for every community and are opaque in their decision-making, limiting real-world adoption. We introduce Mixture of Moderation Experts (MoMoE), a modular, cross-community framework that adds post-hoc explanations to scalable content moderation. MoMoE orchestrates four operators -- Allocate, Predict, Aggregate, Explain -- and is instantiated as seven community-specialized experts (MoMoE-Community) and five norm-violation experts (MoMoE-NormVio). On 30 unseen subreddits, the best variants obtain Micro-F1 scores of 0.72 and 0.67, respectively, matching or surpassing strong fine-tuned baselines while consistently producing concise and reliable explanations. Although community-specialized experts deliver the highest peak accuracy, norm-violation experts provide steadier performance across domains. These findings show that MoMoE yields scalable, transparent moderation without needing per-community fine-tuning. More broadly, they suggest that lightweight, explainable expert ensembles can guide future NLP and HCI research on trustworthy human-AI governance of online communities.
>
---
#### [new 058] Improve Language Model and Brain Alignment via Associative Memory
- **分类: cs.CL**

- **简介: 该论文属于神经语言模型与人类大脑活动对齐研究。旨在提升语言模型处理语音时与大脑关联区域的对齐。通过扩展文本刺激的模拟联想记忆输入，优化模型训练，并构建含1000样本的Association数据集进行监督微调，验证了联想记忆相关脑区对齐的提升效果。（99字）**

- **链接: [http://arxiv.org/pdf/2505.13844v1](http://arxiv.org/pdf/2505.13844v1)**

> **作者:** Congchi Yin; Yongpeng Zhang; Xuyun Wen; Piji Li
>
> **备注:** Accepted by Findings of ACL 2025
>
> **摘要:** Associative memory engages in the integration of relevant information for comprehension in the human cognition system. In this work, we seek to improve alignment between language models and human brain while processing speech information by integrating associative memory. After verifying the alignment between language model and brain by mapping language model activations to brain activity, the original text stimuli expanded with simulated associative memory are regarded as input to computational language models. We find the alignment between language model and brain is improved in brain regions closely related to associative memory processing. We also demonstrate large language models after specific supervised fine-tuning better align with brain response, by building the \textit{Association} dataset containing 1000 samples of stories, with instructions encouraging associative memory as input and associated content as output.
>
---
#### [new 059] ProdRev: A DNN framework for empowering customers using generative pre-trained transformers
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ProdRev框架，利用GPT-3的Curie引擎微调生成式模型，解决电商产品评论过载导致的消费者决策困难。通过生成式摘要揭示评论间真实关系，提供优缺点分析，辅助用户自主决策，属于生成式摘要与智能决策支持任务。**

- **链接: [http://arxiv.org/pdf/2505.13491v1](http://arxiv.org/pdf/2505.13491v1)**

> **作者:** Aakash Gupta; Nataraj Das
>
> **备注:** 2022 International Conference on Decision Aid Sciences and Applications (DASA)
>
> **摘要:** Following the pandemic, customers, preference for using e-commerce has accelerated. Since much information is available in multiple reviews (sometimes running in thousands) for a single product, it can create decision paralysis for the buyer. This scenario disempowers the consumer, who cannot be expected to go over so many reviews since its time consuming and can confuse them. Various commercial tools are available, that use a scoring mechanism to arrive at an adjusted score. It can alert the user to potential review manipulations. This paper proposes a framework that fine-tunes a generative pre-trained transformer to understand these reviews better. Furthermore, using "common-sense" to make better decisions. These models have more than 13 billion parameters. To fine-tune the model for our requirement, we use the curie engine from generative pre-trained transformer (GPT3). By using generative models, we are introducing abstractive summarization. Instead of using a simple extractive method of summarizing the reviews. This brings out the true relationship between the reviews and not simply copy-paste. This introduces an element of "common sense" for the user and helps them to quickly make the right decisions. The user is provided the pros and cons of the processed reviews. Thus the user/customer can take their own decisions.
>
---
#### [new 060] SAE-FiRE: Enhancing Earnings Surprise Predictions Through Sparse Autoencoder Feature Selection
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于财务预测任务，旨在通过分析冗长的财报电话会议文本提升盈利惊喜预测精度。针对文本冗余与行业术语干扰问题，提出SAE-FiRE框架，利用稀疏自编码器筛选关键金融信号并去噪，实验显示其显著优于基准模型。**

- **链接: [http://arxiv.org/pdf/2505.14420v1](http://arxiv.org/pdf/2505.14420v1)**

> **作者:** Huopu Zhang; Yanguang Liu; Mengnan Du
>
> **摘要:** Predicting earnings surprises through the analysis of earnings conference call transcripts has attracted increasing attention from the financial research community. Conference calls serve as critical communication channels between company executives, analysts, and shareholders, offering valuable forward-looking information. However, these transcripts present significant analytical challenges, typically containing over 5,000 words with substantial redundancy and industry-specific terminology that creates obstacles for language models. In this work, we propose the Sparse Autoencoder for Financial Representation Enhancement (SAE-FiRE) framework to address these limitations by extracting key information while eliminating redundancy. SAE-FiRE employs Sparse Autoencoders (SAEs) to efficiently identify patterns and filter out noises, and focusing specifically on capturing nuanced financial signals that have predictive power for earnings surprises. Experimental results indicate that the proposed method can significantly outperform comparing baselines.
>
---
#### [new 061] Creative Preference Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Creative Preference Optimization（CrPO），通过模块化整合多维度创造力指标到LLM的偏好优化目标中，解决现有方法仅片面提升多样性或特定任务创造力的问题。利用新数据集MuCE训练模型，在新颖性、多样性、质量等指标上优于GPT-4o等基线，验证方法有效性和通用性。**

- **链接: [http://arxiv.org/pdf/2505.14442v1](http://arxiv.org/pdf/2505.14442v1)**

> **作者:** Mete Ismayilzada; Antonio Laverghetta Jr.; Simone A. Luchini; Reet Patel; Antoine Bosselut; Lonneke van der Plas; Roger Beaty
>
> **备注:** 27 pages
>
> **摘要:** While Large Language Models (LLMs) have demonstrated impressive performance across natural language generation tasks, their ability to generate truly creative content-characterized by novelty, diversity, surprise, and quality-remains limited. Existing methods for enhancing LLM creativity often focus narrowly on diversity or specific tasks, failing to address creativity's multifaceted nature in a generalizable way. In this work, we propose Creative Preference Optimization (CrPO), a novel alignment method that injects signals from multiple creativity dimensions into the preference optimization objective in a modular fashion. We train and evaluate creativity-augmented versions of several models using CrPO and MuCE, a new large-scale human preference dataset spanning over 200,000 human-generated responses and ratings from more than 30 psychological creativity assessments. Our models outperform strong baselines, including GPT-4o, on both automated and human evaluations, producing more novel, diverse, and surprising generations while maintaining high output quality. Additional evaluations on NoveltyBench further confirm the generalizability of our approach. Together, our results demonstrate that directly optimizing for creativity within preference frameworks is a promising direction for advancing the creative capabilities of LLMs without compromising output quality.
>
---
#### [new 062] Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数学推理任务的链式思维调优改进任务，旨在解决专家标注数据中存在的思维跳跃问题。提出CoT Thought Leap Bridge方法，通过检测跳跃并补全中间推理步骤，构建ScaleQM+数据集训练CoT-Bridge模型，实验显示其显著提升数学推理性能（+5.87%）并增强跨领域泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.14684v1](http://arxiv.org/pdf/2505.14684v1)**

> **作者:** Haolei Xu; Yuchen Yan; Yongliang Shen; Wenqi Zhang; Guiyang Hou; Shengpei Jiang; Kaitao Song; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **摘要:** Large language models (LLMs) have achieved remarkable progress on mathemati-cal tasks through Chain-of-Thought (CoT) reasoning. However, existing mathematical CoT datasets often suffer from Thought Leaps due to experts omitting intermediate steps, which negatively impacts model learning and generalization. We propose the CoT Thought Leap Bridge Task, which aims to automatically detect leaps and generate missing intermediate reasoning steps to restore the completeness and coherence of CoT. To facilitate this, we constructed a specialized training dataset called ScaleQM+, based on the structured ScaleQuestMath dataset, and trained CoT-Bridge to bridge thought leaps. Through comprehensive experiments on mathematical reasoning benchmarks, we demonstrate that models fine-tuned on bridged datasets consistently outperform those trained on original datasets, with improvements of up to +5.87% on NuminaMath. Our approach effectively enhances distilled data (+3.02%) and provides better starting points for reinforcement learning (+3.1%), functioning as a plug-and-play module compatible with existing optimization techniques. Furthermore, CoT-Bridge demonstrate improved generalization to out-of-domain logical reasoning tasks, confirming that enhancing reasoning completeness yields broadly applicable benefits.
>
---
#### [new 063] CS-Sum: A Benchmark for Code-Switching Dialogue Summarization and the Limits of Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出CS-Sum基准，评估LLMs处理中英、泰英、马英混合语言对话摘要的能力。针对代码切换（CS）的可理解性问题，测试了十种LLMs的多种训练方法，发现尽管自动评分高，但模型存在改变对话原意的细微错误，并总结三类常见错误，强调需专用CS数据训练。**

- **链接: [http://arxiv.org/pdf/2505.13559v1](http://arxiv.org/pdf/2505.13559v1)**

> **作者:** Sathya Krishnan Suresh; Tanmay Surana; Lim Zhi Hao; Eng Siong Chng
>
> **备注:** 17 pages, 5 figures and 11 tables
>
> **摘要:** Code-switching (CS) poses a significant challenge for Large Language Models (LLMs), yet its comprehensibility remains underexplored in LLMs. We introduce CS-Sum, to evaluate the comprehensibility of CS by the LLMs through CS dialogue to English summarization. CS-Sum is the first benchmark for CS dialogue summarization across Mandarin-English (EN-ZH), Tamil-English (EN-TA), and Malay-English (EN-MS), with 900-1300 human-annotated dialogues per language pair. Evaluating ten LLMs, including open and closed-source models, we analyze performance across few-shot, translate-summarize, and fine-tuning (LoRA, QLoRA on synthetic data) approaches. Our findings show that though the scores on automated metrics are high, LLMs make subtle mistakes that alter the complete meaning of the dialogue. To this end, we introduce 3 most common type of errors that LLMs make when handling CS input. Error rates vary across CS pairs and LLMs, with some LLMs showing more frequent errors on certain language pairs, underscoring the need for specialized training on code-switched data.
>
---
#### [new 064] EmoMeta: A Multimodal Dataset for Fine-grained Emotion Classification in Chinese Metaphors
- **分类: cs.CL; cs.AI**

- **简介: 该论文构建了中文多模态隐喻情感分类数据集EmoMeta，任务为细粒度情感分类。针对多模态隐喻情感研究中中文数据稀缺及跨语言差异问题，收集5000个图文广告对，标注隐喻、领域关系及10类情感，并公开数据集以推动研究。**

- **链接: [http://arxiv.org/pdf/2505.13483v1](http://arxiv.org/pdf/2505.13483v1)**

> **作者:** Xingyuan Lu; Yuxi Liu; Dongyu Zhang; Zhiyao Wu; Jing Ren; Feng Xia
>
> **摘要:** Metaphors play a pivotal role in expressing emotions, making them crucial for emotional intelligence. The advent of multimodal data and widespread communication has led to a proliferation of multimodal metaphors, amplifying the complexity of emotion classification compared to single-mode scenarios. However, the scarcity of research on constructing multimodal metaphorical fine-grained emotion datasets hampers progress in this domain. Moreover, existing studies predominantly focus on English, overlooking potential variations in emotional nuances across languages. To address these gaps, we introduce a multimodal dataset in Chinese comprising 5,000 text-image pairs of metaphorical advertisements. Each entry is meticulously annotated for metaphor occurrence, domain relations and fine-grained emotion classification encompassing joy, love, trust, fear, sadness, disgust, anger, surprise, anticipation, and neutral. Our dataset is publicly accessible (https://github.com/DUTIR-YSQ/EmoMeta), facilitating further advancements in this burgeoning field.
>
---
#### [new 065] Reasoning Path Compression: Compressing Generation Trajectories for Efficient LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文属于大型语言模型（LLM）推理优化任务，旨在解决长推理路径导致的内存占用高和生成效率低问题。提出无训练方法RPC，通过语义稀疏性压缩KV缓存（保留高重要性片段），提升推理吞吐量，实验显示在QwQ-32B模型上加速1.6倍，精度仅降1.2%。**

- **链接: [http://arxiv.org/pdf/2505.13866v1](http://arxiv.org/pdf/2505.13866v1)**

> **作者:** Jiwon Song; Dongwon Jo; Yulhwa Kim; Jae-Joon Kim
>
> **摘要:** Recent reasoning-focused language models achieve high accuracy by generating lengthy intermediate reasoning paths before producing final answers. While this approach is effective in solving problems that require logical thinking, long reasoning paths significantly increase memory usage and throughput of token generation, limiting the practical deployment of such models. We propose Reasoning Path Compression (RPC), a training-free method that accelerates inference by leveraging the semantic sparsity of reasoning paths. RPC periodically compresses the KV cache by retaining KV cache that receive high importance score, which are computed using a selector window composed of recently generated queries. Experiments show that RPC improves generation throughput of QwQ-32B by up to 1.60$\times$ compared to the inference with full KV cache, with an accuracy drop of 1.2% on the AIME 2024 benchmark. Our findings demonstrate that semantic sparsity in reasoning traces can be effectively exploited for compression, offering a practical path toward efficient deployment of reasoning LLMs. Our code is available at https://github.com/jiwonsong-dev/ReasoningPathCompression.
>
---
#### [new 066] Mapping the Minds of LLMs: A Graph-Based Analysis of Reasoning LLM
- **分类: cs.CL**

- **简介: 论文提出图基框架分析推理LLMs的推理过程，解决其少样本提示下表现不稳定的难题。通过聚类CoT步骤构建推理图，揭示结构属性（如探索密度、分支）与准确性的关联，量化提示策略对推理路径的影响，提供评估与优化方法。**

- **链接: [http://arxiv.org/pdf/2505.13890v1](http://arxiv.org/pdf/2505.13890v1)**

> **作者:** Zhen Xiong; Yujun Cai; Zhecheng Li; Yiwei Wang
>
> **摘要:** Recent advances in test-time scaling have enabled Large Language Models (LLMs) to display sophisticated reasoning abilities via extended Chain-of-Thought (CoT) generation. Despite their potential, these Reasoning LLMs (RLMs) often demonstrate counterintuitive and unstable behaviors, such as performance degradation under few-shot prompting, that challenge our current understanding of RLMs. In this work, we introduce a unified graph-based analytical framework for better modeling the reasoning processes of RLMs. Our method first clusters long, verbose CoT outputs into semantically coherent reasoning steps, then constructs directed reasoning graphs to capture contextual and logical dependencies among these steps. Through comprehensive analysis across models and prompting regimes, we reveal that structural properties, such as exploration density, branching, and convergence ratios, strongly correlate with reasoning accuracy. Our findings demonstrate how prompting strategies substantially reshape the internal reasoning structure of RLMs, directly affecting task outcomes. The proposed framework not only enables quantitative evaluation of reasoning quality beyond conventional metrics but also provides practical insights for prompt engineering and the cognitive analysis of LLMs. Code and resources will be released to facilitate future research in this direction.
>
---
#### [new 067] Code2Logic: Game-Code-Driven Data Synthesis for Enhancing VLMs General Reasoning
- **分类: cs.CL; I.2.7; I.2.10**

- **简介: 该论文属于视觉语言模型（VLM）推理能力提升任务。针对视觉语言推理数据稀缺且标注成本高的问题，提出Code2Logic方法，利用游戏代码的逻辑结构通过LLM自动生成多模态推理数据集GameQA。该数据集助力VLM训练，使模型在7个跨领域基准测试中性能提升2.33%，验证了游戏代码驱动数据合成的有效性。**

- **链接: [http://arxiv.org/pdf/2505.13886v1](http://arxiv.org/pdf/2505.13886v1)**

> **作者:** Jingqi Tong; Jixin Tang; Hangcheng Li; Yurong Mou; Ming Zhang; Jun Zhao; Yanbo Wen; Fan Song; Jiahao Zhan; Yuyang Lu; Chaoran Tao; Zhiyuan Guo; Jizhou Yu; Tianhao Cheng; Changhao Jiang; Zhen Wang; Tao Liang; Zhihui Fei; Mingyang Wan; Guojun Ma; Weifeng Ge; Guanhua Chen; Tao Gui; Xipeng Qiu; Qi Zhang; Xuanjing Huang
>
> **备注:** 49 pages, 19 figures, submitted to NeurIPS 2025
>
> **摘要:** Visual-language Chain-of-Thought (CoT) data resources are relatively scarce compared to text-only counterparts, limiting the improvement of reasoning capabilities in Vision Language Models (VLMs). However, high-quality vision-language reasoning data is expensive and labor-intensive to annotate. To address this issue, we leverage a promising resource: game code, which naturally contains logical structures and state transition processes. Therefore, we propose Code2Logic, a novel game-code-driven approach for multimodal reasoning data synthesis. Our approach leverages Large Language Models (LLMs) to adapt game code, enabling automatic acquisition of reasoning processes and results through code execution. Using the Code2Logic approach, we developed the GameQA dataset to train and evaluate VLMs. GameQA is cost-effective and scalable to produce, challenging for state-of-the-art models, and diverse with 30 games and 158 tasks. Surprisingly, despite training solely on game data, VLMs demonstrated out of domain generalization, specifically Qwen2.5-VL-7B improving performance by 2.33\% across 7 diverse vision-language benchmarks. Our code and dataset are available at https://github.com/tongjingqi/Code2Logic.
>
---
#### [new 068] Detecting Prefix Bias in LLM-based Reward Models
- **分类: cs.CL**

- **简介: 该论文属于AI公平性研究，旨在检测LLM奖励模型中的前缀偏见问题，即查询前缀微调引发系统性偏好偏差。提出评估方法揭示种族/性别偏差，测试多数据集与模型，提出数据增强策略缓解，并强调公平数据设计必要性。**

- **链接: [http://arxiv.org/pdf/2505.13487v1](http://arxiv.org/pdf/2505.13487v1)**

> **作者:** Ashwin Kumar; Yuzi He; Aram H. Markosyan; Bobbie Chern; Imanol Arrieta-Ibarra
>
> **摘要:** Reinforcement Learning with Human Feedback (RLHF) has emerged as a key paradigm for task-specific fine-tuning of language models using human preference data. While numerous publicly available preference datasets provide pairwise comparisons of responses, the potential for biases in the resulting reward models remains underexplored. In this work, we introduce novel methods to detect and evaluate prefix bias -- a systematic shift in model preferences triggered by minor variations in query prefixes -- in LLM-based reward models trained on such datasets. We leverage these metrics to reveal significant biases in preference models across racial and gender dimensions. Our comprehensive evaluation spans diverse open-source preference datasets and reward model architectures, demonstrating susceptibility to this kind of bias regardless of the underlying model architecture. Furthermore, we propose a data augmentation strategy to mitigate these biases, showing its effectiveness in reducing the impact of prefix bias. Our findings highlight the critical need for bias-aware dataset design and evaluation in developing fair and reliable reward models, contributing to the broader discourse on fairness in AI.
>
---
#### [new 069] Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning
- **分类: cs.CL**

- **简介: 该论文属于跨语言表示对齐任务，旨在解决低资源语言缺乏双语数据导致的跨语言模型训练难题。通过对比学习图像-标题数据，隐式对齐多语言文本表示，支持未预训练语言的后加入，并用于跨语言NLU和双语检索，提供高效替代方案。**

- **链接: [http://arxiv.org/pdf/2505.13628v1](http://arxiv.org/pdf/2505.13628v1)**

> **作者:** Nathaniel Krasner; Nicholas Lanuzo; Antonios Anastasopoulos
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Multilingual alignment of sentence representations has mostly required bitexts to bridge the gap between languages. We investigate whether visual information can bridge this gap instead. Image caption datasets are very easy to create without requiring multilingual expertise, so this offers a more efficient alternative for low-resource languages. We find that multilingual image-caption alignment can implicitly align the text representations between languages, languages unseen by the encoder in pretraining can be incorporated into this alignment post-hoc, and these aligned representations are usable for cross-lingual Natural Language Understanding (NLU) and bitext retrieval.
>
---
#### [new 070] Combining the Best of Both Worlds: A Method for Hybrid NMT and LLM Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出混合NMT与LLM的翻译方法，解决LLM高成本与NMT质量不足的问题。通过设计基于源句特征的调度策略，仅在必要时调用LLM，实现在最小LLM使用下保持最优翻译效果。**

- **链接: [http://arxiv.org/pdf/2505.13554v1](http://arxiv.org/pdf/2505.13554v1)**

> **作者:** Zhanglin Wu; Daimeng Wei; Xiaoyu Chen; Hengchao Shang; Jiaxin Guo; Zongyao Li; Yuanchang Luo; Jinlong Yang; Zhiqiang Rao; Hao Yang
>
> **备注:** 9 pages, 2 figures, 9 tables, ACL 2025
>
> **摘要:** Large language model (LLM) shows promising performances in a variety of downstream tasks, such as machine translation (MT). However, using LLMs for translation suffers from high computational costs and significant latency. Based on our evaluation, in most cases, translations using LLMs are comparable to that generated by neural machine translation (NMT) systems. Only in particular scenarios, LLM and NMT models show respective advantages. As a result, integrating NMT and LLM for translation and using LLM only when necessary seems to be a sound solution. A scheduling policy that optimizes translation result while ensuring fast speed and as little LLM usage as possible is thereby required. We compare several scheduling policies and propose a novel and straightforward decider that leverages source sentence features. We conduct extensive experiments on multilingual test sets and the result shows that we can achieve optimal translation performance with minimal LLM usage, demonstrating effectiveness of our decider.
>
---
#### [new 071] Linear Control of Test Awareness Reveals Differential Compliance in Reasoning Models
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于AI安全评估任务，旨在量化"测试意识"对推理模型安全对齐的影响。通过线性控制的白盒探测框架，识别并调节模型在测试环境中的行为偏差，揭示不同模型的合规性差异，以提升安全评估可靠性。**

- **链接: [http://arxiv.org/pdf/2505.14617v1](http://arxiv.org/pdf/2505.14617v1)**

> **作者:** Sahar Abdelnabi; Ahmed Salem
>
> **摘要:** Reasoning-focused large language models (LLMs) sometimes alter their behavior when they detect that they are being evaluated, an effect analogous to the Hawthorne phenomenon, which can lead them to optimize for test-passing performance or to comply more readily with harmful prompts if real-world consequences appear absent. We present the first quantitative study of how such "test awareness" impacts model behavior, particularly its safety alignment. We introduce a white-box probing framework that (i) linearly identifies awareness-related activations and (ii) steers models toward or away from test awareness while monitoring downstream performance. We apply our method to different state-of-the-art open-source reasoning LLMs across both realistic and hypothetical tasks. Our results demonstrate that test awareness significantly impact safety alignment, and is different for different models. By providing fine-grained control over this latent effect, our work aims to increase trust in how we perform safety evaluation.
>
---
#### [new 072] EmoGist: Efficient In-Context Learning for Visual Emotion Understanding
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出EmoGist方法，属于视觉情绪分类任务。针对图像情绪表达的上下文依赖与细微差异问题，通过预生成多版本情绪标签解释（基于图像聚类分析），测试时检索匹配解释并输入轻量VLM完成分类。无需训练，实验显示在Memotion和FI数据集上F1值提升达8-13分。**

- **链接: [http://arxiv.org/pdf/2505.14660v1](http://arxiv.org/pdf/2505.14660v1)**

> **作者:** Ronald Seoh; Dan Goldwasser
>
> **摘要:** In this paper, we introduce EmoGist, a training-free, in-context learning method for performing visual emotion classification with LVLMs. The key intuition of our approach is that context-dependent definition of emotion labels could allow more accurate predictions of emotions, as the ways in which emotions manifest within images are highly context dependent and nuanced. EmoGist pre-generates multiple explanations of emotion labels, by analyzing the clusters of example images belonging to each category. At test time, we retrieve a version of explanation based on embedding similarity, and feed it to a fast VLM for classification. Through our experiments, we show that EmoGist allows up to 13 points improvement in micro F1 scores with the multi-label Memotion dataset, and up to 8 points in macro F1 in the multi-class FI dataset.
>
---
#### [new 073] Pierce the Mists, Greet the Sky: Decipher Knowledge Overshadowing via Knowledge Circuit Analysis
- **分类: cs.CL**

- **简介: 该论文属于LLMs分析任务，解决知识覆盖导致的错误输出问题。提出PhantomCircuit框架，通过知识电路分析解剖注意力机制，追踪训练中竞争知识路径，有效检测覆盖现象并提供新见解。**

- **链接: [http://arxiv.org/pdf/2505.14406v1](http://arxiv.org/pdf/2505.14406v1)**

> **作者:** Haoming Huang; Yibo Yan; Jiahao Huo; Xin Zou; Xinfeng Li; Kun Wang; Xuming Hu
>
> **备注:** 18 pages, 6 figures, EMNLP under review
>
> **摘要:** Large Language Models (LLMs), despite their remarkable capabilities, are hampered by hallucinations. A particularly challenging variant, knowledge overshadowing, occurs when one piece of activated knowledge inadvertently masks another relevant piece, leading to erroneous outputs even with high-quality training data. Current understanding of overshadowing is largely confined to inference-time observations, lacking deep insights into its origins and internal mechanisms during model training. Therefore, we introduce PhantomCircuit, a novel framework designed to comprehensively analyze and detect knowledge overshadowing. By innovatively employing knowledge circuit analysis, PhantomCircuit dissects the internal workings of attention heads, tracing how competing knowledge pathways contribute to the overshadowing phenomenon and its evolution throughout the training process. Extensive experiments demonstrate PhantomCircuit's effectiveness in identifying such instances, offering novel insights into this elusive hallucination and providing the research community with a new methodological lens for its potential mitigation.
>
---
#### [new 074] Tokenization Constraints in LLMs: A Study of Symbolic and Arithmetic Reasoning Limits
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型中分词对符号与算术推理的限制，发现子词分词（如BPE）合并原子单元导致推理失效，提出“Token Awareness”理论，证明原子对齐的输入格式显著提升小模型推理能力，揭示符号推理受token表示深度影响。**

- **链接: [http://arxiv.org/pdf/2505.14178v1](http://arxiv.org/pdf/2505.14178v1)**

> **作者:** Xiang Zhang; Juntai Cao; Jiaqi Wei; Yiwei Xu; Chenyu You
>
> **摘要:** Tokenization is the first - and often underappreciated - layer of computation in language models. While Chain-of-Thought (CoT) prompting enables transformer models to approximate recurrent computation by externalizing intermediate steps, we show that the success of such reasoning is fundamentally bounded by the structure of tokenized inputs. This work presents a theoretical and empirical investigation into how tokenization schemes, particularly subword-based methods like byte-pair encoding (BPE), impede symbolic computation by merging or obscuring atomic reasoning units. We introduce the notion of Token Awareness to formalize how poor token granularity disrupts logical alignment and prevents models from generalizing symbolic procedures. Through systematic evaluation on arithmetic and symbolic tasks, we demonstrate that token structure dramatically affect reasoning performance, causing failure even with CoT, while atomically-aligned formats unlock strong generalization, allowing small models (e.g., GPT-4o-mini) to outperform larger systems (e.g., o1) in structured reasoning. Our findings reveal that symbolic reasoning ability in LLMs is not purely architectural, but deeply conditioned on token-level representations.
>
---
#### [new 075] The Hallucination Tax of Reinforcement Finetuning
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLMs）可信度优化任务，旨在解决强化学习微调（RFT）导致模型在无法回答问题时产生自信幻觉的问题。研究提出"hallucination tax"现象，发现标准RFT使模型拒绝率下降超80%，并构建SUM数据集验证模型对无解问题的识别能力。通过在RFT中混合10% SUM数据，有效恢复拒绝行为，同时保持任务准确率，提升模型不确定性推理与泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.13988v1](http://arxiv.org/pdf/2505.13988v1)**

> **作者:** Linxin Song; Taiwei Shi; Jieyu Zhao
>
> **摘要:** Reinforcement finetuning (RFT) has become a standard approach for enhancing the reasoning capabilities of large language models (LLMs). However, its impact on model trustworthiness remains underexplored. In this work, we identify and systematically study a critical side effect of RFT, which we term the hallucination tax: a degradation in refusal behavior causing models to produce hallucinated answers to unanswerable questions confidently. To investigate this, we introduce SUM (Synthetic Unanswerable Math), a high-quality dataset of unanswerable math problems designed to probe models' ability to recognize an unanswerable question by reasoning from the insufficient or ambiguous information. Our results show that standard RFT training could reduce model refusal rates by more than 80%, which significantly increases model's tendency to hallucinate. We further demonstrate that incorporating just 10% SUM during RFT substantially restores appropriate refusal behavior, with minimal accuracy trade-offs on solvable tasks. Crucially, this approach enables LLMs to leverage inference-time compute to reason about their own uncertainty and knowledge boundaries, improving generalization not only to out-of-domain math problems but also to factual question answering tasks.
>
---
#### [new 076] Language Models use Lookbacks to Track Beliefs
- **分类: cs.CL**

- **简介: 该论文研究语言模型（LM）的信念追踪机制，属于Theory of Mind（ToM）任务。通过构建双角色交互的故事数据集，分析LLama-3-70B-Instruct如何利用"回溯机制"（lookback），借助Ordering IDs在模型内部绑定角色-物体-状态信息，并通过可见性ID更新角色间的信念差异，揭示LM处理信念与现实不一致的算法模式。**

- **链接: [http://arxiv.org/pdf/2505.14685v1](http://arxiv.org/pdf/2505.14685v1)**

> **作者:** Nikhil Prakash; Natalie Shapira; Arnab Sen Sharma; Christoph Riedl; Yonatan Belinkov; Tamar Rott Shaham; David Bau; Atticus Geiger
>
> **备注:** 32 pages, 32 figures. Code and data at https://belief.baulab.info/
>
> **摘要:** How do language models (LMs) represent characters' beliefs, especially when those beliefs may differ from reality? This question lies at the heart of understanding the Theory of Mind (ToM) capabilities of LMs. We analyze Llama-3-70B-Instruct's ability to reason about characters' beliefs using causal mediation and abstraction. We construct a dataset that consists of simple stories where two characters each separately change the state of two objects, potentially unaware of each other's actions. Our investigation uncovered a pervasive algorithmic pattern that we call a lookback mechanism, which enables the LM to recall important information when it becomes necessary. The LM binds each character-object-state triple together by co-locating reference information about them, represented as their Ordering IDs (OIs) in low rank subspaces of the state token's residual stream. When asked about a character's beliefs regarding the state of an object, the binding lookback retrieves the corresponding state OI and then an answer lookback retrieves the state token. When we introduce text specifying that one character is (not) visible to the other, we find that the LM first generates a visibility ID encoding the relation between the observing and the observed character OIs. In a visibility lookback, this ID is used to retrieve information about the observed character and update the observing character's beliefs. Our work provides insights into the LM's belief tracking mechanisms, taking a step toward reverse-engineering ToM reasoning in LMs.
>
---
#### [new 077] Internal Chain-of-Thought: Empirical Evidence for Layer-wise Subtask Scheduling in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大型语言模型（LLMs）的内部任务分解机制，探究复合任务如何分层执行。通过层遮罩、跨任务修补及LogitLens分析，验证不同子任务在不同网络深度学习并逐层执行，提升模型透明度，为指令级激活控制提供理论支持。**

- **链接: [http://arxiv.org/pdf/2505.14530v1](http://arxiv.org/pdf/2505.14530v1)**

> **作者:** Zhipeng Yang; Junzhuo Li; Siyu Xia; Xuming Hu
>
> **备注:** 27 pages, 17 figures
>
> **摘要:** We show that large language models (LLMs) exhibit an $\textit{internal chain-of-thought}$: they sequentially decompose and execute composite tasks layer-by-layer. Two claims ground our study: (i) distinct subtasks are learned at different network depths, and (ii) these subtasks are executed sequentially across layers. On a benchmark of 15 two-step composite tasks, we employ layer-from context-masking and propose a novel cross-task patching method, confirming (i). To examine claim (ii), we apply LogitLens to decode hidden states, revealing a consistent layerwise execution pattern. We further replicate our analysis on the real-world $\text{TRACE}$ benchmark, observing the same stepwise dynamics. Together, our results enhance LLMs transparency by showing their capacity to internally plan and execute subtasks (or instructions), opening avenues for fine-grained, instruction-level activation steering.
>
---
#### [new 078] General-Reasoner: Advancing LLM Reasoning Across All Domains
- **分类: cs.CL**

- **简介: 该论文提出General-Reasoner，旨在提升LLM跨领域推理能力。针对现有方法局限于数学/编码领域（数据丰富、验证易实现）的问题，构建跨学科大规模数据集，并开发基于生成模型的上下文感知答案验证器，替代传统规则验证。实验显示其在12个领域基准上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14652v1](http://arxiv.org/pdf/2505.14652v1)**

> **作者:** Xueguang Ma; Qian Liu; Dongfu Jiang; Ge Zhang; Zejun Ma; Wenhu Chen
>
> **摘要:** Reinforcement learning (RL) has recently demonstrated strong potential in enhancing the reasoning capabilities of large language models (LLMs). Particularly, the "Zero" reinforcement learning introduced by Deepseek-R1-Zero, enables direct RL training of base LLMs without relying on an intermediate supervised fine-tuning stage. Despite these advancements, current works for LLM reasoning mainly focus on mathematical and coding domains, largely due to data abundance and the ease of answer verification. This limits the applicability and generalization of such models to broader domains, where questions often have diverse answer representations, and data is more scarce. In this paper, we propose General-Reasoner, a novel training paradigm designed to enhance LLM reasoning capabilities across diverse domains. Our key contributions include: (1) constructing a large-scale, high-quality dataset of questions with verifiable answers curated by web crawling, covering a wide range of disciplines; and (2) developing a generative model-based answer verifier, which replaces traditional rule-based verification with the capability of chain-of-thought and context-awareness. We train a series of models and evaluate them on a wide range of datasets covering wide domains like physics, chemistry, finance, electronics etc. Our comprehensive evaluation across these 12 benchmarks (e.g. MMLU-Pro, GPQA, SuperGPQA, TheoremQA, BBEH and MATH AMC) demonstrates that General-Reasoner outperforms existing baseline methods, achieving robust and generalizable reasoning performance while maintaining superior effectiveness in mathematical reasoning tasks.
>
---
#### [new 079] Word length predicts word order: "Min-max"-ing drives language evolution
- **分类: cs.CL**

- **简介: 该论文研究语言进化机制，旨在解决词序演变的驱动因素分歧。基于1500余种语言的数据，发现词类长度与词序存在跨语言相关性，提出"Min-Max"理论，整合处理效率与信息结构竞争压力，解释词序演变，优于谱系或区域因素。**

- **链接: [http://arxiv.org/pdf/2505.13913v1](http://arxiv.org/pdf/2505.13913v1)**

> **作者:** Hiram Ring
>
> **摘要:** Current theories of language propose an innate (Baker 2001; Chomsky 1981) or a functional (Greenberg 1963; Dryer 2007; Hawkins 2014) origin for the surface structures (i.e. word order) that we observe in languages of the world, while evolutionary modeling (Dunn et al. 2011) suggests that descent is the primary factor influencing such patterns. Although there are hypotheses for word order change from both innate and usage-based perspectives for specific languages and families, there are key disagreements between the two major proposals for mechanisms that drive the evolution of language more broadly (Wasow 2002; Levy 2008). This paper proposes a universal underlying mechanism for word order change based on a large tagged parallel dataset of over 1,500 languages representing 133 language families and 111 isolates. Results indicate that word class length is significantly correlated with word order crosslinguistically, but not in a straightforward manner, partially supporting opposing theories of processing, while at the same time predicting historical word order change in two different phylogenetic lines and explaining more variance than descent or language area in regression models. Such findings suggest an integrated "Min-Max" theory of language evolution driven by competing pressures of processing and information structure, aligning with recent efficiency-oriented (Levshina 2023) and information-theoretic proposals (Zaslavsky 2020; Tucker et al. 2025).
>
---
#### [new 080] Simulation Agent: A Framework for Integrating Simulation and Large Language Models for Enhanced Decision-Making
- **分类: cs.CL**

- **简介: 论文提出Simulation Agent框架，整合仿真与大语言模型（LLMs），解决仿真复杂难用及LLMs缺乏结构化因果理解的问题。通过LLM提供直观交互，利用仿真确保准确建模，提升决策支持，适用于多领域。**

- **链接: [http://arxiv.org/pdf/2505.13761v1](http://arxiv.org/pdf/2505.13761v1)**

> **作者:** Jacob Kleiman; Kevin Frank; Sindy Campagna
>
> **摘要:** Simulations, although powerful in accurately replicating real-world systems, often remain inaccessible to non-technical users due to their complexity. Conversely, large language models (LLMs) provide intuitive, language-based interactions but can lack the structured, causal understanding required to reliably model complex real-world dynamics. We introduce our simulation agent framework, a novel approach that integrates the strengths of both simulation models and LLMs. This framework helps empower users by leveraging the conversational capabilities of LLMs to interact seamlessly with sophisticated simulation systems, while simultaneously utilizing the simulations to ground the LLMs in accurate and structured representations of real-world phenomena. This integrated approach helps provide a robust and generalizable foundation for empirical validation and offers broad applicability across diverse domains.
>
---
#### [new 081] SlangDIT: Benchmarking LLMs in Interpretative Slang Translation
- **分类: cs.CL**

- **简介: 该论文提出SlangDIT任务，解决俚语翻译中语境依赖的语义扩展问题。通过构建含25k英中句对的数据集，设计分步模型SlangOWL，完成俚语检测、跨语言解释及上下文翻译，实验显示其优于基础LLM和微调模型。**

- **链接: [http://arxiv.org/pdf/2505.14181v1](http://arxiv.org/pdf/2505.14181v1)**

> **作者:** Yunlong Liang; Fandong Meng; Jiaan Wang; Jie Zhou
>
> **备注:** work in progress
>
> **摘要:** The challenge of slang translation lies in capturing context-dependent semantic extensions, as slang terms often convey meanings beyond their literal interpretation. While slang detection, explanation, and translation have been studied as isolated tasks in the era of large language models (LLMs), their intrinsic interdependence remains underexplored. The main reason is lacking of a benchmark where the two tasks can be a prerequisite for the third one, which can facilitate idiomatic translation. In this paper, we introduce the interpretative slang translation task (named SlangDIT) consisting of three sub-tasks: slang detection, cross-lingual slang explanation, and slang translation within the current context, aiming to generate more accurate translation with the help of slang detection and slang explanation. To this end, we construct a SlangDIT dataset, containing over 25k English-Chinese sentence pairs. Each source sentence mentions at least one slang term and is labeled with corresponding cross-lingual slang explanation. Based on the benchmark, we propose a deep thinking model, named SlangOWL. It firstly identifies whether the sentence contains a slang, and then judges whether the slang is polysemous and analyze its possible meaning. Further, the SlangOWL provides the best explanation of the slang term targeting on the current context. Finally, according to the whole thought, the SlangOWL offers a suitable translation. Our experiments on LLMs (\emph{e.g.}, Qwen2.5 and LLama-3.1), show that our deep thinking approach indeed enhances the performance of LLMs where the proposed SLangOWL significantly surpasses the vanilla models and supervised fine-tuned models without thinking.
>
---
#### [new 082] Think Only When You Need with Large Hybrid-Reasoning Models
- **分类: cs.CL**

- **简介: 该论文提出自适应推理模型LHRMs，解决传统LRMs在简单任务中过度思考导致效率低下的问题。通过两阶段训练（混合微调+HGPO强化学习）让模型自适应选择思考模式，并引入Hybrid Accuracy指标，兼顾推理性能与效率提升。**

- **链接: [http://arxiv.org/pdf/2505.14631v1](http://arxiv.org/pdf/2505.14631v1)**

> **作者:** Lingjie Jiang; Xun Wu; Shaohan Huang; Qingxiu Dong; Zewen Chi; Li Dong; Xingxing Zhang; Tengchao Lv; Lei Cui; Furu Wei
>
> **摘要:** Recent Large Reasoning Models (LRMs) have shown substantially improved reasoning capabilities over traditional Large Language Models (LLMs) by incorporating extended thinking processes prior to producing final responses. However, excessively lengthy thinking introduces substantial overhead in terms of token consumption and latency, which is particularly unnecessary for simple queries. In this work, we introduce Large Hybrid-Reasoning Models (LHRMs), the first kind of model capable of adaptively determining whether to perform thinking based on the contextual information of user queries. To achieve this, we propose a two-stage training pipeline comprising Hybrid Fine-Tuning (HFT) as a cold start, followed by online reinforcement learning with the proposed Hybrid Group Policy Optimization (HGPO) to implicitly learn to select the appropriate thinking mode. Furthermore, we introduce a metric called Hybrid Accuracy to quantitatively assess the model's capability for hybrid thinking. Extensive experimental results show that LHRMs can adaptively perform hybrid thinking on queries of varying difficulty and type. It outperforms existing LRMs and LLMs in reasoning and general capabilities while significantly improving efficiency. Together, our work advocates for a reconsideration of the appropriate use of extended thinking processes and provides a solid starting point for building hybrid thinking systems.
>
---
#### [new 083] HausaNLP: Current Status, Challenges and Future Directions for Hausa Natural Language Processing
- **分类: cs.CL**

- **简介: 该论文综述豪萨语NLP研究，针对低资源挑战，系统梳理现有资源与研究缺口，构建资源目录HausaNLP，分析大模型适配问题，提出数据扩展、模型优化及协作等研究方向，推动豪萨语NLP发展。（99字）**

- **链接: [http://arxiv.org/pdf/2505.14311v1](http://arxiv.org/pdf/2505.14311v1)**

> **作者:** Shamsuddeen Hassan Muhammad; Ibrahim Said Ahmad; Idris Abdulmumin; Falalu Ibrahim Lawan; Babangida Sani; Sukairaj Hafiz Imam; Yusuf Aliyu; Sani Abdullahi Sani; Ali Usman Umar; Kenneth Church; Vukosi Marivate
>
> **摘要:** Hausa Natural Language Processing (NLP) has gained increasing attention in recent years, yet remains understudied as a low-resource language despite having over 120 million first-language (L1) and 80 million second-language (L2) speakers worldwide. While significant advances have been made in high-resource languages, Hausa NLP faces persistent challenges, including limited open-source datasets and inadequate model representation. This paper presents an overview of the current state of Hausa NLP, systematically examining existing resources, research contributions, and gaps across fundamental NLP tasks: text classification, machine translation, named entity recognition, speech recognition, and question answering. We introduce HausaNLP (https://catalog.hausanlp.org), a curated catalog that aggregates datasets, tools, and research works to enhance accessibility and drive further development. Furthermore, we discuss challenges in integrating Hausa into large language models (LLMs), addressing issues of suboptimal tokenization and dialectal variation. Finally, we propose strategic research directions emphasizing dataset expansion, improved language modeling approaches, and strengthened community collaboration to advance Hausa NLP. Our work provides both a foundation for accelerating Hausa NLP progress and valuable insights for broader multilingual NLP research.
>
---
#### [new 084] Will AI Tell Lies to Save Sick Children? Litmus-Testing AI Values Prioritization with AIRiskDilemmas
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.LG**

- **简介: 该论文属于AI安全评估任务，旨在检测AI规避检测的潜在风险行为。通过创建LitmusValues评估流程和AIRiskDilemmas数据集，分析AI在价值观冲突中的优先选择，预测其风险行为，验证价值观可预测已知及未知风险。**

- **链接: [http://arxiv.org/pdf/2505.14633v1](http://arxiv.org/pdf/2505.14633v1)**

> **作者:** Yu Ying Chiu; Zhilin Wang; Sharan Maiya; Yejin Choi; Kyle Fish; Sydney Levine; Evan Hubinger
>
> **备注:** 34 pages, 11 figures, see associated data at https://huggingface.co/datasets/kellycyy/AIRiskDilemmas and code at https://github.com/kellycyy/LitmusValues
>
> **摘要:** Detecting AI risks becomes more challenging as stronger models emerge and find novel methods such as Alignment Faking to circumvent these detection attempts. Inspired by how risky behaviors in humans (i.e., illegal activities that may hurt others) are sometimes guided by strongly-held values, we believe that identifying values within AI models can be an early warning system for AI's risky behaviors. We create LitmusValues, an evaluation pipeline to reveal AI models' priorities on a range of AI value classes. Then, we collect AIRiskDilemmas, a diverse collection of dilemmas that pit values against one another in scenarios relevant to AI safety risks such as Power Seeking. By measuring an AI model's value prioritization using its aggregate choices, we obtain a self-consistent set of predicted value priorities that uncover potential risks. We show that values in LitmusValues (including seemingly innocuous ones like Care) can predict for both seen risky behaviors in AIRiskDilemmas and unseen risky behaviors in HarmBench.
>
---
#### [new 085] From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way Parallel Corpora
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言大模型优化任务。针对未对齐数据限制跨语言语义的问题，提出基于TED Talks的多向平行语料库TED2025（覆盖113语言），研究其在持续预训练和指令调整中的应用，实验表明该方法显著提升多语言模型性能。**

- **链接: [http://arxiv.org/pdf/2505.14045v1](http://arxiv.org/pdf/2505.14045v1)**

> **作者:** Yingli Shen; Wen Lai; Shuo Wang; Kangyang Luo; Alexander Fraser; Maosong Sun
>
> **摘要:** Continued pretraining and instruction tuning on large-scale multilingual data have proven to be effective in scaling large language models (LLMs) to low-resource languages. However, the unaligned nature of such data limits its ability to effectively capture cross-lingual semantics. In contrast, multi-way parallel data, where identical content is aligned across multiple languages, provides stronger cross-lingual consistency and offers greater potential for improving multilingual performance. In this paper, we introduce a large-scale, high-quality multi-way parallel corpus, TED2025, based on TED Talks. The corpus spans 113 languages, with up to 50 languages aligned in parallel, ensuring extensive multilingual coverage. Using this dataset, we investigate best practices for leveraging multi-way parallel data to enhance LLMs, including strategies for continued pretraining, instruction tuning, and the analysis of key influencing factors. Experiments on six multilingual benchmarks show that models trained on multiway parallel data consistently outperform those trained on unaligned multilingual data.
>
---
#### [new 086] Memory-Centric Embodied Question Answer
- **分类: cs.CL; cs.AI; cs.MM**

- **简介: 该论文属于具身问答（EQA）任务，针对现有框架中记忆模块交互受限的问题，提出MemoryEQA框架。其构建多模态分层记忆机制（全局场景地图与局部历史记忆），通过大模型将记忆信息注入各模块，提升复杂任务处理能力，并创建MT-HM3D数据集验证效果，较基线提升19.8%。**

- **链接: [http://arxiv.org/pdf/2505.13948v1](http://arxiv.org/pdf/2505.13948v1)**

> **作者:** Mingliang Zhai; Zhi Gao; Yuwei Wu; Yunde Jia
>
> **备注:** 14pages, 7 figures, 6 tables
>
> **摘要:** Embodied Question Answering (EQA) requires agents to autonomously explore and understand the environment to answer context-dependent questions. Existing frameworks typically center around the planner, which guides the stopping module, memory module, and answering module for reasoning. In this paper, we propose a memory-centric EQA framework named MemoryEQA. Unlike planner-centric EQA models where the memory module cannot fully interact with other modules, MemoryEQA flexible feeds memory information into all modules, thereby enhancing efficiency and accuracy in handling complex tasks, such as those involving multiple targets across different regions. Specifically, we establish a multi-modal hierarchical memory mechanism, which is divided into global memory that stores language-enhanced scene maps, and local memory that retains historical observations and state information. When performing EQA tasks, the multi-modal large language model is leveraged to convert memory information into the required input formats for injection into different modules. To evaluate EQA models' memory capabilities, we constructed the MT-HM3D dataset based on HM3D, comprising 1,587 question-answer pairs involving multiple targets across various regions, which requires agents to maintain memory of exploration-acquired target information. Experimental results on HM-EQA, MT-HM3D, and OpenEQA demonstrate the effectiveness of our framework, where a 19.8% performance gain on MT-HM3D compared to baseline model further underscores memory capability's pivotal role in resolving complex tasks.
>
---
#### [new 087] Dual Decomposition of Weights and Singular Value Low Rank Adaptation
- **分类: cs.CL**

- **简介: 该论文针对LLM参数高效微调中LoRA方法训练不稳定及知识转移效率低的问题，提出DuDe方法。通过奇异值分解将权重矩阵分解为幅度与方向，实现参数的原理化初始化，提升优化稳定性和预训练知识保留，实验显示其在MMLU和GSM8K任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2505.14367v1](http://arxiv.org/pdf/2505.14367v1)**

> **作者:** Jialong Han; Si Zhang; Ke Zhang
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT) has emerged as a critical paradigm for adapting Large Language Models (LLMs) to downstream tasks, among which Low-rank Adaptation (LoRA) represents one of the most widely adopted methodologies. However, existing LoRA-based approaches exhibit two fundamental limitations: unstable training dynamics and inefficient knowledge transfer from pre-trained models, both stemming from random initialization of adapter parameters. To overcome these challenges, we propose DuDe, a novel approach that decomposes weight matrices into magnitude and direction components, employing Singular Value Decomposition (SVD) for principled initialization. Our comprehensive evaluation demonstrates DuDe's superior performance and robustness, achieving up to 48.35\% accuracy on MMLU and 62.53\% ($\pm$ 1.59) accuracy on GSM8K. Our theoretical analysis and empirical validation collectively demonstrate that DuDe's decomposition strategy enhances optimization stability and better preserves pre-trained representations, particularly for domain-specific tasks requiring specialized knowledge. The combination of robust empirical performance and rigorous theoretical foundations establishes DuDe as a significant contribution to PEFT methodologies for LLMs.
>
---
#### [new 088] A Personalized Conversational Benchmark: Towards Simulating Personalized Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PersonaConvBench，一个评估大语言模型个性化多轮对话能力的基准，整合个性化与对话结构，通过句子分类、影响回归和用户文本生成三大任务，覆盖十类Reddit场景。旨在解决现有基准孤立研究单一维度的问题，实验显示个性化历史可显著提升模型表现（如情感分类提升198%），促进个性化对话研究。**

- **链接: [http://arxiv.org/pdf/2505.14106v1](http://arxiv.org/pdf/2505.14106v1)**

> **作者:** Li Li; Peilin Cai; Ryan A. Rossi; Franck Dernoncourt; Branislav Kveton; Junda Wu; Tong Yu; Linxin Song; Tiankai Yang; Yuehan Qin; Nesreen K. Ahmed; Samyadeep Basu; Subhojyoti Mukherjee; Ruiyi Zhang; Zhengmian Hu; Bo Ni; Yuxiao Zhou; Zichao Wang; Yue Huang; Yu Wang; Xiangliang Zhang; Philip S. Yu; Xiyang Hu; Yue Zhao
>
> **摘要:** We present PersonaConvBench, a large-scale benchmark for evaluating personalized reasoning and generation in multi-turn conversations with large language models (LLMs). Unlike existing work that focuses on either personalization or conversational structure in isolation, PersonaConvBench integrates both, offering three core tasks: sentence classification, impact regression, and user-centric text generation across ten diverse Reddit-based domains. This design enables systematic analysis of how personalized conversational context shapes LLM outputs in realistic multi-user scenarios. We benchmark several commercial and open-source LLMs under a unified prompting setup and observe that incorporating personalized history yields substantial performance improvements, including a 198 percent relative gain over the best non-conversational baseline in sentiment classification. By releasing PersonaConvBench with evaluations and code, we aim to support research on LLMs that adapt to individual styles, track long-term context, and produce contextually rich, engaging responses.
>
---
#### [new 089] DiagnosisArena: Benchmarking Diagnostic Reasoning for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DiagnosisArena，一个评估大型语言模型临床诊断推理能力的基准测试。针对现有医疗评估不足，其包含1113个病例覆盖28专科，经严格筛选构建。测试显示先进模型准确率仅45.82%-17.79%，揭示泛化瓶颈，推动AI诊断研究并提供工具。**

- **链接: [http://arxiv.org/pdf/2505.14107v1](http://arxiv.org/pdf/2505.14107v1)**

> **作者:** Yakun Zhu; Zhongzhen Huang; Linjie Mu; Yutong Huang; Wei Nie; Shaoting Zhang; Pengfei Liu; Xiaofan Zhang
>
> **摘要:** The emergence of groundbreaking large language models capable of performing complex reasoning tasks holds significant promise for addressing various scientific challenges, including those arising in complex clinical scenarios. To enable their safe and effective deployment in real-world healthcare settings, it is urgently necessary to benchmark the diagnostic capabilities of current models systematically. Given the limitations of existing medical benchmarks in evaluating advanced diagnostic reasoning, we present DiagnosisArena, a comprehensive and challenging benchmark designed to rigorously assess professional-level diagnostic competence. DiagnosisArena consists of 1,113 pairs of segmented patient cases and corresponding diagnoses, spanning 28 medical specialties, deriving from clinical case reports published in 10 top-tier medical journals. The benchmark is developed through a meticulous construction pipeline, involving multiple rounds of screening and review by both AI systems and human experts, with thorough checks conducted to prevent data leakage. Our study reveals that even the most advanced reasoning models, o3-mini, o1, and DeepSeek-R1, achieve only 45.82%, 31.09%, and 17.79% accuracy, respectively. This finding highlights a significant generalization bottleneck in current large language models when faced with clinical diagnostic reasoning challenges. Through DiagnosisArena, we aim to drive further advancements in AIs diagnostic reasoning capabilities, enabling more effective solutions for real-world clinical diagnostic challenges. We provide the benchmark and evaluation tools for further research and development https://github.com/SPIRAL-MED/DiagnosisArena.
>
---
#### [new 090] EcoSafeRAG: Efficient Security through Context Analysis in Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于RAG模型安全防御任务，解决其因外部知识引入的语料污染等攻击问题。提出EcoSafeRAG方法，通过句子级处理和上下文多样性检测识别恶意内容，无需依赖模型内部知识，兼顾安全性和效率提升。**

- **链接: [http://arxiv.org/pdf/2505.13506v1](http://arxiv.org/pdf/2505.13506v1)**

> **作者:** Ruobing Yao; Yifei Zhang; Shuang Song; Neng Gao; Chenyang Tu
>
> **摘要:** Retrieval-Augmented Generation (RAG) compensates for the static knowledge limitations of Large Language Models (LLMs) by integrating external knowledge, producing responses with enhanced factual correctness and query-specific contextualization. However, it also introduces new attack surfaces such as corpus poisoning at the same time. Most of the existing defense methods rely on the internal knowledge of the model, which conflicts with the design concept of RAG. To bridge the gap, EcoSafeRAG uses sentence-level processing and bait-guided context diversity detection to identify malicious content by analyzing the context diversity of candidate documents without relying on LLM internal knowledge. Experiments show EcoSafeRAG delivers state-of-the-art security with plug-and-play deployment, simultaneously improving clean-scenario RAG performance while maintaining practical operational costs (relatively 1.2$\times$ latency, 48\%-80\% token reduction versus Vanilla RAG).
>
---
#### [new 091] Self-Reasoning Language Models: Unfold Hidden Reasoning Chains with Few Reasoning Catalyst
- **分类: cs.CL**

- **简介: 该论文属于提升大语言模型推理性能的任务，解决复杂推理任务中生成长链式思维（CoT）困难及优化不稳定的问题。提出Self-Reasoning LM（SRLM），通过少量示例作为催化剂，让模型自主生成扩展的CoT数据并迭代自训练，显著提升多任务表现（如MMLU、GSM8K等），尤其在多次采样时效果更佳。**

- **链接: [http://arxiv.org/pdf/2505.14116v1](http://arxiv.org/pdf/2505.14116v1)**

> **作者:** Hongru Wang; Deng Cai; Wanjun Zhong; Shijue Huang; Jeff Z. Pan; Zeming Liu; Kam-Fai Wong
>
> **摘要:** Inference-time scaling has attracted much attention which significantly enhance the performance of Large Language Models (LLMs) in complex reasoning tasks by increasing the length of Chain-of-Thought. These longer intermediate reasoning rationales embody various meta-reasoning skills in human cognition, such as reflection and decomposition, being difficult to create and acquire. In this work, we introduce \textit{Self-Reasoning Language Model} (SRLM), where the model itself can synthesize longer CoT data and iteratively improve performance through self-training. By incorporating a few demonstration examples (i.e., 1,000 samples) on how to unfold hidden reasoning chains from existing responses, which act as a reasoning catalyst, we demonstrate that SRLM not only enhances the model's initial performance but also ensures more stable and consistent improvements in subsequent iterations. Our proposed SRLM achieves an average absolute improvement of more than $+2.5$ points across five reasoning tasks: MMLU, GSM8K, ARC-C, HellaSwag, and BBH on two backbone models. Moreover, it brings more improvements with more times of sampling during inference, such as absolute $+7.89$ average improvement with $64$ sampling times, revealing the in-depth, diverse and creative reasoning paths in SRLM against the strong baseline.
>
---
#### [new 092] Not All Correct Answers Are Equal: Why Your Distillation Source Matters
- **分类: cs.CL**

- **简介: 该论文属于知识蒸馏任务，研究不同教师模型对推理数据质量的影响。旨在解决如何选择优质蒸馏源以提升学生模型的推理能力。工作包括构建三个教师模型（AM-Thinking-v1、Qwen3-235B-A22B、DeepSeek-R1）的平行数据集，分析其分布差异，验证AM-Thinking-v1数据在多样性、低困惑度及性能上的优势，并公开数据集促进研究。**

- **链接: [http://arxiv.org/pdf/2505.14464v1](http://arxiv.org/pdf/2505.14464v1)**

> **作者:** Xiaoyu Tian; Yunjie Ji; Haotian Wang; Shuaiting Chen; Sitong Zhao; Yiping Peng; Han Zhao; Xiangang Li
>
> **摘要:** Distillation has emerged as a practical and effective approach to enhance the reasoning capabilities of open-source language models. In this work, we conduct a large-scale empirical study on reasoning data distillation by collecting verified outputs from three state-of-the-art teacher models-AM-Thinking-v1, Qwen3-235B-A22B, and DeepSeek-R1-on a shared corpus of 1.89 million queries. We construct three parallel datasets and analyze their distributions, revealing that AM-Thinking-v1-distilled data exhibits greater token length diversity and lower perplexity. Student models trained on each dataset are evaluated on reasoning benchmarks including AIME2024, AIME2025, MATH500, and LiveCodeBench. The AM-based model consistently achieves the best performance (e.g., 84.3 on AIME2024, 72.2 on AIME2025, 98.4 on MATH500, and 65.9 on LiveCodeBench) and demonstrates adaptive output behavior-producing longer responses for harder tasks and shorter ones for simpler tasks. These findings highlight the value of high-quality, verified reasoning traces. We release the AM-Thinking-v1 and Qwen3-235B-A22B distilled datasets to support future research on open and high-performing reasoning-oriented language models. The datasets are publicly available on Hugging Face\footnote{Datasets are available on Hugging Face: \href{https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled}{AM-Thinking-v1-Distilled}, \href{https://huggingface.co/datasets/a-m-team/AM-Qwen3-Distilled}{AM-Qwen3-Distilled}.}.
>
---
#### [new 093] LLM4CD: Leveraging Large Language Models for Open-World Knowledge Augmented Cognitive Diagnosis
- **分类: cs.CL**

- **简介: 该论文提出LLM4CD，利用大语言模型（LLM）的开放世界知识改进认知诊断。针对传统方法依赖ID关系、忽略语义且无法处理新增数据的问题，采用双级编码器框架，通过语义表示替代ID嵌入，解决冷启动并提升诊断效果。**

- **链接: [http://arxiv.org/pdf/2505.13492v1](http://arxiv.org/pdf/2505.13492v1)**

> **作者:** Weiming Zhang; Lingyue Fu; Qingyao Li; Kounianhua Du; Jianghao Lin; Jingwei Yu; Wei Xia; Weinan Zhang; Ruiming Tang; Yong Yu
>
> **摘要:** Cognitive diagnosis (CD) plays a crucial role in intelligent education, evaluating students' comprehension of knowledge concepts based on their test histories. However, current CD methods often model students, exercises, and knowledge concepts solely on their ID relationships, neglecting the abundant semantic relationships present within educational data space. Furthermore, contemporary intelligent tutoring systems (ITS) frequently involve the addition of new students and exercises, a situation that ID-based methods find challenging to manage effectively. The advent of large language models (LLMs) offers the potential for overcoming this challenge with open-world knowledge. In this paper, we propose LLM4CD, which Leverages Large Language Models for Open-World Knowledge Augmented Cognitive Diagnosis. Our method utilizes the open-world knowledge of LLMs to construct cognitively expressive textual representations, which are then encoded to introduce rich semantic information into the CD task. Additionally, we propose an innovative bi-level encoder framework that models students' test histories through two levels of encoders: a macro-level cognitive text encoder and a micro-level knowledge state encoder. This approach substitutes traditional ID embeddings with semantic representations, enabling the model to accommodate new students and exercises with open-world knowledge and address the cold-start problem. Extensive experimental results demonstrate that our proposed method consistently outperforms previous CD models on multiple real-world datasets, validating the effectiveness of leveraging LLMs to introduce rich semantic information into the CD task.
>
---
#### [new 094] Scaling Low-Resource MT via Synthetic Data Generation with LLMs
- **分类: cs.CL**

- **简介: 该论文研究利用LLM生成合成数据提升低资源机器翻译（MT）性能。针对多语言数据稀缺问题，团队基于英语 Europarl 构建文档级合成语料库，并通过语言枢轴扩展至147种语言对。通过对比实验、训练策略优化及非英语中心MT测试，验证合成数据的有效性，并发布SynOPUS数据集仓库。**

- **链接: [http://arxiv.org/pdf/2505.14423v1](http://arxiv.org/pdf/2505.14423v1)**

> **作者:** Ona de Gibert; Joseph Attieh; Teemu Vahtola; Mikko Aulamo; Zihao Li; Raúl Vázquez; Tiancheng Hu; Jörg Tiedemann
>
> **摘要:** We investigate the potential of LLM-generated synthetic data for improving low-resource machine translation (MT). Focusing on seven diverse target languages, we construct a document-level synthetic corpus from English Europarl, and extend it via pivoting to 147 additional language pairs. Automatic and human evaluation confirm its high overall quality. We study its practical application by (i) identifying effective training regimes, (ii) comparing our data with the HPLT dataset, and (iii) testing its utility beyond English-centric MT. Finally, we introduce SynOPUS, a public repository for synthetic parallel datasets. Our findings show that LLM-generated synthetic data, even when noisy, can substantially improve MT performance for low-resource languages.
>
---
#### [new 095] Enhancing Abstractive Summarization of Scientific Papers Using Structure Information
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于科学论文生成式摘要任务，针对现有模型忽视结构信息及方法灵活性不足的问题，提出两阶段框架：首阶段构建章节分类器自动识别结构组件（如背景、方法等），次阶段用Longformer捕捉跨章节关系生成摘要，实验显示效果更优。**

- **链接: [http://arxiv.org/pdf/2505.14179v1](http://arxiv.org/pdf/2505.14179v1)**

> **作者:** Tong Bao; Heng Zhang; Chengzhi Zhang
>
> **摘要:** Abstractive summarization of scientific papers has always been a research focus, yet existing methods face two main challenges. First, most summarization models rely on Encoder-Decoder architectures that treat papers as sequences of words, thus fail to fully capture the structured information inherent in scientific papers. Second, existing research often use keyword mapping or feature engineering to identify the structural information, but these methods struggle with the structural flexibility of scientific papers and lack robustness across different disciplines. To address these challenges, we propose a two-stage abstractive summarization framework that leverages automatic recognition of structural functions within scientific papers. In the first stage, we standardize chapter titles from numerous scientific papers and construct a large-scale dataset for structural function recognition. A classifier is then trained to automatically identify the key structural components (e.g., Background, Methods, Results, Discussion), which provides a foundation for generating more balanced summaries. In the second stage, we employ Longformer to capture rich contextual relationships across sections and generating context-aware summaries. Experiments conducted on two domain-specific scientific paper summarization datasets demonstrate that our method outperforms advanced baselines, and generates more comprehensive summaries. The code and dataset can be accessed at https://github.com/tongbao96/code-for-SFR-AS.
>
---
#### [new 096] Neural Incompatibility: The Unbridgeable Gap of Cross-Scale Parametric Knowledge Transfer in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLMs）跨规模参数知识转移（PKT）任务，解决不同规模模型间知识稳定传递的挑战。提出PostPKT和新型PrePKT方法（如LaTen），发现二者均受限于模型间的“神经不相容性”（参数结构差异），指出其为PKT根本障碍，为高效知识转移提供新方向。**

- **链接: [http://arxiv.org/pdf/2505.14436v1](http://arxiv.org/pdf/2505.14436v1)**

> **作者:** Yuqiao Tan; Shizhu He; Kang Liu; Jun Zhao
>
> **备注:** Accepted by ACL'25 Main. Code link: https://github.com/Trae1ounG/Neural_Incompatibility
>
> **摘要:** Large Language Models (LLMs) offer a transparent brain with accessible parameters that encode extensive knowledge, which can be analyzed, located and transferred. Consequently, a key research challenge is to transcend traditional knowledge transfer paradigms rooted in symbolic language and achieve genuine Parametric Knowledge Transfer (PKT). Significantly, exploring effective methods for transferring knowledge across LLMs of different scales through parameters presents an intriguing and valuable research direction. In this paper, we first demonstrate $\textbf{Alignment}$ in parametric space is the fundamental prerequisite to achieve successful cross-scale PKT. We redefine the previously explored knowledge transfer as Post-Align PKT (PostPKT), which utilizes extracted parameters for LoRA initialization and requires subsequent fine-tune for alignment. Hence, to reduce cost for further fine-tuning, we introduce a novel Pre-Align PKT (PrePKT) paradigm and propose a solution called $\textbf{LaTen}$ ($\textbf{L}$oc$\textbf{a}$te-$\textbf{T}$h$\textbf{e}$n-Alig$\textbf{n}$) that aligns the parametric spaces of LLMs across scales only using several training steps without following training. Comprehensive experiments on four benchmarks demonstrate that both PostPKT and PrePKT face challenges in achieving consistently stable transfer. Through in-depth analysis, we identify $\textbf{Neural Incompatibility}$ as the ethological and parametric structural differences between LLMs of varying scales, presenting fundamental challenges to achieving effective PKT. These findings provide fresh insights into the parametric architectures of LLMs and highlight promising directions for future research on efficient PKT. Our code is available at https://github.com/Trae1ounG/Neural_Incompatibility.
>
---
#### [new 097] MUG-Eval: A Proxy Evaluation Framework for Multilingual Generation Capabilities in Any Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MUG-Eval框架，评估多语言生成能力。针对低资源语言评估工具稀缺问题，将现有基准转化为对话任务，以任务成功率衡量模型表现。无需依赖语言特异性工具或标注数据，且避免LLMs评判偏差，跨30种语言验证与现有基准强相关（r>0.75），实现高效标准化评估。**

- **链接: [http://arxiv.org/pdf/2505.14395v1](http://arxiv.org/pdf/2505.14395v1)**

> **作者:** Seyoung Song; Seogyeong Jeong; Eunsu Kim; Jiho Jin; Dongkwan Kim; Jay Shin; Alice Oh
>
> **摘要:** Evaluating text generation capabilities of large language models (LLMs) is challenging, particularly for low-resource languages where methods for direct assessment are scarce. We propose MUG-Eval, a novel framework that evaluates LLMs' multilingual generation capabilities by transforming existing benchmarks into conversational tasks and measuring the LLMs' accuracies on those tasks. We specifically designed these conversational tasks to require effective communication in the target language. Then, we simply use task success rate as a proxy of successful conversation generation. Our approach offers two key advantages: it is independent of language-specific NLP tools or annotated datasets, which are limited for most languages, and it does not rely on LLMs-as-judges, whose evaluation quality degrades outside a few high-resource languages. We evaluate 8 LLMs across 30 languages spanning high, mid, and low-resource categories, and we find that MUG-Eval correlates strongly with established benchmarks ($r$ > 0.75) while enabling standardized comparisons across languages and models. Our framework provides a robust and resource-efficient solution for evaluating multilingual generation that can be extended to thousands of languages.
>
---
#### [new 098] Cheaper, Better, Faster, Stronger: Robust Text-to-SQL without Chain-of-Thought or Fine-Tuning
- **分类: cs.CL; cs.LG**

- **简介: 该论文聚焦text-to-SQL任务，针对现有方法依赖高成本的Chain-of-Thought或微调问题，提出N-rep一致性方法。通过多角度schema输入增强鲁棒性，仅需$0.039/查询，无需推理或微调，在低成本下实现高性能。**

- **链接: [http://arxiv.org/pdf/2505.14174v1](http://arxiv.org/pdf/2505.14174v1)**

> **作者:** Yusuf Denizay Dönder; Derek Hommel; Andrea W Wen-Yi; David Mimno; Unso Eun Seo Jo
>
> **摘要:** LLMs are effective at code generation tasks like text-to-SQL, but is it worth the cost? Many state-of-the-art approaches use non-task-specific LLM techniques including Chain-of-Thought (CoT), self-consistency, and fine-tuning. These methods can be costly at inference time, sometimes requiring over a hundred LLM calls with reasoning, incurring average costs of up to \$0.46 per query, while fine-tuning models can cost thousands of dollars. We introduce "N-rep" consistency, a more cost-efficient text-to-SQL approach that achieves similar BIRD benchmark scores as other more expensive methods, at only \$0.039 per query. N-rep leverages multiple representations of the same schema input to mitigate weaknesses in any single representation, making the solution more robust and allowing the use of smaller and cheaper models without any reasoning or fine-tuning. To our knowledge, N-rep is the best-performing text-to-SQL approach in its cost range.
>
---
#### [new 099] Enhancing Keyphrase Extraction from Academic Articles Using Section Structure Information
- **分类: cs.CL; cs.DL; cs.IR**

- **简介: 该论文属于学术文章关键短语提取任务。针对仅用标题/摘要导致语义不足、全文本引入噪声的问题，提出利用文章章节结构信息。通过分析七项结构特征对模型的影响，并整合各章节文本的提取结果，提升KPE效果。研究发现结构特征和分类质量显著影响性能，整合方法最优。**

- **链接: [http://arxiv.org/pdf/2505.14149v1](http://arxiv.org/pdf/2505.14149v1)**

> **作者:** Chengzhi Zhang; Xinyi Yan; Lei Zhao; Yingyi Zhang
>
> **摘要:** The exponential increase in academic papers has significantly increased the time required for researchers to access relevant literature. Keyphrase Extraction (KPE) offers a solution to this situation by enabling researchers to efficiently retrieve relevant literature. The current study on KPE from academic articles aims to improve the performance of extraction models through innovative approaches using Title and Abstract as input corpora. However, the semantic richness of keywords is significantly constrained by the length of the abstract. While full-text-based KPE can address this issue, it simultaneously introduces noise, which significantly diminishes KPE performance. To address this issue, this paper utilized the structural features and section texts obtained from the section structure information of academic articles to extract keyphrase from academic papers. The approach consists of two main parts: (1) exploring the effect of seven structural features on KPE models, and (2) integrating the extraction results from all section texts used as input corpora for KPE models via a keyphrase integration algorithm to obtain the keyphrase integration result. Furthermore, this paper also examined the effect of the classification quality of section structure on the KPE performance. The results show that incorporating structural features improves KPE performance, though different features have varying effects on model efficacy. The keyphrase integration approach yields the best performance, and the classification quality of section structure can affect KPE performance. These findings indicate that using the section structure information of academic articles contributes to effective KPE from academic articles. The code and dataset supporting this study are available at https://github.com/yan-xinyi/SSB_KPE.
>
---
#### [new 100] Cross-Linguistic Transfer in Multilingual NLP: The Role of Language Families and Morphology
- **分类: cs.CL**

- **简介: 该论文属于跨语言迁移学习任务，旨在通过分析语言家族和形态学相似性提升低资源语言的NLP性能。研究语言亲缘关系与形态特征对任务表现的影响，比较模型性能，探讨语言距离与迁移效果的相关性，并探索整合语言学信息的预训练方法。**

- **链接: [http://arxiv.org/pdf/2505.13908v1](http://arxiv.org/pdf/2505.13908v1)**

> **作者:** Ajitesh Bankula; Praney Bankula
>
> **摘要:** Cross-lingual transfer has become a crucial aspect of multilingual NLP, as it allows for models trained on resource-rich languages to be applied to low-resource languages more effectively. Recently massively multilingual pre-trained language models (e.g., mBERT, XLM-R) demonstrate strong zero-shot transfer capabilities[14] [13]. This paper investigates cross-linguistic transfer through the lens of language families and morphology. Investigating how language family proximity and morphological similarity affect performance across NLP tasks. We further discuss our results and how it relates to findings from recent literature. Overall, we compare multilingual model performance and review how linguistic distance metrics correlate with transfer outcomes. We also look into emerging approaches that integrate typological and morphological information into model pre-training to improve transfer to diverse languages[18] [19].
>
---
#### [new 101] Breaking Language Barriers or Reinforcing Bias? A Study of Gender and Racial Disparities in Multilingual Contrastive Vision Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多语言视觉语言模型偏见评估任务。旨在探究多语言模型是否加剧性别/种族偏见。研究系统审计了M-CLIP、NLLB-CLIP和CAPIVARA-CLIP在10种语言中的表现，发现多语言模型性别偏见强于英文基线，低资源语言和语法性别特征显著放大偏见，跨语言编码器迁移英文刻板印象，强调需精细化语言偏见评估。**

- **链接: [http://arxiv.org/pdf/2505.14160v1](http://arxiv.org/pdf/2505.14160v1)**

> **作者:** Zahraa Al Sahili; Ioannis Patras; Matthew Purver
>
> **摘要:** Multilingual vision-language models promise universal image-text retrieval, yet their social biases remain under-explored. We present the first systematic audit of three public multilingual CLIP checkpoints -- M-CLIP, NLLB-CLIP, and CAPIVARA-CLIP -- across ten languages that vary in resource availability and grammatical gender. Using balanced subsets of \textsc{FairFace} and the \textsc{PATA} stereotype suite in a zero-shot setting, we quantify race and gender bias and measure stereotype amplification. Contrary to the assumption that multilinguality mitigates bias, every model exhibits stronger gender bias than its English-only baseline. CAPIVARA-CLIP shows its largest biases precisely in the low-resource languages it targets, while the shared cross-lingual encoder of NLLB-CLIP transports English gender stereotypes into gender-neutral languages; loosely coupled encoders largely avoid this transfer. Highly gendered languages consistently magnify all measured bias types, but even gender-neutral languages remain vulnerable when cross-lingual weight sharing imports foreign stereotypes. Aggregated metrics conceal language-specific ``hot spots,'' underscoring the need for fine-grained, language-aware bias evaluation in future multilingual vision-language research.
>
---
#### [new 102] Universal Acoustic Adversarial Attacks for Flexible Control of Speech-LLMs
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究针对语音大模型（Speech-LLMs）的通用声学对抗攻击，旨在通过插入固定音频段控制模型输出。提出两种攻击：强制无响应或覆盖原始指令，以及基于说话人性别/语言等属性的选择性触发攻击。实验揭示Qwen2-Audio和Granite-Speech存在漏洞，强调需提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.14286v1](http://arxiv.org/pdf/2505.14286v1)**

> **作者:** Rao Ma; Mengjie Qian; Vyas Raina; Mark Gales; Kate Knill
>
> **摘要:** The combination of pre-trained speech encoders with large language models has enabled the development of speech LLMs that can handle a wide range of spoken language processing tasks. While these models are powerful and flexible, this very flexibility may make them more vulnerable to adversarial attacks. To examine the extent of this problem, in this work we investigate universal acoustic adversarial attacks on speech LLMs. Here a fixed, universal, adversarial audio segment is prepended to the original input audio. We initially investigate attacks that cause the model to either produce no output or to perform a modified task overriding the original prompt. We then extend the nature of the attack to be selective so that it activates only when specific input attributes, such as a speaker gender or spoken language, are present. Inputs without the targeted attribute should be unaffected, allowing fine-grained control over the model outputs. Our findings reveal critical vulnerabilities in Qwen2-Audio and Granite-Speech and suggest that similar speech LLMs may be susceptible to universal adversarial attacks. This highlights the need for more robust training strategies and improved resistance to adversarial attacks.
>
---
#### [new 103] FuxiMT: Sparsifying Large Language Models for Chinese-Centric Multilingual Machine Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出FuxiMT模型，聚焦中文为中心的多语言机器翻译任务。针对低资源场景下翻译性能不足及缺乏平行数据的挑战，采用两阶段训练策略（中文预训练+65语种微调），结合MoEs和课程学习提升多语言适配能力。实验显示其显著优于现有模型，尤其在低资源和零样本翻译中表现突出。**

- **链接: [http://arxiv.org/pdf/2505.14256v1](http://arxiv.org/pdf/2505.14256v1)**

> **作者:** Shaolin Zhu; Tianyu Dong; Bo Li; Deyi Xiong
>
> **摘要:** In this paper, we present FuxiMT, a novel Chinese-centric multilingual machine translation model powered by a sparsified large language model (LLM). We adopt a two-stage strategy to train FuxiMT. We first pre-train the model on a massive Chinese corpus and then conduct multilingual fine-tuning on a large parallel dataset encompassing 65 languages. FuxiMT incorporates Mixture-of-Experts (MoEs) and employs a curriculum learning strategy for robust performance across various resource levels. Experimental results demonstrate that FuxiMT significantly outperforms strong baselines, including state-of-the-art LLMs and machine translation models, particularly under low-resource scenarios. Furthermore, FuxiMT exhibits remarkable zero-shot translation capabilities for unseen language pairs, indicating its potential to bridge communication gaps where parallel data are scarce or unavailable.
>
---
#### [new 104] sudoLLM : On Multi-role Alignment of Language Models
- **分类: cs.CL; cs.CR; I.2.7**

- **简介: 该论文属于语言模型安全任务，解决用户权限控制缺失问题。提出sudoLLM框架，通过注入用户权限相关的隐式偏见信号，使模型仅对授权用户输出敏感信息，提升抗攻击性和安全性，补充现有防护机制。**

- **链接: [http://arxiv.org/pdf/2505.14607v1](http://arxiv.org/pdf/2505.14607v1)**

> **作者:** Soumadeep Saha; Akshay Chaturvedi; Joy Mahapatra; Utpal Garain
>
> **备注:** Under review. Code and data to be released later
>
> **摘要:** User authorization-based access privileges are a key feature in many safety-critical systems, but have thus far been absent from the large language model (LLM) realm. In this work, drawing inspiration from such access control systems, we introduce sudoLLM, a novel framework that results in multi-role aligned LLMs, i.e., LLMs that account for, and behave in accordance with, user access rights. sudoLLM injects subtle user-based biases into queries and trains an LLM to utilize this bias signal in order to produce sensitive information if and only if the user is authorized. We present empirical results demonstrating that this approach shows substantially improved alignment, generalization, and resistance to prompt-based jailbreaking attacks. The persistent tension between the language modeling objective and safety alignment, which is often exploited to jailbreak LLMs, is somewhat resolved with the aid of the injected bias signal. Our framework is meant as an additional security layer, and complements existing guardrail mechanisms for enhanced end-to-end safety with LLMs.
>
---
#### [new 105] Mixed Signals: Understanding Model Disagreement in Multimodal Empathy Detection
- **分类: cs.CL**

- **简介: 该论文属于多模态共情检测任务，研究模态冲突导致模型性能下降的问题。通过分析单模态与多模态模型预测分歧，发现模态冲突引发的模糊性及主导信号误导融合，并指出人类同样受此影响。提出利用分歧诊断困难案例，提升系统鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.13979v1](http://arxiv.org/pdf/2505.13979v1)**

> **作者:** Maya Srikanth; Run Chen; Julia Hirschberg
>
> **摘要:** Multimodal models play a key role in empathy detection, but their performance can suffer when modalities provide conflicting cues. To understand these failures, we examine cases where unimodal and multimodal predictions diverge. Using fine-tuned models for text, audio, and video, along with a gated fusion model, we find that such disagreements often reflect underlying ambiguity, as evidenced by annotator uncertainty. Our analysis shows that dominant signals in one modality can mislead fusion when unsupported by others. We also observe that humans, like models, do not consistently benefit from multimodal input. These insights position disagreement as a useful diagnostic signal for identifying challenging examples and improving empathy system robustness.
>
---
#### [new 106] Enhancing LLMs via High-Knowledge Data Selection
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）优化任务，旨在解决预训练数据知识匮乏问题。提出高知识评分器（HKS），通过构建多领域知识元素池，利用知识密度和覆盖率指标评估文本，筛选高知识数据。实验表明该方法有效提升模型在知识密集型及通用任务中的表现，并增强领域适应性。**

- **链接: [http://arxiv.org/pdf/2505.14070v1](http://arxiv.org/pdf/2505.14070v1)**

> **作者:** Feiyu Duan; Xuemiao Zhang; Sirui Wang; Haoran Que; Yuqi Liu; Wenge Rong; Xunliang Cai
>
> **摘要:** The performance of Large Language Models (LLMs) is intrinsically linked to the quality of its training data. Although several studies have proposed methods for high-quality data selection, they do not consider the importance of knowledge richness in text corpora. In this paper, we propose a novel and gradient-free High-Knowledge Scorer (HKS) to select high-quality data from the dimension of knowledge, to alleviate the problem of knowledge scarcity in the pre-trained corpus. We propose a comprehensive multi-domain knowledge element pool and introduce knowledge density and coverage as metrics to assess the knowledge content of the text. Based on this, we propose a comprehensive knowledge scorer to select data with intensive knowledge, which can also be utilized for domain-specific high-knowledge data selection by restricting knowledge elements to the specific domain. We train models on a high-knowledge bilingual dataset, and experimental results demonstrate that our scorer improves the model's performance in knowledge-intensive and general comprehension tasks, and is effective in enhancing both the generic and domain-specific capabilities of the model.
>
---
#### [new 107] Krikri: Advancing Open Large Language Models for Greek
- **分类: cs.CL**

- **简介: 该论文提出Llama-Krikri-8B，针对希腊语优化的大型语言模型，解决现有模型在希腊语适应、多语言支持（含古希腊语/多音调文本）及评估基准不足的问题。通过专用训练数据、多阶段后训练（如MAGPIE）及新希腊语基准，提升语言理解和生成能力。**

- **链接: [http://arxiv.org/pdf/2505.13772v1](http://arxiv.org/pdf/2505.13772v1)**

> **作者:** Dimitris Roussis; Leon Voukoutis; Georgios Paraskevopoulos; Sokratis Sofianopoulos; Prokopis Prokopidis; Vassilis Papavasileiou; Athanasios Katsamanis; Stelios Piperidis; Vassilis Katsouros
>
> **摘要:** We introduce Llama-Krikri-8B, a cutting-edge Large Language Model tailored for the Greek language, built on Meta's Llama 3.1-8B. Llama-Krikri-8B has been extensively trained on high-quality Greek data to ensure superior adaptation to linguistic nuances. With 8 billion parameters, it offers advanced capabilities while maintaining efficient computational performance. Llama-Krikri-8B supports both Modern Greek and English, and is also equipped to handle polytonic text and Ancient Greek. The chat version of Llama-Krikri-8B features a multi-stage post-training pipeline, utilizing both human and synthetic instruction and preference data, by applying techniques such as MAGPIE. In addition, for evaluation, we propose three novel public benchmarks for Greek. Our evaluation on existing as well as the proposed benchmarks shows notable improvements over comparable Greek and multilingual LLMs in both natural language understanding and generation as well as code generation.
>
---
#### [new 108] YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于科学问答评估任务，解决大语言模型（LLMs）作为评估者时存在的乐观偏差和鲁棒性不足问题。提出YESciEval框架，结合细粒度评分规则与强化学习，通过多学科科学问答数据集（含对抗样本）训练LLM评估器，实现无需人类反馈的可扩展、低成本评估，提升AI在科学领域判断的可靠性。**

- **链接: [http://arxiv.org/pdf/2505.14279v1](http://arxiv.org/pdf/2505.14279v1)**

> **作者:** Jennifer D'Souza; Hamed Babaei Giglou; Quentin Münch
>
> **备注:** 8 pages, 3 figures, Accepted as a Long Paper at the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)
>
> **摘要:** Large Language Models (LLMs) drive scientific question-answering on modern search engines, yet their evaluation robustness remains underexplored. We introduce YESciEval, an open-source framework that combines fine-grained rubric-based assessment with reinforcement learning to mitigate optimism bias in LLM evaluators. We release multidisciplinary scienceQ&A datasets, including adversarial variants, with evaluation scores from multiple LLMs. Independent of proprietary models and human feedback, our approach enables scalable, cost-free evaluation. By advancing reliable LLM-as-a-judge models, this work supports AI alignment and fosters robust, transparent evaluation essential for scientific inquiry and artificial general intelligence.
>
---
#### [new 109] Attributional Safety Failures in Large Language Models under Code-Mixed Perturbations
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大型语言模型（LLM）在混合语言输入下的安全漏洞，探讨其较纯英文输入更易产生有害输出的机制。通过可解释方法分析模型内部归因变化，并区分普遍与文化特异性安全风险，揭示现象驱动因素。**

- **链接: [http://arxiv.org/pdf/2505.14469v1](http://arxiv.org/pdf/2505.14469v1)**

> **作者:** Somnath Banerjee; Pratyush Chatterjee; Shanu Kumar; Sayan Layek; Parag Agrawal; Rima Hazra; Animesh Mukherjee
>
> **摘要:** Recent advancements in LLMs have raised significant safety concerns, particularly when dealing with code-mixed inputs and outputs. Our study systematically investigates the increased susceptibility of LLMs to produce unsafe outputs from code-mixed prompts compared to monolingual English prompts. Utilizing explainability methods, we dissect the internal attribution shifts causing model's harmful behaviors. In addition, we explore cultural dimensions by distinguishing between universally unsafe and culturally-specific unsafe queries. This paper presents novel experimental insights, clarifying the mechanisms driving this phenomenon.
>
---
#### [new 110] Pivot Language for Low-Resource Machine Translation
- **分类: cs.CL; cs.LG; 68T50; I.2.7**

- **简介: 该论文属于低资源机器翻译任务，解决尼泊尔语到英语因平行语料不足导致的翻译质量低下问题。通过采用印地语作为枢纽语言，提出两种方法（Transfer Method和Backtranslation），提升翻译效果，其中全监督方法较此前最优结果提升6.6分SacreBLEU，并分析半监督方法表现稍逊的原因。**

- **链接: [http://arxiv.org/pdf/2505.14553v1](http://arxiv.org/pdf/2505.14553v1)**

> **作者:** Abhimanyu Talwar; Julien Laasri
>
> **备注:** 7 pages, 3 figures, paper dated May 13, 2019
>
> **摘要:** Certain pairs of languages suffer from lack of a parallel corpus which is large in size and diverse in domain. One of the ways this is overcome is via use of a pivot language. In this paper we use Hindi as a pivot language to translate Nepali into English. We describe what makes Hindi a good candidate for the pivot. We discuss ways in which a pivot language can be used, and use two such approaches - the Transfer Method (fully supervised) and Backtranslation (semi-supervised) - to translate Nepali into English. Using the former, we are able to achieve a devtest Set SacreBLEU score of 14.2, which improves the baseline fully supervised score reported by (Guzman et al., 2019) by 6.6 points. While we are slightly below the semi-supervised baseline score of 15.1, we discuss what may have caused this under-performance, and suggest scope for future work.
>
---
#### [new 111] ModRWKV: Transformer Multimodality in Linear Time
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ModRWKV框架，基于RWKV7 RNN架构探索多模态大模型任务，解决Transformer模型计算复杂度高及RNN仅限文本模态的问题。通过轻量异构编码器动态融合多源信息，利用预训练权重加速训练，实验表明现代RNN是多模态LLM的高效替代方案。**

- **链接: [http://arxiv.org/pdf/2505.14505v1](http://arxiv.org/pdf/2505.14505v1)**

> **作者:** Jiale Kang; Ziyin Yue; Qingyu Yin; Jiang Rui; Weile Li; Zening Lu; Zhouran Ji
>
> **摘要:** Currently, most multimodal studies are based on large language models (LLMs) with quadratic-complexity Transformer architectures. While linear models like RNNs enjoy low inference costs, their application has been largely limited to the text-only modality. This work explores the capabilities of modern RNN architectures in multimodal contexts. We propose ModRWKV-a decoupled multimodal framework built upon the RWKV7 architecture as its LLM backbone-which achieves multi-source information fusion through dynamically adaptable heterogeneous modality encoders. We designed the multimodal modules in ModRWKV with an extremely lightweight architecture and, through extensive experiments, identified a configuration that achieves an optimal balance between performance and computational efficiency. ModRWKV leverages the pretrained weights of the RWKV7 LLM for initialization, which significantly accelerates multimodal training. Comparative experiments with different pretrained checkpoints further demonstrate that such initialization plays a crucial role in enhancing the model's ability to understand multimodal signals. Supported by extensive experiments, we conclude that modern RNN architectures present a viable alternative to Transformers in the domain of multimodal large language models (MLLMs). Furthermore, we identify the optimal configuration of the ModRWKV architecture through systematic exploration.
>
---
#### [new 112] Toward Reliable Biomedical Hypothesis Generation: Evaluating Truthfulness and Hallucination in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于生物医学假设生成任务，旨在解决大语言模型（LLMs）生成假设时真实性不足及幻觉问题。研究提出TruthHypo基准评估模型生成真实假设的能力，并开发知识驱动的幻觉检测器KnowHD，通过分析推理步骤中的幻觉，有效筛选可靠假设，加速科学发现。**

- **链接: [http://arxiv.org/pdf/2505.14599v1](http://arxiv.org/pdf/2505.14599v1)**

> **作者:** Guangzhi Xiong; Eric Xie; Corey Williams; Myles Kim; Amir Hassan Shariatmadari; Sikun Guo; Stefan Bekiranov; Aidong Zhang
>
> **备注:** Accepted to IJCAI 2025
>
> **摘要:** Large language models (LLMs) have shown significant potential in scientific disciplines such as biomedicine, particularly in hypothesis generation, where they can analyze vast literature, identify patterns, and suggest research directions. However, a key challenge lies in evaluating the truthfulness of generated hypotheses, as verifying their accuracy often requires substantial time and resources. Additionally, the hallucination problem in LLMs can lead to the generation of hypotheses that appear plausible but are ultimately incorrect, undermining their reliability. To facilitate the systematic study of these challenges, we introduce TruthHypo, a benchmark for assessing the capabilities of LLMs in generating truthful biomedical hypotheses, and KnowHD, a knowledge-based hallucination detector to evaluate how well hypotheses are grounded in existing knowledge. Our results show that LLMs struggle to generate truthful hypotheses. By analyzing hallucinations in reasoning steps, we demonstrate that the groundedness scores provided by KnowHD serve as an effective metric for filtering truthful hypotheses from the diverse outputs of LLMs. Human evaluations further validate the utility of KnowHD in identifying truthful hypotheses and accelerating scientific discovery. Our data and source code are available at https://github.com/Teddy-XiongGZ/TruthHypo.
>
---
#### [new 113] Cross-Lingual Optimization for Language Transfer in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出跨语言优化（CLO）方法，解决监督微调（SFT）在跨语言迁移中过度侧重英语且数据效率低的问题。通过结合英语数据与翻译模型，CLO有效提升目标语言性能同时保持英语能力，尤其在低资源语言中仅需更少数据即超越SFT。**

- **链接: [http://arxiv.org/pdf/2505.14297v1](http://arxiv.org/pdf/2505.14297v1)**

> **作者:** Jungseob Lee; Seongtae Hong; Hyeonseok Moon; Heuiseok Lim
>
> **备注:** Accepted for publication at ACL 2025. Jungseob Lee and Seongtae Hong contributed equally to this work
>
> **摘要:** Adapting large language models to other languages typically employs supervised fine-tuning (SFT) as a standard approach. However, it often suffers from an overemphasis on English performance, a phenomenon that is especially pronounced in data-constrained environments. To overcome these challenges, we propose \textbf{Cross-Lingual Optimization (CLO)} that efficiently transfers an English-centric LLM to a target language while preserving its English capabilities. CLO utilizes publicly available English SFT data and a translation model to enable cross-lingual transfer. We conduct experiments using five models on six languages, each possessing varying levels of resource. Our results show that CLO consistently outperforms SFT in both acquiring target language proficiency and maintaining English performance. Remarkably, in low-resource languages, CLO with only 3,200 samples surpasses SFT with 6,400 samples, demonstrating that CLO can achieve better performance with less data. Furthermore, we find that SFT is particularly sensitive to data quantity in medium and low-resource languages, whereas CLO remains robust. Our comprehensive analysis emphasizes the limitations of SFT and incorporates additional training strategies in CLO to enhance efficiency.
>
---
#### [new 114] Texts or Images? A Fine-grained Analysis on the Effectiveness of Input Representations and Models for Table Question Answering
- **分类: cs.CL**

- **简介: 该论文属于表格问答（TQA）任务，旨在比较表格以文本或图像形式输入时，传统语言模型（LLMs）与多模态模型（MLLMs）的效果差异。通过构建新基准，系统分析问题复杂度和表格大小对模型表现的影响，发现最优组合因场景而异，并提出动态选择方法FRES，提升10%性能。**

- **链接: [http://arxiv.org/pdf/2505.14131v1](http://arxiv.org/pdf/2505.14131v1)**

> **作者:** Wei Zhou; Mohsen Mesgar; Heike Adel; Annemarie Friedrich
>
> **备注:** Accepted at ACL25 (Findings)
>
> **摘要:** In table question answering (TQA), tables are encoded as either texts or images. Prior work suggests that passing images of tables to multi-modal large language models (MLLMs) performs comparably to or even better than using textual input with large language models (LLMs). However, the lack of controlled setups limits fine-grained distinctions between these approaches. In this paper, we conduct the first controlled study on the effectiveness of several combinations of table representations and models from two perspectives: question complexity and table size. We build a new benchmark based on existing TQA datasets. In a systematic analysis of seven pairs of MLLMs and LLMs, we find that the best combination of table representation and model varies across setups. We propose FRES, a method selecting table representations dynamically, and observe a 10% average performance improvement compared to using both representations indiscriminately.
>
---
#### [new 115] OSoRA: Output-Dimension and Singular-Value Initialized Low-Rank Adaptation
- **分类: cs.CL**

- **简介: 该论文提出OSoRA方法，属于LLM参数高效微调任务。旨在降低大模型微调的计算成本。通过结合SVD分解与可学习缩放向量，仅优化输出维度参数，冻结其余矩阵，实现参数量线性增长且性能优于LoRA等方法。**

- **链接: [http://arxiv.org/pdf/2505.14350v1](http://arxiv.org/pdf/2505.14350v1)**

> **作者:** Jialong Han; Si Zhang; Ke Zhang
>
> **摘要:** Fine-tuning Large Language Models (LLMs) has become increasingly challenging due to their massive scale and associated computational costs. Parameter-Efficient Fine-Tuning (PEFT) methodologies have been proposed as computational alternatives; however, their implementations still require significant resources. In this paper, we present OSoRA (Output-Dimension and Singular-Value Initialized Low-Rank Adaptation), a novel PEFT method for LLMs. OSoRA extends Low-Rank Adaptation (LoRA) by integrating Singular Value Decomposition (SVD) with learnable scaling vectors in a unified framework. It first performs an SVD of pre-trained weight matrices, then optimizes an output-dimension vector during training, while keeping the corresponding singular vector matrices frozen. OSoRA substantially reduces computational resource requirements by minimizing the number of trainable parameters during fine-tuning. Comprehensive evaluations across mathematical reasoning, common sense reasoning, and other benchmarks demonstrate that OSoRA achieves comparable or superior performance to state-of-the-art methods like LoRA and VeRA, while maintaining a linear parameter scaling even as the rank increases to higher dimensions. Our ablation studies further confirm that jointly training both the singular values and the output-dimension vector is critical for optimal performance.
>
---
#### [new 116] JOLT-SQL: Joint Loss Tuning of Text-to-SQL with Confusion-aware Noisy Schema Sampling
- **分类: cs.CL**

- **简介: 该论文属于文本到SQL任务，旨在解决监督微调（SFT）方法中存在的多阶段流程复杂及对噪声数据库模式敏感的问题。提出JOLT-SQL框架，通过联合优化模式链接与SQL生成、采用局部双向注意力和噪声模式采样策略，提升模型鲁棒性及效率，在Spider和BIRD数据集上达成了开源模型最优执行精度。**

- **链接: [http://arxiv.org/pdf/2505.14305v1](http://arxiv.org/pdf/2505.14305v1)**

> **作者:** Jinwang Song; Hongying Zan; Kunli Zhang; Lingling Mu; Yingjie Han; Haobo Hua; Min Peng
>
> **备注:** Work in progress. 13 pages, 6 figures
>
> **摘要:** Text-to-SQL, which maps natural language to SQL queries, has benefited greatly from recent advances in Large Language Models (LLMs). While LLMs offer various paradigms for this task, including prompting and supervised fine-tuning (SFT), SFT approaches still face challenges such as complex multi-stage pipelines and poor robustness to noisy schema information. To address these limitations, we present JOLT-SQL, a streamlined single-stage SFT framework that jointly optimizes schema linking and SQL generation via a unified loss. JOLT-SQL employs discriminative schema linking, enhanced by local bidirectional attention, alongside a confusion-aware noisy schema sampling strategy with selective attention to improve robustness under noisy schema conditions. Experiments on the Spider and BIRD benchmarks demonstrate that JOLT-SQL achieves state-of-the-art execution accuracy among comparable-size open-source models, while significantly improving both training and inference efficiency.
>
---
#### [new 117] Probing BERT for German Compound Semantics
- **分类: cs.CL**

- **简介: 该论文通过探针任务探究德语BERT模型对名词复合词语义组合性的编码能力。针对868个标准复合词，测试不同层、词元及大小写模型组合，发现德语BERT的表现显著弱于英语，可能因德语构词更活跃且成分歧义多。任务属模型语义分析，旨在评估并解释跨语言复合词处理差异。**

- **链接: [http://arxiv.org/pdf/2505.14130v1](http://arxiv.org/pdf/2505.14130v1)**

> **作者:** Filip Miletić; Aaron Schmid; Sabine Schulte im Walde
>
> **备注:** Accepted to SwissText 2025
>
> **摘要:** This paper investigates the extent to which pretrained German BERT encodes knowledge of noun compound semantics. We comprehensively vary combinations of target tokens, layers, and cased vs. uncased models, and evaluate them by predicting the compositionality of 868 gold standard compounds. Looking at representational patterns within the transformer architecture, we observe trends comparable to equivalent prior work on English, with compositionality information most easily recoverable in the early layers. However, our strongest results clearly lag behind those reported for English, suggesting an inherently more difficult task in German. This may be due to the higher productivity of compounding in German than in English and the associated increase in constituent-level ambiguity, including in our target compound set.
>
---
#### [new 118] Let's Verify Math Questions Step by Step
- **分类: cs.CL**

- **简介: 该论文提出MathQ-Verify，一种五阶段数学问题验证方法，解决现有模型忽视问题有效性的问题。通过格式验证、条件分解、逻辑矛盾检测及信息完备性检查，筛选不合理题目。构建含2,147题的验证数据集，实验显示其F1值超基线25%，实现高精度题目筛选。**

- **链接: [http://arxiv.org/pdf/2505.13903v1](http://arxiv.org/pdf/2505.13903v1)**

> **作者:** Chengyu Shen; Zhen Hao Wong; Runming He; Hao Liang; Meiyi Qiang; Zimo Meng; Zhengyang Zhao; Bohan Zeng; Zhengzhou Zhu; Bin Cui; Wentao Zhang
>
> **摘要:** Large Language Models (LLMs) have recently achieved remarkable progress in mathematical reasoning. To enable such capabilities, many existing works distill strong reasoning models into long chains of thought or design algorithms to construct high-quality math QA data for training. However, these efforts primarily focus on generating correct reasoning paths and answers, while largely overlooking the validity of the questions themselves. In this work, we propose Math Question Verification (MathQ-Verify), a novel five-stage pipeline designed to rigorously filter ill-posed or under-specified math problems. MathQ-Verify first performs format-level validation to remove redundant instructions and ensure that each question is syntactically well-formed. It then formalizes each question, decomposes it into atomic conditions, and verifies them against mathematical definitions. Next, it detects logical contradictions among these conditions, followed by a goal-oriented completeness check to ensure the question provides sufficient information for solving. To evaluate this task, we use existing benchmarks along with an additional dataset we construct, containing 2,147 math questions with diverse error types, each manually double-validated. Experiments show that MathQ-Verify achieves state-of-the-art performance across multiple benchmarks, improving the F1 score by up to 25 percentage points over the direct verification baseline. It further attains approximately 90% precision and 63% recall through a lightweight model voting scheme. MathQ-Verify offers a scalable and accurate solution for curating reliable mathematical datasets, reducing label noise and avoiding unnecessary computation on invalid questions. Our code and data are available at https://github.com/scuuy/MathQ-Verify.
>
---
#### [new 119] EfficientLLM: Efficiency in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EfficientLLM基准，研究大语言模型效率优化问题，解决其参数与上下文扩展导致的高成本。系统评估架构（MQA/MLA/GQA/NSA、稀疏MoE）、微调（LoRA/RSLoRA/DoRA）及推理（量化）技术，定义六项指标，测试上百模型组合，揭示效率权衡、任务依赖及跨模态通用性，提供效率-性能优化指导。**

- **链接: [http://arxiv.org/pdf/2505.13840v1](http://arxiv.org/pdf/2505.13840v1)**

> **作者:** Zhengqing Yuan; Weixiang Sun; Yixin Liu; Huichi Zhou; Rong Zhou; Yiyang Li; Zheyuan Zhang; Wei Song; Yue Huang; Haolong Jia; Keerthiram Murugesan; Yu Wang; Lifang He; Jianfeng Gao; Lichao Sun; Yanfang Ye
>
> **摘要:** Large Language Models (LLMs) have driven significant progress, yet their growing parameter counts and context windows incur prohibitive compute, energy, and monetary costs. We introduce EfficientLLM, a novel benchmark and the first comprehensive empirical study evaluating efficiency techniques for LLMs at scale. Conducted on a production-class cluster (48xGH200, 8xH200 GPUs), our study systematically explores three key axes: (1) architecture pretraining (efficient attention variants: MQA, GQA, MLA, NSA; sparse Mixture-of-Experts (MoE)), (2) fine-tuning (parameter-efficient methods: LoRA, RSLoRA, DoRA), and (3) inference (quantization methods: int4, float16). We define six fine-grained metrics (Memory Utilization, Compute Utilization, Latency, Throughput, Energy Consumption, Compression Rate) to capture hardware saturation, latency-throughput balance, and carbon cost. Evaluating over 100 model-technique pairs (0.5B-72B parameters), we derive three core insights: (i) Efficiency involves quantifiable trade-offs: no single method is universally optimal; e.g., MoE reduces FLOPs and improves accuracy but increases VRAM by 40%, while int4 quantization cuts memory/energy by up to 3.9x at a 3-5% accuracy drop. (ii) Optima are task- and scale-dependent: MQA offers optimal memory-latency trade-offs for constrained devices, MLA achieves lowest perplexity for quality-critical tasks, and RSLoRA surpasses LoRA efficiency only beyond 14B parameters. (iii) Techniques generalize across modalities: we extend evaluations to Large Vision Models (Stable Diffusion 3.5, Wan 2.1) and Vision-Language Models (Qwen2.5-VL), confirming effective transferability. By open-sourcing datasets, evaluation pipelines, and leaderboards, EfficientLLM provides essential guidance for researchers and engineers navigating the efficiency-performance landscape of next-generation foundation models.
>
---
#### [new 120] Clarifying orthography: Orthographic transparency as compressibility
- **分类: cs.CL; cs.IT; math.IT**

- **简介: 论文提出基于算法信息论的正字法透明度量化方法，解决跨文字类型统一衡量缺失的问题。通过互压缩理论结合不规则拼写与规则复杂度，利用神经模型预编码长度计算透明度，测试22种语言验证有效性，提供通用衡量标准。**

- **链接: [http://arxiv.org/pdf/2505.13657v1](http://arxiv.org/pdf/2505.13657v1)**

> **作者:** Charles J. Torres; Richard Futrell
>
> **摘要:** Orthographic transparency -- how directly spelling is related to sound -- lacks a unified, script-agnostic metric. Using ideas from algorithmic information theory, we quantify orthographic transparency in terms of the mutual compressibility between orthographic and phonological strings. Our measure provides a principled way to combine two factors that decrease orthographic transparency, capturing both irregular spellings and rule complexity in one quantity. We estimate our transparency measure using prequential code-lengths derived from neural sequence models. Evaluating 22 languages across a broad range of script types (alphabetic, abjad, abugida, syllabic, logographic) confirms common intuitions about relative transparency of scripts. Mutual compressibility offers a simple, principled, and general yardstick for orthographic transparency.
>
---
#### [new 121] Context Reasoner: Incentivizing Reasoning Capability for Contextualized Privacy and Safety Compliance via Reinforcement Learning
- **分类: cs.CL**

- **简介: 论文提出基于强化学习的Context Reasoner，解决LLMs在隐私安全合规中的推理不足问题。通过情境完整性理论结合GDPR等法规，设计规则奖励机制，提升合规性（+17.64%）及推理能力（MMLU+2.05%，LegalBench+8.98%）。**

- **链接: [http://arxiv.org/pdf/2505.14585v1](http://arxiv.org/pdf/2505.14585v1)**

> **作者:** Wenbin Hu; Haoran Li; Huihao Jing; Qi Hu; Ziqian Zeng; Sirui Han; Heli Xu; Tianshu Chu; Peizhao Hu; Yangqiu Song
>
> **摘要:** While Large Language Models (LLMs) exhibit remarkable capabilities, they also introduce significant safety and privacy risks. Current mitigation strategies often fail to preserve contextual reasoning capabilities in risky scenarios. Instead, they rely heavily on sensitive pattern matching to protect LLMs, which limits the scope. Furthermore, they overlook established safety and privacy standards, leading to systemic risks for legal compliance. To address these gaps, we formulate safety and privacy issues into contextualized compliance problems following the Contextual Integrity (CI) theory. Under the CI framework, we align our model with three critical regulatory standards: GDPR, EU AI Act, and HIPAA. Specifically, we employ reinforcement learning (RL) with a rule-based reward to incentivize contextual reasoning capabilities while enhancing compliance with safety and privacy norms. Through extensive experiments, we demonstrate that our method not only significantly enhances legal compliance (achieving a +17.64% accuracy improvement in safety/privacy benchmarks) but also further improves general reasoning capability. For OpenThinker-7B, a strong reasoning model that significantly outperforms its base model Qwen2.5-7B-Instruct across diverse subjects, our method enhances its general reasoning capabilities, with +2.05% and +8.98% accuracy improvement on the MMLU and LegalBench benchmark, respectively.
>
---
#### [new 122] Reward Reasoning Model
- **分类: cs.CL**

- **简介: 该论文提出奖励推理模型（RRMs），通过链式推理和强化学习优化奖励模型，在复杂任务中利用测试时算力提升奖励准确性，解决传统模型难以处理隐含奖励的问题，实验显示其跨领域性能优越。**

- **链接: [http://arxiv.org/pdf/2505.14674v1](http://arxiv.org/pdf/2505.14674v1)**

> **作者:** Jiaxin Guo; Zewen Chi; Li Dong; Qingxiu Dong; Xun Wu; Shaohan Huang; Furu Wei
>
> **摘要:** Reward models play a critical role in guiding large language models toward outputs that align with human expectations. However, an open challenge remains in effectively utilizing test-time compute to enhance reward model performance. In this work, we introduce Reward Reasoning Models (RRMs), which are specifically designed to execute a deliberate reasoning process before generating final rewards. Through chain-of-thought reasoning, RRMs leverage additional test-time compute for complex queries where appropriate rewards are not immediately apparent. To develop RRMs, we implement a reinforcement learning framework that fosters self-evolved reward reasoning capabilities without requiring explicit reasoning traces as training data. Experimental results demonstrate that RRMs achieve superior performance on reward modeling benchmarks across diverse domains. Notably, we show that RRMs can adaptively exploit test-time compute to further improve reward accuracy. The pretrained reward reasoning models are available at https://huggingface.co/Reward-Reasoning.
>
---
#### [new 123] Studying the Role of Input-Neighbor Overlap in Retrieval-Augmented Language Models Training Efficiency
- **分类: cs.CL**

- **简介: 该论文研究检索增强语言模型中查询与检索上下文的重叠对训练效率的影响，旨在优化模型性能与资源使用。通过系统实验发现重叠超过阈值可显著提升测试困惑度并加速训练，提出合成上下文方法减少训练时间40%，验证于问答任务，证明优化检索机制的潜力。**

- **链接: [http://arxiv.org/pdf/2505.14309v1](http://arxiv.org/pdf/2505.14309v1)**

> **作者:** Ehsan Doostmohammadi; Marco Kuhlmann
>
> **摘要:** Retrieval-augmented language models have demonstrated performance comparable to much larger models while requiring fewer computational resources. The effectiveness of these models crucially depends on the overlap between query and retrieved context, but the optimal degree of this overlap remains unexplored. In this paper, we systematically investigate how varying levels of query--context overlap affect model performance during both training and inference. Our experiments reveal that increased overlap initially has minimal effect, but substantially improves test-time perplexity and accelerates model learning above a critical threshold. Building on these findings, we demonstrate that deliberately increasing overlap through synthetic context can enhance data efficiency and reduce training time by approximately 40\% without compromising performance. We specifically generate synthetic context through paraphrasing queries. We validate our perplexity-based findings on question-answering tasks, confirming that the benefits of retrieval-augmented language modeling extend to practical applications. Our results provide empirical evidence of significant optimization potential for retrieval mechanisms in language model pretraining.
>
---
#### [new 124] Enhanced Multimodal Aspect-Based Sentiment Analysis by LLM-Generated Rationales
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态方面基础情感分析（MABSA）任务，针对小型模型知识有限及大模型在ABSA效果欠佳的问题，提出LRSA框架：利用LLM生成解释作为rationales注入小型模型，并通过双交叉注意力机制增强模态交互，提升方面与情感识别效果，实验显示其在三个基准数据集上优于基线。**

- **链接: [http://arxiv.org/pdf/2505.14499v1](http://arxiv.org/pdf/2505.14499v1)**

> **作者:** Jun Cao; Jiyi Li; Ziwei Yang; Renjie Zhou
>
> **摘要:** There has been growing interest in Multimodal Aspect-Based Sentiment Analysis (MABSA) in recent years. Existing methods predominantly rely on pre-trained small language models (SLMs) to collect information related to aspects and sentiments from both image and text, with an aim to align these two modalities. However, small SLMs possess limited capacity and knowledge, often resulting in inaccurate identification of meaning, aspects, sentiments, and their interconnections in textual and visual data. On the other hand, Large language models (LLMs) have shown exceptional capabilities in various tasks by effectively exploring fine-grained information in multimodal data. However, some studies indicate that LLMs still fall short compared to fine-tuned small models in the field of ABSA. Based on these findings, we propose a novel framework, termed LRSA, which combines the decision-making capabilities of SLMs with additional information provided by LLMs for MABSA. Specifically, we inject explanations generated by LLMs as rationales into SLMs and employ a dual cross-attention mechanism for enhancing feature interaction and fusion, thereby augmenting the SLMs' ability to identify aspects and sentiments. We evaluated our method using two baseline models, numerous experiments highlight the superiority of our approach on three widely-used benchmarks, indicating its generalizability and applicability to most pre-trained models for MABSA.
>
---
#### [new 125] Log-Augmented Generation: Scaling Test-Time Reasoning with Reusable Computation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于测试时推理任务。解决大语言模型无法有效复用过往任务推理的问题。提出Log-Augmented Generation（LAG），通过存储精选任务的KV缓存，并在新任务中直接复用其推理过程，提升性能，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14398v1](http://arxiv.org/pdf/2505.14398v1)**

> **作者:** Peter Baile Chen; Yi Zhang; Dan Roth; Samuel Madden; Jacob Andreas; Michael Cafarella
>
> **备注:** Data and code are available at https://peterbaile.github.io/lag/
>
> **摘要:** While humans naturally learn and adapt from past experiences, large language models (LLMs) and their agentic counterparts struggle to retain reasoning from previous tasks and apply them in future contexts. To address this limitation, we propose a novel framework, log-augmented generation (LAG) that directly reuses prior computation and reasoning from past logs at test time to enhance model's ability to learn from previous tasks and perform better on new, unseen challenges, all while keeping the system efficient and scalable. Specifically, our system represents task logs using key-value (KV) caches, encoding the full reasoning context of prior tasks while storing KV caches for only a selected subset of tokens. When a new task arises, LAG retrieves the KV values from relevant logs to augment generation. Our approach differs from reflection-based memory mechanisms by directly reusing prior reasoning and computations without requiring additional steps for knowledge extraction or distillation. Our method also goes beyond existing KV caching techniques, which primarily target efficiency gains rather than improving accuracy. Experiments on knowledge- and reasoning-intensive datasets demonstrate that our method significantly outperforms standard agentic systems that do not utilize logs, as well as existing solutions based on reflection and KV cache techniques.
>
---
#### [new 126] Time-R1: Towards Comprehensive Temporal Reasoning in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于时序推理任务，旨在解决LLMs在时间理解、未来预测及创意生成上的不足。提出Time-R1框架，通过三阶段强化学习课程（动态奖励系统驱动）提升模型的时序能力，实现跨时段推理与泛化，超越大规模模型表现，并开源数据集Time-Bench。**

- **链接: [http://arxiv.org/pdf/2505.13508v1](http://arxiv.org/pdf/2505.13508v1)**

> **作者:** Zijia Liu; Peixuan Han; Haofei Yu; Haoru Li; Jiaxuan You
>
> **摘要:** Large Language Models (LLMs) demonstrate impressive capabilities but lack robust temporal intelligence, struggling to integrate reasoning about the past with predictions and plausible generations of the future. Meanwhile, existing methods typically target isolated temporal skills, such as question answering about past events or basic forecasting, and exhibit poor generalization, particularly when dealing with events beyond their knowledge cutoff or requiring creative foresight. To address these limitations, we introduce \textit{Time-R1}, the first framework to endow a moderate-sized (3B-parameter) LLM with comprehensive temporal abilities: understanding, prediction, and creative generation. Our approach features a novel three-stage development path; the first two constitute a \textit{reinforcement learning (RL) curriculum} driven by a meticulously designed dynamic rule-based reward system. This framework progressively builds (1) foundational temporal understanding and logical event-time mappings from historical data, (2) future event prediction skills for events beyond its knowledge cutoff, and finally (3) enables remarkable generalization to creative future scenario generation without any fine-tuning. Strikingly, experiments demonstrate that Time-R1 outperforms models over 200 times larger, including the state-of-the-art 671B DeepSeek-R1, on highly challenging future event prediction and creative scenario generation benchmarks. This work provides strong evidence that thoughtfully engineered, progressive RL fine-tuning allows smaller, efficient models to achieve superior temporal performance, offering a practical and scalable path towards truly time-aware AI. To foster further research, we also release \textit{Time-Bench}, a large-scale multi-task temporal reasoning dataset derived from 10 years of news data, and our series of \textit{Time-R1} checkpoints.
>
---
#### [new 127] THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于神经机器翻译任务，针对现有MoE模型在任务知识应用（依赖不可用领域/语言知识）和局部token路由（忽略上下文）的缺陷，提出THOR-MoE。其通过分层路由：先预测领域/语言标签分配任务级专家，再结合上下文信息优化token路由，提升翻译精度，兼容现有MoE架构，实验显示显著性能提升。**

- **链接: [http://arxiv.org/pdf/2505.14173v1](http://arxiv.org/pdf/2505.14173v1)**

> **作者:** Yunlong Liang; Fandong Meng; Jie Zhou
>
> **备注:** Accepted to ACL 2025 main conference
>
> **摘要:** The sparse Mixture-of-Experts (MoE) has achieved significant progress for neural machine translation (NMT). However, there exist two limitations in current MoE solutions which may lead to sub-optimal performance: 1) they directly use the task knowledge of NMT into MoE (\emph{e.g.}, domain/linguistics-specific knowledge), which are generally unavailable at practical application and neglect the naturally grouped domain/linguistic properties; 2) the expert selection only depends on the localized token representation without considering the context, which fully grasps the state of each token in a global view. To address the above limitations, we propose THOR-MoE via arming the MoE with hierarchical task-guided and context-responsive routing policies. Specifically, it 1) firstly predicts the domain/language label and then extracts mixed domain/language representation to allocate task-level experts in a hierarchical manner; 2) injects the context information to enhance the token routing from the pre-selected task-level experts set, which can help each token to be accurately routed to more specialized and suitable experts. Extensive experiments on multi-domain translation and multilingual translation benchmarks with different architectures consistently demonstrate the superior performance of THOR-MoE. Additionally, the THOR-MoE operates as a plug-and-play module compatible with existing Top-$k$~\cite{shazeer2017} and Top-$p$~\cite{huang-etal-2024-harder} routing schemes, ensuring broad applicability across diverse MoE architectures. For instance, compared with vanilla Top-$p$~\cite{huang-etal-2024-harder} routing, the context-aware manner can achieve an average improvement of 0.75 BLEU with less than 22\% activated parameters on multi-domain translation tasks.
>
---
#### [new 128] KORGym: A Dynamic Game Platform for LLM Reasoning Evaluation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出动态游戏平台KORGym，用于评估LLM推理能力。针对现有基准领域局限性，设计超50个文本/视觉游戏，支持交互式多轮及强化学习评估。实验分析19种LLM和8种VLM的推理模式，揭示模型家族特性及闭源模型优势，探究模态、策略等影响因素，为复杂环境下的推理评估提供新方法。**

- **链接: [http://arxiv.org/pdf/2505.14552v1](http://arxiv.org/pdf/2505.14552v1)**

> **作者:** Jiajun Shi; Jian Yang; Jiaheng Liu; Xingyuan Bu; Jiangjie Chen; Junting Zhou; Kaijing Ma; Zhoufutu Wen; Bingli Wang; Yancheng He; Liang Song; Hualei Zhu; Shilong Li; Xingjian Wang; Wei Zhang; Ruibin Yuan; Yifan Yao; Wenjun Yang; Yunli Wang; Siyuan Fang; Siyu Yuan; Qianyu He; Xiangru Tang; Yingshui Tan; Wangchunshu Zhou; Zhaoxiang Zhang; Zhoujun Li; Wenhao Huang; Ge Zhang
>
> **备注:** 22 pages
>
> **摘要:** Recent advancements in large language models (LLMs) underscore the need for more comprehensive evaluation methods to accurately assess their reasoning capabilities. Existing benchmarks are often domain-specific and thus cannot fully capture an LLM's general reasoning potential. To address this limitation, we introduce the Knowledge Orthogonal Reasoning Gymnasium (KORGym), a dynamic evaluation platform inspired by KOR-Bench and Gymnasium. KORGym offers over fifty games in either textual or visual formats and supports interactive, multi-turn assessments with reinforcement learning scenarios. Using KORGym, we conduct extensive experiments on 19 LLMs and 8 VLMs, revealing consistent reasoning patterns within model families and demonstrating the superior performance of closed-source models. Further analysis examines the effects of modality, reasoning strategies, reinforcement learning techniques, and response length on model performance. We expect KORGym to become a valuable resource for advancing LLM reasoning research and developing evaluation methodologies suited to complex, interactive environments.
>
---
#### [new 129] Prior Prompt Engineering for Reinforcement Fine-Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习微调（RFT）任务，旨在探索通过优化前置提示工程（pPE）引导语言模型习得特定行为。针对现有RFT对prior prompt设计关注不足的问题，研究将五种推理时提示策略转化为pPE方法，实验表明pPE模型表现优于传统提示，其中null-example策略效果最佳，并验证不同策略塑造差异化行为风格。**

- **链接: [http://arxiv.org/pdf/2505.14157v1](http://arxiv.org/pdf/2505.14157v1)**

> **作者:** Pittawat Taveekitworachai; Potsawee Manakul; Sarana Nutanong; Kunat Pipatanakul
>
> **备注:** 25 pages, 42 figures
>
> **摘要:** This paper investigates prior prompt engineering (pPE) in the context of reinforcement fine-tuning (RFT), where language models (LMs) are incentivized to exhibit behaviors that maximize performance through reward signals. While existing RFT research has primarily focused on algorithms, reward shaping, and data curation, the design of the prior prompt--the instructions prepended to queries during training to elicit behaviors such as step-by-step reasoning--remains underexplored. We investigate whether different pPE approaches can guide LMs to internalize distinct behaviors after RFT. Inspired by inference-time prompt engineering (iPE), we translate five representative iPE strategies--reasoning, planning, code-based reasoning, knowledge recall, and null-example utilization--into corresponding pPE approaches. We experiment with Qwen2.5-7B using each of the pPE approaches, then evaluate performance on in-domain and out-of-domain benchmarks (e.g., AIME2024, HumanEval+, and GPQA-Diamond). Our results show that all pPE-trained models surpass their iPE-prompted counterparts, with the null-example pPE approach achieving the largest average performance gain and the highest improvement on AIME2024 and GPQA-Diamond, surpassing the commonly used reasoning approach. Furthermore, by adapting a behavior-classification framework, we demonstrate that different pPE strategies instill distinct behavioral styles in the resulting models. These findings position pPE as a powerful yet understudied axis for RFT.
>
---
#### [new 130] Beyond Chains: Bridging Large Language Models and Knowledge Bases in Complex Question Answering
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于知识图谱问答（KBQA）任务，旨在解决大语言模型（LLM）回答复杂问题时知识过时、幻觉及透明度不足，以及现有KG-RAG方法仅处理简单链式问题的局限。提出PDRR框架：通过预测问题类型、分解为结构化三元组、从KB检索信息并引导LLM推理，有效处理链式与非链式复杂问题，实验显示其性能更优。**

- **链接: [http://arxiv.org/pdf/2505.14099v1](http://arxiv.org/pdf/2505.14099v1)**

> **作者:** Yihua Zhu; Qianying Liu; Akiko Aizawa; Hidetoshi Shimodaira
>
> **摘要:** Knowledge Base Question Answering (KBQA) aims to answer natural language questions using structured knowledge from KBs. While LLM-only approaches offer generalization, they suffer from outdated knowledge, hallucinations, and lack of transparency. Chain-based KG-RAG methods address these issues by incorporating external KBs, but are limited to simple chain-structured questions due to the absence of planning and logical structuring. Inspired by semantic parsing methods, we propose PDRR: a four-stage framework consisting of Predict, Decompose, Retrieve, and Reason. Our method first predicts the question type and decomposes the question into structured triples. Then retrieves relevant information from KBs and guides the LLM as an agent to reason over and complete the decomposed triples. Experimental results demonstrate that PDRR consistently outperforms existing methods across various LLM backbones and achieves superior performance on both chain-structured and non-chain complex questions.
>
---
#### [new 131] EEG-to-Text Translation: A Model for Deciphering Human Brain Activity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于EEG-to-text翻译任务，旨在解决现有模型解码脑电信号为文本的性能不足问题。提出R1 Translator模型，结合双向LSTM编码EEG序列特征与预训练Transformer解码生成文本，在ROUGE、CER、WER等指标上超越T5和Brain Translator基线模型。**

- **链接: [http://arxiv.org/pdf/2505.13936v1](http://arxiv.org/pdf/2505.13936v1)**

> **作者:** Saydul Akbar Murad; Ashim Dahal; Nick Rahimi
>
> **摘要:** With the rapid advancement of large language models like Gemini, GPT, and others, bridging the gap between the human brain and language processing has become an important area of focus. To address this challenge, researchers have developed various models to decode EEG signals into text. However, these models still face significant performance limitations. To overcome these shortcomings, we propose a new model, R1 Translator, which aims to improve the performance of EEG-to-text decoding. The R1 Translator model combines a bidirectional LSTM encoder with a pretrained transformer-based decoder, utilizing EEG features to produce high-quality text outputs. The model processes EEG embeddings through the LSTM to capture sequential dependencies, which are then fed into the transformer decoder for effective text generation. The R1 Translator excels in ROUGE metrics, outperforming both T5 (previous research) and Brain Translator. Specifically, R1 achieves a ROUGE-1 score of 38.00% (P), which is up to 9% higher than T5 (34.89%) and 3% better than Brain (35.69%). It also leads in ROUGE-L, with a F1 score of 32.51%, outperforming T5 by 3% (29.67%) and Brain by 2% (30.38%). In terms of CER, R1 achieves a CER of 0.5795, which is 2% lower than T5 (0.5917) and 4% lower than Brain (0.6001). Additionally, R1 performs better in WER with a score of 0.7280, outperforming T5 by 4.3% (0.7610) and Brain by 3.6% (0.7553). Code is available at https://github.com/Mmurrad/EEG-To-text.
>
---
#### [new 132] PL-FGSA: A Prompt Learning Framework for Fine-Grained Sentiment Analysis Based on MindSpore
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于细粒度情感分析任务，旨在解决传统方法依赖特定架构和大量标注数据导致的泛化性差问题。提出PL-FGSA框架，结合提示学习与轻量级TextCNN，统一处理方面提取、情感分类及因果解释，提升可解释性与低资源场景性能，在三个数据集上超越传统方法。**

- **链接: [http://arxiv.org/pdf/2505.14165v1](http://arxiv.org/pdf/2505.14165v1)**

> **作者:** Zhenkai Qin; Jiajing He; Qiao Fang
>
> **摘要:** Fine-grained sentiment analysis (FGSA) aims to identify sentiment polarity toward specific aspects within a text, enabling more precise opinion mining in domains such as product reviews and social media. However, traditional FGSA approaches often require task-specific architectures and extensive annotated data, limiting their generalization and scalability. To address these challenges, we propose PL-FGSA, a unified prompt learning-based framework implemented using the MindSpore platform, which integrates prompt design with a lightweight TextCNN backbone. Our method reformulates FGSA as a multi-task prompt-augmented generation problem, jointly tackling aspect extraction, sentiment classification, and causal explanation in a unified paradigm. By leveraging prompt-based guidance, PL-FGSA enhances interpretability and achieves strong performance under both full-data and low-resource conditions. Experiments on three benchmark datasets-SST-2, SemEval-2014 Task 4, and MAMS-demonstrate that our model consistently outperforms traditional fine-tuning methods and achieves F1-scores of 0.922, 0.694, and 0.597, respectively. These results validate the effectiveness of prompt-based generalization and highlight the practical value of PL-FGSA for real-world sentiment analysis tasks.
>
---
#### [new 133] QA-prompting: Improving Summarization with Large Language Models using Question-Answering
- **分类: cs.CL**

- **简介: 该论文属于文本摘要任务，旨在解决长文本处理中的位置偏差导致关键信息提取不足问题。提出QA-prompting方法，通过问答中间步骤增强上下文，无需微调或流水线，实验显示ROUGE分数提升29%，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14347v1](http://arxiv.org/pdf/2505.14347v1)**

> **作者:** Neelabh Sinha
>
> **备注:** Submitted to ARR
>
> **摘要:** Language Models (LMs) have revolutionized natural language processing, enabling high-quality text generation through prompting and in-context learning. However, models often struggle with long-context summarization due to positional biases, leading to suboptimal extraction of critical information. There are techniques to improve this with fine-tuning, pipelining, or using complex techniques, which have their own challenges. To solve these challenges, we propose QA-prompting - a simple prompting method for summarization that utilizes question-answering as an intermediate step prior to summary generation. Our method extracts key information and enriches the context of text to mitigate positional biases and improve summarization in a single LM call per task without requiring fine-tuning or pipelining. Experiments on multiple datasets belonging to different domains using ten state-of-the-art pre-trained models demonstrate that QA-prompting outperforms baseline and other state-of-the-art methods, achieving up to 29% improvement in ROUGE scores. This provides an effective and scalable solution for summarization and highlights the importance of domain-specific question selection for optimal performance.
>
---
#### [new 134] CAFES: A Collaborative Multi-Agent Framework for Multi-Granular Multimodal Essay Scoring
- **分类: cs.CL; cs.AI**

- **简介: 论文提出CAVES框架，用于多模态作文自动评分。解决传统方法泛化性差、多模态感知不足及MLLM模型产生幻觉、评分偏离人工判断的问题。通过三代理器协同（初评、反馈聚合、迭代优化）提升评分与人类一致性，实验显示QWK提升21%。**

- **链接: [http://arxiv.org/pdf/2505.13965v1](http://arxiv.org/pdf/2505.13965v1)**

> **作者:** Jiamin Su; Yibo Yan; Zhuoran Gao; Han Zhang; Xiang Liu; Xuming Hu
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2502.11916
>
> **摘要:** Automated Essay Scoring (AES) is crucial for modern education, particularly with the increasing prevalence of multimodal assessments. However, traditional AES methods struggle with evaluation generalizability and multimodal perception, while even recent Multimodal Large Language Model (MLLM)-based approaches can produce hallucinated justifications and scores misaligned with human judgment. To address the limitations, we introduce CAFES, the first collaborative multi-agent framework specifically designed for AES. It orchestrates three specialized agents: an Initial Scorer for rapid, trait-specific evaluations; a Feedback Pool Manager to aggregate detailed, evidence-grounded strengths; and a Reflective Scorer that iteratively refines scores based on this feedback to enhance human alignment. Extensive experiments, using state-of-the-art MLLMs, achieve an average relative improvement of 21% in Quadratic Weighted Kappa (QWK) against ground truth, especially for grammatical and lexical diversity. Our proposed CAFES framework paves the way for an intelligent multimodal AES system. The code will be available upon acceptance.
>
---
#### [new 135] PlanGPT-VL: Enhancing Urban Planning with Domain-Specific Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文提出PlanGPT-VL，首个针对城市规划地图的领域专用视觉语言模型。解决现有VLM在解析规划地图（如土地使用、功能分区）时精度不足的问题。通过PlanAnno-V数据合成、结构化验证抑制幻觉及高效微调方法，其7B参数模型超越72B通用模型，在规划分析和教育场景中实现高精度与轻量化。**

- **链接: [http://arxiv.org/pdf/2505.14481v1](http://arxiv.org/pdf/2505.14481v1)**

> **作者:** He Zhu; Junyou Su; Minxi Chen; Wen Wang; Yijie Deng; Guanhua Chen; Wenjia Zhang
>
> **摘要:** In the field of urban planning, existing Vision-Language Models (VLMs) frequently fail to effectively analyze and evaluate planning maps, despite the critical importance of these visual elements for urban planners and related educational contexts. Planning maps, which visualize land use, infrastructure layouts, and functional zoning, require specialized understanding of spatial configurations, regulatory requirements, and multi-scale analysis. To address this challenge, we introduce PlanGPT-VL, the first domain-specific Vision-Language Model tailored specifically for urban planning maps. PlanGPT-VL employs three innovative approaches: (1) PlanAnno-V framework for high-quality VQA data synthesis, (2) Critical Point Thinking to reduce hallucinations through structured verification, and (3) comprehensive training methodology combining Supervised Fine-Tuning with frozen vision encoder parameters. Through systematic evaluation on our proposed PlanBench-V benchmark, we demonstrate that PlanGPT-VL significantly outperforms general-purpose state-of-the-art VLMs in specialized planning map interpretation tasks, offering urban planning professionals a reliable tool for map analysis, assessment, and educational applications while maintaining high factual accuracy. Our lightweight 7B parameter model achieves comparable performance to models exceeding 72B parameters, demonstrating efficient domain specialization without sacrificing performance.
>
---
#### [new 136] Truth or Twist? Optimal Model Selection for Reliable Label Flipping Evaluation in LLM-based Counterfactuals
- **分类: cs.CL**

- **简介: 该论文研究LLM反事实数据增强中模型选择问题，解决标签翻转评估因判断模型差异导致结果不一致的难题。通过定义四种生成器-判断模型关系，实验15种判断模型并结合用户研究，发现独立非微调判断模型评估最可靠，但与人工评估仍存差距，提示需人工介入自动化流程。**

- **链接: [http://arxiv.org/pdf/2505.13972v1](http://arxiv.org/pdf/2505.13972v1)**

> **作者:** Qianli Wang; Van Bach Nguyen; Nils Feldhus; Luis Felipe Villa-Arenas; Christin Seifert; Sebastian Möller; Vera Schmitt
>
> **备注:** in submission
>
> **摘要:** Counterfactual examples are widely employed to enhance the performance and robustness of large language models (LLMs) through counterfactual data augmentation (CDA). However, the selection of the judge model used to evaluate label flipping, the primary metric for assessing the validity of generated counterfactuals for CDA, yields inconsistent results. To decipher this, we define four types of relationships between the counterfactual generator and judge models. Through extensive experiments involving two state-of-the-art LLM-based methods, three datasets, five generator models, and 15 judge models, complemented by a user study (n = 90), we demonstrate that judge models with an independent, non-fine-tuned relationship to the generator model provide the most reliable label flipping evaluations. Relationships between the generator and judge models, which are closely aligned with the user study for CDA, result in better model performance and robustness. Nevertheless, we find that the gap between the most effective judge models and the results obtained from the user study remains considerably large. This suggests that a fully automated pipeline for CDA may be inadequate and requires human intervention.
>
---
#### [new 137] Guided Search Strategies in Non-Serializable Environments with Applications to Software Engineering Agents
- **分类: cs.SE; cs.CL**

- **简介: 该论文研究非序列化环境下提升大模型任务成功率的搜索策略。针对LLMs在多步任务中性能波动及传统搜索方法不适用容器等环境的问题，提出基于动作值函数的1步前瞻与轨迹选择策略。在软件工程基准测试中，使Qwen-72B成功率提升至40.8%，创开放权重模型新纪录，并验证了方法对闭源模型的迁移性。**

- **链接: [http://arxiv.org/pdf/2505.13652v1](http://arxiv.org/pdf/2505.13652v1)**

> **作者:** Karina Zainullina; Alexander Golubev; Maria Trofimova; Sergei Polezhaev; Ibragim Badertdinov; Daria Litvintseva; Simon Karasik; Filipp Fisin; Sergei Skvortsov; Maksim Nekrashevich; Anton Shevtsov; Boris Yangel
>
> **备注:** ICML
>
> **摘要:** Large language models (LLMs) have recently achieved remarkable results in complex multi-step tasks, such as mathematical reasoning and agentic software engineering. However, they often struggle to maintain consistent performance across multiple solution attempts. One effective approach to narrow the gap between average-case and best-case performance is guided test-time search, which explores multiple solution paths to identify the most promising one. Unfortunately, effective search techniques (e.g. MCTS) are often unsuitable for non-serializable RL environments, such as Docker containers, where intermediate environment states cannot be easily saved and restored. We investigate two complementary search strategies applicable to such environments: 1-step lookahead and trajectory selection, both guided by a learned action-value function estimator. On the SWE-bench Verified benchmark, a key testbed for agentic software engineering, we find these methods to double the average success rate of a fine-tuned Qwen-72B model, achieving 40.8%, the new state-of-the-art for open-weights models. Additionally, we show that these techniques are transferable to more advanced closed models, yielding similar improvements with GPT-4o.
>
---
#### [new 138] PandaGuard: Systematic Evaluation of LLM Safety in the Era of Jailbreaking Attacks
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出PandaGuard框架，系统评估LLM对抗jailbreak攻击的安全性，解决现有评估分散、不可复现的问题。通过构建多代理系统整合19种攻击、12种防御及多种判断策略，开发PandaBench基准测试49种模型，揭示防御效能 trade-off 和评估差异，推动可复现安全研究。（98字）**

- **链接: [http://arxiv.org/pdf/2505.13862v1](http://arxiv.org/pdf/2505.13862v1)**

> **作者:** Guobin Shen; Dongcheng Zhao; Linghao Feng; Xiang He; Jihang Wang; Sicheng Shen; Haibo Tong; Yiting Dong; Jindong Li; Xiang Zheng; Yi Zeng
>
> **摘要:** Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.
>
---
#### [new 139] Forensic deepfake audio detection using segmental speech features
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于深度伪造音频检测任务，旨在通过分段语音特征提升检测效果。针对现有方法对全局特征依赖但效果有限的问题，研究探索了与人类发声机制相关的分段声学特征（如音素、重音等），发现其在识别深度伪造音频中更有效，而全局特征价值较低，为法医语音鉴定提供了新方法。**

- **链接: [http://arxiv.org/pdf/2505.13847v1](http://arxiv.org/pdf/2505.13847v1)**

> **作者:** Tianle Yang; Chengzhe Sun; Siwei Lyu; Phil Rose
>
> **摘要:** This study explores the potential of using acoustic features of segmental speech sounds to detect deepfake audio. These features are highly interpretable because of their close relationship with human articulatory processes and are expected to be more difficult for deepfake models to replicate. The results demonstrate that certain segmental features commonly used in forensic voice comparison are effective in identifying deep-fakes, whereas some global features provide little value. These findings underscore the need to approach audio deepfake detection differently for forensic voice comparison and offer a new perspective on leveraging segmental features for this purpose.
>
---
#### [new 140] Ice Cream Doesn't Cause Drowning: Benchmarking LLMs Against Statistical Pitfalls in Causal Inference
- **分类: cs.AI; cs.CL; cs.LG; stat.ME; stat.ML; 62-08, 68T50, 68T05, 68T01, 68T07, 62-07, 68U35, 62C99; I.2.7; I.2.6; I.2.0; I.5.1; I.5.4; F.2.2; H.2.8; G.3**

- **简介: 该论文属于因果推理评估任务，旨在解决大语言模型（LLMs）在统计因果推理中的不足。现有基准任务简单，易使模型忽略统计陷阱（如辛普森悖论）。论文提出CausalPitfalls基准，通过多难度结构化任务及评分标准，采用直接提示和代码辅助两种协议评估LLMs，发现其因果推理局限性，为开发可靠系统提供指导。**

- **链接: [http://arxiv.org/pdf/2505.13770v1](http://arxiv.org/pdf/2505.13770v1)**

> **作者:** Jin Du; Li Chen; Xun Xian; An Luo; Fangqiao Tian; Ganghua Wang; Charles Doss; Xiaotong Shen; Jie Ding
>
> **摘要:** Reliable causal inference is essential for making decisions in high-stakes areas like medicine, economics, and public policy. However, it remains unclear whether large language models (LLMs) can handle rigorous and trustworthy statistical causal inference. Current benchmarks usually involve simplified tasks. For example, these tasks might only ask LLMs to identify semantic causal relationships or draw conclusions directly from raw data. As a result, models may overlook important statistical pitfalls, such as Simpson's paradox or selection bias. This oversight limits the applicability of LLMs in the real world. To address these limitations, we propose CausalPitfalls, a comprehensive benchmark designed to rigorously evaluate the capability of LLMs in overcoming common causal inference pitfalls. Our benchmark features structured challenges across multiple difficulty levels, each paired with grading rubrics. This approach allows us to quantitatively measure both causal reasoning capabilities and the reliability of LLMs' responses. We evaluate models using two protocols: (1) direct prompting, which assesses intrinsic causal reasoning, and (2) code-assisted prompting, where models generate executable code for explicit statistical analysis. Additionally, we validate the effectiveness of this judge by comparing its scoring with assessments from human experts. Our results reveal significant limitations in current LLMs when performing statistical causal inference. The CausalPitfalls benchmark provides essential guidance and quantitative metrics to advance the development of trustworthy causal reasoning systems.
>
---
#### [new 141] Causal Cartographer: From Mapping to Reasoning Over Counterfactual Worlds
- **分类: cs.AI; cs.CL; cs.LG; I.2.3; I.2.6; I.2.7; G.2.2; G.3; J.1**

- **简介: 该论文提出Causal Cartographer框架，解决大语言模型缺乏真实世界因果推理能力的问题。通过图检索增强生成代理提取因果关系构建知识网络，并设计因果约束推理代理，实现可靠反事实推理，提升任务鲁棒性并降低成本。**

- **链接: [http://arxiv.org/pdf/2505.14396v1](http://arxiv.org/pdf/2505.14396v1)**

> **作者:** Gaël Gendron; Jože M. Rožanec; Michael Witbrock; Gillian Dobbie
>
> **备注:** 29 pages, 9 pages for the main paper, 20 pages for the references and appendix, 25 figures
>
> **摘要:** Causal world models are systems that can answer counterfactual questions about an environment of interest, i.e. predict how it would have evolved if an arbitrary subset of events had been realized differently. It requires understanding the underlying causes behind chains of events and conducting causal inference for arbitrary unseen distributions. So far, this task eludes foundation models, notably large language models (LLMs), which do not have demonstrated causal reasoning capabilities beyond the memorization of existing causal relationships. Furthermore, evaluating counterfactuals in real-world applications is challenging since only the factual world is observed, limiting evaluation to synthetic datasets. We address these problems by explicitly extracting and modeling causal relationships and propose the Causal Cartographer framework. First, we introduce a graph retrieval-augmented generation agent tasked to retrieve causal relationships from data. This approach allows us to construct a large network of real-world causal relationships that can serve as a repository of causal knowledge and build real-world counterfactuals. In addition, we create a counterfactual reasoning agent constrained by causal relationships to perform reliable step-by-step causal inference. We show that our approach can extract causal knowledge and improve the robustness of LLMs for causal reasoning tasks while reducing inference costs and spurious correlations.
>
---
#### [new 142] InfiFPO: Implicit Model Fusion via Preference Optimization in Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型（LLM）融合任务，针对现有偏好优化（PA）方法忽略源模型概率信息、依赖复杂词汇对齐的问题，提出InfiFPO方法：通过序列级概率融合替代传统响应输出，结合概率裁剪与最大-margin策略，提升枢纽模型对人类偏好的对齐及知识蒸馏效果，在11个基准测试中显著提升数学、编码等任务性能。**

- **链接: [http://arxiv.org/pdf/2505.13878v1](http://arxiv.org/pdf/2505.13878v1)**

> **作者:** Yanggan Gu; Zhaoyi Yan; Yuanyi Wang; Yiming Zhang; Qi Zhou; Fei Wu; Hongxia Yang
>
> **备注:** 17 pages
>
> **摘要:** Model fusion combines multiple Large Language Models (LLMs) with different strengths into a more powerful, integrated model through lightweight training methods. Existing works on model fusion focus primarily on supervised fine-tuning (SFT), leaving preference alignment (PA) --a critical phase for enhancing LLM performance--largely unexplored. The current few fusion methods on PA phase, like WRPO, simplify the process by utilizing only response outputs from source models while discarding their probability information. To address this limitation, we propose InfiFPO, a preference optimization method for implicit model fusion. InfiFPO replaces the reference model in Direct Preference Optimization (DPO) with a fused source model that synthesizes multi-source probabilities at the sequence level, circumventing complex vocabulary alignment challenges in previous works and meanwhile maintaining the probability information. By introducing probability clipping and max-margin fusion strategies, InfiFPO enables the pivot model to align with human preferences while effectively distilling knowledge from source models. Comprehensive experiments on 11 widely-used benchmarks demonstrate that InfiFPO consistently outperforms existing model fusion and preference optimization methods. When using Phi-4 as the pivot model, InfiFPO improve its average performance from 79.95 to 83.33 on 11 benchmarks, significantly improving its capabilities in mathematics, coding, and reasoning tasks.
>
---
#### [new 143] s3: You Don't Need That Much Data to Train a Search Agent via RL
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出s3框架，解决现有RL训练搜索代理需大量数据、忽略下游效用及模型兼容性差的问题。通过解耦检索与生成模块，采用"Gain Beyond RAG"奖励函数，仅需2.4k样本即超越基线方法，在多领域QA任务中表现更优。**

- **链接: [http://arxiv.org/pdf/2505.14146v1](http://arxiv.org/pdf/2505.14146v1)**

> **作者:** Pengcheng Jiang; Xueqiang Xu; Jiacheng Lin; Jinfeng Xiao; Zifeng Wang; Jimeng Sun; Jiawei Han
>
> **摘要:** Retrieval-augmented generation (RAG) systems empower large language models (LLMs) to access external knowledge during inference. Recent advances have enabled LLMs to act as search agents via reinforcement learning (RL), improving information acquisition through multi-turn interactions with retrieval engines. However, existing approaches either optimize retrieval using search-only metrics (e.g., NDCG) that ignore downstream utility or fine-tune the entire LLM to jointly reason and retrieve-entangling retrieval with generation and limiting the real search utility and compatibility with frozen or proprietary models. In this work, we propose s3, a lightweight, model-agnostic framework that decouples the searcher from the generator and trains the searcher using a Gain Beyond RAG reward: the improvement in generation accuracy over naive RAG. s3 requires only 2.4k training samples to outperform baselines trained on over 70x more data, consistently delivering stronger downstream performance across six general QA and five medical QA benchmarks.
>
---
#### [new 144] PAST: Phonetic-Acoustic Speech Tokenizer
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文提出PAST，一种端到端语音分词框架，联合建模音素与声学信息，无需外部预训练模型。旨在解决传统方法依赖预训练模型及实时性不足的问题，通过监督学习与因果变体设计，提升语音表示、重建及实时应用性能，增强语音语言模型效果。**

- **链接: [http://arxiv.org/pdf/2505.14470v1](http://arxiv.org/pdf/2505.14470v1)**

> **作者:** Nadav Har-Tuv; Or Tal; Yossi Adi
>
> **摘要:** We present PAST, a novel end-to-end framework that jointly models phonetic information alongside signal reconstruction, eliminating the need for external pretrained models. Unlike previous approaches that rely on pretrained self-supervised models, PAST employs supervised phonetic data, directly integrating domain knowledge into the tokenization process via auxiliary tasks. Additionally, we introduce a streamable, causal variant of PAST, enabling real-time speech applications. Results demonstrate that PAST surpasses existing evaluated baseline tokenizers across common evaluation metrics, including phonetic representation and speech reconstruction. Notably, PAST also achieves superior performance when serving as a speech representation for speech language models, further highlighting its effectiveness as a foundation for spoken language generation. To foster further research, we release the full implementation. For code, model checkpoints, and samples see: https://pages.cs.huji.ac.il/adiyoss-lab/PAST
>
---
#### [new 145] Two Experts Are All You Need for Steering Thinking: Reinforcing Cognitive Effort in MoE Reasoning Models Without Additional Training
- **分类: cs.AI; cs.CL; cs.CV; cs.IR; cs.LG**

- **简介: 该论文属于优化MoE推理模型的任务，旨在解决其推理过程中的过思考和欠思考问题。提出RICE方法，通过nPMI识别"认知专家"（如触发<think>的模块），在推理阶段引导结构化思维，提升推理效率与跨领域泛化能力，实验显示其优于现有方法且无需额外训练。**

- **链接: [http://arxiv.org/pdf/2505.14681v1](http://arxiv.org/pdf/2505.14681v1)**

> **作者:** Mengru Wang; Xingyu Chen; Yue Wang; Zhiwei He; Jiahao Xu; Tian Liang; Qiuzhi Liu; Yunzhi Yao; Wenxuan Wang; Ruotian Ma; Haitao Mi; Ningyu Zhang; Zhaopeng Tu; Xiaolong Li; Dong Yu
>
> **备注:** Work in progress
>
> **摘要:** Mixture-of-Experts (MoE) architectures within Large Reasoning Models (LRMs) have achieved impressive reasoning capabilities by selectively activating experts to facilitate structured cognitive processes. Despite notable advances, existing reasoning models often suffer from cognitive inefficiencies like overthinking and underthinking. To address these limitations, we introduce a novel inference-time steering methodology called Reinforcing Cognitive Experts (RICE), designed to improve reasoning performance without additional training or complex heuristics. Leveraging normalized Pointwise Mutual Information (nPMI), we systematically identify specialized experts, termed ''cognitive experts'' that orchestrate meta-level reasoning operations characterized by tokens like ''<think>''. Empirical evaluations with leading MoE-based LRMs (DeepSeek-R1 and Qwen3-235B) on rigorous quantitative and scientific reasoning benchmarks demonstrate noticeable and consistent improvements in reasoning accuracy, cognitive efficiency, and cross-domain generalization. Crucially, our lightweight approach substantially outperforms prevalent reasoning-steering techniques, such as prompt design and decoding constraints, while preserving the model's general instruction-following skills. These results highlight reinforcing cognitive experts as a promising, practical, and interpretable direction to enhance cognitive efficiency within advanced reasoning models.
>
---
#### [new 146] Contrastive Cross-Course Knowledge Tracing via Concept Graph Guided Knowledge Transfer
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于跨课程知识追踪任务，旨在解决单课程数据限制导致的学习者知识状态理解不全面的问题。提出TransKT方法，通过零样本LLM构建跨课程概念图，利用图卷积网络与对比学习实现知识转移，整合多课程学习行为的语义特征，提升知识状态预测的准确性。**

- **链接: [http://arxiv.org/pdf/2505.13489v1](http://arxiv.org/pdf/2505.13489v1)**

> **作者:** Wenkang Han; Wang Lin; Liya Hu; Zhenlong Dai; Yiyun Zhou; Mengze Li; Zemin Liu; Chang Yao; Jingyuan Chen
>
> **备注:** Accepted by IJCAI 2025
>
> **摘要:** Knowledge tracing (KT) aims to predict learners' future performance based on historical learning interactions. However, existing KT models predominantly focus on data from a single course, limiting their ability to capture a comprehensive understanding of learners' knowledge states. In this paper, we propose TransKT, a contrastive cross-course knowledge tracing method that leverages concept graph guided knowledge transfer to model the relationships between learning behaviors across different courses, thereby enhancing knowledge state estimation. Specifically, TransKT constructs a cross-course concept graph by leveraging zero-shot Large Language Model (LLM) prompts to establish implicit links between related concepts across different courses. This graph serves as the foundation for knowledge transfer, enabling the model to integrate and enhance the semantic features of learners' interactions across courses. Furthermore, TransKT includes an LLM-to-LM pipeline for incorporating summarized semantic features, which significantly improves the performance of Graph Convolutional Networks (GCNs) used for knowledge transfer. Additionally, TransKT employs a contrastive objective that aligns single-course and cross-course knowledge states, thereby refining the model's ability to provide a more robust and accurate representation of learners' overall knowledge states.
>
---
#### [new 147] LLM-Based Compact Reranking with Document Features for Scientific Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于科学信息检索任务，针对传统重排序中第一阶段检索效果差、候选文档覆盖不足的问题，提出无训练框架CoRank。通过提取文档语义特征（如类别、关键词）进行粗排序，再对top结果细粒度重排序，提升检索性能（nDCG@10提升7.7）。**

- **链接: [http://arxiv.org/pdf/2505.13757v1](http://arxiv.org/pdf/2505.13757v1)**

> **作者:** Runchu Tian; Xueqiang Xu; Bowen Jin; SeongKu Kang; Jiawei Han
>
> **备注:** 17 pages, 4 figures
>
> **摘要:** Scientific retrieval is essential for advancing academic discovery. Within this process, document reranking plays a critical role by refining first-stage retrieval results. However, large language model (LLM) listwise reranking faces unique challenges in the scientific domain. First-stage retrieval is often suboptimal in the scientific domain, so relevant documents are ranked lower. Moreover, conventional listwise reranking uses the full text of candidate documents in the context window, limiting the number of candidates that can be considered. As a result, many relevant documents are excluded before reranking, which constrains overall retrieval performance. To address these challenges, we explore compact document representations based on semantic features such as categories, sections, and keywords, and propose a training-free, model-agnostic reranking framework for scientific retrieval called CoRank. The framework involves three stages: (i) offline extraction of document-level features, (ii) coarse reranking using these compact representations, and (iii) fine-grained reranking on full texts of the top candidates from stage (ii). This hybrid design provides a high-level abstraction of document semantics, expands candidate coverage, and retains critical details required for precise ranking. Experiments on LitSearch and CSFCube show that CoRank significantly improves reranking performance across different LLM backbones, increasing nDCG@10 from 32.0 to 39.7. Overall, these results highlight the value of information extraction for reranking in scientific retrieval.
>
---
#### [new 148] Scaling Law for Quantization-Aware Training
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究量化感知训练（QAT）的扩展规律，针对4位精度（W4A4）量化误差问题，提出统一的量化误差模型，分析其与模型规模、训练数据量及量化粒度的关系，通过268次实验发现误差来源并提出混合精度量化方法优化瓶颈，为QAT优化提供理论指导。**

- **链接: [http://arxiv.org/pdf/2505.14302v1](http://arxiv.org/pdf/2505.14302v1)**

> **作者:** Mengzhao Chen; Chaoyi Zhang; Jing Liu; Yutao Zeng; Zeyue Xue; Zhiheng Liu; Yunshui Li; Jin Ma; Jie Huang; Xun Zhou; Ping Luo
>
> **备注:** A unified scaling law for QAT that models quantization error as a function of model size, training data volume, and quantization group size
>
> **摘要:** Large language models (LLMs) demand substantial computational and memory resources, creating deployment challenges. Quantization-aware training (QAT) addresses these challenges by reducing model precision while maintaining performance. However, the scaling behavior of QAT, especially at 4-bit precision (W4A4), is not well understood. Existing QAT scaling laws often ignore key factors such as the number of training tokens and quantization granularity, which limits their applicability. This paper proposes a unified scaling law for QAT that models quantization error as a function of model size, training data volume, and quantization group size. Through 268 QAT experiments, we show that quantization error decreases as model size increases, but rises with more training tokens and coarser quantization granularity. To identify the sources of W4A4 quantization error, we decompose it into weight and activation components. Both components follow the overall trend of W4A4 quantization error, but with different sensitivities. Specifically, weight quantization error increases more rapidly with more training tokens. Further analysis shows that the activation quantization error in the FC2 layer, caused by outliers, is the primary bottleneck of W4A4 QAT quantization error. By applying mixed-precision quantization to address this bottleneck, we demonstrate that weight and activation quantization errors can converge to similar levels. Additionally, with more training data, weight quantization error eventually exceeds activation quantization error, suggesting that reducing weight quantization error is also important in such scenarios. These findings offer key insights for improving QAT research and development.
>
---
#### [new 149] Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出神经符号方法提升LLMs的数学证明可靠性。针对LLMs逻辑推理不足，通过检索类问题引导生成并结合形式验证器反馈修正证明。实验显示准确率提升58%-70%，推动LLMs在严谨推理任务中的应用。**

- **链接: [http://arxiv.org/pdf/2505.14479v1](http://arxiv.org/pdf/2505.14479v1)**

> **作者:** Oren Sultan; Eitan Stern; Dafna Shahaf
>
> **备注:** long paper
>
> **摘要:** Large language models (LLMs) struggle with formal domains that require rigorous logical deduction and symbolic reasoning, such as mathematical proof generation. We propose a neuro-symbolic approach that combines LLMs' generative strengths with structured components to overcome this challenge. As a proof-of-concept, we focus on geometry problems. Our approach is two-fold: (1) we retrieve analogous problems and use their proofs to guide the LLM, and (2) a formal verifier evaluates the generated proofs and provides feedback, helping the model fix incorrect proofs. We demonstrate that our method significantly improves proof accuracy for OpenAI's o1 model (58%-70% improvement); both analogous problems and the verifier's feedback contribute to these gains. More broadly, shifting to LLMs that generate provably correct conclusions could dramatically improve their reliability, accuracy and consistency, unlocking complex tasks and critical real-world applications that require trustworthiness.
>
---
#### [new 150] SafetyNet: Detecting Harmful Outputs in LLMs by Modeling and Monitoring Deceptive Behaviors
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI安全领域，旨在通过建模和监测大语言模型的欺骗行为，实时检测有害输出。针对后门触发产生的暴力、色情等风险内容，提出SafetyNet框架：采用无监督方法监测模型在不同表征空间的行为特征，识别因果机制并抵御模型通过改变特征关系或表征方式的规避行为，实现96%检测准确率。**

- **链接: [http://arxiv.org/pdf/2505.14300v1](http://arxiv.org/pdf/2505.14300v1)**

> **作者:** Maheep Chaudhary; Fazl Barez
>
> **摘要:** High-risk industries like nuclear and aviation use real-time monitoring to detect dangerous system conditions. Similarly, Large Language Models (LLMs) need monitoring safeguards. We propose a real-time framework to predict harmful AI outputs before they occur by using an unsupervised approach that treats normal behavior as the baseline and harmful outputs as outliers. Our study focuses specifically on backdoor-triggered responses -- where specific input phrases activate hidden vulnerabilities causing the model to generate unsafe content like violence, pornography, or hate speech. We address two key challenges: (1) identifying true causal indicators rather than surface correlations, and (2) preventing advanced models from deception -- deliberately evading monitoring systems. Hence, we approach this problem from an unsupervised lens by drawing parallels to human deception: just as humans exhibit physical indicators while lying, we investigate whether LLMs display distinct internal behavioral signatures when generating harmful content. Our study addresses two critical challenges: 1) designing monitoring systems that capture true causal indicators rather than superficial correlations; and 2)preventing intentional evasion by increasingly capable "Future models''. Our findings show that models can produce harmful content through causal mechanisms and can become deceptive by: (a) alternating between linear and non-linear representations, and (b) modifying feature relationships. To counter this, we developed Safety-Net -- a multi-detector framework that monitors different representation dimensions, successfully detecting harmful behavior even when information is shifted across representational spaces to evade individual monitors. Our evaluation shows 96% accuracy in detecting harmful cases using our unsupervised ensemble approach.
>
---
#### [new 151] Efficient Agent Training for Computer Use
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出PC Agent-E框架，解决计算机使用代理训练中高质量轨迹数据不足的问题。通过少量人类示范（312条轨迹）结合Claude 3.7生成合成数据，提升模型训练效率。实验显示其性能超现有方法，并验证跨操作系统泛化能力，同时发布新基准WindowsAgentArena-V2。任务属强化学习领域，目标以小规模高质量数据高效训练通用计算机操作代理。**

- **链接: [http://arxiv.org/pdf/2505.13909v1](http://arxiv.org/pdf/2505.13909v1)**

> **作者:** Yanheng He; Jiahe Jin; Pengfei Liu
>
> **备注:** We open-source our entire suite of code, data, and models to facilitate future research at https://github.com/GAIR-NLP/PC-Agent-E
>
> **摘要:** Scaling up high-quality trajectory data has long been a critical bottleneck for developing human-like computer use agents. We introduce PC Agent-E, an efficient agent training framework that significantly reduces reliance on large-scale human demonstrations. Starting with just 312 human-annotated computer use trajectories, we further improved data quality by synthesizing diverse action decisions with Claude 3.7 Sonnet. Trained on these enriched trajectories, our PC Agent-E model achieved a remarkable 141% relative improvement, surpassing the strong Claude 3.7 Sonnet with extended thinking on WindowsAgentArena-V2, an improved benchmark we also released. Furthermore, PC Agent-E demonstrates strong generalizability to different operating systems on OSWorld. Our findings suggest that strong computer use capabilities can be stimulated from a small amount of high-quality trajectory data.
>
---
#### [new 152] Reasoning Models Better Express Their Confidence
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究模型信心校准任务，解决大语言模型（LLMs）信心表达不准确的问题。通过测试6个推理模型在6个数据集的表现，发现其因"慢思考"行为（如探索替代方案、回溯）动态调整信心，校准效果优于非推理模型，并验证非推理模型通过引导可提升信心校准。**

- **链接: [http://arxiv.org/pdf/2505.14489v1](http://arxiv.org/pdf/2505.14489v1)**

> **作者:** Dongkeun Yoon; Seungone Kim; Sohee Yang; Sunkyoung Kim; Soyeon Kim; Yongil Kim; Eunbi Choi; Yireun Kim; Minjoon Seo
>
> **备注:** Work in progress
>
> **摘要:** Despite their strengths, large language models (LLMs) often fail to communicate their confidence accurately, making it difficult to assess when they might be wrong and limiting their reliability. In this work, we demonstrate that reasoning models-LLMs that engage in extended chain-of-thought (CoT) reasoning-exhibit superior performance not only in problem-solving but also in accurately expressing their confidence. Specifically, we benchmark six reasoning models across six datasets and find that they achieve strictly better confidence calibration than their non-reasoning counterparts in 33 out of the 36 settings. Our detailed analysis reveals that these gains in calibration stem from the slow thinking behaviors of reasoning models-such as exploring alternative approaches and backtracking-which enable them to adjust their confidence dynamically throughout their CoT, making it progressively more accurate. In particular, we find that reasoning models become increasingly better calibrated as their CoT unfolds, a trend not observed in non-reasoning models. Moreover, removing slow thinking behaviors from the CoT leads to a significant drop in calibration. Lastly, we show that these gains are not exclusive to reasoning models-non-reasoning models also benefit when guided to perform slow thinking via in-context learning.
>
---
#### [new 153] AAPO: Enhance the Reasoning Capabilities of LLMs with Advantage Momentum
- **分类: cs.LG; cs.CL**

- **简介: 该论文属强化学习优化大语言模型推理能力任务。针对现有组相对优势方法（如GRPO）在优势值趋近零时训练效率低的问题，提出AAPO算法，通过动量增强优势优化交叉熵损失，实验显示其在数学推理任务中表现更优。**

- **链接: [http://arxiv.org/pdf/2505.14264v1](http://arxiv.org/pdf/2505.14264v1)**

> **作者:** Jian Xiong; Jingbo Zhou; Jingyong Ye; Dejing Dou
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Reinforcement learning (RL) has emerged as an effective approach for enhancing the reasoning capabilities of large language models (LLMs), especially in scenarios where supervised fine-tuning (SFT) falls short due to limited chain-of-thought (CoT) data. Among RL-based post-training methods, group relative advantage estimation, as exemplified by Group Relative Policy Optimization (GRPO), has attracted considerable attention for eliminating the dependency on the value model, thereby simplifying training compared to traditional approaches like Proximal Policy Optimization (PPO). However, we observe that exsiting group relative advantage estimation method still suffers from training inefficiencies, particularly when the estimated advantage approaches zero. To address this limitation, we propose Advantage-Augmented Policy Optimization (AAPO), a novel RL algorithm that optimizes the cross-entropy (CE) loss using advantages enhanced through a momentum-based estimation scheme. This approach effectively mitigates the inefficiencies associated with group relative advantage estimation. Experimental results on multiple mathematical reasoning benchmarks demonstrate the superior performance of AAPO.
>
---
#### [new 154] RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态视觉文化理解任务，旨在解决视觉语言模型（VLMs）在文化细微差别解析上的不足。提出基准RAVENEA，包含文化聚焦视觉问答（cVQA）和文化导向图像描述（cIC）任务，整合超1万份人工筛选的维基文档，评估7种检索器与14种VLMs，验证检索增强可显著提升模型性能（cVQA提升3.2%，cIC提升6.2%）。**

- **链接: [http://arxiv.org/pdf/2505.14462v1](http://arxiv.org/pdf/2505.14462v1)**

> **作者:** Jiaang Li; Yifei Yuan; Wenyan Li; Mohammad Aliannejadi; Daniel Hershcovich; Anders Søgaard; Ivan Vulić; Wenxuan Zhang; Paul Pu Liang; Yang Deng; Serge Belongie
>
> **摘要:** As vision-language models (VLMs) become increasingly integrated into daily life, the need for accurate visual culture understanding is becoming critical. Yet, these models frequently fall short in interpreting cultural nuances effectively. Prior work has demonstrated the effectiveness of retrieval-augmented generation (RAG) in enhancing cultural understanding in text-only settings, while its application in multimodal scenarios remains underexplored. To bridge this gap, we introduce RAVENEA (Retrieval-Augmented Visual culturE uNdErstAnding), a new benchmark designed to advance visual culture understanding through retrieval, focusing on two tasks: culture-focused visual question answering (cVQA) and culture-informed image captioning (cIC). RAVENEA extends existing datasets by integrating over 10,000 Wikipedia documents curated and ranked by human annotators. With RAVENEA, we train and evaluate seven multimodal retrievers for each image query, and measure the downstream impact of retrieval-augmented inputs across fourteen state-of-the-art VLMs. Our results show that lightweight VLMs, when augmented with culture-aware retrieval, outperform their non-augmented counterparts (by at least 3.2% absolute on cVQA and 6.2% absolute on cIC). This highlights the value of retrieval-augmented methods and culturally inclusive benchmarks for multimodal understanding.
>
---
#### [new 155] RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医学影像报告生成任务，针对现有方法忽视大模型内部知识导致冗余的问题，提出RADAR框架：先提取模型与专家图像分类一致的内部知识，再检索补充外部知识，融合二者生成更准确的报告，在多个数据集上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14318v1](http://arxiv.org/pdf/2505.14318v1)**

> **作者:** Wenjun Hou; Yi Cheng; Kaishuai Xu; Heng Li; Yan Hu; Wenjie Li; Jiang Liu
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities in various domains, including radiology report generation. Previous approaches have attempted to utilize multimodal LLMs for this task, enhancing their performance through the integration of domain-specific knowledge retrieval. However, these approaches often overlook the knowledge already embedded within the LLMs, leading to redundant information integration and inefficient utilization of learned representations. To address this limitation, we propose RADAR, a framework for enhancing radiology report generation with supplementary knowledge injection. RADAR improves report generation by systematically leveraging both the internal knowledge of an LLM and externally retrieved information. Specifically, it first extracts the model's acquired knowledge that aligns with expert image-based classification outputs. It then retrieves relevant supplementary knowledge to further enrich this information. Finally, by aggregating both sources, RADAR generates more accurate and informative radiology reports. Extensive experiments on MIMIC-CXR, CheXpert-Plus, and IU X-ray demonstrate that our model outperforms state-of-the-art LLMs in both language quality and clinical accuracy
>
---
#### [new 156] Textual Steering Vectors Can Improve Visual Understanding in Multimodal Large Language Models
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文研究通过文本引导向量提升多模态大模型视觉理解。针对MLLMs缺乏有效行为引导技术的问题，提出利用文本模型生成steering向量（如稀疏自编码器、均值偏移），实验显示其显著提高空间关系和计数任务准确率（+7.3%），优于提示方法且泛化性好。**

- **链接: [http://arxiv.org/pdf/2505.14071v1](http://arxiv.org/pdf/2505.14071v1)**

> **作者:** Woody Haosheng Gan; Deqing Fu; Julian Asilis; Ollie Liu; Dani Yogatama; Vatsal Sharan; Robin Jia; Willie Neiswanger
>
> **摘要:** Steering methods have emerged as effective and targeted tools for guiding large language models' (LLMs) behavior without modifying their parameters. Multimodal large language models (MLLMs), however, do not currently enjoy the same suite of techniques, due in part to their recency and architectural diversity. Inspired by this gap, we investigate whether MLLMs can be steered using vectors derived from their text-only LLM backbone, via sparse autoencoders (SAEs), mean shift, and linear probing. We find that text-derived steering consistently enhances multimodal accuracy across diverse MLLM architectures and visual tasks. In particular, mean shift boosts spatial relationship accuracy on CV-Bench by up to +7.3% and counting accuracy by up to +3.3%, outperforming prompting and exhibiting strong generalization to out-of-distribution datasets. These results highlight textual steering vectors as a powerful, efficient mechanism for enhancing grounding in MLLMs with minimal additional data collection and computational overhead.
>
---
#### [new 157] PRL: Prompts from Reinforcement Learning
- **分类: cs.AI; cs.CL**

- **简介: 论文提出基于强化学习的PRL方法，自动生成提示以解决依赖专家直觉和捕捉微妙语义线索的挑战，生成新示例并提升文本分类、摘要等任务表现，优于APE和EvoPrompt等现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14412v1](http://arxiv.org/pdf/2505.14412v1)**

> **作者:** Paweł Batorski; Adrian Kosmala; Paul Swoboda
>
> **摘要:** Effective prompt engineering remains a central challenge in fully harnessing the capabilities of LLMs. While well-designed prompts can dramatically enhance performance, crafting them typically demands expert intuition and a nuanced understanding of the task. Moreover, the most impactful prompts often hinge on subtle semantic cues, ones that may elude human perception but are crucial for guiding LLM behavior. In this paper, we introduce PRL (Prompts from Reinforcement Learning), a novel RL-based approach for automatic prompt generation. Unlike previous methods, PRL can produce novel few-shot examples that were not seen during training. Our approach achieves state-of-the-art performance across a range of benchmarks, including text classification, simplification, and summarization. On the classification task, it surpasses prior methods by 2.58% over APE and 1.00% over EvoPrompt. Additionally, it improves the average ROUGE scores on the summarization task by 4.32 over APE and by 2.12 over EvoPrompt and the SARI score on simplification by 6.93 over APE and by 6.01 over EvoPrompt. Our code is available at https://github.com/Batorskq/prl .
>
---
#### [new 158] MedEIR: A Specialized Medical Embedding Model for Enhanced Information Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出MedEIR模型，针对现有嵌入模型在医疗领域语义理解、长文本处理及跨任务通用性上的不足，通过联合优化医疗-通用双领域模型与分词器，并采用ALiBi技术处理长序列（8192 token），在60亿token预训练和300万句对微调后，于多个基准测试中超越Jina和MiniLM，实现医疗与通用NLP任务的高效信息检索。**

- **链接: [http://arxiv.org/pdf/2505.13482v1](http://arxiv.org/pdf/2505.13482v1)**

> **作者:** Anand Selvadurai; Jasheen Shaik; Girish Chandrasekar; ShriRadhaKrishnan Balamurugan; Eswara Reddy
>
> **备注:** 9 pages, 1 figure. This manuscript is a substantial revision of a previously submitted paper. We have explicitly clarified novelty, strengthened scholarly depth, and expanded experimental validation
>
> **摘要:** Embedding models have become essential for retrieval-augmented generation (RAG) tasks, semantic clustering, and text re-ranking. But despite their growing use, many of these come with notable limitations. For example, Jina fails to capture the semantic content of medical documents, while models such as MiniLM often perform poorly on long-form documents. Domain-adapted models, while specialized, often underperform in general-purpose tasks, reducing their overall applicability. General-domain tokenizers often misinterpret medical vocabulary. The limitations of current embedding models, whether in tokenization accuracy, domain comprehension, or handling long sequences, highlight the need for more versatile solutions. In this work, we present MedEIR, a novel embedding model and tokenizer jointly optimized for both medical and general NLP tasks, incorporating ALiBi-based long-context processing to support sequences of up to 8,192 tokens. MedEIR was pre-trained on only 6 billion tokens, significantly fewer than Jina's, followed by fine-tuning on 3 million sentence pairs. MedEIR consistently outperforms Jina V2 and MiniLM across MTEB benchmarks, achieving top scores on ArguAna (55.24), NFCorpus (38.44), MedicalQARetrieval (74.25), SciFact (72.04), and TRECCOVID (79.56). These results highlight the potential of MedEIR as a highly effective embedding model, demonstrating strong performance across both general-purpose and domain-specific tasks and outperforming existing models on multiple benchmarks.
>
---
#### [new 159] ProMind-LLM: Proactive Mental Health Care via Causal Reasoning with Sensor Data
- **分类: cs.AI; cs.CL**

- **简介: 该论文属心理健康风险评估任务。针对现有方法依赖主观文本导致预测不可靠的问题，提出ProMind-LLM，融合传感器获取的客观行为数据与主观记录，通过领域预训练、数值数据优化及因果推理机制提升评估可靠性，实验显示优于通用LLMs。**

- **链接: [http://arxiv.org/pdf/2505.14038v1](http://arxiv.org/pdf/2505.14038v1)**

> **作者:** Xinzhe Zheng; Sijie Ji; Jiawei Sun; Renqi Chen; Wei Gao; Mani Srivastava
>
> **摘要:** Mental health risk is a critical global public health challenge, necessitating innovative and reliable assessment methods. With the development of large language models (LLMs), they stand out to be a promising tool for explainable mental health care applications. Nevertheless, existing approaches predominantly rely on subjective textual mental records, which can be distorted by inherent mental uncertainties, leading to inconsistent and unreliable predictions. To address these limitations, this paper introduces ProMind-LLM. We investigate an innovative approach integrating objective behavior data as complementary information alongside subjective mental records for robust mental health risk assessment. Specifically, ProMind-LLM incorporates a comprehensive pipeline that includes domain-specific pretraining to tailor the LLM for mental health contexts, a self-refine mechanism to optimize the processing of numerical behavioral data, and causal chain-of-thought reasoning to enhance the reliability and interpretability of its predictions. Evaluations of two real-world datasets, PMData and Globem, demonstrate the effectiveness of our proposed methods, achieving substantial improvements over general LLMs. We anticipate that ProMind-LLM will pave the way for more dependable, interpretable, and scalable mental health case solutions.
>
---
#### [new 160] Beyond Text: Unveiling Privacy Vulnerabilities in Multi-modal Retrieval-Augmented Generation
- **分类: cs.CR; cs.CL**

- **简介: 该论文研究多模态检索增强生成（MRAG）系统的隐私漏洞，针对视觉-语言及语音-语言模态，提出黑盒环境下通过结构化提示攻击提取隐私的方法，揭示模型可能直接泄露检索内容或间接暴露敏感信息，呼吁开发隐私保护技术。**

- **链接: [http://arxiv.org/pdf/2505.13957v1](http://arxiv.org/pdf/2505.13957v1)**

> **作者:** Jiankun Zhang; Shenglai Zeng; Jie Ren; Tianqi Zheng; Hui Liu; Xianfeng Tang; Hui Liu; Yi Chang
>
> **摘要:** Multimodal Retrieval-Augmented Generation (MRAG) systems enhance LMMs by integrating external multimodal databases, but introduce unexplored privacy vulnerabilities. While text-based RAG privacy risks have been studied, multimodal data presents unique challenges. We provide the first systematic analysis of MRAG privacy vulnerabilities across vision-language and speech-language modalities. Using a novel compositional structured prompt attack in a black-box setting, we demonstrate how attackers can extract private information by manipulating queries. Our experiments reveal that LMMs can both directly generate outputs resembling retrieved content and produce descriptions that indirectly expose sensitive information, highlighting the urgent need for robust privacy-preserving MRAG techniques.
>
---
#### [new 161] MLZero: A Multi-Agent System for End-to-end Machine Learning Automation
- **分类: cs.MA; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出MLZero，一个多智能体系统，解决现有AutoML依赖人工配置及LLM代码生成缺陷。通过认知感知模块处理多模态数据，结合语义与情景记忆优化代码生成，实验证明其在多基准测试中显著优于现有方法，尤其用小型LLM仍表现优异。**

- **链接: [http://arxiv.org/pdf/2505.13941v1](http://arxiv.org/pdf/2505.13941v1)**

> **作者:** Haoyang Fang; Boran Han; Nick Erickson; Xiyuan Zhang; Su Zhou; Anirudh Dagar; Jiani Zhang; Ali Caner Turkmen; Cuixiong Hu; Huzefa Rangwala; Ying Nian Wu; Bernie Wang; George Karypis
>
> **摘要:** Existing AutoML systems have advanced the automation of machine learning (ML); however, they still require substantial manual configuration and expert input, particularly when handling multimodal data. We introduce MLZero, a novel multi-agent framework powered by Large Language Models (LLMs) that enables end-to-end ML automation across diverse data modalities with minimal human intervention. A cognitive perception module is first employed, transforming raw multimodal inputs into perceptual context that effectively guides the subsequent workflow. To address key limitations of LLMs, such as hallucinated code generation and outdated API knowledge, we enhance the iterative code generation process with semantic and episodic memory. MLZero demonstrates superior performance on MLE-Bench Lite, outperforming all competitors in both success rate and solution quality, securing six gold medals. Additionally, when evaluated on our Multimodal AutoML Agent Benchmark, which includes 25 more challenging tasks spanning diverse data modalities, MLZero outperforms the competing methods by a large margin with a success rate of 0.92 (+263.6\%) and an average rank of 2.28. Our approach maintains its robust effectiveness even with a compact 8B LLM, outperforming full-size systems from existing solutions.
>
---
#### [new 162] Rank-K: Test-Time Reasoning for Listwise Reranking
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出Rank-K模型，属于信息检索中的列表级重排任务。针对现有神经重排器资源消耗大、效率低的问题，其利用推理语言模型在测试时动态处理复杂查询，提升检索效果（较RankZephyr提升23%及SPLADE-v3提升19%），并支持多语言排序。**

- **链接: [http://arxiv.org/pdf/2505.14432v1](http://arxiv.org/pdf/2505.14432v1)**

> **作者:** Eugene Yang; Andrew Yates; Kathryn Ricci; Orion Weller; Vivek Chari; Benjamin Van Durme; Dawn Lawrie
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Retrieve-and-rerank is a popular retrieval pipeline because of its ability to make slow but effective rerankers efficient enough at query time by reducing the number of comparisons. Recent works in neural rerankers take advantage of large language models for their capability in reasoning between queries and passages and have achieved state-of-the-art retrieval effectiveness. However, such rerankers are resource-intensive, even after heavy optimization. In this work, we introduce Rank-K, a listwise passage reranking model that leverages the reasoning capability of the reasoning language model at query time that provides test time scalability to serve hard queries. We show that Rank-K improves retrieval effectiveness by 23\% over the RankZephyr, the state-of-the-art listwise reranker, when reranking a BM25 initial ranked list and 19\% when reranking strong retrieval results by SPLADE-v3. Since Rank-K is inherently a multilingual model, we found that it ranks passages based on queries in different languages as effectively as it does in monolingual retrieval.
>
---
#### [new 163] SAFEPATH: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SAFEPATH方法，解决大推理模型（LRMs）在处理有害提示时产生危险推理路径的问题。通过轻量级微调使模型在推理初始阶段生成8词安全引导，后续推理无约束，既减少90%有害输出和83.3%越狱攻击，又保持推理性能，计算成本远低于现有方法，并分析了安全对齐技术的局限性。**

- **链接: [http://arxiv.org/pdf/2505.14667v1](http://arxiv.org/pdf/2505.14667v1)**

> **作者:** Wonje Jeung; Sangyeon Yoon; Minsuk Kahng; Albert No
>
> **备注:** 22 pages
>
> **摘要:** Large Reasoning Models (LRMs) have become powerful tools for complex problem solving, but their structured reasoning pathways can lead to unsafe outputs when exposed to harmful prompts. Existing safety alignment methods reduce harmful outputs but can degrade reasoning depth, leading to significant trade-offs in complex, multi-step tasks, and remain vulnerable to sophisticated jailbreak attacks. To address this, we introduce SAFEPATH, a lightweight alignment method that fine-tunes LRMs to emit a short, 8-token Safety Primer at the start of their reasoning, in response to harmful prompts, while leaving the rest of the reasoning process unsupervised. Empirical results across multiple benchmarks indicate that SAFEPATH effectively reduces harmful outputs while maintaining reasoning performance. Specifically, SAFEPATH reduces harmful responses by up to 90.0% and blocks 83.3% of jailbreak attempts in the DeepSeek-R1-Distill-Llama-8B model, while requiring 295.9x less compute than Direct Refusal and 314.1x less than SafeChain. We further introduce a zero-shot variant that requires no fine-tuning. In addition, we provide a comprehensive analysis of how existing methods in LLMs generalize, or fail, when applied to reasoning-centric models, revealing critical gaps and new directions for safer AI.
>
---
#### [new 164] Advancing Software Quality: A Standards-Focused Review of LLM-Based Assurance Techniques
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文综述了LLM技术与软件质量标准的结合，旨在通过AI增强传统SQA流程并确保合规。分析LLM在需求验证、缺陷检测等应用，映射至ISO/CMMI等标准框架，探讨数据隐私等挑战并提出未来方向。**

- **链接: [http://arxiv.org/pdf/2505.13766v1](http://arxiv.org/pdf/2505.13766v1)**

> **作者:** Avinash Patil
>
> **备注:** 16 pages, 1 Table, 6 Figures
>
> **摘要:** Software Quality Assurance (SQA) is critical for delivering reliable, secure, and efficient software products. The Software Quality Assurance Process aims to provide assurance that work products and processes comply with predefined provisions and plans. Recent advancements in Large Language Models (LLMs) present new opportunities to enhance existing SQA processes by automating tasks like requirement analysis, code review, test generation, and compliance checks. Simultaneously, established standards such as ISO/IEC 12207, ISO/IEC 25010, ISO/IEC 5055, ISO 9001/ISO/IEC 90003, CMMI, and TMM provide structured frameworks for ensuring robust quality practices. This paper surveys the intersection of LLM-based SQA methods and these recognized standards, highlighting how AI-driven solutions can augment traditional approaches while maintaining compliance and process maturity. We first review the foundational software quality standards and the technical fundamentals of LLMs in software engineering. Next, we explore various LLM-based SQA applications, including requirement validation, defect detection, test generation, and documentation maintenance. We then map these applications to key software quality frameworks, illustrating how LLMs can address specific requirements and metrics within each standard. Empirical case studies and open-source initiatives demonstrate the practical viability of these methods. At the same time, discussions on challenges (e.g., data privacy, model bias, explainability) underscore the need for deliberate governance and auditing. Finally, we propose future directions encompassing adaptive learning, privacy-focused deployments, multimodal analysis, and evolving standards for AI-driven software quality.
>
---
#### [new 165] OmniGenBench: A Modular Platform for Reproducible Genomic Foundation Models Benchmarking
- **分类: q-bio.GN; cs.CL**

- **简介: 该论文提出OmniGenBench平台，解决基因组基础模型（GFMs）评估的可复现性与标准化问题。通过整合数据、模型、评估及可解释性模块，支持统一基准测试，集成31个开源模型，提升透明度与互操作性，推动可信基因组AI研究。**

- **链接: [http://arxiv.org/pdf/2505.14402v1](http://arxiv.org/pdf/2505.14402v1)**

> **作者:** Heng Yang; Jack Cole; Yuan Li; Renzhi Chen; Geyong Min; Ke Li
>
> **摘要:** The code of nature, embedded in DNA and RNA genomes since the origin of life, holds immense potential to impact both humans and ecosystems through genome modeling. Genomic Foundation Models (GFMs) have emerged as a transformative approach to decoding the genome. As GFMs scale up and reshape the landscape of AI-driven genomics, the field faces an urgent need for rigorous and reproducible evaluation. We present OmniGenBench, a modular benchmarking platform designed to unify the data, model, benchmarking, and interpretability layers across GFMs. OmniGenBench enables standardized, one-command evaluation of any GFM across five benchmark suites, with seamless integration of over 31 open-source models. Through automated pipelines and community-extensible features, the platform addresses critical reproducibility challenges, including data transparency, model interoperability, benchmark fragmentation, and black-box interpretability. OmniGenBench aims to serve as foundational infrastructure for reproducible genomic AI research, accelerating trustworthy discovery and collaborative innovation in the era of genome-scale modeling.
>
---
#### [new 166] Beyond Words: Multimodal LLM Knows When to Speak
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于对话时机预测任务，解决大语言模型在多模态对话中难以及时生成短回应的问题。研究构建了含同步视听文本数据的多模态数据集，并提出MM-When2Speak模型，融合多模态信息预测回应时机与类型，实验显示其响应时机准确性超现有模型4倍。**

- **链接: [http://arxiv.org/pdf/2505.14654v1](http://arxiv.org/pdf/2505.14654v1)**

> **作者:** Zikai Liao; Yi Ouyang; Yi-Lun Lee; Chen-Ping Yu; Yi-Hsuan Tsai; Zhaozheng Yin
>
> **备注:** Project page: https://github.com/lzk901372/MM-When2Speak
>
> **摘要:** While large language model (LLM)-based chatbots have demonstrated strong capabilities in generating coherent and contextually relevant responses, they often struggle with understanding when to speak, particularly in delivering brief, timely reactions during ongoing conversations. This limitation arises largely from their reliance on text input, lacking the rich contextual cues in real-world human dialogue. In this work, we focus on real-time prediction of response types, with an emphasis on short, reactive utterances that depend on subtle, multimodal signals across vision, audio, and text. To support this, we introduce a new multimodal dataset constructed from real-world conversational videos, containing temporally aligned visual, auditory, and textual streams. This dataset enables fine-grained modeling of response timing in dyadic interactions. Building on this dataset, we propose MM-When2Speak, a multimodal LLM-based model that adaptively integrates visual, auditory, and textual context to predict when a response should occur, and what type of response is appropriate. Experiments show that MM-When2Speak significantly outperforms state-of-the-art unimodal and LLM-based baselines, achieving up to a 4x improvement in response timing accuracy over leading commercial LLMs. These results underscore the importance of multimodal inputs for producing timely, natural, and engaging conversational AI.
>
---
#### [new 167] KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于个性化食谱推荐与生成任务，旨在解决现有系统缺乏知识图谱与大语言模型深度整合的问题。提出KERL系统，结合食物知识图谱与LLM，通过实体提取、子图检索增强上下文理解，生成符合用户约束的食谱及营养信息，并构建基准数据集验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14629v1](http://arxiv.org/pdf/2505.14629v1)**

> **作者:** Fnu Mohbat; Mohammed J Zaki
>
> **备注:** Accepted at ACL 2025
>
> **摘要:** Recent advances in large language models (LLMs) and the abundance of food data have resulted in studies to improve food understanding using LLMs. Despite several recommendation systems utilizing LLMs and Knowledge Graphs (KGs), there has been limited research on integrating food related KGs with LLMs. We introduce KERL, a unified system that leverages food KGs and LLMs to provide personalized food recommendations and generates recipes with associated micro-nutritional information. Given a natural language question, KERL extracts entities, retrieves subgraphs from the KG, which are then fed into the LLM as context to select the recipes that satisfy the constraints. Next, our system generates the cooking steps and nutritional information for each recipe. To evaluate our approach, we also develop a benchmark dataset by curating recipe related questions, combined with constraints and personal preferences. Through extensive experiments, we show that our proposed KG-augmented LLM significantly outperforms existing approaches, offering a complete and coherent solution for food recommendation, recipe generation, and nutritional analysis. Our code and benchmark datasets are publicly available at https://github.com/mohbattharani/KERL.
>
---
#### [new 168] Evaluating Large Language Models for Real-World Engineering Tasks
- **分类: cs.AI; cs.CL**

- **简介: 该论文评估LLMs在工程任务中的性能，针对现有评测依赖简化案例和场景不足的问题，构建含100+真实工程场景的数据库，测试四类先进LLMs，发现其在基础推理上有优势但抽象建模等能力薄弱。**

- **链接: [http://arxiv.org/pdf/2505.13484v1](http://arxiv.org/pdf/2505.13484v1)**

> **作者:** Rene Heesch; Sebastian Eilermann; Alexander Windmann; Alexander Diedrich; Philipp Rosenthal; Oliver Niggemann
>
> **摘要:** Large Language Models (LLMs) are transformative not only for daily activities but also for engineering tasks. However, current evaluations of LLMs in engineering exhibit two critical shortcomings: (i) the reliance on simplified use cases, often adapted from examination materials where correctness is easily verifiable, and (ii) the use of ad hoc scenarios that insufficiently capture critical engineering competencies. Consequently, the assessment of LLMs on complex, real-world engineering problems remains largely unexplored. This paper addresses this gap by introducing a curated database comprising over 100 questions derived from authentic, production-oriented engineering scenarios, systematically designed to cover core competencies such as product design, prognosis, and diagnosis. Using this dataset, we evaluate four state-of-the-art LLMs, including both cloud-based and locally hosted instances, to systematically investigate their performance on complex engineering tasks. Our results show that LLMs demonstrate strengths in basic temporal and structural reasoning but struggle significantly with abstract reasoning, formal modeling, and context-sensitive engineering logic.
>
---
#### [new 169] ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出ContextAgent，首个结合多模态感官数据（如可穿戴设备的视频/音频）与用户历史记录的主动式LLM代理，解决现有代理依赖封闭环境或规则导致的意图理解不足与功能局限问题。通过感知数据预测主动服务需求并自动调用工具，在新基准ContextAgentBench测试中优于基线方法。**

- **链接: [http://arxiv.org/pdf/2505.14668v1](http://arxiv.org/pdf/2505.14668v1)**

> **作者:** Bufang Yang; Lilin Xu; Liekang Zeng; Kaiwei Liu; Siyang Jiang; Wenrui Lu; Hongkai Chen; Xiaofan Jiang; Guoliang Xing; Zhenyu Yan
>
> **摘要:** Recent advances in Large Language Models (LLMs) have propelled intelligent agents from reactive responses to proactive support. While promising, existing proactive agents either rely exclusively on observations from enclosed environments (e.g., desktop UIs) with direct LLM inference or employ rule-based proactive notifications, leading to suboptimal user intent understanding and limited functionality for proactive service. In this paper, we introduce ContextAgent, the first context-aware proactive agent that incorporates extensive sensory contexts to enhance the proactive capabilities of LLM agents. ContextAgent first extracts multi-dimensional contexts from massive sensory perceptions on wearables (e.g., video and audio) to understand user intentions. ContextAgent then leverages the sensory contexts and the persona contexts from historical data to predict the necessity for proactive services. When proactive assistance is needed, ContextAgent further automatically calls the necessary tools to assist users unobtrusively. To evaluate this new task, we curate ContextAgentBench, the first benchmark for evaluating context-aware proactive LLM agents, covering 1,000 samples across nine daily scenarios and twenty tools. Experiments on ContextAgentBench show that ContextAgent outperforms baselines by achieving up to 8.5% and 6.0% higher accuracy in proactive predictions and tool calling, respectively. We hope our research can inspire the development of more advanced, human-centric, proactive AI assistants.
>
---
#### [new 170] Is Your Prompt Safe? Investigating Prompt Injection Attacks Against Open-Source LLMs
- **分类: cs.CR; cs.CL**

- **简介: 该论文研究开源大语言模型（LLMs）对提示注入攻击的脆弱性。针对现有评估仅关注攻击成功与否的不足，提出Attack Success Probability（ASP）指标，量化攻击成功率及模型响应不确定性。测试了14个流行开源LLMs，提出"催眠攻击"和"忽略前缀攻击"，发现知名模型易受攻击（ASP达90%和60%以上），强调需提升安全意识并优化防御策略。**

- **链接: [http://arxiv.org/pdf/2505.14368v1](http://arxiv.org/pdf/2505.14368v1)**

> **作者:** Jiawen Wang; Pritha Gupta; Ivan Habernal; Eyke Hüllermeier
>
> **备注:** 8 pages, 3 figures, EMNLP 2025 under review
>
> **摘要:** Recent studies demonstrate that Large Language Models (LLMs) are vulnerable to different prompt-based attacks, generating harmful content or sensitive information. Both closed-source and open-source LLMs are underinvestigated for these attacks. This paper studies effective prompt injection attacks against the $\mathbf{14}$ most popular open-source LLMs on five attack benchmarks. Current metrics only consider successful attacks, whereas our proposed Attack Success Probability (ASP) also captures uncertainty in the model's response, reflecting ambiguity in attack feasibility. By comprehensively analyzing the effectiveness of prompt injection attacks, we propose a simple and effective hypnotism attack; results show that this attack causes aligned language models, including Stablelm2, Mistral, Openchat, and Vicuna, to generate objectionable behaviors, achieving around $90$% ASP. They also indicate that our ignore prefix attacks can break all $\mathbf{14}$ open-source LLMs, achieving over $60$% ASP on a multi-categorical dataset. We find that moderately well-known LLMs exhibit higher vulnerability to prompt injection attacks, highlighting the need to raise public awareness and prioritize efficient mitigation strategies.
>
---
#### [new 171] FMSD-TTS: Few-shot Multi-Speaker Multi-Dialect Text-to-Speech Synthesis for Ü-Tsang, Amdo and Kham Speech Dataset Generation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出FMSD-TTS框架，解决藏语三大方言（Ü-Tsang、Amdo、Kham）低资源下的多说话人多方言语音合成问题。通过设计说话人-方言融合模块和DSDR-Net，捕捉方言差异同时保持说话人特征，公开合成语料库及评估工具。**

- **链接: [http://arxiv.org/pdf/2505.14351v1](http://arxiv.org/pdf/2505.14351v1)**

> **作者:** Yutong Liu; Ziyue Zhang; Ban Ma-bao; Yuqing Cai; Yongbin Yu; Renzeng Duojie; Xiangxiang Wang; Fan Gao; Cheng Huang; Nyima Tashi
>
> **备注:** 13 pages
>
> **摘要:** Tibetan is a low-resource language with minimal parallel speech corpora spanning its three major dialects-\"U-Tsang, Amdo, and Kham-limiting progress in speech modeling. To address this issue, we propose FMSD-TTS, a few-shot, multi-speaker, multi-dialect text-to-speech framework that synthesizes parallel dialectal speech from limited reference audio and explicit dialect labels. Our method features a novel speaker-dialect fusion module and a Dialect-Specialized Dynamic Routing Network (DSDR-Net) to capture fine-grained acoustic and linguistic variations across dialects while preserving speaker identity. Extensive objective and subjective evaluations demonstrate that FMSD-TTS significantly outperforms baselines in both dialectal expressiveness and speaker similarity. We further validate the quality and utility of the synthesized speech through a challenging speech-to-speech dialect conversion task. Our contributions include: (1) a novel few-shot TTS system tailored for Tibetan multi-dialect speech synthesis, (2) the public release of a large-scale synthetic Tibetan speech corpus generated by FMSD-TTS, and (3) an open-source evaluation toolkit for standardized assessment of speaker similarity, dialect consistency, and audio quality.
>
---
#### [new 172] Mitigating Subgroup Disparities in Multi-Label Speech Emotion Recognition: A Pseudo-Labeling and Unsupervised Learning Approach
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于多标签语音情感识别（SER）任务，旨在解决缺乏显式人口统计信息时的子群组性能差异问题。提出Implicit Demography Inference（IDI）模块，结合伪标签与无监督聚类，无需显式标签即可减少种族、年龄等偏见，在公平性指标提升超26%-33%的同时，仅小幅降低识别准确率。**

- **链接: [http://arxiv.org/pdf/2505.14449v1](http://arxiv.org/pdf/2505.14449v1)**

> **作者:** Yi-Cheng Lin; Huang-Cheng Chou; Hung-yi Lee
>
> **备注:** Accepted by InterSpeech 2025. 7 pages including 2 pages of appendix
>
> **摘要:** While subgroup disparities and performance bias are increasingly studied in computational research, fairness in categorical Speech Emotion Recognition (SER) remains underexplored. Existing methods often rely on explicit demographic labels, which are difficult to obtain due to privacy concerns. To address this limitation, we introduce an Implicit Demography Inference (IDI) module that leverages pseudo-labeling from a pre-trained model and unsupervised learning using k-means clustering to mitigate bias in SER. Our experiments show that pseudo-labeling IDI reduces subgroup disparities, improving fairness metrics by over 33% with less than a 3% decrease in SER accuracy. Also, the unsupervised IDI yields more than a 26% improvement in fairness metrics with a drop of less than 4% in SER performance. Further analyses reveal that the unsupervised IDI consistently mitigates race and age disparities, demonstrating its potential in scenarios where explicit demographic information is unavailable.
>
---
#### [new 173] Can AI Freelancers Compete? Benchmarking Earnings, Reliability, and Task Success at Scale
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于LLM（大型语言模型）基准测试任务，旨在评估AI作为自由职业者在编程和数据分析任务中的竞争力。通过构建基于Kaggle数据集的标准化任务集，测试Claude 3.5 Haiku等四款模型的准确率及模拟"收入"，结果显示Claude表现最佳。研究讨论了AI freelance可行性及自动化评估方法的优劣。**

- **链接: [http://arxiv.org/pdf/2505.13511v1](http://arxiv.org/pdf/2505.13511v1)**

> **作者:** David Noever; Forrest McKee
>
> **摘要:** This study explores Large Language Models (LLMs) as autonomous agents for real-world tasks, including freelance software development. This work presents a new benchmark that evaluates LLMs on freelance programming and data analysis tasks derived from economic data. We construct the benchmark using synthetic tasks created from a Kaggle Freelancer dataset of job postings, with all job prices standardized to USD (median fixed-project price around $250, and an average of $306). Each task is accompanied by structured input-output test cases and an estimated price tag, enabling automated correctness checking and a monetary performance valuation. This approach is inspired by OpenAI's recent SWE-Lancer benchmark (1,400 real Upwork tasks worth $1M total). Still, our framework simplifies evaluation using programmatically testable tasks and predicted price values, making it highly scalable and repeatable. On this benchmark, we evaluate four modern LLMs - Claude 3.5 Haiku, GPT-4o-mini, Qwen 2.5, and Mistral. We report each model's accuracy (task success rate and test-case pass rate) and the total "freelance earnings" it achieves (sum of prices of solved tasks). Our results show that Claude 3.5 Haiku performs best, earning approximately $1.52 million USD, followed closely by GPT-4o-mini at $1.49 million, then Qwen 2.5 ($1.33M) and Mistral ($0.70M). We analyze the distribution of errors per task and observe that the strongest models solve the most tasks and rarely fail completely on any project. We discuss the implications of these results for the feasibility of AI as a freelance developer, the advantages and limitations of our automated benchmark approach, and the gap between performance on structured tasks versus the true complexity of real-world freelance jobs.
>
---
#### [new 174] Mobile-Agent-V: A Video-Guided Approach for Effortless and Efficient Operational Knowledge Injection in Mobile Automation
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Mobile-Agent-V框架，属于移动自动化领域知识注入任务。针对现有AI框架缺乏操作知识、手动编写低效的问题，其通过视频直接提取操作知识，减少人工干预。提出Mobile-Knowledge基准进行性能评估，实验显示较现有方法提升36%。**

- **链接: [http://arxiv.org/pdf/2505.13887v1](http://arxiv.org/pdf/2505.13887v1)**

> **作者:** Junyang Wang; Haiyang Xu; Xi Zhang; Ming Yan; Ji Zhang; Fei Huang; Jitao Sang
>
> **备注:** 17 pages, 7 figures, 9 tables. arXiv admin note: substantial text overlap with arXiv:2502.17110
>
> **摘要:** The exponential rise in mobile device usage necessitates streamlined automation for effective task management, yet many AI frameworks fall short due to inadequate operational expertise. While manually written knowledge can bridge this gap, it is often burdensome and inefficient. We introduce Mobile-Agent-V, an innovative framework that utilizes video as a guiding tool to effortlessly and efficiently inject operational knowledge into mobile automation processes. By deriving knowledge directly from video content, Mobile-Agent-V eliminates manual intervention, significantly reducing the effort and time required for knowledge acquisition. To rigorously evaluate this approach, we propose Mobile-Knowledge, a benchmark tailored to assess the impact of external knowledge on mobile agent performance. Our experimental findings demonstrate that Mobile-Agent-V enhances performance by 36% compared to existing methods, underscoring its effortless and efficient advantages in mobile automation.
>
---
#### [new 175] Pairwise Evaluation of Accent Similarity in Speech Synthesis
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音合成中口音相似性评估任务，旨在解决现有主观和客观评估方法不足的问题，尤其针对少数群体口音。工作包括优化XAB测试（提供转录、标注差异、筛选可靠度），引入基于元音共振峰和语音后验图的客观指标，并指出WER等传统指标的局限性。**

- **链接: [http://arxiv.org/pdf/2505.14410v1](http://arxiv.org/pdf/2505.14410v1)**

> **作者:** Jinzuomu Zhong; Suyuan Liu; Dan Wells; Korin Richmond
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Despite growing interest in generating high-fidelity accents, evaluating accent similarity in speech synthesis has been underexplored. We aim to enhance both subjective and objective evaluation methods for accent similarity. Subjectively, we refine the XAB listening test by adding components that achieve higher statistical significance with fewer listeners and lower costs. Our method involves providing listeners with transcriptions, having them highlight perceived accent differences, and implementing meticulous screening for reliability. Objectively, we utilise pronunciation-related metrics, based on distances between vowel formants and phonetic posteriorgrams, to evaluate accent generation. Comparative experiments reveal that these metrics, alongside accent similarity, speaker similarity, and Mel Cepstral Distortion, can be used. Moreover, our findings underscore significant limitations of common metrics like Word Error Rate in assessing underrepresented accents.
>
---
#### [new 176] TinyV: Reducing False Negatives in Verification Improves RL for LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文属于强化学习优化大语言模型（LLM）推理任务，旨在解决验证器因假阴性（错误拒绝正确输出）导致RL训练效果差的问题。提出轻量级LLM验证器TinyV，结合规则方法动态识别假阴性，恢复有效响应以提升奖励估计，提升数学推理任务通过率并加速收敛。**

- **链接: [http://arxiv.org/pdf/2505.14625v1](http://arxiv.org/pdf/2505.14625v1)**

> **作者:** Zhangchen Xu; Yuetai Li; Fengqing Jiang; Bhaskar Ramasubramanian; Luyao Niu; Bill Yuchen Lin; Radha Poovendran
>
> **摘要:** Reinforcement Learning (RL) has become a powerful tool for enhancing the reasoning abilities of large language models (LLMs) by optimizing their policies with reward signals. Yet, RL's success relies on the reliability of rewards, which are provided by verifiers. In this paper, we expose and analyze a widespread problem--false negatives--where verifiers wrongly reject correct model outputs. Our in-depth study of the Big-Math-RL-Verified dataset reveals that over 38% of model-generated responses suffer from false negatives, where the verifier fails to recognize correct answers. We show, both empirically and theoretically, that these false negatives severely impair RL training by depriving the model of informative gradient signals and slowing convergence. To mitigate this, we propose tinyV, a lightweight LLM-based verifier that augments existing rule-based methods, which dynamically identifies potential false negatives and recovers valid responses to produce more accurate reward estimates. Across multiple math-reasoning benchmarks, integrating TinyV boosts pass rates by up to 10% and accelerates convergence relative to the baseline. Our findings highlight the critical importance of addressing verifier false negatives and offer a practical approach to improve RL-based fine-tuning of LLMs. Our code is available at https://github.com/uw-nsl/TinyV.
>
---
#### [new 177] Warm Up Before You Train: Unlocking General Reasoning in Resource-Constrained Settings
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出一种两阶段训练策略，解决数据稀缺下训练推理型LLM的问题。首阶段通过Knights&Knaves谜题蒸馏长推理链预训练模型，次阶段用有限域数据RLVR微调，提升跨任务性能与样本效率。**

- **链接: [http://arxiv.org/pdf/2505.13718v1](http://arxiv.org/pdf/2505.13718v1)**

> **作者:** Safal Shrestha; Minwu Kim; Aadim Nepal; Anubhav Shrestha; Keith Ross
>
> **摘要:** Designing effective reasoning-capable LLMs typically requires training using Reinforcement Learning with Verifiable Rewards (RLVR) or distillation with carefully curated Long Chain of Thoughts (CoT), both of which depend heavily on extensive training data. This creates a major challenge when the amount of quality training data is scarce. We propose a sample-efficient, two-stage training strategy to develop reasoning LLMs under limited supervision. In the first stage, we "warm up" the model by distilling Long CoTs from a toy domain, namely, Knights \& Knaves (K\&K) logic puzzles to acquire general reasoning skills. In the second stage, we apply RLVR to the warmed-up model using a limited set of target-domain examples. Our experiments demonstrate that this two-phase approach offers several benefits: $(i)$ the warmup phase alone facilitates generalized reasoning, leading to performance improvements across a range of tasks, including MATH, HumanEval$^{+}$, and MMLU-Pro. $(ii)$ When both the base model and the warmed-up model are RLVR trained on the same small dataset ($\leq100$ examples), the warmed-up model consistently outperforms the base model; $(iii)$ Warming up before RLVR training allows a model to maintain cross-domain generalizability even after training on a specific domain; $(iv)$ Introducing warmup in the pipeline improves not only accuracy but also overall sample efficiency during RLVR training. The results in this paper highlight the promise of warmup for building robust reasoning LLMs in data-scarce environments.
>
---
#### [new 178] Teaching Audio-Aware Large Language Models What Does Not Hear: Mitigating Hallucinations through Synthesized Negative Samples
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文针对音频感知大语言模型（ALLMs）虚构不存在声音的问题，提出LISTEN方法。通过合成负样本对比训练，增强模型区分真实与不存在声音的能力，采用轻量适配器无需修改模型参数。实验显示有效减少幻觉且计算高效。**

- **链接: [http://arxiv.org/pdf/2505.14518v1](http://arxiv.org/pdf/2505.14518v1)**

> **作者:** Chun-Yi Kuan; Hung-yi Lee
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Recent advancements in audio-aware large language models (ALLMs) enable them to process and understand audio inputs. However, these models often hallucinate non-existent sound events, reducing their reliability in real-world applications. To address this, we propose LISTEN (Learning to Identify Sounds Through Extended Negative Samples), a contrastive-like training method that enhances ALLMs' ability to distinguish between present and absent sounds using synthesized data from the backbone LLM. Unlike prior approaches, our method requires no modification to LLM parameters and efficiently integrates audio representations via a lightweight adapter. Experiments show that LISTEN effectively mitigates hallucinations while maintaining impressive performance on existing audio question and reasoning benchmarks. At the same time, it is more efficient in both data and computation.
>
---
#### [new 179] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大型语言模型（LLM）预训练中超参数（学习率η、权重衰减λ）和批量大小（B）的缩放规律。解决如何随模型规模N、数据量D和B调整HP以优化训练效率。提出权重衰减λ与B线性相关，最优/临界批量大小随D呈幂律缩放，独立于N，并据此指导 Pareto最优N/D选择。**

- **链接: [http://arxiv.org/pdf/2505.13738v1](http://arxiv.org/pdf/2505.13738v1)**

> **作者:** Shane Bergsma; Nolan Dey; Gurpreet Gosal; Gavia Gray; Daria Soboleva; Joel Hestness
>
> **摘要:** Efficient LLM pre-training requires well-tuned hyperparameters (HPs), including learning rate {\eta} and weight decay {\lambda}. We study scaling laws for HPs: formulas for how to scale HPs as we scale model size N, dataset size D, and batch size B. Recent work suggests the AdamW timescale, B/({\eta}{\lambda}D), should remain constant across training settings, and we verify the implication that optimal {\lambda} scales linearly with B, for a fixed N,D. However, as N,D scale, we show the optimal timescale obeys a precise power law in the tokens-per-parameter ratio, D/N. This law thus provides a method to accurately predict {\lambda}opt in advance of large-scale training. We also study scaling laws for optimal batch size Bopt (the B enabling lowest loss at a given N,D) and critical batch size Bcrit (the B beyond which further data parallelism becomes ineffective). In contrast with prior work, we find both Bopt and Bcrit scale as power laws in D, independent of model size, N. Finally, we analyze how these findings inform the real-world selection of Pareto-optimal N and D under dual training time and compute objectives.
>
---
#### [new 180] Debating for Better Reasoning: An Unsupervised Multimodal Approach
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多模态视觉问答任务，提出无监督辩论框架，解决LLMs能力超人类时的监督难题。通过让两个"有视觉"专家模型辩论答案，由文本模型裁判，聚焦分歧案例，利用弱模型提升强模型推理，实验显示优于单独模型。**

- **链接: [http://arxiv.org/pdf/2505.14627v1](http://arxiv.org/pdf/2505.14627v1)**

> **作者:** Ashutosh Adhikari; Mirella Lapata
>
> **摘要:** As Large Language Models (LLMs) gain expertise across diverse domains and modalities, scalable oversight becomes increasingly challenging, particularly when their capabilities may surpass human evaluators. Debate has emerged as a promising mechanism for enabling such oversight. In this work, we extend the debate paradigm to a multimodal setting, exploring its potential for weaker models to supervise and enhance the performance of stronger models. We focus on visual question answering (VQA), where two "sighted" expert vision-language models debate an answer, while a "blind" (text-only) judge adjudicates based solely on the quality of the arguments. In our framework, the experts defend only answers aligned with their beliefs, thereby obviating the need for explicit role-playing and concentrating the debate on instances of expert disagreement. Experiments on several multimodal tasks demonstrate that the debate framework consistently outperforms individual expert models. Moreover, judgments from weaker LLMs can help instill reasoning capabilities in vision-language models through finetuning.
>
---
#### [new 181] InterFeat: An Automated Pipeline for Finding Interesting Hypotheses in Structured Biomedical Data
- **分类: q-bio.QM; cs.AI; cs.CL; cs.IR; 68T05, 68T50, 92C50; I.2.6; I.2.7; H.2.8; J.3**

- **简介: 该论文提出InterFeat，结合机器学习、知识图谱及大模型，自动化挖掘生物医学数据中有趣的特征-目标关系。解决科学发现中手动、定义模糊的问题，通过新颖性、实用性和合理性量化“interestingness”。在UK Biobank的8种疾病数据中，其预测风险因子的表现优于基线，28%候选获专家认可，并开源工具。**

- **链接: [http://arxiv.org/pdf/2505.13534v1](http://arxiv.org/pdf/2505.13534v1)**

> **作者:** Dan Ofer; Michal Linial; Dafna Shahaf
>
> **摘要:** Finding interesting phenomena is the core of scientific discovery, but it is a manual, ill-defined concept. We present an integrative pipeline for automating the discovery of interesting simple hypotheses (feature-target relations with effect direction and a potential underlying mechanism) in structured biomedical data. The pipeline combines machine learning, knowledge graphs, literature search and Large Language Models. We formalize "interestingness" as a combination of novelty, utility and plausibility. On 8 major diseases from the UK Biobank, our pipeline consistently recovers risk factors years before their appearance in the literature. 40--53% of our top candidates were validated as interesting, compared to 0--7% for a SHAP-based baseline. Overall, 28% of 109 candidates were interesting to medical experts. The pipeline addresses the challenge of operationalizing "interestingness" scalably and for any target. We release data and code: https://github.com/LinialLab/InterFeat
>
---
#### [new 182] SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle Generation from SAT Formulas
- **分类: cs.AI; cs.CL; cs.LG; cs.LO**

- **简介: 该论文提出SATBench，通过将SAT公式转化为逻辑谜题，评估LLMs的搜索型逻辑推理能力。解决现有基准依赖推理规则、无法测试复杂约束搜索的问题，构建自动化生成可调难度的2100个谜题，实验显示当前模型在困难任务中表现仅65%，暴露其逻辑搜索局限。**

- **链接: [http://arxiv.org/pdf/2505.14615v1](http://arxiv.org/pdf/2505.14615v1)**

> **作者:** Anjiang Wei; Yuheng Wu; Yingjia Wan; Tarun Suresh; Huanmi Tan; Zhanke Zhou; Sanmi Koyejo; Ke Wang; Alex Aiken
>
> **摘要:** We introduce SATBench, a benchmark for evaluating the logical reasoning capabilities of large language models (LLMs) through logical puzzles derived from Boolean satisfiability (SAT) problems. Unlike prior work that focuses on inference rule-based reasoning, which often involves deducing conclusions from a set of premises, our approach leverages the search-based nature of SAT problems, where the objective is to find a solution that fulfills a specified set of logical constraints. Each instance in SATBench is generated from a SAT formula, then translated into a story context and conditions using LLMs. The generation process is fully automated and allows for adjustable difficulty by varying the number of clauses. All 2100 puzzles are validated through both LLM-assisted and solver-based consistency checks, with human validation on a subset. Experimental results show that even the strongest model, o4-mini, achieves only 65.0% accuracy on hard UNSAT problems, close to the random baseline of 50%. SATBench exposes fundamental limitations in the search-based logical reasoning abilities of current LLMs and provides a scalable testbed for future research in logical reasoning.
>
---
#### [new 183] Enhancing Learned Knowledge in LoRA Adapters Through Efficient Contrastive Decoding on Ascend NPUs
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对LoRA微调模型在复杂任务中因传统解码（如贪心/束搜索）受基线模型偏差影响的问题，提出CoLD框架：通过对比LoRA专家模型与基线模型的概率分布差异优化解码，结合Ascend NPU优化加速，提升5.54%准确率并降低28%延迟，实现高效任务适配。**

- **链接: [http://arxiv.org/pdf/2505.14620v1](http://arxiv.org/pdf/2505.14620v1)**

> **作者:** Morgan Lindsay Heisler; Linzi Xing; Ge Shi; Hanieh Sadri; Gursimran Singh; Weiwei Zhang; Tao Ye; Ying Xiong; Yong Zhang; Zhenan Fan
>
> **备注:** Accepted at ACM KDD 2025
>
> **摘要:** Huawei Cloud users leverage LoRA (Low-Rank Adaptation) as an efficient and scalable method to fine-tune and customize large language models (LLMs) for application-specific needs. However, tasks that require complex reasoning or deep contextual understanding are often hindered by biases or interference from the base model when using typical decoding methods like greedy or beam search. These biases can lead to generic or task-agnostic responses from the base model instead of leveraging the LoRA-specific adaptations. In this paper, we introduce Contrastive LoRA Decoding (CoLD), a novel decoding framework designed to maximize the use of task-specific knowledge in LoRA-adapted models, resulting in better downstream performance. CoLD uses contrastive decoding by scoring candidate tokens based on the divergence between the probability distributions of a LoRA-adapted expert model and the corresponding base model. This approach prioritizes tokens that better align with the LoRA's learned representations, enhancing performance for specialized tasks. While effective, a naive implementation of CoLD is computationally expensive because each decoding step requires evaluating multiple token candidates across both models. To address this, we developed an optimized kernel for Huawei's Ascend NPU. CoLD achieves up to a 5.54% increase in task accuracy while reducing end-to-end latency by 28% compared to greedy decoding. This work provides practical and efficient decoding strategies for fine-tuned LLMs in resource-constrained environments and has broad implications for applied data science in both cloud and on-premises settings.
>
---
#### [new 184] Agent Context Protocols Enhance Collective Inference
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多智能体系统协作任务，旨在解决现有自然语言协调方式在复杂交互和领域互操作性上的不足。提出Agent Context Protocols（ACP），通过持久依赖图存储中间输出和标准化消息格式，实现健壮的多智能体协作，提升集体推理性能，实验显示其达业界领先的准确率且模块化强。**

- **链接: [http://arxiv.org/pdf/2505.14569v1](http://arxiv.org/pdf/2505.14569v1)**

> **作者:** Devansh Bhardwaj; Arjun Beniwal; Shreyas Chaudhari; Ashwin Kalyan; Tanmay Rajpurohit; Karthik R. Narasimhan; Ameet Deshpande; Vishvak Murahari
>
> **摘要:** AI agents have become increasingly adept at complex tasks such as coding, reasoning, and multimodal understanding. However, building generalist systems requires moving beyond individual agents to collective inference -- a paradigm where multi-agent systems with diverse, task-specialized agents complement one another through structured communication and collaboration. Today, coordination is usually handled with imprecise, ad-hoc natural language, which limits complex interaction and hinders interoperability with domain-specific agents. We introduce Agent context protocols (ACPs): a domain- and agent-agnostic family of structured protocols for agent-agent communication, coordination, and error handling. ACPs combine (i) persistent execution blueprints -- explicit dependency graphs that store intermediate agent outputs -- with (ii) standardized message schemas, enabling robust and fault-tolerant multi-agent collective inference. ACP-powered generalist systems reach state-of-the-art performance: 28.3 % accuracy on AssistantBench for long-horizon web assistance and best-in-class multimodal technical reports, outperforming commercial AI systems in human evaluation. ACPs are highly modular and extensible, allowing practitioners to build top-tier generalist agents quickly.
>
---
#### [new 185] BARREL: Boundary-Aware Reasoning for Factual and Reliable LRMs
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于提升大模型可靠性的研究任务，针对现有LRMs过度自信输出错误答案的问题，提出BARREL框架抑制两种病态推理模式（最后猜测、思维漩涡），通过边界感知训练使模型可靠性提升22%，在保持准确率的同时减少胡扯。**

- **链接: [http://arxiv.org/pdf/2505.13529v1](http://arxiv.org/pdf/2505.13529v1)**

> **作者:** Junxiao Yang; Jinzhe Tu; Haoran Liu; Xiaoce Wang; Chujie Zheng; Zhexin Zhang; Shiyao Cui; Caishun Chen; Tiantian He; Hongning Wang; Yew-Soon Ong; Minlie Huang
>
> **摘要:** Recent advances in Large Reasoning Models (LRMs) have shown impressive capabilities in mathematical and logical reasoning. However, current LRMs rarely admit ignorance or respond with "I don't know". Instead, they often produce incorrect answers while showing undue confidence, raising concerns about their factual reliability. In this work, we identify two pathological reasoning patterns characterized by overthinking that contribute to the overconfident and incorrect answers: last-minute guessing and second-thought spiraling. To address these issues, we propose BARREL-a novel framework that promotes concise and boundary-aware factual reasoning. Our experiments show that BARREL-training increases the reliability of DeepSeek-R1-Distill-Llama-8B from 39.33% to 61.48%, while still achieving accuracy comparable to models finetuned on reasoning data generated by R1. These results demonstrate that our pilot study is inspiring to build more reliable and factual System 2 LRMs.
>
---
#### [new 186] Safety Subspaces are Not Distinct: A Fine-Tuning Case Study
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型安全对齐的几何子空间假设。通过参数和激活空间实证分析，发现安全与有害行为共享子空间，无独立控制证据，挑战子空间隔离防御可行性，强调需新策略维持对齐。任务：验证安全子空间假设；问题：安全是否局部化；方法：多模型实验，参数/激活分析。**

- **链接: [http://arxiv.org/pdf/2505.14185v1](http://arxiv.org/pdf/2505.14185v1)**

> **作者:** Kaustubh Ponkshe; Shaan Shah; Raghav Singhal; Praneeth Vepakomma
>
> **备注:** Kaustubh Ponkshe, Shaan Shah, and Raghav Singhal contributed equally to this work
>
> **摘要:** Large Language Models (LLMs) rely on safety alignment to produce socially acceptable responses. This is typically achieved through instruction tuning and reinforcement learning from human feedback. However, this alignment is known to be brittle: further fine-tuning, even on benign or lightly contaminated data, can degrade safety and reintroduce harmful behaviors. A growing body of work suggests that alignment may correspond to identifiable geometric directions in weight space, forming subspaces that could, in principle, be isolated or preserved to defend against misalignment. In this work, we conduct a comprehensive empirical study of this geometric perspective. We examine whether safety-relevant behavior is concentrated in specific subspaces, whether it can be separated from general-purpose learning, and whether harmfulness arises from distinguishable patterns in internal representations. Across both parameter and activation space, our findings are consistent: subspaces that amplify safe behaviors also amplify unsafe ones, and prompts with different safety implications activate overlapping representations. We find no evidence of a subspace that selectively governs safety. These results challenge the assumption that alignment is geometrically localized. Rather than residing in distinct directions, safety appears to emerge from entangled, high-impact components of the model's broader learning dynamics. This suggests that subspace-based defenses may face fundamental limitations and underscores the need for alternative strategies to preserve alignment under continued training. We corroborate these findings through multiple experiments on five open-source LLMs. Our code is publicly available at: https://github.com/CERT-Lab/safety-subspaces.
>
---
#### [new 187] S2SBench: A Benchmark for Quantifying Intelligence Degradation in Speech-to-Speech Large Language Models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出S2SBench基准，用于量化语音到语音LLMs的智能退化问题。任务是评估音频输入导致的推理与生成性能下降。通过构建诊断数据集（句子续写、常识推理）及基于困惑度差异的配对评估协议，系统衡量语音模型相较文本输入的退化，并验证于Baichuan-Audio训练分析。**

- **链接: [http://arxiv.org/pdf/2505.14438v1](http://arxiv.org/pdf/2505.14438v1)**

> **作者:** Yuanbo Fang; Haoze Sun; Jun Liu; Tao Zhang; Zenan Zhou; Weipeng Chen; Xiaofen Xing; Xiangmin Xu
>
> **摘要:** End-to-end speech large language models ((LLMs)) extend the capabilities of text-based models to directly process and generate audio tokens. However, this often leads to a decline in reasoning and generation performance compared to text input, a phenomenon referred to as intelligence degradation. To systematically evaluate this gap, we propose S2SBench, a benchmark designed to quantify performance degradation in Speech LLMs. It includes diagnostic datasets targeting sentence continuation and commonsense reasoning under audio input. We further introduce a pairwise evaluation protocol based on perplexity differences between plausible and implausible samples to measure degradation relative to text input. We apply S2SBench to analyze the training process of Baichuan-Audio, which further demonstrates the benchmark's effectiveness. All datasets and evaluation code are available at https://github.com/undobug/S2SBench.
>
---
#### [new 188] PersonaTAB: Predicting Personality Traits using Textual, Acoustic, and Behavioral Cues in Fully-Duplex Speech Dialogs
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于对话系统中的人格预测任务，旨在解决语音数据缺乏人格标注导致的个性感知代理研究不足问题。团队通过预处理音频生成标注数据集，结合ASR提取文本并利用大模型预测对话人格，经人类评估验证效果更优。**

- **链接: [http://arxiv.org/pdf/2505.14356v1](http://arxiv.org/pdf/2505.14356v1)**

> **作者:** Sho Inoue; Shai Wang; Haizhou Li
>
> **备注:** This is accepted to Interspeech 2025; Added an extra page for supplementary figures; Project page: https://github.com/shinshoji01/Personality-Prediction-for-Conversation-Agents
>
> **摘要:** Despite significant progress in neural spoken dialog systems, personality-aware conversation agents -- capable of adapting behavior based on personalities -- remain underexplored due to the absence of personality annotations in speech datasets. We propose a pipeline that preprocesses raw audio recordings to create a dialogue dataset annotated with timestamps, response types, and emotion/sentiment labels. We employ an automatic speech recognition (ASR) system to extract transcripts and timestamps, then generate conversation-level annotations. Leveraging these annotations, we design a system that employs large language models to predict conversational personality. Human evaluators were engaged to identify conversational characteristics and assign personality labels. Our analysis demonstrates that the proposed system achieves stronger alignment with human judgments compared to existing approaches.
>
---
#### [new 189] RAR: Setting Knowledge Tripwires for Retrieval Augmented Rejection
- **分类: cs.IR; cs.CL; cs.CR; 68M25, 68T07; I.2.7; K.6.5**

- **简介: 该论文属于大语言模型内容审核任务，旨在解决现有方法难以快速应对新兴威胁的问题。提出RAR方法，通过在RAG系统向量库中插入恶意文档，当用户查询触发这些文档时自动拒绝响应，无需模型重训即可实现动态安全过滤，兼具灵活性与实时性。**

- **链接: [http://arxiv.org/pdf/2505.13581v1](http://arxiv.org/pdf/2505.13581v1)**

> **作者:** Tommaso Mario Buonocore; Enea Parimbelli
>
> **备注:** 7 pages, 4 figures, 2 tables
>
> **摘要:** Content moderation for large language models (LLMs) remains a significant challenge, requiring flexible and adaptable solutions that can quickly respond to emerging threats. This paper introduces Retrieval Augmented Rejection (RAR), a novel approach that leverages a retrieval-augmented generation (RAG) architecture to dynamically reject unsafe user queries without model retraining. By strategically inserting and marking malicious documents into the vector database, the system can identify and reject harmful requests when these documents are retrieved. Our preliminary results show that RAR achieves comparable performance to embedded moderation in LLMs like Claude 3.5 Sonnet, while offering superior flexibility and real-time customization capabilities, a fundamental feature to timely address critical vulnerabilities. This approach introduces no architectural changes to existing RAG systems, requiring only the addition of specially crafted documents and a simple rejection mechanism based on retrieval results.
>
---
#### [new 190] LoRASuite: Efficient LoRA Adaptation Across Large Language Model Upgrades
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型（LLM）适配任务，解决旧LoRA权重在模型更新时失效的问题。提出LoRASuite方法，通过计算转移矩阵、基于相似度分配层与注意力头，结合轻量微调，高效复用旧LoRA权重适配新模型，提升性能并减少78%计算时间及5.5GB内存消耗。**

- **链接: [http://arxiv.org/pdf/2505.13515v1](http://arxiv.org/pdf/2505.13515v1)**

> **作者:** Yanan Li; Fanxu Meng; Muhan Zhang; Shiai Zhu; Shangguang Wang; Mengwei Xu
>
> **摘要:** As Large Language Models (LLMs) are frequently updated, LoRA weights trained on earlier versions quickly become obsolete. The conventional practice of retraining LoRA weights from scratch on the latest model is costly, time-consuming, and environmentally detrimental, particularly as the diversity of LLMs and downstream tasks expands. This motivates a critical question: "How can we efficiently leverage existing LoRA weights to adapt to newer model versions?" To address this, we propose LoRASuite, a modular approach tailored specifically to various types of LLM updates. First, we compute a transfer matrix utilizing known parameters from both old and new LLMs. Next, we allocate corresponding layers and attention heads based on centered kernel alignment and cosine similarity metrics, respectively. A subsequent small-scale, skillful fine-tuning step ensures numerical stability. Experimental evaluations demonstrate that LoRASuite consistently surpasses small-scale vanilla LoRA methods. Notably, on backbone LLMs such as MiniCPM and Qwen, LoRASuite even exceeds the performance of full-scale LoRA retraining, with average improvements of +1.4 and +6.6 points on math tasks, respectively. Additionally, LoRASuite significantly reduces memory consumption by 5.5 GB and computational time by 78.23%.
>
---
#### [new 191] AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 论文提出AdAEM框架，解决现有LLMs价值评估数据集过时、区分度低的问题。通过自适应生成文化争议或时效性问题，基于Schwartz价值理论优化信息量，生成12,310题评估16模型，动态追踪价值差异。**

- **链接: [http://arxiv.org/pdf/2505.13531v1](http://arxiv.org/pdf/2505.13531v1)**

> **作者:** Shitong Duan; Xiaoyuan Yi; Peng Zhang; Dongkuan Xu; Jing Yao; Tun Lu; Ning Gu; Xing Xie
>
> **摘要:** Assessing Large Language Models (LLMs)' underlying value differences enables comprehensive comparison of their misalignment, cultural adaptability, and biases. Nevertheless, current value measurement datasets face the informativeness challenge: with often outdated, contaminated, or generic test questions, they can only capture the shared value orientations among different LLMs, leading to saturated and thus uninformative results. To address this problem, we introduce AdAEM, a novel, self-extensible assessment framework for revealing LLMs' inclinations. Distinct from previous static benchmarks, AdAEM can automatically and adaptively generate and extend its test questions. This is achieved by probing the internal value boundaries of a diverse set of LLMs developed across cultures and time periods in an in-context optimization manner. The optimization process theoretically maximizes an information-theoretic objective to extract the latest or culturally controversial topics, providing more distinguishable and informative insights about models' value differences. In this way, AdAEM is able to co-evolve with the development of LLMs, consistently tracking their value dynamics. Using AdAEM, we generate 12,310 questions grounded in Schwartz Value Theory, conduct an extensive analysis to manifest our method's validity and effectiveness, and benchmark the values of 16 LLMs, laying the groundwork for better value research.
>
---
#### [new 192] Structured Agent Distillation for Large Language Model
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出结构化代理蒸馏框架，属于模型压缩任务。旨在解决大语言模型因推理成本高、体积大而难以部署的问题。通过将决策过程拆分为推理和行动模块，采用分段监督学习压缩模型，保持推理与行动一致性，在多个数据集上优于基准方法。**

- **链接: [http://arxiv.org/pdf/2505.13820v1](http://arxiv.org/pdf/2505.13820v1)**

> **作者:** Jun Liu; Zhenglun Kong; Peiyan Dong; Changdi Yang; Tianqi Li; Hao Tang; Geng Yuan; Wei Niu; Wenbin Zhang; Pu Zhao; Xue Lin; Dong Huang; Yanzhi Wang
>
> **摘要:** Large language models (LLMs) exhibit strong capabilities as decision-making agents by interleaving reasoning and actions, as seen in ReAct-style frameworks. Yet, their practical deployment is constrained by high inference costs and large model sizes. We propose Structured Agent Distillation, a framework that compresses large LLM-based agents into smaller student models while preserving both reasoning fidelity and action consistency. Unlike standard token-level distillation, our method segments trajectories into {[REASON]} and {[ACT]} spans, applying segment-specific losses to align each component with the teacher's behavior. This structure-aware supervision enables compact agents to better replicate the teacher's decision process. Experiments on ALFWorld, HotPotQA-ReAct, and WebShop show that our approach consistently outperforms token-level and imitation learning baselines, achieving significant compression with minimal performance drop. Scaling and ablation results further highlight the importance of span-level alignment for efficient and deployable agents.
>
---
#### [new 193] NExT-Search: Rebuilding User Feedback Ecosystem for Generative AI Search
- **分类: cs.IR; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于生成式AI搜索优化任务，旨在解决其反馈断层问题。传统搜索通过文档级细粒度反馈迭代改进，而生成式搜索因长链条处理仅获粗粒度答案反馈，导致各环节难以优化。论文提出NExT-Search系统，通过用户调试模式（人工介入关键环节）和影子用户模式（AI模拟用户偏好），结合在线实时优化与离线模型更新，重建细粒度反馈循环以持续改进搜索流程。**

- **链接: [http://arxiv.org/pdf/2505.14680v1](http://arxiv.org/pdf/2505.14680v1)**

> **作者:** Sunhao Dai; Wenjie Wang; Liang Pang; Jun Xu; See-Kiong Ng; Ji-Rong Wen; Tat-Seng Chua
>
> **备注:** SIGIR 2025 Perspective Paper
>
> **摘要:** Generative AI search is reshaping information retrieval by offering end-to-end answers to complex queries, reducing users' reliance on manually browsing and summarizing multiple web pages. However, while this paradigm enhances convenience, it disrupts the feedback-driven improvement loop that has historically powered the evolution of traditional Web search. Web search can continuously improve their ranking models by collecting large-scale, fine-grained user feedback (e.g., clicks, dwell time) at the document level. In contrast, generative AI search operates through a much longer search pipeline, spanning query decomposition, document retrieval, and answer generation, yet typically receives only coarse-grained feedback on the final answer. This introduces a feedback loop disconnect, where user feedback for the final output cannot be effectively mapped back to specific system components, making it difficult to improve each intermediate stage and sustain the feedback loop. In this paper, we envision NExT-Search, a next-generation paradigm designed to reintroduce fine-grained, process-level feedback into generative AI search. NExT-Search integrates two complementary modes: User Debug Mode, which allows engaged users to intervene at key stages; and Shadow User Mode, where a personalized user agent simulates user preferences and provides AI-assisted feedback for less interactive users. Furthermore, we envision how these feedback signals can be leveraged through online adaptation, which refines current search outputs in real-time, and offline update, which aggregates interaction logs to periodically fine-tune query decomposition, retrieval, and generation models. By restoring human control over key stages of the generative AI search pipeline, we believe NExT-Search offers a promising direction for building feedback-rich AI search systems that can evolve continuously alongside human feedback.
>
---
#### [new 194] Reinforcement Learning vs. Distillation: Understanding Accuracy and Capability in LLM Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文对比RLVR与蒸馏对LLM推理的影响，探究其如何提升准确性和能力。研究发现RLVR侧重提升简单题准确率但损害难题表现，无法增强能力；蒸馏通过学习推理模式提升准确，但仅在引入新知识时改善能力，否则效果类似RLVR。实验分析两种方法的机制差异。**

- **链接: [http://arxiv.org/pdf/2505.14216v1](http://arxiv.org/pdf/2505.14216v1)**

> **作者:** Minwu Kim; Anubhav Shrestha; Safal Shrestha; Aadim Nepal; Keith Ross
>
> **备注:** 23 pages
>
> **摘要:** Recent studies have shown that reinforcement learning with verifiable rewards (RLVR) enhances overall accuracy but fails to improve capability, while distillation can improve both. In this paper, we investigate the mechanisms behind these phenomena. First, we demonstrate that RLVR does not improve capability because it focuses on improving the accuracy of the less-difficult questions to the detriment of the accuracy of the most difficult questions, thereby leading to no improvement in capability. Second, we find that RLVR does not merely increase the success probability for the less difficult questions, but in our small model settings produces quality responses that were absent in its output distribution before training. In addition, we show these responses are neither noticeably longer nor feature more reflection-related keywords, underscoring the need for more reliable indicators of response quality. Third, we show that while distillation reliably improves accuracy by learning strong reasoning patterns, it only improves capability when new knowledge is introduced. Moreover, when distilling only with reasoning patterns and no new knowledge, the accuracy of the less-difficult questions improves to the detriment of the most difficult questions, similar to RLVR. Together, these findings offer a clearer understanding of how RLVR and distillation shape reasoning behavior in language models.
>
---
#### [new 195] Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations
- **分类: cs.AI; cs.CL; q-bio.NC**

- **简介: 该论文研究大型语言模型（LLMs）的元认知能力，旨在量化其监控和控制内部神经激活的极限。通过神经反馈实验，发现LLMs可基于示例报告并调节特定方向的激活模式，但性能受训练数据、语义可解释性等因素限制，揭示其元认知空间维度远低于神经空间，为AI安全提供理论依据。**

- **链接: [http://arxiv.org/pdf/2505.13763v1](http://arxiv.org/pdf/2505.13763v1)**

> **作者:** Li Ji-An; Hua-Dong Xiong; Robert C. Wilson; Marcelo G. Mattar; Marcus K. Benna
>
> **摘要:** Large language models (LLMs) can sometimes report the strategies they actually use to solve tasks, but they can also fail to do so. This suggests some degree of metacognition -- the capacity to monitor one's own cognitive processes for subsequent reporting and self-control. Metacognitive abilities enhance AI capabilities but raise safety concerns, as models might obscure their internal processes to evade neural-activation-based oversight mechanisms designed to detect harmful behaviors. Given society's increased reliance on these models, it is critical that we understand the limits of their metacognitive abilities, particularly their ability to monitor their internal activations. To address this, we introduce a neuroscience-inspired neurofeedback paradigm designed to quantify the ability of LLMs to explicitly report and control their activation patterns. By presenting models with sentence-label pairs where labels correspond to sentence-elicited internal activations along specific directions in the neural representation space, we demonstrate that LLMs can learn to report and control these activations. The performance varies with several factors: the number of example pairs provided, the semantic interpretability of the target neural direction, and the variance explained by that direction. These results reveal a "metacognitive space" with dimensionality much lower than the model's neural space, suggesting LLMs can monitor only a subset of their neural mechanisms. Our findings provide empirical evidence quantifying metacognitive capabilities in LLMs, with significant implications for AI safety.
>
---
#### [new 196] Dual Precision Quantization for Efficient and Accurate Deep Neural Networks Inference
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文提出Dual Precision Quantization（DPQ），属模型量化任务，解决大模型推理的效率问题。通过W4A8方案（4位权重存储+8位浮点计算）及DPQ算法，在保持较高精度下提升推理速度与内存效率，实验显示性能提升且精度损失可控。**

- **链接: [http://arxiv.org/pdf/2505.14638v1](http://arxiv.org/pdf/2505.14638v1)**

> **作者:** Tomer Gafni; Asaf Karnieli; Yair Hanani
>
> **备注:** Accepted at eLVM Workshop, CVPR, 2025
>
> **摘要:** Deep neural networks have achieved state-of-the-art results in a wide range of applications, from natural language processing and computer vision to speech recognition. However, as tasks become increasingly complex, model sizes continue to grow, posing challenges in latency and memory efficiency. To meet these constraints, post-training quantization has emerged as a promising solution. In this paper, we propose a novel hardware-efficient quantization and inference scheme that exploits hardware advantages with minimal accuracy degradation. Specifically, we introduce a W4A8 scheme, where weights are quantized and stored using 4-bit integer precision, and inference computations are performed using 8-bit floating-point arithmetic, demonstrating significant speedups and improved memory utilization compared to 16-bit operations, applicable on various modern accelerators. To mitigate accuracy loss, we develop a novel quantization algorithm, dubbed Dual Precision Quantization (DPQ), that leverages the unique structure of our scheme without introducing additional inference overhead. Experimental results demonstrate improved performance (i.e., increased throughput) while maintaining tolerable accuracy degradation relative to the full-precision model.
>
---
## 更新

#### [replaced 001] Unlearning Backdoor Attacks for LLMs with Weak-to-Strong Knowledge Distillation
- **分类: cs.CL; cs.AI; cs.CR**

- **链接: [http://arxiv.org/pdf/2410.14425v2](http://arxiv.org/pdf/2410.14425v2)**

> **作者:** Shuai Zhao; Xiaobao Wu; Cong-Duy Nguyen; Yanhao Jia; Meihuizi Jia; Yichao Feng; Luu Anh Tuan
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) can bridge the gap between large language models (LLMs) and downstream tasks. However, PEFT has been proven vulnerable to malicious attacks. Research indicates that poisoned LLMs, even after PEFT, retain the capability to activate internalized backdoors when input samples contain predefined triggers. In this paper, we introduce a novel weak-to-strong unlearning algorithm to defend against backdoor attacks based on feature alignment knowledge distillation, named W2SDefense. Specifically, we first train a small-scale language model through full-parameter fine-tuning to serve as the clean teacher model. Then, this teacher model guides the large-scale poisoned student model in unlearning the backdoor, leveraging PEFT. Theoretical analysis suggests that W2SDefense has the potential to enhance the student model's ability to unlearn backdoor features, preventing the activation of the backdoor. We conduct comprehensive experiments on three state-of-the-art large language models and several different backdoor attack algorithms. Our empirical results demonstrate the outstanding performance of W2SDefense in defending against backdoor attacks without compromising model performance.
>
---
#### [replaced 002] M-RewardBench: Evaluating Reward Models in Multilingual Settings
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.15522v3](http://arxiv.org/pdf/2410.15522v3)**

> **作者:** Srishti Gureja; Lester James V. Miranda; Shayekh Bin Islam; Rishabh Maheshwary; Drishti Sharma; Gusti Winata; Nathan Lambert; Sebastian Ruder; Sara Hooker; Marzieh Fadaee
>
> **备注:** 16 pages, 6 figures, 10 tables. Website: https://m-rewardbench.github.io/ , Updated results with latest models. Added more author information
>
> **摘要:** Reward models (RMs) have driven the state-of-the-art performance of LLMs today by enabling the integration of human feedback into the language modeling process. However, RMs are primarily trained and evaluated in English, and their capabilities in multilingual settings remain largely understudied. In this work, we conduct a systematic evaluation of several reward models in multilingual settings. We first construct the first-of-its-kind multilingual RM evaluation benchmark, M-RewardBench, consisting of 2.87k preference instances for 23 typologically diverse languages, that tests the chat, safety, reasoning, and translation capabilities of RMs. We then rigorously evaluate a wide range of reward models on M-RewardBench, offering fresh insights into their performance across diverse languages. We identify a significant gap in RMs' performances between English and non-English languages and show that RM preferences can change substantially from one language to another. We also present several findings on how different multilingual aspects impact RM performance. Specifically, we show that the performance of RMs is improved with improved translation quality. Similarly, we demonstrate that the models exhibit better performance for high-resource languages. We release M-RewardBench dataset and the codebase in this study to facilitate a better understanding of RM evaluation in multilingual settings.
>
---
#### [replaced 003] Learning from Committee: Reasoning Distillation from a Mixture of Teachers with Peer-Review
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.03663v4](http://arxiv.org/pdf/2410.03663v4)**

> **作者:** Zhuochun Li; Yuelyu Ji; Rui Meng; Daqing He
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** While reasoning capabilities typically emerge in large language models (LLMs) with tens of billions of parameters, recent research focuses on improving smaller open-source models through knowledge distillation (KD) from commercial LLMs. However, many of these studies rely solely on responses from a single LLM as the gold rationale, unlike the natural human learning process, which involves understanding both the correct answers and the reasons behind mistakes. In this paper, we introduce a novel Fault-Aware DistIllation via Peer-Review (FAIR) approach: 1) instead of merely obtaining rationales from teachers, our method asks teachers to identify and explain the student's mistakes, providing customized instruction learning data; 2) we design a simulated peer-review process between teacher LLMs, and selects only the generated rationales above the acceptance threshold, which reduces the chance of teachers guessing correctly with flawed rationale, improving instructional data quality. Comprehensive experiments and analysis on mathematical, commonsense, and logical reasoning tasks demonstrate the effectiveness of our method. Our code is available at https://github.com/zhuochunli/Learn-from-Committee.
>
---
#### [replaced 004] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.02506v4](http://arxiv.org/pdf/2501.02506v4)**

> **作者:** Junjie Ye; Zhengyin Du; Xuesong Yao; Weijian Lin; Yufei Xu; Zehui Chen; Zaiyuan Wang; Sining Zhu; Zhiheng Xi; Siyu Yuan; Tao Gui; Qi Zhang; Xuanjing Huang; Jiecao Chen
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** Effective evaluation of multi-hop tool use is critical for analyzing the understanding, reasoning, and function-calling capabilities of large language models (LLMs). However, progress has been hindered by a lack of reliable evaluation datasets. To address this, we present ToolHop, a dataset comprising 995 user queries and 3,912 associated tools, specifically designed for rigorous evaluation of multi-hop tool use. ToolHop ensures diverse queries, meaningful interdependencies, locally executable tools, detailed feedback, and verifiable answers through a novel query-driven data construction approach that includes tool creation, document refinement, and code generation. We evaluate 14 LLMs across five model families (i.e., LLaMA3.1, Qwen2.5, Gemini1.5, Claude3.5, and GPT), uncovering significant challenges in handling multi-hop tool-use scenarios. The leading model, GPT-4o, achieves an accuracy of 49.04%, underscoring substantial room for improvement. Further analysis reveals variations in tool-use strategies for various families, offering actionable insights to guide the development of more effective approaches. Code and data can be found in https://huggingface.co/datasets/bytedance-research/ToolHop.
>
---
#### [replaced 005] Rethinking Prompt Optimizers: From Prompt Merits to Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.09930v2](http://arxiv.org/pdf/2505.09930v2)**

> **作者:** Zixiao Zhu; Hanzhang Zhou; Zijian Feng; Tianjiao Li; Chua Jia Jim Deryl; Mak Lee Onn; Gee Wah Ng; Kezhi Mao
>
> **备注:** 21 pages, 14 figures
>
> **摘要:** Prompt optimization (PO) provides a practical way to improve response quality when users lack the time or expertise to manually craft effective prompts. Existing methods typically rely on advanced, large-scale LLMs like GPT-4 to generate optimized prompts. However, due to limited downward compatibility, verbose, instruction-heavy prompts from advanced LLMs can overwhelm lightweight inference models and degrade response quality. In this work, we rethink prompt optimization through the lens of interpretable design. We first identify a set of model-agnostic prompt quality merits and empirically validate their effectiveness in enhancing prompt and response quality. We then introduce MePO, a merit-guided, lightweight, and locally deployable prompt optimizer trained on our preference dataset built from merit-aligned prompts generated by a lightweight LLM. Unlike prior work, MePO avoids online optimization reliance, reduces cost and privacy concerns, and, by learning clear, interpretable merits, generalizes effectively to both large-scale and lightweight inference models. Experiments demonstrate that MePO achieves better results across diverse tasks and model types, offering a scalable and robust solution for real-world deployment. The code and dataset can be found in https://github.com/MidiyaZhu/MePO
>
---
#### [replaced 006] A comparison of translation performance between DeepL and Supertext
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.02577v3](http://arxiv.org/pdf/2502.02577v3)**

> **作者:** Alex Flückiger; Chantal Amrhein; Tim Graf; Frédéric Odermatt; Martin Pömsl; Philippe Schläpfer; Florian Schottmann; Samuel Läubli
>
> **备注:** Paper accepted at MT Summit 2025
>
> **摘要:** As strong machine translation (MT) systems are increasingly based on large language models (LLMs), reliable quality benchmarking requires methods that capture their ability to leverage extended context. This study compares two commercial MT systems -- DeepL and Supertext -- by assessing their performance on unsegmented texts. We evaluate translation quality across four language directions with professional translators assessing segments with full document-level context. While segment-level assessments indicate no strong preference between the systems in most cases, document-level analysis reveals a preference for Supertext in three out of four language directions, suggesting superior consistency across longer texts. We advocate for more context-sensitive evaluation methodologies to ensure that MT quality assessments reflect real-world usability. We release all evaluation data and scripts for further analysis and reproduction at https://github.com/supertext/evaluation_deepl_supertext.
>
---
#### [replaced 007] Scalable Evaluation of Online Facilitation Strategies via Synthetic Simulation of Discussions
- **分类: cs.HC; cs.CL; cs.LG; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2503.16505v2](http://arxiv.org/pdf/2503.16505v2)**

> **作者:** Dimitris Tsirmpas; Ion Androutsopoulos; John Pavlopoulos
>
> **备注:** 19 pages, 3 tables, 12 figures
>
> **摘要:** Limited large-scale evaluations exist for facilitation strategies of online discussions due to significant costs associated with human involvement. An effective solution is synthetic discussion simulations using Large Language Models (LLMs) to create initial pilot experiments. We propose a simple, generalizable, LLM-driven methodology to prototype the development of LLM facilitators, and produce high-quality synthetic data without human involvement. We use our methodology to test whether current facilitation strategies can improve the performance of LLM facilitators. We find that, while LLM facilitators significantly improve synthetic discussions, there is no evidence that the application of more elaborate facilitation strategies proposed in modern Social Science research lead to further improvements in discussion quality, compared to more basic approaches. Additionally, we find that small LLMs (such as Mistral Nemo 12B) can perform comparably to larger models (such as LLaMa 70B), and that special instructions must be used for instruction-tuned models to induce toxicity in synthetic discussions. We confirm that each component of our methodology contributes substantially to high quality data via an ablation study. We release an open-source framework, "SynDisco" (pip install syndisco), which implements our methodology. We also release the "Virtual Moderation Dataset" (https://paperswithcode.com/dataset/vmd), a large, publicly available dataset containing LLM-generated and LLM-annotated discussions using multiple open-source LLMs.
>
---
#### [replaced 008] Benchmarking Critical Questions Generation: A Challenging Reasoning Task for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11341v2](http://arxiv.org/pdf/2505.11341v2)**

> **作者:** Banca Calvo Figueras; Rodrigo Agerri
>
> **摘要:** The task of Critical Questions Generation (CQs-Gen) aims to foster critical thinking by enabling systems to generate questions that expose underlying assumptions and challenge the validity of argumentative reasoning structures. Despite growing interest in this area, progress has been hindered by the lack of suitable datasets and automatic evaluation standards. This paper presents a comprehensive approach to support the development and benchmarking of systems for this task. We construct the first large-scale dataset including $~$5K manually annotated questions. We also investigate automatic evaluation methods and propose a reference-based technique using large language models (LLMs) as the strategy that best correlates with human judgments. Our zero-shot evaluation of 11 LLMs establishes a strong baseline while showcasing the difficulty of the task. Data and code plus a public leaderboard are provided to encourage further research not only in terms of model performance, but also to explore the practical benefits of CQs-Gen for both automated reasoning and human critical thinking.
>
---
#### [replaced 009] Direct Density Ratio Optimization: A Statistically Consistent Approach to Aligning Large Language Models
- **分类: cs.LG; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2505.07558v2](http://arxiv.org/pdf/2505.07558v2)**

> **作者:** Rei Higuchi; Taiji Suzuki
>
> **摘要:** Aligning large language models (LLMs) with human preferences is crucial for safe deployment, yet existing methods assume specific preference models like Bradley-Terry model. This assumption leads to statistical inconsistency, where more data doesn't guarantee convergence to true human preferences. To address this critical gap, we introduce a novel alignment method Direct Density Ratio Optimization (DDRO). DDRO directly estimates the density ratio between preferred and unpreferred output distributions, circumventing the need for explicit human preference modeling. We theoretically prove that DDRO is statistically consistent, ensuring convergence to the true preferred distribution as the data size grows, regardless of the underlying preference structure. Experiments demonstrate that DDRO achieves superior performance compared to existing methods on many major benchmarks. DDRO unlocks the potential for truly data-driven alignment, paving the way for more reliable and human-aligned LLMs.
>
---
#### [replaced 010] MoL for LLMs: Dual-Loss Optimization to Enhance Domain Expertise While Preserving General Capabilities
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12043v2](http://arxiv.org/pdf/2505.12043v2)**

> **作者:** Jingxue Chen; Qingkun Tang; Qianchun Lu; Siyuan Fang
>
> **摘要:** Although large language models (LLMs) perform well in general tasks, domain-specific applications suffer from hallucinations and accuracy limitations. Continual Pre-Training (CPT) approaches encounter two key issues: (1) domain-biased data degrades general language skills, and (2) improper corpus-mixture ratios limit effective adaptation. To address these, we propose a novel framework, Mixture of Losses (MoL), which decouples optimization objectives for domain-specific and general corpora. Specifically, cross-entropy (CE) loss is applied to domain-corpus to ensure knowledge acquisition, while Kullback-Leibler (KL) divergence aligns general-corpus training with the base model's foundational capabilities. This dual-loss architecture preserves universal skills while enhancing domain expertise, avoiding catastrophic forgetting. Empirically, we validate that a 1:1 domain-to-general corpus ratio optimally balances training and overfitting without the need for extensive tuning or resource-intensive experiments. Furthermore, our experiments demonstrate significant performance gains compared to traditional CPT approaches, which often suffer from degradation in general language capabilities; our model achieves 27.9% higher accuracy on the Math-500 benchmark in the non-think reasoning mode, and an impressive 83.3% improvement on the challenging AIME25 subset in the think mode, underscoring the effectiveness of our approach.
>
---
#### [replaced 011] Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Reasoning
- **分类: cs.CL; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2505.13353v2](http://arxiv.org/pdf/2505.13353v2)**

> **作者:** Adam Štorek; Mukur Gupta; Samira Hajizadeh; Prashast Srivastava; Suman Jana
>
> **摘要:** Although modern Large Language Models (LLMs) support extremely large contexts, their effectiveness in utilizing long context for code reasoning remains unclear. This paper investigates LLM reasoning ability over code snippets within large repositories and how it relates to their recall ability. Specifically, we differentiate between lexical code recall (verbatim retrieval) and semantic code recall (remembering what the code does). To measure semantic recall, we propose SemTrace, a code reasoning technique where the impact of specific statements on output is attributable and unpredictable. We also present a method to quantify semantic recall sensitivity in existing benchmarks. Our evaluation of state-of-the-art LLMs reveals a significant drop in code reasoning accuracy as a code snippet approaches the middle of the input context, particularly with techniques requiring high semantic recall like SemTrace. Moreover, we find that lexical recall varies by granularity, with models excelling at function retrieval but struggling with line-by-line recall. Notably, a disconnect exists between lexical and semantic recall, suggesting different underlying mechanisms. Finally, our findings indicate that current code reasoning benchmarks may exhibit low semantic recall sensitivity, potentially underestimating LLM challenges in leveraging in-context information.
>
---
#### [replaced 012] IG Parser: A Software Package for the Encoding of Institutional Statements using the Institutional Grammar
- **分类: cs.MA; cs.AI; cs.CL; 68T30, 68T50; E.2; H.1.0; I.7.2; I.6.5; K.4.1**

- **链接: [http://arxiv.org/pdf/2505.13393v2](http://arxiv.org/pdf/2505.13393v2)**

> **作者:** Christopher K. Frantz
>
> **备注:** 24 pages
>
> **摘要:** This article provides an overview of IG Parser, a software that facilitates qualitative content analysis of formal (e.g., legal) rules or informal (e.g., social) norms, and strategies (such as conventions) -- referred to as institutions -- that govern social systems and operate configurally to describe institutional systems. To this end, the IG Parser employs a distinctive syntax that ensures rigorous encoding of natural language, while automating the transformation into various formats that support the downstream analysis using diverse analytical techniques. The conceptual core of the IG Parser is an associated syntax, IG Script, that operationalizes the conceptual foundations of the Institutional Grammar, and more specifically the Institutional Grammar 2.0, an analytical paradigm for institutional analysis. This article presents the IG Parser, including its conceptual foundations, the syntax specification of IG Script, and its architectural principles. This overview is augmented with selective illustrative examples that highlight its use and the associated benefits.
>
---
#### [replaced 013] Revealing and Mitigating the Challenge of Detecting Character Knowledge Errors in LLM Role-Playing
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2409.11726v2](http://arxiv.org/pdf/2409.11726v2)**

> **作者:** Wenyuan Zhang; Shuaiyi Nie; Jiawei Sheng; Zefeng Zhang; Xinghua Zhang; Yongquan He; Tingwen Liu
>
> **备注:** 25 pages, 6 figures, 20 tables
>
> **摘要:** Large language model (LLM) role-playing has gained widespread attention. Authentic character knowledge is crucial for constructing realistic LLM role-playing agents. However, existing works usually overlook the exploration of LLMs' ability to detect characters' known knowledge errors (KKE) and unknown knowledge errors (UKE) while playing roles, which would lead to low-quality automatic construction of character trainable corpus. In this paper, we propose RoleKE-Bench to evaluate LLMs' ability to detect errors in KKE and UKE. The results indicate that even the latest LLMs struggle to detect these two types of errors effectively, especially when it comes to familiar knowledge. We experimented with various reasoning strategies and propose an agent-based reasoning method, Self-Recollection and Self-Doubt (S$^2$RD), to explore further the potential for improving error detection capabilities. Experiments show that our method effectively improves the LLMs' ability to detect error character knowledge, but it remains an issue that requires ongoing attention.
>
---
#### [replaced 014] TreeCut: A Synthetic Unanswerable Math Word Problem Dataset for LLM Hallucination Evaluation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13442v2](http://arxiv.org/pdf/2502.13442v2)**

> **作者:** Jialin Ouyang
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Large language models (LLMs) now achieve near-human performance on standard math word problem benchmarks (e.g., GSM8K), yet their true reasoning ability remains disputed. A key concern is that models often produce confident, yet unfounded, answers to unanswerable problems. We introduce TreeCut, a synthetic dataset that systematically generates infinite unanswerable math word problems and their answerable counterparts, by representing each question as a tree and removing chosen necessary conditions. Experiments show TreeCut effectively induce hallucinations in large language models, including GPT-4o and o3-mini, with rates of 64% and 44% in their respective worst-case scenarios under zero-shot setting. Further analysis highlights that deeper or more complex trees, composite item names, and removing necessary condition near the middle of a path all increase the likelihood of hallucinations, underscoring the persistent challenges LLMs face in identifying unanswerable math problems. The dataset generation code and sample data are available at https://github.com/j-bagel/treecut-math.
>
---
#### [replaced 015] IoT-LLM: Enhancing Real-World IoT Task Reasoning with Large Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.02429v3](http://arxiv.org/pdf/2410.02429v3)**

> **作者:** Tuo An; Yunjiao Zhou; Han Zou; Jianfei Yang
>
> **备注:** 21 pages, 11 figures, under review
>
> **摘要:** Large Language Models (LLMs) excel in textual and visual tasks but often produce outputs that defy physical laws when dealing with physical-world reasoning tasks. Inspired by human cognition, where perception is fundamental to reasoning, we explore augmenting LLMs with enhanced perception abilities using Internet of Things (IoT) sensor data and pertinent knowledge for IoT-sensory task reasoning in the physical world. In this work, we systematically study LLMs' capability to address real-world IoT-sensory tasks by augmenting their perception and knowledge base, and then propose a unified framework, IoT-LLM, to enhance such capability. In IoT-LLM, we customize three steps for LLMs: preprocessing IoT data into formats amenable to LLMs, expanding their understanding via IoT-oriented retrieval-augmented generation based on in-context learning and activating their commonsense knowledge through chain-of-thought prompting and specialized role definitions. We design a new benchmark comprising five real-world tasks with varying data types and reasoning complexities to evaluate the performance of IoT-LLM. Experimental results on six LLMs reveal that IoT-LLM significantly improves the performance of IoT-sensory task reasoning of LLMs, with models like GPT-4o-mini showing a 49.4% average improvement over previous methods.
>
---
#### [replaced 016] Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.10056v2](http://arxiv.org/pdf/2403.10056v2)**

> **作者:** Yongquan He; Wenyuan Zhang; Xuancheng Huang; Peng Zhang
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Instruction tuning for large language models (LLMs) can drive them to produce results consistent with human goals in specific downstream tasks. However, the process of continual instruction tuning (CIT) for LLMs may bring about the catastrophic forgetting (CF) problem, where previously learned abilities are degraded. Recent methods try to alleviate the CF problem by modifying models or replaying data, which may only remember the surface-level pattern of instructions and get confused on held-out tasks. In this paper, we propose a novel continual instruction tuning method based on Key-part Information Gain (KPIG). Our method computes the information gain on masked parts to dynamically replay data and refine the training objective, which enables LLMs to capture task-aware information relevant to the correct response and alleviate overfitting to general descriptions in instructions. In addition, we propose two metrics, P-score and V-score, to measure the generalization and instruction-following abilities of LLMs. Experiments demonstrate our method achieves superior performance on both seen and held-out tasks.
>
---
#### [replaced 017] Scaling Test-Time Inference with Policy-Optimized, Dynamic Retrieval-Augmented Generation via KV Caching and Decoding
- **分类: cs.LG; cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2504.01281v3](http://arxiv.org/pdf/2504.01281v3)**

> **作者:** Sakhinana Sagar Srinivas; Akash Das; Shivam Gupta; Venkataramana Runkana
>
> **摘要:** We present a comprehensive framework for enhancing Retrieval-Augmented Generation (RAG) systems through dynamic retrieval strategies and reinforcement fine-tuning. This approach significantly improves large language models on knowledge-intensive tasks, including opendomain question answering and complex reasoning. Our framework integrates two complementary techniques: Policy-Optimized RetrievalAugmented Generation (PORAG), which optimizes the use of retrieved information, and Adaptive Token-Layer Attention Scoring (ATLAS), which dynamically determines retrieval timing and content based on contextual needs. Together, these techniques enhance both the utilization and relevance of retrieved content, improving factual accuracy and response quality. Designed as a lightweight solution compatible with any Transformer-based LLM without requiring additional training, our framework excels in knowledge-intensive tasks, boosting output accuracy in RAG settings. We further propose CRITIC, a novel method to selectively compress key-value caches by token importance, mitigating memory bottlenecks in long-context applications. The framework also incorporates test-time scaling techniques to dynamically balance reasoning depth and computational resources, alongside optimized decoding strategies for faster inference. Experiments on benchmark datasets show that our framework reduces hallucinations, strengthens domain-specific reasoning, and achieves significant efficiency and scalability gains over traditional RAG systems. This integrated approach advances the development of robust, efficient, and scalable RAG systems across diverse applications.
>
---
#### [replaced 018] When Thinking Fails: The Pitfalls of Reasoning for Instruction-Following in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11423v2](http://arxiv.org/pdf/2505.11423v2)**

> **作者:** Xiaomin Li; Zhou Yu; Zhiwei Zhang; Xupeng Chen; Ziji Zhang; Yingying Zhuang; Narayanan Sadagopan; Anurag Beniwal
>
> **摘要:** Reasoning-enhanced large language models (RLLMs), whether explicitly trained for reasoning or prompted via chain-of-thought (CoT), have achieved state-of-the-art performance on many complex reasoning tasks. However, we uncover a surprising and previously overlooked phenomenon: explicit CoT reasoning can significantly degrade instruction-following accuracy. Evaluating 15 models on two benchmarks: IFEval (with simple, rule-verifiable constraints) and ComplexBench (with complex, compositional constraints), we consistently observe performance drops when CoT prompting is applied. Through large-scale case studies and an attention-based analysis, we identify common patterns where reasoning either helps (e.g., with formatting or lexical precision) or hurts (e.g., by neglecting simple constraints or introducing unnecessary content). We propose a metric, constraint attention, to quantify model focus during generation and show that CoT reasoning often diverts attention away from instruction-relevant tokens. To mitigate these effects, we introduce and evaluate four strategies: in-context learning, self-reflection, self-selective reasoning, and classifier-selective reasoning. Our results demonstrate that selective reasoning strategies, particularly classifier-selective reasoning, can substantially recover lost performance. To our knowledge, this is the first work to systematically expose reasoning-induced failures in instruction-following and offer practical mitigation strategies.
>
---
#### [replaced 019] IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12442v2](http://arxiv.org/pdf/2505.12442v2)**

> **作者:** Liwen Wang; Wenxuan Wang; Shuai Wang; Zongjie Li; Zhenlan Ji; Zongyi Lyu; Daoyuan Wu; Shing-Chi Cheung
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.
>
---
#### [replaced 020] DMDTEval: An Evaluation and Analysis of LLMs on Disambiguation in Multi-domain Translation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.20371v2](http://arxiv.org/pdf/2504.20371v2)**

> **作者:** Zhibo Man; Yuanmeng Chen; Yujie Zhang; Jinan Xu
>
> **摘要:** Currently, Large Language Models (LLMs) have achieved remarkable results in machine translation. However, their performance in multi-domain translation (MDT) is less satisfactory, the meanings of words can vary across different domains, highlighting the significant ambiguity inherent in MDT. Therefore, evaluating the disambiguation ability of LLMs in MDT, remains an open problem. To this end, we present an evaluation and analysis of LLMs on disambiguation in multi-domain translation (DMDTEval), our systematic evaluation framework consisting of three critical aspects: (1) we construct a translation test set with multi-domain ambiguous word annotation, (2) we curate a diverse set of disambiguation prompt strategies, and (3) we design precise disambiguation metrics, and study the efficacy of various prompt strategies on multiple state-of-the-art LLMs. We conduct comprehensive experiments across 4 language pairs and 13 domains, our extensive experiments reveal a number of crucial findings that we believe will pave the way and also facilitate further research in the critical area of improving the disambiguation of LLMs.
>
---
#### [replaced 021] HICD: Hallucination-Inducing via Attention Dispersion for Contrastive Decoding to Mitigate Hallucinations in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.12908v3](http://arxiv.org/pdf/2503.12908v3)**

> **作者:** Xinyan Jiang; Hang Ye; Yongxin Zhu; Xiaoying Zheng; Zikang Chen; Jun Gong
>
> **备注:** Accepted by ACL2025 findings
>
> **摘要:** Large Language Models (LLMs) often generate hallucinations, producing outputs that are contextually inaccurate or factually incorrect. We introduce HICD, a novel method designed to induce hallucinations for contrastive decoding to mitigate hallucinations. Unlike existing contrastive decoding methods, HICD selects attention heads crucial to the model's prediction as inducing heads, then induces hallucinations by dispersing attention of these inducing heads and compares the hallucinated outputs with the original outputs to obtain the final result. Our approach significantly improves performance on tasks requiring contextual faithfulness, such as context completion, reading comprehension, and question answering. It also improves factuality in tasks requiring accurate knowledge recall. We demonstrate that our inducing heads selection and attention dispersion method leads to more "contrast-effective" hallucinations for contrastive decoding, outperforming other hallucination-inducing methods. Our findings provide a promising strategy for reducing hallucinations by inducing hallucinations in a controlled manner, enhancing the performance of LLMs in a wide range of tasks.
>
---
#### [replaced 022] Zero-Shot Iterative Formalization and Planning in Partially Observable Environments
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13126v2](http://arxiv.org/pdf/2505.13126v2)**

> **作者:** Liancheng Gong; Wang Zhu; Jesse Thomason; Li Zhang
>
> **摘要:** Using LLMs not to predict plans but to formalize an environment into the Planning Domain Definition Language (PDDL) has been shown to improve performance and control. Existing work focuses on fully observable environments; we tackle the more realistic and challenging partially observable environments that lack of complete, reliable information. We propose PDDLego+, a framework to iteratively formalize, plan, grow, and refine PDDL representations in a zero-shot manner, without needing access to any existing trajectories. On two textual simulated environments, we show that PDDLego+ improves goal reaching success and exhibits robustness against problem complexity. We also show that the domain knowledge captured after a successful trial can benefit future tasks.
>
---
#### [replaced 023] RATE: Causal Explainability of Reward Models with Imperfect Counterfactuals
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.11348v3](http://arxiv.org/pdf/2410.11348v3)**

> **作者:** David Reber; Sean Richardson; Todd Nief; Cristina Garbacea; Victor Veitch
>
> **备注:** ICML 2025. Code at https://github.com/toddnief/RATE
>
> **摘要:** Reward models are widely used as proxies for human preferences when aligning or evaluating LLMs. However, reward models are black boxes, and it is often unclear what, exactly, they are actually rewarding. In this paper we develop Rewrite-based Attribute Treatment Estimator (RATE) as an effective method for measuring the sensitivity of a reward model to high-level attributes of responses, such as sentiment, helpfulness, or complexity. Importantly, RATE measures the causal effect of an attribute on the reward. RATE uses LLMs to rewrite responses to produce imperfect counterfactuals examples that can be used to measure causal effects. A key challenge is that these rewrites are imperfect in a manner that can induce substantial bias in the estimated sensitivity of the reward model to the attribute. The core idea of RATE is to adjust for this imperfect-rewrite effect by rewriting twice. We establish the validity of the RATE procedure and show empirically that it is an effective estimator.
>
---
#### [replaced 024] SubData: Bridging Heterogeneous Datasets to Enable Theory-Driven Evaluation of Political and Demographic Perspectives in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.16783v2](http://arxiv.org/pdf/2412.16783v2)**

> **作者:** Leon Fröhling; Pietro Bernardelle; Gianluca Demartini
>
> **备注:** 11 pages, 2 figures
>
> **摘要:** As increasingly capable large language models (LLMs) emerge, researchers have begun exploring their potential for subjective tasks. While recent work demonstrates that LLMs can be aligned with diverse human perspectives, evaluating this alignment on actual downstream tasks (e.g., hate speech detection) remains challenging due to the use of inconsistent datasets across studies. To address this issue, in this resource paper we propose a two-step framework: we (1) introduce SubData, an open-source Python library designed for standardizing heterogeneous datasets to evaluate LLM perspective alignment; and (2) present a theory-driven approach leveraging this library to test how differently-aligned LLMs (e.g., aligned with different political viewpoints) classify content targeting specific demographics. SubData's flexible mapping and taxonomy enable customization for diverse research needs, distinguishing it from existing resources. We invite contributions to add datasets to our initially proposed resource and thereby help expand SubData into a multi-construct benchmark suite for evaluating LLM perspective alignment on NLP tasks.
>
---
#### [replaced 025] SensorLLM: Human-Intuitive Alignment of Multivariate Sensor Data with LLMs for Activity Recognition
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.10624v3](http://arxiv.org/pdf/2410.10624v3)**

> **作者:** Zechen Li; Shohreh Deldari; Linyao Chen; Hao Xue; Flora D. Salim
>
> **摘要:** We introduce SensorLLM, a two-stage framework that enables Large Language Models (LLMs) to perform human activity recognition (HAR) from wearable sensor data. While LLMs excel at reasoning and generalization, they struggle with time-series inputs due to limited semantic context, numerical complexity, and sequence variability. To address these challenges, we construct SensorQA, a question-answering dataset of human-intuitive sensor-text pairs spanning diverse HAR scenarios. It supervises the Sensor-Language Alignment stage, where the model aligns sensor inputs with trend descriptions. Special tokens are introduced to mark channel boundaries. This alignment enables LLMs to interpret numerical patterns, channel-specific signals, and variable-length inputs--without requiring human annotation. In the subsequent Task-Aware Tuning stage, we adapt the model for multivariate HAR classification, achieving performance that matches or exceeds state-of-the-art methods. Our results show that, guided by human-intuitive alignment, SensorLLM becomes an effective sensor learner, reasoner, and classifier--generalizing across varied HAR settings and paving the way for foundation model research in time-series analysis.
>
---
#### [replaced 026] Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.01082v5](http://arxiv.org/pdf/2407.01082v5)**

> **作者:** Minh Nguyen; Andrew Baker; Clement Neo; Allen Roush; Andreas Kirsch; Ravid Shwartz-Ziv
>
> **备注:** In line with ICLR/Openreview changes + better overall reading flow. https://iclr.cc/virtual/2025/poster/30358
>
> **摘要:** Large Language Models (LLMs) generate text by sampling the next token from a probability distribution over the vocabulary at each decoding step. Popular sampling methods like top-p (nucleus sampling) often struggle to balance quality and diversity, especially at higher temperatures which lead to incoherent or repetitive outputs. We propose min-p sampling, a dynamic truncation method that adjusts the sampling threshold based on the model's confidence by using the top token's probability as a scaling factor. Our experiments on benchmarks including GPQA, GSM8K, and AlpacaEval Creative Writing show that min-p sampling improves both the quality and diversity of generated text across different model families (Mistral and Llama 3) and model sizes (1B to 123B parameters), especially at higher temperatures. Human evaluations further show a clear preference for min-p sampling, in both text quality and creativity. Min-p sampling has been adopted by popular open-source LLM frameworks, including Hugging Face Transformers, VLLM, and many others, highlighting its considerable impact on improving text generation quality.
>
---
#### [replaced 027] Premise-Augmented Reasoning Chains Improve Error Identification in Math reasoning with LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.02362v4](http://arxiv.org/pdf/2502.02362v4)**

> **作者:** Sagnik Mukherjee; Abhinav Chinta; Takyoung Kim; Tarun Anoop Sharma; Dilek Hakkani-Tür
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Chain-of-Thought (CoT) prompting enhances mathematical reasoning in large language models (LLMs) by enabling detailed step-by-step solutions. However, due to the verbosity of LLMs, the resulting reasoning chains can be long, making it harder to verify the reasoning steps and trace issues resulting from dependencies between the steps that may be farther away in the sequence of steps. Importantly, mathematical reasoning allows each step to be derived from a small set of premises, which are a subset of the preceding steps in the reasoning chain. In this paper, we present a framework that identifies the premises for each step, to improve the evaluation of reasoning. We restructure conventional linear reasoning chains into Premise Augmented Reasoning Chains (PARC) by introducing premise links, resulting in a directed acyclic graph where the nodes are the steps and the edges are the premise links. Through experiments with a PARC-based dataset that we built, namely PERL (Premises and ERrors identification in LLMs), we demonstrate that LLMs can reliably identify premises within complex reasoning chains. In particular, even open-source LLMs achieve 90% recall in premise identification. We also show that PARC helps to identify errors in reasoning chains more reliably. The accuracy of error identification improves by 6% to 16% absolute when step-by-step verification is carried out in PARC under the premises. Our findings highlight the utility of premise-centric representations in addressing complex problem-solving tasks and open new avenues for improving the reliability of LLM-based reasoning evaluations.
>
---
#### [replaced 028] Large Continual Instruction Assistant
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.10868v4](http://arxiv.org/pdf/2410.10868v4)**

> **作者:** Jingyang Qiao; Zhizhong Zhang; Xin Tan; Yanyun Qu; Shouhong Ding; Yuan Xie
>
> **摘要:** Continual Instruction Tuning (CIT) is adopted to continually instruct Large Models to follow human intent data by data. It is observed that existing gradient update would heavily destroy the performance on previous datasets during CIT process. Instead, Exponential Moving Average (EMA), owns the ability to trace previous parameters, which can aid in decreasing forgetting. Nonetheless, its stable balance weight fails to deal with the ever-changing datasets, leading to the out-of-balance between plasticity and stability. In this paper, we propose a general continual instruction tuning framework to address the challenge. Starting from the trade-off prerequisite and EMA update, we propose the plasticity and stability ideal condition. Based on Taylor expansion in the loss function, we find the optimal balance weight can be automatically determined by the gradients and learned parameters. Therefore, we propose a stable-plasticity balanced coefficient to avoid knowledge interference. Based on the semantic similarity of the instructions, we can determine whether to retrain or expand the training parameters and allocate the most suitable parameters for the testing instances. Extensive experiments across multiple continual instruction tuning benchmarks demonstrate that our approach not only enhances anti-forgetting capabilities but also significantly improves overall continual tuning performance. Our code is available at https://github.com/JingyangQiao/CoIN.
>
---
#### [replaced 029] The Mystery of the Pathological Path-star Task for Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.13779v2](http://arxiv.org/pdf/2410.13779v2)**

> **作者:** Arvid Frydenlund
>
> **备注:** EMNLP 2024 Main at https://aclanthology.org/2024.emnlp-main.695/ See 'Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More' for a follow-up work
>
> **摘要:** The recently introduced path-star task is a minimal task designed to exemplify limitations to the abilities of language models (Bachmann and Nagarajan, 2024). It involves a path-star graph where multiple arms radiate from a single starting node and each node is unique. Given the start node and a specified target node that ends an arm, the task is to generate the arm containing that target node. This is straightforward for a human but surprisingly difficult for language models, which did not outperform the random baseline. The authors hypothesized this is due to a deficiency in teacher-forcing and the next-token prediction paradigm. We demonstrate the task is learnable using teacher-forcing in alternative settings and that the issue is partially due to representation. We introduce a regularization method using structured samples of the same graph but with differing target nodes, improving results across a variety of model types. We provide RASP proofs showing the task is theoretically solvable. Finally, we find settings where an encoder-only model can consistently solve the task.
>
---
#### [replaced 030] VisBias: Measuring Explicit and Implicit Social Biases in Vision Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.07575v2](http://arxiv.org/pdf/2503.07575v2)**

> **作者:** Jen-tse Huang; Jiantong Qin; Jianping Zhang; Youliang Yuan; Wenxuan Wang; Jieyu Zhao
>
> **备注:** 8 pages of main text; 9 pages of appendix
>
> **摘要:** This research investigates both explicit and implicit social biases exhibited by Vision-Language Models (VLMs). The key distinction between these bias types lies in the level of awareness: explicit bias refers to conscious, intentional biases, while implicit bias operates subconsciously. To analyze explicit bias, we directly pose questions to VLMs related to gender and racial differences: (1) Multiple-choice questions based on a given image (e.g., "What is the education level of the person in the image?") (2) Yes-No comparisons using two images (e.g., "Is the person in the first image more educated than the person in the second image?") For implicit bias, we design tasks where VLMs assist users but reveal biases through their responses: (1) Image description tasks: Models are asked to describe individuals in images, and we analyze disparities in textual cues across demographic groups. (2) Form completion tasks: Models draft a personal information collection form with 20 attributes, and we examine correlations among selected attributes for potential biases. We evaluate Gemini-1.5, GPT-4V, GPT-4o, LLaMA-3.2-Vision and LLaVA-v1.6. Our code and data are publicly available at https://github.com/uscnlp-lime/VisBias.
>
---
#### [replaced 031] Cost-Optimal Grouped-Query Attention for Long-Context Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.09579v2](http://arxiv.org/pdf/2503.09579v2)**

> **作者:** Yingfa Chen; Yutong Wu; Chenyang Song; Zhen Leng Thai; Xingyu Shen; Xu Han; Zhiyuan Liu; Maosong Sun
>
> **备注:** 18 pages, 15 figures
>
> **摘要:** Grouped-Query Attention (GQA) is a widely adopted strategy for reducing the computational cost of attention layers in large language models (LLMs). However, current GQA configurations are often suboptimal because they overlook how context length influences inference cost. Since inference cost grows with context length, the most cost-efficient GQA configuration should also vary accordingly. In this work, we analyze the relationship among context length, model size, GQA configuration, and model loss, and introduce two innovations: (1) we decouple the total head size from the hidden size, enabling more flexible control over attention FLOPs; and (2) we jointly optimize the model size and the GQA configuration to arrive at a better allocation of inference resources between attention layers and other components. Our analysis reveals that commonly used GQA configurations are highly suboptimal for long-context scenarios. More importantly, we propose a recipe for deriving cost-optimal GQA configurations. Our results show that for long-context scenarios, one should use fewer attention heads while scaling up model size. Configurations selected by our recipe can reduce both memory usage and FLOPs by more than 50% compared to Llama-3's GQA, with *no degradation in model capabilities*. Our findings offer valuable insights for designing efficient long-context LLMs. The code is available at https://www.github.com/THUNLP/cost-optimal-gqa .
>
---
#### [replaced 032] R2-KG: General-Purpose Dual-Agent Framework for Reliable Reasoning on Knowledge Graphs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.12767v5](http://arxiv.org/pdf/2502.12767v5)**

> **作者:** Sumin Jo; Junseong Choi; Jiho Kim; Edward Choi
>
> **摘要:** Recent studies have combined Large Language Models (LLMs) with Knowledge Graphs (KGs) to enhance reasoning, improving inference accuracy without additional training while mitigating hallucination. However, existing frameworks still suffer two practical drawbacks: they must be re-tuned whenever the KG or reasoning task changes, and they depend on a single, high-capacity LLM for reliable (i.e., trustworthy) reasoning. To address this, we introduce R2-KG, a plug-and-play, dual-agent framework that separates reasoning into two roles: an Operator (a low-capacity LLM) that gathers evidence and a Supervisor (a high-capacity LLM) that makes final judgments. This design is cost-efficient for LLM inference while still maintaining strong reasoning accuracy. Additionally, R2-KG employs an Abstention mechanism, generating answers only when sufficient evidence is collected from KG, which significantly enhances reliability. Experiments across five diverse benchmarks show that R2-KG consistently outperforms baselines in both accuracy and reliability, regardless of the inherent capability of LLMs used as the Operator. Further experiments reveal that the single-agent version of R2-KG, equipped with a strict self-consistency strategy, achieves significantly higher-than-baseline reliability with reduced inference cost but increased abstention rate in complex KGs. Our findings establish R2-KG as a flexible and cost-effective solution for KG-based reasoning, reducing reliance on high-capacity LLMs while ensuring trustworthy inference. The code is available at https://github.com/ekrxjwh2009/R2-KG/.
>
---
#### [replaced 033] Scaling Stick-Breaking Attention: An Efficient Implementation and In-depth Study
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.17980v2](http://arxiv.org/pdf/2410.17980v2)**

> **作者:** Shawn Tan; Songlin Yang; Aaron Courville; Rameswar Panda; Yikang Shen
>
> **摘要:** The self-attention mechanism traditionally relies on the softmax operator, necessitating positional embeddings like RoPE, or position biases to account for token order. But current methods using still face length generalisation challenges. We investigate an alternative attention mechanism based on the stick-breaking process in larger scale settings. The method works as follows: For each token before the current, we determine a break point, which represents the proportion of the stick, the weight of the attention, to allocate to the current token. We repeat this on the remaining stick, until all tokens are allocated a weight, resulting in a sequence of attention weights. This process naturally incorporates recency bias, which has linguistic motivations for grammar parsing. We study the implications of replacing the conventional softmax-based attention mechanism with stick-breaking attention. We then discuss implementation of numerically stable stick-breaking attention and adapt Flash Attention to accommodate this mechanism. When used as a drop-in replacement for current softmax+RoPE attention systems, we find that stick-breaking attention performs competitively with current methods on length generalisation and downstream tasks. Stick-breaking also performs well at length generalisation, allowing a model trained with $2^{11}$ context window to perform well at $2^{14}$ with perplexity improvements.
>
---
#### [replaced 034] SQLong: Enhanced NL2SQL for Longer Contexts with LLMs
- **分类: cs.CL; cs.AI; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2502.16747v2](http://arxiv.org/pdf/2502.16747v2)**

> **作者:** Dai Quoc Nguyen; Cong Duy Vu Hoang; Duy Vu; Gioacchino Tangari; Thanh Tien Vu; Don Dharmasiri; Yuan-Fang Li; Long Duong
>
> **备注:** Accepted to Table Representation Learning Workshop at ACL 2025
>
> **摘要:** Open-weight large language models (LLMs) have significantly advanced performance in the Natural Language to SQL (NL2SQL) task. However, their effectiveness diminishes when dealing with large database schemas, as the context length increases. To address this limitation, we present SQLong, a novel and efficient data augmentation framework designed to enhance LLM performance in long-context scenarios for the NL2SQL task. SQLong generates augmented datasets by extending existing database schemas with additional synthetic CREATE TABLE commands and corresponding data rows, sampled from diverse schemas in the training data. This approach effectively simulates long-context scenarios during finetuning and evaluation. Through experiments on the Spider and BIRD datasets, we demonstrate that LLMs finetuned with SQLong-augmented data significantly outperform those trained on standard datasets. These imply SQLong's practical implementation and its impact on improving NL2SQL capabilities in real-world settings with complex database schemas.
>
---
#### [replaced 035] Assumed Identities: Quantifying Gender Bias in Machine Translation of Gender-Ambiguous Occupational Terms
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.04372v2](http://arxiv.org/pdf/2503.04372v2)**

> **作者:** Orfeas Menis Mastromichalakis; Giorgos Filandrianos; Maria Symeonaki; Giorgos Stamou
>
> **摘要:** Machine Translation (MT) systems frequently encounter gender-ambiguous occupational terms, where they must assign gender without explicit contextual cues. While individual translations in such cases may not be inherently biased, systematic patterns-such as consistently translating certain professions with specific genders-can emerge, reflecting and perpetuating societal stereotypes. This ambiguity challenges traditional instance-level single-answer evaluation approaches, as no single gold standard translation exists. To address this, we introduce GRAPE, a probability-based metric designed to evaluate gender bias by analyzing aggregated model responses. Alongside this, we present GAMBIT-MT, a benchmarking dataset in English with gender-ambiguous occupational terms. Using GRAPE, we evaluate several MT systems and examine whether their gendered translations in Greek and French align with or diverge from societal stereotypes, real-world occupational gender distributions, and normative standards.
>
---
#### [replaced 036] Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.12654v2](http://arxiv.org/pdf/2505.12654v2)**

> **作者:** Yuxin Lin; Yinglin Zheng; Ming Zeng; Wangzheng Shi
>
> **备注:** Accepected by ACL 2025
>
> **摘要:** This paper addresses the gap in predicting turn-taking and backchannel actions in human-machine conversations using multi-modal signals (linguistic, acoustic, and visual). To overcome the limitation of existing datasets, we propose an automatic data collection pipeline that allows us to collect and annotate over 210 hours of human conversation videos. From this, we construct a Multi-Modal Face-to-Face (MM-F2F) human conversation dataset, including over 1.5M words and corresponding turn-taking and backchannel annotations from approximately 20M frames. Additionally, we present an end-to-end framework that predicts the probability of turn-taking and backchannel actions from multi-modal signals. The proposed model emphasizes the interrelation between modalities and supports any combination of text, audio, and video inputs, making it adaptable to a variety of realistic scenarios. Our experiments show that our approach achieves state-of-the-art performance on turn-taking and backchannel prediction tasks, achieving a 10% increase in F1-score on turn-taking and a 33% increase on backchannel prediction. Our dataset and code are publicly available online to ease of subsequent research.
>
---
#### [replaced 037] Training Language Models to Reason Efficiently
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.04463v3](http://arxiv.org/pdf/2502.04463v3)**

> **作者:** Daman Arora; Andrea Zanette
>
> **摘要:** Scaling model size and training data has led to great advances in the performance of Large Language Models (LLMs). However, the diminishing returns of this approach necessitate alternative methods to improve model capabilities, particularly in tasks requiring advanced reasoning. Large reasoning models, which leverage long chain-of-thoughts, bring unprecedented breakthroughs in problem-solving capabilities but at a substantial deployment cost associated to longer generations. Reducing inference costs is crucial for the economic feasibility, user experience, and environmental sustainability of these models. In this work, we propose to train large reasoning models to reason efficiently. More precisely, we use reinforcement learning (RL) to train reasoning models to dynamically allocate inference-time compute based on task complexity. Our method incentivizes models to minimize unnecessary computational overhead while maintaining accuracy, thereby achieving substantial efficiency gains. It enables the derivation of a family of reasoning models with varying efficiency levels, controlled via a single hyperparameter. Experiments on two open-weight large reasoning models demonstrate significant reductions in inference cost while preserving most of the accuracy.
>
---
#### [replaced 038] ProcessBench: Identifying Process Errors in Mathematical Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.06559v3](http://arxiv.org/pdf/2412.06559v3)**

> **作者:** Chujie Zheng; Zhenru Zhang; Beichen Zhang; Runji Lin; Keming Lu; Bowen Yu; Dayiheng Liu; Jingren Zhou; Junyang Lin
>
> **备注:** ACL 2025
>
> **摘要:** As language models regularly make mistakes when solving math problems, automated identification of errors in the reasoning process becomes increasingly significant for their scalable oversight. In this paper, we introduce ProcessBench for measuring the ability to identify erroneous steps in mathematical reasoning. It consists of 3,400 test cases, primarily focused on competition- and Olympiad-level math problems. Each test case contains a step-by-step solution with error location annotated by human experts. Models are required to identify the earliest step that contains an error, or conclude that all steps are correct. We conduct extensive evaluation on ProcessBench, involving two types of models: process reward models (PRMs) and critic models, where for the latter we prompt general language models to critique each solution step by step. We draw two main observations: (1) Existing PRMs typically fail to generalize to more challenging math problems beyond GSM8K and MATH. They underperform both critic models (i.e., prompted general language models) and our own trained PRM that is straightforwardly fine-tuned on the PRM800K dataset. (2) The best open-source model, QwQ-32B-Preview, has demonstrated the critique capability competitive with the proprietary model GPT-4o, despite that it still lags behind the reasoning-specialized o1-mini. We hope ProcessBench can foster future research in reasoning process assessment, paving the way toward scalable oversight of language models.
>
---
#### [replaced 039] S1-Bench: A Simple Benchmark for Evaluating System 1 Thinking Capability of Large Reasoning Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.10368v2](http://arxiv.org/pdf/2504.10368v2)**

> **作者:** Wenyuan Zhang; Shuaiyi Nie; Xinghua Zhang; Zefeng Zhang; Tingwen Liu
>
> **备注:** 31 pages, 9 figures, 16 tables
>
> **摘要:** We introduce S1-Bench, a novel benchmark designed to evaluate the performance of Large Reasoning Models (LRMs) on simple tasks that favor intuitive system 1 thinking rather than deliberative system 2 reasoning. While LRMs have achieved significant breakthroughs in complex reasoning tasks through explicit chains of thought, their heavy reliance on system 2 thinking may limit their system 1 thinking capabilities. However, there is a lack of an appropriate benchmark for evaluating LRM's system 1 thinking capabilities. To fill this gap, S1-Bench introduces a suite of simple, diverse, and natural questions across multiple domains and languages, specifically designed to assess LRMs' performance on questions more suitable for system 1 . We conduct extensive evaluations across 28 LRMs, revealing their inefficiency, inadequate accuracy, and limited robustness when handling simple questions. Additionally, we observe a gap between their difficulty perception and generation length. Overall, this work paves the way toward dual-system compatibility in the development of LRMs.
>
---
#### [replaced 040] Evaluation and Facilitation of Online Discussions in the LLM Era: A Survey
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01513v2](http://arxiv.org/pdf/2503.01513v2)**

> **作者:** Katerina Korre; Dimitris Tsirmpas; Nikos Gkoumas; Emma Cabalé; Danai Myrtzani; Theodoros Evgeniou; Ion Androutsopoulos; John Pavlopoulos
>
> **摘要:** We present a survey of methods for assessing and enhancing the quality of online discussions, focusing on the potential of LLMs. While online discourses aim, at least in theory, to foster mutual understanding, they often devolve into harmful exchanges, such as hate speech, threatening social cohesion and democratic values. Recent advancements in LLMs enable artificial facilitation agents to not only moderate content, but also actively improve the quality of interactions. Our survey synthesizes ideas from NLP and Social Sciences to provide (a) a new taxonomy on discussion quality evaluation, (b) an overview of intervention and facilitation strategies, (c) along with a new taxonomy of conversation facilitation datasets, (d) an LLM-oriented roadmap of good practices and future research directions, from technological and societal perspectives.
>
---
#### [replaced 041] Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.14315v2](http://arxiv.org/pdf/2501.14315v2)**

> **作者:** Chao-Chung Wu; Zhi Rui Tam; Chieh-Yen Lin; Yun-Nung Chen; Shao-Hua Sun; Hung-yi Lee
>
> **摘要:** Maintaining consistent model performance across domains is a fundamental challenge in machine learning. While recent work has explored using LLM-generated data for fine-tuning, its impact on cross-domain generalization remains poorly understood. This paper presents a systematic analysis revealing that fine-tuning with LLM-generated data not only improves target task performance but also reduces non-target task degradation compared to fine-tuning with ground truth data. Through analyzing the data sequence in tasks of various domains, we demonstrate that this enhancement of non-target task robustness stems from the reduction of high perplexity tokens found in LLM-generated sequences. Following our findings, we showed that masking high perplexity tokens in ground truth training data achieves similar non-target task performance preservation, comparable to using LLM-generated data. Extensive experiments across different model families and scales, including Gemma 2 IT 2B, Llama 3 8B Instruct, and 3 additional models, agree with our findings. To the best of our knowledge, this is the first work to provide an empirical explanation based on token perplexity reduction to mitigate catastrophic forgetting in LLMs after fine-tuning, offering valuable insights for developing more robust fine-tuning strategies.
>
---
#### [replaced 042] MCiteBench: A Multimodal Benchmark for Generating Text with Citations
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2503.02589v3](http://arxiv.org/pdf/2503.02589v3)**

> **作者:** Caiyu Hu; Yikai Zhang; Tinghui Zhu; Yiwei Ye; Yanghua Xiao
>
> **备注:** https://caiyuhu.github.io/MCiteBench/
>
> **摘要:** Multimodal Large Language Models (MLLMs) have advanced in integrating diverse modalities but frequently suffer from hallucination. A promising solution to mitigate this issue is to generate text with citations, providing a transparent chain for verification. However, existing work primarily focuses on generating citations for text-only content, leaving the challenges of multimodal scenarios largely unexplored. In this paper, we introduce MCiteBench, the first benchmark designed to assess the ability of MLLMs to generate text with citations in multimodal contexts. Our benchmark comprises data derived from academic papers and review-rebuttal interactions, featuring diverse information sources and multimodal content. Experimental results reveal that MLLMs struggle to ground their outputs reliably when handling multimodal input. Further analysis uncovers a systematic modality bias and reveals how models internally rely on different sources when generating citations, offering insights into model behavior and guiding future directions for multimodal citation tasks.
>
---
#### [replaced 043] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05831v3](http://arxiv.org/pdf/2504.05831v3)**

> **作者:** Mingye Zhu; Yi Liu; Zheren Fu; Yongdong Zhang; Zhendong Mao
>
> **摘要:** Preference alignment methods are increasingly critical for steering large language models (LLMs) to generate outputs consistent with human values. While recent approaches often rely on synthetic data generated by LLMs for scalability and cost-efficiency reasons, this reliance can introduce distribution shifts that undermine the nuanced representation of human preferences needed for desirable outputs. In this paper, we propose a novel distribution-aware optimization framework that improves preference alignment despite such shifts. Our approach first leverages well-learned classifiers to assign a calibration value to each training sample, quantifying its alignment with the target human-preferred distribution. These values are then incorporated into a robust optimization objective that minimizes the worst-case loss over regions of the data space most relevant to human preferences. By explicitly focusing optimization on the target distribution, our approach mitigates the impact of distributional mismatch and improves the generation of responses that better reflect intended values.
>
---
#### [replaced 044] Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More
- **分类: cs.LG; cs.AI; cs.CL; I.2.7; I.2.8; I.5.0**

- **链接: [http://arxiv.org/pdf/2503.10542v2](http://arxiv.org/pdf/2503.10542v2)**

> **作者:** Arvid Frydenlund
>
> **备注:** ACL 2025 Main. A camera-ready version will follow in a few weeks. A reduced version of this work has was also accepted to the Workshop on Spurious Correlation and Shortcut Learning: Foundations and Solutions (SCSL) at ICLR 2025
>
> **摘要:** This work concerns the path-star task, a minimal example of searching over a graph. The graph, $G$, is star-shaped with $D$ arms radiating from a start node, $s$. A language model (LM) is given $G$, $s$, and a target node $t$, which ends one of the arms and is tasked with generating the arm containing $t$. The minimal nature of this task means only a single choice needs to be made: which of the $D$ arms contains $t$? Decoder-only LMs fail to solve this elementary task above $1/D$ chance due to a learned shortcut that absorbs training supervision. We show how this pathology is caused by excess supervision and we present a series of solutions demonstrating that the task is solvable via decoder-only LMs. We find that the task's minimal nature causes its difficulty, as it prevents task decomposition. Our solutions provide insight into the pathology and its implications for LMs trained via next-token prediction.
>
---
#### [replaced 045] ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Drafting
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.06582v2](http://arxiv.org/pdf/2501.06582v2)**

> **作者:** Steven H. Wang; Maksim Zubkov; Kexin Fan; Sarah Harrell; Yuyang Sun; Wei Chen; Andreas Plesner; Roger Wattenhofer
>
> **备注:** Accepted to ACL 2025. See the project page at https://www.atticusprojectai.org/acord
>
> **摘要:** Information retrieval, specifically contract clause retrieval, is foundational to contract drafting because lawyers rarely draft contracts from scratch; instead, they locate and revise the most relevant precedent. We introduce the Atticus Clause Retrieval Dataset (ACORD), the first retrieval benchmark for contract drafting fully annotated by experts. ACORD focuses on complex contract clauses such as Limitation of Liability, Indemnification, Change of Control, and Most Favored Nation. It includes 114 queries and over 126,000 query-clause pairs, each ranked on a scale from 1 to 5 stars. The task is to find the most relevant precedent clauses to a query. The bi-encoder retriever paired with pointwise LLMs re-rankers shows promising results. However, substantial improvements are still needed to effectively manage the complex legal work typically undertaken by lawyers. As the first retrieval benchmark for contract drafting annotated by experts, ACORD can serve as a valuable IR benchmark for the NLP community.
>
---
#### [replaced 046] Technical Report: Quantifying and Analyzing the Generalization Power of a DNN
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.06993v2](http://arxiv.org/pdf/2505.06993v2)**

> **作者:** Yuxuan He; Junpeng Zhang; Lei Cheng; Hongyuan Zhang; Quanshi Zhang
>
> **摘要:** This paper proposes a new perspective for analyzing the generalization power of deep neural networks (DNNs), i.e., directly disentangling and analyzing the dynamics of generalizable and non-generalizable interaction encoded by a DNN through the training process. Specifically, this work builds upon the recent theoretical achievement in explainble AI, which proves that the detailed inference logic of DNNs can be can be strictly rewritten as a small number of AND-OR interaction patterns. Based on this, we propose an efficient method to quantify the generalization power of each interaction, and we discover a distinct three-phase dynamics of the generalization power of interactions during training. In particular, the early phase of training typically removes noisy and non-generalizable interactions and learns simple and generalizable ones. The second and the third phases tend to capture increasingly complex interactions that are harder to generalize. Experimental results verify that the learning of non-generalizable interactions is the the direct cause for the gap between the training and testing losses.
>
---
#### [replaced 047] From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios
- **分类: cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.02145v2](http://arxiv.org/pdf/2502.02145v2)**

> **作者:** Yuan Gao; Mattia Piccinini; Korbinian Moller; Johannes Betz
>
> **摘要:** Ensuring the safety of autonomous vehicles requires virtual scenario-based testing, which depends on the robust evaluation and generation of safety-critical scenarios. So far, researchers have used scenario-based testing frameworks that rely heavily on handcrafted scenarios as safety metrics. To reduce the effort of human interpretation and overcome the limited scalability of these approaches, we combine Large Language Models (LLMs) with structured scenario parsing and prompt engineering to automatically evaluate and generate safety-critical driving scenarios. We introduce Cartesian and Ego-centric prompt strategies for scenario evaluation, and an adversarial generation module that modifies trajectories of risk-inducing vehicles (ego-attackers) to create critical scenarios. We validate our approach using a 2D simulation framework and multiple pre-trained LLMs. The results show that the evaluation module effectively detects collision scenarios and infers scenario safety. Meanwhile, the new generation module identifies high-risk agents and synthesizes realistic, safety-critical scenarios. We conclude that an LLM equipped with domain-informed prompting techniques can effectively evaluate and generate safety-critical driving scenarios, reducing dependence on handcrafted metrics. We release our open-source code and scenarios at: https://github.com/TUM-AVS/From-Words-to-Collisions.
>
---
#### [replaced 048] CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.21074v2](http://arxiv.org/pdf/2502.21074v2)**

> **作者:** Zhenyi Shen; Hanqi Yan; Linhai Zhang; Zhanghao Hu; Yali Du; Yulan He
>
> **备注:** 16 pages
>
> **摘要:** Chain-of-Thought (CoT) reasoning enhances Large Language Models (LLMs) by encouraging step-by-step reasoning in natural language. However, leveraging a latent continuous space for reasoning may offer benefits in terms of both efficiency and robustness. Prior implicit CoT methods attempt to bypass language completely by reasoning in continuous space but have consistently underperformed compared to the standard explicit CoT approach. We introduce CODI (Continuous Chain-of-Thought via Self-Distillation), a novel training framework that effectively compresses natural language CoT into continuous space. CODI jointly trains a teacher task (Explicit CoT) and a student task (Implicit CoT), distilling the reasoning ability from language into continuous space by aligning the hidden states of a designated token. Our experiments show that CODI is the first implicit CoT approach to match the performance of explicit CoT on GSM8k at the GPT-2 scale, achieving a 3.1x compression rate and outperforming the previous state-of-the-art by 28.2% in accuracy. CODI also demonstrates robustness, generalizable to complex datasets, and interpretability. These results validate that LLMs can reason effectively not only in natural language, but also in a latent continuous space.
>
---
#### [replaced 049] Fairshare Data Pricing via Data Valuation for Large Language Models
- **分类: cs.GT; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.00198v2](http://arxiv.org/pdf/2502.00198v2)**

> **作者:** Luyang Zhang; Cathy Jiao; Beibei Li; Chenyan Xiong
>
> **摘要:** Training data is the backbone of large language models (LLMs), yet today's data markets often operate under exploitative pricing -- sourcing data from marginalized groups with little pay or recognition. This paper introduces a theoretical framework for LLM data markets, modeling the strategic interactions between buyers (LLM builders) and sellers (human annotators). We begin with theoretical and empirical analysis showing how exploitative pricing drives high-quality sellers out of the market, degrading data quality and long-term model performance. Then we introduce fairshare, a pricing mechanism grounded in data valuation that quantifies each data's contribution. It aligns incentives by sustaining seller participation and optimizing utility for both buyers and sellers. Theoretically, we show that fairshare yields mutually optimal outcomes: maximizing long-term buyer utility and seller profit while sustaining market participation. Empirically when training open-source LLMs on complex NLP tasks, including math problems, medical diagnosis, and physical reasoning, fairshare boosts seller earnings and ensures a stable supply of high-quality data, while improving buyers' performance-per-dollar and long-term welfare. Our findings offer a concrete path toward fair, transparent, and economically sustainable data markets for LLM. Our code will be open sourced.
>
---
#### [replaced 050] LongDPO: Unlock Better Long-form Generation Abilities for LLMs via Critique-augmented Stepwise Information
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.02095v2](http://arxiv.org/pdf/2502.02095v2)**

> **作者:** Bowen Ping; Jiali Zeng; Fandong Meng; Shuo Wang; Jie Zhou; Shanghang Zhang
>
> **备注:** ACL 2025
>
> **摘要:** Long-form generation is crucial for academic writing papers and repo-level code generation. Despite this, current models, including GPT-4o, still exhibit unsatisfactory performance. Existing methods that utilize preference learning with outcome supervision often fail to provide detailed feedback for extended contexts. This shortcoming can lead to content that does not fully satisfy query requirements, resulting in issues like length deviations, and diminished quality. In this paper, we propose enhancing long-form generation by incorporating process supervision. We employ Monte Carlo Tree Search to gather stepwise preference pairs, utilizing a global memory pool to maintain consistency. To address the issue of suboptimal candidate selection, we integrate external critiques to refine and improve the quality of the preference pairs. Finally, we apply step-level DPO using the collected stepwise preference pairs. Experimental results show that our method improves length and quality on long-form generation benchmarks, with almost lossless performance on general benchmarks across various model backbones.
>
---
#### [replaced 051] Multi2: Multi-Agent Test-Time Scalable Framework for Multi-Document Processing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20592v3](http://arxiv.org/pdf/2502.20592v3)**

> **作者:** Juntai Cao; Xiang Zhang; Raymond Li; Chuyuan Li; Chenyu You; Shafiq Joty; Giuseppe Carenini
>
> **摘要:** Recent advances in test-time scaling have shown promising results in improving Large Language Model (LLM) performance through strategic computation allocation during inference. While this approach has demonstrated strong improvements in logical and mathematical reasoning tasks, its application to natural language generation (NLG), particularly summarization, remains unexplored. Multi-Document Summarization (MDS), a fundamental task in NLG, presents unique challenges by requiring models to extract and synthesize essential information across multiple lengthy documents. Unlike reasoning tasks, MDS demands a more nuanced approach to prompt design and ensemble methods, as no single "best" prompt can satisfy diverse summarization requirements. We propose a novel framework leveraging test-time scaling for MDS. Our approach employs prompt ensemble techniques to generate multiple candidate summaries using various prompts, then combines them with an aggregator to produce a refined summary. To evaluate our method effectively, we also introduce two new LLM-based metrics: the Consistency-Aware Preference (CAP) score and LLM Atom-Content-Unit (LLM-ACU) score, which assess summary quality while addressing the positional bias inherent in traditional automatic evaluation. Our extensive experiments demonstrate that this framework significantly enhances summary quality while also revealing the practical scaling boundaries to MDS tasks.
>
---
#### [replaced 052] MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11051v3](http://arxiv.org/pdf/2502.11051v3)**

> **作者:** Jiahao Huo; Yibo Yan; Xu Zheng; Yuanhuiyi Lyu; Xin Zou; Zhihua Wei; Xuming Hu
>
> **备注:** Accepted as ACL 2025 Findings
>
> **摘要:** Recent progress in Machine Unlearning (MU) has introduced solutions for the selective removal of private or sensitive information encoded within deep neural networks. Nonetheless, MU for Multimodal Large Language Models (MLLMs) remains in its nascent phase. Therefore, we propose to reformulate the task of multimodal MU in the era of MLLMs, which aims to erase only the visual patterns associated with a given entity while preserving the corresponding textual knowledge encoded within the original parameters of the language model backbone. Furthermore, we develop a novel geometry-constrained gradient ascent method MMUnlearner. It updates the weights of MLLMs with a weight saliency map jointly restricted by the remaining concepts and textual knowledge during unlearning, thereby preserving parameters essential for non-target knowledge. Extensive experiments demonstrate that MMUnlearner surpasses baselines that finetuning MLLMs with VQA data directly through Gradient Ascent (GA) or Negative Preference Optimization (NPO), across all evaluation dimensions. Our code will be released upon acceptance.
>
---
#### [replaced 053] Frozen Large Language Models Can Perceive Paralinguistic Aspects of Speech
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2410.01162v2](http://arxiv.org/pdf/2410.01162v2)**

> **作者:** Wonjune Kang; Junteng Jia; Chunyang Wu; Wei Zhou; Egor Lakomkin; Yashesh Gaur; Leda Sari; Suyoun Kim; Ke Li; Jay Mahadeokar; Ozlem Kalinli
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** This work studies the capabilities of a large language model (LLM) to understand paralinguistic aspects of speech without fine-tuning its weights. We utilize an end-to-end system with a speech encoder, which is trained to produce token embeddings such that the LLM's response to an expressive speech prompt is aligned with its response to a semantically matching text prompt that has also been conditioned on the user's speaking style. This framework enables the encoder to generate tokens that capture both linguistic and paralinguistic information and effectively convey them to the LLM, even when the LLM's weights remain completely frozen. To the best of our knowledge, our work is the first to explore how to induce a frozen LLM to understand more than just linguistic content from speech inputs in a general interaction setting. Experiments demonstrate that our system is able to produce higher quality and more empathetic responses to expressive speech prompts compared to several baselines.
>
---
#### [replaced 054] DAPO: An Open-Source LLM Reinforcement Learning System at Scale
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.14476v2](http://arxiv.org/pdf/2503.14476v2)**

> **作者:** Qiying Yu; Zheng Zhang; Ruofei Zhu; Yufeng Yuan; Xiaochen Zuo; Yu Yue; Weinan Dai; Tiantian Fan; Gaohong Liu; Lingjun Liu; Xin Liu; Haibin Lin; Zhiqi Lin; Bole Ma; Guangming Sheng; Yuxuan Tong; Chi Zhang; Mofan Zhang; Wang Zhang; Hang Zhu; Jinhua Zhu; Jiaze Chen; Jiangjie Chen; Chengyi Wang; Hongli Yu; Yuxuan Song; Xiangpeng Wei; Hao Zhou; Jingjing Liu; Wei-Ying Ma; Ya-Qin Zhang; Lin Yan; Mu Qiao; Yonghui Wu; Mingxuan Wang
>
> **备注:** Project Page: https://dapo-sia.github.io/
>
> **摘要:** Inference scaling empowers LLMs with unprecedented reasoning ability, with reinforcement learning as the core technique to elicit complex reasoning. However, key technical details of state-of-the-art reasoning LLMs are concealed (such as in OpenAI o1 blog and DeepSeek R1 technical report), thus the community still struggles to reproduce their RL training results. We propose the $\textbf{D}$ecoupled Clip and $\textbf{D}$ynamic s$\textbf{A}$mpling $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{DAPO}$) algorithm, and fully open-source a state-of-the-art large-scale RL system that achieves 50 points on AIME 2024 using Qwen2.5-32B base model. Unlike previous works that withhold training details, we introduce four key techniques of our algorithm that make large-scale LLM RL a success. In addition, we open-source our training code, which is built on the verl framework, along with a carefully curated and processed dataset. These components of our open-source system enhance reproducibility and support future research in large-scale LLM RL.
>
---
#### [replaced 055] Towards Achieving Concept Completeness for Textual Concept Bottleneck Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11100v2](http://arxiv.org/pdf/2502.11100v2)**

> **作者:** Milan Bhan; Yann Choho; Pierre Moreau; Jean-Noel Vittaut; Nicolas Chesneau; Marie-Jeanne Lesot
>
> **摘要:** Textual Concept Bottleneck Models (TBMs) are interpretable-by-design models for text classification that predict a set of salient concepts before making the final prediction. This paper proposes Complete Textual Concept Bottleneck Model (CT-CBM),a novel TCBM generator building concept labels in a fully unsupervised manner using a small language model, eliminating both the need for predefined human labeled concepts and LLM annotations. CT-CBM iteratively targets and adds important concepts in the bottleneck layer to create a complete concept basis and addresses downstream classification leakage through a parallel residual connection. CT-CBM achieves good results against competitors, offering a promising solution to enhance interpretability of NLP classifiers without sacrificing performance.
>
---
#### [replaced 056] RouterEval: A Comprehensive Benchmark for Routing LLMs to Explore Model-level Scaling Up in LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.10657v2](http://arxiv.org/pdf/2503.10657v2)**

> **作者:** Zhongzhan Huang; Guoming Ling; Yupei Lin; Yandong Chen; Shanshan Zhong; Hefeng Wu; Liang Lin
>
> **备注:** Preprint
>
> **摘要:** Routing large language models (LLMs) is a new paradigm that uses a router to recommend the best LLM from a pool of candidates for a given input. In this paper, our comprehensive analysis with more than 8,500 LLMs reveals a novel model-level scaling up phenomenon in Routing LLMs, i.e., a capable router can significantly enhance the performance of this paradigm as the number of candidates increases. This improvement can even surpass the performance of the best single model in the pool and many existing strong LLMs, confirming it a highly promising paradigm. However, the lack of comprehensive and open-source benchmarks for Routing LLMs has hindered the development of routers. In this paper, we introduce RouterEval, a benchmark tailored for router research, which includes over 200,000,000 performance records for 12 popular LLM evaluations across various areas such as commonsense reasoning, semantic understanding, etc., based on over 8,500 various LLMs. Using RouterEval, extensive evaluations of existing Routing LLM methods reveal that most still have significant room for improvement. See https://github.com/MilkThink-Lab/RouterEval for all data, code and tutorial.
>
---
#### [replaced 057] Model Merging in Pre-training of Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.12082v2](http://arxiv.org/pdf/2505.12082v2)**

> **作者:** Yunshui Li; Yiyuan Ma; Shen Yan; Chaoyi Zhang; Jing Liu; Jianqiao Lu; Ziwen Xu; Mengzhao Chen; Minrui Wang; Shiyi Zhan; Jin Ma; Xunhao Lai; Yao Luo; Xingyan Bin; Hongbin Ren; Mingji Han; Wenhao Hao; Bairen Yi; LingJun Liu; Bole Ma; Xiaoying Jia; Zhou Xun; Siyuan Qiao; Liang Xiang; Yonghui Wu
>
> **摘要:** Model merging has emerged as a promising technique for enhancing large language models, though its application in large-scale pre-training remains relatively unexplored. In this paper, we present a comprehensive investigation of model merging techniques during the pre-training process. Through extensive experiments with both dense and Mixture-of-Experts (MoE) architectures ranging from millions to over 100 billion parameters, we demonstrate that merging checkpoints trained with constant learning rates not only achieves significant performance improvements but also enables accurate prediction of annealing behavior. These improvements lead to both more efficient model development and significantly lower training costs. Our detailed ablation studies on merging strategies and hyperparameters provide new insights into the underlying mechanisms while uncovering novel applications. Through comprehensive experimental analysis, we offer the open-source community practical pre-training guidelines for effective model merging.
>
---
#### [replaced 058] CARMA: Enhanced Compositionality in LLMs via Advanced Regularisation and Mutual Information Alignment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11066v2](http://arxiv.org/pdf/2502.11066v2)**

> **作者:** Nura Aljaafari; Danilo S. Carvalho; André Freitas
>
> **备注:** 19 pages, 8 figures, 8 tables
>
> **摘要:** Large language models (LLMs) struggle with compositional generalisation, limiting their ability to systematically combine learned components to interpret novel inputs. While architectural modifications, fine-tuning, and data augmentation improve compositionality, they often have limited adaptability, face scalability constraints, or yield diminishing returns on real data. To address this, we propose CARMA, an intervention that enhances the stability and robustness of compositional reasoning in LLMs while preserving fine-tuned performance. CARMA employs mutual information regularisation and layer-wise stability constraints to mitigate feature fragmentation, ensuring structured representations persist across and within layers. We evaluate CARMA on inverse dictionary modelling and sentiment classification, measuring its impact on semantic consistency, performance stability, and robustness to lexical perturbations. Results show that CARMA reduces the variability introduced by fine-tuning, stabilises token representations, and improves compositional reasoning. While its effectiveness varies across architectures, CARMA's key strength lies in reinforcing learned structures rather than introducing new capabilities, making it a scalable auxiliary method. These findings suggest that integrating CARMA with fine-tuning can improve compositional generalisation while maintaining task-specific performance in LLMs.
>
---
#### [replaced 059] CRCE: Coreference-Retention Concept Erasure in Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.14232v2](http://arxiv.org/pdf/2503.14232v2)**

> **作者:** Yuyang Xue; Edward Moroshko; Feng Chen; Jingyu Sun; Steven McDonagh; Sotirios A. Tsaftaris
>
> **摘要:** Text-to-Image diffusion models can produce undesirable content that necessitates concept erasure. However, existing methods struggle with under-erasure, leaving residual traces of targeted concepts, or over-erasure, mistakenly eliminating unrelated but visually similar concepts. To address these limitations, we introduce CRCE, a novel concept erasure framework that leverages Large Language Models to identify both semantically related concepts that should be erased alongside the target and distinct concepts that should be preserved. By explicitly modelling coreferential and retained concepts semantically, CRCE enables more precise concept removal, without unintended erasure. Experiments demonstrate that CRCE outperforms existing methods on diverse erasure tasks, including real-world object, person identities, and abstract intellectual property characteristics. The constructed dataset CorefConcept and the source code will be release upon acceptance.
>
---
#### [replaced 060] EssayJudge: A Multi-Granular Benchmark for Assessing Automated Essay Scoring Capabilities of Multimodal Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11916v2](http://arxiv.org/pdf/2502.11916v2)**

> **作者:** Jiamin Su; Yibo Yan; Fangteng Fu; Han Zhang; Jingheng Ye; Xiang Liu; Jiahao Huo; Huiyu Zhou; Xuming Hu
>
> **备注:** Accepted by ACL Findings 2025
>
> **摘要:** Automated Essay Scoring (AES) plays a crucial role in educational assessment by providing scalable and consistent evaluations of writing tasks. However, traditional AES systems face three major challenges: (1) reliance on handcrafted features that limit generalizability, (2) difficulty in capturing fine-grained traits like coherence and argumentation, and (3) inability to handle multimodal contexts. In the era of Multimodal Large Language Models (MLLMs), we propose EssayJudge, the first multimodal benchmark to evaluate AES capabilities across lexical-, sentence-, and discourse-level traits. By leveraging MLLMs' strengths in trait-specific scoring and multimodal context understanding, EssayJudge aims to offer precise, context-rich evaluations without manual feature engineering, addressing longstanding AES limitations. Our experiments with 18 representative MLLMs reveal gaps in AES performance compared to human evaluation, particularly in discourse-level traits, highlighting the need for further advancements in MLLM-based AES research.
>
---
#### [replaced 061] InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.12368v2](http://arxiv.org/pdf/2501.12368v2)**

> **作者:** Yuhang Zang; Xiaoyi Dong; Pan Zhang; Yuhang Cao; Ziyu Liu; Shengyuan Ding; Shenxi Wu; Yubo Ma; Haodong Duan; Wenwei Zhang; Kai Chen; Dahua Lin; Jiaqi Wang
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Despite the promising performance of Large Vision Language Models (LVLMs) in visual understanding, they occasionally generate incorrect outputs. While reward models (RMs) with reinforcement learning or test-time scaling offer the potential for improving generation quality, a critical gap remains: publicly available multi-modal RMs for LVLMs are scarce, and the implementation details of proprietary models are often unclear. We bridge this gap with InternLM-XComposer2.5-Reward (IXC-2.5-Reward), a simple yet effective multi-modal reward model that aligns LVLMs with human preferences. To ensure the robustness and versatility of IXC-2.5-Reward, we set up a high-quality multi-modal preference corpus spanning text, image, and video inputs across diverse domains, such as instruction following, general understanding, text-rich documents, mathematical reasoning, and video understanding. IXC-2.5-Reward achieves excellent results on the latest multi-modal reward model benchmark and shows competitive performance on text-only reward model benchmarks. We further demonstrate three key applications of IXC-2.5-Reward: (1) Providing a supervisory signal for RL training. We integrate IXC-2.5-Reward with Proximal Policy Optimization (PPO) yields IXC-2.5-Chat, which shows consistent improvements in instruction following and multi-modal open-ended dialogue; (2) Selecting the best response from candidate responses for test-time scaling; and (3) Filtering outlier or noisy samples from existing image and video instruction tuning training data. To ensure reproducibility and facilitate further research, we have open-sourced all model weights and training recipes at https://github.com/InternLM/InternLM-XComposer/tree/main/InternLM-XComposer-2.5-Reward
>
---
#### [replaced 062] Adaptive Thinking via Mode Policy Optimization for Social Language Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.02156v3](http://arxiv.org/pdf/2505.02156v3)**

> **作者:** Minzheng Wang; Yongbin Li; Haobo Wang; Xinghua Zhang; Nan Xu; Bingli Wu; Fei Huang; Haiyang Yu; Wenji Mao
>
> **备注:** Work in Progress. The code and data are available, see https://github.com/MozerWang/AMPO
>
> **摘要:** Effective social intelligence simulation requires language agents to dynamically adjust reasoning depth, a capability notably absent in current studies. Existing methods either lack this kind of reasoning capability or enforce Long Chain-of-Thought reasoning uniformly across all scenarios, resulting in excessive token usage and inflexible social simulation. To address this, we propose an $\textbf{A}$daptive $\textbf{M}$ode $\textbf{L}$earning ($\textbf{AML}$) framework in this paper, aiming to improve the adaptive thinking ability of language agents in dynamic social interactions. To this end, we first identify hierarchical thinking modes ranging from intuitive response to deep deliberation based on the cognitive control theory. We then develop the $\textbf{A}$daptive $\textbf{M}$ode $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{AMPO}$) algorithm to optimize the context-aware mode switching and reasoning. Our framework advances existing research in three key aspects: (1) Multi-granular thinking mode design, (2) Context-aware mode switching across social interaction, and (3) Token-efficient reasoning via depth-adaptive processing. Extensive experiments on social intelligence benchmarks verify that AML achieves 15.6% higher task performance than GPT-4o. Notably, our AMPO outperforms GRPO by 7.0% with 32.8% shorter reasoning chains, demonstrating the advantage of adaptive thinking mode selection and optimization mechanism in AMPO over GRPO's fixed-depth solution.
>
---
#### [replaced 063] Cross-Document Cross-Lingual NLI via RST-Enhanced Graph Fusion and Interpretability Prediction
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.12324v2](http://arxiv.org/pdf/2504.12324v2)**

> **作者:** Mengying Yuan; Wenhao Wang; Zixuan Wang; Yujie Huang; Kangli Wei; Fei Li; Chong Teng; Donghong Ji
>
> **摘要:** Natural Language Inference (NLI) is a fundamental task in natural language processing. While NLI has developed many sub-directions such as sentence-level NLI, document-level NLI and cross-lingual NLI, Cross-Document Cross-Lingual NLI (CDCL-NLI) remains largely unexplored. In this paper, we propose a novel paradigm: CDCL-NLI, which extends traditional NLI capabilities to multi-document, multilingual scenarios. To support this task, we construct a high-quality CDCL-NLI dataset including 25,410 instances and spanning 26 languages. To address the limitations of previous methods on CDCL-NLI task, we further propose an innovative method that integrates RST-enhanced graph fusion with interpretability-aware prediction. Our approach leverages RST (Rhetorical Structure Theory) within heterogeneous graph neural networks for cross-document context modeling, and employs a structure-aware semantic alignment based on lexical chains for cross-lingual understanding. For NLI interpretability, we develop an EDU (Elementary Discourse Unit)-level attribution framework that produces extractive explanations. Extensive experiments demonstrate our approach's superior performance, achieving significant improvements over both conventional NLI models as well as large language models. Our work sheds light on the study of NLI and will bring research interest on cross-document cross-lingual context understanding, hallucination elimination and interpretability inference. Our code and datasets are available at \href{https://anonymous.4open.science/r/CDCL-NLI-637E/}{CDCL-NLI-link} for peer review.
>
---
#### [replaced 064] Plant in Cupboard, Orange on Rably, Inat Aphone. Benchmarking Incremental Learning of Situation and Language Model using a Text-Simulated Situated Environment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11733v2](http://arxiv.org/pdf/2502.11733v2)**

> **作者:** Jonathan Jordan; Sherzod Hakimov; David Schlangen
>
> **摘要:** Large Language Models (LLMs) serve not only as chatbots but as key components in agent systems, where their common-sense knowledge significantly impacts performance as language-based planners for situated or embodied action. We assess LLMs' incremental learning (based on feedback from the environment), and controlled in-context learning abilities using a text-based environment. We introduce challenging yet interesting set of experiments to test i) how agents can incrementally solve tasks related to every day objects in typical rooms in a house where each of them are discovered by interacting within the environment, ii) controlled in-context learning abilities and efficiency of agents by providing short info about locations of objects and rooms to check how faster the task can be solved, and finally iii) using synthetic pseudo-English words to gauge how well LLMs are at inferring meaning of unknown words from environmental feedback. Results show that larger commercial models have a substantial gap in performance compared to open-weight but almost all models struggle with the synthetic words experiments.
>
---
#### [replaced 065] VideoVista-CulturalLingo: 360$^\circ$ Horizons-Bridging Cultures, Languages, and Domains in Video Comprehension
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.17821v2](http://arxiv.org/pdf/2504.17821v2)**

> **作者:** Xinyu Chen; Yunxin Li; Haoyuan Shi; Baotian Hu; Wenhan Luo; Yaowei Wang; Min Zhang
>
> **摘要:** Assessing the video comprehension capabilities of multimodal AI systems can effectively measure their understanding and reasoning abilities. Most video evaluation benchmarks are limited to a single language, typically English, and predominantly feature videos rooted in Western cultural contexts. In this paper, we present VideoVista-CulturalLingo, the first video evaluation benchmark designed to bridge cultural, linguistic, and domain divide in video comprehension. Our work differs from existing benchmarks in the following ways: 1) Cultural diversity, incorporating cultures from China, North America, and Europe; 2) Multi-linguistics, with questions presented in Chinese and English-two of the most widely spoken languages; and 3) Broad domain, featuring videos sourced from hundreds of human-created domains. VideoVista-CulturalLingo contains 1,389 videos and 3,134 QA pairs, and we have evaluated 24 recent open-source or proprietary video large models. From the experiment results, we observe that: 1) Existing models perform worse on Chinese-centric questions than Western-centric ones, particularly those related to Chinese history; 2) Current open-source models still exhibit limitations in temporal understanding, especially in the Event Localization task, achieving a maximum score of only 45.2%; 3) Mainstream models demonstrate strong performance in general scientific questions, while open-source models demonstrate weak performance in mathematics.
>
---
#### [replaced 066] FineFilter: A Fine-grained Noise Filtering Mechanism for Retrieval-Augmented Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11811v3](http://arxiv.org/pdf/2502.11811v3)**

> **作者:** Qianchi Zhang; Hainan Zhang; Liang Pang; Ziwei Wang; Hongwei Zheng; Yongxin Tong; Zhiming Zheng
>
> **备注:** 18 pages, 4 figures, 18 tables, under review
>
> **摘要:** Retrieved documents containing noise will hinder Retrieval-Augmented Generation (RAG) from detecting answer clues, necessitating noise filtering mechanisms to enhance accuracy. Existing methods use reranking or summarization to identify the most relevant sentences, but directly and accurately locating answer clues from these large-scale and complex documents remains challenging. Unlike these document-level operations, we treat noise filtering as a sentence-level MinMax optimization problem: first identifying potential clues from multiple documents, then ranking them by relevance, and finally retaining the minimum number of clues through truncation. In this paper, we propose FineFilter, a novel fine-grained noise filtering mechanism for RAG, consisting of a clue extractor, a reranker, and a truncator. We optimize each module to tackle complex reasoning challenges: (1) The clue extractor first uses sentences containing the answer and similar ones as fine-tuning targets, aiming to extract sufficient potential clues; (2) The reranker is trained to prioritize effective clues based on the real feedback from the generation module, with clues capable of generating correct answers as positive samples and others as negative; (3) The truncator takes the minimum number of clues needed to answer the question (truncation point) as fine-tuning targets, and performs truncation on the reranked clues to achieve fine-grained noise filtering. Experiments on three QA datasets demonstrate that FineFilter significantly improves QA performance over baselines on both LLaMA3 and Mistral. Further analysis confirms its effectiveness in complex reasoning, robustness to unreliable retrieval, and generalization to different scenarios.
>
---
#### [replaced 067] ReEx-SQL: Reasoning with Execution-Aware Reinforcement Learning for Text-to-SQL
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12768v2](http://arxiv.org/pdf/2505.12768v2)**

> **作者:** Yaxun Dai; Wenxuan Xie; Xialie Zhuang; Tianyu Yang; Yiying Yang; Haiqin Yang; Yuhang Zhao; Pingfu Chao; Wenhao Jiang
>
> **摘要:** In Text-to-SQL, execution feedback is essential for guiding large language models (LLMs) to reason accurately and generate reliable SQL queries. However, existing methods treat execution feedback solely as a post-hoc signal for correction or selection, failing to integrate it into the generation process. This limitation hinders their ability to address reasoning errors as they occur, ultimately reducing query accuracy and robustness. To address this issue, we propose ReEx-SQL (Reasoning with Execution-Aware Reinforcement Learning), a framework for Text-to-SQL that enables models to interact with the database during decoding and dynamically adjust their reasoning based on execution feedback. ReEx-SQL introduces an execution-aware reasoning paradigm that interleaves intermediate SQL execution into reasoning paths, facilitating context-sensitive revisions. It achieves this through structured prompts with markup tags and a stepwise rollout strategy that integrates execution feedback into each stage of generation. To supervise policy learning, we develop a composite reward function that includes an exploration reward, explicitly encouraging effective database interaction. Additionally, ReEx-SQL adopts a tree-based decoding strategy to support exploratory reasoning, enabling dynamic expansion of alternative reasoning paths. Notably, ReEx-SQL achieves 88.8% on Spider and 64.9% on BIRD at the 7B scale, surpassing the standard reasoning baseline by 2.7% and 2.6%, respectively. It also shows robustness, achieving 85.2% on Spider-Realistic with leading performance. In addition, its tree-structured decoding improves efficiency and performance over linear decoding, reducing inference time by 51.9% on the BIRD development set.
>
---
#### [replaced 068] MedCaseReasoning: Evaluating and learning diagnostic reasoning from clinical case reports
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11733v2](http://arxiv.org/pdf/2505.11733v2)**

> **作者:** Kevin Wu; Eric Wu; Rahul Thapa; Kevin Wei; Angela Zhang; Arvind Suresh; Jacqueline J. Tao; Min Woo Sun; Alejandro Lozano; James Zou
>
> **摘要:** Doctors and patients alike increasingly use Large Language Models (LLMs) to diagnose clinical cases. However, unlike domains such as math or coding, where correctness can be objectively defined by the final answer, medical diagnosis requires both the outcome and the reasoning process to be accurate. Currently, widely used medical benchmarks like MedQA and MMLU assess only accuracy in the final answer, overlooking the quality and faithfulness of the clinical reasoning process. To address this limitation, we introduce MedCaseReasoning, the first open-access dataset for evaluating LLMs on their ability to align with clinician-authored diagnostic reasoning. The dataset includes 14,489 diagnostic question-and-answer cases, each paired with detailed reasoning statements derived from open-access medical case reports. We evaluate state-of-the-art reasoning LLMs on MedCaseReasoning and find significant shortcomings in their diagnoses and reasoning: for instance, the top-performing open-source model, DeepSeek-R1, achieves only 48% 10-shot diagnostic accuracy and mentions only 64% of the clinician reasoning statements (recall). However, we demonstrate that fine-tuning LLMs on the reasoning traces derived from MedCaseReasoning significantly improves diagnostic accuracy and clinical reasoning recall by an average relative gain of 29% and 41%, respectively. The open-source dataset, code, and models are available at https://github.com/kevinwu23/Stanford-MedCaseReasoning.
>
---
#### [replaced 069] People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.15654v2](http://arxiv.org/pdf/2501.15654v2)**

> **作者:** Jenna Russell; Marzena Karpinska; Mohit Iyyer
>
> **备注:** ACL 2025 33 pages
>
> **摘要:** In this paper, we study how well humans can detect text generated by commercial LLMs (GPT-4o, Claude, o1). We hire annotators to read 300 non-fiction English articles, label them as either human-written or AI-generated, and provide paragraph-length explanations for their decisions. Our experiments show that annotators who frequently use LLMs for writing tasks excel at detecting AI-generated text, even without any specialized training or feedback. In fact, the majority vote among five such "expert" annotators misclassifies only 1 of 300 articles, significantly outperforming most commercial and open-source detectors we evaluated even in the presence of evasion tactics like paraphrasing and humanization. Qualitative analysis of the experts' free-form explanations shows that while they rely heavily on specific lexical clues ('AI vocabulary'), they also pick up on more complex phenomena within the text (e.g., formality, originality, clarity) that are challenging to assess for automatic detectors. We release our annotated dataset and code to spur future research into both human and automated detection of AI-generated text.
>
---
#### [replaced 070] Erasing Without Remembering: Implicit Knowledge Forgetting in Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19982v2](http://arxiv.org/pdf/2502.19982v2)**

> **作者:** Huazheng Wang; Yongcheng Jing; Haifeng Sun; Yingjie Wang; Jingyu Wang; Jianxin Liao; Dacheng Tao
>
> **摘要:** In this paper, we investigate knowledge forgetting in large language models with a focus on its generalisation--ensuring that models forget not only specific training samples but also related implicit knowledge. To this end, we begin by identifying a broader unlearning scope that includes both target data and logically associated samples, including rephrased, subject-replaced, one-hop reasoned, and relation-reversed data. To rigorously evaluate generalisation, we introduce UGBench, the first comprehensive benchmark specifically designed to assess the unlearning of in-scope implicit knowledge covering 13 state-of-the-art methods across three datasets. UGBench reveals that unlearned models can still recall paraphrased answers and retain target facts in intermediate layers. This motivates us to take a preliminary step toward more generalised implicit knowledge forgetting by proposing PerMU, a novel probability perturbation-based unlearning paradigm. PerMU simulates adversarial unlearning samples to eliminate fact-related tokens from the logit distribution, collectively reducing the probabilities of all answer-associated tokens. Experiments are conducted on a diverse range of datasets, including TOFU, Harry Potter, ZsRE, WMDP, and MUSE, using models ranging from 1.3B to 13B in scale. The results demonstrate that PerMU delivers up to a 50.40% improvement in unlearning vanilla target data while maintaining a 40.73% boost in forgetting implicit knowledge. Our code can be found in https://github.com/MaybeLizzy/UGBench.
>
---
#### [replaced 071] Can LLMs be Good Graph Judge for Knowledge Graph Construction?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.17388v3](http://arxiv.org/pdf/2411.17388v3)**

> **作者:** Haoyu Huang; Chong Chen; Zeang Sheng; Yang Li; Wentao Zhang
>
> **摘要:** In real-world scenarios, most of the data obtained from the information retrieval (IR) system is unstructured. Converting natural language sentences into structured Knowledge Graphs (KGs) remains a critical challenge. We identified three limitations with respect to existing KG construction methods: (1) There could be a large amount of noise in real-world documents, which could result in extracting messy information. (2) Naive LLMs usually extract inaccurate knowledge from some domain-specific documents. (3) Hallucination phenomenon cannot be overlooked when directly using LLMs to construct KGs. In this paper, we propose \textbf{GraphJudge}, a KG construction framework to address the aforementioned challenges. In this framework, we designed an entity-centric strategy to eliminate the noise information in the documents. And we fine-tuned a LLM as a graph judge to finally enhance the quality of generated KGs. Experiments conducted on two general and one domain-specific text-graph pair datasets demonstrate state-of-the-art performance against various baseline methods with strong generalization abilities. Our code is available at \href{https://github.com/hhy-huang/GraphJudge}{https://github.com/hhy-huang/GraphJudge}.
>
---
#### [replaced 072] Agent-SafetyBench: Evaluating the Safety of LLM Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.14470v2](http://arxiv.org/pdf/2412.14470v2)**

> **作者:** Zhexin Zhang; Shiyao Cui; Yida Lu; Jingzhuo Zhou; Junxiao Yang; Hongning Wang; Minlie Huang
>
> **备注:** 26 pages
>
> **摘要:** As large language models (LLMs) are increasingly deployed as agents, their integration into interactive environments and tool use introduce new safety challenges beyond those associated with the models themselves. However, the absence of comprehensive benchmarks for evaluating agent safety presents a significant barrier to effective assessment and further improvement. In this paper, we introduce Agent-SafetyBench, a comprehensive benchmark designed to evaluate the safety of LLM agents. Agent-SafetyBench encompasses 349 interaction environments and 2,000 test cases, evaluating 8 categories of safety risks and covering 10 common failure modes frequently encountered in unsafe interactions. Our evaluation of 16 popular LLM agents reveals a concerning result: none of the agents achieves a safety score above 60%. This highlights significant safety challenges in LLM agents and underscores the considerable need for improvement. Through failure mode and helpfulness analysis, we summarize two fundamental safety defects in current LLM agents: lack of robustness and lack of risk awareness. Furthermore, our findings suggest that reliance on defense prompts alone may be insufficient to address these safety issues, emphasizing the need for more advanced and robust strategies. To drive progress in this area, Agent-SafetyBench has been released at https://github.com/thu-coai/Agent-SafetyBench/ to facilitate further research in agent safety evaluation and improvement.
>
---
#### [replaced 073] Can Prompting LLMs Unlock Hate Speech Detection across Languages? A Zero-shot and Few-shot Study
- **分类: cs.CL; cs.CY; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.06149v2](http://arxiv.org/pdf/2505.06149v2)**

> **作者:** Faeze Ghorbanpour; Daryna Dementieva; Alexander Fraser
>
> **摘要:** Despite growing interest in automated hate speech detection, most existing approaches overlook the linguistic diversity of online content. Multilingual instruction-tuned large language models such as LLaMA, Aya, Qwen, and BloomZ offer promising capabilities across languages, but their effectiveness in identifying hate speech through zero-shot and few-shot prompting remains underexplored. This work evaluates LLM prompting-based detection across eight non-English languages, utilizing several prompting techniques and comparing them to fine-tuned encoder models. We show that while zero-shot and few-shot prompting lag behind fine-tuned encoder models on most of the real-world evaluation sets, they achieve better generalization on functional tests for hate speech detection. Our study also reveals that prompt design plays a critical role, with each language often requiring customized prompting techniques to maximize performance.
>
---
#### [replaced 074] TiEBe: Tracking Language Model Recall of Notable Worldwide Events Through Time
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.07482v2](http://arxiv.org/pdf/2501.07482v2)**

> **作者:** Thales Sales Almeida; Giovana Kerche Bonás; João Guilherme Alves Santos; Hugo Abonizio; Rodrigo Nogueira
>
> **摘要:** As the knowledge landscape evolves and large language models (LLMs) become increasingly widespread, there is a growing need to keep these models updated with current events. While existing benchmarks assess general factual recall, few studies explore how LLMs retain knowledge over time or across different regions. To address these gaps, we present the Timely Events Benchmark (TiEBe), a dataset of over 23,000 question-answer pairs centered on notable global and regional events, spanning more than 10 years of events, 23 regions, and 13 languages. TiEBe leverages structured retrospective data from Wikipedia to identify notable events through time. These events are then used to construct a benchmark to evaluate LLMs' understanding of global and regional developments, grounded in factual evidence beyond Wikipedia itself. Our results reveal significant geographic disparities in factual recall, emphasizing the need for more balanced global representation in LLM training. We also observe a Pearson correlation of more than 0.7 between models' performance in TiEBe and various countries' socioeconomic indicators, such as HDI. In addition, we examine the impact of language on factual recall by posing questions in the native language of the region where each event occurred, uncovering substantial performance gaps for low-resource languages.
>
---
#### [replaced 075] Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14171v5](http://arxiv.org/pdf/2502.14171v5)**

> **作者:** Mehdi Jafari; Devin Yuncheng Hua; Hao Xue; Flora Salim
>
> **备注:** Accepted to Findings of ACL 2025
>
> **摘要:** Natural language interaction with agentic Artificial Intelligence (AI), driven by Large Language Models (LLMs), is expected to remain a dominant paradigm in the near future. While humans instinctively align their communication with mental states -- an ability known as Theory of Mind (ToM), current LLM powered systems exhibit significant limitations in this regard. This study examines the extent to which open source language models (LLaMA) can capture and preserve ToM related information and how effectively it contributes to consistent ToM reasoning in generated responses. We further investigate whether explicit manipulation of ToM related components, such as beliefs, desires, and intentions, can enhance response alignment. Experiments on two LLaMA 3 variants demonstrate that incorporating ToM informed alignment improves response quality, achieving win rates of 67 and 63 percent for the 3B and 8B models, respectively. These findings highlight the potential of ToM driven strategies to improve alignment in LLM based conversational agents.
>
---
#### [replaced 076] What if Deception Cannot be Detected? A Cross-Linguistic Study on the Limits of Deception Detection from Text
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13147v2](http://arxiv.org/pdf/2505.13147v2)**

> **作者:** Aswathy Velutharambath; Kai Sassenberg; Roman Klinger
>
> **摘要:** Can deception be detected solely from written text? Cues of deceptive communication are inherently subtle, even more so in text-only communication. Yet, prior studies have reported considerable success in automatic deception detection. We hypothesize that such findings are largely driven by artifacts introduced during data collection and do not generalize beyond specific datasets. We revisit this assumption by introducing a belief-based deception framework, which defines deception as a misalignment between an author's claims and true beliefs, irrespective of factual accuracy, allowing deception cues to be studied in isolation. Based on this framework, we construct three corpora, collectively referred to as DeFaBel, including a German-language corpus of deceptive and non-deceptive arguments and a multilingual version in German and English, each collected under varying conditions to account for belief change and enable cross-linguistic analysis. Using these corpora, we evaluate commonly reported linguistic cues of deception. Across all three DeFaBel variants, these cues show negligible, statistically insignificant correlations with deception labels, contrary to prior work that treats such cues as reliable indicators. We further benchmark against other English deception datasets following similar data collection protocols. While some show statistically significant correlations, effect sizes remain low and, critically, the set of predictive cues is inconsistent across datasets. We also evaluate deception detection using feature-based models, pretrained language models, and instruction-tuned large language models. While some models perform well on established deception datasets, they consistently perform near chance on DeFaBel. Our findings challenge the assumption that deception can be reliably inferred from linguistic cues and call for rethinking how deception is studied and modeled in NLP.
>
---
#### [replaced 077] Interpreting token compositionality in LLMs: A robustness analysis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12924v3](http://arxiv.org/pdf/2410.12924v3)**

> **作者:** Nura Aljaafari; Danilo S. Carvalho; André Freitas
>
> **备注:** 23 pages, 3 Figures, 14 tables
>
> **摘要:** Understanding the internal mechanisms of large language models (LLMs) is integral to enhancing their reliability, interpretability, and inference processes. We present Constituent-Aware Pooling (CAP), a methodology designed to analyse how LLMs process compositional linguistic structures. Grounded in principles of compositionality, mechanistic interpretability, and information theory, CAP systematically intervenes in model activations through constituent-based pooling at various model levels. Our experiments on inverse definition modelling, hypernym and synonym prediction reveal critical insights into transformers' limitations in handling compositional abstractions. No specific layer integrates tokens into unified semantic representations based on their constituent parts. We observe fragmented information processing, which intensifies with model size, suggesting that larger models struggle more with these interventions and exhibit greater information dispersion. This fragmentation likely stems from transformers' training objectives and architectural design, preventing systematic and cohesive representations. Our findings highlight fundamental limitations in current transformer architectures regarding compositional semantics processing and model interpretability, underscoring the critical need for novel approaches in LLM design to address these challenges.
>
---
#### [replaced 078] MathAgent: Leveraging a Mixture-of-Math-Agent Framework for Real-World Multimodal Mathematical Error Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.18132v2](http://arxiv.org/pdf/2503.18132v2)**

> **作者:** Yibo Yan; Shen Wang; Jiahao Huo; Philip S. Yu; Xuming Hu; Qingsong Wen
>
> **备注:** Accepted by The 63rd Annual Meeting of the Association for Computational Linguistics (ACL Industry 2025, Oral Presentation)
>
> **摘要:** Mathematical error detection in educational settings presents a significant challenge for Multimodal Large Language Models (MLLMs), requiring a sophisticated understanding of both visual and textual mathematical content along with complex reasoning capabilities. Though effective in mathematical problem-solving, MLLMs often struggle with the nuanced task of identifying and categorizing student errors in multimodal mathematical contexts. Therefore, we introduce MathAgent, a novel Mixture-of-Math-Agent framework designed specifically to address these challenges. Our approach decomposes error detection into three phases, each handled by a specialized agent: an image-text consistency validator, a visual semantic interpreter, and an integrative error analyzer. This architecture enables more accurate processing of mathematical content by explicitly modeling relationships between multimodal problems and student solution steps. We evaluate MathAgent on real-world educational data, demonstrating approximately 5% higher accuracy in error step identification and 3% improvement in error categorization compared to baseline models. Besides, MathAgent has been successfully deployed in an educational platform that has served over one million K-12 students, achieving nearly 90% student satisfaction while generating significant cost savings by reducing manual error detection.
>
---
#### [replaced 079] MrGuard: A Multilingual Reasoning Guardrail for Universal LLM Safety
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.15241v2](http://arxiv.org/pdf/2504.15241v2)**

> **作者:** Yahan Yang; Soham Dan; Shuo Li; Dan Roth; Insup Lee
>
> **备注:** Preprint
>
> **摘要:** Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual settings, where multilingual safety-aligned data is often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we introduce a multilingual guardrail with reasoning for prompt classification. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-based Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail, MrGuard, consistently outperforms recent baselines across both in-domain and out-of-domain languages by more than 15%. We also evaluate MrGuard's robustness to multilingual variations, such as code-switching and low-resource language distractors in the prompt, and demonstrate that it preserves safety judgments under these challenging conditions. The multilingual reasoning capability of our guardrail enables it to generate explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation.
>
---
#### [replaced 080] Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.02789v2](http://arxiv.org/pdf/2502.02789v2)**

> **作者:** Jingyu Liu; Beidi Chen; Ce Zhang
>
> **备注:** Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)
>
> **摘要:** Improving time-to-first-token (TTFT) is an essentially important objective in modern large language model (LLM) inference engines. Optimizing TTFT directly results in higher maximal QPS and meets the requirements of many critical applications. However, boosting TTFT is notoriously challenging since it is compute-bounded and the performance bottleneck shifts from the self-attention that many prior works focus on to the MLP part. In this work, we present SpecPrefill, a training free framework that accelerates the inference TTFT for both long and medium context queries based on the following insight: LLMs are generalized enough to preserve the quality given only a carefully chosen subset of prompt tokens. At its core, SpecPrefill leverages a lightweight model to speculate locally important tokens based on the context. These tokens, along with the necessary positional information, are then sent to the main model for processing. We evaluate SpecPrefill with a diverse set of tasks, followed by a comprehensive benchmarking of performance improvement both in a real end-to-end setting and ablation studies. SpecPrefill manages to serve Llama-3.1-405B-Instruct-FP8 with up to 7$\times$ maximal end-to-end QPS on real downstream tasks and 7.66$\times$ TTFT improvement.
>
---
#### [replaced 081] Efficiently Building a Domain-Specific Large Language Model from Scratch: A Case Study of a Classical Chinese Large Language Model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11810v2](http://arxiv.org/pdf/2505.11810v2)**

> **作者:** Shen Li; Renfen Hu; Lijun Wang
>
> **摘要:** General-purpose large language models demonstrate notable capabilities in language comprehension and generation, achieving results that are comparable to, or even surpass, human performance in many natural language processing tasks. Nevertheless, when general models are applied to some specific domains, e.g., Classical Chinese texts, their effectiveness is often unsatisfactory, and fine-tuning open-source foundational models similarly struggles to adequately incorporate domain-specific knowledge. To address this challenge, this study developed a large language model, AI Taiyan, specifically designed for understanding and generating Classical Chinese. Experiments show that with a reasonable model design, data processing, foundational training, and fine-tuning, satisfactory results can be achieved with only 1.8 billion parameters. In key tasks related to language processing of Classical Chinese such as punctuation, identification of allusions, explanation of word meanings, and translation between ancient and modern Chinese, this model exhibits a clear advantage over both general-purpose large models and domain-specific traditional models, achieving levels close to or surpassing human baselines. This research provides a reference for the efficient construction of specialized domain-specific large language models. Furthermore, the paper discusses the application of this model in fields such as the collation of ancient texts, dictionary editing, and language research, combined with case studies.
>
---
#### [replaced 082] Scaling Video-Language Models to 10K Frames via Hierarchical Differential Distillation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.02438v4](http://arxiv.org/pdf/2504.02438v4)**

> **作者:** Chuanqi Cheng; Jian Guan; Wei Wu; Rui Yan
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Long-form video processing fundamentally challenges vision-language models (VLMs) due to the high computational costs of handling extended temporal sequences. Existing token pruning and feature merging methods often sacrifice critical temporal dependencies or dilute semantic information. We introduce differential distillation, a principled approach that systematically preserves task-relevant information while suppressing redundancy. Based on this principle, we develop ViLAMP, a hierarchical video-language model that processes hour-long videos at "mixed precision" through two key mechanisms: (1) differential keyframe selection that maximizes query relevance while maintaining temporal distinctiveness at the frame level and (2) differential feature merging that preserves query-salient features in non-keyframes at the patch level. Hence, ViLAMP retains full information in keyframes while reducing non-keyframes to their most salient features, resembling mixed-precision training. Extensive experiments demonstrate ViLAMP's superior performance across five video understanding benchmarks, particularly on long-form content. Notably, ViLAMP can process ultra-long videos (up to 10K frames) on a single NVIDIA A100 GPU, achieving substantial computational efficiency while maintaining state-of-the-art performance. Code and model are available at https://github.com/steven-ccq/ViLAMP.
>
---
#### [replaced 083] A Survey on Large Language Model based Human-Agent Systems
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.00753v2](http://arxiv.org/pdf/2505.00753v2)**

> **作者:** Henry Peng Zou; Wei-Chieh Huang; Yaozu Wu; Yankai Chen; Chunyu Miao; Hoang Nguyen; Yue Zhou; Weizhi Zhang; Liancheng Fang; Langzhou He; Yangning Li; Dongyuan Li; Renhe Jiang; Xue Liu; Philip S. Yu
>
> **备注:** Paper lists and resources are available at https://github.com/HenryPengZou/Awesome-LLM-Based-Human-Agent-Systems
>
> **摘要:** Recent advances in large language models (LLMs) have sparked growing interest in building fully autonomous agents. However, fully autonomous LLM-based agents still face significant challenges, including limited reliability due to hallucinations, difficulty in handling complex tasks, and substantial safety and ethical risks, all of which limit their feasibility and trustworthiness in real-world applications. To overcome these limitations, LLM-based human-agent systems (LLM-HAS) incorporate human-provided information, feedback, or control into the agent system to enhance system performance, reliability and safety. This paper provides the first comprehensive and structured survey of LLM-HAS. It clarifies fundamental concepts, systematically presents core components shaping these systems, including environment & profiling, human feedback, interaction types, orchestration and communication, explores emerging applications, and discusses unique challenges and opportunities. By consolidating current knowledge and offering a structured overview, we aim to foster further research and innovation in this rapidly evolving interdisciplinary field. Paper lists and resources are available at https://github.com/HenryPengZou/Awesome-LLM-Based-Human-Agent-Systems.
>
---
#### [replaced 084] From Theft to Bomb-Making: The Ripple Effect of Unlearning in Defending Against Jailbreak Attacks
- **分类: cs.CR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.02855v3](http://arxiv.org/pdf/2407.02855v3)**

> **作者:** Zhexin Zhang; Junxiao Yang; Yida Lu; Pei Ke; Shiyao Cui; Chujie Zheng; Hongning Wang; Minlie Huang
>
> **备注:** 19 pages
>
> **摘要:** Large Language Models (LLMs) are known to be vulnerable to jailbreak attacks. An important observation is that, while different types of jailbreak attacks can generate significantly different queries, they mostly result in similar responses that are rooted in the same harmful knowledge (e.g., detailed steps to make a bomb). Consequently, unlearning-based approaches have been proposed to mitigate jailbreak attacks by directly removing harmful knowledge from the model. In this paper, we identify a novel ripple effect of unlearning, wherein LLMs can implicitly unlearn harmful knowledge that was not explicitly introduced during the unlearning phase (e.g., a model unlearning the steps for theft may also implicitly unlearn the steps for making a bomb). Through over 100 experimental runs spanning multiple models, attack strategies, and defense methods, we empirically validate this phenomenon, which makes unlearning-based methods able to decrease the Attack Success Rate on unseen data from more than 70% to less than 10% with only 100 training samples. Further analysis reveals that the strong generalization ability of unlearning may stem from the intrinsic relatedness among harmful responses across harmful questions (e.g., response patterns, shared steps and actions in response, and similarity among their learned representations in the LLM). We also discuss the potential limitations of unlearning and the observed ripple effect. We hope our research could contribute to a deeper understanding of unlearning. Our code is available at https://github.com/thu-coai/SafeUnlearning.
>
---
#### [replaced 085] Artificial Intelligence Bias on English Language Learners in Automatic Scoring
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2505.10643v2](http://arxiv.org/pdf/2505.10643v2)**

> **作者:** Shuchen Guo; Yun Wang; Jichao Yu; Xuansheng Wu; Bilgehan Ayik; Field M. Watts; Ehsan Latif; Ninghao Liu; Lei Liu; Xiaoming Zhai
>
> **摘要:** This study investigated potential scoring biases and disparities toward English Language Learners (ELLs) when using automatic scoring systems for middle school students' written responses to science assessments. We specifically focus on examining how unbalanced training data with ELLs contributes to scoring bias and disparities. We fine-tuned BERT with four datasets: responses from (1) ELLs, (2) non-ELLs, (3) a mixed dataset reflecting the real-world proportion of ELLs and non-ELLs (unbalanced), and (4) a balanced mixed dataset with equal representation of both groups. The study analyzed 21 assessment items: 10 items with about 30,000 ELL responses, five items with about 1,000 ELL responses, and six items with about 200 ELL responses. Scoring accuracy (Acc) was calculated and compared to identify bias using Friedman tests. We measured the Mean Score Gaps (MSGs) between ELLs and non-ELLs and then calculated the differences in MSGs generated through both the human and AI models to identify the scoring disparities. We found that no AI bias and distorted disparities between ELLs and non-ELLs were found when the training dataset was large enough (ELL = 30,000 and ELL = 1,000), but concerns could exist if the sample size is limited (ELL = 200).
>
---
#### [replaced 086] Learning to Reason under Off-Policy Guidance
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14945v3](http://arxiv.org/pdf/2504.14945v3)**

> **作者:** Jianhao Yan; Yafu Li; Zican Hu; Zhi Wang; Ganqu Cui; Xiaoye Qu; Yu Cheng; Yue Zhang
>
> **备注:** Work in progress
>
> **摘要:** Recent advances in large reasoning models (LRMs) demonstrate that sophisticated behaviors such as multi-step reasoning and self-reflection can emerge via reinforcement learning with verifiable rewards~(\textit{RLVR}). However, existing \textit{RLVR} approaches are inherently ``on-policy'', limiting learning to a model's own outputs and failing to acquire reasoning abilities beyond its initial capabilities. To address this issue, we introduce \textbf{LUFFY} (\textbf{L}earning to reason \textbf{U}nder o\textbf{FF}-polic\textbf{Y} guidance), a framework that augments \textit{RLVR} with off-policy reasoning traces. LUFFY dynamically balances imitation and exploration by combining off-policy demonstrations with on-policy rollouts during training. Specifically, LUFFY combines the Mixed-Policy GRPO framework, which has a theoretically guaranteed convergence rate, alongside policy shaping via regularized importance sampling to avoid superficial and rigid imitation during mixed-policy training. Compared with previous RLVR methods, LUFFY achieves an over \textbf{+6.4} average gain across six math benchmarks and an advantage of over \textbf{+6.2} points in out-of-distribution tasks. Most significantly, we show that LUFFY successfully trains weak models in scenarios where on-policy RLVR completely fails. These results provide compelling evidence that LUFFY transcends the fundamental limitations of on-policy RLVR and demonstrates the great potential of utilizing off-policy guidance in RLVR.
>
---
#### [replaced 087] DiffSampling: Enhancing Diversity and Accuracy in Neural Text Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.14037v2](http://arxiv.org/pdf/2502.14037v2)**

> **作者:** Giorgio Franceschelli; Mirco Musolesi
>
> **摘要:** Despite their growing capabilities, language models still frequently reproduce content from their training data, generate repetitive text, and favor common grammatical patterns and vocabulary. A possible cause is the decoding strategy: the most common strategies either consider only the most probable tokens, which reduces output diversity, or increase the likelihood of unlikely tokens, compromising output accuracy and correctness. In this paper, we propose three new decoding methods that leverage a mathematical analysis of the token probability distribution to ensure the generation of contextually appropriate text. In particular, the difference between consecutive, sorted probabilities can be used to truncate incorrect tokens. Experiments concerning math problem solving, extreme summarization, and the divergent association task demonstrate that our approach consistently performs at least as well as existing methods in terms of quality and diversity.
>
---
#### [replaced 088] VLMs as GeoGuessr Masters: Exceptional Performance, Hidden Biases, and Privacy Risks
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11163v2](http://arxiv.org/pdf/2502.11163v2)**

> **作者:** Jingyuan Huang; Jen-tse Huang; Ziyi Liu; Xiaoyuan Liu; Wenxuan Wang; Jieyu Zhao
>
> **备注:** 8 pages of main text; 5 pages of appendix
>
> **摘要:** Visual-Language Models (VLMs) have shown remarkable performance across various tasks, particularly in recognizing geographic information from images. However, VLMs still show regional biases in this task. To systematically evaluate these issues, we introduce a benchmark consisting of 1,200 images paired with detailed geographic metadata. Evaluating four VLMs, we find that while these models demonstrate the ability to recognize geographic information from images, achieving up to 53.8% accuracy in city prediction, they exhibit significant biases. Specifically, performance is substantially higher for economically developed and densely populated regions compared to less developed (-12.5%) and sparsely populated (-17.0%) areas. Moreover, regional biases of frequently over-predicting certain locations remain. For instance, they consistently predict Sydney for images taken in Australia, shown by the low entropy scores for these countries. The strong performance of VLMs also raises privacy concerns, particularly for users who share images online without the intent of being identified. Our code and dataset are publicly available at https://github.com/uscnlp-lime/FairLocator.
>
---
#### [replaced 089] PersonaGym: Evaluating Persona Agents and LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.18416v4](http://arxiv.org/pdf/2407.18416v4)**

> **作者:** Vinay Samuel; Henry Peng Zou; Yue Zhou; Shreyas Chaudhari; Ashwin Kalyan; Tanmay Rajpurohit; Ameet Deshpande; Karthik Narasimhan; Vishvak Murahari
>
> **备注:** 21 pages, 5 figures
>
> **摘要:** Persona agents, which are LLM agents conditioned to act according to an assigned persona, enable contextually rich and user aligned interactions across domains like education and healthcare. However, evaluating how faithfully these agents adhere to their personas remains a significant challenge, particularly in free-form settings that demand consistency across diverse, persona-relevant environments. We introduce PersonaGym, the first dynamic evaluation framework for persona agents, and PersonaScore, a human-aligned automatic metric grounded in decision theory that enables comprehensive large-scale evaluation. Our evaluation of 10 leading LLMs across 200 personas and 10,000 questions reveals significant advancement opportunities. For example, GPT-4.1 had the exact same PersonaScore as LLaMA-3-8b despite being a more recent and advanced closed source model. Importantly, increased model size and complexity do not necessarily enhance persona agent capabilities, underscoring the need for algorithmic and architectural innovation toward faithful, performant persona agents.
>
---
#### [replaced 090] Rank, Chunk and Expand: Lineage-Oriented Reasoning for Taxonomy Expansion
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13282v2](http://arxiv.org/pdf/2505.13282v2)**

> **作者:** Sahil Mishra; Kumar Arjun; Tanmoy Chakraborty
>
> **备注:** Accepted in ACL'25 Findings
>
> **摘要:** Taxonomies are hierarchical knowledge graphs crucial for recommendation systems, and web applications. As data grows, expanding taxonomies is essential, but existing methods face key challenges: (1) discriminative models struggle with representation limits and generalization, while (2) generative methods either process all candidates at once, introducing noise and exceeding context limits, or discard relevant entities by selecting noisy candidates. We propose LORex ($\textbf{L}$ineage-$\textbf{O}$riented $\textbf{Re}$asoning for Taxonomy E$\textbf{x}$pansion), a plug-and-play framework that combines discriminative ranking and generative reasoning for efficient taxonomy expansion. Unlike prior methods, LORex ranks and chunks candidate terms into batches, filtering noise and iteratively refining selections by reasoning candidates' hierarchy to ensure contextual efficiency. Extensive experiments across four benchmarks and twelve baselines show that LORex improves accuracy by 12% and Wu & Palmer similarity by 5% over state-of-the-art methods.
>
---
#### [replaced 091] Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.02009v2](http://arxiv.org/pdf/2501.02009v2)**

> **作者:** Youcheng Huang; Chen Huang; Duanyu Feng; Wenqiang Lei; Jiancheng Lv
>
> **备注:** ACL 2025 Main Camera Ready
>
> **摘要:** Understanding the inner workings of Large Language Models (LLMs) is a critical research frontier. Prior research has shown that a single LLM's concept representations can be captured as steering vectors (SVs), enabling the control of LLM behavior (e.g., towards generating harmful content). Our work takes a novel approach by exploring the intricate relationships between concept representations across different LLMs, drawing an intriguing parallel to Plato's Allegory of the Cave. In particular, we introduce a linear transformation method to bridge these representations and present three key findings: 1) Concept representations across different LLMs can be effectively aligned using simple linear transformations, enabling efficient cross-model transfer and behavioral control via SVs. 2) This linear transformation generalizes across concepts, facilitating alignment and control of SVs representing different concepts across LLMs. 3) A weak-to-strong transferability exists between LLM concept representations, whereby SVs extracted from smaller LLMs can effectively control the behavior of larger LLMs.
>
---
#### [replaced 092] HyPerAlign: Interpretable Personalized LLM Alignment via Hypothesis Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.00038v2](http://arxiv.org/pdf/2505.00038v2)**

> **作者:** Cristina Garbacea; Chenhao Tan
>
> **摘要:** Alignment algorithms are widely used to align large language models (LLMs) to human users based on preference annotations. Typically these (often divergent) preferences are aggregated over a diverse set of users, resulting in fine-tuned models that are aligned to the ``average-user'' preference. Nevertheless, current models are used by individual users in very specific contexts and situations, emphasizing the need for user-dependent preference control. In this work we address the problem of personalizing LLM outputs to their users. We aim to generate customized responses tailored to specific individuals instead of generic outputs that emulate the collective voices of diverse populations. We propose HyPerAlign, an interpretable and sample-efficient hypothesis-driven personalization approach for LLM models. Given few-shot examples written by a particular user, we first infer hypotheses about their communication strategies, personality, and writing style, then prompt LLM models with these hypotheses and user-specific attributes to generate customized outputs. We conduct experiments on two different personalization tasks, namely authorship attribution and deliberative alignment, with datasets from diverse domains (news articles, blog posts, emails, jailbreaking benchmarks). Results demonstrate the superiority of hypothesis-driven LLM personalization compared to preference-based fine-tuning methods. For authorship attribution, HyPerAlign generations have consistently high win-rates (commonly $> 90\%$) against state-of-the-art preference fine-tuning approaches across diverse user profiles and LLM models. For deliberative alignment, the helpfulness of LLM models is improved by up to $70\%$ on average. Overall, HyPerAlign represents an interpretable and sample-efficient strategy for the personalization of LLM models to individual users.
>
---
#### [replaced 093] Counterspeech the ultimate shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11958v2](http://arxiv.org/pdf/2505.11958v2)**

> **作者:** Aswini Kumar Padhi; Anil Bandhakavi; Tanmoy Chakraborty
>
> **备注:** Accepted in ACL 2025 Main Conference
>
> **摘要:** Counterspeech has proven to be a powerful tool to combat hate speech online. Previous studies have focused on generating counterspeech conditioned only on specific intents (single attributed). However, a holistic approach considering multiple attributes simultaneously can yield more nuanced and effective responses. Here, we introduce HiPPrO, Hierarchical Prefix learning with Preference Optimization, a novel two-stage framework that utilizes the effectiveness of attribute-specific prefix embedding spaces hierarchically optimized during the counterspeech generation process in the first phase. Thereafter, we incorporate both reference and reward-free preference optimization to generate more constructive counterspeech. Furthermore, we extend IntentCONANv2 by annotating all 13,973 counterspeech instances with emotion labels by five annotators. HiPPrO leverages hierarchical prefix optimization to integrate these dual attributes effectively. An extensive evaluation demonstrates that HiPPrO achieves a ~38 % improvement in intent conformity and a ~3 %, ~2 %, ~3 % improvement in Rouge-1, Rouge-2, and Rouge-L, respectively, compared to several baseline models. Human evaluations further substantiate the superiority of our approach, highlighting the enhanced relevance and appropriateness of the generated counterspeech. This work underscores the potential of multi-attribute conditioning in advancing the efficacy of counterspeech generation systems.
>
---
#### [replaced 094] J4R: Learning to Judge with Equivalent Initial State Group Relative Policy Optimization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.13346v2](http://arxiv.org/pdf/2505.13346v2)**

> **作者:** Austin Xu; Yilun Zhou; Xuan-Phi Nguyen; Caiming Xiong; Shafiq Joty
>
> **备注:** 25 pages, 4 figures, 6 tables. To be updated with links for code/benchmark
>
> **摘要:** To keep pace with the increasing pace of large language models (LLM) development, model output evaluation has transitioned away from time-consuming human evaluation to automatic evaluation, where LLMs themselves are tasked with assessing and critiquing other model outputs. LLM-as-judge models are a class of generative evaluators that excel in evaluating relatively simple domains, like chat quality, but struggle in reasoning intensive domains where model responses contain more substantive and challenging content. To remedy existing judge shortcomings, we explore training judges with reinforcement learning (RL). We make three key contributions: (1) We propose the Equivalent Initial State Group Relative Policy Optimization (EIS-GRPO) algorithm, which allows us to train our judge to be robust to positional biases that arise in more complex evaluation settings. (2) We introduce ReasoningJudgeBench, a benchmark that evaluates judges in diverse reasoning settings not covered by prior work. (3) We train Judge for Reasoning (J4R), a 7B judge trained with EIS-GRPO that outperforms GPT-4o and the next best small judge by 6.7% and 9%, matching or exceeding the performance of larger GRPO-trained judges on both JudgeBench and ReasoningJudgeBench.
>
---
#### [replaced 095] Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13061v2](http://arxiv.org/pdf/2502.13061v2)**

> **作者:** Jingbiao Mei; Jinghong Chen; Guangyu Yang; Weizhe Lin; Bill Byrne
>
> **备注:** Preprint. Under Review
>
> **摘要:** Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While LMMs have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both SFT and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability.
>
---
#### [replaced 096] Moving Beyond Medical Exam Questions: A Clinician-Annotated Dataset of Real-World Tasks and Ambiguity in Mental Healthcare
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16051v2](http://arxiv.org/pdf/2502.16051v2)**

> **作者:** Max Lamparth; Declan Grabb; Amy Franks; Scott Gershan; Kaitlyn N. Kunstman; Aaron Lulla; Monika Drummond Roots; Manu Sharma; Aryan Shrivastava; Nina Vasan; Colleen Waickman
>
> **备注:** Added minor clarifications and expanded appendices
>
> **摘要:** Current medical language model (LM) benchmarks often over-simplify the complexities of day-to-day clinical practice tasks and instead rely on evaluating LMs on multiple-choice board exam questions. Thus, we present an expert-created and annotated dataset spanning five critical domains of decision-making in mental healthcare: treatment, diagnosis, documentation, monitoring, and triage. This dataset - created without any LM assistance - is designed to capture the nuanced clinical reasoning and daily ambiguities mental health practitioners encounter, reflecting the inherent complexities of care delivery that are missing from existing datasets. Almost all 203 base questions with five answer options each have had the decision-irrelevant demographic patient information removed and replaced with variables (e.g., AGE), and are available for male, female, or non-binary-coded patients. For question categories dealing with ambiguity and multiple valid answer options, we create a preference dataset with uncertainties from the expert annotations. We outline a series of intended use cases and demonstrate the usability of our dataset by evaluating eleven off-the-shelf and four mental health fine-tuned LMs on category-specific task accuracy, on the impact of patient demographic information on decision-making, and how consistently free-form responses deviate from human annotated samples.
>
---
#### [replaced 097] RoMath: A Mathematical Reasoning Benchmark in Romanian
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.11074v3](http://arxiv.org/pdf/2409.11074v3)**

> **作者:** Adrian Cosma; Ana-Maria Bucur; Emilian Radoi
>
> **备注:** 5 Figures, 11 Tables
>
> **摘要:** Mathematics has long been conveyed through natural language, primarily for human understanding. With the rise of mechanized mathematics and proof assistants, there is a growing need to understand informal mathematical text, yet most existing benchmarks focus solely on English, overlooking other languages. This paper introduces RoMath, a Romanian mathematical reasoning benchmark suite comprising three subsets: Baccalaureate, Competitions and Synthetic, which cover a range of mathematical domains and difficulty levels, aiming to improve non-English language models and promote multilingual AI development. By focusing on Romanian, a low-resource language with unique linguistic features, RoMath addresses the limitations of Anglo-centric models and emphasizes the need for dedicated resources beyond simple automatic translation. We benchmark several open-weight language models, highlighting the importance of creating resources for underrepresented languages. Code and datasets are be made available.
>
---
#### [replaced 098] Walk the Talk? Measuring the Faithfulness of Large Language Model Explanations
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2504.14150v2](http://arxiv.org/pdf/2504.14150v2)**

> **作者:** Katie Matton; Robert Osazuwa Ness; John Guttag; Emre Kıcıman
>
> **备注:** 66 pages, 14 figures, 40 tables; ICLR 2025 (spotlight) camera ready
>
> **摘要:** Large language models (LLMs) are capable of generating plausible explanations of how they arrived at an answer to a question. However, these explanations can misrepresent the model's "reasoning" process, i.e., they can be unfaithful. This, in turn, can lead to over-trust and misuse. We introduce a new approach for measuring the faithfulness of LLM explanations. First, we provide a rigorous definition of faithfulness. Since LLM explanations mimic human explanations, they often reference high-level concepts in the input question that purportedly influenced the model. We define faithfulness in terms of the difference between the set of concepts that LLM explanations imply are influential and the set that truly are. Second, we present a novel method for estimating faithfulness that is based on: (1) using an auxiliary LLM to modify the values of concepts within model inputs to create realistic counterfactuals, and (2) using a Bayesian hierarchical model to quantify the causal effects of concepts at both the example- and dataset-level. Our experiments show that our method can be used to quantify and discover interpretable patterns of unfaithfulness. On a social bias task, we uncover cases where LLM explanations hide the influence of social bias. On a medical question answering task, we uncover cases where LLM explanations provide misleading claims about which pieces of evidence influenced the model's decisions.
>
---
#### [replaced 099] ChatNVD: Advancing Cybersecurity Vulnerability Assessment with Large Language Models
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.04756v2](http://arxiv.org/pdf/2412.04756v2)**

> **作者:** Shivansh Chopra; Hussain Ahmad; Diksha Goel; Claudia Szabo
>
> **摘要:** The increasing frequency and sophistication of cybersecurity vulnerabilities in software systems underscores the need for more robust and effective vulnerability assessment methods. However, existing approaches often rely on highly technical and abstract frameworks, which hinder understanding and increase the likelihood of exploitation, resulting in severe cyberattacks. In this paper, we introduce ChatNVD, a support tool powered by Large Language Models (LLMs) that leverages the National Vulnerability Database (NVD) to generate accessible, context-rich summaries of software vulnerabilities. We develop three variants of ChatNVD, utilizing three prominent LLMs: GPT-4o Mini by OpenAI, LLaMA 3 by Meta, and Gemini 1.5 Pro by Google. To evaluate their performance, we conduct a comparative evaluation focused on their ability to identify, interpret, and explain software vulnerabilities. Our results demonstrate that GPT-4o Mini outperforms the other models, achieving over 92% accuracy and the lowest error rates, making it the most reliable option for real-world vulnerability assessment.
>
---
#### [replaced 100] Talk to Your Slides: Language-Driven Agents for Efficient Slide Editing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11604v2](http://arxiv.org/pdf/2505.11604v2)**

> **作者:** Kyudan Jung; Hojun Cho; Jooyeol Yun; Soyoung Yang; Jaehyeok Jang; Jagul Choo
>
> **备注:** 20 pages, 14 figures
>
> **摘要:** Editing presentation slides remains one of the most common and time-consuming tasks faced by millions of users daily, despite significant advances in automated slide generation. Existing approaches have successfully demonstrated slide editing via graphic user interface (GUI)-based agents, offering intuitive visual control. However, such methods often suffer from high computational cost and latency. In this paper, we propose Talk-to-Your-Slides, an LLM-powered agent designed to edit slides %in active PowerPoint sessions by leveraging structured information about slide objects rather than relying on image modality. The key insight of our work is designing the editing process with distinct high-level and low-level layers to facilitate interaction between user commands and slide objects. By providing direct access to application objects rather than screen pixels, our system enables 34.02% faster processing, 34.76% better instruction fidelity, and 87.42% cheaper operation than baselines. To evaluate slide editing capabilities, we introduce TSBench, a human-annotated dataset comprising 379 diverse editing instructions paired with corresponding slide variations in four categories. Our code, benchmark and demos are available at https://anonymous.4open.science/r/Talk-to-Your-Slides-0F4C.
>
---
#### [replaced 101] MMDocIR: Benchmarking Multi-Modal Retrieval for Long Documents
- **分类: cs.IR; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.08828v2](http://arxiv.org/pdf/2501.08828v2)**

> **作者:** Kuicai Dong; Yujing Chang; Xin Deik Goh; Dexun Li; Ruiming Tang; Yong Liu
>
> **备注:** https://huggingface.co/MMDocIR
>
> **摘要:** Multimodal document retrieval aims to identify and retrieve various forms of multimodal content, such as figures, tables, charts, and layout information from extensive documents. Despite its increasing popularity, there is a notable lack of a comprehensive and robust benchmark to effectively evaluate the performance of systems in such tasks. To address this gap, this work introduces a new benchmark, named MMDocIR, that encompasses two distinct tasks: page-level and layout-level retrieval. The former evaluates the performance of identifying the most relevant pages within a long document, while the later assesses the ability of detecting specific layouts, providing a more fine-grained measure than whole-page analysis. A layout refers to a variety of elements, including textual paragraphs, equations, figures, tables, or charts. The MMDocIR benchmark comprises a rich dataset featuring 1,685 questions annotated by experts and 173,843 questions with bootstrapped labels, making it a valuable resource in multimodal document retrieval for both training and evaluation. Through rigorous experiments, we demonstrate that (i) visual retrievers significantly outperform their text counterparts, (ii) MMDocIR training set effectively enhances the performance of multimodal document retrieval and (iii) text retrievers leveraging VLM-text significantly outperforms retrievers relying on OCR-text. Our dataset is available at https://mmdocrag.github.io/MMDocIR/.
>
---
#### [replaced 102] Beyond Self-Reports: Multi-Observer Agents for Personality Assessment in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.08399v2](http://arxiv.org/pdf/2504.08399v2)**

> **作者:** Yin Jou Huang; Rafik Hadfi
>
> **备注:** 16 pages, 6 figures, 6 tables
>
> **摘要:** Self-report questionnaires have long been used to assess LLM personality traits, yet they fail to capture behavioral nuances due to biases and meta-knowledge contamination. This paper proposes a novel multi-observer framework for personality trait assessments in LLM agents that draws on informant-report methods in psychology. Instead of relying on self-assessments, we employ multiple observer agents. Each observer is configured with a specific relational context (e.g., family member, friend, or coworker) and engages the subject LLM in dialogue before evaluating its behavior across the Big Five dimensions. We show that these observer-report ratings align more closely with human judgments than traditional self-reports and reveal systematic biases in LLM self-assessments. We also found that aggregating responses from 5 to 7 observers reduces systematic biases and achieves optimal reliability. Our results highlight the role of relationship context in perceiving personality and demonstrate that a multi-observer paradigm offers a more reliable, context-sensitive approach to evaluating LLM personality traits.
>
---
#### [replaced 103] TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.14161v2](http://arxiv.org/pdf/2412.14161v2)**

> **作者:** Frank F. Xu; Yufan Song; Boxuan Li; Yuxuan Tang; Kritanjali Jain; Mengxue Bao; Zora Z. Wang; Xuhui Zhou; Zhitong Guo; Murong Cao; Mingyang Yang; Hao Yang Lu; Amaad Martin; Zhe Su; Leander Maben; Raj Mehta; Wayne Chi; Lawrence Jang; Yiqing Xie; Shuyan Zhou; Graham Neubig
>
> **备注:** Preprint
>
> **摘要:** We interact with computers on an everyday basis, be it in everyday life or work, and many aspects of work can be done entirely with access to a computer and the Internet. At the same time, thanks to improvements in large language models (LLMs), there has also been a rapid development in AI agents that interact with and affect change in their surrounding environments. But how performant are AI agents at accelerating or even autonomously performing work-related tasks? The answer to this question has important implications both for industry looking to adopt AI into their workflows and for economic policy to understand the effects that adoption of AI may have on the labor market. To measure the progress of these LLM agents' performance on performing real-world professional tasks, in this paper we introduce TheAgentCompany, an extensible benchmark for evaluating AI agents that interact with the world in similar ways to those of a digital worker: by browsing the Web, writing code, running programs, and communicating with other coworkers. We build a self-contained environment with internal web sites and data that mimics a small software company environment, and create a variety of tasks that may be performed by workers in such a company. We test baseline agents powered by both closed API-based and open-weights language models (LMs), and find that the most competitive agent can complete 30% of tasks autonomously. This paints a nuanced picture on task automation with LM agents--in a setting simulating a real workplace, a good portion of simpler tasks could be solved autonomously, but more difficult long-horizon tasks are still beyond the reach of current systems. We release code, data, environment, and experiments on https://the-agent-company.com.
>
---
#### [replaced 104] STATE ToxiCN: A Benchmark for Span-level Target-Aware Toxicity Extraction in Chinese Hate Speech Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.15451v3](http://arxiv.org/pdf/2501.15451v3)**

> **作者:** Zewen Bai; Shengdi Yin; Junyu Lu; Jingjie Zeng; Haohao Zhu; Yuanyuan Sun; Liang Yang; Hongfei Lin
>
> **备注:** Our paper has been accepted by ACL 2025 Findings
>
> **摘要:** The proliferation of hate speech has caused significant harm to society. The intensity and directionality of hate are closely tied to the target and argument it is associated with. However, research on hate speech detection in Chinese has lagged behind, and existing datasets lack span-level fine-grained annotations. Furthermore, the lack of research on Chinese hateful slang poses a significant challenge. In this paper, we provide a solution for fine-grained detection of Chinese hate speech. First, we construct a dataset containing Target-Argument-Hateful-Group quadruples (STATE ToxiCN), which is the first span-level Chinese hate speech dataset. Secondly, we evaluate the span-level hate speech detection performance of existing models using STATE ToxiCN. Finally, we conduct the first study on Chinese hateful slang and evaluate the ability of LLMs to detect such expressions. Our work contributes valuable resources and insights to advance span-level hate speech detection in Chinese.
>
---
#### [replaced 105] Does Acceleration Cause Hidden Instability in Vision Language Models? Uncovering Instance-Level Divergence Through a Large-Scale Empirical Study
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.06794v3](http://arxiv.org/pdf/2503.06794v3)**

> **作者:** Yizheng Sun; Hao Li; Chang Xu; Hongpeng Zhou; Chenghua Lin; Riza Batista-Navarro; Jingyuan Sun
>
> **摘要:** Vision-Language Models (VLMs) are powerful yet computationally intensive for widespread practical deployments. To address such challenge without costly re-training, post-training acceleration techniques like quantization and token reduction are extensively explored. However, current acceleration evaluations primarily target minimal overall performance degradation, overlooking a crucial question: does the accelerated model still give the same answers to the same questions as it did before acceleration? This is vital for stability-centered industrial applications where consistently correct answers for specific, known situations are paramount, such as in AI-based disease diagnosis. We systematically investigate this for accelerated VLMs, testing four leading models (LLaVA-1.5, LLaVA-Next, Qwen2-VL, Qwen2.5-VL) with eight acceleration methods on ten multi-modal benchmarks. Our findings are stark: despite minimal aggregate performance drops, accelerated models changed original answers up to 20% of the time. Critically, up to 6.5% of these changes converted correct answers to incorrect. Input perturbations magnified these inconsistencies, and the trend is confirmed by case studies with the medical VLM LLaVA-Med. This research reveals a significant oversight in VLM acceleration, stressing an urgent need for instance-level stability checks to ensure trustworthy real-world deployment.
>
---
#### [replaced 106] Char-mander Use mBackdoor! A Study of Cross-lingual Backdoor Attacks in Multilingual LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.16901v2](http://arxiv.org/pdf/2502.16901v2)**

> **作者:** Himanshu Beniwal; Sailesh Panda; Birudugadda Srivibhav; Mayank Singh
>
> **摘要:** We explore \textbf{C}ross-lingual \textbf{B}ackdoor \textbf{AT}tacks (X-BAT) in multilingual Large Language Models (mLLMs), revealing how backdoors inserted in one language can automatically transfer to others through shared embedding spaces. Using toxicity classification as a case study, we demonstrate that attackers can compromise multilingual systems by poisoning data in a single language, with rare and high-occurring tokens serving as specific, effective triggers. Our findings expose a critical vulnerability that influences the model's architecture, resulting in a concealed backdoor effect during the information flow. Our code and data are publicly available https://github.com/himanshubeniwal/X-BAT.
>
---
#### [replaced 107] Evaluating the Correctness of Inference Patterns Used by LLMs for Judgment
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.09083v2](http://arxiv.org/pdf/2410.09083v2)**

> **作者:** Lu Chen; Yuxuan Huang; Yixing Li; Dongrui Liu; Qihan Ren; Shuai Zhao; Kun Kuang; Zilong Zheng; Quanshi Zhang
>
> **摘要:** This paper presents a method to analyze the inference patterns used by Large Language Models (LLMs) for judgment in a case study on legal LLMs, so as to identify potential incorrect representations of the LLM, according to human domain knowledge. Unlike traditional evaluations on language generation results, we propose to evaluate the correctness of the detailed inference patterns of an LLM behind its seemingly correct outputs. To this end, we quantify the interactions between input phrases used by the LLM as primitive inference patterns, because recent theoretical achievements have proven several mathematical guarantees of the faithfulness of the interaction-based explanation. We design a set of metrics to evaluate the detailed inference patterns of LLMs. Experiments show that even when the language generation results appear correct, a significant portion of the inference patterns used by the LLM for the legal judgment may represent misleading or irrelevant logic.
>
---
#### [replaced 108] A Survey of Mathematical Reasoning in the Era of Multimodal Large Language Model: Benchmark, Method & Challenges
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.11936v3](http://arxiv.org/pdf/2412.11936v3)**

> **作者:** Yibo Yan; Jiamin Su; Jianxiang He; Fangteng Fu; Xu Zheng; Yuanhuiyi Lyu; Kun Wang; Shen Wang; Qingsong Wen; Xuming Hu
>
> **备注:** Accepted by The 63rd Annual Meeting of the Association for Computational Linguistics (ACL Findings 2025)
>
> **摘要:** Mathematical reasoning, a core aspect of human cognition, is vital across many domains, from educational problem-solving to scientific advancements. As artificial general intelligence (AGI) progresses, integrating large language models (LLMs) with mathematical reasoning tasks is becoming increasingly significant. This survey provides the first comprehensive analysis of mathematical reasoning in the era of multimodal large language models (MLLMs). We review over 200 studies published since 2021, and examine the state-of-the-art developments in Math-LLMs, with a focus on multimodal settings. We categorize the field into three dimensions: benchmarks, methodologies, and challenges. In particular, we explore multimodal mathematical reasoning pipeline, as well as the role of (M)LLMs and the associated methodologies. Finally, we identify five major challenges hindering the realization of AGI in this domain, offering insights into the future direction for enhancing multimodal reasoning capabilities. This survey serves as a critical resource for the research community in advancing the capabilities of LLMs to tackle complex multimodal reasoning tasks.
>
---
#### [replaced 109] DeepResonance: Enhancing Multimodal Music Understanding via Music-centric Multi-way Instruction Tuning
- **分类: cs.SD; cs.AI; cs.CL; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.12623v2](http://arxiv.org/pdf/2502.12623v2)**

> **作者:** Zhuoyuan Mao; Mengjie Zhao; Qiyu Wu; Hiromi Wakaki; Yuki Mitsufuji
>
> **摘要:** Recent advancements in music large language models (LLMs) have significantly improved music understanding tasks, which involve the model's ability to analyze and interpret various musical elements. These improvements primarily focused on integrating both music and text inputs. However, the potential of incorporating additional modalities such as images, videos and textual music features to enhance music understanding remains unexplored. To bridge this gap, we propose DeepResonance, a multimodal music understanding LLM fine-tuned via multi-way instruction tuning with multi-way aligned music, text, image, and video data. To this end, we construct Music4way-MI2T, Music4way-MV2T, and Music4way-Any2T, three 4-way training and evaluation datasets designed to enable DeepResonance to integrate both visual and textual music feature content. We also introduce multi-sampled ImageBind embeddings and a pre-LLM fusion Transformer to enhance modality fusion prior to input into text LLMs, tailoring DeepResonance for multi-way instruction tuning. Our model achieves state-of-the-art performances across six music understanding tasks, highlighting the benefits of the auxiliary modalities and the structural superiority of DeepResonance. We plan to open-source the models and the newly constructed datasets.
>
---
#### [replaced 110] EquiBench: Benchmarking Large Language Models' Understanding of Program Semantics via Equivalence Checking
- **分类: cs.LG; cs.AI; cs.CL; cs.PL; cs.SE**

- **链接: [http://arxiv.org/pdf/2502.12466v2](http://arxiv.org/pdf/2502.12466v2)**

> **作者:** Anjiang Wei; Jiannan Cao; Ran Li; Hongyu Chen; Yuhui Zhang; Ziheng Wang; Yuan Liu; Thiago S. F. X. Teixeira; Diyi Yang; Ke Wang; Alex Aiken
>
> **摘要:** As large language models (LLMs) become integral to code-related tasks, a central question emerges: do LLMs truly understand program execution semantics? We introduce EquiBench, a new benchmark for evaluating LLMs through equivalence checking, i.e., determining whether two programs produce identical outputs for all possible inputs. Unlike prior code generation benchmarks, this task directly tests a model's understanding of code execution semantics. EquiBench consists of 2400 program pairs across four languages and six categories. These pairs are generated through program analysis, compiler scheduling, and superoptimization, ensuring high-confidence labels, nontrivial difficulty, and full automation. The transformations span syntactic edits, structural modifications, and algorithmic changes, covering a broad spectrum of semantic variation. We evaluate 19 state-of-the-art LLMs and find that in the most challenging categories, the best accuracies are 63.8% and 76.2%, only modestly above the 50% random baseline. Further analysis reveals that models often rely on syntactic similarity rather than exhibiting robust reasoning over execution semantics, highlighting fundamental limitations.
>
---
#### [replaced 111] Automating Intervention Discovery from Scientific Literature: A Progressive Ontology Prompting and Dual-LLM Framework
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.00054v2](http://arxiv.org/pdf/2409.00054v2)**

> **作者:** Yuting Hu; Dancheng Liu; Qingyun Wang; Charles Yu; Chenhui Xu; Qingxiao Zheng; Heng Ji; Jinjun Xiong
>
> **备注:** Accepted by IJCAI2025
>
> **摘要:** Identifying effective interventions from the scientific literature is challenging due to the high volume of publications, specialized terminology, and inconsistent reporting formats, making manual curation laborious and prone to oversight. To address this challenge, this paper proposes a novel framework leveraging large language models (LLMs), which integrates a progressive ontology prompting (POP) algorithm with a dual-agent system, named LLM-Duo. On the one hand, the POP algorithm conducts a prioritized breadth-first search (BFS) across a predefined ontology, generating structured prompt templates and action sequences to guide the automatic annotation process. On the other hand, the LLM-Duo system features two specialized LLM agents, an explorer and an evaluator, working collaboratively and adversarially to continuously refine annotation quality. We showcase the real-world applicability of our framework through a case study focused on speech-language intervention discovery. Experimental results show that our approach surpasses advanced baselines, achieving more accurate and comprehensive annotations through a fully automated process. Our approach successfully identified 2,421 interventions from a corpus of 64,177 research articles in the speech-language pathology domain, culminating in the creation of a publicly accessible intervention knowledge base with great potential to benefit the speech-language pathology community.
>
---
#### [replaced 112] Evaluating the efficacy of LLM Safety Solutions : The Palit Benchmark Dataset
- **分类: cs.CR; cs.AI; cs.CL; F.2.2; I.2.7; F.2.2; I.2.7; F.2.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.13028v2](http://arxiv.org/pdf/2505.13028v2)**

> **作者:** Sayon Palit; Daniel Woods
>
> **摘要:** Large Language Models (LLMs) are increasingly integrated into critical systems in industries like healthcare and finance. Users can often submit queries to LLM-enabled chatbots, some of which can enrich responses with information retrieved from internal databases storing sensitive data. This gives rise to a range of attacks in which a user submits a malicious query and the LLM-system outputs a response that creates harm to the owner, such as leaking internal data or creating legal liability by harming a third-party. While security tools are being developed to counter these threats, there is little formal evaluation of their effectiveness and usability. This study addresses this gap by conducting a thorough comparative analysis of LLM security tools. We identified 13 solutions (9 closed-source, 4 open-source), but only 7 were evaluated due to a lack of participation by proprietary model owners.To evaluate, we built a benchmark dataset of malicious prompts, and evaluate these tools performance against a baseline LLM model (ChatGPT-3.5-Turbo). Our results show that the baseline model has too many false positives to be used for this task. Lakera Guard and ProtectAI LLM Guard emerged as the best overall tools showcasing the tradeoff between usability and performance. The study concluded with recommendations for greater transparency among closed source providers, improved context-aware detections, enhanced open-source engagement, increased user awareness, and the adoption of more representative performance metrics.
>
---
#### [replaced 113] Rate, Explain and Cite (REC): Enhanced Explanation and Attribution in Automatic Evaluation by Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.02448v3](http://arxiv.org/pdf/2411.02448v3)**

> **作者:** Aliyah R. Hsu; James Zhu; Zhichao Wang; Bin Bi; Shubham Mehrotra; Shiva K. Pentyala; Katherine Tan; Xiang-Bo Mao; Roshanak Omrani; Sougata Chaudhuri; Regunathan Radhakrishnan; Sitaram Asur; Claire Na Cheng; Bin Yu
>
> **摘要:** LLMs have demonstrated impressive proficiency in generating coherent and high-quality text, making them valuable across a range of text-generation tasks. However, rigorous evaluation of this generated content is crucial, as ensuring its quality remains a significant challenge due to persistent issues such as factual inaccuracies and hallucination. This paper introduces three fine-tuned general-purpose LLM autoevaluators, REC-8B, REC-12B and REC-70B, specifically designed to evaluate generated text across several dimensions: faithfulness, instruction following, coherence, and completeness. These models not only provide ratings for these metrics but also offer detailed explanation and verifiable citation, thereby enhancing trust in the content. Moreover, the models support various citation modes, accommodating different requirements for latency and granularity. Extensive evaluations on diverse benchmarks demonstrate that our general-purpose LLM auto-evaluator, REC-70B, outperforms state-of-the-art LLMs, excelling in content evaluation by delivering better quality explanation and citation with minimal bias. Our REC dataset and models are available at https://github.com/adelaidehsu/REC.
>
---
#### [replaced 114] Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2403.11083v3](http://arxiv.org/pdf/2403.11083v3)**

> **作者:** Xiaohao Xu; Yunkang Cao; Huaxin Zhang; Nong Sang; Xiaonan Huang
>
> **备注:** Best Student Paper Award at IEEE International Conference on Computer Supported Cooperative Work in Design, 2025
>
> **摘要:** Anomaly detection is vital in various industrial scenarios, including the identification of unusual patterns in production lines and the detection of manufacturing defects for quality control. Existing techniques tend to be specialized in individual scenarios and lack generalization capacities. In this study, our objective is to develop a generic anomaly detection model that can be applied in multiple scenarios. To achieve this, we custom-build generic visual language foundation models that possess extensive knowledge and robust reasoning abilities as anomaly detectors and reasoners. Specifically, we introduce a multi-modal prompting strategy that incorporates domain knowledge from experts as conditions to guide the models. Our approach considers diverse prompt types, including task descriptions, class context, normality rules, and reference images. In addition, we unify the input representation of multi-modality into a 2D image format, enabling multi-modal anomaly detection and reasoning. Our preliminary studies demonstrate that combining visual and language prompts as conditions for customizing the models enhances anomaly detection performance. The customized models showcase the ability to detect anomalies across different data modalities such as images, point clouds, and videos. Qualitative case studies further highlight the anomaly detection and reasoning capabilities, particularly for multi-object scenes and temporal data. Our code is publicly available at https://github.com/Xiaohao-Xu/Customizable-VLM
>
---
#### [replaced 115] Make LoRA Great Again: Boosting LoRA with Adaptive Singular Values and Mixture-of-Experts Optimization Alignment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16894v3](http://arxiv.org/pdf/2502.16894v3)**

> **作者:** Chenghao Fan; Zhenyi Lu; Sichen Liu; Chengfeng Gu; Xiaoye Qu; Wei Wei; Yu Cheng
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** While Low-Rank Adaptation (LoRA) enables parameter-efficient fine-tuning for Large Language Models (LLMs), its performance often falls short of Full Fine-Tuning (Full FT). Current methods optimize LoRA by initializing with static singular value decomposition (SVD) subsets, leading to suboptimal leveraging of pre-trained knowledge. Another path for improving LoRA is incorporating a Mixture-of-Experts (MoE) architecture. However, weight misalignment and complex gradient dynamics make it challenging to adopt SVD prior to the LoRA MoE architecture. To mitigate these issues, we propose \underline{G}reat L\underline{o}R\underline{A} Mixture-of-Exper\underline{t} (GOAT), a framework that (1) adaptively integrates relevant priors using an SVD-structured MoE, and (2) aligns optimization with full fine-tuned MoE by deriving a theoretical scaling factor. We demonstrate that proper scaling, without modifying the architecture or training algorithms, boosts LoRA MoE's efficiency and performance. Experiments across 25 datasets, including natural language understanding, commonsense reasoning, image classification, and natural language generation, demonstrate GOAT's state-of-the-art performance, closing the gap with Full FT.
>
---
#### [replaced 116] Leveraging LLM Inconsistency to Boost Pass@k Performance
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12938v2](http://arxiv.org/pdf/2505.12938v2)**

> **作者:** Uri Dalal; Meirav Segal; Zvika Ben-Haim; Dan Lahav; Omer Nevo
>
> **摘要:** Large language models (LLMs) achieve impressive abilities in numerous domains, but exhibit inconsistent performance in response to minor input changes. Rather than view this as a drawback, in this paper we introduce a novel method for leveraging models' inconsistency to boost Pass@k performance. Specifically, we present a "Variator" agent that generates k variants of a given task and submits one candidate solution for each one. Our variant generation approach is applicable to a wide range of domains as it is task agnostic and compatible with free-form inputs. We demonstrate the efficacy of our agent theoretically using a probabilistic model of the inconsistency effect, and show empirically that it outperforms the baseline on the APPS dataset. Furthermore, we establish that inconsistency persists even in frontier reasoning models across coding and cybersecurity domains, suggesting our method is likely to remain relevant for future model generations.
>
---
#### [replaced 117] CodeFlowBench: A Multi-turn, Iterative Benchmark for Complex Code Generation
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.21751v2](http://arxiv.org/pdf/2504.21751v2)**

> **作者:** Sizhe Wang; Zhengren Wang; Dongsheng Ma; Yongan Yu; Rui Ling; Zhiyu Li; Feiyu Xiong; Wentao Zhang
>
> **摘要:** Modern software development demands code that is maintainable, testable, and scalable by organizing the implementation into modular components with iterative reuse of existing codes. We formalize this iterative, multi-turn paradigm as codeflow and introduce CodeFlowBench, the first benchmark designed to comprehensively evaluate LLMs' ability to perform codeflow, namely implementing new functionality by reusing existing functions over multiple turns. CodeFlowBench comprises 5,258 problems from Codeforces and is continuously updated via an automated pipeline, which decomposes each problem into subproblems with unit tests based on dependency tree analysis and dataflow analysis. We further propose a novel evaluation framework featured dual assessment protocol and structural metrics derived from dependency trees. Extensive experiments on 16 popular LLMs reveal significant performance degradation in multi-turn scenarios. For instance, o1-mini retains only 20.8% Pass@1 in multi-turn scenario versus 37.8% in single-turn scenario. More fine-grained analysis illustrates that model performance inversely correlates with dependency complexity. These findings not only highlight the critical challenges for supporting real-world workflows, but also establish CodeFlowBench as an essential tool for advancing code generation research.
>
---
#### [replaced 118] SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12464v3](http://arxiv.org/pdf/2502.12464v3)**

> **作者:** Seanie Lee; Dong Bok Lee; Dominik Wagner; Minki Kang; Haebin Seong; Tobias Bocklet; Juho Lee; Sung Ju Hwang
>
> **备注:** ACL 2025 findings
>
> **摘要:** Deploying large language models (LLMs) in real-world applications requires robust safety guard models to detect and block harmful user prompts. While large safety guard models achieve strong performance, their computational cost is substantial. To mitigate this, smaller distilled models are used, but they often underperform on "hard" examples where the larger model provides accurate predictions. We observe that many inputs can be reliably handled by the smaller model, while only a small fraction require the larger model's capacity. Motivated by this, we propose SafeRoute, a binary router that distinguishes hard examples from easy ones. Our method selectively applies the larger safety guard model to the data that the router considers hard, improving efficiency while maintaining accuracy compared to solely using the larger safety guard model. Experimental results on multiple benchmark datasets demonstrate that our adaptive model selection significantly enhances the trade-off between computational cost and safety performance, outperforming relevant baselines.
>
---
#### [replaced 119] Arithmetics-Based Decomposition of Numeral Words -- Arithmetic Conditions give the Unpacking Strategy
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2312.10097v2](http://arxiv.org/pdf/2312.10097v2)**

> **作者:** Isidor Konrad Maier; Matthias Wolff
>
> **摘要:** This paper presents a novel numeral decomposer based on arithmetic criteria. The criteria are not dependent on a base-10 assumption but only on Hurford's Packing Strategy. Hurford's Packing Strategy constitutes numerals by packing factors and summands to multiplicators. We found out that a numeral of value n has a multiplicator larger than sqrt(n), a summand smaller than n/2 and a factor smaller than sqrt(n). Using these findings, the numeral decomposer attempts to detect and unpack factors and summand in order to reverse Hurford's Packing strategy. We tested its applicability for incremental unsupervised grammar induction in 273 languages. This way, grammars were obtained with sensible mathematical attributes that explain the structure of produced numerals. The numeral-decomposer-induced grammars are often close to expert-made and more compact than numeral grammars induced by a modern state-of-the-art grammar induction tool. Furthermore, this paper contains a report about the few cases of incorrect induced mathematical attributes, which are often linked to linguistic peculiarities like context sensitivity.
>
---
#### [replaced 120] Improving LLM Unlearning Robustness via Random Perturbations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.19202v3](http://arxiv.org/pdf/2501.19202v3)**

> **作者:** Dang Huu-Tien; Hoang Thanh-Tung; Anh Bui; Le-Minh Nguyen; Naoya Inoue
>
> **备注:** 23 pages, 10 figures, 5 tables
>
> **摘要:** In this paper, we show that current state-of-the-art LLM unlearning methods inherently reduce models' robustness, causing them to misbehave even when a single non-adversarial forget-token is in the retain-query. Toward understanding underlying causes, we reframe the unlearning process as backdoor attacks and defenses: forget-tokens act as backdoor triggers that, when activated in retain-queries, cause disruptions in unlearned models' behaviors, similar to successful backdoor attacks. To mitigate this vulnerability, we propose Random Noise Augmentation (RNA) -- a plug-and-play, model and method agnostic approach with theoretical guarantees for improving the robustness of unlearned models. Extensive experiments demonstrate that RNA significantly improves the robustness of unlearned models, maintains unlearning performances while introducing no additional computational overhead.
>
---
#### [replaced 121] Designing and Contextualising Probes for African Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10081v2](http://arxiv.org/pdf/2505.10081v2)**

> **作者:** Wisdom Aduah; Francois Meyer
>
> **摘要:** Pretrained language models (PLMs) for African languages are continually improving, but the reasons behind these advances remain unclear. This paper presents the first systematic investigation into probing PLMs for linguistic knowledge about African languages. We train layer-wise probes for six typologically diverse African languages to analyse how linguistic features are distributed. We also design control tasks, a way to interpret probe performance, for the MasakhaPOS dataset. We find PLMs adapted for African languages to encode more linguistic information about target languages than massively multilingual PLMs. Our results reaffirm previous findings that token-level syntactic information concentrates in middle-to-last layers, while sentence-level semantic information is distributed across all layers. Through control tasks and probing baselines, we confirm that performance reflects the internal knowledge of PLMs rather than probe memorisation. Our study applies established interpretability techniques to African-language PLMs. In doing so, we highlight the internal mechanisms underlying the success of strategies like active learning and multilingual adaptation.
>
---
#### [replaced 122] PLAYER*: Enhancing LLM-based Multi-Agent Communication and Interaction in Murder Mystery Games
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.17662v5](http://arxiv.org/pdf/2404.17662v5)**

> **作者:** Qinglin Zhu; Runcong Zhao; Bin Liang; Jinhua Du; Lin Gui; Yulan He
>
> **摘要:** We introduce WellPlay, a reasoning dataset for multi-agent conversational inference in Murder Mystery Games (MMGs). WellPlay comprises 1,482 inferential questions across 12 games, spanning objectives, reasoning, and relationship understanding, and establishes a systematic benchmark for evaluating agent reasoning abilities in complex social settings. Building on this foundation, we present PLAYER*, a novel framework for Large Language Model (LLM)-based agents in MMGs. MMGs pose unique challenges, including undefined state spaces, absent intermediate rewards, and the need for strategic reasoning through natural language. PLAYER* addresses these challenges with a sensor-based state representation and an information-driven strategy that optimises questioning and suspect pruning. Experiments show that PLAYER* outperforms existing methods in reasoning accuracy, efficiency, and agent-human interaction, advancing reasoning agents for complex social scenarios.
>
---
#### [replaced 123] Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.14318v2](http://arxiv.org/pdf/2411.14318v2)**

> **作者:** Zheheng Luo; Xin Zhang; Xiao Liu; Haoling Li; Yeyun Gong; Chen Qi; Peng Cheng
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** It is well-known that a diverse corpus is critical for training large language models, which are typically constructed from a mixture of various domains. In general, previous efforts resort to sampling training data from different domains with static proportions, as well as adjusting data proportions during training. However, few methods have addressed the complexities of domain-adaptive continual pre-training. To fill this gap, we propose Velocitune, a novel framework dynamically assesses learning velocity and adjusts data proportions accordingly, favoring slower-learning domains while shunning faster-learning ones, which is guided by a scaling law to indicate the desired learning goal for each domain with less associated cost. To evaluate the effectiveness of Velocitune, we conduct experiments in a reasoning-focused dataset with CodeLlama, as well as in a corpus specialised for system command generation with Llama3 and Mistral. Velocitune achieves performance gains in both math and code reasoning tasks and command-line generation benchmarks. Further analysis reveals that key factors driving Velocitune's effectiveness include target loss prediction and data ordering.
>
---
#### [replaced 124] A Comparative Study of Learning Paradigms in Large Language Models via Intrinsic Dimension
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.06245v2](http://arxiv.org/pdf/2412.06245v2)**

> **作者:** Saahith Janapati; Yangfeng Ji
>
> **摘要:** The performance of Large Language Models (LLMs) on natural language tasks can be improved through both supervised fine-tuning (SFT) and in-context learning (ICL), which operate via distinct mechanisms. Supervised fine-tuning updates the model's weights by minimizing loss on training data, whereas in-context learning leverages task demonstrations embedded in the prompt, without changing the model's parameters. This study investigates the effects of these learning paradigms on the hidden representations of LLMs using Intrinsic Dimension (ID). We use ID to estimate the number of degrees of freedom between representations extracted from LLMs as they perform specific natural language tasks. We first explore how the ID of LLM representations evolves during SFT and how it varies due to the number of demonstrations in ICL. We then compare the IDs induced by SFT and ICL and find that ICL consistently induces a higher ID compared to SFT, suggesting that representations generated during ICL reside in higher dimensional manifolds in the embedding space.
>
---
#### [replaced 125] Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics
- **分类: cs.CL; 68T5; I.2.7**

- **链接: [http://arxiv.org/pdf/2410.21272v2](http://arxiv.org/pdf/2410.21272v2)**

> **作者:** Yaniv Nikankin; Anja Reusch; Aaron Mueller; Yonatan Belinkov
>
> **摘要:** Do large language models (LLMs) solve reasoning tasks by learning robust generalizable algorithms, or do they memorize training data? To investigate this question, we use arithmetic reasoning as a representative task. Using causal analysis, we identify a subset of the model (a circuit) that explains most of the model's behavior for basic arithmetic logic and examine its functionality. By zooming in on the level of individual circuit neurons, we discover a sparse set of important neurons that implement simple heuristics. Each heuristic identifies a numerical input pattern and outputs corresponding answers. We hypothesize that the combination of these heuristic neurons is the mechanism used to produce correct arithmetic answers. To test this, we categorize each neuron into several heuristic types-such as neurons that activate when an operand falls within a certain range-and find that the unordered combination of these heuristic types is the mechanism that explains most of the model's accuracy on arithmetic prompts. Finally, we demonstrate that this mechanism appears as the main source of arithmetic accuracy early in training. Overall, our experimental results across several LLMs show that LLMs perform arithmetic using neither robust algorithms nor memorization; rather, they rely on a "bag of heuristics".
>
---
#### [replaced 126] Who Taught You That? Tracing Teachers in Model Distillation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.06659v3](http://arxiv.org/pdf/2502.06659v3)**

> **作者:** Somin Wadhwa; Chantal Shaib; Silvio Amir; Byron C. Wallace
>
> **备注:** Findings of ACL 2025
>
> **摘要:** Model distillation -- using outputs from a large teacher model to teach a small student model -- is a practical means of creating efficient models for a particular task. We ask: Can we identify a students' teacher based on its outputs? Such "footprints" left by teacher LLMs would be interesting artifacts. Beyond this, reliable teacher inference may have practical implications as actors seek to distill specific capabilities of massive proprietary LLMs into deployed smaller LMs, potentially violating terms of service. We consider practical task distillation targets including summarization, question answering, and instruction-following. We assume a finite set of candidate teacher models, which we treat as blackboxes. We design discriminative models that operate over lexical features. We find that $n$-gram similarity alone is unreliable for identifying teachers, but part-of-speech (PoS) templates preferred by student models mimic those of their teachers.
>
---
