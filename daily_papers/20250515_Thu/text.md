# 自然语言处理 cs.CL

- **最新发布 31 篇**

- **更新 24 篇**

## 最新发布

#### [new 001] Atomic Consistency Preference Optimization for Long-Form Question Answering
- **分类: cs.CL**

- **简介: 该论文针对长问答任务中LLMs易生成事实错误的问题，提出自监督偏好优化方法ACPO。通过捕捉多组随机响应间的原子一致性信号，自动筛选高质量数据对进行模型对齐，无需依赖外部模型或知识库。实验表明ACPO在事实准确性上优于传统监督基线，提供了高效可扩展的解决方案。**

- **链接: [http://arxiv.org/pdf/2505.09039v1](http://arxiv.org/pdf/2505.09039v1)**

> **作者:** Jingfeng Chen; Raghuveer Thirukovalluru; Junlin Wang; Kaiwei Luo; Bhuwan Dhingra
>
> **备注:** 16 pages, 2 figures
>
> **摘要:** Large Language Models (LLMs) frequently produce factoid hallucinations - plausible yet incorrect answers. A common mitigation strategy is model alignment, which improves factual accuracy by training on curated factual and non-factual pairs. However, this approach often relies on a stronger model (e.g., GPT-4) or an external knowledge base to assess factual correctness, which may not always be accessible. To address this, we propose Atomic Consistency Preference Optimization (ACPO), a self-supervised preference-tuning method that enhances factual accuracy without external supervision. ACPO leverages atomic consistency signals, i.e., the agreement of individual facts across multiple stochastic responses, to identify high- and low-quality data pairs for model alignment. By eliminating the need for costly GPT calls, ACPO provides a scalable and efficient approach to improving factoid question-answering. Despite being self-supervised, empirical results demonstrate that ACPO outperforms FactAlign, a strong supervised alignment baseline, by 1.95 points on the LongFact and BioGen datasets, highlighting its effectiveness in enhancing factual reliability without relying on external models or knowledge bases.
>
---
#### [new 002] Multilingual Machine Translation with Quantum Encoder Decoder Attention-based Convolutional Variational Circuits
- **分类: cs.CL; cs.AI; cs.ET**

- **简介: 该论文研究多语言机器翻译，提出量子编码-解码架构QEDACVC，替代传统经典计算模型。通过量子卷积、池化、变分电路及注意力机制，在OPUS数据集（英/法/德/印）实现82%翻译准确率，探索量子计算在NLP任务中的应用潜力。**

- **链接: [http://arxiv.org/pdf/2505.09407v1](http://arxiv.org/pdf/2505.09407v1)**

> **作者:** Subrit Dikshit; Ritu Tiwari; Priyank Jain
>
> **备注:** 12 pages, 12 figures
>
> **摘要:** Cloud-based multilingual translation services like Google Translate and Microsoft Translator achieve state-of-the-art translation capabilities. These services inherently use large multilingual language models such as GRU, LSTM, BERT, GPT, T5, or similar encoder-decoder architectures with attention mechanisms as the backbone. Also, new age natural language systems, for instance ChatGPT and DeepSeek, have established huge potential in multiple tasks in natural language processing. At the same time, they also possess outstanding multilingual translation capabilities. However, these models use the classical computing realm as a backend. QEDACVC (Quantum Encoder Decoder Attention-based Convolutional Variational Circuits) is an alternate solution that explores the quantum computing realm instead of the classical computing realm to study and demonstrate multilingual machine translation. QEDACVC introduces the quantum encoder-decoder architecture that simulates and runs on quantum computing hardware via quantum convolution, quantum pooling, quantum variational circuit, and quantum attention as software alterations. QEDACVC achieves an Accuracy of 82% when trained on the OPUS dataset for English, French, German, and Hindi corpora for multilingual translations.
>
---
#### [new 003] S-DAT: A Multilingual, GenAI-Driven Framework for Automated Divergent Thinking Assessment
- **分类: cs.CL; cs.HC**

- **简介: 该论文提出S-DAT框架，属于自动化创造力评估任务，解决传统发散思维测评效率低、语言依赖性强和主观偏差问题。通过大语言模型和多语言语义嵌入计算语义距离，实现跨语言标准化评估，验证了11种语言场景下的有效性和区分度，促进全球创造力研究的公平性与扩展性。**

- **链接: [http://arxiv.org/pdf/2505.09068v1](http://arxiv.org/pdf/2505.09068v1)**

> **作者:** Jennifer Haase; Paul H. P. Hanel; Sebastian Pokutta
>
> **摘要:** This paper introduces S-DAT (Synthetic-Divergent Association Task), a scalable, multilingual framework for automated assessment of divergent thinking (DT) -a core component of human creativity. Traditional creativity assessments are often labor-intensive, language-specific, and reliant on subjective human ratings, limiting their scalability and cross-cultural applicability. In contrast, S-DAT leverages large language models and advanced multilingual embeddings to compute semantic distance -- a language-agnostic proxy for DT. We evaluate S-DAT across eleven diverse languages, including English, Spanish, German, Russian, Hindi, and Japanese (Kanji, Hiragana, Katakana), demonstrating robust and consistent scoring across linguistic contexts. Unlike prior DAT approaches, the S-DAT shows convergent validity with other DT measures and correct discriminant validity with convergent thinking. This cross-linguistic flexibility allows for more inclusive, global-scale creativity research, addressing key limitations of earlier approaches. S-DAT provides a powerful tool for fairer, more comprehensive evaluation of cognitive flexibility in diverse populations and can be freely assessed online: https://sdat.iol.zib.de/.
>
---
#### [new 004] A suite of LMs comprehend puzzle statements as well as humans
- **分类: cs.CL**

- **简介: 该论文属于语言模型评估任务，旨在纠正先前低估模型语言理解能力的结论。通过限制人类重读的对比实验，发现模型（如GPT-4、Falcon-180B）表现优于人类（81% vs 73%），且共享语用挑战（如互惠动作）。分析表明实验设计和编码偏差导致模型能力被系统性低估，需改进评估方法。**

- **链接: [http://arxiv.org/pdf/2505.08996v1](http://arxiv.org/pdf/2505.08996v1)**

> **作者:** Adele E Goldberg; Supantho Rakshit; Jennifer Hu; Kyle Mahowald
>
> **摘要:** Recent claims suggest that large language models (LMs) underperform humans in comprehending minimally complex English statements (Dentella et al., 2024). Here, we revisit those findings and argue that human performance was overestimated, while LLM abilities were underestimated. Using the same stimuli, we report a preregistered study comparing human responses in two conditions: one allowed rereading (replicating the original study), and one that restricted rereading (a more naturalistic comprehension test). Human accuracy dropped significantly when rereading was restricted (73%), falling below that of Falcon-180B-Chat (76%) and GPT-4 (81%). The newer GPT-o1 model achieves perfect accuracy. Results further show that both humans and models are disproportionately challenged by queries involving potentially reciprocal actions (e.g., kissing), suggesting shared pragmatic sensitivities rather than model-specific deficits. Additional analyses using Llama-2-70B log probabilities, a recoding of open-ended model responses, and grammaticality ratings of other sentences reveal systematic underestimation of model performance. We find that GPT-4o can align with either naive or expert grammaticality judgments, depending on prompt framing. These findings underscore the need for more careful experimental design and coding practices in LLM evaluation, and they challenge the assumption that current models are inherently weaker than humans at language comprehension.
>
---
#### [new 005] Scent of Knowledge: Optimizing Search-Enhanced Reasoning with Information Foraging
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于检索增强生成任务，旨在解决传统静态检索策略在复杂多步推理中的不足。提出InForage强化学习框架，模拟动态信息觅食过程，通过奖励中间检索质量实现自适应搜索，并构建数据集训练模型，提升实时问答和推理效果。**

- **链接: [http://arxiv.org/pdf/2505.09316v1](http://arxiv.org/pdf/2505.09316v1)**

> **作者:** Hongjin Qian; Zheng Liu
>
> **备注:** 16 pages
>
> **摘要:** Augmenting large language models (LLMs) with external retrieval has become a standard method to address their inherent knowledge cutoff limitations. However, traditional retrieval-augmented generation methods employ static, pre-inference retrieval strategies, making them inadequate for complex tasks involving ambiguous, multi-step, or evolving information needs. Recent advances in test-time scaling techniques have demonstrated significant potential in enabling LLMs to dynamically interact with external tools, motivating the shift toward adaptive inference-time retrieval. Inspired by Information Foraging Theory (IFT), we propose InForage, a reinforcement learning framework that formalizes retrieval-augmented reasoning as a dynamic information-seeking process. Unlike existing approaches, InForage explicitly rewards intermediate retrieval quality, encouraging LLMs to iteratively gather and integrate information through adaptive search behaviors. To facilitate training, we construct a human-guided dataset capturing iterative search and reasoning trajectories for complex, real-world web tasks. Extensive evaluations across general question answering, multi-hop reasoning tasks, and a newly developed real-time web QA dataset demonstrate InForage's superior performance over baseline methods. These results highlight InForage's effectiveness in building robust, adaptive, and efficient reasoning agents.
>
---
#### [new 006] A Scalable Unsupervised Framework for multi-aspect labeling of Multilingual and Multi-Domain Review Data
- **分类: cs.CL**

- **简介: 该论文提出一种无监督多语言跨领域评论数据多标签标注框架，解决传统方法依赖标注数据、受限于单域/单语言的问题。通过聚类提取候选类别，结合负采样生成嵌入向量，验证标签质量并微调预训练模型，实验表明其标注效果优于大模型且接近人工水平，具备扩展性。任务属于文本挖掘中的多维度信息抽取。**

- **链接: [http://arxiv.org/pdf/2505.09286v1](http://arxiv.org/pdf/2505.09286v1)**

> **作者:** Jiin Park; Misuk Kim
>
> **备注:** 36 pages, 3 figures
>
> **摘要:** Effectively analyzing online review data is essential across industries. However, many existing studies are limited to specific domains and languages or depend on supervised learning approaches that require large-scale labeled datasets. To address these limitations, we propose a multilingual, scalable, and unsupervised framework for cross-domain aspect detection. This framework is designed for multi-aspect labeling of multilingual and multi-domain review data. In this study, we apply automatic labeling to Korean and English review datasets spanning various domains and assess the quality of the generated labels through extensive experiments. Aspect category candidates are first extracted through clustering, and each review is then represented as an aspect-aware embedding vector using negative sampling. To evaluate the framework, we conduct multi-aspect labeling and fine-tune several pretrained language models to measure the effectiveness of the automatically generated labels. Results show that these models achieve high performance, demonstrating that the labels are suitable for training. Furthermore, comparisons with publicly available large language models highlight the framework's superior consistency and scalability when processing large-scale data. A human evaluation also confirms that the quality of the automatic labels is comparable to those created manually. This study demonstrates the potential of a robust multi-aspect labeling approach that overcomes limitations of supervised methods and is adaptable to multilingual, multi-domain environments. Future research will explore automatic review summarization and the integration of artificial intelligence agents to further improve the efficiency and depth of review analysis.
>
---
#### [new 007] PT-MoE: An Efficient Finetuning Framework for Integrating Mixture-of-Experts into Prompt Tuning
- **分类: cs.CL**

- **简介: 该论文属于参数高效微调领域，旨在提升提示调优的跨任务性能。针对现有方法效率与效果不均衡的问题，提出PT-MoE框架，将矩阵分解与混合专家路由结合，通过参数共享和动态适配实现高效训练。实验证明其在问答和数学任务中参数减少25%的同时性能超越主流方法。**

- **链接: [http://arxiv.org/pdf/2505.09519v1](http://arxiv.org/pdf/2505.09519v1)**

> **作者:** Zongqian Li; Yixuan Su; Nigel Collier
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) methods have shown promise in adapting large language models, yet existing approaches exhibit counter-intuitive phenomena: integrating router into prompt tuning (PT) increases training efficiency yet does not improve performance universally; parameter reduction through matrix decomposition can improve performance in specific domains. Motivated by these observations and the modular nature of PT, we propose PT-MoE, a novel framework that integrates matrix decomposition with mixture-of-experts (MoE) routing for efficient PT. Results across 17 datasets demonstrate that PT-MoE achieves state-of-the-art performance in both question answering (QA) and mathematical problem solving tasks, improving F1 score by 1.49 points over PT and 2.13 points over LoRA in QA tasks, while enhancing mathematical accuracy by 10.75 points over PT and 0.44 points over LoRA, all while using 25% fewer parameters than LoRA. Our analysis reveals that while PT methods generally excel in QA tasks and LoRA-based methods in math datasets, the integration of matrix decomposition and MoE in PT-MoE yields complementary benefits: decomposition enables efficient parameter sharing across experts while MoE provides dynamic adaptation, collectively enabling PT-MoE to demonstrate cross-task consistency and generalization abilities. These findings, along with ablation studies on routing mechanisms and architectural components, provide insights for future PEFT methods.
>
---
#### [new 008] Human-AI Collaboration or Academic Misconduct? Measuring AI Use in Student Writing Through Stylometric Evidence
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究如何通过作者验证技术（AV）量化学生写作中AI的使用，区分合作与学术不端。任务属于文本分类/检测，解决教育场景中AI辅助的透明化问题。工作涵盖构建含AI生成文本的数据集，开发特征向量差异方法建立学生写作特征，并评估模型在识别AI文本及抗模仿攻击的能力，为教育者提供检测工具。**

- **链接: [http://arxiv.org/pdf/2505.08828v1](http://arxiv.org/pdf/2505.08828v1)**

> **作者:** Eduardo Araujo Oliveira; Madhavi Mohoni; Sonsoles López-Pernas; Mohammed Saqr
>
> **备注:** 19 pages, 10 figures, 11 tables
>
> **摘要:** As human-AI collaboration becomes increasingly prevalent in educational contexts, understanding and measuring the extent and nature of such interactions pose significant challenges. This research investigates the use of authorship verification (AV) techniques not as a punitive measure, but as a means to quantify AI assistance in academic writing, with a focus on promoting transparency, interpretability, and student development. Building on prior work, we structured our investigation into three stages: dataset selection and expansion, AV method development, and systematic evaluation. Using three datasets - including a public dataset (PAN-14) and two from University of Melbourne students from various courses - we expanded the data to include LLM-generated texts, totalling 1,889 documents and 540 authorship problems from 506 students. We developed an adapted Feature Vector Difference AV methodology to construct robust academic writing profiles for students, designed to capture meaningful, individual characteristics of their writing. The method's effectiveness was evaluated across multiple scenarios, including distinguishing between student-authored and LLM-generated texts and testing resilience against LLMs' attempts to mimic student writing styles. Results demonstrate the enhanced AV classifier's ability to identify stylometric discrepancies and measure human-AI collaboration at word and sentence levels while providing educators with a transparent tool to support academic integrity investigations. This work advances AV technology, offering actionable insights into the dynamics of academic writing in an AI-driven era.
>
---
#### [new 009] CEC-Zero: Chinese Error Correction Solution Based on LLM
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对中文拼写纠错任务，提出CEC-Zero框架，解决传统LLM可靠性不足和泛化受限问题。通过强化学习使大模型自主习得纠错策略，无需标注数据或辅助模型，实现高精度跨领域纠错，优化中文NLP应用的可靠性。**

- **链接: [http://arxiv.org/pdf/2505.09082v1](http://arxiv.org/pdf/2505.09082v1)**

> **作者:** Sophie Zhang; Zhiming Lin
>
> **摘要:** Recent advancements in large language models (LLMs) demonstrate exceptional Chinese text processing capabilities, particularly in Chinese Spelling Correction (CSC). While LLMs outperform traditional BERT-based models in accuracy and robustness, challenges persist in reliability and generalization. This paper proposes CEC-Zero, a novel reinforcement learning (RL) framework enabling LLMs to self-correct through autonomous error strategy learning without external supervision. By integrating RL with LLMs' generative power, the method eliminates dependency on annotated data or auxiliary models. Experiments reveal RL-enhanced LLMs achieve industry-viable accuracy and superior cross-domain generalization, offering a scalable solution for reliability optimization in Chinese NLP applications. This breakthrough facilitates LLM deployment in practical Chinese text correction scenarios while establishing a new paradigm for self-improving language models.
>
---
#### [new 010] A Comprehensive Analysis of Large Language Model Outputs: Similarity, Diversity, and Bias
- **分类: cs.CL**

- **简介: 该论文分析大语言模型（LLM）输出的相似性、多样性和偏见，属于自然语言处理评估任务。研究通过5000个提示生成300万文本，对比12个模型的输出特性，揭示同一模型内容相似度高、不同模型风格差异显著，并发现部分模型在伦理平衡与多样性上的优势，为模型优化和伦理评估提供依据。**

- **链接: [http://arxiv.org/pdf/2505.09056v1](http://arxiv.org/pdf/2505.09056v1)**

> **作者:** Brandon Smith; Mohamed Reda Bouadjenek; Tahsin Alamgir Kheya; Phillip Dawson; Sunil Aryal
>
> **摘要:** Large Language Models (LLMs) represent a major step toward artificial general intelligence, significantly advancing our ability to interact with technology. While LLMs perform well on Natural Language Processing tasks -- such as translation, generation, code writing, and summarization -- questions remain about their output similarity, variability, and ethical implications. For instance, how similar are texts generated by the same model? How does this compare across different models? And which models best uphold ethical standards? To investigate, we used 5{,}000 prompts spanning diverse tasks like generation, explanation, and rewriting. This resulted in approximately 3 million texts from 12 LLMs, including proprietary and open-source systems from OpenAI, Google, Microsoft, Meta, and Mistral. Key findings include: (1) outputs from the same LLM are more similar to each other than to human-written texts; (2) models like WizardLM-2-8x22b generate highly similar outputs, while GPT-4 produces more varied responses; (3) LLM writing styles differ significantly, with Llama 3 and Mistral showing higher similarity, and GPT-4 standing out for distinctiveness; (4) differences in vocabulary and tone underscore the linguistic uniqueness of LLM-generated content; (5) some LLMs demonstrate greater gender balance and reduced bias. These results offer new insights into the behavior and diversity of LLM outputs, helping guide future development and ethical evaluation.
>
---
#### [new 011] WorldView-Bench: A Benchmark for Evaluating Global Cultural Perspectives in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY; cs.MA**

- **简介: 该论文提出WorldView-Bench基准，用于评估大语言模型的全球文化包容性。针对现有评测框架忽视文化多样性、导致模型西方中心化的问题，通过自由生成任务衡量文化极化现象，并提出基于多智能体协作和上下文嵌入的干预方法，将多元视角分布熵提升至94%，促进AI系统的文化平衡与伦理对齐。**

- **链接: [http://arxiv.org/pdf/2505.09595v1](http://arxiv.org/pdf/2505.09595v1)**

> **作者:** Abdullah Mushtaq; Imran Taj; Rafay Naeem; Ibrahim Ghaznavi; Junaid Qadir
>
> **备注:** Preprint. Submitted to the Journal of Artificial Intelligence Research (JAIR) on April 29, 2025
>
> **摘要:** Large Language Models (LLMs) are predominantly trained and aligned in ways that reinforce Western-centric epistemologies and socio-cultural norms, leading to cultural homogenization and limiting their ability to reflect global civilizational plurality. Existing benchmarking frameworks fail to adequately capture this bias, as they rely on rigid, closed-form assessments that overlook the complexity of cultural inclusivity. To address this, we introduce WorldView-Bench, a benchmark designed to evaluate Global Cultural Inclusivity (GCI) in LLMs by analyzing their ability to accommodate diverse worldviews. Our approach is grounded in the Multiplex Worldview proposed by Senturk et al., which distinguishes between Uniplex models, reinforcing cultural homogenization, and Multiplex models, which integrate diverse perspectives. WorldView-Bench measures Cultural Polarization, the exclusion of alternative perspectives, through free-form generative evaluation rather than conventional categorical benchmarks. We implement applied multiplexity through two intervention strategies: (1) Contextually-Implemented Multiplex LLMs, where system prompts embed multiplexity principles, and (2) Multi-Agent System (MAS)-Implemented Multiplex LLMs, where multiple LLM agents representing distinct cultural perspectives collaboratively generate responses. Our results demonstrate a significant increase in Perspectives Distribution Score (PDS) entropy from 13% at baseline to 94% with MAS-Implemented Multiplex LLMs, alongside a shift toward positive sentiment (67.7%) and enhanced cultural balance. These findings highlight the potential of multiplex-aware AI evaluation in mitigating cultural bias in LLMs, paving the way for more inclusive and ethically aligned AI systems.
>
---
#### [new 012] For GPT-4 as with Humans: Information Structure Predicts Acceptability of Long-Distance Dependencies
- **分类: cs.CL**

- **简介: 该论文属于语言模型评估任务，探究GPT-4是否像人类一样能通过信息结构预测长距离依存句的合理性。研究通过零样本测试验证了GPT-4能复现人类句法-信息结构的因果关系，表明其具备元语言推理能力，揭示了语言模型与人类语言认知的潜在关联。**

- **链接: [http://arxiv.org/pdf/2505.09005v1](http://arxiv.org/pdf/2505.09005v1)**

> **作者:** Nicole Cuneo; Eleanor Graves; Supantho Rakshit; Adele E. Goldberg
>
> **摘要:** It remains debated how well any LM understands natural language or generates reliable metalinguistic judgments. Moreover, relatively little work has demonstrated that LMs can represent and respect subtle relationships between form and function proposed by linguists. We here focus on a particular such relationship established in recent work: English speakers' judgments about the information structure of canonical sentences predicts independently collected acceptability ratings on corresponding 'long distance dependency' [LDD] constructions, across a wide array of base constructions and multiple types of LDDs. To determine whether any LM captures this relationship, we probe GPT-4 on the same tasks used with humans and new extensions.Results reveal reliable metalinguistic skill on the information structure and acceptability tasks, replicating a striking interaction between the two, despite the zero-shot, explicit nature of the tasks, and little to no chance of contamination [Studies 1a, 1b]. Study 2 manipulates the information structure of base sentences and confirms a causal relationship: increasing the prominence of a constituent in a context sentence increases the subsequent acceptability ratings on an LDD construction. The findings suggest a tight relationship between natural and GPT-4 generated English, and between information structure and syntax, which begs for further exploration.
>
---
#### [new 013] How an unintended Side Effect of a Research Project led to Boosting the Power of UML
- **分类: cs.CL**

- **简介: 该论文属于软件工程领域，旨在提升UML建模工具的功能。针对传统工具无法整合类图与对象图、缺乏动态执行能力的问题，研究者开发了支持多图表集成与对象执行的新型工具，促进了软件架构创新和教学实践。成果源于国际多级架构研究项目的意外发现，体现了科研副产品价值。**

- **链接: [http://arxiv.org/pdf/2505.09269v1](http://arxiv.org/pdf/2505.09269v1)**

> **作者:** Ulrich Frank; Pierre Maier
>
> **摘要:** This paper describes the design, implementation and use of a new UML modeling tool that represents a significant advance over conventional tools. Among other things, it allows the integration of class diagrams and object diagrams as well as the execution of objects. This not only enables new software architectures characterized by the integration of software with corresponding object models, but is also ideal for use in teaching, as it provides students with a particularly stimulating learning experience. A special feature of the project is that it has emerged from a long-standing international research project, which is aimed at a comprehensive multi-level architecture. The project is therefore an example of how research can lead to valuable results that arise as a side effect of other work.
>
---
#### [new 014] Qwen3 Technical Report
- **分类: cs.CL**

- **简介: 该论文属于大语言模型优化任务，旨在提升性能、效率与多语言能力。Qwen3通过整合动态推理模式（思考/非思考）统一框架解决模型切换问题，引入思维预算机制平衡计算资源与延迟，并缩减小模型训练成本。模型支持119种语言，性能超越前代及同类模型，全部开源。**

- **链接: [http://arxiv.org/pdf/2505.09388v1](http://arxiv.org/pdf/2505.09388v1)**

> **作者:** An Yang; Anfeng Li; Baosong Yang; Beichen Zhang; Binyuan Hui; Bo Zheng; Bowen Yu; Chang Gao; Chengen Huang; Chenxu Lv; Chujie Zheng; Dayiheng Liu; Fan Zhou; Fei Huang; Feng Hu; Hao Ge; Haoran Wei; Huan Lin; Jialong Tang; Jian Yang; Jianhong Tu; Jianwei Zhang; Jianxin Yang; Jiaxi Yang; Jing Zhou; Jingren Zhou; Junyang Lin; Kai Dang; Keqin Bao; Kexin Yang; Le Yu; Lianghao Deng; Mei Li; Mingfeng Xue; Mingze Li; Pei Zhang; Peng Wang; Qin Zhu; Rui Men; Ruize Gao; Shixuan Liu; Shuang Luo; Tianhao Li; Tianyi Tang; Wenbiao Yin; Xingzhang Ren; Xinyu Wang; Xinyu Zhang; Xuancheng Ren; Yang Fan; Yang Su; Yichang Zhang; Yinger Zhang; Yu Wan; Yuqiong Liu; Zekun Wang; Zeyu Cui; Zhenru Zhang; Zhipeng Zhou; Zihan Qiu
>
> **摘要:** In this work, we present Qwen3, the latest version of the Qwen model family. Qwen3 comprises a series of large language models (LLMs) designed to advance performance, efficiency, and multilingual capabilities. The Qwen3 series includes models of both dense and Mixture-of-Expert (MoE) architectures, with parameter scales ranging from 0.6 to 235 billion. A key innovation in Qwen3 is the integration of thinking mode (for complex, multi-step reasoning) and non-thinking mode (for rapid, context-driven responses) into a unified framework. This eliminates the need to switch between different models--such as chat-optimized models (e.g., GPT-4o) and dedicated reasoning models (e.g., QwQ-32B)--and enables dynamic mode switching based on user queries or chat templates. Meanwhile, Qwen3 introduces a thinking budget mechanism, allowing users to allocate computational resources adaptively during inference, thereby balancing latency and performance based on task complexity. Moreover, by leveraging the knowledge from the flagship models, we significantly reduce the computational resources required to build smaller-scale models, while ensuring their highly competitive performance. Empirical evaluations demonstrate that Qwen3 achieves state-of-the-art results across diverse benchmarks, including tasks in code generation, mathematical reasoning, agent tasks, etc., competitive against larger MoE models and proprietary models. Compared to its predecessor Qwen2.5, Qwen3 expands multilingual support from 29 to 119 languages and dialects, enhancing global accessibility through improved cross-lingual understanding and generation capabilities. To facilitate reproducibility and community-driven research and development, all Qwen3 models are publicly accessible under Apache 2.0.
>
---
#### [new 015] Clicking some of the silly options: Exploring Player Motivation in Static and Dynamic Educational Interactive Narratives
- **分类: cs.CL**

- **简介: 该论文属于教育游戏设计研究，旨在比较静态与AI驱动的动态互动叙事对学习动机的影响。通过开发两个版本的教育游戏（传统分支叙事 vs 动态序列生成），研究发现动态叙事能提升玩家参与度但需平衡教学与叙事灵活性，为AI教育游戏设计提供实践启示。**

- **链接: [http://arxiv.org/pdf/2505.08891v1](http://arxiv.org/pdf/2505.08891v1)**

> **作者:** Daeun Hwang; Samuel Shields; Alex Calderwood; Shi Johnson-Bey; Michael Mateas; Noah Wardrip-Fruin; Edward F. Melcer
>
> **备注:** 8 pages, 3 figures, 1 table, 1 appendix. Workshop paper, CHI 2025 Augmented Educators and AI
>
> **摘要:** Motivation is an important factor underlying successful learning. Previous research has demonstrated the positive effects that static interactive narrative games can have on motivation. Concurrently, advances in AI have made dynamic and adaptive approaches to interactive narrative increasingly accessible. However, limited work has explored the impact that dynamic narratives can have on learner motivation. In this paper, we compare two versions of Academical, a choice-based educational interactive narrative game about research ethics. One version employs a traditional hand-authored branching plot (i.e., static narrative) while the other dynamically sequences plots during play (i.e., dynamic narrative). Results highlight the importance of responsive content and a variety of choices for player engagement, while also illustrating the challenge of balancing pedagogical goals with the dynamic aspects of narrative. We also discuss design implications that arise from these findings. Ultimately, this work provides initial steps to illuminate the emerging potential of AI-driven dynamic narrative in educational games.
>
---
#### [new 016] Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs
- **分类: cs.CL**

- **简介: 该论文研究大语言模型中无关上下文导致分心的机制（任务：模型机理分析），揭示"上下文牵引"现象：模型会偏向重复提示中的任意词元，证明其与语义无关但受语义调控。通过可微分掩码定位"牵引注意力头"，关闭后可显著降低干扰，为缓解分心问题提供新方法。**

- **链接: [http://arxiv.org/pdf/2505.09338v1](http://arxiv.org/pdf/2505.09338v1)**

> **作者:** Jingcheng Niu; Xingdi Yuan; Tong Wang; Hamidreza Saghir; Amir H. Abdi
>
> **摘要:** We observe a novel phenomenon, contextual entrainment, across a wide range of language models (LMs) and prompt settings, providing a new mechanistic perspective on how LMs become distracted by ``irrelevant'' contextual information in the input prompt. Specifically, LMs assign significantly higher logits (or probabilities) to any tokens that have previously appeared in the context prompt, even for random tokens. This suggests that contextual entrainment is a mechanistic phenomenon, occurring independently of the relevance or semantic relation of the tokens to the question or the rest of the sentence. We find statistically significant evidence that the magnitude of contextual entrainment is influenced by semantic factors. Counterfactual prompts have a greater effect compared to factual ones, suggesting that while contextual entrainment is a mechanistic phenomenon, it is modulated by semantic factors. We hypothesise that there is a circuit of attention heads -- the entrainment heads -- that corresponds to the contextual entrainment phenomenon. Using a novel entrainment head discovery method based on differentiable masking, we identify these heads across various settings. When we ``turn off'' these heads, i.e., set their outputs to zero, the effect of contextual entrainment is significantly attenuated, causing the model to generate output that capitulates to what it would produce if no distracting context were provided. Our discovery of contextual entrainment, along with our investigation into LM distraction via the entrainment heads, marks a key step towards the mechanistic analysis and mitigation of the distraction problem.
>
---
#### [new 017] Automated Meta Prompt Engineering for Alignment with the Theory of Mind
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究如何通过自动元提示工程使大语言模型（LLM）生成内容与人类心理预期对齐。提出基于强化学习的代理框架（LLMaaJ），利用人类编辑数据优化文本生成，解决理论心智（ToM）对齐问题，通过希尔伯特空间量化内容特征，提升事实性和相关性。实验显示53.8%场景实现完全对齐，应用于体育赛事内容生产。**

- **链接: [http://arxiv.org/pdf/2505.09024v1](http://arxiv.org/pdf/2505.09024v1)**

> **作者:** Aaron Baughman; Rahul Agarwal; Eduardo Morales; Gozde Akay
>
> **备注:** 9 pages, 6 figures, 3 tables
>
> **摘要:** We introduce a method of meta-prompting that jointly produces fluent text for complex tasks while optimizing the similarity of neural states between a human's mental expectation and a Large Language Model's (LLM) neural processing. A technique of agentic reinforcement learning is applied, in which an LLM as a Judge (LLMaaJ) teaches another LLM, through in-context learning, how to produce content by interpreting the intended and unintended generated text traits. To measure human mental beliefs around content production, users modify long form AI-generated text articles before publication at the US Open 2024 tennis Grand Slam. Now, an LLMaaJ can solve the Theory of Mind (ToM) alignment problem by anticipating and including human edits within the creation of text from an LLM. Throughout experimentation and by interpreting the results of a live production system, the expectations of human content reviewers had 100% of alignment with AI 53.8% of the time with an average iteration count of 4.38. The geometric interpretation of content traits such as factualness, novelty, repetitiveness, and relevancy over a Hilbert vector space combines spatial volume (all trait importance) with vertices alignment (individual trait relevance) enabled the LLMaaJ to optimize on Human ToM. This resulted in an increase in content quality by extending the coverage of tennis action. Our work that was deployed at the US Open 2024 has been used across other live events within sports and entertainment.
>
---
#### [new 018] Behind Maya: Building a Multilingual Vision Language Model
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态视觉语言任务，旨在解决现有模型在低资源语言和文化多样性场景下的性能缺陷。研究者提出了开源模型Maya，通过构建基于LLaVA的八语种图文预训练数据集，并开发支持多语言的视觉语言模型，提升跨文化场景下的图文理解能力。**

- **链接: [http://arxiv.org/pdf/2505.08910v1](http://arxiv.org/pdf/2505.08910v1)**

> **作者:** Nahid Alam; Karthik Reddy Kanjula; Surya Guthikonda; Timothy Chung; Bala Krishna S Vegesna; Abhipsha Das; Anthony Susevski; Ryan Sze-Yin Chan; S M Iftekhar Uddin; Shayekh Bin Islam; Roshan Santhosh; Snegha A; Drishti Sharma; Chen Liu; Isha Chaturvedi; Genta Indra Winata; Ashvanth. S; Snehanshu Mukherjee; Alham Fikri Aji
>
> **备注:** Accepted at VLM4ALL CVPR 2025 Workshop
>
> **摘要:** In recent times, we have seen a rapid development of large Vision-Language Models (VLMs). They have shown impressive results on academic benchmarks, primarily in widely spoken languages but lack performance on low-resource languages and varied cultural contexts. To address these limitations, we introduce Maya, an open-source Multilingual VLM. Our contributions are: 1) a multilingual image-text pretraining dataset in eight languages, based on the LLaVA pretraining dataset; and 2) a multilingual image-text model supporting these languages, enhancing cultural and linguistic comprehension in vision-language tasks. Code available at https://github.com/nahidalam/maya.
>
---
#### [new 019] Improving the Reliability of LLMs: Combining CoT, RAG, Self-Consistency, and Self-Verification
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对大语言模型（LLMs）生成内容存在幻觉（不准确/无关信息）的问题，研究结合思维链（CoT）、检索增强（RAG）与自洽性、自我验证策略，提升复杂任务中的可靠性与事实准确性。通过融合外部知识验证和多策略协同优化，实验比较了不同方法的有效性，最终提出综合解决方案以平衡准确度与推理深度。**

- **链接: [http://arxiv.org/pdf/2505.09031v1](http://arxiv.org/pdf/2505.09031v1)**

> **作者:** Adarsh Kumar; Hwiyoon Kim; Jawahar Sai Nathani; Neil Roy
>
> **摘要:** Hallucination, where large language models (LLMs) generate confident but incorrect or irrelevant information, remains a key limitation in their application to complex, open-ended tasks. Chain-of-thought (CoT) prompting has emerged as a promising method for improving multistep reasoning by guiding models through intermediate steps. However, CoT alone does not fully address the hallucination problem. In this work, we investigate how combining CoT with retrieval-augmented generation (RAG), as well as applying self-consistency and self-verification strategies, can reduce hallucinations and improve factual accuracy. By incorporating external knowledge sources during reasoning and enabling models to verify or revise their own outputs, we aim to generate more accurate and coherent responses. We present a comparative evaluation of baseline LLMs against CoT, CoT+RAG, self-consistency, and self-verification techniques. Our results highlight the effectiveness of each method and identify the most robust approach for minimizing hallucinations while preserving fluency and reasoning depth.
>
---
#### [new 020] Performance Gains of LLMs With Humans in a World of LLMs Versus Humans
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于人机协作研究，针对当前"LLM vs人类"对比范式的问题，提出需转向"LLM与人类共生"策略。解决LLM快速迭代威胁临床安全体系的问题，主张构建可持续的医疗LLM安全应用框架，强调协作而非对抗的研究方向。**

- **链接: [http://arxiv.org/pdf/2505.08902v1](http://arxiv.org/pdf/2505.08902v1)**

> **作者:** Lucas McCullum; Pelagie Ami Agassi; Leo Anthony Celi; Daniel K. Ebner; Chrystinne Oliveira Fernandes; Rachel S. Hicklen; Mkliwa Koumbia; Lisa Soleymani Lehmann; David Restrepo
>
> **摘要:** Currently, a considerable research effort is devoted to comparing LLMs to a group of human experts, where the term "expert" is often ill-defined or variable, at best, in a state of constantly updating LLM releases. Without proper safeguards in place, LLMs will threaten to cause harm to the established structure of safe delivery of patient care which has been carefully developed throughout history to keep the safety of the patient at the forefront. A key driver of LLM innovation is founded on community research efforts which, if continuing to operate under "humans versus LLMs" principles, will expedite this trend. Therefore, research efforts moving forward must focus on effectively characterizing the safe use of LLMs in clinical settings that persist across the rapid development of novel LLM models. In this communication, we demonstrate that rather than comparing LLMs to humans, there is a need to develop strategies enabling efficient work of humans with LLMs in an almost symbiotic manner.
>
---
#### [new 021] Prioritizing Image-Related Tokens Enhances Vision-Language Pre-Training
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文针对视觉语言预训练任务，解决传统方法因均匀处理所有文本标记导致的噪声拟合和幻觉问题。提出PRIOR方法，通过纯文本LLM生成重要性权重，在损失函数中优先学习图像相关标记，在两种模型架构中分别实现19%和8%性能提升，并展现出更强扩展性。**

- **链接: [http://arxiv.org/pdf/2505.08971v1](http://arxiv.org/pdf/2505.08971v1)**

> **作者:** Yangyi Chen; Hao Peng; Tong Zhang; Heng Ji
>
> **备注:** The code will be available at https://github.com/Yangyi-Chen/PRIOR
>
> **摘要:** In standard large vision-language models (LVLMs) pre-training, the model typically maximizes the joint probability of the caption conditioned on the image via next-token prediction (NTP); however, since only a small subset of caption tokens directly relates to the visual content, this naive NTP unintentionally fits the model to noise and increases the risk of hallucination. We present PRIOR, a simple vision-language pre-training approach that addresses this issue by prioritizing image-related tokens through differential weighting in the NTP loss, drawing from the importance sampling framework. PRIOR introduces a reference model-a text-only large language model (LLM) trained on the captions without image inputs, to weight each token based on its probability for LVLMs training. Intuitively, tokens that are directly related to the visual inputs are harder to predict without the image and thus receive lower probabilities from the text-only reference LLM. During training, we implement a token-specific re-weighting term based on the importance scores to adjust each token's loss. We implement PRIOR in two distinct settings: LVLMs with visual encoders and LVLMs without visual encoders. We observe 19% and 8% average relative improvement, respectively, on several vision-language benchmarks compared to NTP. In addition, PRIOR exhibits superior scaling properties, as demonstrated by significantly higher scaling coefficients, indicating greater potential for performance gains compared to NTP given increasing compute and data.
>
---
#### [new 022] Ornithologist: Towards Trustworthy "Reasoning" about Central Bank Communications
- **分类: econ.GN; cs.CL; q-fin.EC; J.4; I.2.7**

- **简介: 该论文属于文本分类任务，旨在可信地分析央行文本的政策倾向（鹰派/鸽派）。为解决传统方法监督需求高、透明度低的问题，作者开发了Ornithologist系统，结合人类决策树与语言模型，降低幻觉风险并提升可解释性，可泛化应用于其他文本分析，且预测利率路径有效。**

- **链接: [http://arxiv.org/pdf/2505.09083v1](http://arxiv.org/pdf/2505.09083v1)**

> **作者:** Dominic Zaun Eu Jones
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** I develop Ornithologist, a weakly-supervised textual classification system and measure the hawkishness and dovishness of central bank text. Ornithologist uses ``taxonomy-guided reasoning'', guiding a large language model with human-authored decision trees. This increases the transparency and explainability of the system and makes it accessible to non-experts. It also reduces hallucination risk. Since it requires less supervision than traditional classification systems, it can more easily be applied to other problems or sources of text (e.g. news) without much modification. Ornithologist measurements of hawkishness and dovishness of RBA communication carry information about the future of the cash rate path and of market expectations.
>
---
#### [new 023] Language Agents Mirror Human Causal Reasoning Biases. How Can We Help Them Think Like Scientists?
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究语言模型（LM）的因果推理能力，属于认知评估与改进任务。发现LM存在与人类相似的析因偏见（倾向析取关系，忽视合取关系），影响科学推理。通过Blicket Test验证跨模型一致性，提出假设采样方法有效降低偏差，推动LM更科学的因果推断。**

- **链接: [http://arxiv.org/pdf/2505.09614v1](http://arxiv.org/pdf/2505.09614v1)**

> **作者:** Anthony GX-Chen; Dongyan Lin; Mandana Samiei; Doina Precup; Blake A. Richards; Rob Fergus; Kenneth Marino
>
> **摘要:** Language model (LM) agents are increasingly used as autonomous decision-makers who need to actively gather information to guide their decisions. A crucial cognitive skill for such agents is the efficient exploration and understanding of the causal structure of the world -- key to robust, scientifically grounded reasoning. Yet, it remains unclear whether LMs possess this capability or exhibit systematic biases leading to erroneous conclusions. In this work, we examine LMs' ability to explore and infer causal relationships, using the well-established "Blicket Test" paradigm from developmental psychology. We find that LMs reliably infer the common, intuitive disjunctive causal relationships but systematically struggle with the unusual, yet equally (or sometimes even more) evidenced conjunctive ones. This "disjunctive bias" persists across model families, sizes, and prompting strategies, and performance further declines as task complexity increases. Interestingly, an analogous bias appears in human adults, suggesting that LMs may have inherited deep-seated reasoning heuristics from their training data. To this end, we quantify similarities between LMs and humans, finding that LMs exhibit adult-like inference profiles (but not children-like). Finally, we propose a test-time sampling method which explicitly samples and eliminates hypotheses about causal relationships from the LM. This scalable approach significantly reduces the disjunctive bias and moves LMs closer to the goal of scientific, causally rigorous reasoning.
>
---
#### [new 024] The Geometry of Meaning: Perfect Spacetime Representations of Hierarchical Structures
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究层次结构的几何表示，提出一种基于三维Minkowski时空的快速嵌入算法，仅通过局部层次信号（定向标记对）将离散数据（如WordNet语义网络）的因果关系编码到时空中。解决了传统方法依赖全局结构的问题，实现了包含歧义节点和超8万名词的完美几何映射，并通过因果检索机制验证了层次意义本质上是几何的，揭示了与相对论和场论的潜在联系。**

- **链接: [http://arxiv.org/pdf/2505.08795v1](http://arxiv.org/pdf/2505.08795v1)**

> **作者:** Andres Anabalon; Hugo Garces; Julio Oliva; Jose Cifuentes
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** We show that there is a fast algorithm that embeds hierarchical structures in three-dimensional Minkowski spacetime. The correlation of data ends up purely encoded in the causal structure. Our model relies solely on oriented token pairs -- local hierarchical signals -- with no access to global symbolic structure. We apply our method to the corpus of \textit{WordNet}. We provide a perfect embedding of the mammal sub-tree including ambiguities (more than one hierarchy per node) in such a way that the hierarchical structures get completely codified in the geometry and exactly reproduce the ground-truth. We extend this to a perfect embedding of the maximal unambiguous subset of the \textit{WordNet} with 82{,}115 noun tokens and a single hierarchy per token. We introduce a novel retrieval mechanism in which causality, not distance, governs hierarchical access. Our results seem to indicate that all discrete data has a perfect geometrical representation that is three-dimensional. The resulting embeddings are nearly conformally invariant, indicating deep connections with general relativity and field theory. These results suggest that concepts, categories, and their interrelations, namely hierarchical meaning itself, is geometric.
>
---
#### [new 025] LibVulnWatch: A Deep Assessment Agent System and Leaderboard for Uncovering Hidden Vulnerabilities in Open-Source AI Libraries
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出LibVulnWatch图式代理系统，针对开源AI库的潜在安全漏洞、合规风险等问题，构建多代理协作框架进行深度风险评估。通过整合仓库、漏洞数据库等可信数据源，生成可复现的治理评分并建立公开排行榜，覆盖超80%现有检测标准，发现每库多达19项新风险（如RCE漏洞、许可证缺陷），实现AI供应链风险的持续监测与量化评估。**

- **链接: [http://arxiv.org/pdf/2505.08842v1](http://arxiv.org/pdf/2505.08842v1)**

> **作者:** Zekun Wu; Seonglae Cho; Umar Mohammed; Cristian Munoz; Kleyton Costa; Xin Guan; Theo King; Ze Wang; Emre Kazim; Adriano Koshiyama
>
> **摘要:** Open-source AI libraries are foundational to modern AI systems but pose significant, underexamined risks across security, licensing, maintenance, supply chain integrity, and regulatory compliance. We present LibVulnWatch, a graph-based agentic assessment framework that performs deep, source-grounded evaluations of these libraries. Built on LangGraph, the system coordinates a directed acyclic graph of specialized agents to extract, verify, and quantify risk using evidence from trusted sources such as repositories, documentation, and vulnerability databases. LibVulnWatch generates reproducible, governance-aligned scores across five critical domains, publishing them to a public leaderboard for longitudinal ecosystem monitoring. Applied to 20 widely used libraries, including ML frameworks, LLM inference engines, and agent orchestration tools, our system covers up to 88% of OpenSSF Scorecard checks while uncovering up to 19 additional risks per library. These include critical Remote Code Execution (RCE) vulnerabilities, absent Software Bills of Materials (SBOMs), licensing constraints, undocumented telemetry, and widespread gaps in regulatory documentation and auditability. By translating high-level governance principles into practical, verifiable metrics, LibVulnWatch advances technical AI governance with a scalable, transparent mechanism for continuous supply chain risk assessment and informed library selection.
>
---
#### [new 026] Customizing a Large Language Model for VHDL Design of High-Performance Microprocessors
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理应用任务，针对高性能处理器设计中VHDL代码解释需求，定制大语言模型。通过扩展预训练、构建专用测试集及专家评估，开发了指令调优模型，将专家认可率从43%提升至71%，并提出结合新基模型可突破85%。解决了现有LLM在VHDL领域支持不足的问题。**

- **链接: [http://arxiv.org/pdf/2505.09610v1](http://arxiv.org/pdf/2505.09610v1)**

> **作者:** Nicolas Dupuis; Ravi Nair; Shyam Ramji; Sean McClintock; Nishant Chauhan; Priyanka Nagpal; Bart Blaner; Ken Valk; Leon Stok; Ruchir Puri
>
> **摘要:** The use of Large Language Models (LLMs) in hardware design has taken off in recent years, principally through its incorporation in tools that increase chip designer productivity. There has been considerable discussion about the use of LLMs in RTL specifications of chip designs, for which the two most popular languages are Verilog and VHDL. LLMs and their use in Verilog design has received significant attention due to the higher popularity of the language, but little attention so far has been given to VHDL despite its continued popularity in the industry. There has also been little discussion about the unique needs of organizations that engage in high-performance processor design, and techniques to deploy AI solutions in these settings. In this paper, we describe our journey in developing a Large Language Model (LLM) specifically for the purpose of explaining VHDL code, a task that has particular importance in an organization with decades of experience and assets in high-performance processor design. We show how we developed test sets specific to our needs and used them for evaluating models as we performed extended pretraining (EPT) of a base LLM. Expert evaluation of the code explanations produced by the EPT model increased to 69% compared to a base model rating of 43%. We further show how we developed an LLM-as-a-judge to gauge models similar to expert evaluators. This led us to deriving and evaluating a host of new models, including an instruction-tuned version of the EPT model with an expected expert evaluator rating of 71%. Our experiments also indicate that with the potential use of newer base models, this rating can be pushed to 85% and beyond. We conclude with a discussion on further improving the quality of hardware design LLMs using exciting new developments in the Generative AI world.
>
---
#### [new 027] An Extra RMSNorm is All You Need for Fine Tuning to 1.58 Bits
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型量化任务，旨在解决大语言模型低比特量化（1.58位）时精度损失和不稳定的问题。通过在每个线性层前添加RMS归一化，结合分层渐进量化策略，将全精度模型稳定微调为三值模型，无需复杂蒸馏即可保持性能，缩小低比特与全精度模型的精度差距。**

- **链接: [http://arxiv.org/pdf/2505.08823v1](http://arxiv.org/pdf/2505.08823v1)**

> **作者:** Cody Steinmetz; Gavin Childress; Aaron Herbst; Gavin Jones; Jasdeep Singh; Eli Vang; Keagan Weinstock
>
> **摘要:** Large language models (LLMs) have transformed natural-language processing, yet their scale makes real-world deployment costly. Post-training quantization reduces memory and computation but often degrades accuracy, while quantization-aware training can recover performance at the cost of extra training. Pushing quantization to the ternary (2-bit) regime yields even larger savings but is notoriously unstable. Building on recent work showing that a bias-free, RMS-normalized Transformer with straight-through estimation can reach 1.58-bit precision, we demonstrate that simply inserting RMS normalization before every linear projection and applying a gradual, layer-wise quantization schedule stably fine-tunes full-precision checkpoints into ternary LLMs. Our approach matches or surpasses more elaborate knowledge-distillation pipelines on standard language-modeling benchmarks without adding model complexity. These results indicate that careful normalization alone can close much of the accuracy gap between ternary and full-precision LLMs, making ultra-low-bit inference practical.
>
---
#### [new 028] ForeCite: Adapting Pre-Trained Language Models to Predict Future Citation Rates of Academic Papers
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于回归预测任务，旨在解决学术论文长期影响力评估问题。通过为预训练因果语言模型添加线性预测头，构建ForeCite框架预测论文未来月均引用率。在90万+生物医学论文数据集上，其测试相关系数达0.826（较先前提升27%），并通过规模分析和时序验证证明有效性，揭示了模型对标题/摘要的过度依赖特征。**

- **链接: [http://arxiv.org/pdf/2505.08941v1](http://arxiv.org/pdf/2505.08941v1)**

> **作者:** Gavin Hull; Alex Bihlo
>
> **备注:** 16 pages, 13 figures
>
> **摘要:** Predicting the future citation rates of academic papers is an important step toward the automation of research evaluation and the acceleration of scientific progress. We present $\textbf{ForeCite}$, a simple but powerful framework to append pre-trained causal language models with a linear head for average monthly citation rate prediction. Adapting transformers for regression tasks, ForeCite achieves a test correlation of $\rho = 0.826$ on a curated dataset of 900K+ biomedical papers published between 2000 and 2024, a 27-point improvement over the previous state-of-the-art. Comprehensive scaling-law analysis reveals consistent gains across model sizes and data volumes, while temporal holdout experiments confirm practical robustness. Gradient-based saliency heatmaps suggest a potentially undue reliance on titles and abstract texts. These results establish a new state-of-the-art in forecasting the long-term influence of academic research and lay the groundwork for the automated, high-fidelity evaluation of scientific contributions.
>
---
#### [new 029] Focus, Merge, Rank: Improved Question Answering Based on Semi-structured Knowledge Bases
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于多跳问答任务，旨在解决结合结构化和非结构化知识的难题。提出FocusedRetriever框架，通过整合向量搜索、LLM生成查询与重排序组件，利用半结构化知识库实现知识关联，在STaRK基准中以25.7%优势超越现有方法，提升答案检索准确率。**

- **链接: [http://arxiv.org/pdf/2505.09246v1](http://arxiv.org/pdf/2505.09246v1)**

> **作者:** Derian Boer; Stephen Roth; Stefan Kramer
>
> **摘要:** In many real-world settings, machine learning models and interactive systems have access to both structured knowledge, e.g., knowledge graphs or tables, and unstructured content, e.g., natural language documents. However, most rely on either. Semi-Structured Knowledge Bases (SKBs) bridge this gap by linking unstructured content to nodes within structured data, thereby enabling new strategies for knowledge access and use. In this work, we present FocusedRetriever, a modular SKB-based framework for multi-hop question answering. It integrates components (VSS-based entity search, LLM-based generation of Cypher queries and pairwise re-ranking) in a way that enables it to outperform state-of-the-art methods across all three STaRK benchmark test sets, covering diverse domains and multiple performance metrics. The average first-hit rate exceeds that of the second-best method by 25.7%. FocusedRetriever leverages (1) the capacity of Large Language Models (LLMs) to extract relational facts and entity attributes from unstructured text, (2) node set joins to filter answer candidates based on these extracted triplets and constraints, (3) vector similarity search to retrieve and rank relevant unstructured content, and (4) the contextual capabilities of LLMs to finally rank the top-k answers. For generality, we only incorporate base LLMs in FocusedRetriever in our evaluation. However, our analysis of intermediate results highlights several opportunities for further upgrades including finetuning. The source code is publicly available at https://github.com/kramerlab/FocusedRetriever .
>
---
#### [new 030] CXMArena: Unified Dataset to benchmark performance in realistic CXM Scenarios
- **分类: cs.LG; cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出CXMArena——面向客户体验管理（CXM）的合成基准数据集，解决现有评估方法因数据隐私、缺乏真实场景（如知识库整合、噪声干扰）导致的AI模型实用性验证难题。通过LLM生成模拟品牌实体数据（知识库、对话），注入专家验证的噪声，构建涵盖知识优化、意图预测等5个核心任务的评测体系，揭示当前模型在复杂操作任务中的显著性能瓶颈。**

- **链接: [http://arxiv.org/pdf/2505.09436v1](http://arxiv.org/pdf/2505.09436v1)**

> **作者:** Raghav Garg; Kapil Sharma; Karan Gupta
>
> **摘要:** Large Language Models (LLMs) hold immense potential for revolutionizing Customer Experience Management (CXM), particularly in contact center operations. However, evaluating their practical utility in complex operational environments is hindered by data scarcity (due to privacy concerns) and the limitations of current benchmarks. Existing benchmarks often lack realism, failing to incorporate deep knowledge base (KB) integration, real-world noise, or critical operational tasks beyond conversational fluency. To bridge this gap, we introduce CXMArena, a novel, large-scale synthetic benchmark dataset specifically designed for evaluating AI in operational CXM contexts. Given the diversity in possible contact center features, we have developed a scalable LLM-powered pipeline that simulates the brand's CXM entities that form the foundation of our datasets-such as knowledge articles including product specifications, issue taxonomies, and contact center conversations. The entities closely represent real-world distribution because of controlled noise injection (informed by domain experts) and rigorous automated validation. Building on this, we release CXMArena, which provides dedicated benchmarks targeting five important operational tasks: Knowledge Base Refinement, Intent Prediction, Agent Quality Adherence, Article Search, and Multi-turn RAG with Integrated Tools. Our baseline experiments underscore the benchmark's difficulty: even state of the art embedding and generation models achieve only 68% accuracy on article search, while standard embedding methods yield a low F1 score of 0.3 for knowledge base refinement, highlighting significant challenges for current models necessitating complex pipelines and solutions over conventional techniques.
>
---
#### [new 031] Grounding Synthetic Data Evaluations of Language Models in Unsupervised Document Corpora
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出自动化构建基于事实的合成数据评估方法，解决人工评测基准效率低、覆盖不足的问题。通过无监督文档自动生成领域知识评测问题，验证其与人工评测高度相关（Spearman 0.96），支持多题型诊断模型能力，并应用于检测Gemma3模型的意外高性能。**

- **链接: [http://arxiv.org/pdf/2505.08905v1](http://arxiv.org/pdf/2505.08905v1)**

> **作者:** Michael Majurski; Cynthia Matuszek
>
> **摘要:** Language Models (LMs) continue to advance, improving response quality and coherence. Given Internet-scale training datasets, LMs have likely encountered much of what users might ask them to generate in some form during their training. A plethora of evaluation benchmarks have been constructed to assess model quality, response appropriateness, and reasoning capabilities. However, the human effort required for benchmark construction is limited and being rapidly outpaced by the size and scope of the models under evaluation. Additionally, having humans build a benchmark for every possible domain of interest is impractical. Therefore, we propose a methodology for automating the construction of fact-based synthetic data model evaluations grounded in document populations. This work leverages those very same LMs to evaluate domain-specific knowledge automatically, using only grounding documents (e.g., a textbook) as input. This synthetic data benchmarking approach corresponds well with human curated questions with a Spearman ranking correlation of 0.96 and a benchmark evaluation Pearson accuracy correlation of 0.79. This novel tool supports generating both multiple choice and open-ended synthetic data questions to gain diagnostic insight of LM capability. We apply this methodology to evaluate model performance on a recent relevant arXiv preprint, discovering a surprisingly strong performance from Gemma3 models.
>
---
## 更新

#### [replaced 001] Principled Data Selection for Alignment: The Hidden Risks of Difficult Examples
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.09650v2](http://arxiv.org/pdf/2502.09650v2)**

> **作者:** Chengqian Gao; Haonan Li; Liu Liu; Zeke Xie; Peilin Zhao; Zhiqiang Xu
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** The alignment of large language models (LLMs) often assumes that using more clean data yields better outcomes, overlooking the match between model capacity and example difficulty. Challenging this, we propose a new principle: Preference data vary in difficulty, and overly difficult examples hinder alignment, by exceeding the model's capacity. Through systematic experimentation, we validate this principle with three key findings: (1) preference examples vary in difficulty, as evidenced by consistent learning orders across alignment runs; (2) overly difficult examples significantly degrade performance across four LLMs and two datasets; and (3) the capacity of a model dictates its threshold for handling difficult examples, underscoring a critical relationship between data selection and model capacity. Building on this principle, we introduce Selective DPO, which filters out overly difficult examples. This simple adjustment improves alignment performance by 9-16% in win rates on the AlpacaEval 2 benchmark compared to the DPO baseline, suppressing a series of DPO variants with different algorithmic adjustments. Together, these results illuminate the importance of aligning data difficulty with model capacity, offering a transformative perspective for improving alignment strategies in LLMs. Code is available at https://github.com/glorgao/SelectiveDPO.
>
---
#### [replaced 002] Activation Steering in Neural Theorem Provers
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15507v3](http://arxiv.org/pdf/2502.15507v3)**

> **作者:** Shashank Kirtania
>
> **备注:** incorrect explanation for a concept, need to revise and update!
>
> **摘要:** Large Language Models (LLMs) have shown promise in proving formal theorems using proof assistants like Lean. However, current state of the art language models struggles to predict next step in proofs leading practitioners to use different sampling techniques to improve LLMs capabilities. We observe that the LLM is capable of predicting the correct tactic; however, it faces challenges in ranking it appropriately within the set of candidate tactics, affecting the overall selection process. To overcome this hurdle, we use activation steering to guide LLMs responses to improve the generations at the time of inference. Our results suggest that activation steering offers a promising lightweight alternative to specialized fine-tuning for enhancing theorem proving capabilities in LLMs, particularly valuable in resource-constrained environments.
>
---
#### [replaced 003] What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.03343v2](http://arxiv.org/pdf/2411.03343v2)**

> **作者:** Nathalie Kirch; Constantin Weisser; Severin Field; Helen Yannakoudakis; Stephen Casper
>
> **摘要:** Jailbreaks have been a central focus of research regarding the safety and reliability of large language models (LLMs), yet the mechanisms underlying these attacks remain poorly understood. While previous studies have predominantly relied on linear methods to detect jailbreak attempts and model refusals, we take a different approach by examining both linear and non-linear features in prompts that lead to successful jailbreaks. First, we introduce a novel dataset comprising 10,800 jailbreak attempts spanning 35 diverse attack methods. Leveraging this dataset, we train probes to classify successful from unsuccessful jailbreaks using the latent representations corresponding to prompt tokens. Notably, we find that even when probes achieve high accuracy in predicting the success of jailbreaks, their performance often fails to generalize to unseen attack methods. This reveals that different jailbreaking strategies exploit different non-linear, non-universal features. Next, we demonstrate that non-linear probes provide a powerful tool for steering model behavior. Specifically, we use these probes to guide targeted latent space perturbations, enabling us to effectively modulate the model's robustness against jailbreaks. Overall, our findings challenge the assumption that jailbreaks can be fully understood through linear or simple universal prompt features alone, highlighting the importance of a nuanced understanding of the mechanisms behind LLM vulnerabilities.
>
---
#### [replaced 004] Embodied-Reasoner: Synergizing Visual Search, Reasoning, and Action for Embodied Interactive Tasks
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21696v2](http://arxiv.org/pdf/2503.21696v2)**

> **作者:** Wenqi Zhang; Mengna Wang; Gangao Liu; Xu Huixin; Yiwei Jiang; Yongliang Shen; Guiyang Hou; Zhe Zheng; Hang Zhang; Xin Li; Weiming Lu; Peng Li; Yueting Zhuang
>
> **备注:** Code: https://github.com/zwq2018/embodied_reasoner Dataset: https://huggingface.co/datasets/zwq2018/embodied_reasoner
>
> **摘要:** Recent advances in deep thinking models have demonstrated remarkable reasoning capabilities on mathematical and coding tasks. However, their effectiveness in embodied domains which require continuous interaction with environments through image action interleaved trajectories remains largely -unexplored. We present Embodied Reasoner, a model that extends o1 style reasoning to interactive embodied search tasks. Unlike mathematical reasoning that relies primarily on logical deduction, embodied scenarios demand spatial understanding, temporal reasoning, and ongoing self-reflection based on interaction history. To address these challenges, we synthesize 9.3k coherent Observation-Thought-Action trajectories containing 64k interactive images and 90k diverse thinking processes (analysis, spatial reasoning, reflection, planning, and verification). We develop a three-stage training pipeline that progressively enhances the model's capabilities through imitation learning, self-exploration via rejection sampling, and self-correction through reflection tuning. The evaluation shows that our model significantly outperforms those advanced visual reasoning models, e.g., it exceeds OpenAI o1, o3-mini, and Claude-3.7 by +9\%, 24\%, and +13\%. Analysis reveals our model exhibits fewer repeated searches and logical inconsistencies, with particular advantages in complex long-horizon tasks. Real-world environments also show our superiority while exhibiting fewer repeated searches and logical inconsistency cases.
>
---
#### [replaced 005] OAEI-LLM-T: A TBox Benchmark Dataset for Understanding Large Language Model Hallucinations in Ontology Matching
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2503.21813v3](http://arxiv.org/pdf/2503.21813v3)**

> **作者:** Zhangcheng Qiang; Kerry Taylor; Weiqing Wang; Jing Jiang
>
> **备注:** 14 pages, 4 figures, 4 tables, 2 prompt templates
>
> **摘要:** Hallucinations are often inevitable in downstream tasks using large language models (LLMs). To tackle the substantial challenge of addressing hallucinations for LLM-based ontology matching (OM) systems, we introduce a new benchmark dataset OAEI-LLM-T. The dataset evolves from seven TBox datasets in the Ontology Alignment Evaluation Initiative (OAEI), capturing hallucinations of ten different LLMs performing OM tasks. These OM-specific hallucinations are organised into two primary categories and six sub-categories. We showcase the usefulness of the dataset in constructing an LLM leaderboard for OM tasks and for fine-tuning LLMs used in OM tasks.
>
---
#### [replaced 006] An Analytical Emotion Framework of Rumour Threads on Social Media
- **分类: cs.AI; cs.CL; cs.SI**

- **链接: [http://arxiv.org/pdf/2502.16560v2](http://arxiv.org/pdf/2502.16560v2)**

> **作者:** Rui Xing; Boyang Sun; Kun Zhang; Preslav Nakov; Timothy Baldwin; Jey Han Lau
>
> **备注:** Accepted to ICWSM 2025 MisD Workshop
>
> **摘要:** Rumours in online social media pose significant risks to modern society, motivating the need for better understanding of how they develop. We focus specifically on the interface between emotion and rumours in threaded discourses, building on the surprisingly sparse literature on the topic which has largely focused on single aspect of emotions within the original rumour posts themselves, and largely overlooked the comparative differences between rumours and non-rumours. In this work, we take one step further to provide a comprehensive analytical emotion framework with multi-aspect emotion detection, contrasting rumour and non-rumour threads and provide both correlation and causal analysis of emotions. We applied our framework on existing widely-used rumour datasets to further understand the emotion dynamics in online social media threads. Our framework reveals that rumours trigger more negative emotions (e.g., anger, fear, pessimism), while non-rumours evoke more positive ones. Emotions are contagious, rumours spread negativity, non-rumours spread positivity. Causal analysis shows surprise bridges rumours and other emotions; pessimism comes from sadness and fear, while optimism arises from joy and love.
>
---
#### [replaced 007] Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.05084v2](http://arxiv.org/pdf/2505.05084v2)**

> **作者:** Xiaowei Zhu; Yubing Ren; Yanan Cao; Xixun Lin; Fang Fang; Yangxi Li
>
> **摘要:** The rapid advancement of large language models has raised significant concerns regarding their potential misuse by malicious actors. As a result, developing effective detectors to mitigate these risks has become a critical priority. However, most existing detection methods focus excessively on detection accuracy, often neglecting the societal risks posed by high false positive rates (FPRs). This paper addresses this issue by leveraging Conformal Prediction (CP), which effectively constrains the upper bound of FPRs. While directly applying CP constrains FPRs, it also leads to a significant reduction in detection performance. To overcome this trade-off, this paper proposes a Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction (MCP), which both enforces the FPR constraint and improves detection performance. This paper also introduces RealDet, a high-quality dataset that spans a wide range of domains, ensuring realistic calibration and enabling superior detection performance when combined with MCP. Empirical evaluations demonstrate that MCP effectively constrains FPRs, significantly enhances detection performance, and increases robustness against adversarial attacks across multiple detectors and datasets.
>
---
#### [replaced 008] Is analogy enough to draw novel adjective-noun inferences?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.24293v2](http://arxiv.org/pdf/2503.24293v2)**

> **作者:** Hayley Ross; Kathryn Davidson; Najoung Kim
>
> **备注:** 9 pages (17 pages with appendix). Accepted to SCiL 2025
>
> **摘要:** Recent work (Ross et al., 2025, 2024) has argued that the ability of humans and LLMs respectively to generalize to novel adjective-noun combinations shows that they each have access to a compositional mechanism to determine the phrase's meaning and derive inferences. We study whether these inferences can instead be derived by analogy to known inferences, without need for composition. We investigate this by (1) building a model of analogical reasoning using similarity over lexical items, and (2) asking human participants to reason by analogy. While we find that this strategy works well for a large proportion of the dataset of Ross et al. (2025), there are novel combinations for which both humans and LLMs derive convergent inferences but which are not well handled by analogy. We thus conclude that the mechanism humans and LLMs use to generalize in these cases cannot be fully reduced to analogy, and likely involves composition.
>
---
#### [replaced 009] Simulating and Analysing Human Survey Responses with Large Language Models: A Case Study in Energy Stated Preference
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2503.10652v2](http://arxiv.org/pdf/2503.10652v2)**

> **作者:** Han Wang; Jacek Pawlak; Aruna Sivakumar
>
> **摘要:** Survey research plays a crucial role in studies by capturing consumer preferences and informing policy decisions. Stated preference (SP) surveys help researchers understand how individuals make trade-offs in hypothetical, potentially futuristic, scenarios. However, traditional methods are costly, time-consuming, and affected by respondent fatigue and ethical constraints. Large language models (LLMs) have shown remarkable capabilities in generating human-like responses, prompting interest in their use in survey research. This study investigates LLMs for simulating consumer choices in energy-related SP surveys and explores their integration into data collection and analysis workflows. Test scenarios were designed to assess the simulation performance of several LLMs (LLaMA 3.1, Mistral, GPT-3.5, DeepSeek-R1) at individual and aggregated levels, considering prompt design, in-context learning (ICL), chain-of-thought (CoT) reasoning, model types, integration with traditional choice models, and potential biases. While LLMs achieve accuracy above random guessing, performance remains insufficient for practical simulation use. Cloud-based LLMs do not consistently outperform smaller local models. DeepSeek-R1 achieves the highest average accuracy (77%) and outperforms non-reasoning LLMs in accuracy, factor identification, and choice distribution alignment. Previous SP choices are the most effective input; longer prompts with more factors reduce accuracy. Mixed logit models can support LLM prompt refinement. Reasoning LLMs show potential in data analysis by indicating factor significance, offering a qualitative complement to statistical models. Despite limitations, pre-trained LLMs offer scalability and require minimal historical data. Future work should refine prompts, further explore CoT reasoning, and investigate fine-tuning techniques.
>
---
#### [replaced 010] InductionBench: LLMs Fail in the Simplest Complexity Class
- **分类: cs.LG; cs.AI; cs.CL; cs.FL**

- **链接: [http://arxiv.org/pdf/2502.15823v4](http://arxiv.org/pdf/2502.15823v4)**

> **作者:** Wenyue Hua; Tyler Wong; Sun Fei; Liangming Pan; Adam Jardine; William Yang Wang
>
> **备注:** 25 pages, 10 figures, more details including examples and prompts are added
>
> **摘要:** Large language models (LLMs) have shown remarkable improvements in reasoning and many existing benchmarks have been addressed by models such as o1 and o3 either fully or partially. However, a majority of these benchmarks emphasize deductive reasoning, including mathematical and coding tasks in which rules such as mathematical axioms or programming syntax are clearly defined, based on which LLMs can plan and apply these rules to arrive at a solution. In contrast, inductive reasoning, where one infers the underlying rules from observed data, remains less explored. Such inductive processes lie at the heart of scientific discovery, as they enable researchers to extract general principles from empirical observations. To assess whether LLMs possess this capacity, we introduce InductionBench, a new benchmark designed to evaluate the inductive reasoning ability of LLMs. Our experimental findings reveal that even the most advanced models available struggle to master the simplest complexity classes within the subregular hierarchy of functions, highlighting a notable deficiency in current LLMs' inductive reasoning capabilities. Coda and data are available https://github.com/Wenyueh/inductive_reasoning_benchmark.
>
---
#### [replaced 011] TiSpell: A Semi-Masked Methodology for Tibetan Spelling Correction covering Multi-Level Error with Data Augmentation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.08037v2](http://arxiv.org/pdf/2505.08037v2)**

> **作者:** Yutong Liu; Feng Xiao; Ziyue Zhang; Yongbin Yu; Cheng Huang; Fan Gao; Xiangxiang Wang; Ma-bao Ban; Manping Fan; Thupten Tsering; Cheng Huang; Gadeng Luosang; Renzeng Duojie; Nyima Tashi
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Multi-level Tibetan spelling correction addresses errors at both the character and syllable levels within a unified model. Existing methods focus mainly on single-level correction and lack effective integration of both levels. Moreover, there are no open-source datasets or augmentation methods tailored for this task in Tibetan. To tackle this, we propose a data augmentation approach using unlabeled text to generate multi-level corruptions, and introduce TiSpell, a semi-masked model capable of correcting both character- and syllable-level errors. Although syllable-level correction is more challenging due to its reliance on global context, our semi-masked strategy simplifies this process. We synthesize nine types of corruptions on clean sentences to create a robust training set. Experiments on both simulated and real-world data demonstrate that TiSpell, trained on our dataset, outperforms baseline models and matches the performance of state-of-the-art approaches, confirming its effectiveness.
>
---
#### [replaced 012] FAMMA: A Benchmark for Financial Domain Multilingual Multimodal Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.04526v3](http://arxiv.org/pdf/2410.04526v3)**

> **作者:** Siqiao Xue; Xiaojing Li; Fan Zhou; Qingyang Dai; Zhixuan Chu; Hongyuan Mei
>
> **摘要:** In this paper, we introduce FAMMA, an open-source benchmark for \underline{f}in\underline{a}ncial \underline{m}ultilingual \underline{m}ultimodal question \underline{a}nswering (QA). Our benchmark aims to evaluate the abilities of large language models (LLMs) in answering complex reasoning questions that require advanced financial knowledge. The benchmark has two versions: FAMMA-Basic consists of 1,945 questions extracted from university textbooks and exams, along with human-annotated answers and rationales; FAMMA-LivePro consists of 103 novel questions created by human domain experts, with answers and rationales held out from the public for a contamination-free evaluation. These questions cover advanced knowledge of 8 major subfields in finance (e.g., corporate finance, derivatives, and portfolio management). Some are in Chinese or French, while a majority of them are in English. Each question has some non-text data such as charts, diagrams, or tables. Our experiments reveal that FAMMA poses a significant challenge on LLMs, including reasoning models such as GPT-o1 and DeepSeek-R1. Additionally, we curated 1,270 reasoning trajectories of DeepSeek-R1 on the FAMMA-Basic data, and fine-tuned a series of open-source Qwen models using this reasoning data. We found that training a model on these reasoning trajectories can significantly improve its performance on FAMMA-LivePro. We released our leaderboard, data, code, and trained models at https://famma-bench.github.io/famma/.
>
---
#### [replaced 013] PropNet: a White-Box and Human-Like Network for Sentence Representation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.10725v3](http://arxiv.org/pdf/2502.10725v3)**

> **作者:** Fei Yang
>
> **备注:** Clarified some ambiguities in the previous version
>
> **摘要:** Transformer-based embedding methods have dominated the field of sentence representation in recent years. Although they have achieved remarkable performance on NLP missions, such as semantic textual similarity (STS) tasks, their black-box nature and large-data-driven training style have raised concerns, including issues related to bias, trust, and safety. Many efforts have been made to improve the interpretability of embedding models, but these problems have not been fundamentally resolved. To achieve inherent interpretability, we propose a purely white-box and human-like sentence representation network, PropNet. Inspired by findings from cognitive science, PropNet constructs a hierarchical network based on the propositions contained in a sentence. While experiments indicate that PropNet has a significant gap compared to state-of-the-art (SOTA) embedding models in STS tasks, case studies reveal substantial room for improvement. Additionally, PropNet enables us to analyze and understand the human cognitive processes underlying STS benchmarks.
>
---
#### [replaced 014] TSLFormer: A Lightweight Transformer Model for Turkish Sign Language Recognition Using Skeletal Landmarks
- **分类: cs.CL; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.07890v2](http://arxiv.org/pdf/2505.07890v2)**

> **作者:** Kutay Ertürk; Furkan Altınışık; İrem Sarıaltın; Ömer Nezih Gerek
>
> **摘要:** This study presents TSLFormer, a light and robust word-level Turkish Sign Language (TSL) recognition model that treats sign gestures as ordered, string-like language. Instead of using raw RGB or depth videos, our method only works with 3D joint positions - articulation points - extracted using Google's Mediapipe library, which focuses on the hand and torso skeletal locations. This creates efficient input dimensionality reduction while preserving important semantic gesture information. Our approach revisits sign language recognition as sequence-to-sequence translation, inspired by the linguistic nature of sign languages and the success of transformers in natural language processing. Since TSLFormer uses the self-attention mechanism, it effectively captures temporal co-occurrence within gesture sequences and highlights meaningful motion patterns as words unfold. Evaluated on the AUTSL dataset with over 36,000 samples and 227 different words, TSLFormer achieves competitive performance with minimal computational cost. These results show that joint-based input is sufficient for enabling real-time, mobile, and assistive communication systems for hearing-impaired individuals.
>
---
#### [replaced 015] FAS: Fast ANN-SNN Conversion for Spiking Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.04405v2](http://arxiv.org/pdf/2502.04405v2)**

> **作者:** Long Chen; Xiaotian Song; Andy Song; BaDong Chen; Jiancheng Lv; Yanan Sun
>
> **摘要:** Spiking Large Language Models have been shown as a good alternative to LLMs in various scenarios. Existing methods for creating Spiking LLMs, i.e., direct training and ANN-SNN conversion, often suffer from performance degradation and relatively high computational costs. To address these issues, we propose a novel Fast ANN-SNN conversion strategy (FAS) that transforms LLMs into spiking LLMs in two stages. The first stage employs a full-parameter fine-tuning of pre-trained models, so it does not need any direct training from scratch. The second stage introduces a coarse-to-fine calibration method to reduce conversion errors and improve accuracy. Experiments on both language and vision-language tasks across four different scales of LLMs demonstrate that FAS can achieve state-of-the-art performance yet with significantly reduced inference latency and computational costs. Notably, FAS only takes eight timesteps to achieve an accuracy of 3\% higher than that of the OPT-7B model, while reducing energy consumption by 96.63\%. The source code is available at https://github.com/lc783/FAS
>
---
#### [replaced 016] Beyond Single-Turn: A Survey on Multi-Turn Interactions with Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.04717v4](http://arxiv.org/pdf/2504.04717v4)**

> **作者:** Yubo Li; Xiaobin Shen; Xinyu Yao; Xueying Ding; Yidi Miao; Ramayya Krishnan; Rema Padman
>
> **摘要:** Recent advancements in large language models (LLMs) have revolutionized their ability to handle single-turn tasks, yet real-world applications demand sophisticated multi-turn interactions. This survey provides a comprehensive review of recent advancements in evaluating and enhancing multi-turn interactions in LLMs. Focusing on task-specific scenarios, from instruction following in diverse domains such as math and coding to complex conversational engagements in roleplay, healthcare, education, and even adversarial jailbreak settings, we systematically examine the challenges of maintaining context, coherence, fairness, and responsiveness over prolonged dialogues. The paper organizes current benchmarks and datasets into coherent categories that reflect the evolving landscape of multi-turn dialogue evaluation. In addition, we review a range of enhancement methodologies under multi-turn settings, including model-centric strategies (contextual learning, supervised fine-tuning, reinforcement learning, and new architectures), external integration approaches (memory-augmented, retrieval-based methods, and knowledge graph), and agent-based techniques for collaborative interactions. Finally, we discuss open challenges and propose future directions for research to further advance the robustness and effectiveness of multi-turn interactions in LLMs. Related resources and papers are available at https://github.com/yubol-cmu/Awesome-Multi-Turn-LLMs.
>
---
#### [replaced 017] Evaluating Clinical Competencies of Large Language Models with a General Practice Benchmark
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.17599v2](http://arxiv.org/pdf/2503.17599v2)**

> **作者:** Zheqing Li; Yiying Yang; Jiping Lang; Wenhao Jiang; Yuhang Zhao; Shuang Li; Dingqian Wang; Zhu Lin; Xuanna Li; Yuze Tang; Jiexian Qiu; Xiaolin Lu; Hongji Yu; Shuang Chen; Yuhua Bi; Xiaofei Zeng; Yixian Chen; Junrong Chen; Lin Yao
>
> **摘要:** Large Language Models (LLMs) have demonstrated considerable potential in general practice. However, existing benchmarks and evaluation frameworks primarily depend on exam-style or simplified question-answer formats, lacking a competency-based structure aligned with the real-world clinical responsibilities encountered in general practice. Consequently, the extent to which LLMs can reliably fulfill the duties of general practitioners (GPs) remains uncertain. In this work, we propose a novel evaluation framework to assess the capability of LLMs to function as GPs. Based on this framework, we introduce a general practice benchmark (GPBench), whose data are meticulously annotated by domain experts in accordance with routine clinical practice standards. We evaluate ten state-of-the-art LLMs and analyze their competencies. Our findings indicate that current LLMs are not yet ready for deployment in such settings without human oversight, and further optimization specifically tailored to the daily responsibilities of GPs is essential.
>
---
#### [replaced 018] P-MMEval: A Parallel Multilingual Multitask Benchmark for Consistent Evaluation of LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.09116v2](http://arxiv.org/pdf/2411.09116v2)**

> **作者:** Yidan Zhang; Yu Wan; Boyi Deng; Baosong Yang; Haoran Wei; Fei Huang; Bowen Yu; Junyang Lin; Fei Huang; Jingren Zhou
>
> **摘要:** Recent advancements in large language models (LLMs) showcase varied multilingual capabilities across tasks like translation, code generation, and reasoning. Previous assessments often limited their scope to fundamental natural language processing (NLP) or isolated capability-specific tasks. To alleviate this drawback, we aim to present a comprehensive multilingual multitask benchmark. First, we introduce P-MMEval, a large-scale benchmark covering effective fundamental and capability-specialized datasets. Furthermore, P-MMEval delivers consistent language coverage across various datasets and provides parallel samples. Finally, we conduct extensive experiments on representative multilingual model series to compare performances across models and tasks, explore the relationship between multilingual performances and factors such as tasks, model sizes, languages, and prompts, and examine the effectiveness of knowledge transfer from English to other languages. The resulting insights are intended to offer valuable guidance for future research. The dataset is available at https://huggingface.co/datasets/Qwen/P-MMEval.
>
---
#### [replaced 019] Fusing Bidirectional Chains of Thought and Reward Mechanisms A Method for Enhancing Question-Answering Capabilities of Large Language Models for Chinese Intangible Cultural Heritage
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.08167v2](http://arxiv.org/pdf/2505.08167v2)**

> **作者:** Ruilin Liu; Zhixiao Zhao; Jieqiong Li; Chang Liu; Dongbo Wang
>
> **备注:** 22 pages, 5 figures
>
> **摘要:** The rapid development of large language models (LLMs) has provided significant support and opportunities for the advancement of domain-specific LLMs. However, fine-tuning these large models using Intangible Cultural Heritage (ICH) data inevitably faces challenges such as bias, incorrect knowledge inheritance, and catastrophic forgetting. To address these issues, we propose a novel training method that integrates a bidirectional chains of thought and a reward mechanism. This method is built upon ICH-Qwen, a large language model specifically designed for the field of intangible cultural heritage. The proposed method enables the model to not only perform forward reasoning but also enhances the accuracy of the generated answers by utilizing reverse questioning and reverse reasoning to activate the model's latent knowledge. Additionally, a reward mechanism is introduced during training to optimize the decision-making process. This mechanism improves the quality of the model's outputs through structural and content evaluations with different weighting schemes. We conduct comparative experiments on ICH-Qwen, with results demonstrating that our method outperforms 0-shot, step-by-step reasoning, knowledge distillation, and question augmentation methods in terms of accuracy, Bleu-4, and Rouge-L scores on the question-answering task. Furthermore, the paper highlights the effectiveness of combining the bidirectional chains of thought and reward mechanism through ablation experiments. In addition, a series of generalizability experiments are conducted, with results showing that the proposed method yields improvements on various domain-specific datasets and advanced models in areas such as Finance, Wikidata, and StrategyQA. This demonstrates that the method is adaptable to multiple domains and provides a valuable approach for model training in future applications across diverse fields.
>
---
#### [replaced 020] Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.11197v4](http://arxiv.org/pdf/2503.11197v4)**

> **作者:** Gang Li; Jizhong Liu; Heinrich Dinkel; Yadong Niu; Junbo Zhang; Jian Luan
>
> **摘要:** Recently, reinforcement learning (RL) has been shown to greatly enhance the reasoning capabilities of large language models (LLMs), and RL-based approaches have been progressively applied to visual multimodal tasks. However, the audio modality has largely been overlooked in these developments. Thus, we conduct a series of RL explorations in audio understanding and reasoning, specifically focusing on the audio question answering (AQA) task. We leverage the group relative policy optimization (GRPO) algorithm to Qwen2-Audio-7B-Instruct, and our experiments demonstrated state-of-the-art performance on the MMAU Test-mini benchmark, achieving an accuracy rate of 64.5%. The main findings in this technical report are as follows: 1) The GRPO algorithm can be effectively applied to large audio language models (LALMs), even when the model has only 8.2B parameters; 2) With only 38k post-training samples, RL significantly outperforms supervised fine-tuning (SFT), indicating that RL-based approaches can be effective without large datasets; 3) The explicit reasoning process has not shown significant benefits for AQA tasks, and how to efficiently utilize deep thinking remains an open question for further research; 4) LALMs still lag far behind humans auditory-language reasoning, suggesting that the RL-based approaches warrant further exploration. Our project is available at https://github.com/xiaomi-research/r1-aqa and https://huggingface.co/mispeech/r1-aqa.
>
---
#### [replaced 021] Llama-Nemotron: Efficient Reasoning Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.00949v3](http://arxiv.org/pdf/2505.00949v3)**

> **作者:** Akhiad Bercovich; Itay Levy; Izik Golan; Mohammad Dabbah; Ran El-Yaniv; Omri Puny; Ido Galil; Zach Moshe; Tomer Ronen; Najeeb Nabwani; Ido Shahaf; Oren Tropp; Ehud Karpas; Ran Zilberstein; Jiaqi Zeng; Soumye Singhal; Alexander Bukharin; Yian Zhang; Tugrul Konuk; Gerald Shen; Ameya Sunil Mahabaleshwarkar; Bilal Kartal; Yoshi Suhara; Olivier Delalleau; Zijia Chen; Zhilin Wang; David Mosallanezhad; Adi Renduchintala; Haifeng Qian; Dima Rekesh; Fei Jia; Somshubra Majumdar; Vahid Noroozi; Wasi Uddin Ahmad; Sean Narenthiran; Aleksander Ficek; Mehrzad Samadi; Jocelyn Huang; Siddhartha Jain; Igor Gitman; Ivan Moshkov; Wei Du; Shubham Toshniwal; George Armstrong; Branislav Kisacanin; Matvei Novikov; Daria Gitman; Evelina Bakhturina; Jane Polak Scowcroft; John Kamalu; Dan Su; Kezhi Kong; Markus Kliegl; Rabeeh Karimi; Ying Lin; Sanjeev Satheesh; Jupinder Parmar; Pritam Gundecha; Brandon Norick; Joseph Jennings; Shrimai Prabhumoye; Syeda Nahida Akter; Mostofa Patwary; Abhinav Khattar; Deepak Narayanan; Roger Waleffe; Jimmy Zhang; Bor-Yiing Su; Guyue Huang; Terry Kong; Parth Chadha; Sahil Jain; Christine Harvey; Elad Segal; Jining Huang; Sergey Kashirsky; Robert McQueen; Izzy Putterman; George Lam; Arun Venkatesan; Sherry Wu; Vinh Nguyen; Manoj Kilaru; Andrew Wang; Anna Warno; Abhilash Somasamudramath; Sandip Bhaskar; Maka Dong; Nave Assaf; Shahar Mor; Omer Ullman Argov; Scot Junkin; Oleksandr Romanenko; Pedro Larroy; Monika Katariya; Marco Rovinelli; Viji Balas; Nicholas Edelman; Anahita Bhiwandiwalla; Muthu Subramaniam; Smita Ithape; Karthik Ramamoorthy; Yuting Wu; Suguna Varshini Velury; Omri Almog; Joyjit Daw; Denys Fridman; Erick Galinkin; Michael Evans; Shaona Ghosh; Katherine Luna; Leon Derczynski; Nikki Pope; Eileen Long; Seth Schneider; Guillermo Siman; Tomasz Grzegorzek; Pablo Ribalta; Monika Katariya; Chris Alexiuk; Joey Conway; Trisha Saar; Ann Guan; Krzysztof Pawelec; Shyamala Prayaga; Oleksii Kuchaiev; Boris Ginsburg; Oluwatobi Olabiyi; Kari Briski; Jonathan Cohen; Bryan Catanzaro; Jonah Alben; Yonatan Geifman; Eric Chung
>
> **摘要:** We introduce the Llama-Nemotron series of models, an open family of heterogeneous reasoning models that deliver exceptional reasoning capabilities, inference efficiency, and an open license for enterprise use. The family comes in three sizes -- Nano (8B), Super (49B), and Ultra (253B) -- and performs competitively with state-of-the-art reasoning models such as DeepSeek-R1 while offering superior inference throughput and memory efficiency. In this report, we discuss the training procedure for these models, which entails using neural architecture search from Llama 3 models for accelerated inference, knowledge distillation, and continued pretraining, followed by a reasoning-focused post-training stage consisting of two main parts: supervised fine-tuning and large scale reinforcement learning. Llama-Nemotron models are the first open-source models to support a dynamic reasoning toggle, allowing users to switch between standard chat and reasoning modes during inference. To further support open research and facilitate model development, we provide the following resources: 1. We release the Llama-Nemotron reasoning models -- LN-Nano, LN-Super, and LN-Ultra -- under the commercially permissive NVIDIA Open Model License Agreement. 2. We release the complete post-training dataset: Llama-Nemotron-Post-Training-Dataset. 3. We also release our training codebases: NeMo, NeMo-Aligner, and Megatron-LM.
>
---
#### [replaced 022] Hakim: Farsi Text Embedding Model
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.08435v2](http://arxiv.org/pdf/2505.08435v2)**

> **作者:** Mehran Sarmadi; Morteza Alikhani; Erfan Zinvandi; Zahra Pourbahman
>
> **摘要:** Recent advancements in text embedding have significantly improved natural language understanding across many languages, yet Persian remains notably underrepresented in large-scale embedding research. In this paper, we present Hakim, a novel state-of-the-art Persian text embedding model that achieves a 8.5% performance improvement over existing approaches on the FaMTEB benchmark, outperforming all previously developed Persian language models. As part of this work, we introduce three new datasets - Corpesia, Pairsia-sup, and Pairsia-unsup - to support supervised and unsupervised training scenarios. Additionally, Hakim is designed for applications in chatbots and retrieval-augmented generation (RAG) systems, particularly addressing retrieval tasks that require incorporating message history within these systems. We also propose a new baseline model built on the BERT architecture. Our language model consistently achieves higher accuracy across various Persian NLP tasks, while the RetroMAE-based model proves particularly effective for textual information retrieval applications. Together, these contributions establish a new foundation for advancing Persian language understanding.
>
---
#### [replaced 023] Construction and Application of Materials Knowledge Graph in Multidisciplinary Materials Science via Large Language Model
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2404.03080v4](http://arxiv.org/pdf/2404.03080v4)**

> **作者:** Yanpeng Ye; Jie Ren; Shaozhou Wang; Yuwei Wan; Imran Razzak; Bram Hoex; Haofen Wang; Tong Xie; Wenjie Zhang
>
> **备注:** 14 pages, 7 figures, 3 tables; Accepted by 38th Conference on Neural Information Processing Systems (NeurIPS 2024)
>
> **摘要:** Knowledge in materials science is widely dispersed across extensive scientific literature, posing significant challenges to the efficient discovery and integration of new materials. Traditional methods, often reliant on costly and time-consuming experimental approaches, further complicate rapid innovation. Addressing these challenges, the integration of artificial intelligence with materials science has opened avenues for accelerating the discovery process, though it also demands precise annotation, data extraction, and traceability of information. To tackle these issues, this article introduces the Materials Knowledge Graph (MKG), which utilizes advanced natural language processing techniques integrated with large language models to extract and systematically organize a decade's worth of high-quality research into structured triples, contains 162,605 nodes and 731,772 edges. MKG categorizes information into comprehensive labels such as Name, Formula, and Application, structured around a meticulously designed ontology, thus enhancing data usability and integration. By implementing network-based algorithms, MKG not only facilitates efficient link prediction but also significantly reduces reliance on traditional experimental methods. This structured approach not only streamlines materials research but also lays the groundwork for more sophisticated science knowledge graphs.
>
---
#### [replaced 024] LLM-based NLG Evaluation: Current Status and Challenges
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2402.01383v3](http://arxiv.org/pdf/2402.01383v3)**

> **作者:** Mingqi Gao; Xinyu Hu; Jie Ruan; Xiao Pu; Xiaojun Wan
>
> **摘要:** Evaluating natural language generation (NLG) is a vital but challenging problem in natural language processing. Traditional evaluation metrics mainly capturing content (e.g. n-gram) overlap between system outputs and references are far from satisfactory, and large language models (LLMs) such as ChatGPT have demonstrated great potential in NLG evaluation in recent years. Various automatic evaluation methods based on LLMs have been proposed, including metrics derived from LLMs, prompting LLMs, fine-tuning LLMs, and human-LLM collaborative evaluation. In this survey, we first give a taxonomy of LLM-based NLG evaluation methods, and discuss their pros and cons, respectively. Lastly, we discuss several open problems in this area and point out future research directions.
>
---
