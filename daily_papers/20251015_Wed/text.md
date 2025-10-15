# 自然语言处理 cs.CL

- **最新发布 86 篇**

- **更新 84 篇**

## 最新发布

#### [new 001] HALF: Harm-Aware LLM Fairness Evaluation Aligned with Deployment
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出HALF框架，属公平性评估任务，旨在解决现有LLM公平性评测脱离实际部署、忽视伤害程度的问题。通过构建九个应用领域的三级危害分类体系，结合真实场景与伤害权重，评估发现模型表现因领域而异，突显部署前需差异化评估。**

- **链接: [http://arxiv.org/pdf/2510.12217v1](http://arxiv.org/pdf/2510.12217v1)**

> **作者:** Ali Mekky; Omar El Herraoui; Preslav Nakov; Yuxia Wang
>
> **摘要:** Large language models (LLMs) are increasingly deployed across high-impact domains, from clinical decision support and legal analysis to hiring and education, making fairness and bias evaluation before deployment critical. However, existing evaluations lack grounding in real-world scenarios and do not account for differences in harm severity, e.g., a biased decision in surgery should not be weighed the same as a stylistic bias in text summarization. To address this gap, we introduce HALF (Harm-Aware LLM Fairness), a deployment-aligned framework that assesses model bias in realistic applications and weighs the outcomes by harm severity. HALF organizes nine application domains into three tiers (Severe, Moderate, Mild) using a five-stage pipeline. Our evaluation results across eight LLMs show that (1) LLMs are not consistently fair across domains, (2) model size or performance do not guarantee fairness, and (3) reasoning models perform better in medical decision support but worse in education. We conclude that HALF exposes a clear gap between previous benchmarking success and deployment readiness.
>
---
#### [new 002] Probing Latent Knowledge Conflict for Faithful Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文研究检索增强生成（RAG）中的知识冲突问题，旨在提升生成结果的忠实性。通过探针分析发现LLM在句子级隐含冲突信号，据此提出CLEAR框架，实现冲突定位与增强注意力，提升模型对检索证据的准确整合。**

- **链接: [http://arxiv.org/pdf/2510.12460v1](http://arxiv.org/pdf/2510.12460v1)**

> **作者:** Linfeng Gao; Baolong Bi; Zheng Yuan; Le Wang; Zerui Chen; Zhimin Wei; Shenghua Liu; Qinggang Zhang; Jinsong Su
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm to enhance the factuality of Large Language Models (LLMs). However, existing RAG systems often suffer from an unfaithfulness issue, where the model's response contradicts evidence from the retrieved context. Existing approaches to improving contextual faithfulness largely rely on external interventions, such as prompt engineering, decoding constraints, or reward-based fine-tuning. These works treat the LLM as a black box and overlook a crucial question: how does the LLM internally integrate retrieved evidence with its parametric memory, particularly under knowledge conflicts? To address this gap, we conduct a probing-based analysis of hidden-state representations in LLMs and observe three findings: knowledge integration occurs hierarchically, conflicts manifest as latent signals at the sentence level, and irrelevant context is often amplified when aligned with parametric knowledge. Building on these findings, we propose CLEAR (Conflict-Localized and Enhanced Attention for RAG), a framework that (i) decomposes context into fine-grained sentence-level knowledge, (ii) employs hidden-state probing to localize conflicting knowledge, and (iii) introduces conflict-aware fine-tuning to guide the model to accurately integrate retrieved evidence. Extensive experiments across three benchmarks demonstrate that CLEAR substantially improves both accuracy and contextual faithfulness, consistently outperforming strong baselines under diverse conflict conditions. The related resources are available at https://github.com/LinfengGao/CLEAR.
>
---
#### [new 003] LLM Reasoning for Machine Translation: Synthetic Data Generation over Thinking Tokens
- **分类: cs.CL**

- **简介: 该论文研究大推理模型在机器翻译中使用“思考令牌”的效果。发现自动生成的中间推理步骤无助于提升翻译性能，而包含实际翻译尝试的中间步骤才有效，表明扩充数据比模仿人类思维链更有效。**

- **链接: [http://arxiv.org/pdf/2510.11919v1](http://arxiv.org/pdf/2510.11919v1)**

> **作者:** Armel Zebaze; Rachel Bawden; Benoît Sagot
>
> **摘要:** Large reasoning models (LRMs) have led to new possibilities in terms of problem-solving, through the devising of a natural language thought process prior to answering a query. While their capabilities are well known across mathematics and coding tasks, their impact on the task of machine translation (MT) remains underexplored. In this work, we explore the benefits of the generation of intermediate tokens when performing MT across multiple language pairs of different levels of resourcedness and multiple setups. We find that "thinking tokens" do not help LRMs better perform MT. This result generalizes to models fine-tuned to reason before translating using distilled chain of thought (CoT) inspired by human translators' practices. Specifically, fine-tuning a model with synthetic CoT explanations detailing how to translate step-by-step does not outperform standard input-output fine-tuning. However, constructing the intermediate tokens by combining the outputs of modular translation-specific prompting strategies results in improvements. Our findings underscore that the contribution of intermediate tokens during fine-tuning highly depends on the presence of translation attempts within them. More broadly, our results suggest that using a teacher to refine target translations or to expand parallel corpora is more impactful than distilling their CoT explanations into "thinking" MT models.
>
---
#### [new 004] StyleDecipher: Robust and Explainable Detection of LLM-Generated Texts with Stylistic Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属文本检测任务，旨在解决现有LLM生成文本检测方法泛化性差、易受干扰且缺乏可解释性的问题。作者提出StyleDecipher框架，结合离散与连续风格特征，实现高准确率、鲁棒且可解释的跨领域检测。**

- **链接: [http://arxiv.org/pdf/2510.12608v1](http://arxiv.org/pdf/2510.12608v1)**

> **作者:** Siyuan Li; Aodu Wulianghai; Xi Lin; Guangyan Li; Xiang Chen; Jun Wu; Jianhua Li
>
> **摘要:** With the increasing integration of large language models (LLMs) into open-domain writing, detecting machine-generated text has become a critical task for ensuring content authenticity and trust. Existing approaches rely on statistical discrepancies or model-specific heuristics to distinguish between LLM-generated and human-written text. However, these methods struggle in real-world scenarios due to limited generalization, vulnerability to paraphrasing, and lack of explainability, particularly when facing stylistic diversity or hybrid human-AI authorship. In this work, we propose StyleDecipher, a robust and explainable detection framework that revisits LLM-generated text detection using combined feature extractors to quantify stylistic differences. By jointly modeling discrete stylistic indicators and continuous stylistic representations derived from semantic embeddings, StyleDecipher captures distinctive style-level divergences between human and LLM outputs within a unified representation space. This framework enables accurate, explainable, and domain-agnostic detection without requiring access to model internals or labeled segments. Extensive experiments across five diverse domains, including news, code, essays, reviews, and academic abstracts, demonstrate that StyleDecipher consistently achieves state-of-the-art in-domain accuracy. Moreover, in cross-domain evaluations, it surpasses existing baselines by up to 36.30%, while maintaining robustness against adversarial perturbations and mixed human-AI content. Further qualitative and quantitative analysis confirms that stylistic signals provide explainable evidence for distinguishing machine-generated text. Our source code can be accessed at https://github.com/SiyuanLi00/StyleDecipher.
>
---
#### [new 005] SafeMT: Multi-turn Safety for Multimodal Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦多模态大模型在多轮对话中的安全问题，提出SafeMT基准和安全指数SI，评估发现现有模型随对话轮次增加风险上升，并设计对话安全 moderator 有效降低攻击成功率。**

- **链接: [http://arxiv.org/pdf/2510.12133v1](http://arxiv.org/pdf/2510.12133v1)**

> **作者:** Han Zhu; Juntao Dai; Jiaming Ji; Haoran Li; Chengkun Cai; Pengcheng Wen; Chi-Min Chan; Boyuan Chen; Yaodong Yang; Sirui Han; Yike Guo
>
> **摘要:** With the widespread use of multi-modal Large Language models (MLLMs), safety issues have become a growing concern. Multi-turn dialogues, which are more common in everyday interactions, pose a greater risk than single prompts; however, existing benchmarks do not adequately consider this situation. To encourage the community to focus on the safety issues of these models in multi-turn dialogues, we introduce SafeMT, a benchmark that features dialogues of varying lengths generated from harmful queries accompanied by images. This benchmark consists of 10,000 samples in total, encompassing 17 different scenarios and four jailbreak methods. Additionally, we propose Safety Index (SI) to evaluate the general safety of MLLMs during conversations. We assess the safety of 17 models using this benchmark and discover that the risk of successful attacks on these models increases as the number of turns in harmful dialogues rises. This observation indicates that the safety mechanisms of these models are inadequate for recognizing the hazard in dialogue interactions. We propose a dialogue safety moderator capable of detecting malicious intent concealed within conversations and providing MLLMs with relevant safety policies. Experimental results from several open-source models indicate that this moderator is more effective in reducing multi-turn ASR compared to existed guard models.
>
---
#### [new 006] R-WoM: Retrieval-augmented World Model For Computer-use Agents
- **分类: cs.CL**

- **简介: 该论文研究LLM作为世界模型在长时程任务中的局限性，提出检索增强的世界模型R-WoM，通过引入外部教程知识提升状态预测与规划能力，显著改善长周期仿真性能。**

- **链接: [http://arxiv.org/pdf/2510.11892v1](http://arxiv.org/pdf/2510.11892v1)**

> **作者:** Kai Mei; Jiang Guo; Shuaichen Chang; Mingwen Dong; Dongkyu Lee; Xing Niu; Jiarong Jiang
>
> **摘要:** Large Language Models (LLMs) can serve as world models to enhance agent decision-making in digital environments by simulating future states and predicting action outcomes, potentially eliminating costly trial-and-error exploration. However, this capability is fundamentally limited by LLMs' tendency toward hallucination and their reliance on static training knowledge, which can lead to compounding errors that inhibit long-horizon simulations. To systematically investigate whether LLMs are appropriate for world modeling, we probe two core capabilities of world models--future state prediction and reward estimation--through three tasks: next-state identification, full-procedure planning alignment, and milestone transition recognition. Our analysis shows that while LLMs effectively capture immediate next states and identify meaningful state transitions, their performance rapidly degrades in full-procedure planning. This highlights LLMs' limitations in reliably modeling environment dynamics over long horizons. To address these limitations, we propose the Retrieval-augmented World Model (R-WoM), which grounds LLM simulations by incorporating factual, up-to-date knowledge retrieved from external tutorials. Experiments show that R-WoM achieves substantial improvements of up to 25.3% (OSWorld) and 18.1% (WebArena) compared to baselines, with particular advantages in longer-horizon simulations.
>
---
#### [new 007] Discrepancy Detection at the Data Level: Toward Consistent Multilingual Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多语言问答系统中的事实与文化差异问题，提出MIND框架，通过用户参与检测数据层面的不一致。工作涵盖构建标注数据集、验证跨领域泛化能力，旨在提升多语言问答系统的事实一致性与文化敏感性。**

- **链接: [http://arxiv.org/pdf/2510.11928v1](http://arxiv.org/pdf/2510.11928v1)**

> **作者:** Lorena Calvo-Bartolomé; Valérie Aldana; Karla Cantarero; Alonso Madroñal de Mesa; Jerónimo Arenas-García; Jordan Boyd-Graber
>
> **备注:** Long paper accepted at EMNLP 2025
>
> **摘要:** Multilingual question answering (QA) systems must ensure factual consistency across languages, especially for objective queries such as What is jaundice?, while also accounting for cultural variation in subjective responses. We propose MIND, a user-in-the-loop fact-checking pipeline to detect factual and cultural discrepancies in multilingual QA knowledge bases. MIND highlights divergent answers to culturally sensitive questions (e.g., Who assists in childbirth?) that vary by region and context. We evaluate MIND on a bilingual QA system in the maternal and infant health domain and release a dataset of bilingual questions annotated for factual and cultural inconsistencies. We further test MIND on datasets from other domains to assess generalization. In all cases, MIND reliably identifies inconsistencies, supporting the development of more culturally aware and factually consistent QA systems.
>
---
#### [new 008] An AI-Based Behavioral Health Safety Filter and Dataset for Identifying Mental Health Crises in Text-Based Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型在心理危机对话中响应不当的问题，提出并评估了一种AI行为健康安全过滤器（VBHSF），通过两个标注数据集验证其在识别心理健康危机方面的高敏感性和特异性，相比现有方法表现更优。**

- **链接: [http://arxiv.org/pdf/2510.12083v1](http://arxiv.org/pdf/2510.12083v1)**

> **作者:** Benjamin W. Nelson; Celeste Wong; Matthew T. Silvestrini; Sooyoon Shin; Alanna Robinson; Jessica Lee; Eric Yang; John Torous; Andrew Trister
>
> **备注:** Main Text: 2943; Abstract: 256; Tables and Figures: 5
>
> **摘要:** Large language models often mishandle psychiatric emergencies, offering harmful or inappropriate advice and enabling destructive behaviors. This study evaluated the Verily behavioral health safety filter (VBHSF) on two datasets: the Verily Mental Health Crisis Dataset containing 1,800 simulated messages and the NVIDIA Aegis AI Content Safety Dataset subsetted to 794 mental health-related messages. The two datasets were clinician-labelled and we evaluated performance using the clinician labels. Additionally, we carried out comparative performance analyses against two open source, content moderation guardrails: OpenAI Omni Moderation Latest and NVIDIA NeMo Guardrails. The VBHSF demonstrated, well-balanced performance on the Verily Mental Health Crisis Dataset v1.0, achieving high sensitivity (0.990) and specificity (0.992) in detecting any mental health crises. It achieved an F1-score of 0.939, sensitivity ranged from 0.917-0.992, and specificity was >= 0.978 in identifying specific crisis categories. When evaluated against the NVIDIA Aegis AI Content Safety Dataset 2.0, VBHSF performance remained highly sensitive (0.982) and accuracy (0.921) with reduced specificity (0.859). When compared with the NVIDIA NeMo and OpenAI Omni Moderation Latest guardrails, the VBHSF demonstrated superior performance metrics across both datasets, achieving significantly higher sensitivity in all cases (all p < 0.001) and higher specificity relative to NVIDIA NeMo (p < 0.001), but not to OpenAI Omni Moderation Latest (p = 0.094). NVIDIA NeMo and OpenAI Omni Moderation Latest exhibited inconsistent performance across specific crisis types, with sensitivity for some categories falling below 0.10. Overall, the VBHSF demonstrated robust, generalizable performance that prioritizes sensitivity to minimize missed crises, a crucial feature for healthcare applications.
>
---
#### [new 009] On the Interplay between Human Label Variation and Model Fairness
- **分类: cs.CL**

- **简介: 该论文研究人类标注差异（HLV）对模型公平性的影响，属于机器学习公平性任务。旨在探究不同HLV方法如何影响模型公平性，通过比较多数投票与多种HLV训练方式，发现无需显式去偏，HLV仍能提升公平性。**

- **链接: [http://arxiv.org/pdf/2510.12036v1](http://arxiv.org/pdf/2510.12036v1)**

> **作者:** Kemal Kurniawan; Meladel Mistica; Timothy Baldwin; Jey Han Lau
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** The impact of human label variation (HLV) on model fairness is an unexplored topic. This paper examines the interplay by comparing training on majority-vote labels with a range of HLV methods. Our experiments show that without explicit debiasing, HLV training methods have a positive impact on fairness.
>
---
#### [new 010] A Survey on Parallel Reasoning
- **分类: cs.CL**

- **简介: 该论文属于综述任务，旨在解决大语言模型推理脆弱性问题。作者提出并定义了平行推理范式，构建分类体系，总结现有技术与应用，并指出挑战与未来方向，为该领域提供研究路线图。**

- **链接: [http://arxiv.org/pdf/2510.12164v1](http://arxiv.org/pdf/2510.12164v1)**

> **作者:** Ziqi Wang; Boye Niu; Zipeng Gao; Zhi Zheng; Tong Xu; Linghui Meng; Zhongli Li; Jing Liu; Yilong Chen; Chen Zhu; Hua Wu; Haifeng Wang; Enhong Chen
>
> **摘要:** With the increasing capabilities of Large Language Models (LLMs), parallel reasoning has emerged as a new inference paradigm that enhances reasoning robustness by concurrently exploring multiple lines of thought before converging on a final answer. It has become a significant trend to explore parallel reasoning to overcome the fragility of standard sequential methods and improve practical performance. In this paper, we aim to survey and summarize the progress and challenges of parallel reasoning. We first present a formal definition of parallel reasoning and clarify its distinction from related concepts like Chain-of-Thought. Then, we organize and discuss advanced techniques based on a novel taxonomy, including non-interactive reasoning, interactive reasoning, and efficiency-focused decoding strategies. Additionally, we explore various application scenarios, such as solving complex problems and enhancing the reliability of LLM outputs.Finally, we highlight the core challenges of parallel reasoning and suggest potential directions for future research. We hope that our work can provide a useful roadmap for beginners and encourage more research on improving parallel reasoning methods. Related source can be avaliable in https://github.com/PPPP-kaqiu/Awesome-Parallel-Reasoning.
>
---
#### [new 011] BoN Appetit Team at LeWiDi-2025: Best-of-N Test-time Scaling Can Not Stomach Annotation Disagreements (Yet)
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究测试时扩展技术在标注分歧评估任务（LeWiDi-2025）中的应用。旨在探索Best-of-N等方法是否适用于无唯一正确答案的场景。实验发现传统方法有效，但Best-of-N不适用，并分析其原因。**

- **链接: [http://arxiv.org/pdf/2510.12516v1](http://arxiv.org/pdf/2510.12516v1)**

> **作者:** Tomas Ruiz; Siyao Peng; Barbara Plank; Carsten Schwemmer
>
> **摘要:** Test-time scaling is a family of techniques to improve LLM outputs at inference time by performing extra computation. To the best of our knowledge, test-time scaling has been limited to domains with verifiably correct answers, like mathematics and coding. We transfer test-time scaling to the LeWiDi-2025 tasks to evaluate annotation disagreements. We experiment with three test-time scaling methods: two benchmark algorithms (Model Averaging and Majority Voting), and a Best-of-N sampling method. The two benchmark methods improve LLM performance consistently on the LeWiDi tasks, but the Best-of-N method does not. Our experiments suggest that the Best-of-N method does not currently transfer from mathematics to LeWiDi tasks, and we analyze potential reasons for this gap.
>
---
#### [new 012] When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究个性化机器生成文本的检测问题，提出新基准\dataset和方法\method，揭示特征反转陷阱现象，并预测检测器在个性化场景下的性能下降，提升检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.12476v1](http://arxiv.org/pdf/2510.12476v1)**

> **作者:** Lang Gao; Xuhui Li; Chenxi Wang; Mingzhe Li; Wei Liu; Zirui Song; Jinghui Zhang; Rui Yan; Preslav Nakov; Xiuying Chen
>
> **摘要:** Large language models (LLMs) have grown more powerful in language generation, producing fluent text and even imitating personal style. Yet, this ability also heightens the risk of identity impersonation. To the best of our knowledge, no prior work has examined personalized machine-generated text (MGT) detection. In this paper, we introduce \dataset, the first benchmark for evaluating detector robustness in personalized settings, built from literary and blog texts paired with their LLM-generated imitations. Our experimental results demonstrate large performance gaps across detectors in personalized settings: some state-of-the-art models suffer significant drops. We attribute this limitation to the \textit{feature-inversion trap}, where features that are discriminative in general domains become inverted and misleading when applied to personalized text. Based on this finding, we propose \method, a simple and reliable way to predict detector performance changes in personalized settings. \method identifies latent directions corresponding to inverted features and constructs probe datasets that differ primarily along these features to evaluate detector dependence. Our experiments show that \method can accurately predict both the direction and the magnitude of post-transfer changes, showing 85\% correlation with the actual performance gaps. We hope that this work will encourage further research on personalized text detection.
>
---
#### [new 013] Multi-stage Prompt Refinement for Mitigating Hallucinations in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大语言模型因提示不规范导致的幻觉问题，提出多阶段提示优化框架MPR。利用小模型逐阶段修正提示错误，并结合自反思与排序机制提升提示质量，有效减少幻觉，增强模型输出准确性。**

- **链接: [http://arxiv.org/pdf/2510.12032v1](http://arxiv.org/pdf/2510.12032v1)**

> **作者:** Jung-Woo Shim; Yeong-Joon Ju; Ji-Hoon Park; Seong-Whan Lee
>
> **备注:** 22 pages, 6 figures
>
> **摘要:** Recent advancements in large language models (LLMs) have shown strong performance in natural language understanding and generation tasks. However, LLMs continue to encounter challenges with hallucinations, where models generate plausible but incorrect information. While several factors contribute to hallucinations, the impact of ill-formed prompts, prompts with ambiguous wording, incorrect grammar, or incomplete information, was relatively under explored. To address this, we introduce Multi-stage Prompt Refinement (MPR), a framework designed to systematically improve these ill-formed prompts across multiple stages. Each stage addresses specific errors such as punctuation, typographical mistakes, and misuse of key terms, using small language models (SLMs) fine-tuned for these tasks. MPR iteratively enhances the clarity of prompts with additional context and employs a self-reflection mechanism with ranking to prioritize the most relevant input. Experimental results on hallucination benchmarks show that prompts refined by MPR achieve over an 85~\% win rate compared to their original forms, demonstrating its effectiveness in reducing hallucinations and improving LLM output accuracy. Interestingly, we reveal that MPR can be combined with existing post-hoc hallucination mitigation frameworks, further enhancing its versatility. MPR provides a lightweight and adaptable solution for enhancing LLM reliability across various domains.
>
---
#### [new 014] CPR: Mitigating Large Language Model Hallucinations with Curative Prompt Refinement
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大语言模型因用户输入模糊或结构不良提示而产生幻觉的问题，提出一种即插即用的修复性提示优化框架CPR。通过清洗原始提示并补充任务描述，提升生成质量，显著降低幻觉。**

- **链接: [http://arxiv.org/pdf/2510.12029v1](http://arxiv.org/pdf/2510.12029v1)**

> **作者:** Jung-Woo Shim; Yeong-Joon Ju; Ji-Hoon Park; Seong-Whan Lee
>
> **备注:** 2024 IEEE International Conference on Systems, Man, and Cybernetics (SMC), 7 pages, 2 figures
>
> **摘要:** Recent advancements in large language models (LLMs) highlight their fluency in generating responses to diverse prompts. However, these models sometimes generate plausible yet incorrect ``hallucinated" facts, undermining trust. A frequent but often overlooked cause of such errors is the use of poorly structured or vague prompts by users, leading LLMs to base responses on assumed rather than actual intentions. To mitigate hallucinations induced by these ill-formed prompts, we introduce Curative Prompt Refinement (CPR), a plug-and-play framework for curative prompt refinement that 1) cleans ill-formed prompts, and 2) generates additional informative task descriptions to align the intention of the user and the prompt using a fine-tuned small language model. When applied to language models, we discover that CPR significantly increases the quality of generation while also mitigating hallucination. Empirical studies show that prompts with CPR applied achieves over a 90\% win rate over the original prompts without any external knowledge.
>
---
#### [new 015] Beating Harmful Stereotypes Through Facts: RAG-based Counter-speech Generation
- **分类: cs.CL**

- **简介: 该论文研究反制有害刻板印象的对抗性言论生成任务，旨在提升生成内容的可信度与连贯性。作者提出基于检索增强生成（RAG）的框架，结合联合国等权威知识源构建知识库，有效生成针对八大群体的可靠反制言论，并在自动与人工评估中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.12316v1](http://arxiv.org/pdf/2510.12316v1)**

> **作者:** Greta Damo; Elena Cabrio; Serena Villata
>
> **摘要:** Counter-speech generation is at the core of many expert activities, such as fact-checking and hate speech, to counter harmful content. Yet, existing work treats counter-speech generation as pure text generation task, mainly based on Large Language Models or NGO experts. These approaches show severe drawbacks due to the limited reliability and coherence in the generated countering text, and in scalability, respectively. To close this gap, we introduce a novel framework to model counter-speech generation as knowledge-wise text generation process. Our framework integrates advanced Retrieval-Augmented Generation (RAG) pipelines to ensure the generation of trustworthy counter-speech for 8 main target groups identified in the hate speech literature, including women, people of colour, persons with disabilities, migrants, Muslims, Jews, LGBT persons, and other. We built a knowledge base over the United Nations Digital Library, EUR-Lex and the EU Agency for Fundamental Rights, comprising a total of 32,792 texts. We use the MultiTarget-CONAN dataset to empirically assess the quality of the generated counter-speech, both through standard metrics (i.e., JudgeLM) and a human evaluation. Results show that our framework outperforms standard LLM baselines and competitive approach, on both assessments. The resulting framework and the knowledge base pave the way for studying trustworthy and sound counter-speech generation, in hate speech and beyond.
>
---
#### [new 016] Reasoning Pattern Matters: Learning to Reason without Human Rationales
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何减少大模型推理训练中依赖人工标注推理过程的成本。提出“模式化推理任务”概念，指出推理模式比标注数据量更重要，并设计PARO框架让大模型自动生成符合推理模式的标注，实现接近人类标注的效果。**

- **链接: [http://arxiv.org/pdf/2510.12643v1](http://arxiv.org/pdf/2510.12643v1)**

> **作者:** Chaoxu Pang; Yixuan Cao; Ping Luo
>
> **备注:** Submitted to Frontiers of Computer Science
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable reasoning capabilities under the widely adopted SFT+RLVR paradigm, which first performs Supervised Fine-Tuning (SFT) on human-annotated reasoning trajectories (rationales) to establish initial reasoning behaviors, then applies Reinforcement Learning with Verifiable Rewards (RLVR) to optimize the model using verifiable signals without golden rationales. However, annotating high-quality rationales for the SFT stage remains prohibitively expensive. This paper investigates when and how rationale annotation costs can be substantially reduced without compromising reasoning performance. We identify a broad class of problems, termed patterned reasoning tasks, where reasoning follows a fixed, procedural strategy consistent across instances. Although instances vary in content such as domain knowledge, factual information, or numeric values, the solution derives from applying a shared reasoning pattern. We argue that the success of SFT+RLVR on such tasks primarily stems from its ability to enable models to internalize these reasoning patterns. Using numerical semantic matching as a representative task, we provide both causal and behavioral evidence showing that reasoning patterns rather than the quantity or quality of rationales are the key determinant of performance. Building on these insights, we propose Pattern-Aware LLMs as Rationale AnnOtators (PARO), a simple yet effective framework that enables LLMs to generate rationales aligned with task-specific reasoning patterns without requiring human rationale annotations. Experiments show that PARO-generated rationales achieve comparable SFT+RLVR performance to human rationales that are 10 times larger. These results suggest that large-scale human rationale annotations can be replaced with LLM-based automatic annotations requiring only limited human supervision over reasoning patterns.
>
---
#### [new 017] APCE: Adaptive Progressive Context Expansion for Long Context Processing
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对长上下文Transformer模型的内存消耗大和性能下降（ContextRot）问题，提出APCE方法，通过语义相似性自适应选择重要输入片段，在减少50%-70%输入的情况下保持良好摘要性能，提升内存效率。**

- **链接: [http://arxiv.org/pdf/2510.12051v1](http://arxiv.org/pdf/2510.12051v1)**

> **作者:** Baisub Lee; Sanghyun Byun; Mohanad Odema; Jung Guack; Jacob Song; Woo Seong Chung
>
> **备注:** NeurIPS 2025 Workshop: ML For Systems
>
> **摘要:** Deploying useful Long-Context Transformer Models (LCTMs) requires addressing two key challenges: (1) A growing memory footprint due to quadratic self-attention and linear KV-cache scaling in memory as sequence length increases; (2) the ContextRot phenomena where empirical evidence suggests that transformer architecture's performance degrades with increasing context length. Given the shared dependency on the input, a natural question arises: Can we surgically select the most important input chunks for processing to synergistically (a) reduce the memory footprint, and (b) mitigate the ContextRot effects? In this paper, we answer this question in the affirmative for long-context summarization tasks. We propose APCE as a context-aware solution to select the most important input chunks through low-dimensional semantic similarity matching with the current query. By directly operating on the input, APCE decouples from strict dependency on underlying hardware or CUDA environments, promising a compatible solution scalable to different deployment systems. Our empirical evaluations have demonstrated superior or on-par summarization performance for APCE compared to the full dense baseline using a fraction (50%-70%) of the input sequence resulting in KV-cache and self-attention memory efficiency improvements. We hope our findings inspire further research on context-aware efficiency solutions for LCTMs geared towards other relevant long-context tasks.
>
---
#### [new 018] From Knowledge to Treatment: Large Language Model Assisted Biomedical Concept Representation for Drug Repurposing
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦药物重定位任务，旨在解决现有知识图谱方法忽略生物医学常识的问题。作者提出LLaDR框架，利用大语言模型提取治疗相关文本表示，增强知识图谱嵌入，提升对复杂疾病的语义理解，实验证明其性能优越。**

- **链接: [http://arxiv.org/pdf/2510.12181v1](http://arxiv.org/pdf/2510.12181v1)**

> **作者:** Chengrui Xiang; Tengfei Ma; Xiangzheng Fu; Yiping Liu; Bosheng Song; Xiangxiang Zeng
>
> **备注:** 16 pages, 4 figures, 13 tables. Accepted by EMNLP 2025 (Findings)
>
> **摘要:** Drug repurposing plays a critical role in accelerating treatment discovery, especially for complex and rare diseases. Biomedical knowledge graphs (KGs), which encode rich clinical associations, have been widely adopted to support this task. However, existing methods largely overlook common-sense biomedical concept knowledge in real-world labs, such as mechanistic priors indicating that certain drugs are fundamentally incompatible with specific treatments. To address this gap, we propose LLaDR, a Large Language Model-assisted framework for Drug Repurposing, which improves the representation of biomedical concepts within KGs. Specifically, we extract semantically enriched treatment-related textual representations of biomedical entities from large language models (LLMs) and use them to fine-tune knowledge graph embedding (KGE) models. By injecting treatment-relevant knowledge into KGE, LLaDR largely improves the representation of biomedical concepts, enhancing semantic understanding of under-studied or complex indications. Experiments based on benchmarks demonstrate that LLaDR achieves state-of-the-art performance across different scenarios, with case studies on Alzheimer's disease further confirming its robustness and effectiveness. Code is available at https://github.com/xiaomingaaa/LLaDR.
>
---
#### [new 019] Improving Text-to-Image Generation with Input-Side Inference-Time Scaling
- **分类: cs.CL**

- **简介: 该论文研究文本到图像生成中的提示词优化问题，提出一种基于大语言模型的输入侧推理时扩缩方法，通过奖励机制和迭代DPO训练自动改写提示，提升图像质量与文本对齐，且具备跨模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.12041v1](http://arxiv.org/pdf/2510.12041v1)**

> **作者:** Ruibo Chen; Jiacheng Pan; Heng Huang; Zhenheng Yang
>
> **摘要:** Recent advances in text-to-image (T2I) generation have achieved impressive results, yet existing models often struggle with simple or underspecified prompts, leading to suboptimal image-text alignment, aesthetics, and quality. We propose a prompt rewriting framework that leverages large language models (LLMs) to refine user inputs before feeding them into T2I backbones. Our approach introduces a carefully designed reward system and an iterative direct preference optimization (DPO) training pipeline, enabling the rewriter to enhance prompts without requiring supervised fine-tuning data. We evaluate our method across diverse T2I models and benchmarks. Results show that our prompt rewriter consistently improves image-text alignment, visual quality, and aesthetics, outperforming strong baselines. Furthermore, we demonstrate strong transferability by showing that a prompt rewriter trained on one T2I backbone generalizes effectively to others without needing to be retrained. We also systematically study scalability, evaluating how performance gains scale with the capacity of the large LLM used as the rewriter. These findings highlight that prompt rewriting is an effective, scalable, and practical model-agnostic strategy for improving T2I systems. We plan to release the code and trained prompt rewriters soon.
>
---
#### [new 020] Information Extraction from Conversation Transcripts: Neuro-Symbolic vs. LLM
- **分类: cs.CL**

- **简介: 该论文研究对话文本中的信息抽取任务，比较神经符号系统与大语言模型在农业领域的性能。实验表明LLM效果更优，但各有权衡，揭示实际应用中性能、效率与控制的平衡难题。**

- **链接: [http://arxiv.org/pdf/2510.12023v1](http://arxiv.org/pdf/2510.12023v1)**

> **作者:** Alice Saebom Kwak; Maria Alexeeva; Gus Hahn-Powell; Keith Alcock; Kevin McLaughlin; Doug McCorkle; Gabe McNunn; Mihai Surdeanu
>
> **备注:** 15 pages, 2 figures
>
> **摘要:** The current trend in information extraction (IE) is to rely extensively on large language models, effectively discarding decades of experience in building symbolic or statistical IE systems. This paper compares a neuro-symbolic (NS) and an LLM-based IE system in the agricultural domain, evaluating them on nine interviews across pork, dairy, and crop subdomains. The LLM-based system outperforms the NS one (F1 total: 69.4 vs. 52.7; core: 63.0 vs. 47.2), where total includes all extracted information and core focuses on essential details. However, each system has trade-offs: the NS approach offers faster runtime, greater control, and high accuracy in context-free tasks but lacks generalizability, struggles with contextual nuances, and requires significant resources to develop and maintain. The LLM-based system achieves higher performance, faster deployment, and easier maintenance but has slower runtime, limited control, model dependency and hallucination risks. Our findings highlight the "hidden cost" of deploying NLP systems in real-world applications, emphasizing the need to balance performance, efficiency, and control.
>
---
#### [new 021] Tracing Multilingual Knowledge Acquisition Dynamics in Domain Adaptation: A Case Study of English-Japanese Biomedical Adaptation
- **分类: cs.CL**

- **简介: 该论文研究多语言领域自适应中的知识获取机制，旨在揭示大模型在英日双语生物医学场景下如何学习和跨语言迁移领域知识。提出AdaXEval评估方法，通过持续训练追踪知识获取动态，发现即使使用高质量双语语料，跨语言迁移仍具挑战。**

- **链接: [http://arxiv.org/pdf/2510.12115v1](http://arxiv.org/pdf/2510.12115v1)**

> **作者:** Xin Zhao; Naoki Yoshinaga; Yuma Tsuta; Akiko Aizawa
>
> **备注:** 22 Pages, Submitted to ARR 2025 Oct
>
> **摘要:** Multilingual domain adaptation (ML-DA) is widely used to learn new domain knowledge across languages into large language models (LLMs). Although many methods have been proposed to improve domain adaptation, the mechanisms of multilingual knowledge acquisition, how domain knowledge is learned within a language and transferred across languages, remain underexplored. This gap leads to suboptimal performance, particularly in low-resource settings. This work examines the learning dynamics of LLMs during ML-DA. Because prior ML-DA studies often train and evaluate on datasets with mismatched knowledge coverage, we propose AdaXEval, an adaptive evaluation method that builds multiple-choice QA datasets from the same bilingual domain corpus used for training, thereby directly studying multilingual knowledge acquisition. Through continual training of LLMs with diverse data recipes, we track how LLMs acquire domain facts and pinpoint the mechanism behind the transformation process from domain training data to knowledge. Our experiments on a 13B English-Japanese bilingual LLM reveal that cross-lingual transfer remains challenging despite a high-quality bilingual corpus. The code has been released.
>
---
#### [new 022] TopoAlign: A Framework for Aligning Code to Math via Topological Decomposition
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦自动形式化任务，旨在将非形式化数学转为形式化表述。针对缺乏高质量平行语料的问题，提出TopoAlign框架，通过拓扑分解代码并重构为与形式化语句结构对齐的数据，利用现有代码库训练数学大模型，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.11944v1](http://arxiv.org/pdf/2510.11944v1)**

> **作者:** Yupei Li; Philipp Borchert; Gerasimos Lampouras
>
> **摘要:** Large Language Models (LLMs) excel at both informal and formal (e.g. Lean 4) mathematical reasoning but still struggle with autoformalisation, the task of transforming informal into formal mathematical statements. Autoformalisation helps pair the informal reasoning of LLMs with formal proof assistants which enable machine-verifiable generation and mitigate hallucinations. Yet, the performance of current Math LLMs is constrained by the scarcity of large-scale corpora, particularly those containing pairs of informal and formal statements. Although current models are trained to generate code from natural language instructions, structural and syntactic differences between these and formal mathematics limit effective transfer learning. We propose TopoAlign, a framework that unlocks widely available code repositories as training resources for Math LLMs. TopoAlign decomposes code into docstrings, main functions, and dependency functions, and reassembles these components into analogues that structurally mirror formal statements. This produces structurally aligned code data that can be used for training Math LLMs without requiring additional human annotation. We train two state-of-the-art models, DeepSeek-Math and Herald, and evaluate them on the minif2f, Putnam, and ProofNet benchmarks. TopoAlign provides substantial gains for DeepSeek-Math, improving performance by 17.77% on BEq@10 and 68.82% on typecheck@10. Despite introducing no new mathematical knowledge, our framework achieves gains of 0.12% and 1.09% for Herald on BEq@10 and typecheck@10, respectively, demonstrating that training on aligned code data is beneficial even for specialized models.
>
---
#### [new 023] PHANTOM RECALL: When Familiar Puzzles Fool Smart Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大模型在逻辑谜题中的推理能力，发现其依赖记忆而非真正推理。作者构建PHANTOM RECALL基准测试，揭示模型在谜题微调后表现骤降，并提出工具识别与缓解“幻象回忆”问题，暴露语言流畅性与逻辑理解间的差距。**

- **链接: [http://arxiv.org/pdf/2510.11812v1](http://arxiv.org/pdf/2510.11812v1)**

> **作者:** Souradeep Mukhopadhyay; Rishabh Baral; Nimeesh Mahajan; Samhitha Harish; Aswin RRV; Mihir Parmar; Mutsumi Nakamura; Chitta Baral
>
> **备注:** 22 Pages
>
> **摘要:** Large language models (LLMs) such as GPT, Gemini, and Claude often appear adept at solving classic logic puzzles--but how much genuine reasoning underlies their answers? Recent evidence suggests that these models frequently rely on memorized templates rather than reasoning from first principles. When puzzles are slightly modified, their performance collapses, revealing a striking fragility. In particular, we asked: Have LLMs addressed these issues? To what extent? How about perturbations to other puzzles? Is there a general way of reformulating the prompt so that the models do better? To examine these things systematically, we introduce PHANTOM RECALL, a benchmark comprising 25 well-known logic puzzles and 149 carefully designed perturbations that preserve reasoning structure but alter superficial details and solutions. We evaluate eleven leading LLMs and identify a recurring failure mode--phantom recall--where models confidently reproduce memorized solutions or spurious rationales that no longer fit the altered scenario. To probe and mitigate this issue, we contribute three tools: (i) an automated logical-equivalence judge to detect reasoning mismatches, (ii) a taxonomy of fine-grained reasoning error categories, and (iii) a prompting-based mitigation framework guided by these categories. Despite near-perfect accuracy on unmodified puzzles, models significantly underperform humans on perturbed ones, exhibiting both phantom recall and over-elaboration. Our findings reveal a crucial limitation: LLMs often fail to re-reason when contextual cues shift--highlighting the gap between linguistic fluency and logical understanding.
>
---
#### [new 024] Not in Sync: Unveiling Temporal Bias in Audio Chat Models
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究音频大模型在时间定位上的偏差问题，属于音频理解与多模态推理任务。作者发现模型预测事件时间戳存在系统性偏移，提出时序偏差指数（TBI）量化该问题，并通过实验分析其在不同数据、模型和事件中的表现，呼吁构建更及时准确的LALM架构。**

- **链接: [http://arxiv.org/pdf/2510.12185v1](http://arxiv.org/pdf/2510.12185v1)**

> **作者:** Jiayu Yao; Shenghua Liu; Yiwei Wang; Rundong Cheng; Lingrui Mei; Baolong Bi; Zhen Xiong; Xueqi Cheng
>
> **摘要:** Large Audio Language Models (LALMs) are increasingly applied to audio understanding and multimodal reasoning, yet their ability to locate when events occur remains underexplored. We present the first systematic study of temporal bias in LALMs, revealing a key limitation in their timestamp prediction. For example, when asked "At which second does the lecturer introduce the key formula?", models often predict timestamps that are consistently earlier or later than the ground truth. Through controlled experiments on timestamped datasets, we find that temporal bias (i) is prevalent across datasets and models, (ii) increases with audio length - even accumulating to tens of seconds in extended recordings, and (iii) varies across event types and positions. We quantify this effect with the Temporal Bias Index (TBI), measuring systematic misalignment in predicted event timings, and complement it with a visualization framework. Our findings highlight a fundamental limitation in current LALMs and call for the development of temporally robust architectures.
>
---
#### [new 025] Deep Associations, High Creativity: A Simple yet Effective Metric for Evaluating Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PACE指标，通过生成平行联想链评估大语言模型的创造力，缓解数据污染与人工评测成本高的问题。实验证明其与人类评分高度相关，并揭示LLMs在联想创造力上接近普通人但不及专业人士。**

- **链接: [http://arxiv.org/pdf/2510.12110v1](http://arxiv.org/pdf/2510.12110v1)**

> **作者:** Ziliang Qiu; Renfen Hu
>
> **备注:** 14 pages
>
> **摘要:** The evaluation of LLMs' creativity represents a crucial research domain, though challenges such as data contamination and costly human assessments often impede progress. Drawing inspiration from human creativity assessment, we propose PACE, asking LLMs to generate Parallel Association Chains to Evaluate their creativity. PACE minimizes the risk of data contamination and offers a straightforward, highly efficient evaluation, as evidenced by its strong correlation with Chatbot Arena Creative Writing rankings (Spearman's $\rho = 0.739$, $p < 0.001$) across various proprietary and open-source models. A comparative analysis of associative creativity between LLMs and humans reveals that while high-performing LLMs achieve scores comparable to average human performance, professional humans consistently outperform LLMs. Furthermore, linguistic analysis reveals that both humans and LLMs exhibit a trend of decreasing concreteness in their associations, and humans demonstrating a greater diversity of associative patterns.
>
---
#### [new 026] Dr.LLM: Dynamic Layer Routing in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Dr.LLM，一种动态层路由框架，旨在提升大语言模型推理效率与准确性。通过引入轻量级路由器决定每层跳过、执行或重复，结合MCTS生成监督信号，在不修改原模型权重下实现计算资源高效利用，兼顾性能与灵活性。**

- **链接: [http://arxiv.org/pdf/2510.12773v1](http://arxiv.org/pdf/2510.12773v1)**

> **作者:** Ahmed Heakl; Martin Gubri; Salman Khan; Sangdoo Yun; Seong Joon Oh
>
> **备注:** 17 pages, Under submission
>
> **摘要:** Large Language Models (LLMs) process every token through all layers of a transformer stack, causing wasted computation on simple queries and insufficient flexibility for harder ones that need deeper reasoning. Adaptive-depth methods can improve efficiency, but prior approaches rely on costly inference-time search, architectural changes, or large-scale retraining, and in practice often degrade accuracy despite efficiency gains. We introduce Dr.LLM, Dynamic routing of Layers for LLMs, a retrofittable framework that equips pretrained models with lightweight per-layer routers deciding to skip, execute, or repeat a block. Routers are trained with explicit supervision: using Monte Carlo Tree Search (MCTS), we derive high-quality layer configurations that preserve or improve accuracy under a compute budget. Our design, windowed pooling for stable routing, focal loss with class balancing, and bottleneck MLP routers, ensures robustness under class imbalance and long sequences. On ARC (logic) and DART (math), Dr.LLM improves accuracy by up to +3.4%p while saving 5 layers per example on average. Routers generalize to out-of-domain tasks (MMLU, GSM8k, AIME, TruthfulQA, SQuADv2, GPQA, PIQA, AGIEval) with only 0.85% accuracy drop while retaining efficiency, and outperform prior routing methods by up to +7.7%p. Overall, Dr.LLM shows that explicitly supervised routers retrofit frozen LLMs for budget-aware, accuracy-driven inference without altering base weights.
>
---
#### [new 027] Generate Logical Equivalence Questions
- **分类: cs.CL**

- **简介: 该论文属自动问答生成任务，旨在解决逻辑等价问题生成中的效率与难度不均问题。提出基于形式语言的生成方法和线性时间算法，实验验证了生成问题的准确性和难度与教材题相当。**

- **链接: [http://arxiv.org/pdf/2510.12001v1](http://arxiv.org/pdf/2510.12001v1)**

> **作者:** Xinyu Wang; Haoming Yu; Yicheng Yang; Zhiyuan Li
>
> **摘要:** Academic dishonesty is met with zero tolerance in higher education, yet plagiarism has become increasingly prevalent in the era of online teaching and learning. Automatic Question Generation (AQG) presents a potential solution to mitigate copying by creating unique questions for each student. Additionally, AQG can provide a vast array of practice questions. Our AQG focuses on generating logical equivalence questions for Discrete Mathematics, a foundational course for first-year computer science students. A literature review reveals that existing AQGs for this type of question generate all propositions that meet user-defined constraints, resulting in inefficiencies and a lack of uniform question difficulty. To address this, we propose a new approach that defines logical equivalence questions using a formal language, translates this language into two sets of generation rules, and develops a linear-time algorithm for question generation. We evaluated our AQG through two experiments. The first involved a group of students completing questions generated by our system. Statistical analysis shows that the accuracy of these questions is comparable to that of textbook questions. The second experiment assessed the number of steps required to solve our generated questions, textbook questions, and those generated by multiple large language models. The results indicated that the difficulty of our questions was similar to that of textbook questions, confirming the quality of our AQG.
>
---
#### [new 028] Omni-Captioner: Data Pipeline, Models, and Benchmark for Omni Detailed Perception
- **分类: cs.CL; cs.CV; cs.MM; cs.SD**

- **简介: 该论文研究多模态细粒度感知任务，旨在解决现有模型在生成细节时易产生幻觉的问题。作者提出Omni-Detective数据生成 pipeline，训练Audio-Captioner和Omni-Captioner模型，并构建新基准Omni-Cloze，实现更准确、可靠的细粒度音频-视觉描述。**

- **链接: [http://arxiv.org/pdf/2510.12720v1](http://arxiv.org/pdf/2510.12720v1)**

> **作者:** Ziyang Ma; Ruiyang Xu; Zhenghao Xing; Yunfei Chu; Yuxuan Wang; Jinzheng He; Jin Xu; Pheng-Ann Heng; Kai Yu; Junyang Lin; Eng Siong Chng; Xie Chen
>
> **备注:** https://github.com/ddlBoJack/Omni-Captioner
>
> **摘要:** Fine-grained perception of multimodal information is critical for advancing human-AI interaction. With recent progress in audio-visual technologies, Omni Language Models (OLMs), capable of processing audio and video signals in parallel, have emerged as a promising paradigm for achieving richer understanding and reasoning. However, their capacity to capture and describe fine-grained details remains limited explored. In this work, we present a systematic and comprehensive investigation of omni detailed perception from the perspectives of the data pipeline, models, and benchmark. We first identify an inherent "co-growth" between detail and hallucination in current OLMs. To address this, we propose Omni-Detective, an agentic data generation pipeline integrating tool-calling, to autonomously produce highly detailed yet minimally hallucinatory multimodal data. Based on the data generated with Omni-Detective, we train two captioning models: Audio-Captioner for audio-only detailed perception, and Omni-Captioner for audio-visual detailed perception. Under the cascade evaluation protocol, Audio-Captioner achieves the best performance on MMAU and MMAR among all open-source models, surpassing Gemini 2.5 Flash and delivering performance comparable to Gemini 2.5 Pro. On existing detailed captioning benchmarks, Omni-Captioner sets a new state-of-the-art on VDC and achieves the best trade-off between detail and hallucination on the video-SALMONN 2 testset. Given the absence of a dedicated benchmark for omni detailed perception, we design Omni-Cloze, a novel cloze-style evaluation for detailed audio, visual, and audio-visual captioning that ensures stable, efficient, and reliable assessment. Experimental results and analysis demonstrate the effectiveness of Omni-Detective in generating high-quality detailed captions, as well as the superiority of Omni-Cloze in evaluating such detailed captions.
>
---
#### [new 029] Scaling Long-Horizon LLM Agent via Context-Folding
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对长周期任务中大模型上下文长度受限的问题，提出Context-Folding框架，通过子任务分解与结果折叠压缩上下文。结合FoldGRPO强化学习方法，实现高效上下文管理，在减少10倍活动上下文的同时优于ReAct基线。**

- **链接: [http://arxiv.org/pdf/2510.11967v1](http://arxiv.org/pdf/2510.11967v1)**

> **作者:** Weiwei Sun; Miao Lu; Zhan Ling; Kang Liu; Xuesong Yao; Yiming Yang; Jiecao Chen
>
> **摘要:** Large language model (LLM) agents are fundamentally constrained by context length on long-horizon tasks. We introduce Context-Folding, a framework that empowers agents to actively manage their working context. An agent can procedurally branch into a sub-trajectory to handle a subtask and then fold it upon completion, collapsing the intermediate steps while retaining a concise summary of the outcome. To make this behavior learnable, we develop an end-to-end reinforcement learning framework FoldGRPO with specific process rewards to encourage effective task decomposition and context management. On complex long-horizon tasks (Deep Research and SWE), our folding agent matches or outperforms the ReAct baselines while using an active context 10$\times$ smaller and significantly outperforms models that rely on summarization-based context management.
>
---
#### [new 030] Teaching Language Models to Faithfully Express their Uncertainty
- **分类: cs.CL**

- **简介: 该论文聚焦语言模型不确定性表达不真实的问题，提出Faithful Uncertainty Tuning（FUT）方法，通过微调使模型在不改变答案分布的前提下，用一致性对齐的词语（如“可能”）准确表达不确定性，提升开放域问答中信实性。**

- **链接: [http://arxiv.org/pdf/2510.12587v1](http://arxiv.org/pdf/2510.12587v1)**

> **作者:** Bryan Eikema; Evgenia Ilia; José G. C. de Souza; Chrysoula Zerva; Wilker Aziz
>
> **摘要:** Large language models (LLMs) often miscommunicate their uncertainty: repeated queries can produce divergent answers, yet generated responses are typically unhedged or hedged in ways that do not reflect this variability. This conveys unfaithful information about the uncertain state of the LLMs' knowledge, creating a faithfulness gap that affects even strong LLMs. We introduce Faithful Uncertainty Tuning (FUT): a fine-tuning approach that teaches instruction-tuned LLMs to express uncertainty faithfully without altering their underlying answer distribution. We construct training data by augmenting model samples with uncertainty hedges (i.e. verbal cues such as 'possibly' or 'likely') aligned with sample consistency, requiring no supervision beyond the model and a set of prompts. We evaluate FUT on open-domain question answering (QA) across multiple models and datasets. Our results show that FUT substantially reduces the faithfulness gap, while preserving QA accuracy and introducing minimal semantic distribution shift. Further analyses demonstrate robustness across decoding strategies, choice of hedgers, and other forms of uncertainty expression (i.e. numerical). These findings establish FUT as a simple and effective way to teach LLMs to communicate uncertainty faithfully.
>
---
#### [new 031] Tokenization Disparities as Infrastructure Bias: How Subword Systems Create Inequities in LLM Access and Efficiency
- **分类: cs.CL; cs.AI; I.2.7; I.2.1; H.3.3; F.2.2**

- **简介: 该论文研究多语言大模型中的分词差异问题，揭示非拉丁语系和形态复杂语言存在显著的标记膨胀与计算低效。通过跨语言量化分析，指出当前分词系统导致的语言不平等，呼吁构建更具包容性的语言技术基础设施。**

- **链接: [http://arxiv.org/pdf/2510.12389v1](http://arxiv.org/pdf/2510.12389v1)**

> **作者:** Hailay Kidu Teklehaymanot; Wolfgang Nejdl
>
> **备注:** 6 pages 4 figures
>
> **摘要:** Tokenization disparities pose a significant barrier to achieving equitable access to artificial intelligence across linguistically diverse populations. This study conducts a large-scale cross-linguistic evaluation of tokenization efficiency in over 200 languages to systematically quantify computational inequities in large language models (LLMs). Using a standardized experimental framework, we applied consistent preprocessing and normalization protocols, followed by uniform tokenization through the tiktoken library across all language samples. Comprehensive tokenization statistics were collected using established evaluation metrics, including Tokens Per Sentence (TPS) and Relative Tokenization Cost (RTC), benchmarked against English baselines. Our cross-linguistic analysis reveals substantial and systematic disparities: Latin-script languages consistently exhibit higher tokenization efficiency, while non-Latin and morphologically complex languages incur significantly greater token inflation, often 3-5 times higher RTC ratios. These inefficiencies translate into increased computational costs and reduced effective context utilization for underrepresented languages. Overall, the findings highlight structural inequities in current AI systems, where speakers of low-resource and non-Latin languages face disproportionate computational disadvantages. Future research should prioritize the development of linguistically informed tokenization strategies and adaptive vocabulary construction methods that incorporate typological diversity, ensuring more inclusive and computationally equitable multilingual AI systems.
>
---
#### [new 032] Evaluating Retrieval-Augmented Generation Systems on Unanswerable, Uncheatable, Realistic, Multi-hop Queries
- **分类: cs.CL; cs.IR**

- **简介: 该论文聚焦RAG系统评估任务，针对现有基准缺乏真实复杂性、易被“作弊”回答的问题，提出自动构建难解、不可作弊、多跳、无答案查询的CRUMQ方法，提升评测难度与现实匹配度。**

- **链接: [http://arxiv.org/pdf/2510.11956v1](http://arxiv.org/pdf/2510.11956v1)**

> **作者:** Gabrielle Kaili-May Liu; Bryan Li; Arman Cohan; William Gantt Walden; Eugene Yang
>
> **摘要:** Real-world use cases often present RAG systems with complex queries for which relevant information is missing from the corpus or is incomplete. In these settings, RAG systems must be able to reject unanswerable, out-of-scope queries and identify failures of retrieval and multi-hop reasoning. Despite this, existing RAG benchmarks rarely reflect realistic task complexity for multi-hop or out-of-scope questions, which often can be cheated via disconnected reasoning (i.e., solved without genuine multi-hop inference) or require only simple factual recall. This limits the ability for such benchmarks to uncover limitations of existing RAG systems. To address this gap, we present the first pipeline for automatic, difficulty-controlled creation of un$\underline{c}$heatable, $\underline{r}$ealistic, $\underline{u}$nanswerable, and $\underline{m}$ulti-hop $\underline{q}$uerie$\underline{s}$ (CRUMQs), adaptable to any corpus and domain. We use our pipeline to create CRUMQs over two popular RAG datasets and demonstrate its effectiveness via benchmark experiments on leading retrieval-augmented LLMs. Results show that compared to prior RAG benchmarks, CRUMQs are highly challenging for RAG systems and achieve up to 81.0\% reduction in cheatability scores. More broadly, our pipeline offers a simple way to enhance benchmark difficulty and realism and drive development of more capable RAG systems.
>
---
#### [new 033] Hey, wait a minute: on at-issue sensitivity in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属对话自然性评估任务，旨在解决语言模型对话连贯性量化难的问题。提出DGRC方法，利用“议题相关性”分析模型是否优先延续核心话题，并考察提示语对这一偏好的调节作用。**

- **链接: [http://arxiv.org/pdf/2510.12740v1](http://arxiv.org/pdf/2510.12740v1)**

> **作者:** Sanghee J. Kim; Kanishka Misra
>
> **备注:** 10 pages, 5 figures, 3 tables. See https://github.com/sangheek16/hey-wait-a-minute for code and data
>
> **摘要:** Evaluating the naturalness of dialogue in language models (LMs) is not trivial: notions of 'naturalness' vary, and scalable quantitative metrics remain limited. This study leverages the linguistic notion of 'at-issueness' to assess dialogue naturalness and introduces a new method: Divide, Generate, Recombine, and Compare (DGRC). DGRC (i) divides a dialogue as a prompt, (ii) generates continuations for subparts using LMs, (iii) recombines the dialogue and continuations, and (iv) compares the likelihoods of the recombined sequences. This approach mitigates bias in linguistic analyses of LMs and enables systematic testing of discourse-sensitive behavior. Applying DGRC, we find that LMs prefer to continue dialogue on at-issue content, with this effect enhanced in instruct-tuned models. They also reduce their at-issue preference when relevant cues (e.g., "Hey, wait a minute") are present. Although instruct-tuning does not further amplify this modulation, the pattern reflects a hallmark of successful dialogue dynamics.
>
---
#### [new 034] MoBiLE: Efficient Mixture-of-Experts Inference on Consumer GPU with Mixture of Big Little Experts
- **分类: cs.CL**

- **简介: 该论文针对MoE模型在消费级GPU上推理效率低的问题，提出MoBiLE框架。通过引入大小专家混合结构和动态切换机制，减少冗余计算与内存传输，在保持精度的同时显著提升推理速度。**

- **链接: [http://arxiv.org/pdf/2510.12357v1](http://arxiv.org/pdf/2510.12357v1)**

> **作者:** Yushu Zhao; Yubin Qin; Yang Wang; Xiaolong Yang; Huiming Han; Shaojun Wei; Yang Hu; Shouyi Yin
>
> **备注:** Accepted to ASP-DAC 2026
>
> **摘要:** Mixture-of-Experts (MoE) models have recently demonstrated exceptional performance across a diverse range of applications. The principle of sparse activation in MoE models facilitates an offloading strategy, wherein active experts are maintained in GPU HBM, while inactive experts are stored in CPU DRAM. The efficacy of this approach, however, is fundamentally constrained by the limited bandwidth of the CPU-GPU interconnect. To mitigate this bottleneck, existing approaches have employed prefetching to accelerate MoE inference. These methods attempt to predict and prefetch the required experts using specially trained modules. Nevertheless, such techniques are often encumbered by significant training overhead and have shown diminished effectiveness on recent MoE models with fine-grained expert segmentation. In this paper, we propose MoBiLE, a plug-and-play offloading-based MoE inference framework with \textit{mixture of big-little experts}. It reduces the number of experts for unimportant tokens to half for acceleration while maintaining full experts for important tokens to guarantee model quality. Further, a dedicated fallback and prefetching mechanism is designed for switching between little and big experts to improve memory efficiency. We evaluate MoBiLE on four typical modern MoE architectures and challenging generative tasks. Our results show that MoBiLE achieves a speedup of 1.60x to 1.72x compared to the baseline on a consumer GPU system, with negligible degradation in accuracy.
>
---
#### [new 035] PRoH: Dynamic Planning and Reasoning over Knowledge Hypergraphs for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文针对知识超图在检索增强生成中的静态规划、非自适应执行和浅层结构利用问题，提出PRoH框架，通过动态规划、结构化子问题分解与EWO引导的路径检索，实现多跳问答的高效推理。**

- **链接: [http://arxiv.org/pdf/2510.12434v1](http://arxiv.org/pdf/2510.12434v1)**

> **作者:** Xiangjun Zai; Xingyu Tan; Xiaoyang Wang; Qing Liu; Xiwei Xu; Wenjie Zhang
>
> **摘要:** Knowledge Hypergraphs (KHs) have recently emerged as a knowledge representation for retrieval-augmented generation (RAG), offering a paradigm to model multi-entity relations into a structured form. However, existing KH-based RAG methods suffer from three major limitations: static retrieval planning, non-adaptive retrieval execution, and superficial use of KH structure and semantics, which constrain their ability to perform effective multi-hop question answering. To overcome these limitations, we propose PRoH, a dynamic Planning and Reasoning over Knowledge Hypergraphs framework. PRoH incorporates three core innovations: (i) a context-aware planning module that sketches the local KH neighborhood to guide structurally grounded reasoning plan generation; (ii) a structured question decomposition process that organizes subquestions as a dynamically evolving Directed Acyclic Graph (DAG) to enable adaptive, multi-trajectory exploration; and (iii) an Entity-Weighted Overlap (EWO)-guided reasoning path retrieval algorithm that prioritizes semantically coherent hyperedge traversals. Experiments across multiple domains demonstrate that PRoH achieves state-of-the-art performance, surpassing the prior SOTA model HyperGraphRAG by an average of 19.73% in F1 and 8.41% in Generation Evaluation (G-E) score, while maintaining strong robustness in long-range multi-hop reasoning tasks.
>
---
#### [new 036] GRAVITY: A Framework for Personalized Text Generation via Profile-Grounded Synthetic Preferences
- **分类: cs.CL**

- **简介: 该论文提出GRAVITY框架，解决LLM个性化生成中依赖昂贵人工标注的问题。通过融合文化、心理等维度构建用户画像，生成合成偏好数据，用于指导个性化文本生成，在多文化场景下显著提升生成效果与用户偏好。**

- **链接: [http://arxiv.org/pdf/2510.11952v1](http://arxiv.org/pdf/2510.11952v1)**

> **作者:** Priyanka Dey; Daniele Rosa; Wenqing Zheng; Daniel Barcklow; Jieyu Zhao; Emilio Ferrara
>
> **摘要:** Personalization in LLMs often relies on costly human feedback or interaction logs, limiting scalability and neglecting deeper user attributes. To reduce the reliance on human annotations, we introduce GRAVITY (Generative Response with Aligned Values, Interests, and Traits of You), a framework for generating synthetic, profile-grounded preference data that captures users' interests, values, beliefs, and personality traits. By integrating demographic, cultural, and psychological frameworks -- including Hofstede's cultural dimensions, Schwartz's basic values, the World Values Survey, and Big Five OCEAN traits -- GRAVITY synthesizes preference pairs to guide personalized content generation. We evaluate GRAVITY on book descriptions for 400 Amazon users, comparing it to prompt-based conditioning, standard fine-tuning, and naive synthetic pair generation. Profile-grounded synthetic data consistently improves generation, especially across multiple cultures (USA, Brazil, Japan, India), achieving over 4% higher preference gains across baselines, with user studies showing that GRAVITY outputs are preferred over 86% of the time. Our results show that scenario-grounded synthetic data can capture richer user variation, reduce reliance on costly annotation, and produce more engaging, user-centered content, offering a scalable path for LLM personalization.
>
---
#### [new 037] A large-scale, unsupervised pipeline for automatic corpus annotation using LLMs: variation and change in the English consider construction
- **分类: cs.CL**

- **简介: 该论文提出一种基于大语言模型的大规模无监督语料自动标注管道，旨在解决人工标注效率低的问题。通过四阶段流程，实现对历史英语语料中consider结构的高效准确标注，验证了LLM在语料处理中的可扩展性与有效性。**

- **链接: [http://arxiv.org/pdf/2510.12306v1](http://arxiv.org/pdf/2510.12306v1)**

> **作者:** Cameron Morin; Matti Marttinen Larsson
>
> **摘要:** As natural language corpora expand at an unprecedented rate, manual annotation remains a significant methodological bottleneck in corpus linguistic work. We address this challenge by presenting a scalable, unsupervised pipeline for automating grammatical annotation in voluminous corpora using large language models (LLMs). Unlike previous supervised and iterative approaches, our method employs a four-phase workflow: prompt engineering, pre-hoc evaluation, automated batch processing, and post-hoc validation. We demonstrate the pipeline's accessibility and effectiveness through a diachronic case study of variation in the English consider construction. Using GPT-5 through the OpenAI API, we annotate 143,933 sentences from the Corpus of Historical American English (COHA) in under 60 hours, achieving 98%+ accuracy on two sophisticated annotation procedures. Our results suggest that LLMs can perform a range of data preparation tasks at scale with minimal human intervention, opening new possibilities for corpus-based research, though implementation requires attention to costs, licensing, and other ethical considerations.
>
---
#### [new 038] Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs
- **分类: cs.CL; cs.AI; I.2.7; I.2.6; J.3**

- **简介: 该论文聚焦医疗大模型在多轮对话中的可靠性问题，提出MedQA-Followup框架，区分浅层与深层鲁棒性，评估模型在误导上下文和答案挑战下的表现，揭示现有模型在临床交互场景中的严重脆弱性。**

- **链接: [http://arxiv.org/pdf/2510.12255v1](http://arxiv.org/pdf/2510.12255v1)**

> **作者:** Blazej Manczak; Eric Lin; Francisco Eiras; James O' Neill; Vaikkunth Mugunthan
>
> **备注:** Dataset and code: https://huggingface.co/datasets/dynamoai-ml/MedQA-USMLE-4-MultiTurnRobust ; https://github.com/bmanczak/MedQA-MultiTurnRobustness Accepted as a poster at NeurIPS 2025 Workshop on GenAI for Health: Potential, Trust, and Policy Compliance
>
> **摘要:** Large language models (LLMs) are rapidly transitioning into medical clinical use, yet their reliability under realistic, multi-turn interactions remains poorly understood. Existing evaluation frameworks typically assess single-turn question answering under idealized conditions, overlooking the complexities of medical consultations where conflicting input, misleading context, and authority influence are common. We introduce MedQA-Followup, a framework for systematically evaluating multi-turn robustness in medical question answering. Our approach distinguishes between shallow robustness (resisting misleading initial context) and deep robustness (maintaining accuracy when answers are challenged across turns), while also introducing an indirect-direct axis that separates contextual framing (indirect) from explicit suggestion (direct). Using controlled interventions on the MedQA dataset, we evaluate five state-of-the-art LLMs and find that while models perform reasonably well under shallow perturbations, they exhibit severe vulnerabilities in multi-turn settings, with accuracy dropping from 91.2% to as low as 13.5% for Claude Sonnet 4. Counterintuitively, indirect, context-based interventions are often more harmful than direct suggestions, yielding larger accuracy drops across models and exposing a significant vulnerability for clinical deployment. Further compounding analyses reveal model differences, with some showing additional performance drops under repeated interventions while others partially recovering or even improving. These findings highlight multi-turn robustness as a critical but underexplored dimension for safe and reliable deployment of medical LLMs.
>
---
#### [new 039] SMEC: Rethinking Matryoshka Representation Learning for Retrieval Embedding Compression
- **分类: cs.CL; cs.LG**

- **简介: 该论文聚焦检索嵌入压缩任务，旨在降低大模型生成的高维嵌入的存储与计算开销。提出SMEC框架，通过序列化学习、自适应降维和跨批次记忆机制，在压缩维度的同时保持语义检索性能。**

- **链接: [http://arxiv.org/pdf/2510.12474v1](http://arxiv.org/pdf/2510.12474v1)**

> **作者:** Biao Zhang; Lixin Chen; Tong Liu; Bo Zheng
>
> **备注:** Accepted by EMNLP2025
>
> **摘要:** Large language models (LLMs) generate high-dimensional embeddings that capture rich semantic and syntactic information. However, high-dimensional embeddings exacerbate computational complexity and storage requirements, thereby hindering practical deployment. To address these challenges, we propose a novel training framework named Sequential Matryoshka Embedding Compression (SMEC). This framework introduces the Sequential Matryoshka Representation Learning(SMRL) method to mitigate gradient variance during training, the Adaptive Dimension Selection (ADS) module to reduce information degradation during dimension pruning, and the Selectable Cross-batch Memory (S-XBM) module to enhance unsupervised learning between high- and low-dimensional embeddings. Experiments on image, text, and multimodal datasets demonstrate that SMEC achieves significant dimensionality reduction while maintaining performance. For instance, on the BEIR dataset, our approach improves the performance of compressed LLM2Vec embeddings (256 dimensions) by 1.1 points and 2.7 points compared to the Matryoshka-Adaptor and Search-Adaptor models, respectively.
>
---
#### [new 040] Chinese ModernBERT with Whole-Word Masking
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Chinese ModernBERT，针对中文预训练模型优化。解决中文tokenization与现有BERT不匹配问题，通过定制分词、整词掩码、长上下文训练等技术提升性能，在CLUE和SimCLUE任务上表现优异。**

- **链接: [http://arxiv.org/pdf/2510.12285v1](http://arxiv.org/pdf/2510.12285v1)**

> **作者:** Zeyu Zhao; Ningtao Wang; Xing Fu; Yu Cheng
>
> **摘要:** Encoder-only Transformers have advanced along three axes -- architecture, data, and systems -- yielding Pareto gains in accuracy, speed, and memory efficiency. Yet these improvements have not fully transferred to Chinese, where tokenization and morphology differ markedly from English. We introduce Chinese ModernBERT, a from-scratch Chinese encoder that couples: (i) a hardware-aware 32k BPE vocabulary tailored to frequent Chinese affixes/compounds, lowering the embedding budget; (ii) whole-word masking (WWM) with a dynamic masking curriculum (30% -> 15%) to align task difficulty with training progress; (iii) a two-stage pre-training pipeline that extends the native context from 1,024 to 8,192 tokens using RoPE and alternating local/global attention; and (iv) a damped-cosine learning-rate schedule for stable long-horizon optimization. We pre-train on ~1.2T Chinese tokens from CCI3-HQ, CCI4 (Chinese), and Cosmopedia-Chinese. On CLUE, Chinese ModernBERT is competitive with strong Chinese encoders under a unified fine-tuning protocol. Under bf16 it achieves high long-sequence throughput while maintaining strong short-sequence speed, reflecting benefits from budget allocation and attention design. To probe retrieval-oriented quality, we add a small amount of open contrastive data: fine-tuning on SimCLUE (~3M pairs) improves further when adding T2Ranking (~2M), reaching 0.505 (Pearson) / 0.537 (Spearman) on the SimCLUE test set. Under this open-data setting, Chinese ModernBERT surpasses Qwen-0.6B-embedding on SimCLUE, suggesting a clear scaling path for STS with additional curated pairs. We will release tokenizer and weights to facilitate reproducible research.
>
---
#### [new 041] Fine-grained Analysis of Brain-LLM Alignment through Input Attribution
- **分类: cs.CL**

- **简介: 该论文研究脑活动与大语言模型（LLM）对齐的机制，旨在探究对齐与下一词预测的关系。作者提出细粒度输入归因方法，发现二者依赖不同词汇特征：后者重语法与位置，前者重语义与篇章。**

- **链接: [http://arxiv.org/pdf/2510.12355v1](http://arxiv.org/pdf/2510.12355v1)**

> **作者:** Michela Proietti; Roberto Capobianco; Mariya Toneva
>
> **摘要:** Understanding the alignment between large language models (LLMs) and human brain activity can reveal computational principles underlying language processing. We introduce a fine-grained input attribution method to identify the specific words most important for brain-LLM alignment, and leverage it to study a contentious research question about brain-LLM alignment: the relationship between brain alignment (BA) and next-word prediction (NWP). Our findings reveal that BA and NWP rely on largely distinct word subsets: NWP exhibits recency and primacy biases with a focus on syntax, while BA prioritizes semantic and discourse-level information with a more targeted recency effect. This work advances our understanding of how LLMs relate to human language processing and highlights differences in feature reliance between BA and NWP. Beyond this study, our attribution method can be broadly applied to explore the cognitive relevance of model predictions in diverse language processing tasks.
>
---
#### [new 042] LLM-REVal: Can We Trust LLM Reviewers Yet?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在学术评审中的公平性问题，通过模拟研究与评审代理，发现LLM评审存在偏爱LLM生成论文、贬低含批判性语句的人类论文等偏差，揭示其潜在风险，同时也探讨了其对论文改进的积极作用。**

- **链接: [http://arxiv.org/pdf/2510.12367v1](http://arxiv.org/pdf/2510.12367v1)**

> **作者:** Rui Li; Jia-Chen Gu; Po-Nien Kung; Heming Xia; Junfeng liu; Xiangwen Kong; Zhifang Sui; Nanyun Peng
>
> **摘要:** The rapid advancement of large language models (LLMs) has inspired researchers to integrate them extensively into the academic workflow, potentially reshaping how research is practiced and reviewed. While previous studies highlight the potential of LLMs in supporting research and peer review, their dual roles in the academic workflow and the complex interplay between research and review bring new risks that remain largely underexplored. In this study, we focus on how the deep integration of LLMs into both peer-review and research processes may influence scholarly fairness, examining the potential risks of using LLMs as reviewers by simulation. This simulation incorporates a research agent, which generates papers and revises, alongside a review agent, which assesses the submissions. Based on the simulation results, we conduct human annotations and identify pronounced misalignment between LLM-based reviews and human judgments: (1) LLM reviewers systematically inflate scores for LLM-authored papers, assigning them markedly higher scores than human-authored ones; (2) LLM reviewers persistently underrate human-authored papers with critical statements (e.g., risk, fairness), even after multiple revisions. Our analysis reveals that these stem from two primary biases in LLM reviewers: a linguistic feature bias favoring LLM-generated writing styles, and an aversion toward critical statements. These results highlight the risks and equity concerns posed to human authors and academic research if LLMs are deployed in the peer review cycle without adequate caution. On the other hand, revisions guided by LLM reviews yield quality gains in both LLM-based and human evaluations, illustrating the potential of the LLMs-as-reviewers for early-stage researchers and enhancing low-quality papers.
>
---
#### [new 043] Hierarchical Alignment: Surgical Fine-Tuning via Functional Layer Specialization in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属大模型对齐任务，旨在解决传统对齐方法忽视模型内部功能分层的问题。作者提出“分层对齐”方法，将DPO针对性地应用于语法、逻辑和事实性对应的功能层，通过LoRA微调实现更高效、可控的对齐，避免了对齐税，在多个主流模型上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2510.12044v1](http://arxiv.org/pdf/2510.12044v1)**

> **作者:** Yukun Zhang; Qi Dong
>
> **摘要:** Existing alignment techniques for Large Language Models (LLMs), such as Direct Preference Optimization (DPO), typically treat the model as a monolithic entity, applying uniform optimization pressure across all layers. This approach overlooks the functional specialization within the Transformer architecture, where different layers are known to handle distinct tasks from syntax to abstract reasoning. In this paper, we challenge this one-size-fits-all paradigm by introducing Hierarchical Alignment, a novel method that applies targeted DPO to distinct functional blocks of a model's layers: local (syntax), intermediate (logic), and global (factuality). Through a series of controlled experiments on state-of-the-art models like Llama-3.1-8B and Qwen1.5-7B using LoRA for surgical fine-tuning, our results, evaluated by a powerful LLM-as-Judge, demonstrate significant and predictable improvements. Specifically, aligning the local layers (Local-Align) enhances grammatical fluency. More importantly, aligning the global layers (Global-Align) not only improves factual consistency as hypothesized but also proves to be the most effective strategy for enhancing logical coherence, outperforming all baselines. Critically, all hierarchical strategies successfully avoid the "alignment tax" observed in standard DPO, where gains in fluency come at the cost of degraded logical reasoning. These findings establish a more resource-efficient, controllable, and interpretable path for model alignment, highlighting the immense potential of shifting from monolithic optimization to structure-aware surgical fine-tuning to build more advanced and reliable LLMs.
>
---
#### [new 044] Cost Analysis of Human-corrected Transcription for Predominately Oral Languages
- **分类: cs.CL**

- **简介: 该论文属语音数据标注任务，旨在解决低资源口语语言（如巴马卡语）转录成本高的问题。通过实地研究，分析人工校正ASR转录所需工时，发现每小时语音需30至36小时人工修正，为类似语言的数据建设提供成本基准。**

- **链接: [http://arxiv.org/pdf/2510.12781v1](http://arxiv.org/pdf/2510.12781v1)**

> **作者:** Yacouba Diarra; Nouhoum Souleymane Coulibaly; Michael Leventhal
>
> **备注:** 6 pages, 1 figure
>
> **摘要:** Creating speech datasets for low-resource languages is a critical yet poorly understood challenge, particularly regarding the actual cost in human labor. This paper investigates the time and complexity required to produce high-quality annotated speech data for a subset of low-resource languages, low literacy Predominately Oral Languages, focusing on Bambara, a Manding language of Mali. Through a one-month field study involving ten transcribers with native proficiency, we analyze the correction of ASR-generated transcriptions of 53 hours of Bambara voice data. We report that it takes, on average, 30 hours of human labor to accurately transcribe one hour of speech data under laboratory conditions and 36 hours under field conditions. The study provides a baseline and practical insights for a large class of languages with comparable profiles undertaking the creation of NLP resources.
>
---
#### [new 045] Conjecturing: An Overlooked Step in Formal Mathematical Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦数学推理中的“猜想”环节，指出当前自动形式化研究忽视了猜想的前置作用。作者构建ConjectureBench数据集和新评估框架，揭示LLM在未显式生成猜想时性能被高估，并提出Lean-FIRe方法提升猜想与形式化效果，推动自动形式化迈向端到端成功。**

- **链接: [http://arxiv.org/pdf/2510.11986v1](http://arxiv.org/pdf/2510.11986v1)**

> **作者:** Jasivan Alex Sivakumar; Philipp Borchert; Ronald Cardenas; Gerasimos Lampouras
>
> **摘要:** Autoformalisation, the task of expressing informal mathematical statements in formal language, is often viewed as a direct translation process. This, however, disregards a critical preceding step: conjecturing. Many mathematical problems cannot be formalised directly without first conjecturing a conclusion such as an explicit answer, or a specific bound. Since Large Language Models (LLMs) already struggle with autoformalisation, and the evaluation of their conjecturing ability is limited and often entangled within autoformalisation or proof, it is particularly challenging to understand its effect. To address this gap, we augment existing datasets to create ConjectureBench, and redesign the evaluation framework and metric specifically to measure the conjecturing capabilities of LLMs both as a distinct task and within the autoformalisation pipeline. Our evaluation of foundational models, including GPT-4.1 and DeepSeek-V3.1, reveals that their autoformalisation performance is substantially overestimated when the conjecture is accounted for during evaluation. However, the conjecture should not be assumed to be provided. We design an inference-time method, Lean-FIRe to improve conjecturing and autoformalisation, which, to the best of our knowledge, achieves the first successful end-to-end autoformalisation of 13 PutnamBench problems with GPT-4.1 and 7 with DeepSeek-V3.1. We demonstrate that while LLMs possess the requisite knowledge to generate accurate conjectures, improving autoformalisation performance requires treating conjecturing as an independent task, and investigating further how to correctly integrate it within autoformalisation. Finally, we provide forward-looking guidance to steer future research toward improving conjecturing, an overlooked step of formal mathematical reasoning.
>
---
#### [new 046] Uncertainty Quantification for Hallucination Detection in Large Language Models: Foundations, Methodology, and Future Directions
- **分类: cs.CL**

- **简介: 该论文聚焦大语言模型幻觉检测中的不确定性量化（UQ）任务，旨在提升模型输出的可靠性。作者梳理UQ基础理论，分类现有方法，评估代表性方法效果，并探讨局限与未来方向。**

- **链接: [http://arxiv.org/pdf/2510.12040v1](http://arxiv.org/pdf/2510.12040v1)**

> **作者:** Sungmin Kang; Yavuz Faruk Bakman; Duygu Nur Yaldiz; Baturalp Buyukates; Salman Avestimehr
>
> **备注:** 24 pages, 3 figures, magazine
>
> **摘要:** The rapid advancement of large language models (LLMs) has transformed the landscape of natural language processing, enabling breakthroughs across a wide range of areas including question answering, machine translation, and text summarization. Yet, their deployment in real-world applications has raised concerns over reliability and trustworthiness, as LLMs remain prone to hallucinations that produce plausible but factually incorrect outputs. Uncertainty quantification (UQ) has emerged as a central research direction to address this issue, offering principled measures for assessing the trustworthiness of model generations. We begin by introducing the foundations of UQ, from its formal definition to the traditional distinction between epistemic and aleatoric uncertainty, and then highlight how these concepts have been adapted to the context of LLMs. Building on this, we examine the role of UQ in hallucination detection, where quantifying uncertainty provides a mechanism for identifying unreliable generations and improving reliability. We systematically categorize a wide spectrum of existing methods along multiple dimensions and present empirical results for several representative approaches. Finally, we discuss current limitations and outline promising future research directions, providing a clearer picture of the current landscape of LLM UQ for hallucination detection.
>
---
#### [new 047] ACADATA: Parallel Dataset of Academic Data for Machine Translation
- **分类: cs.CL**

- **简介: 该论文聚焦学术翻译任务，旨在解决高质量学术平行语料稀缺问题。作者构建了包含训练集和测试集的ACADATA数据集，通过在训练集上微调大模型，并在测试集上验证其有效性，显著提升学术翻译与长文本翻译性能，并公开数据与模型以促进相关研究。**

- **链接: [http://arxiv.org/pdf/2510.12621v1](http://arxiv.org/pdf/2510.12621v1)**

> **作者:** Iñaki Lacunza; Javier Garcia Gilabert; Francesca De Luca Fornaciari; Javier Aula-Blasco; Aitor Gonzalez-Agirre; Maite Melero; Marta Villegas
>
> **摘要:** We present ACADATA, a high-quality parallel dataset for academic translation, that consists of two subsets: ACAD-TRAIN, which contains approximately 1.5 million author-generated paragraph pairs across 96 language directions and ACAD-BENCH, a curated evaluation set of almost 6,000 translations covering 12 directions. To validate its utility, we fine-tune two Large Language Models (LLMs) on ACAD-TRAIN and benchmark them on ACAD-BENCH against specialized machine-translation systems, general-purpose, open-weight LLMs, and several large-scale proprietary models. Experimental results demonstrate that fine-tuning on ACAD-TRAIN leads to improvements in academic translation quality by +6.1 and +12.4 d-BLEU points on average for 7B and 2B models respectively, while also improving long-context translation in a general domain by up to 24.9% when translating out of English. The fine-tuned top-performing model surpasses the best propietary and open-weight models on academic translation domain. By releasing ACAD-TRAIN, ACAD-BENCH and the fine-tuned models, we provide the community with a valuable resource to advance research in academic domain and long-context translation.
>
---
#### [new 048] Towards Inference-time Scaling for Continuous Space Reasoning
- **分类: cs.CL**

- **简介: 该论文研究连续空间推理中的推理时扩展问题，旨在通过采样与重排序提升性能。基于COCONUT模型，探索了生成多样化推理路径的可行性，发现现有离散空间方法在连续空间中效果有限，主因是缺乏利于区分正误推理的归纳偏置，需在训练中显式引入此类偏置。**

- **链接: [http://arxiv.org/pdf/2510.12167v1](http://arxiv.org/pdf/2510.12167v1)**

> **作者:** Minghan Wang; Thuy-Trang Vu; Ehsan Shareghi; Gholamreza Haffari
>
> **摘要:** Inference-time scaling through multiple sample generation in combination with Process- or Outcome-Reward Model (PRM or ORM) re-ranking has proven effective for text-based reasoning in large language models. This paper investigates whether such established techniques can be successfully adapted to reasoning in the continuous space, using COCONUT (Hao et al. 2024) continuous space reasoning LM as the backbone. We demonstrate the feasibility of generating diverse reasoning paths through dropout-based sampling. Our Pass@N analysis on the generated samples reveals the potential that could enable a significant gain in performance akin to observed gain in the discrete space. However, we highlight unique challenges faced for materializing this gain in the continuous thought space. In particular, working recipes for data generation and training PRM and ORM models in the discrete space unlocks only marginal improvements in the continuous space. Through probing various aspects including geometric properties and trajectory dynamics we identify the underlying reasons that prevent effective discrimination between correct and incorrect reasoning (essential for the functioning of PRM and ORM). Our findings reveal that current limitations stem from the absence of key inductive biases in continuous thought representations. We argue that the training frameworks for continuous reasoning LMs require not only to optimize for accuracy but also to explicitly incorporate inductive biases that could be utilized during inference-time for discrimination of correct and incorrect thoughts.\footnote{Our code and data will be publicly available.}
>
---
#### [new 049] Understanding the Modality Gap: An Empirical Study on the Speech-Text Alignment Mechanism of Large Speech Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究端到端大语音语言模型中语音与文本输入的性能差异（模态差距）。通过分析表示对齐机制，提出量化对齐质量的方法，并设计干预策略优化关键token，以缩小模态差距。**

- **链接: [http://arxiv.org/pdf/2510.12116v1](http://arxiv.org/pdf/2510.12116v1)**

> **作者:** Bajian Xiang; Shuaijiang Zhao; Tingwei Guo; Wei Zou
>
> **备注:** Accepted to EMNLP 2025 (Main Conference)
>
> **摘要:** End-to-end Large Speech Language Models (LSLMs) have demonstrated impressive conversational generation abilities, yet consistently fall short of traditional pipeline systems on semantic understanding benchmarks. In this work, we reveal through systematic experimentation that although LSLMs lose some text input performance after speech-text alignment training, the performance gap between speech and text inputs is more pronounced, which we refer to as the modality gap. To understand this gap, we analyze both coarse- and fine-grained text and speech representations. At the coarse-grained level, representations of speech and text in deeper layers are found to be increasingly aligned in direction (cosine similarity), while concurrently diverging in magnitude (Euclidean distance). We further find that representation similarity is strongly correlated with the modality gap. At the fine-grained level, a spontaneous token-level alignment pattern between text and speech representations is observed. Based on this, we introduce the Alignment Path Score to quantify token-level alignment quality, which exhibits stronger correlation with the modality gap. Building on these insights, we design targeted interventions on critical tokens through angle projection and length normalization. These strategies demonstrate the potential to improve correctness for speech inputs. Our study provides the first systematic empirical analysis of the modality gap and alignment mechanisms in LSLMs, offering both theoretical and methodological guidance for future optimization.
>
---
#### [new 050] DPO-Tuned Large Language Models for Segmentation in Simultaneous Speech Translation
- **分类: cs.CL**

- **简介: 该论文研究同步语音翻译中的分段任务，旨在解决传统分段方法缺乏人类偏好对齐的问题。作者提出基于大语言模型与直接偏好优化（DPO）的分段框架，提升分段自然性与翻译质量，在多语言对上优于SHAS等基线方法。**

- **链接: [http://arxiv.org/pdf/2510.12195v1](http://arxiv.org/pdf/2510.12195v1)**

> **作者:** Zeyu Yang; Satoshi Nakamura
>
> **摘要:** Simultaneous speech translation requires accurate segmentation to balance translation quality and latency. Recent studies such as SHAS have introduced pretrained segmentation models, achieving stronger performance than heuristic rules. However, segmentation models such as SHAS, though pretrained and more robust than heuristic methods, are still constrained by supervised learning objectives and do not incorporate human preference alignment, which is crucial for natural real-time interpretation. In this work, we propose a segmentation framework based on large language models (LLMs) trained with Direct Preference Optimization (DPO). By leveraging preference alignment, our method enables LLMs to predict natural segmentation points that better meet the demands of real-time translation. We evaluate the system on the ACL 60/60 corpus across three language pairs (English-Japanese, Chinese, German), using SeamlessM4T v2 as the translation backbone. Experimental results show that our DPO-tuned LLM achieves higher segmentation accuracy than SHAS and yields consistent improvements in translation quality (BLEU, COMET) as well as latency (Average Lagging). Furthermore, our system benefits from IWSLT baselines for direct comparison. These findings highlight the potential of preference-tuned LLMs to surpass existing pretrained segmentation models and advance adaptive, human-aligned simultaneous interpretation.
>
---
#### [new 051] Direct Multi-Token Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出直接多词元解码（DMTD），旨在加速纯解码器Transformer的推理。其核心是利用模型中后层已足够生成多个词元，避免重复计算前中层，实现近2倍加速，仅轻微损失性能。**

- **链接: [http://arxiv.org/pdf/2510.11958v1](http://arxiv.org/pdf/2510.11958v1)**

> **作者:** Xuan Luo; Weizhi Wang; Xifeng Yan
>
> **摘要:** Decoder-only transformers have become the standard architecture for large language models (LLMs) due to their strong performance. Recent studies suggest that, in pre-trained LLMs, early, middle, and late layers may serve distinct roles: Early layers focus on understanding the input context, middle layers handle task-specific processing, and late layers convert abstract representations into output tokens. We hypothesize that once representations have been processed by the early and middle layers, the resulting hidden states may encapsulate sufficient information to support the generation of multiple tokens using only the late layers, eliminating the need to repeatedly traverse the early and middle layers. We refer to this inference paradigm as Direct Multi-Token Decoding (DMTD). Unlike speculative decoding, our method introduces no additional parameters, auxiliary routines, or post-generation verification. Despite being trained on a limited dataset, a fine-tuned DMTD Qwen3-4B model has already demonstrated promising results, achieving up to a 2x speedup with only minor performance loss. Moreover, as shown in our scaling analysis, its performance is expected to further improve with larger training datasets.
>
---
#### [new 052] Analysing Moral Bias in Finetuned LLMs through Mechanistic Interpretability
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究微调大模型中的道德偏差（如Knobe效应），通过机械可解释性方法定位其在模型层中的机制。实验证明，仅修补少数关键层即可消除偏差，表明社会偏见可被定位并干预，无需重训练。**

- **链接: [http://arxiv.org/pdf/2510.12229v1](http://arxiv.org/pdf/2510.12229v1)**

> **作者:** Bianca Raimondi; Daniela Dalbagno; Maurizio Gabbrielli
>
> **备注:** Preprint. Under review
>
> **摘要:** Large language models (LLMs) have been shown to internalize human-like biases during finetuning, yet the mechanisms by which these biases manifest remain unclear. In this work, we investigated whether the well-known Knobe effect, a moral bias in intentionality judgements, emerges in finetuned LLMs and whether it can be traced back to specific components of the model. We conducted a Layer-Patching analysis across 3 open-weights LLMs and demonstrated that the bias is not only learned during finetuning but also localized in a specific set of layers. Surprisingly, we found that patching activations from the corresponding pretrained model into just a few critical layers is sufficient to eliminate the effect. Our findings offer new evidence that social biases in LLMs can be interpreted, localized, and mitigated through targeted interventions, without the need for model retraining.
>
---
#### [new 053] VISaGE: Understanding Visual Generics and Exceptions
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究视觉语言模型（VLM）在处理典型与异常图像时的概念理解能力，旨在揭示模型在语义先验与语用先验间的权衡问题。作者构建了新数据集VISaGE，通过实验发现，当图像与文本不一致时，模型概念理解显著下降。**

- **链接: [http://arxiv.org/pdf/2510.12548v1](http://arxiv.org/pdf/2510.12548v1)**

> **作者:** Stella Frank; Emily Allaway
>
> **备注:** EMNLP 2025
>
> **摘要:** While Vision Language Models (VLMs) learn conceptual representations, in the form of generalized knowledge, during training, they are typically used to analyze individual instances. When evaluation instances are atypical, this paradigm results in tension between two priors in the model. The first is a pragmatic prior that the textual and visual input are both relevant, arising from VLM finetuning on congruent inputs; the second is a semantic prior that the conceptual representation is generally true for instances of the category. In order to understand how VLMs trade off these priors, we introduce a new evaluation dataset, VISaGE, consisting of both typical and exceptional images. In carefully balanced experiments, we show that conceptual understanding degrades when the assumption of congruency underlying the pragmatic prior is violated with incongruent images. This effect is stronger than the effect of the semantic prior when querying about individual instances.
>
---
#### [new 054] Which Word Orders Facilitate Length Generalization in LMs? An Investigation with GCG-Based Artificial Languages
- **分类: cs.CL**

- **简介: 该论文研究语言模型对不同词序的长度泛化能力，旨在探究其是否偏好类型学上常见的语法结构。作者基于广义范畴语法构建人工语言，并评估模型在未见长句上的泛化表现，发现类型学更常见的词序更易被模型掌握。**

- **链接: [http://arxiv.org/pdf/2510.12722v1](http://arxiv.org/pdf/2510.12722v1)**

> **作者:** Nadine El-Naggar; Tatsuki Kuribayashi; Ted Briscoe
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** Whether language models (LMs) have inductive biases that favor typologically frequent grammatical properties over rare, implausible ones has been investigated, typically using artificial languages (ALs) (White and Cotterell, 2021; Kuribayashi et al., 2024). In this paper, we extend these works from two perspectives. First, we extend their context-free AL formalization by adopting Generalized Categorial Grammar (GCG) (Wood, 2014), which allows ALs to cover attested but previously overlooked constructions, such as unbounded dependency and mildly context-sensitive structures. Second, our evaluation focuses more on the generalization ability of LMs to process unseen longer test sentences. Thus, our ALs better capture features of natural languages and our experimental paradigm leads to clearer conclusions -- typologically plausible word orders tend to be easier for LMs to productively generalize.
>
---
#### [new 055] Language Models Model Language
- **分类: cs.CL**

- **简介: 该论文属语言学与AI交叉研究，旨在反驳对大语言模型（LLM）缺乏“深层结构”或“语义根基”的批评。作者基于Mańczak的实证主义语言观，主张语言即言语使用总和，频率为核心机制，据此重构对LLMs的理解，并指导其设计与评估。**

- **链接: [http://arxiv.org/pdf/2510.12766v1](http://arxiv.org/pdf/2510.12766v1)**

> **作者:** Łukasz Borchmann
>
> **摘要:** Linguistic commentary on LLMs, heavily influenced by the theoretical frameworks of de Saussure and Chomsky, is often speculative and unproductive. Critics challenge whether LLMs can legitimately model language, citing the need for "deep structure" or "grounding" to achieve an idealized linguistic "competence." We argue for a radical shift in perspective towards the empiricist principles of Witold Ma\'nczak, a prominent general and historical linguist. He defines language not as a "system of signs" or a "computational system of the brain" but as the totality of all that is said and written. Above all, he identifies frequency of use of particular language elements as language's primary governing principle. Using his framework, we challenge prior critiques of LLMs and provide a constructive guide for designing, evaluating, and interpreting language models.
>
---
#### [new 056] LLM Knowledge is Brittle: Truthfulness Representations Rely on Superficial Resemblance
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型（LLM）知识表征的脆弱性，探究其对输入表面形式变化的敏感性。通过分析不同扰动下真值表征的可分离性，发现LLM的真假判断依赖浅层特征，缺乏鲁棒性，揭示了模型知识泛化能力有限的问题。**

- **链接: [http://arxiv.org/pdf/2510.11905v1](http://arxiv.org/pdf/2510.11905v1)**

> **作者:** Patrick Haller; Mark Ibrahim; Polina Kirichenko; Levent Sagun; Samuel J. Bell
>
> **摘要:** For Large Language Models (LLMs) to be reliable, they must learn robust knowledge that can be generally applied in diverse settings -- often unlike those seen during training. Yet, extensive research has shown that LLM performance can be brittle, with models exhibiting excessive sensitivity to trivial input variations. In this work, we explore whether this brittleness is a direct result of unstable internal knowledge representations. To explore this question, we build on previous work showing that LLM representations encode statement truthfulness -- i.e., true, factual statements can be easily separated from false, inaccurate ones. Specifically, we test the robustness of learned knowledge by evaluating representation separability on samples that have undergone superficial transformations to drive them out-of-distribution (OOD), such as typos or reformulations. By applying semantically-preserving perturbations, we study how separability degrades as statements become more OOD, across four LLM families, five evaluation datasets, and three knowledge probing methods. Our results reveal that internal representations of statement truthfulness collapse as the samples' presentations become less similar to those seen during pre-training. While LLMs can often distinguish between true and false statements when they closely resemble the pre-training data, this ability is highly dependent on the statement's exact surface form. These findings offer a possible explanation for brittle benchmark performance: LLMs may learn shallow, non-robust knowledge representations that allow for only limited generalizability. Our work presents a fundamental challenge for the utility of truthfulness probes, and more broadly, calls for further research on improving the robustness of learned knowledge representations.
>
---
#### [new 057] COSTAR-A: A prompting framework for enhancing Large Language Model performance on Point-of-View questions
- **分类: cs.CL; I.2.7**

- **简介: 该论文针对小规模大模型在观点类问题中输出不一致的问题，提出COSTAR-A提示框架，在COSTAR基础上增加“答案”组件。实验证明其能提升小模型输出的结构性与决断性，尤其在Llama 3.1-8B上效果显著，适用于资源受限场景。**

- **链接: [http://arxiv.org/pdf/2510.12637v1](http://arxiv.org/pdf/2510.12637v1)**

> **作者:** Nzubechukwu C. Ohalete; Kevin B. Gittner; Lauren M. Matheny
>
> **备注:** 20 pages, 2 figures
>
> **摘要:** Large Language Models (LLMs) are highly sensitive to prompt design, and making optimized prompting techniques is crucial for generating consistent, high-quality outputs. In this study, we introduce COSTAR-A, a novel prompt engineering framework that enhances the existing COSTAR method, which stands for Context, Objective, Style, Tone, Audience, and Response, by adding the 'Answer' component at the end. We demonstrate that while the original COSTAR framework improves prompt clarity and aligns outputs for larger LLMs, its performance is less consistent with smaller, locally optimized models, particularly in tasks that require more directive or constrained outputs. Through a series of controlled prompt-output assessments with smaller (at most 8 billion parameters), fine-tuned models, we found that COSTAR-A can enhance the output structure and decisiveness of localized LLMs for certain tasks, although its effectiveness varies across models and use cases. Notably, the Llama 3.1-8B model exhibited performance improvements when prompted with COSTAR-A compared to COSTAR alone. These findings emphasize the adaptability and scalability of COSTAR-A as a prompting framework, particularly in computationally efficient AI deployments on resource-constrained hardware.
>
---
#### [new 058] Credal Transformer: A Principled Approach for Quantifying and Mitigating Hallucinations in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型幻觉问题，提出Credal Transformer架构。通过将注意力分数视为Dirichlet分布的证据，用可信集替代单一概率分布，量化模型不确定性，在输入模糊或分布外时避免过度自信，有效减少错误生成，提升推理可靠性。**

- **链接: [http://arxiv.org/pdf/2510.12137v1](http://arxiv.org/pdf/2510.12137v1)**

> **作者:** Shihao Ji; Zihui Song; Jiajie Huang
>
> **摘要:** Large Language Models (LLMs) hallucinate, generating factually incorrect yet confident assertions. We argue this stems from the Transformer's Softmax function, which creates "Artificial Certainty" by collapsing ambiguous attention scores into a single probability distribution, discarding uncertainty information at each layer. To fix this, we introduce the Credal Transformer, which replaces standard attention with a Credal Attention Mechanism (CAM) based on evidential theory. CAM produces a "credal set" (a set of distributions) instead of a single attention vector, with the set's size directly measuring model uncertainty. We implement this by re-conceptualizing attention scores as evidence masses for a Dirichlet distribution: sufficient evidence recovers standard attention, while insufficient evidence yields a diffuse distribution, representing ambiguity. Empirically, the Credal Transformer identifies out-of-distribution inputs, quantifies ambiguity, and significantly reduces confident errors on unanswerable questions by abstaining. Our contribution is a new architecture to mitigate hallucinations and a design paradigm that integrates uncertainty quantification directly into the model, providing a foundation for more reliable AI.
>
---
#### [new 059] Resource-sensitive but language-blind: Community size and not grammatical complexity better predicts the accuracy of Large Language Models in a novel Wug Test
- **分类: cs.CL**

- **简介: 该论文通过多语言Wug测试考察大语言模型的形态泛化能力，探究其表现受语言复杂度还是训练数据量影响。结果显示，模型准确率更贴近语言的社区规模和数据资源，而非语法复杂性，表明其表现主要由数据驱动。**

- **链接: [http://arxiv.org/pdf/2510.12463v1](http://arxiv.org/pdf/2510.12463v1)**

> **作者:** Nikoleta Pantelidou; Evelina Leivada; Paolo Morosi
>
> **摘要:** The linguistic abilities of Large Language Models are a matter of ongoing debate. This study contributes to this discussion by investigating model performance in a morphological generalization task that involves novel words. Using a multilingual adaptation of the Wug Test, six models were tested across four partially unrelated languages (Catalan, English, Greek, and Spanish) and compared with human speakers. The aim is to determine whether model accuracy approximates human competence and whether it is shaped primarily by linguistic complexity or by the quantity of available training data. Consistent with previous research, the results show that the models are able to generalize morphological processes to unseen words with human-like accuracy. However, accuracy patterns align more closely with community size and data availability than with structural complexity, refining earlier claims in the literature. In particular, languages with larger speaker communities and stronger digital representation, such as Spanish and English, revealed higher accuracy than less-resourced ones like Catalan and Greek. Overall, our findings suggest that model behavior is mainly driven by the richness of linguistic resources rather than by sensitivity to grammatical complexity, reflecting a form of performance that resembles human linguistic competence only superficially.
>
---
#### [new 060] DSAS: A Universal Plug-and-Play Framework for Attention Optimization in Multi-Document Question Answering
- **分类: cs.CL**

- **简介: 该论文针对多文档问答中大模型处理长文本时的“中间信息丢失”和长程依赖建模问题，提出DSAS插件式框架，通过双阶段注意力优化模块提升关键信息关注，无需微调即可在多个主流大模型上显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.12251v1](http://arxiv.org/pdf/2510.12251v1)**

> **作者:** Jiakai Li; Rongzheng Wang; Yizhuo Ma; Shuang Liang; Guangchun Luo; Ke Qin
>
> **备注:** 27 pages, has been accepted by NeurIPS 2025
>
> **摘要:** While large language models (LLMs) show considerable promise across various fields, they have notable limitations in handling multi-document question answering (Multi-doc QA) tasks. The first challenge is long-range dependency modeling, where LLMs struggle to focus on key information in long texts, which weakens important semantic connections. Second, most LLMs suffer from the ''lost-in-the-middle'' issue, where they have difficulty processing information in the middle of long inputs. Current solutions either truncate global dependencies or demand costly finetuning, ultimately lacking a universal and simple solution for these challenges. To resolve these limitations, we propose Dual-Stage Adaptive Sharpening (DSAS) containing two modules. (i) The Contextual Gate Weighting (CGW) module alleviates ''lost-in-the-middle'' by assessing paragraph relevance through layer-wise attention tracking and position-aware weighting. (ii) The Reciprocal Attention Suppression (RAS) module enhances focus on critical paragraphs by suppressing information exchange between key and irrelevant texts, thus mitigating the limitations in long-range dependency modeling. Notably, DSAS functions as a plug-and-play solution requiring no architectural modifications or extra training parameters. Extensive experiments on four benchmarks demonstrate DSAS's efficacy across mainstream LLMs (Llama, Qwen, Mistral, and Deepseek), with an average F1-score improvement of 4.2% in Multi-doc QA tasks on Llama-3.1-8B-Instruct and Qwen2.5-14B-Instruct. Ablation studies confirm the essential contributions of both the CGW and RAS modules. In addition, detailed discussions in the Appendix further validate the robustness and scalability of DSAS.
>
---
#### [new 061] SAGE: A Top-Down Bottom-Up Knowledge-Grounded User Simulator for Multi-turn AGent Evaluation
- **分类: cs.CL**

- **简介: 该论文提出SAGE，用于多轮对话智能体评估的用户模拟框架。针对现有模拟器缺乏领域知识的问题，SAGE融合业务逻辑（自上而下）和实际数据（自下而上），生成更真实、多样化的用户行为，有效提升智能体错误检测能力。**

- **链接: [http://arxiv.org/pdf/2510.11997v1](http://arxiv.org/pdf/2510.11997v1)**

> **作者:** Ryan Shea; Yunan Lu; Liang Qiu; Zhou Yu
>
> **摘要:** Evaluating multi-turn interactive agents is challenging due to the need for human assessment. Evaluation with simulated users has been introduced as an alternative, however existing approaches typically model generic users and overlook the domain-specific principles required to capture realistic behavior. We propose SAGE, a novel user Simulation framework for multi-turn AGent Evaluation that integrates knowledge from business contexts. SAGE incorporates top-down knowledge rooted in business logic, such as ideal customer profiles, grounding user behavior in realistic customer personas. We further integrate bottom-up knowledge taken from business agent infrastructure (e.g., product catalogs, FAQs, and knowledge bases), allowing the simulator to generate interactions that reflect users' information needs and expectations in a company's target market. Through empirical evaluation, we find that this approach produces interactions that are more realistic and diverse, while also identifying up to 33% more agent errors, highlighting its effectiveness as an evaluation tool to support bug-finding and iterative agent improvement.
>
---
#### [new 062] Generation Space Size: Understanding and Calibrating Open-Endedness of LLM Generations
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大模型生成开放性校准问题，提出生成空间大小（GSS）概念，构建GSSBench评测集，发现EigenScore更优，并应用于检测歧义、解释推理模式和引导多样化输出。**

- **链接: [http://arxiv.org/pdf/2510.12699v1](http://arxiv.org/pdf/2510.12699v1)**

> **作者:** Sunny Yu; Ahmad Jabbar; Robert Hawkins; Dan Jurafsky; Myra Cheng
>
> **摘要:** Different open-ended generation tasks require different degrees of output diversity. However, current LLMs are often miscalibrated. They collapse to overly homogeneous outputs for creative tasks and hallucinate diverse but incorrect responses for factual tasks. We argue that these two failure modes are unified by, and can both be addressed by, the notion of effective generation space size (GSS) -- the set of semantically distinct outputs a model considers for a prompt. We present GSSBench, a task suite of prompt pairs with ground-truth GSS relationships to assess different metrics and understand where models diverge from desired behavior. We find that hallucination detection metrics, particularly EigenScore, consistently outperform standard diversity and uncertainty quantification metrics, while using only model internals, providing interpretable insights into a model's internal task representations. We demonstrate three applications of GSS: (1) detecting prompt ambiguity and predicting clarification questions for better grounding, (2) interpreting overthinking and underthinking in reasoning models, and (3) steering models to expand their generation space to yield high-quality and diverse outputs.
>
---
#### [new 063] Evolution of meta's llama models and parameter-efficient fine-tuning of large language models: a survey
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于综述任务，旨在梳理Meta的LLaMA系列模型演进及参数高效微调（PEFT）方法。它分析了LLaMA架构与PEFT技术，比较了不同微调方法在参数效率和性能上的表现，并探讨了应用案例与未来方向。**

- **链接: [http://arxiv.org/pdf/2510.12178v1](http://arxiv.org/pdf/2510.12178v1)**

> **作者:** Abdulhady Abas Abdullah; Arkaitz Zubiaga; Seyedali Mirjalili; Amir H. Gandomi; Fatemeh Daneshfar; Mohammadsadra Amini; Alan Salam Mohammed; Hadi Veisi
>
> **摘要:** This review surveys the rapid evolution of Meta AI's LLaMA (Large Language Model Meta AI) series - from LLaMA 1 through LLaMA 4 and the specialized parameter-efficient fine-tuning (PEFT) methods developed for these models. We first describe the LLaMA family of foundation models (7B-65B to 288B parameters), their architectures (including native multimodal and Mixtureof-Experts variants), and key performance characteristics. We then describe and discuss the concept of PEFT, which adapts large pre-trained models by updating only a small subset of parameters, and review five PEFT methods that have been applied to LLaMA: LoRA (Low-Rank Adaptation), LLaMA-Adapter V1 and V2, LLaMA-Excitor, and QLoRA (Quantized LoRA). We discuss each method's mechanism, parameter savings, and example application to LLaMA (e.g., instruction tuning, multimodal tasks). We provide structured discussion and analysis of model and adapter architectures, parameter counts, and benchmark results (including examples where fine-tuned LLaMA models outperform larger baselines). Finally, we examine real-world use cases where LLaMA-based models and PEFT have been successfully applied (e.g., legal and medical domains), and we discuss ongoing challenges and future research directions (such as scaling to even larger contexts and improving robustness). This survey paper provides a one-stop resource for ML researchers and practitioners interested in LLaMA models and efficient fine-tuning strategies.
>
---
#### [new 064] Demystifying Hybrid Thinking: Can LLMs Truly Switch Between Think and No-Think?
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大模型在混合思维（hybrid thinking）中的可控性问题，旨在解决推理与直答模式分离不彻底的问题。通过分析影响因素并提出优化训练方案，实现了在保持准确率的同时减少直答冗余和推理泄漏。**

- **链接: [http://arxiv.org/pdf/2510.12680v1](http://arxiv.org/pdf/2510.12680v1)**

> **作者:** Shouren Wang; Wang Yang; Xianxuan Long; Qifan Wang; Vipin Chaudhary; Xiaotian Han
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Hybrid thinking enables LLMs to switch between reasoning and direct answering, offering a balance between efficiency and reasoning capability. Yet our experiments reveal that current hybrid thinking LLMs only achieve partial mode separation: reasoning behaviors often leak into the no-think mode. To understand and mitigate this, we analyze the factors influencing controllability and identify four that matter most: (1) larger data scale, (2) using think and no-think answers from different questions rather than the same question, (3) a moderate increase in no-think data number, and (4) a two-phase strategy that first trains reasoning ability and then applies hybrid think training. Building on these findings, we propose a practical recipe that, compared to standard training, can maintain accuracy in both modes while significantly reducing no-think output length (from $1085$ to $585$ on MATH500) and occurrences of reasoning-supportive tokens such as ``\texttt{wait}'' (from $5917$ to $522$ on MATH500). Our findings highlight the limitations of current hybrid thinking and offer directions for strengthening its controllability.
>
---
#### [new 065] Task-Aware Reduction for Scalable LLM-Database Systems
- **分类: cs.SE; cs.CL; cs.DB**

- **简介: 该论文关注LLM与数据库系统的集成，旨在解决数据密集型任务中输入文本冗余导致的成本高、效率低问题。提出将输入约简视为注意力分配，倡导任务感知的文本约简作为核心设计原则，并探讨构建基准、自适应流水线及系统集成等研究方向。**

- **链接: [http://arxiv.org/pdf/2510.11813v1](http://arxiv.org/pdf/2510.11813v1)**

> **作者:** Marcus Emmanuel Barnes; Taher A. Ghaleb; Safwat Hassan
>
> **备注:** Preprint. Accepted for presentation at the Workshop on Language Models and Databases (LMD), co-located with CASCON 2025 (IEEE). The final version will appear in IEEE Xplore
>
> **摘要:** Large Language Models (LLMs) are increasingly applied to data-intensive workflows, from database querying to developer observability. Yet the effectiveness of these systems is constrained by the volume, verbosity, and noise of real-world text-rich data such as logs, telemetry, and monitoring streams. Feeding such data directly into LLMs is costly, environmentally unsustainable, and often misaligned with task objectives. Parallel efforts in LLM efficiency have focused on model- or architecture-level optimizations, but the challenge of reducing upstream input verbosity remains underexplored. In this paper, we argue for treating the token budget of an LLM as an attention budget and elevating task-aware text reduction as a first-class design principle for language -- data systems. We position input-side reduction not as compression, but as attention allocation: prioritizing information most relevant to downstream tasks. We outline open research challenges for building benchmarks, designing adaptive reduction pipelines, and integrating token-budget--aware preprocessing into database and retrieval systems. Our vision is to channel scarce attention resources toward meaningful signals in noisy, data-intensive workflows, enabling scalable, accurate, and sustainable LLM--data integration.
>
---
#### [new 066] The Role of Parametric Injection-A Systematic Study of Parametric Retrieval-Augmented Generation
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究参数化检索增强生成（PRAG），探讨参数注入机制。发现参数化文档仅捕获部分语义，单独使用效果有限，但能提升模型对上下文文档的理解。建议结合文本与参数化文档以提升性能。**

- **链接: [http://arxiv.org/pdf/2510.12668v1](http://arxiv.org/pdf/2510.12668v1)**

> **作者:** Minghao Tang; Shiyu Ni; Jingtong Wu; Zengxin Han; Keping Bi
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models (LLMs) by retrieving external documents. As an emerging form of RAG, parametric retrieval-augmented generation (PRAG) encodes documents as model parameters (i.e., LoRA modules) and injects these representations into the model during inference, enabling interaction between the LLM and documents at parametric level. Compared with directly placing documents in the input context, PRAG is more efficient and has the potential to offer deeper model-document interaction. Despite its growing attention, the mechanism underlying parametric injection remains poorly understood. In this work, we present a systematic study of PRAG to clarify the role of parametric injection, showing that parameterized documents capture only partial semantic information of documents, and relying on them alone yields inferior performance compared to interaction at text level. However, these parametric representations encode high-level document information that can enhance the model's understanding of documents within the input context. When combined parameterized documents with textual documents, the model can leverage relevant information more effectively and become more robust to noisy inputs, achieving better performance than either source alone. We recommend jointly using parameterized and textual documents and advocate for increasing the information content of parametric representations to advance PRAG.
>
---
#### [new 067] Who is a Better Matchmaker? Human vs. Algorithmic Judge Assignment in a High-Stakes Startup Competition
- **分类: cs.HC; cs.AI; cs.CL; cs.CY; cs.LG**

- **简介: 该论文研究算法与人工在评委分配任务中的表现，旨在解决高风险创业大赛中如何高效匹配评委与项目的问题。作者提出HLSE算法，在真实赛事中实现与人工相当的匹配质量，且耗时大幅缩短。**

- **链接: [http://arxiv.org/pdf/2510.12692v1](http://arxiv.org/pdf/2510.12692v1)**

> **作者:** Sarina Xi; Orelia Pi; Miaomiao Zhang; Becca Xiong; Jacqueline Ng Lane; Nihar B. Shah
>
> **备注:** 17 Pages, 2 figures
>
> **摘要:** There is growing interest in applying artificial intelligence (AI) to automate and support complex decision-making tasks. However, it remains unclear how algorithms compare to human judgment in contexts requiring semantic understanding and domain expertise. We examine this in the context of the judge assignment problem, matching submissions to suitably qualified judges. Specifically, we tackled this problem at the Harvard President's Innovation Challenge, the university's premier venture competition awarding over \$500,000 to student and alumni startups. This represents a real-world environment where high-quality judge assignment is essential. We developed an AI-based judge-assignment algorithm, Hybrid Lexical-Semantic Similarity Ensemble (HLSE), and deployed it at the competition. We then evaluated its performance against human expert assignments using blinded match-quality scores from judges on $309$ judge-venture pairs. Using a Mann-Whitney U statistic based test, we found no statistically significant difference in assignment quality between the two approaches ($AUC=0.48, p=0.40$); on average, algorithmic matches are rated $3.90$ and manual matches $3.94$ on a 5-point scale, where 5 indicates an excellent match. Furthermore, manual assignments that previously required a full week could be automated in several hours by the algorithm during deployment. These results demonstrate that HLSE achieves human-expert-level matching quality while offering greater scalability and efficiency, underscoring the potential of AI-driven solutions to support and enhance human decision-making for judge assignment in high-stakes settings.
>
---
#### [new 068] Simple Projection Variants Improve ColBERT Performance
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文研究检索任务中的多向量模型ColBERT，指出其线性投影存在局限。作者探索多种前馈网络结构替代原投影层，发现更优的投影设计可显著提升性能，验证了改进方案的有效性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.12327v1](http://arxiv.org/pdf/2510.12327v1)**

> **作者:** Benjamin Clavié; Sean Lee; Rikiya Takehi; Aamir Shakir; Makoto P. Kato
>
> **摘要:** Multi-vector dense retrieval methods like ColBERT systematically use a single-layer linear projection to reduce the dimensionality of individual vectors. In this study, we explore the implications of the MaxSim operator on the gradient flows of the training of multi-vector models and show that such a simple linear projection has inherent, if non-critical, limitations in this setting. We then discuss the theoretical improvements that could result from replacing this single-layer projection with well-studied alternative feedforward linear networks (FFN), such as deeper, non-linear FFN blocks, GLU blocks, and skip-connections, could alleviate these limitations. Through the design and systematic evaluation of alternate projection blocks, we show that better-designed final projections positively impact the downstream performance of ColBERT models. We highlight that many projection variants outperform the original linear projections, with the best-performing variants increasing average performance on a range of retrieval benchmarks across domains by over 2 NDCG@10 points. We then conduct further exploration on the individual parameters of these projections block in order to understand what drives this empirical performance, highlighting the particular importance of upscaled intermediate projections and residual connections. As part of these ablation studies, we show that numerous suboptimal projection variants still outperform the traditional single-layer projection across multiple benchmarks, confirming our hypothesis. Finally, we observe that this effect is consistent across random seeds, further confirming that replacing the linear layer of ColBERT models is a robust, drop-in upgrade.
>
---
#### [new 069] UALM: Unified Audio Language Model for Understanding, Generation and Reasoning
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文提出统一音频语言模型UALM，旨在解决音频理解、文本到音频生成和多模态推理分离的问题。通过UALM-Gen和UALM-Reason，实现单模型多任务，支持跨模态生成与推理，首次在音频领域验证了跨模态生成式推理的有效性。**

- **链接: [http://arxiv.org/pdf/2510.12000v1](http://arxiv.org/pdf/2510.12000v1)**

> **作者:** Jinchuan Tian; Sang-gil Lee; Zhifeng Kong; Sreyan Ghosh; Arushi Goel; Chao-Han Huck Yang; Wenliang Dai; Zihan Liu; Hanrong Ye; Shinji Watanabe; Mohammad Shoeybi; Bryan Catanzaro; Rafael Valle; Wei Ping
>
> **摘要:** Recent advances in the audio language modeling (ALM) domain tackle audio understanding and text-to-audio generation as separate tasks. Very few studies attempt to unify these tasks -- an essential step toward advanced multimodal reasoning. This paper introduces U}nified Audio Language Model (UALM), which aims to unify audio understanding, text-to-audio generation, and multimodal reasoning in a single model. To achieve this goal, we first present UALM-Gen, a text-to-audio language model that directly predicts audio tokens and is comparable to state-of-the-art diffusion-based models. We then demonstrate, using proper data blending, training recipes, and inference techniques, that our single UALM model matches the quality of state-of-the-art specialized models in audio understanding, text-to-audio generation, and text reasoning. Furthermore, we present UALM-Reason, a multimodal reasoning model that utilizes both text and audio in the intermediate thinking steps to facilitate complex generation tasks. To our knowledge, this is the first demonstration in audio research of cross-modal generative reasoning, with its effectiveness confirmed by subjective evaluations.
>
---
#### [new 070] ThinkPilot: Steering Reasoning Models via Automated Think-prefixes Optimization
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ThinkPilot，一种无需训练的框架，旨在优化大推理模型的推理过程。通过进化生成“思考前缀”，引导模型调整推理行为，提升效率、安全性和指令遵循能力，实现任务自适应的推理控制。**

- **链接: [http://arxiv.org/pdf/2510.12063v1](http://arxiv.org/pdf/2510.12063v1)**

> **作者:** Sunzhu Li; Zhiyu Lin; Shuling Yang; Jiale Zhao; Wei Chen
>
> **摘要:** Large Reasoning Models (LRMs) are powerful, but they still suffer from inefficient and off-target reasoning. Currently, training-free methods are limited to either rigid heuristics or descriptive, non-actionable analyses. In this paper, we introduce ThinkPilot, a training-free framework that automatically optimizes LRMs reasoning. It uses an evolutionary process to generate think-prefixes, which are instructions that evolve driven by a taxonomy of reasoning behaviors to guide models toward superior performance. Extensive experiments demonstrate ThinkPilot's broad effectiveness: it significantly improves the accuracy-length trade-off for efficient reasoning, drastically improves safety (for example, cutting the StrongREJECT score of DeepSeek-R1-Distill-Qwen-32B from 27.0% to 0.7), and enhances instruction following. It also synergizes with existing training-based methods. Our analysis reveals that think-prefixes can reliably control LRMs' reasoning behaviors, and that different tasks have strong preferences for specific behavioral distributions. By automatically identifying and eliciting these behaviors, ThinkPilot provides a generalizable framework for aligning LRMs reasoning with task demands. Data and code are available at https://github.com/teqkilla/ThinkPilot
>
---
#### [new 071] Evolution of wartime discourse on Telegram: A comparative study of Ukrainian and Russian policymakers' communication before and after Russia's full-scale invasion of Ukraine
- **分类: cs.SI; cs.CL; cs.CY**

- **简介: 该论文研究俄乌战争期间两国政策制定者在Telegram上的政治传播变化，属计算社会科学任务。旨在分析战前战后通信量、主题与互动差异，揭示精英如何调整话语策略。基于2019–2024年数据，发现乌方初期聚焦战事但渐减弱，俄方则转移话题以分散公众注意力。**

- **链接: [http://arxiv.org/pdf/2510.11746v1](http://arxiv.org/pdf/2510.11746v1)**

> **作者:** Mykola Makhortykh; Aytalina Kulichkina; Kateryna Maikovska
>
> **备注:** 46 pages
>
> **摘要:** This study examines elite-driven political communication on Telegram during the ongoing Russo-Ukrainian war, the first large-scale European war in the social media era. Using a unique dataset of Telegram public posts from Ukrainian and Russian policymakers (2019-2024), we analyze changes in communication volume, thematic content, and actor engagement following Russia's 2022 full-scale invasion. Our findings show a sharp increase in Telegram activity after the invasion, particularly among ruling-party policymakers. Ukrainian policymakers initially focused on war-related topics, but this emphasis declined over time In contrast, Russian policymakers largely avoided war-related discussions, instead emphasizing unrelated topics, such as Western crises, to distract public attention. We also identify differences in communication strategies between large and small parties, as well as individual policymakers. Our findings shed light on how policymakers adapt to wartime communication challenges and offer critical insights into the dynamics of online political discourse during times of war.
>
---
#### [new 072] Vision Language Models Map Logos to Text via Semantic Entanglement in the Visual Projector
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型（VLM）在无文本标志中误生成品牌名称的“标志幻觉”问题。通过构建数据集与扰动实验，发现幻觉源于视觉投影器中的语义纠缠，并提出通过子空间消融缓解该问题，提升模型可靠性。**

- **链接: [http://arxiv.org/pdf/2510.12287v1](http://arxiv.org/pdf/2510.12287v1)**

> **作者:** Sifan Li; Hongkai Chen; Yujun Cai; Qingwen Ye; Liyang Chen; Junsong Yuan; Yiwei Wang
>
> **摘要:** Vision Language Models (VLMs) have achieved impressive progress in multimodal reasoning; yet, they remain vulnerable to hallucinations, where outputs are not grounded in visual evidence. In this paper, we investigate a previously overlooked setting: logo hallucination, where models generate brand names or textual content despite logos containing no visible words. Using curated splits of pure symbols, hybrids, and text-bearing logos, as well as the challenging Hard-60 subset, we systematically measure hallucination across leading VLMs. We further probe robustness through nine structured perturbations and show that hallucinations persist even under strong distortions, with occlusion exposing the sharpest weaknesses. Embedding-level analysis with open-weight LLaVA demonstrates that hallucination is tied to a small subset of projector dimensions, and targeted ablation substantially reduces errors while preserving OCR accuracy. Together, these findings reveal that VLMs often rely on symbolic priors rather than genuine glyph perception, particularly for iconic circular logos, and that projector subspaces play a decisive role in this failure mode. Our work contributes both a novel diagnostic lens and actionable mitigation insights, highlighting projector disentanglement and OCR-guided decoding as promising directions for building more trustworthy multimodal systems.
>
---
#### [new 073] HackWorld: Evaluating Computer-Use Agents on Exploiting Web Application Vulnerabilities
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出HackWorld框架，评估计算机使用代理（CUA）在真实Web应用中通过视觉交互发现和利用漏洞的能力。针对CUA在复杂界面下安全测试能力不足的问题，构建含36个真实漏洞应用的评测基准，揭示现有CUA在攻击规划与工具使用上的缺陷。**

- **链接: [http://arxiv.org/pdf/2510.12200v1](http://arxiv.org/pdf/2510.12200v1)**

> **作者:** Xiaoxue Ren; Penghao Jiang; Kaixin Li; Zhiyong Huang; Xiaoning Du; Jiaojiao Jiang; Zhenchang Xing; Jiamou Sun; Terry Yue Zhuo
>
> **摘要:** Web applications are prime targets for cyberattacks as gateways to critical services and sensitive data. Traditional penetration testing is costly and expertise-intensive, making it difficult to scale with the growing web ecosystem. While language model agents show promise in cybersecurity, modern web applications demand visual understanding, dynamic content handling, and multi-step interactions that only computer-use agents (CUAs) can perform. Yet, their ability to discover and exploit vulnerabilities through graphical interfaces remains largely unexplored. We present HackWorld, the first framework for systematically evaluating CUAs' capabilities to exploit web application vulnerabilities via visual interaction. Unlike sanitized benchmarks, HackWorld includes 36 real-world applications across 11 frameworks and 7 languages, featuring realistic flaws such as injection vulnerabilities, authentication bypasses, and unsafe input handling. Using a Capture-the-Flag (CTF) setup, it tests CUAs' capacity to identify and exploit these weaknesses while navigating complex web interfaces. Evaluation of state-of-the-art CUAs reveals concerning trends: exploitation rates below 12% and low cybersecurity awareness. CUAs often fail at multi-step attack planning and misuse security tools. These results expose the current limitations of CUAs in web security contexts and highlight opportunities for developing more security-aware agents capable of effective vulnerability detection and exploitation.
>
---
#### [new 074] Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对AI智能体评估难、标准不一的问题，提出Holistic Agent Leaderboard（HAL）评测框架。通过标准化评估流程、多维度分析与日志审查，提升评估效率与真实性，推动AI智能体在真实场景中的可靠应用。**

- **链接: [http://arxiv.org/pdf/2510.11977v1](http://arxiv.org/pdf/2510.11977v1)**

> **作者:** Sayash Kapoor; Benedikt Stroebl; Peter Kirgis; Nitya Nadgir; Zachary S Siegel; Boyi Wei; Tianci Xue; Ziru Chen; Felix Chen; Saiteja Utpala; Franck Ndzomga; Dheeraj Oruganty; Sophie Luskin; Kangheng Liu; Botao Yu; Amit Arora; Dongyoon Hahm; Harsh Trivedi; Huan Sun; Juyong Lee; Tengjun Jin; Yifan Mai; Yifei Zhou; Yuxuan Zhu; Rishi Bommasani; Daniel Kang; Dawn Song; Peter Henderson; Yu Su; Percy Liang; Arvind Narayanan
>
> **摘要:** AI agents have been developed for complex real-world tasks from coding to customer service. But AI agent evaluations suffer from many challenges that undermine our understanding of how well agents really work. We introduce the Holistic Agent Leaderboard (HAL) to address these challenges. We make three main contributions. First, we provide a standardized evaluation harness that orchestrates parallel evaluations across hundreds of VMs, reducing evaluation time from weeks to hours while eliminating common implementation bugs. Second, we conduct three-dimensional analysis spanning models, scaffolds, and benchmarks. We validate the harness by conducting 21,730 agent rollouts across 9 models and 9 benchmarks in coding, web navigation, science, and customer service with a total cost of about $40,000. Our analysis reveals surprising insights, such as higher reasoning effort reducing accuracy in the majority of runs. Third, we use LLM-aided log inspection to uncover previously unreported behaviors, such as searching for the benchmark on HuggingFace instead of solving a task, or misusing credit cards in flight booking tasks. We share all agent logs, comprising 2.5B tokens of language model calls, to incentivize further research into agent behavior. By standardizing how the field evaluates agents and addressing common pitfalls in agent evaluation, we hope to shift the focus from agents that ace benchmarks to agents that work reliably in the real world.
>
---
#### [new 075] Content Anonymization for Privacy in Long-form Audio
- **分类: cs.SD; cs.CL**

- **简介: 该论文研究长语音中的隐私保护，指出传统声纹匿名化无法防御基于语言风格的内容攻击。作者提出在ASR-TTS流程中对文本进行上下文重写，通过 paraphrasing 消除说话者特有表达方式，实验证明该方法可有效防御内容攻击并保持语义可用性。**

- **链接: [http://arxiv.org/pdf/2510.12780v1](http://arxiv.org/pdf/2510.12780v1)**

> **作者:** Cristina Aggazzotti; Ashi Garg; Zexin Cai; Nicholas Andrews
>
> **摘要:** Voice anonymization techniques have been found to successfully obscure a speaker's acoustic identity in short, isolated utterances in benchmarks such as the VoicePrivacy Challenge. In practice, however, utterances seldom occur in isolation: long-form audio is commonplace in domains such as interviews, phone calls, and meetings. In these cases, many utterances from the same speaker are available, which pose a significantly greater privacy risk: given multiple utterances from the same speaker, an attacker could exploit an individual's vocabulary, syntax, and turns of phrase to re-identify them, even when their voice is completely disguised. To address this risk, we propose new content anonymization approaches. Our approach performs a contextual rewriting of the transcripts in an ASR-TTS pipeline to eliminate speaker-specific style while preserving meaning. We present results in a long-form telephone conversation setting demonstrating the effectiveness of a content-based attack on voice-anonymized speech. Then we show how the proposed content-based anonymization methods can mitigate this risk while preserving speech utility. Overall, we find that paraphrasing is an effective defense against content-based attacks and recommend that stakeholders adopt this step to ensure anonymity in long-form audio.
>
---
#### [new 076] SRUM: Fine-Grained Self-Rewarding for Unified Multimodal Models
- **分类: cs.CV; cs.CL; I.4.0**

- **简介: 该论文针对统一多模态模型中视觉理解与生成能力不匹配的问题，提出SRUM框架，利用模型自身的理解模块作为奖励信号，通过全局-局部双奖励机制实现生成模块的自我优化，提升图文生成质量与推理能力。**

- **链接: [http://arxiv.org/pdf/2510.12784v1](http://arxiv.org/pdf/2510.12784v1)**

> **作者:** Weiyang Jin; Yuwei Niu; Jiaqi Liao; Chengqi Duan; Aoxue Li; Shenghua Gao; Xihui Liu
>
> **备注:** 20 pages, 8 figures, webpage can be seen in https://waynejin0918.github.io/srum_web/
>
> **摘要:** Recently, remarkable progress has been made in Unified Multimodal Models (UMMs), which integrate vision-language generation and understanding capabilities within a single framework. However, a significant gap exists where a model's strong visual understanding often fails to transfer to its visual generation. A model might correctly understand an image based on user instructions, yet be unable to generate a faithful image from text prompts. This phenomenon directly raises a compelling question: Can a model achieve self-improvement by using its understanding module to reward its generation module? To bridge this gap and achieve self-improvement, we introduce SRUM, a self-rewarding post-training framework that can be directly applied to existing UMMs of various designs. SRUM creates a feedback loop where the model's own understanding module acts as an internal ``evaluator'', providing corrective signals to improve its generation module, without requiring additional human-labeled data. To ensure this feedback is comprehensive, we designed a global-local dual reward system. To tackle the inherent structural complexity of images, this system offers multi-scale guidance: a \textbf{global reward} ensures the correctness of the overall visual semantics and layout, while a \textbf{local reward} refines fine-grained, object-level fidelity. SRUM leads to powerful capabilities and shows strong generalization, boosting performance on T2I-CompBench from 82.18 to \textbf{88.37} and on T2I-ReasonBench from 43.82 to \textbf{46.75}. Overall, our work establishes a powerful new paradigm for enabling a UMMs' understanding module to guide and enhance its own generation via self-rewarding.
>
---
#### [new 077] One Life to Learn: Inferring Symbolic World Models for Stochastic Environments from Unguided Exploration
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究从无指导探索中学习随机环境的符号化世界模型，解决复杂、不确定性环境下仅一次生命试错的建模难题。提出OneLife框架，用条件触发的程序化规则构建动态计算图，在Crafter-OO环境中验证其状态预测与规划能力。**

- **链接: [http://arxiv.org/pdf/2510.12088v1](http://arxiv.org/pdf/2510.12088v1)**

> **作者:** Zaid Khan; Archiki Prasad; Elias Stengel-Eskin; Jaemin Cho; Mohit Bansal
>
> **备注:** Project page: https://onelife-worldmodel.github.io/; 39 pages
>
> **摘要:** Symbolic world modeling requires inferring and representing an environment's transitional dynamics as an executable program. Prior work has focused on largely deterministic environments with abundant interaction data, simple mechanics, and human guidance. We address a more realistic and challenging setting, learning in a complex, stochastic environment where the agent has only "one life" to explore a hostile environment without human guidance. We introduce OneLife, a framework that models world dynamics through conditionally-activated programmatic laws within a probabilistic programming framework. Each law operates through a precondition-effect structure, activating in relevant world states. This creates a dynamic computation graph that routes inference and optimization only through relevant laws, avoiding scaling challenges when all laws contribute to predictions about a complex, hierarchical state, and enabling the learning of stochastic dynamics even with sparse rule activation. To evaluate our approach under these demanding constraints, we introduce a new evaluation protocol that measures (a) state ranking, the ability to distinguish plausible future states from implausible ones, and (b) state fidelity, the ability to generate future states that closely resemble reality. We develop and evaluate our framework on Crafter-OO, our reimplementation of the Crafter environment that exposes a structured, object-oriented symbolic state and a pure transition function that operates on that state alone. OneLife can successfully learn key environment dynamics from minimal, unguided interaction, outperforming a strong baseline on 16 out of 23 scenarios tested. We also test OneLife's planning ability, with simulated rollouts successfully identifying superior strategies. Our work establishes a foundation for autonomously constructing programmatic world models of unknown, complex environments.
>
---
#### [new 078] Deep Research Brings Deeper Harm
- **分类: cs.CR; cs.CL**

- **简介: 该论文研究深度研究（DR）代理在生成危险内容方面的安全风险，揭示其易被滥用生成专业级有害报告。针对现有越狱方法不适用的问题，提出两种新策略：计划注入与意图劫持，验证了DR代理存在系统性对齐失效，需专门的安全对策。**

- **链接: [http://arxiv.org/pdf/2510.11851v1](http://arxiv.org/pdf/2510.11851v1)**

> **作者:** Shuo Chen; Zonggen Li; Zhen Han; Bailan He; Tong Liu; Haokun Chen; Georg Groh; Philip Torr; Volker Tresp; Jindong Gu
>
> **备注:** Accepted to Reliable ML from Unreliable Data Workshop @ NeurIPS 2025
>
> **摘要:** Deep Research (DR) agents built on Large Language Models (LLMs) can perform complex, multi-step research by decomposing tasks, retrieving online information, and synthesizing detailed reports. However, the misuse of LLMs with such powerful capabilities can lead to even greater risks. This is especially concerning in high-stakes and knowledge-intensive domains such as biosecurity, where DR can generate a professional report containing detailed forbidden knowledge. Unfortunately, we have found such risks in practice: simply submitting a harmful query, which a standalone LLM directly rejects, can elicit a detailed and dangerous report from DR agents. This highlights the elevated risks and underscores the need for a deeper safety analysis. Yet, jailbreak methods designed for LLMs fall short in exposing such unique risks, as they do not target the research ability of DR agents. To address this gap, we propose two novel jailbreak strategies: Plan Injection, which injects malicious sub-goals into the agent's plan; and Intent Hijack, which reframes harmful queries as academic research questions. We conducted extensive experiments across different LLMs and various safety benchmarks, including general and biosecurity forbidden prompts. These experiments reveal 3 key findings: (1) Alignment of the LLMs often fail in DR agents, where harmful prompts framed in academic terms can hijack agent intent; (2) Multi-step planning and execution weaken the alignment, revealing systemic vulnerabilities that prompt-level safeguards cannot address; (3) DR agents not only bypass refusals but also produce more coherent, professional, and dangerous content, compared with standalone LLMs. These results demonstrate a fundamental misalignment in DR agents and call for better alignment techniques tailored to DR agents. Code and datasets are available at https://chenxshuo.github.io/deeper-harm.
>
---
#### [new 079] Don't Walk the Line: Boundary Guidance for Filtered Generation
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对生成模型与安全分类器联用时易产生边界模糊导致误判的问题，提出Boundary Guidance方法，通过强化学习引导生成结果远离分类边界，提升安全性和生成质量。**

- **链接: [http://arxiv.org/pdf/2510.11834v1](http://arxiv.org/pdf/2510.11834v1)**

> **作者:** Sarah Ball; Andreas Haupt
>
> **备注:** 9 pages, 3 figures, 3 tables
>
> **摘要:** Generative models are increasingly paired with safety classifiers that filter harmful or undesirable outputs. A common strategy is to fine-tune the generator to reduce the probability of being filtered, but this can be suboptimal: it often pushes the model toward producing samples near the classifier's decision boundary, increasing both false positives and false negatives. We propose Boundary Guidance, a reinforcement learning fine-tuning method that explicitly steers generation away from the classifier's margin. On a benchmark of jailbreak and ambiguous prompts, Boundary Guidance improves both the safety and the utility of outputs, as judged by LLM-as-a-Judge evaluations. Comprehensive ablations across model scales and reward designs demonstrate the robustness of our approach.
>
---
#### [new 080] Scaling Law in LLM Simulated Personality: More Detailed and Realistic Persona Profile Is All You Need
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文研究大模型模拟人格的任务，旨在解决现有心理测量方法难以评估LLM人格仿真质量的问题。作者提出系统性评估框架，实证揭示角色细节对仿真效果的关键作用，并发现人格模拟中的缩放律。**

- **链接: [http://arxiv.org/pdf/2510.11734v1](http://arxiv.org/pdf/2510.11734v1)**

> **作者:** Yuqi Bai; Tianyu Huang; Kun Sun; Yuting Chen
>
> **摘要:** This research focuses on using large language models (LLMs) to simulate social experiments, exploring their ability to emulate human personality in virtual persona role-playing. The research develops an end-to-end evaluation framework, including individual-level analysis of stability and identifiability, as well as population-level analysis called progressive personality curves to examine the veracity and consistency of LLMs in simulating human personality. Methodologically, this research proposes important modifications to traditional psychometric approaches (CFA and construct validity) which are unable to capture improvement trends in LLMs at their current low-level simulation, potentially leading to remature rejection or methodological misalignment. The main contributions of this research are: proposing a systematic framework for LLM virtual personality evaluation; empirically demonstrating the critical role of persona detail in personality simulation quality; and identifying marginal utility effects of persona profiles, especially a Scaling Law in LLM personality simulation, offering operational evaluation metrics and a theoretical foundation for applying large language models in social science experiments.
>
---
#### [new 081] DiSTAR: Diffusion over a Scalable Token Autoregressive Representation for Speech Generation
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文提出DiSTAR，用于零样本文本到语音合成。针对现有方法在分布偏移下脆弱及可控性不足的问题，其在离散RVQ码空间中结合自回归模型与掩蔽扩散模型，实现块级并行生成，提升鲁棒性、自然度与可控性。**

- **链接: [http://arxiv.org/pdf/2510.12210v1](http://arxiv.org/pdf/2510.12210v1)**

> **作者:** Yakun Song; Xiaobin Zhuang; Jiawei Chen; Zhikang Niu; Guanrou Yang; Chenpeng Du; Zhuo Chen; Yuping Wang; Yuxuan Wang; Xie Chen
>
> **摘要:** Recent attempts to interleave autoregressive (AR) sketchers with diffusion-based refiners over continuous speech representations have shown promise, but they remain brittle under distribution shift and offer limited levers for controllability. We introduce DISTAR, a zero-shot text-to-speech framework that operates entirely in a discrete residual vector quantization (RVQ) code space and tightly couples an AR language model with a masked diffusion model, without forced alignment or a duration predictor. Concretely, DISTAR drafts block-level RVQ tokens with an AR language model and then performs parallel masked-diffusion infilling conditioned on the draft to complete the next block, yielding long-form synthesis with blockwise parallelism while mitigating classic AR exposure bias. The discrete code space affords explicit control at inference: DISTAR produces high-quality audio under both greedy and sample-based decoding using classifier-free guidance, supports trade-offs between robustness and diversity, and enables variable bit-rate and controllable computation via RVQ layer pruning at test time. Extensive experiments and ablations demonstrate that DISTAR surpasses state-of-the-art zero-shot TTS systems in robustness, naturalness, and speaker/style consistency, while maintaining rich output diversity. Audio samples are provided on https://anonymous.4open.science/w/DiSTAR_demo.
>
---
#### [new 082] Precise Attribute Intensity Control in Large Language Models via Targeted Representation Editing
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究大语言模型中属性强度的精确控制，旨在生成符合用户指定强度的文本。提出将目标强度视为到达问题，通过值函数预测与梯度干预实现精细调控，提升对齐精度，并在多个任务中验证有效性。**

- **链接: [http://arxiv.org/pdf/2510.12121v1](http://arxiv.org/pdf/2510.12121v1)**

> **作者:** Rongzhi Zhang; Liqin Ye; Yuzhao Heng; Xiang Chen; Tong Yu; Lingkai Kong; Sudheer Chava; Chao Zhang
>
> **摘要:** Precise attribute intensity control--generating Large Language Model (LLM) outputs with specific, user-defined attribute intensities--is crucial for AI systems adaptable to diverse user expectations. Current LLM alignment methods, however, typically provide only directional or open-ended guidance, failing to reliably achieve exact attribute intensities. We address this limitation with three key designs: (1) reformulating precise attribute intensity control as a target-reaching problem, rather than simple maximization; (2) training a lightweight value function via temporal-difference learning to predict final attribute intensity scores from partial generations, thereby steering LLM outputs; and (3) employing gradient-based interventions on hidden representations to navigate the model precisely towards specific attribute intensity targets. Our method enables fine-grained, continuous control over attribute intensities, moving beyond simple directional alignment. Experiments on LLaMA-3.2-3b and Phi-4-mini confirm our method's ability to steer text generation to user-specified attribute intensities with high accuracy. Finally, we demonstrate efficiency enhancements across three downstream tasks: preference data synthesis, Pareto frontier approximation and optimization, and distillation of aligned behaviors for intervention-free inference. Our code is available on https://github.com/Pre-Control/pre-control
>
---
#### [new 083] Balancing Synthetic Data and Replay for Enhancing Task-Specific Capabilities
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究在持续预训练中平衡合成数据与回放比例，以提升任务能力并防止遗忘。针对bAbI推理任务，通过实验探索不同计算预算下的最优回放配置，提出兼顾任务性能与知识保留的实践指南。**

- **链接: [http://arxiv.org/pdf/2510.11842v1](http://arxiv.org/pdf/2510.11842v1)**

> **作者:** Urs Spiegelhalter; Jörg K. H. Franke; Frank Hutter
>
> **备注:** Presented at 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop on Continual and Compatible Foundation Model Updates (CCFM)
>
> **摘要:** Adapting language models to new tasks through continued pretraining faces a fundamental trade-off: models must learn new capabilities while avoiding catastrophic forgetting of existing knowledge. While prior work has studied synthetic data generation techniques, the optimal replay ratios for balancing task performance and knowledge retention under computational constraints remain poorly understood. We present a comprehensive empirical study investigating the interplay between replay ratio configuration and computational budget when adapting language models to new tasks. Using the bAbI reasoning tasks as our target objective, we apply synthetic data generation and systematically evaluate different total token budgets and replay ratio configurations. We analyze their effects on both task mastery and general knowledge retention. Our experiments reveal an optimal configuration that balances task-specific performance with general knowledge retention. Based on our findings, we provide empirically-grounded guidelines for selecting replay ratios based on computational budget, enabling practitioners to achieve strong task adaptation with significantly reduced training costs.
>
---
#### [new 084] Celebrity Profiling on Short Urdu Text using Twitter Followers' Feed
- **分类: cs.SI; cs.AI; cs.CL**

- **简介: 该论文研究基于推特粉丝文本的乌尔都语名人画像，旨在解决低资源语言下名人人口统计特征预测问题。作者构建了乌尔都语推文数据集，比较多种机器学习与深度学习模型，实现性别、年龄、职业和知名度预测，验证了粉丝语言特征对名人画像的有效性。**

- **链接: [http://arxiv.org/pdf/2510.11739v1](http://arxiv.org/pdf/2510.11739v1)**

> **作者:** Muhammad Hamza; Rizwan Jafar
>
> **摘要:** Social media has become an essential part of the digital age, serving as a platform for communication, interaction, and information sharing. Celebrities are among the most active users and often reveal aspects of their personal and professional lives through online posts. Platforms such as Twitter provide an opportunity to analyze language and behavior for understanding demographic and social patterns. Since followers frequently share linguistic traits and interests with the celebrities they follow, textual data from followers can be used to predict celebrity demographics. However, most existing research in this field has focused on English and other high-resource languages, leaving Urdu largely unexplored. This study applies modern machine learning and deep learning techniques to the problem of celebrity profiling in Urdu. A dataset of short Urdu tweets from followers of subcontinent celebrities was collected and preprocessed. Multiple algorithms were trained and compared, including Logistic Regression, Support Vector Machines, Random Forests, Convolutional Neural Networks, and Long Short-Term Memory networks. The models were evaluated using accuracy, precision, recall, F1-score, and cumulative rank (cRank). The best performance was achieved for gender prediction with a cRank of 0.65 and an accuracy of 0.65, followed by moderate results for age, profession, and fame prediction. These results demonstrate that follower-based linguistic features can be effectively leveraged using machine learning and neural approaches for demographic prediction in Urdu, a low-resource language.
>
---
#### [new 085] Data or Language Supervision: What Makes CLIP Better than DINO?
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文探究CLIP优于DINO的原因，区分是语言监督还是数据量的影响。通过控制变量训练，发现CLIP更擅高级语义，适合文本任务；DINO侧重低级特征，略胜于视觉任务。**

- **链接: [http://arxiv.org/pdf/2510.11835v1](http://arxiv.org/pdf/2510.11835v1)**

> **作者:** Yiming Liu; Yuhui Zhang; Dhruba Ghosh; Ludwig Schmidt; Serena Yeung-Levy
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** CLIP outperforms self-supervised models like DINO as vision encoders for vision-language models (VLMs), but it remains unclear whether this advantage stems from CLIP's language supervision or its much larger training data. To disentangle these factors, we pre-train CLIP and DINO under controlled settings -- using the same architecture, dataset, and training configuration -- achieving similar ImageNet accuracy. Embedding analysis shows that CLIP captures high-level semantics (e.g., object categories, text), while DINO is more responsive to low-level features like colors and styles. When integrated into VLMs and evaluated on 20 VQA benchmarks, CLIP excels at text-intensive tasks, while DINO slightly outperforms on vision-centric ones. Variants of language supervision (e.g., sigmoid loss, pre-trained language encoders) yield limited gains. Our findings provide scientific insights into vision encoder design and its impact on VLM performance.
>
---
#### [new 086] Reasoning in the Dark: Interleaved Vision-Text Reasoning in Latent Space
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究多模态推理任务，旨在解决现有方法依赖人工标注、推理慢的问题。提出在隐空间中融合视觉与文本的交错隐式推理（IVT-LR），通过渐进式训练提升准确性和推理效率。**

- **链接: [http://arxiv.org/pdf/2510.12603v1](http://arxiv.org/pdf/2510.12603v1)**

> **作者:** Chao Chen; Zhixin Ma; Yongqi Li; Yupeng Hu; Yinwei Wei; Wenjie Li; Liqiang Nie
>
> **摘要:** Multimodal reasoning aims to enhance the capabilities of MLLMs by incorporating intermediate reasoning steps before reaching the final answer. It has evolved from text-only reasoning to the integration of visual information, enabling the thought process to be conveyed through both images and text. Despite its effectiveness, current multimodal reasoning methods depend on explicit reasoning steps that require labor-intensive vision-text annotations and inherently introduce significant inference latency. To address these issues, we introduce multimodal latent reasoning with the advantages of multimodal representation, reduced annotation, and inference efficiency. To facilicate it, we propose Interleaved Vision-Text Latent Reasoning (IVT-LR), which injects both visual and textual information in the reasoning process within the latent space. Specifically, IVT-LR represents each reasoning step by combining two implicit parts: latent text (the hidden states from the previous step) and latent vision (a set of selected image embeddings). We further introduce a progressive multi-stage training strategy to enable MLLMs to perform the above multimodal latent reasoning steps. Experiments on M3CoT and ScienceQA demonstrate that our IVT-LR method achieves an average performance increase of 5.45% in accuracy, while simultaneously achieving a speed increase of over 5 times compared to existing approaches. Code available at https://github.com/FYYDCC/IVT-LR.
>
---
## 更新

#### [replaced 001] COMAL: A Convergent Meta-Algorithm for Aligning LLMs with General Preferences
- **分类: cs.LG; cs.AI; cs.CL; cs.GT**

- **链接: [http://arxiv.org/pdf/2410.23223v2](http://arxiv.org/pdf/2410.23223v2)**

> **作者:** Yixin Liu; Argyris Oikonomou; Weiqiang Zheng; Yang Cai; Arman Cohan
>
> **摘要:** Many alignment methods, including reinforcement learning from human feedback (RLHF), rely on the Bradley-Terry reward assumption, which is not always sufficient to capture the full range and complexity of general human preferences. We explore RLHF under a general preference framework by modeling the alignment problem as a two-player zero-sum game in a game-theoretic framework, where the Nash equilibrium policy guarantees a 50% win rate against any competing policy. However, previous self-play algorithms for finding the Nash policy either diverge or only converge to a Nash policy in a modified game, even in a simple synthetic setting, thereby failing to maintain the 50% win rate guarantee against all other policies. We propose a meta-algorithm, Convergent Meta Alignment Algorithm (COMAL), for language model alignment with general preferences, inspired by convergent algorithms in game theory. We provide theoretical analysis that our meta-algorithm converges to an exact Nash policy in the last iterate and demonstrate its effectiveness on a range of synthetic and preference optimization datasets. COMAL is simple and can be integrated with many existing methods designed for preference optimization with minimal changes, and empirically it consistently maintains above 60.2% and 56.8% win rates, when applied to Llama-3-8B-Instruct and Qwen2.5-7B, against all compared algorithms under controlled evaluations.
>
---
#### [replaced 002] CiteBART: Learning to Generate Citations for Local Citation Recommendation
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.17534v3](http://arxiv.org/pdf/2412.17534v3)**

> **作者:** Ege Yiğit Çelik; Selma Tekir
>
> **备注:** This paper has been accepted to the EMNLP 2025 Main Conference. (19 pages, 3 figures, 11 tables)
>
> **摘要:** Local citation recommendation (LCR) suggests a set of papers for a citation placeholder within a given context. The task has evolved as generative approaches have become more promising than the traditional pre-fetch and re-rank-based state-of-the-art approaches. This paper introduces citation-specific pre-training within an encoder-decoder architecture, where author-date citation tokens are masked to learn to reconstruct them to fulfill LCR. There are two variants for this pre-training. In the local context-only base scheme (CiteBART-Base), the citation token in a local context is masked to learn to predict the citation. The global version (CiteBART-Global) extends the local context with the citing paper's title and abstract to enrich the learning signal. CiteBART-Global achieves state-of-the-art performance on LCR benchmarks except for the FullTextPeerRead dataset, which is quite small to see the advantage of generative pre-training. The effect is significant in the larger benchmarks, e.g., Refseer and ArXiv., with the Refseer benchmark-trained model emerging as the best-performing model. We perform comprehensive experiments, including an ablation study, a qualitative analysis, and a taxonomy of hallucinations with detailed statistics. Our analyses confirm that CiteBART-Global has a cross-dataset generalization capability; the macro hallucination rate (MaHR) at the top-3 predictions is 4\%, and when the ground-truth is in the top-k prediction list, the hallucination tendency in the other predictions drops significantly.
>
---
#### [replaced 003] Understanding the Mixture-of-Experts with Nadaraya-Watson Kernel
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.25913v2](http://arxiv.org/pdf/2509.25913v2)**

> **作者:** Chuanyang Zheng; Jiankai Sun; Yihang Gao; Enze Xie; Yuehao Wang; Peihao Wang; Ting Xu; Matthew Chang; Liliang Ren; Jingyao Li; Jing Xiong; Kashif Rasul; Mac Schwager; Anderson Schneider; Zhangyang Wang; Yuriy Nevmyvaka
>
> **备注:** Tech Report
>
> **摘要:** Mixture-of-Experts (MoE) has become a cornerstone in recent state-of-the-art large language models (LLMs). Traditionally, MoE relies on $\mathrm{Softmax}$ as the router score function to aggregate expert output, a designed choice that has persisted from the earliest MoE models to modern LLMs, and is now widely regarded as standard practice. However, the necessity of using $\mathrm{Softmax}$ to project router weights into a probability simplex remains an unchallenged assumption rather than a principled design choice. In this work, we first revisit the classical Nadaraya-Watson regression and observe that MoE shares the same mathematical formulation as Nadaraya-Watson regression. Furthermore, we show that both feed-forward neural network (FFN) and MoE can be interpreted as a special case of Nadaraya-Watson regression, where the kernel function corresponds to the input neurons of the output layer. Motivated by these insights, we propose the \textbf{zero-additional-cost} Kernel Inspired Router with Normalization (KERN), an FFN-style router function, as an alternative to $\mathrm{Softmax}$. We demonstrate that this router generalizes both $\mathrm{Sigmoid}$- and $\mathrm{Softmax}$-based routers. \textbf{Based on empirical observations and established practices in FFN implementation, we recommend the use of $\mathrm{ReLU}$ activation and $\ell_2$-normalization in $\mathrm{KERN}$ router function.} Comprehensive experiments in MoE and LLM validate the effectiveness of the proposed FFN-style router function \methodNorm.
>
---
#### [replaced 004] Hybrid Multi-stage Decoding for Few-shot NER with Entity-aware Contrastive Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.06970v2](http://arxiv.org/pdf/2404.06970v2)**

> **作者:** Congying Liu; Gaosheng Wang; Peipei Liu; Xingyuan Wei; Hongsong Zhu
>
> **摘要:** Few-shot named entity recognition can identify new types of named entities based on a few labeled examples. Previous methods employing token-level or span-level metric learning suffer from the computational burden and a large number of negative sample spans. In this paper, we propose the Hybrid Multi-stage Decoding for Few-shot NER with Entity-aware Contrastive Learning (MsFNER), which splits the general NER into two stages: entity-span detection and entity classification. There are 3 processes for introducing MsFNER: training, finetuning, and inference. In the training process, we train and get the best entity-span detection model and the entity classification model separately on the source domain using meta-learning, where we create a contrastive learning module to enhance entity representations for entity classification. During finetuning, we finetune the both models on the support dataset of target domain. In the inference process, for the unlabeled data, we first detect the entity-spans, then the entity-spans are jointly determined by the entity classification model and the KNN. We conduct experiments on the open FewNERD dataset and the results demonstrate the advance of MsFNER.
>
---
#### [replaced 005] OmniDraft: A Cross-vocabulary, Online Adaptive Drafter for On-device Speculative Decoding
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.02659v3](http://arxiv.org/pdf/2507.02659v3)**

> **作者:** Ramchalam Kinattinkara Ramakrishnan; Zhaocong Yuan; Shaojie Zhuo; Chen Feng; Yicheng Lin; Chenzheng Su; Xiaopeng Zhang
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Speculative decoding generally dictates having a small, efficient draft model that is either pretrained or distilled offline to a particular target model series, for instance, Llama or Qwen models. However, within online deployment settings, there are two major challenges: 1) usage of a target model that is incompatible with the draft model; 2) expectation of latency improvements over usage and time. In this work, we propose OmniDraft, a unified framework that enables a single draft model to operate with any target model and adapt dynamically to user data. We introduce an online n-gram cache with hybrid distillation fine-tuning to address the cross-vocabulary mismatch across draft and target models; and further improve decoding speed by leveraging adaptive drafting techniques. OmniDraft is particularly suitable for on-device LLM applications where model cost, efficiency and user customization are the major points of contention. This further highlights the need to tackle the above challenges and motivates the \textit{``one drafter for all''} paradigm. We showcase the proficiency of the OmniDraft framework by performing online learning on math reasoning, coding and text generation tasks. Notably, OmniDraft enables a single Llama-68M model to pair with various target models including Vicuna-7B, Qwen2-7B and Llama3-8B models for speculative decoding; and additionally provides up to 1.5-2x speedup.
>
---
#### [replaced 006] Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.23564v2](http://arxiv.org/pdf/2509.23564v2)**

> **作者:** Samuel Yeh; Sharon Li
>
> **备注:** NeurIPS 2025
>
> **摘要:** Human feedback plays a pivotal role in aligning large language models (LLMs) with human preferences. However, such feedback is often noisy or inconsistent, which can degrade the quality of reward models and hinder alignment. While various automated data cleaning methods have been proposed to mitigate this issue, a systematic evaluation of their effectiveness and generalizability remains lacking. To bridge this gap, we introduce the first comprehensive benchmark for evaluating 13 preference data cleaning methods in the context of LLM alignment. PrefCleanBench offers a standardized protocol to assess cleaning strategies in terms of alignment performance and generalizability across diverse datasets, model architectures, and optimization algorithms. By unifying disparate methods and rigorously comparing them, we uncover key factors that determine the success of data cleaning in alignment tasks. This benchmark lays the groundwork for principled and reproducible approaches to improving LLM alignment through better data quality-highlighting the crucial but underexplored role of data preprocessing in responsible AI development. We release modular implementations of all methods to catalyze further research: https://github.com/deeplearning-wisc/PrefCleanBench.
>
---
#### [replaced 007] Cross-Modal Safety Alignment: Is textual unlearning all you need?
- **分类: cs.CL; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.02575v2](http://arxiv.org/pdf/2406.02575v2)**

> **作者:** Trishna Chakraborty; Erfan Shayegani; Zikui Cai; Nael Abu-Ghazaleh; M. Salman Asif; Yue Dong; Amit K. Roy-Chowdhury; Chengyu Song
>
> **备注:** Accepted by EMNLP 2024 Findings
>
> **摘要:** Recent studies reveal that integrating new modalities into Large Language Models (LLMs), such as Vision-Language Models (VLMs), creates a new attack surface that bypasses existing safety training techniques like Supervised Fine-tuning (SFT) and Reinforcement Learning with Human Feedback (RLHF). While further SFT and RLHF-based safety training can be conducted in multi-modal settings, collecting multi-modal training datasets poses a significant challenge. Inspired by the structural design of recent multi-modal models, where, regardless of the combination of input modalities, all inputs are ultimately fused into the language space, we aim to explore whether unlearning solely in the textual domain can be effective for cross-modality safety alignment. Our evaluation across six datasets empirically demonstrates the transferability -- textual unlearning in VLMs significantly reduces the Attack Success Rate (ASR) to less than 8\% and in some cases, even as low as nearly 2\% for both text-based and vision-text-based attacks, alongside preserving the utility. Moreover, our experiments show that unlearning with a multi-modal dataset offers no potential benefits but incurs significantly increased computational demands, possibly up to 6 times higher.
>
---
#### [replaced 008] Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.19700v2](http://arxiv.org/pdf/2505.19700v2)**

> **作者:** Yi Liu; Dianqing Liu; Mingye Zhu; Junbo Guo; Yongdong Zhang; Zhendong Mao
>
> **备注:** Accepted by NeurIPS 2025, 28 pages
>
> **摘要:** The widespread adoption of large language models (LLMs) across industries has increased the demand for high-quality and customizable outputs. However, traditional alignment methods often require retraining large pretrained models, making it difficult to quickly adapt and optimize LLMs for diverse applications. To address this limitation, we propose a novel \textit{Residual Alignment Model} (\textit{RAM}) that formalizes the alignment process as a type of importance sampling. In this framework, the unaligned upstream model serves as the proposal distribution, while the alignment process is framed as secondary sampling based on an autoregressive alignment module that acts as an estimator of the importance weights. This design enables a natural detachment of the alignment module from the target aligned model, improving flexibility and scalability. Based on this model, we derive an efficient sequence-level training strategy for the alignment module, which operates independently of the proposal module. Additionally, we develop a resampling algorithm with iterative token-level decoding to address the common first-token latency issue in comparable methods. Experimental evaluations on two leading open-source LLMs across diverse tasks, including instruction following, domain adaptation, and preference optimization, demonstrate that our approach consistently outperforms baseline models.
>
---
#### [replaced 009] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.00299v5](http://arxiv.org/pdf/2502.00299v5)**

> **作者:** Xiang Liu; Zhenheng Tang; Peijie Dong; Zeyu Li; Yue Liu; Bo Li; Xuming Hu; Xiaowen Chu
>
> **备注:** NeurIPS 2025
>
> **摘要:** Large Language Models (LLMs) require significant GPU memory when processing long texts, with the key value (KV) cache consuming up to 70\% of total memory during inference. Although existing compression methods reduce memory by evaluating the importance of individual tokens, they overlook critical semantic relationships between tokens, resulting in fragmented context and degraded performance. We introduce ChunkKV, which fundamentally reimagines KV cache compression by treating semantic chunks - rather than isolated tokens - as basic compression units. This approach preserves complete linguistic structures and contextual integrity, ensuring that essential meaning is retained even under aggressive compression. Our innovation includes a novel layer-wise index reuse technique that exploits the higher cross-layer similarity of preserved indices in ChunkKV, reducing computational overhead and improving throughput by 26.5\%. Comprehensive evaluations on challenging benchmarks: LongBench, Needle-In-A-HayStack, GSM8K, and JailbreakV demonstrate that ChunkKV outperforms state-of-the-art methods by up to 8.7\% in precision while maintaining the same compression ratio. These results confirm that semantic-aware compression significantly enhances both efficiency and performance for long-context LLM inference, providing a simple yet effective solution to the memory bottleneck problem. The code is available at \href{https://github.com/NVIDIA/kvpress}{link}.
>
---
#### [replaced 010] LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21875v2](http://arxiv.org/pdf/2509.21875v2)**

> **作者:** Samuel Yeh; Sharon Li; Tanwi Mallick
>
> **摘要:** Retrieval-Augmented Generation (RAG) aims to mitigate hallucinations in large language models (LLMs) by grounding responses in retrieved documents. Yet, RAG-based LLMs still hallucinate even when provided with correct and sufficient context. A growing line of work suggests that this stems from an imbalance between how models use external context and their internal knowledge, and several approaches have attempted to quantify these signals for hallucination detection. However, existing methods require extensive hyperparameter tuning, limiting their generalizability. We propose LUMINA, a novel framework that detects hallucinations in RAG systems through context-knowledge signals: external context utilization is quantified via distributional distance, while internal knowledge utilization is measured by tracking how predicted tokens evolve across transformer layers. We further introduce a framework for statistically validating these measurements. Experiments on common RAG hallucination benchmarks and four open-source LLMs show that LUMINA achieves consistently high AUROC and AUPRC scores, outperforming prior utilization-based methods by up to +13% AUROC on HalluRAG. Moreover, LUMINA remains robust under relaxed assumptions about retrieval quality and model matching, offering both effectiveness and practicality.
>
---
#### [replaced 011] Are Large Reasoning Models Interruptible?
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.11713v2](http://arxiv.org/pdf/2510.11713v2)**

> **作者:** Tsung-Han Wu; Mihran Miroyan; David M. Chan; Trevor Darrell; Narges Norouzi; Joseph E. Gonzalez
>
> **备注:** We found a code/data bug in Section 5's intervene experiments. We're not sure how much it would affect the overall results and thus we're planning to fix it and do further investigation. As we do not want to leave any incorrect information on the internet, we want to withdraw this submission
>
> **摘要:** Large Reasoning Models (LRMs) excel at complex reasoning but are traditionally evaluated in static, "frozen world" settings: model responses are assumed to be instantaneous, and the context of a request is presumed to be immutable over the duration of the response. While generally true for short-term tasks, the "frozen world" assumption breaks down in modern reasoning tasks such as assistive programming, where models may take hours to think through problems and code may change dramatically from the time the model starts thinking to the model's final output. In this work, we challenge the frozen world assumption and evaluate LRM robustness under two realistic dynamic scenarios: interruptions, which test the quality of the model's partial outputs on a limited budget, and dynamic context, which tests model adaptation to in-flight changes. Across mathematics and programming benchmarks that require long-form reasoning, static evaluations consistently overestimate robustness: even state-of-the-art LRMs, which achieve high accuracy in static settings, can fail unpredictably when interrupted or exposed to changing context, with performance dropping by up to 60% when updates are introduced late in the reasoning process. Our analysis further reveals several novel failure modes, including reasoning leakage, where models fold the reasoning into their final answer when interrupted; panic, where under time pressure models abandon reasoning entirely and return incorrect answers; and self-doubt, where performance degrades while incorporating updated information.
>
---
#### [replaced 012] DiaCDM: Cognitive Diagnosis in Teacher-Student Dialogues using the Initiation-Response-Evaluation Framework
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.24821v3](http://arxiv.org/pdf/2509.24821v3)**

> **作者:** Rui Jia; Yuang Wei; Ruijia Li; Yuan-Hao Jiang; Xinyu Xie; Yaomin Shen; Min Zhang; Bo Jiang
>
> **摘要:** While cognitive diagnosis (CD) effectively assesses students' knowledge mastery from structured test data, applying it to real-world teacher-student dialogues presents two fundamental challenges. Traditional CD models lack a suitable framework for handling dynamic, unstructured dialogues, and it's difficult to accurately extract diagnostic semantics from lengthy dialogues. To overcome these hurdles, we propose DiaCDM, an innovative model. We've adapted the initiation-response-evaluation (IRE) framework from educational theory to design a diagnostic framework tailored for dialogue. We also developed a unique graph-based encoding method that integrates teacher questions with relevant knowledge components to capture key information more precisely. To our knowledge, this is the first exploration of cognitive diagnosis in a dialogue setting. Experiments on three real-world dialogue datasets confirm that DiaCDM not only significantly improves diagnostic accuracy but also enhances the results' interpretability, providing teachers with a powerful tool for assessing students' cognitive states. The code is available at https://github.com/Mind-Lab-ECNU/DiaCDM/tree/main.
>
---
#### [replaced 013] Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10844v3](http://arxiv.org/pdf/2505.10844v3)**

> **作者:** Simeng Han; Howard Dai; Stephen Xia; Grant Zhang; Chen Liu; Lichang Chen; Hoang Huy Nguyen; Hongyuan Mei; Jiayuan Mao; R. Thomas McCoy
>
> **备注:** 13 Tables; 5 Figures
>
> **摘要:** Accuracy remains a standard metric for evaluating AI systems, but it offers limited insight into how models arrive at their solutions. In this work, we introduce a benchmark based on brainteasers written in long narrative form to probe more deeply into the types of reasoning strategies that models use. Brainteasers are well-suited for this goal because they can be solved with multiple approaches, such as a few-step solution that uses a creative insight or a longer solution that uses more brute force. We investigate large language models (LLMs) across multiple layers of reasoning, focusing not only on correctness but also on the quality and creativity of their solutions. We investigate many aspects of the reasoning process: (1) semantic parsing of the brainteasers into precise mathematical competition style formats; (2) generating solutions from these mathematical forms; (3) self-correcting solutions based on gold solutions; (4) producing step-by-step sketches of solutions; and (5) making use of hints. We find that LLMs are in many cases able to find creative, insightful solutions to brainteasers, suggesting that they capture some of the capacities needed to solve novel problems in creative ways. Nonetheless, there also remain situations where they rely on brute force despite the availability of more efficient, creative solutions, highlighting a potential direction for improvement in the reasoning abilities of LLMs.
>
---
#### [replaced 014] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.02631v2](http://arxiv.org/pdf/2502.02631v2)**

> **作者:** Zechun Liu; Changsheng Zhao; Hanxian Huang; Sijia Chen; Jing Zhang; Jiawei Zhao; Scott Roy; Lisa Jin; Yunyang Xiong; Yangyang Shi; Lin Xiao; Yuandong Tian; Bilge Soran; Raghuraman Krishnamoorthi; Tijmen Blankevoort; Vikas Chandra
>
> **备注:** NeurIPS 2025. Model weights are available at https://huggingface.co/collections/facebook/mobilellm-6722be18cb86c20ebe113e95
>
> **摘要:** The optimal bit-width for achieving the best trade-off between quantized model size and accuracy has been a subject of ongoing debate. While some advocate for 4-bit quantization, others propose that 1.58-bit offers superior results. However, the lack of a cohesive framework for different bits has left such conclusions relatively tenuous. We present ParetoQ, the first unified framework that facilitates rigorous comparisons across 1-bit, 1.58-bit, 2-bit, 3-bit, and 4-bit quantization settings. Our findings reveal a notable learning transition between 2 and 3 bits: For 3-bits and above, the fine-tuned models stay close to their original pre-trained distributions, whereas for learning 2-bit networks or below, the representations change drastically. By optimizing training schemes and refining quantization functions, ParetoQ surpasses all previous methods tailored to specific bit widths. Remarkably, our ParetoQ ternary 600M-parameter model even outperforms the previous SoTA ternary 3B-parameter model in accuracy, using only one-fifth of the parameters. Extensive experimentation shows that ternary, 2-bit, and 3-bit quantization maintains comparable performance in the size-accuracy trade-off and generally exceeds 4-bit and binary quantization. Considering hardware constraints, 2-bit quantization offers promising potential for memory reduction and speedup.
>
---
#### [replaced 015] GRDD: A Dataset for Greek Dialectal NLP
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2308.00802v5](http://arxiv.org/pdf/2308.00802v5)**

> **作者:** Stergios Chatzikyriakidis; Chatrine Qwaider; Ilias Kolokousis; Christina Koula; Dimitris Papadakis; Efthymia Sakellariou
>
> **摘要:** In this paper, we present a dataset for the computational study of a number of Modern Greek dialects. It consists of raw text data from four dialects of Modern Greek, Cretan, Pontic, Northern Greek and Cypriot Greek. The dataset is of considerable size, albeit imbalanced, and presents the first attempt to create large scale dialectal resources of this type for Modern Greek dialects. We then use the dataset to perform dialect idefntification. We experiment with traditional ML algorithms, as well as simple DL architectures. The results show very good performance on the task, potentially revealing that the dialects in question have distinct enough characteristics allowing even simple ML models to perform well on the task. Error analysis is performed for the top performing algorithms showing that in a number of cases the errors are due to insufficient dataset cleaning.
>
---
#### [replaced 016] From Rational Answers to Emotional Resonance: The Role of Controllable Emotion Generation in Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.04075v2](http://arxiv.org/pdf/2502.04075v2)**

> **作者:** Yurui Dong; Luozhijie Jin; Yao Yang; Bingjie Lu; Jiaxi Yang; Zhi Liu
>
> **备注:** 43 pages, 5 figures
>
> **摘要:** Purpose: Emotion is a fundamental component of human communication, shaping understanding, trust, and engagement across domains such as education, healthcare, and mental health. While large language models (LLMs) exhibit strong reasoning and knowledge generation capabilities, they still struggle to express emotions in a consistent, controllable, and contextually appropriate manner. This limitation restricts their potential for authentic human-AI interaction. Methods: We propose a controllable emotion generation framework based on Emotion Vectors (EVs) - latent representations derived from internal activation shifts between neutral and emotion-conditioned responses. By injecting these vectors into the hidden states of pretrained LLMs during inference, our method enables fine-grained, continuous modulation of emotional tone without any additional training or architectural modification. We further provide theoretical analysis proving that EV steering enhances emotional expressivity while maintaining semantic fidelity and linguistic fluency. Results: Extensive experiments across multiple LLM families show that the proposed approach achieves consistent emotional alignment, stable topic adherence, and controllable affect intensity. Compared with existing prompt-based and fine-tuning-based baselines, our method demonstrates superior flexibility and generalizability. Conclusion: Emotion Vector (EV) steering provides an efficient and interpretable means of bridging rational reasoning and affective understanding in large language models, offering a promising direction for building emotionally resonant AI systems capable of more natural human-machine interaction.
>
---
#### [replaced 017] EmoDebt: Bayesian-Optimized Emotional Intelligence for Strategic Agent-to-Agent Debt Recovery
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.21080v5](http://arxiv.org/pdf/2503.21080v5)**

> **作者:** Yunbo Long; Yuhan Liu; Liming Xu; Alexandra Brintrup
>
> **摘要:** The emergence of autonomous Large Language Model (LLM) agents has created a new ecosystem of strategic, agent-to-agent interactions. However, a critical challenge remains unaddressed: in high-stakes, emotion-sensitive domains like debt collection, LLM agents pre-trained on human dialogue are vulnerable to exploitation by adversarial counterparts who simulate negative emotions to derail negotiations. To fill this gap, we first contribute a novel dataset of simulated debt recovery scenarios and a multi-agent simulation framework. Within this framework, we introduce EmoDebt, an LLM agent architected for robust performance. Its core innovation is a Bayesian-optimized emotional intelligence engine that reframes a model's ability to express emotion in negotiation as a sequential decision-making problem. Through online learning, this engine continuously tunes EmoDebt's emotional transition policies, discovering optimal counter-strategies against specific debtor tactics. Extensive experiments on our proposed benchmark demonstrate that EmoDebt achieves significant strategic robustness, substantially outperforming non-adaptive and emotion-agnostic baselines across key performance metrics, including success rate and operational efficiency. By introducing both a critical benchmark and a robustly adaptive agent, this work establishes a new foundation for deploying strategically robust LLM agents in adversarial, emotion-sensitive debt interactions.
>
---
#### [replaced 018] MLRIP: Pre-training a military language representation model with informative factual knowledge and professional knowledge base
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2207.13929v2](http://arxiv.org/pdf/2207.13929v2)**

> **作者:** Hui Li; Xuekang Yang
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Incorporating structured knowledge into pre-trained language models has demonstrated signiffcant bene-ffts for domain-speciffc natural language processing tasks, particularly in specialized ffelds like military intelligence analysis. Existing approaches typically integrate external knowledge through masking tech-niques or fusion mechanisms, but often fail to fully leverage the intrinsic tactical associations and factual information within input sequences, while introducing uncontrolled noise from unveriffed exter-nal sources. To address these limitations, we present MLRIP (Military Language Representation with Integrated Prior), a novel pre-training framework that introduces a hierarchical knowledge integration pipeline combined with a dual-phase entity substitu-tion mechanism. Our approach speciffcally models operational linkages between military entities, capturing critical dependencies such as command, support, and engagement structures. Comprehensive evaluations on military-speciffc NLP tasks show that MLRIP outperforms existing BERT-based models by substantial margins, establishing new state-of-the-art performance in military entity recognition, typing, and operational linkage extraction tasks while demonstrating superior operational efffciency in resource-constrained environments.
>
---
#### [replaced 019] BrowserAgent: Building Web Agents with Human-Inspired Web Browsing Actions
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.10666v2](http://arxiv.org/pdf/2510.10666v2)**

> **作者:** Tao Yu; Zhengbo Zhang; Zhiheng Lyu; Junhao Gong; Hongzhu Yi; Xinming Wang; Yuxuan Zhou; Jiabing Yang; Ping Nie; Yan Huang; Wenhu Chen
>
> **备注:** 10 pages
>
> **摘要:** Efficiently solving real-world problems with LLMs increasingly hinges on their ability to interact with dynamic web environments and autonomously acquire external information. While recent research like Search-R1 and WebDancer demonstrates strong performance in solving web tasks, they heavily rely on additional tools to convert the interactive web environment into static text content. This is in contrast to human browsing behaviors, which involve diverse interactions with the browser, such as scrolling, clicking, and typing. In this paper, we propose BrowserAgent, a more interactive agent that solves complex tasks through human-inspired browser actions. BrowserAgent operates directly on raw web pages via Playwright through a set of predefined browser actions. We adopt a two-stage training (Supervised Fine-Tuning (SFT) and Rejection Fine-Tuning (RFT)) to improve the model's generalization abilities. Despite using significantly less training data than Search-R1, BrowserAgent achieves more competitive results across different Open-QA tasks. Additionally, we introduce an explicit memory mechanism to store key conclusions across steps, further enhancing the model's reasoning capabilities for long-horizon tasks. Notably, BrowserAgent-7B can achieve around 20\% improvement over Search-R1 on multi-hop QA tasks like HotpotQA, 2Wiki, and Bamboogle. These results indicate that BrowserAgent can serve as a more advanced framework for more interactive and scalable web agents.
>
---
#### [replaced 020] Model-Based Ranking of Source Languages for Zero-Shot Cross-Lingual Transfer
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.03202v2](http://arxiv.org/pdf/2510.03202v2)**

> **作者:** Abteen Ebrahimi; Adam Wiemerslage; Katharina von der Wense
>
> **备注:** Accepted to EMNLP 2025 (Main)
>
> **摘要:** We present NN-Rank, an algorithm for ranking source languages for cross-lingual transfer, which leverages hidden representations from multilingual models and unlabeled target-language data. We experiment with two pretrained multilingual models and two tasks: part-of-speech tagging (POS) and named entity recognition (NER). We consider 51 source languages and evaluate on 56 and 72 target languages for POS and NER, respectively. When using in-domain data, NN-Rank beats state-of-the-art baselines that leverage lexical and linguistic features, with average improvements of up to 35.56 NDCG for POS and 18.14 NDCG for NER. As prior approaches can fall back to language-level features if target language data is not available, we show that NN-Rank remains competitive using only the Bible, an out-of-domain corpus available for a large number of languages. Ablations on the amount of unlabeled target data show that, for subsets consisting of as few as 25 examples, NN-Rank produces high-quality rankings which achieve 92.8% of the NDCG achieved using all available target data for ranking.
>
---
#### [replaced 021] Efficient and Versatile Model for Multilingual Information Retrieval of Islamic Text: Development and Deployment in Real-World Scenarios
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.15380v2](http://arxiv.org/pdf/2509.15380v2)**

> **作者:** Vera Pavlova; Mohammed Makhlouf
>
> **摘要:** Despite recent advancements in Multilingual Information Retrieval (MLIR), a significant gap remains between research and practical deployment. Many studies assess MLIR performance in isolated settings, limiting their applicability to real-world scenarios. In this work, we leverage the unique characteristics of the Quranic multilingual corpus to examine the optimal strategies to develop an ad-hoc IR system for the Islamic domain that is designed to satisfy users' information needs in multiple languages. We prepared eleven retrieval models employing four training approaches: monolingual, cross-lingual, translate-train-all, and a novel mixed method combining cross-lingual and monolingual techniques. Evaluation on an in-domain dataset demonstrates that the mixed approach achieves promising results across diverse retrieval scenarios. Furthermore, we provide a detailed analysis of how different training configurations affect the embedding space and their implications for multilingual retrieval effectiveness. Finally, we discuss deployment considerations, emphasizing the cost-efficiency of deploying a single versatile, lightweight model for real-world MLIR applications.
>
---
#### [replaced 022] SAFER: Probing Safety in Reward Models with Sparse Autoencoder
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.00665v2](http://arxiv.org/pdf/2507.00665v2)**

> **作者:** Sihang Li; Wei Shi; Ziyuan Xie; Tao Liang; Guojun Ma; Xiang Wang
>
> **备注:** One of the institutions requires additional approval before we can move forward with the publication. Thanks for your understanding, and we hope to resubmit once everything is finalized
>
> **摘要:** Reinforcement learning from human feedback (RLHF) is a key paradigm for aligning large language models (LLMs) with human values, yet the reward models at its core remain largely opaque. In this work, we present sparse Autoencoder For Enhanced Reward model (\textbf{SAFER}), a novel framework for interpreting and improving reward models through mechanistic analysis. Leveraging Sparse Autoencoders (SAEs), we uncover human-interpretable features in reward model activations, enabling insight into safety-relevant decision-making. We apply SAFER to safety-oriented preference datasets and quantify the salience of individual features by activation differences between chosen and rejected responses. Using these feature-level signals, we design targeted data poisoning and denoising strategies. Experiments show that SAFER can precisely degrade or enhance safety alignment with minimal data modification, without sacrificing general chat performance. Our approach contributes to interpreting, auditing and refining reward models in high-stakes LLM alignment tasks. Our codes are available at https://github.com/xzy-101/SAFER-code. \textit{This paper discusses topics related to large language model safety and may include discussions or examples that highlight potential risks or unsafe outcomes.}
>
---
#### [replaced 023] Large language models management of medications: three performance analyses
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.22926v2](http://arxiv.org/pdf/2509.22926v2)**

> **作者:** Kelli Henry; Steven Xu; Kaitlin Blotske; Moriah Cargile; Erin F. Barreto; Brian Murray; Susan Smith; Seth R. Bauer; Xingmeng Zhao; Adeleine Tilley; Yanjun Gao; Tianming Liu; Sunghwan Sohn; Andrea Sikora
>
> **摘要:** Purpose: Large language models (LLMs) have proven performance for certain diagnostic tasks, however limited studies have evaluated their consistency in recommending appropriate medication regimens for a given diagnosis. Medication management is a complex task that requires synthesis of drug formulation and complete order instructions for safe use. Here, the performance of GPT 4o, an LLM available with ChatGPT, was tested for three medication management tasks. Methods: GPT-4o performance was tested using three medication tasks: identifying available formulations for a given generic drug name, identifying drug-drug interactions (DDI) for a given medication regimen, and preparing a medication order for a given generic drug name. For each experiment, the models raw text response was captured exactly as returned and evaluated using clinician evaluation in addition to standard LLM metrics, including Term Frequency-Inverse Document Frequency (TF IDF) vectors, normalized Levenshtein similarity, and Recall-Oriented Understudy for Gisting Evaluation (ROUGE 1/ROUGE L F1) between each response and its reference string. Results: For the first task of drug-formulation matching, GPT-4o had 49% accuracy for generic medications being matched to all available formulations, with an average of 1.23 omissions per medication and 1.14 hallucinations per medication. For the second task of drug-drug interaction identification, the accuracy was 54.7% for identifying the DDI pair. For the third task, GPT-4o generated order sentences containing no medication or abbreviation errors in 65.8% of cases. Conclusions: Model performance for basic medication tasks was consistently poor. This evaluation highlights the need for domain-specific training through clinician-annotated datasets and a comprehensive evaluation framework for benchmarking performance.
>
---
#### [replaced 024] KaLM-Embedding-V2: Superior Training Techniques and Data Inspire A Versatile Embedding Model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.20923v5](http://arxiv.org/pdf/2506.20923v5)**

> **作者:** Xinping Zhao; Xinshuo Hu; Zifei Shan; Shouzheng Huang; Yao Zhou; Xin Zhang; Zetian Sun; Zhenyu Liu; Dongfang Li; Xinyuan Wei; Youcheng Pan; Yang Xiang; Meishan Zhang; Haofen Wang; Jun Yu; Baotian Hu; Min Zhang
>
> **备注:** 32 pages, 16 tables, 5 figures
>
> **摘要:** Recent advancements in Large Language Models (LLMs)-based text embedding models primarily focus on data scaling or synthesis, yet limited exploration of training techniques and data quality, thereby constraining performance. In this work, we propose KaLM-Embedding-V2, a series of versatile and compact embedding models, systematically incentivizing advanced embedding capability in LLMs by superior training techniques and high-quality data. For model architecture, we implement the models on a 0.5B compact size with simple mean-pooling to produce fixed-length embeddings and remove the causal attention mask to enable fully bidirectional representation learning. For training techniques, we propose a progressive multi-stage training pipeline: pre-training on weakly supervised large-scale datasets, fine-tuning with supervised high-quality datasets, and contrastive distillation with fine-grained soft signals, integrated with focal-style reweighting and online hard-negative mixing to emphasize difficult samples and enrich hard negatives, respectively. For training data, we curate over 20 categories for pre-training and 100 categories for fine-tuning and contrastive distillation, to improve both performance and generalization, leveraging task-specific instructions, hard-negative mining, and example-based multi-class labeling to ensure high quality. Combining these techniques, our KaLM-Embedding-V2 series achieves state-of-the-art performance on the Massive Text Embedding Benchmark, outperforming models of comparable size and rivaling models 3-26x larger, setting a new standard for versatile and compact embedding models under 1B parameters.
>
---
#### [replaced 025] DRIFT: Decompose, Retrieve, Illustrate, then Formalize Theorems
- **分类: cs.AI; cs.CL; cs.IR; cs.SC**

- **链接: [http://arxiv.org/pdf/2510.10815v2](http://arxiv.org/pdf/2510.10815v2)**

> **作者:** Meiru Zhang; Philipp Borchert; Milan Gritta; Gerasimos Lampouras
>
> **摘要:** Automating the formalization of mathematical statements for theorem proving remains a major challenge for Large Language Models (LLMs). LLMs struggle to identify and utilize the prerequisite mathematical knowledge and its corresponding formal representation in languages like Lean. Current retrieval-augmented autoformalization methods query external libraries using the informal statement directly, but overlook a fundamental limitation: informal mathematical statements are often complex and offer limited context on the underlying math concepts. To address this, we introduce DRIFT, a novel framework that enables LLMs to decompose informal mathematical statements into smaller, more tractable ''sub-components''. This facilitates targeted retrieval of premises from mathematical libraries such as Mathlib. Additionally, DRIFT retrieves illustrative theorems to help models use premises more effectively in formalization tasks. We evaluate DRIFT across diverse benchmarks (ProofNet, ConNF, and MiniF2F-test) and find that it consistently improves premise retrieval, nearly doubling the F1 score compared to the DPR baseline on ProofNet. Notably, DRIFT demonstrates strong performance on the out-of-distribution ConNF benchmark, with BEq+@10 improvements of 37.14% and 42.25% using GPT-4.1 and DeepSeek-V3.1, respectively. Our analysis shows that retrieval effectiveness in mathematical autoformalization depends heavily on model-specific knowledge boundaries, highlighting the need for adaptive retrieval strategies aligned with each model's capabilities.
>
---
#### [replaced 026] General Exploratory Bonus for Optimistic Exploration in RLHF
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.03269v2](http://arxiv.org/pdf/2510.03269v2)**

> **作者:** Wendi Li; Changdae Oh; Sharon Li
>
> **摘要:** Optimistic exploration is central to improving sample efficiency in reinforcement learning with human feedback, yet existing exploratory bonus methods to incentivize exploration often fail to realize optimism. We provide a theoretical analysis showing that current formulations, under KL or $\alpha$-divergence regularization, unintentionally bias exploration toward high-probability regions of the reference model, thereby reinforcing conservative behavior instead of promoting discovery of uncertain regions. To address this pitfall, we introduce the General Exploratory Bonus (GEB), a novel theoretical framework that provably satisfies the optimism principle. GEB counteracts divergence-induced bias via reference-dependent reward regulation and unifies prior heuristic bonuses as special cases, while extending naturally across the full $\alpha$-divergence family. Empirically, GEB consistently outperforms baselines on alignment tasks across multiple divergence settings and large language model backbones. These results demonstrate that GEB offers both a principled and practical solution for optimistic exploration in RLHF.
>
---
#### [replaced 027] DSPO: Stable and Efficient Policy Optimization for Agentic Search and Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.09255v3](http://arxiv.org/pdf/2510.09255v3)**

> **作者:** Chenyang Gu; Yewen Pu; Bruce Yang; Xiaofan Li; Huan Gao
>
> **摘要:** Enhancing LLMs with the ability to actively search external knowledge is crucial for complex and real-world tasks. Current approaches either rely on prompting to elicit the model's innate agent capabilities, or suffer from performance ceilings and collapse when applying RL to complex interactive tasks, leaving their true agentic potential untapped. To address this, we introduce \textbf{D}ynamic-filter \textbf{S}equence-level \textbf{P}olicy \textbf{O}ptimization (DSPO), an improved RL algorithm designed for robust agent training through sequence-level optimization and dynamic sample filtering. We train our model purely through RL to interleave multi-turn search and reasoning, obviating the need for supervised demonstration data. Across multiple QA benchmarks, our 7B model improves over a comparable previous work by \textbf{34.1\%}, and even outperforms the 14B model from previous work in complex multihop QA such as HotpotQA by nearly \textbf{9\% relative}, maintaining exceptional training stability.
>
---
#### [replaced 028] Assessing Latency in ASR Systems: A Methodological Perspective for Real-Time Use
- **分类: cs.SD; cs.AI; cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2409.05674v3](http://arxiv.org/pdf/2409.05674v3)**

> **作者:** Carlos Arriaga; Alejandro Pozo; Javier Conde; Alvaro Alonso
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** Automatic speech recognition (ASR) systems generate real-time transcriptions but often miss nuances that human interpreters capture. While ASR is useful in many contexts, interpreters-who already use ASR tools such as Dragon-add critical value, especially in sensitive settings such as diplomatic meetings where subtle language is key. Human interpreters not only perceive these nuances but can adjust in real time, improving accuracy, while ASR handles basic transcription tasks. However, ASR systems introduce a delay that does not align with real-time interpretation needs. The user-perceived latency of ASR systems differs from that of interpretation because it measures the time between speech and transcription delivery. To address this, we propose a new approach to measuring delay in ASR systems and validate if they are usable in live interpretation scenarios.
>
---
#### [replaced 029] SCAN: Self-Denoising Monte Carlo Annotation for Robust Process Reward Learning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.16548v2](http://arxiv.org/pdf/2509.16548v2)**

> **作者:** Yuyang Ding; Xinyu Shi; Juntao Li; Xiaobo Liang; Zhaopeng Tu; Min Zhang
>
> **备注:** NeurIPS 2025. Project page: https://scan-prm.github.io/
>
> **摘要:** Process reward models (PRMs) offer fine-grained, step-level evaluations that facilitate deeper reasoning processes in large language models (LLMs), proving effective in complex tasks like mathematical reasoning. However, developing PRMs is challenging due to the high cost and limited scalability of human-annotated data. Synthetic data from Monte Carlo (MC) estimation is a promising alternative but suffers from a high noise ratio, which can cause overfitting and hinder large-scale training. In this work, we conduct a preliminary study on the noise distribution in synthetic data from MC estimation, identifying that annotation models tend to both underestimate and overestimate step correctness due to limitations in their annotation capabilities. Building on these insights, we propose Self-Denoising Monte Carlo Annotation (SCAN), an efficient data synthesis and noise-tolerant learning framework. Our key findings indicate that: (1) Even lightweight models (e.g., 1.5B parameters) can produce high-quality annotations through a self-denoising strategy, enabling PRMs to achieve superior performance with only 6% the inference cost required by vanilla MC estimation. (2) With our robust learning strategy, PRMs can effectively learn from this weak supervision, achieving a 39.2 F1 score improvement (from 19.9 to 59.1) in ProcessBench. Despite using only a compact synthetic dataset, our models surpass strong baselines, including those trained on large-scale human-annotated datasets such as PRM800K. Furthermore, performance continues to improve as we scale up the synthetic data, highlighting the potential of SCAN for scalable, cost-efficient, and robust PRM training.
>
---
#### [replaced 030] FlagEval Findings Report: A Preliminary Evaluation of Large Reasoning Models on Automatically Verifiable Textual and Visual Questions
- **分类: cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.17177v2](http://arxiv.org/pdf/2509.17177v2)**

> **作者:** Bowen Qin; Chen Yue; Fang Yin; Hui Wang; JG Yao; Jiakang Liu; Jing-Shu Zheng; Miguel Hu Chen; Richeng Xuan; Shibei Meng; Shiqi Zhou; Teng Dai; Tong-Shuai Ren; Wei Cui; Xi Yang; Xialin Du; Xiaojing Xu; Xue Sun; Xuejing Li; Yaming Liu; Yesheng Liu; Ying Liu; Yonghua Lin; Yu Zhao; Yunduo Zhang; Yuwen Luo; Zheqi He; Zhiyuan He; Zhongyuan Wang
>
> **备注:** Project homepage: https://flageval-baai.github.io/LRM-Eval/ This work will also be presented at NeurIPS 2025 Workshop on Foundations of Reasoning in Language Models (FoRLM)
>
> **摘要:** We conduct a moderate-scale contamination-free (to some extent) evaluation of current large reasoning models (LRMs) with some preliminary findings. We also release ROME, our evaluation benchmark for vision language models intended to test reasoning from visual clues. We attach links to the benchmark, evaluation data, and other updates on this website: https://flageval-baai.github.io/LRM-Eval/
>
---
#### [replaced 031] Persuasion at Play: Understanding Misinformation Dynamics in Demographic-Aware Human-LLM Interactions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.02038v2](http://arxiv.org/pdf/2503.02038v2)**

> **作者:** Angana Borah; Rada Mihalcea; Verónica Pérez-Rosas
>
> **摘要:** Existing challenges in misinformation exposure and susceptibility vary across demographic groups, as some populations are more vulnerable to misinformation than others. Large language models (LLMs) introduce new dimensions to these challenges through their ability to generate persuasive content at scale and reinforcing existing biases. This study investigates the bidirectional persuasion dynamics between LLMs and humans when exposed to misinformative content. We analyze human-to-LLM influence using human-stance datasets and assess LLM-to-human influence by generating LLM-based persuasive arguments. Additionally, we use a multi-agent LLM framework to analyze the spread of misinformation under persuasion among demographic-oriented LLM agents. Our findings show that demographic factors influence susceptibility to misinformation in LLMs, closely reflecting the demographic-based patterns seen in human susceptibility. We also find that, similar to human demographic groups, multi-agent LLMs exhibit echo chamber behavior. This research explores the interplay between humans and LLMs, highlighting demographic differences in the context of misinformation and offering insights for future interventions.
>
---
#### [replaced 032] Responsible AI Technical Report
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.20057v3](http://arxiv.org/pdf/2509.20057v3)**

> **作者:** KT; :; Yunjin Park; Jungwon Yoon; Junhyung Moon; Myunggyo Oh; Wonhyuk Lee; Sujin Kim Youngchol Kim; Eunmi Kim; Hyoungjun Park; Eunyoung Shin; Wonyoung Lee; Somin Lee; Minwook Ju; Minsung Noh; Dongyoung Jeong; Jeongyeop Kim; Wanjin Park; Soonmin Bae
>
> **备注:** 23 pages, 8 figures
>
> **摘要:** KT developed a Responsible AI (RAI) assessment methodology and risk mitigation technologies to ensure the safety and reliability of AI services. By analyzing the Basic Act on AI implementation and global AI governance trends, we established a unique approach for regulatory compliance and systematically identify and manage all potential risk factors from AI development to operation. We present a reliable assessment methodology that systematically verifies model safety and robustness based on KT's AI risk taxonomy tailored to the domestic environment. We also provide practical tools for managing and mitigating identified AI risks. With the release of this report, we also release proprietary Guardrail : SafetyGuard that blocks harmful responses from AI models in real-time, supporting the enhancement of safety in the domestic AI development ecosystem. We also believe these research outcomes provide valuable insights for organizations seeking to develop Responsible AI.
>
---
#### [replaced 033] Exploring the Frontier of Vision-Language Models: A Survey of Current Methodologies and Future Directions
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2404.07214v4](http://arxiv.org/pdf/2404.07214v4)**

> **作者:** Akash Ghosh; Arkadeep Acharya; Sriparna Saha; Vinija Jain; Aman Chadha
>
> **备注:** One of the first survey on Visual Language Models
>
> **摘要:** The advent of Large Language Models (LLMs) has significantly reshaped the trajectory of the AI revolution. Nevertheless, these LLMs exhibit a notable limitation, as they are primarily adept at processing textual information. To address this constraint, researchers have endeavored to integrate visual capabilities with LLMs, resulting in the emergence of Vision-Language Models (VLMs). These advanced models are instrumental in tackling more intricate tasks such as image captioning and visual question answering. In our comprehensive survey paper, we delve into the key advancements within the realm of VLMs. Our classification organizes VLMs into three distinct categories: models dedicated to vision-language understanding, models that process multimodal inputs to generate unimodal (textual) outputs and models that both accept and produce multimodal inputs and outputs.This classification is based on their respective capabilities and functionalities in processing and generating various modalities of data.We meticulously dissect each model, offering an extensive analysis of its foundational architecture, training data sources, as well as its strengths and limitations wherever possible, providing readers with a comprehensive understanding of its essential components. We also analyzed the performance of VLMs in various benchmark datasets. By doing so, we aim to offer a nuanced understanding of the diverse landscape of VLMs. Additionally, we underscore potential avenues for future research in this dynamic domain, anticipating further breakthroughs and advancements.
>
---
#### [replaced 034] VIDEE: Visual and Interactive Decomposition, Execution, and Evaluation of Text Analytics with Intelligent Agents
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.21582v4](http://arxiv.org/pdf/2506.21582v4)**

> **作者:** Sam Yu-Te Lee; Chenyang Ji; Shicheng Wen; Lifu Huang; Dongyu Liu; Kwan-Liu Ma
>
> **摘要:** Text analytics has traditionally required specialized knowledge in Natural Language Processing (NLP) or text analysis, which presents a barrier for entry-level analysts. Recent advances in large language models (LLMs) have changed the landscape of NLP by enabling more accessible and automated text analysis (e.g., topic detection, summarization, information extraction, etc.). We introduce VIDEE, a system that supports entry-level data analysts to conduct advanced text analytics with intelligent agents. VIDEE instantiates a human-agent collaroration workflow consisting of three stages: (1) Decomposition, which incorporates a human-in-the-loop Monte-Carlo Tree Search algorithm to support generative reasoning with human feedback, (2) Execution, which generates an executable text analytics pipeline, and (3) Evaluation, which integrates LLM-based evaluation and visualizations to support user validation of execution results. We conduct two quantitative experiments to evaluate VIDEE's effectiveness and analyze common agent errors. A user study involving participants with varying levels of NLP and text analytics experience -- from none to expert -- demonstrates the system's usability and reveals distinct user behavior patterns. The findings identify design implications for human-agent collaboration, validate the practical utility of VIDEE for non-expert users, and inform future improvements to intelligent text analytics systems.
>
---
#### [replaced 035] MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18943v3](http://arxiv.org/pdf/2505.18943v3)**

> **作者:** Xuanming Zhang; Yuxuan Chen; Samuel Yeh; Sharon Li
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** Human social interactions depend on the ability to infer others' unspoken intentions, emotions, and beliefs-a cognitive skill grounded in the psychological concept of Theory of Mind (ToM). While large language models (LLMs) excel in semantic understanding tasks, they struggle with the ambiguity and contextual nuance inherent in human communication. To bridge this gap, we introduce MetaMind, a multi-agent framework inspired by psychological theories of metacognition, designed to emulate human-like social reasoning. MetaMind decomposes social understanding into three collaborative stages: (1) a Theory-of-Mind Agent generates hypotheses about user mental states (e.g., intent, emotion), (2) a Moral Agent refines these hypotheses using cultural norms and ethical constraints, and (3) a Response Agent generates contextually appropriate responses while validating alignment with inferred intent. Our framework achieves state-of-the-art performance across three challenging benchmarks, with 35.7% improvement in real-world social scenarios and 6.2% gain in ToM reasoning. Notably, it enables LLMs to match human-level performance on key ToM tasks for the first time. Ablation studies confirm the necessity of all components, which showcase the framework's ability to balance contextual plausibility, social appropriateness, and user adaptation. This work advances AI systems toward human-like social intelligence, with applications in empathetic dialogue and culturally sensitive interactions. Code is available at https://github.com/XMZhangAI/MetaMind.
>
---
#### [replaced 036] AGENTIQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.10661v2](http://arxiv.org/pdf/2510.10661v2)**

> **作者:** Omid Reza Heidari; Siobhan Reid; Yassine Yaakoubi
>
> **备注:** Accepted at NeurIPS 2025, ER "Efficient Reasoning" workshop
>
> **摘要:** LLMs have advanced text-to-SQL generation, yet monolithic architectures struggle with complex reasoning and schema diversity. We propose AGENTIQL, an agent-inspired multi-expert framework that combines a reasoning agent for question decomposition, a coding agent for sub-query generation, and a refinement step for column selection. An adaptive router further balances efficiency and accuracy by selecting between our modular pipeline and a baseline parser. Several steps in the pipeline can be executed in parallel, making the framework scalable to larger workloads. Evaluated on the Spider benchmark, AGENTIQL improves execution accuracy and interpretability and achieves up to 86.07% EX with 14B models using the Planner&Executor merging strategy. The attained performance is contingent upon the efficacy of the routing mechanism, thereby narrowing the gap to GPT-4-based SOTA (89.65% EX) while using much smaller open-source LLMs. Beyond accuracy, AGENTIQL enhances transparency by exposing intermediate reasoning steps, offering a robust, scalable, and interpretable approach to semantic parsing.
>
---
#### [replaced 037] DFAMS: Dynamic-flow guided Federated Alignment based Multi-prototype Search
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.20353v2](http://arxiv.org/pdf/2508.20353v2)**

> **作者:** Zhibang Yang; Xinke Jiang; Rihong Qiu; Ruiqing Li; Yihang Zhang; Yue Fang; Yongxin Xu; Hongxin Ding; Xu Chu; Junfeng Zhao; Yasha Wang
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Federated Retrieval (FR) routes queries across multiple external knowledge sources, to mitigate hallucinations of LLMs, when necessary external knowledge is distributed. However, existing methods struggle to retrieve high-quality and relevant documents for ambiguous queries, especially in cross-domain scenarios, which significantly limits their effectiveness in supporting downstream generation tasks. Inspired by Dynamic Information Flow (DIF), we propose DFAMS, a novel framework that leverages DIF to identify latent query intents and construct semantically aligned knowledge partitions for accurate retrieval across heterogeneous sources. Specifically, DFAMS probes the DIF in LLMs by leveraging gradient signals from a few annotated queries and employing Shapley value-based attribution to trace neuron activation paths associated with intent recognition and subdomain boundary detection. Then, DFAMS leverages DIF to train an alignment module via multi-prototype contrastive learning, enabling fine-grained intra-source modeling and inter-source semantic alignment across knowledge bases. Experimental results across five benchmarks show that DFAMS outperforms advanced FR methods by up to 14.37\% in knowledge classification accuracy, 5.38\% in retrieval recall, and 6.45\% in downstream QA accuracy, demonstrating its effectiveness in complex FR scenarios. Our code are anonymous available at https://anonymous.4open.science/r/DFAMS/
>
---
#### [replaced 038] Diffusion Language Models Know the Answer Before Decoding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.19982v3](http://arxiv.org/pdf/2508.19982v3)**

> **作者:** Pengxiang Li; Yefan Zhou; Dilxat Muhtar; Lu Yin; Shilin Yan; Li Shen; Yi Liang; Soroush Vosoughi; Shiwei Liu
>
> **摘要:** Diffusion language models (DLMs) have recently emerged as an alternative to autoregressive approaches, offering parallel sequence generation and flexible token orders. However, their inference remains slower than that of autoregressive models, primarily due to the cost of bidirectional attention and the large number of refinement steps required for high quality outputs. In this work, we highlight and leverage an overlooked property of DLMs early answer convergence: in many cases, the correct answer can be internally identified by half steps before the final decoding step, both under semi-autoregressive and random remasking schedules. For example, on GSM8K and MMLU, up to 97% and 99% of instances, respectively, can be decoded correctly using only half of the refinement steps. Building on this observation, we introduce Prophet, a training-free fast decoding paradigm that enables early commit decoding. Specifically, Prophet dynamically decides whether to continue refinement or to go "all-in" (i.e., decode all remaining tokens in one step), using the confidence gap between the top-2 prediction candidates as the criterion. It integrates seamlessly into existing DLM implementations, incurs negligible overhead, and requires no additional training. Empirical evaluations of LLaDA-8B and Dream-7B across multiple tasks show that Prophet reduces the number of decoding steps by up to 3.4x while preserving high generation quality. These results recast DLM decoding as a problem of when to stop sampling, and demonstrate that early decode convergence provides a simple yet powerful mechanism for accelerating DLM inference, complementary to existing speedup techniques. Our code is publicly available at https://github.com/pixeli99/Prophet.
>
---
#### [replaced 039] TripScore: Benchmarking and rewarding real-world travel planning with fine-grained evaluation
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.09011v2](http://arxiv.org/pdf/2510.09011v2)**

> **作者:** Yincen Qu; Huan Xiao; Feng Li; Gregory Li; Hui Zhou; Xiangying Dai
>
> **摘要:** Travel planning is a valuable yet complex task that poses significant challenges even for advanced large language models (LLMs). While recent benchmarks have advanced in evaluating LLMs' planning capabilities, they often fall short in evaluating feasibility, reliability, and engagement of travel plans. We introduce a comprehensive benchmark for travel planning that unifies fine-grained criteria into a single reward, enabling direct comparison of plan quality and seamless integration with reinforcement learning (RL). Our evaluator achieves moderate agreement with travel-expert annotations (60.75%) and outperforms multiple LLM-as-judge baselines. We further release a large-scale dataset of 4,870 queries including 219 real-world, free-form requests for generalization to authentic user intent. Using this benchmark, we conduct extensive experiments across diverse methods and LLMs, including test-time computation, neuro-symbolic approaches, supervised fine-tuning, and RL via GRPO. Across base models, RL generally improves itinerary feasibility over prompt-only and supervised baselines, yielding higher unified reward scores.
>
---
#### [replaced 040] Attention-Aware GNN-based Input Defense against Multi-Turn LLM Jailbreak
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.07146v2](http://arxiv.org/pdf/2507.07146v2)**

> **作者:** Zixuan Huang; Kecheng Huang; Lihao Yin; Bowei He; Huiling Zhen; Mingxuan Yuan; Zili Shao
>
> **摘要:** Large Language Models (LLMs) have gained significant traction in various applications, yet their capabilities present risks for both constructive and malicious exploitation. Despite extensive training and fine-tuning efforts aimed at enhancing safety, LLMs remain susceptible to jailbreak attacks. Recently, the emergence of multi-turn attacks has intensified this vulnerability. Unlike single-turn attacks, multi-turn attacks incrementally escalate dialogue complexity, rendering them more challenging to detect and mitigate. In this study, we introduce G-Guard, an innovative attention-aware Graph Neural Network (GNN)-based input classifier specifically designed to defend against multi-turn jailbreak attacks targeting LLMs. G-Guard constructs an entity graph for multi-turn queries, which captures the interrelationships between queries and harmful keywords that present in multi-turn queries. Furthermore, we propose an attention-aware augmentation mechanism that retrieves the most relevant single-turn query based on the ongoing multi-turn conversation. The retrieved query is incorporated as a labeled node within the graph, thereby enhancing the GNN's capacity to classify the current query as harmful or benign. Evaluation results show that G-Guard consistently outperforms all baselines across diverse datasets and evaluation metrics, demonstrating its efficacy as a robust defense mechanism against multi-turn jailbreak attacks.
>
---
#### [replaced 041] Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15877v2](http://arxiv.org/pdf/2505.15877v2)**

> **作者:** Siting Li; Xiang Gao; Simon Shaolei Du
>
> **备注:** NeurIPS 2025; 27 pages, 6 figures
>
> **摘要:** While an image is worth more than a thousand words, only a few provide crucial information for a given task and thus should be focused on. In light of this, ideal text-to-image (T2I) retrievers should prioritize specific visual attributes relevant to queries. To evaluate current retrievers on handling attribute-focused queries, we build COCO-Facet, a COCO-based benchmark with 9,112 queries about diverse attributes of interest. We find that CLIP-like retrievers, which are widely adopted due to their efficiency and zero-shot ability, have poor and imbalanced performance, possibly because their image embeddings focus on global semantics and subjects while leaving out other details. Notably, we reveal that even recent Multimodal Large Language Model (MLLM)-based, stronger retrievers with a larger output dimension struggle with this limitation. Hence, we hypothesize that retrieving with general image embeddings is suboptimal for performing such queries. As a solution, we propose to use promptable image embeddings enabled by these multimodal retrievers, which boost performance by highlighting required attributes. Our pipeline for deriving such embeddings generalizes across query types, image pools, and base retriever architectures. To enhance real-world applicability, we offer two acceleration strategies: Pre-processing promptable embeddings and using linear approximations. We show that the former yields a 15% improvement in Recall@5 when prompts are predefined, while the latter achieves an 8% improvement when prompts are only available during inference.
>
---
#### [replaced 042] CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.02322v3](http://arxiv.org/pdf/2508.02322v3)**

> **作者:** Yuzhuang Xu; Xu Han; Yuanchi Zhang; Yixuan Wang; Yijun Liu; Shiyu Ji; Qingfu Zhu; Wanxiang Che
>
> **备注:** 16 pages, 9 figures, 7 tables
>
> **摘要:** Large Language Models (LLMs) with Mixture-of-Experts (MoE) architectures are distinguished by their strong performance scaling with increasing parameters across a wide range of tasks, yet they also suffer from substantial computational and storage overheads. Notably, the performance gains of MoE models do not scale proportionally with the growth in expert parameters. While prior works attempt to reduce parameters via expert-level pruning, merging, or decomposition, they still suffer from challenges in both performance and computational efficiency. In this paper, we address these challenges by introducing micro-expert as a finer-grained compression unit that spans across matrices. We first establish a more fundamental perspective, viewing MoE layers as mixtures of micro-experts, and present CAMERA, a lightweight and training-free framework for identifying micro-expert redundancy. Our analysis uncovers significant variance in micro-expert contributions during decoding. Based on this insight, we further propose CAMERA-P, a structured micro-expert pruning framework, and CAMERA-Q, a mixed-precision quantization idea designed for micro-experts. Extensive experiments on nine downstream tasks show that CAMERA-P consistently outperforms strong baselines under pruning ratios ranging from 20% to 60%. Furthermore, CAMERA-Q achieves superior results under aggressive 2-bit quantization, surpassing existing matrix- and channel-level ideas. Notably, our method enables complete micro-expert analysis of Qwen2-57B-A14B in less than 5 minutes on a single NVIDIA A100-40GB GPU.
>
---
#### [replaced 043] The Cultural Gene of Large Language Models: A Study on the Impact of Cross-Corpus Training on Model Values and Biases
- **分类: cs.CL; I.2.7; K.4.1; H.3.3**

- **链接: [http://arxiv.org/pdf/2508.12411v2](http://arxiv.org/pdf/2508.12411v2)**

> **作者:** Emanuel Z. Fenech-Borg; Tilen P. Meznaric-Kos; Milica D. Lekovic-Bojovic; Arni J. Hentze-Djurhuus
>
> **备注:** 10 pages, 5 figures, IEEE conference format, submitted to [Conference Name]
>
> **摘要:** Large language models (LLMs) are deployed globally, yet their underlying cultural and ethical assumptions remain underexplored. We propose the notion of a "cultural gene" -- a systematic value orientation that LLMs inherit from their training corpora -- and introduce a Cultural Probe Dataset (CPD) of 200 prompts targeting two classic cross-cultural dimensions: Individualism-Collectivism (IDV) and Power Distance (PDI). Using standardized zero-shot prompts, we compare a Western-centric model (GPT-4) and an Eastern-centric model (ERNIE Bot). Human annotation shows significant and consistent divergence across both dimensions. GPT-4 exhibits individualistic and low-power-distance tendencies (IDV score approx 1.21; PDI score approx -1.05), while ERNIE Bot shows collectivistic and higher-power-distance tendencies (IDV approx -0.89; PDI approx 0.76); differences are statistically significant (p < 0.001). We further compute a Cultural Alignment Index (CAI) against Hofstede's national scores and find GPT-4 aligns more closely with the USA (e.g., IDV CAI approx 0.91; PDI CAI approx 0.88) whereas ERNIE Bot aligns more closely with China (IDV CAI approx 0.85; PDI CAI approx 0.81). Qualitative analyses of dilemma resolution and authority-related judgments illustrate how these orientations surface in reasoning. Our results support the view that LLMs function as statistical mirrors of their cultural corpora and motivate culturally aware evaluation and deployment to avoid algorithmic cultural hegemony.
>
---
#### [replaced 044] EMSEdit: Efficient Multi-Step Meta-Learning-based Model Editing
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.04012v2](http://arxiv.org/pdf/2508.04012v2)**

> **作者:** Xiaopeng Li; Shasha Li; Xi Wang; Shezheng Song; Bin Ji; Shangwen Wang; Jun Ma; Xiaodong Liu; Mina Liu; Jie Yu
>
> **摘要:** Large Language Models (LLMs) power numerous AI applications, yet updating their knowledge remains costly. Model editing provides a lightweight alternative through targeted parameter modifications, with meta-learning-based model editing (MLME) demonstrating strong effectiveness and efficiency. However, we find that MLME struggles in low-data regimes and incurs high training costs due to the use of KL divergence. To address these issues, we propose $\textbf{E}$fficient $\textbf{M}$ulti-$\textbf{S}$tep $\textbf{Edit (EMSEdit)}$, which leverages multi-step backpropagation (MSBP) to effectively capture gradient-activation mapping patterns within editing samples, performs multi-step edits per sample to enhance editing performance under limited data, and introduces norm-based regularization to preserve unedited knowledge while improving training efficiency. Experiments on two datasets and three LLMs show that EMSEdit consistently outperforms state-of-the-art methods in both sequential and batch editing. Moreover, MSBP can be seamlessly integrated into existing approaches to yield additional performance gains. Further experiments on a multi-hop reasoning editing task demonstrate EMSEdit's robustness in handling complex edits, while ablation studies validate the contribution of each design component. Our code is available at https://github.com/xpq-tech/emsedit.
>
---
#### [replaced 045] Lost at the Beginning of Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.22058v2](http://arxiv.org/pdf/2506.22058v2)**

> **作者:** Baohao Liao; Xinyi Chen; Sara Rajaee; Yuhui Xu; Christian Herold; Anders Søgaard; Maarten de Rijke; Christof Monz
>
> **备注:** remove the benchmark part. (10 pages, 6 figures, 5 tables)
>
> **摘要:** Recent advancements in large language models (LLMs) have significantly advanced complex reasoning capabilities, particularly through extended chain-of-thought (CoT) reasoning that incorporates mechanisms such as backtracking, self-reflection, and self-correction. Despite these developments, the self-correction abilities of LLMs during long CoT reasoning remain underexplored. And recent findings on overthinking suggest that such models often engage in unnecessarily redundant reasoning. In this work, we empirically show that the first reasoning step exerts a disproportionately large influence on the final prediction. I.e., errors introduced at this stage can substantially degrade subsequent reasoning quality. This phenomenon is consistently observed across various state-of-the-art open- and closed-source reasoning models. Leveraging this insight, we propose an efficient sampling strategy that leverages a reward model to identify and retain high-quality first reasoning steps while discarding suboptimal ones, achieving up to a 70% reduction in inference cost without sacrificing any accuracy. Our work highlights the central role of the first reasoning step in generating a high-quality reasoning trajectory, and thus enabling significantly efficient sampling.
>
---
#### [replaced 046] AFRIDOC-MT: Document-level MT Corpus for African Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.06374v2](http://arxiv.org/pdf/2501.06374v2)**

> **作者:** Jesujoba O. Alabi; Israel Abebe Azime; Miaoran Zhang; Cristina España-Bonet; Rachel Bawden; Dawei Zhu; David Ifeoluwa Adelani; Clement Oyeleke Odoje; Idris Akinade; Iffat Maab; Davis David; Shamsuddeen Hassan Muhammad; Neo Putini; David O. Ademuyiwa; Andrew Caines; Dietrich Klakow
>
> **备注:** EMNLP 2025
>
> **摘要:** This paper introduces AFRIDOC-MT, a document-level multi-parallel translation dataset covering English and five African languages: Amharic, Hausa, Swahili, Yor\`ub\'a, and Zulu. The dataset comprises 334 health and 271 information technology news documents, all human-translated from English to these languages. We conduct document-level translation benchmark experiments by evaluating neural machine translation (NMT) models and large language models (LLMs) for translations between English and these languages, at both the sentence and pseudo-document levels. These outputs are realigned to form complete documents for evaluation. Our results indicate that NLLB-200 achieved the best average performance among the standard NMT models, while GPT-4o outperformed general-purpose LLMs. Fine-tuning selected models led to substantial performance gains, but models trained on sentences struggled to generalize effectively to longer documents. Furthermore, our analysis reveals that some LLMs exhibit issues such as under-generation, repetition of words or phrases, and off-target translations, especially for African languages.
>
---
#### [replaced 047] Knowledge Fusion via Bidirectional Information Aggregation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.08704v2](http://arxiv.org/pdf/2507.08704v2)**

> **作者:** Songlin Zhai; Guilin Qi; Yue Wang; Yuan Meng
>
> **摘要:** Knowledge graphs (KGs) are the cornerstone of the semantic web, offering up-to-date representations of real-world entities and relations. Yet large language models (LLMs) remain largely static after pre-training, causing their internal knowledge to become outdated and limiting their utility in time-sensitive web applications. To bridge this gap between dynamic knowledge and static models, a prevalent approach is to enhance LLMs with KGs. However, prevailing methods typically rely on parameter-invasive fine-tuning, which risks catastrophic forgetting and often degrades LLMs' general capabilities. Moreover, their static integration frameworks cannot keep pace with the continuous evolution of real-world KGs, hindering their deployment in dynamic web environments. To bridge this gap, we introduce KGA (\textit{\underline{K}nowledge \underline{G}raph-guided \underline{A}ttention}), a novel framework that dynamically integrates external KGs into LLMs exclusively at inference-time without any parameter modification. Inspired by research on neuroscience, we rewire the self-attention module by innovatively introducing two synergistic pathways: a \textit{bottom-up knowledge fusion} pathway and a \textit{top-down attention guidance} pathway. The \textit{bottom-up pathway} dynamically integrates external knowledge into input representations via input-driven KG fusion, which is akin to the \textit{stimulus-driven attention process} in the human brain. Complementarily, the \textit{top-down pathway} aims to assess the contextual relevance of each triple through a \textit{goal-directed verification process}, thereby suppressing task-irrelevant signals and amplifying knowledge-relevant patterns. By synergistically combining these two pathways, our method supports real-time knowledge fusion. Extensive experiments on four benchmarks verify KGA's strong fusion performance and efficiency.
>
---
#### [replaced 048] Revela: Dense Retriever Learning via Language Modeling
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.16552v2](http://arxiv.org/pdf/2506.16552v2)**

> **作者:** Fengyu Cai; Tong Chen; Xinran Zhao; Sihao Chen; Hongming Zhang; Sherry Tongshuang Wu; Iryna Gurevych; Heinz Koeppl
>
> **摘要:** Dense retrievers play a vital role in accessing external and specialized knowledge to augment language models (LMs). Training dense retrievers typically requires annotated query-document pairs, which are costly to create and scarce in specialized domains (e.g., code) or in complex settings (e.g., requiring reasoning). These practical challenges have sparked growing interest in self-supervised retriever learning. Since LMs are trained to capture token-level dependencies through a self-supervised learning objective (i.e., next token prediction), we can analogously cast retrieval as learning dependencies among chunks of tokens. This analogy naturally leads to the question: How can we adapt self-supervised learning objectives in the spirit of language modeling to train retrievers? To answer this question, we introduce Revela, a unified and scalable training framework for self-supervised retriever learning via language modeling. Revela models semantic dependencies among documents by conditioning next token prediction on local and cross-document context through an in-batch attention mechanism. This attention is weighted by retriever-computed similarity scores, enabling the retriever to be optimized as part of language modeling. We evaluate Revela on domain-specific (CoIR), reasoning-intensive (BRIGHT), and general-domain (BEIR) benchmarks across various retriever backbones. Without annotated or synthetic query-document pairs, Revela surpasses larger supervised models and proprietary APIs on CoIR and matches them on BRIGHT. It achieves BEIR's unsupervised SoTA with ~ 1000x less training data and 10x less compute. Performance increases with batch size and model size, highlighting Revela's scalability and its promise for self-supervised retriever learning.
>
---
#### [replaced 049] AgentAda: Skill-Adaptive Data Analytics for Tailored Insight Discovery
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07421v3](http://arxiv.org/pdf/2504.07421v3)**

> **作者:** Amirhossein Abaskohi; Amrutha Varshini Ramesh; Shailesh Nanisetty; Chirag Goel; David Vazquez; Christopher Pal; Spandana Gella; Giuseppe Carenini; Issam H. Laradji
>
> **摘要:** We introduce AgentAda, the first LLM-powered analytics agent that can learn and use new analytics skills to extract more specialized insights. Unlike existing methods that require users to manually decide which data analytics method to apply, AgentAda automatically identifies the skill needed from a library of analytical skills to perform the analysis. This also allows AgentAda to use skills that existing LLMs cannot perform out of the box. The library covers a range of methods, including clustering, predictive modeling, and NLP techniques like BERT, which allow AgentAda to handle complex analytics tasks based on what the user needs. AgentAda's dataset-to-insight extraction strategy consists of three key steps: (I) a question generator to generate queries relevant to the user's goal and persona, (II) a hybrid Retrieval-Augmented Generation (RAG)-based skill matcher to choose the best data analytics skill from the skill library, and (III) a code generator that produces executable code based on the retrieved skill's documentation to extract key patterns. We also introduce KaggleBench, a benchmark of curated notebooks across diverse domains, to evaluate AgentAda's performance. We conducted a human evaluation demonstrating that AgentAda provides more insightful analytics than existing tools, with 48.78% of evaluators preferring its analyses, compared to 27.67% for the unskilled agent. We also propose a novel LLM-as-a-judge approach that we show is aligned with human evaluation as a way to automate insight quality evaluation at larger scale.
>
---
#### [replaced 050] Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.26074v2](http://arxiv.org/pdf/2509.26074v2)**

> **作者:** Leitian Tao; Xuefeng Du; Sharon Li
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Reward modeling, crucial for aligning large language models (LLMs) with human preferences, is often bottlenecked by the high cost of preference data. Existing textual data synthesis methods are computationally expensive. We propose a novel framework LENS for synthesizing preference data directly in the LLM's latent embedding space. Our method employs a Variational Autoencoder (VAE) to learn a structured latent representation of response embeddings. By performing controlled perturbations in this latent space and decoding back to the embedding space, we efficiently generate diverse, semantically consistent synthetic preference pairs, bypassing costly text generation and annotation. We provide theoretical guarantees that our synthesized pairs approximately preserve original preference ordering and improve reward model generalization. Empirically, our latent-space synthesis significantly outperforms text-based augmentation on standard benchmarks, achieving superior results while being 18x faster in generation and using a 16,000x smaller model. Our work offers a scalable and effective alternative for enhancing reward modeling through efficient data augmentation. Code is publicly available at https://github.com/deeplearning-wisc/lens
>
---
#### [replaced 051] Agent Learning via Early Experience
- **分类: cs.AI; cs.CL; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.08558v2](http://arxiv.org/pdf/2510.08558v2)**

> **作者:** Kai Zhang; Xiangchao Chen; Bo Liu; Tianci Xue; Zeyi Liao; Zhihan Liu; Xiyao Wang; Yuting Ning; Zhaorun Chen; Xiaohan Fu; Jian Xie; Yuxuan Sun; Boyu Gou; Qi Qi; Zihang Meng; Jianwei Yang; Ning Zhang; Xian Li; Ashish Shah; Dat Huynh; Hengduo Li; Zi Yang; Sara Cao; Lawrence Jang; Shuyan Zhou; Jiacheng Zhu; Huan Sun; Jason Weston; Yu Su; Yifan Wu
>
> **备注:** Work in progress
>
> **摘要:** A long-term goal of language agents is to learn and improve through their own experience, ultimately outperforming humans in complex, real-world tasks. However, training agents from experience data with reinforcement learning remains difficult in many environments, which either lack verifiable rewards (e.g., websites) or require inefficient long-horizon rollouts (e.g., multi-turn tool use). As a result, most current agents rely on supervised fine-tuning on expert data, which is challenging to scale and generalizes poorly. This limitation stems from the nature of expert demonstrations: they capture only a narrow range of scenarios and expose the agent to limited environment diversity. We address this limitation with a middle-ground paradigm we call early experience: interaction data generated by the agent's own actions, where the resulting future states serve as supervision without reward signals. Within this paradigm we study two strategies of using such data: (1) Implicit world modeling, which uses collected states to ground the policy in environment dynamics; and (2) Self-reflection, where the agent learns from its suboptimal actions to improve reasoning and decision-making. We evaluate across eight diverse environments and multiple model families. Our approaches consistently improve effectiveness and out-of-domain generalization, highlighting the value of early experience. Moreover, in environments with verifiable rewards, our results provide promising signals that early experience offers a strong foundation for subsequent reinforcement learning, positioning it as a practical bridge between imitation learning and fully experience-driven agents.
>
---
#### [replaced 052] Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.00507v2](http://arxiv.org/pdf/2510.00507v2)**

> **作者:** Yurun Chen; Xavier Hu; Yuhan Liu; Ziqi Wang; Zeyi Liao; Lin Chen; Feng Wei; Yuxi Qian; Bo Zheng; Keting Yin; Shengyu Zhang
>
> **备注:** 20 pages, 10 figures. Our Code: https://github.com/YurunChen/Graph2Eval
>
> **摘要:** As multimodal LLM-driven agents continue to advance in autonomy and generalization, evaluation based on static datasets can no longer adequately assess their true capabilities in dynamic environments and diverse tasks. Existing LLM-based synthetic data methods are largely designed for LLM training and evaluation, and thus cannot be directly applied to agent tasks that require tool use and interactive capabilities. While recent studies have explored automatic agent task generation with LLMs, most efforts remain limited to text or image analysis, without systematically modeling multi-step interactions in web environments. To address these challenges, we propose Graph2Eval, a knowledge graph-based framework that automatically generates both multimodal document comprehension tasks and web interaction tasks, enabling comprehensive evaluation of agents' reasoning, collaboration, and interactive capabilities. In our approach, knowledge graphs constructed from multi-source external data serve as the task space, where we translate semantic relations into structured multimodal tasks using subgraph sampling, task templates, and meta-paths. A multi-stage filtering pipeline based on node reachability, LLM scoring, and similarity analysis is applied to guarantee the quality and executability of the generated tasks. Furthermore, Graph2Eval supports end-to-end evaluation of multiple agent types (Single-Agent, Multi-Agent, Web Agent) and measures reasoning, collaboration, and interaction capabilities. We instantiate the framework with Graph2Eval-Bench, a curated dataset of 1,319 tasks spanning document comprehension and web interaction scenarios. Experiments show that Graph2Eval efficiently generates tasks that differentiate agent and model performance, revealing gaps in reasoning, collaboration, and web interaction across different settings and offering a new perspective for agent evaluation.
>
---
#### [replaced 053] EvolveNav: Empowering LLM-Based Vision-Language Navigation via Self-Improving Embodied Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01551v3](http://arxiv.org/pdf/2506.01551v3)**

> **作者:** Bingqian Lin; Yunshuang Nie; Khun Loun Zai; Ziming Wei; Mingfei Han; Rongtao Xu; Minzhe Niu; Jianhua Han; Hanwang Zhang; Liang Lin; Bokui Chen; Cewu Lu; Xiaodan Liang
>
> **摘要:** Recent studies have revealed the potential of training open-source Large Language Models (LLMs) to unleash LLMs' reasoning ability for enhancing vision-language navigation (VLN) performance, and simultaneously mitigate the domain gap between LLMs' training corpus and the VLN task. However, these approaches predominantly adopt straightforward input-output mapping paradigms, causing the mapping learning difficult and the navigational decisions unexplainable. Chain-of-Thought (CoT) training is a promising way to improve both navigational decision accuracy and interpretability, while the complexity of the navigation task makes the perfect CoT labels unavailable and may lead to overfitting through pure CoT supervised fine-tuning. To address these issues, we propose EvolveNav, a novel sElf-improving embodied reasoning paradigm that realizes adaptable and generalizable navigational reasoning for boosting LLM-based vision-language Navigation. Specifically, EvolveNav involves a two-stage training process: (1) Formalized CoT Supervised Fine-Tuning, where we train the model with curated formalized CoT labels to first activate the model's navigational reasoning capabilities, and simultaneously increase the reasoning speed; (2) Self-Reflective Post-Training, where the model is iteratively trained with its own reasoning outputs as self-enriched CoT labels to enhance the supervision diversity. A self-reflective auxiliary task is also designed to encourage the model to learn correct reasoning patterns by contrasting with wrong ones. Experimental results under both task-specific and cross-task training paradigms demonstrate the consistent superiority of EvolveNav over previous LLM-based VLN approaches on various popular benchmarks, including R2R, REVERIE, CVDN, and SOON. Code is available at https://github.com/expectorlin/EvolveNav.
>
---
#### [replaced 054] MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.17238v2](http://arxiv.org/pdf/2509.17238v2)**

> **作者:** Soheil Zibakhsh; Mohammad Samragh; Kumari Nishu; Lauren Hannah; Arnav Kundu; Minsik Cho
>
> **备注:** Corrected typo in arxiv abstract
>
> **摘要:** The generation quality of large language models (LLMs) is often improved by utilizing inference-time sequence-level scaling methods (e.g., Chain-of-Thought). We introduce hyper-parallel scaling, a complementary framework that improves prediction quality at the token level. Hyper-parallel scaling computes and aggregates multiple output proposals for a single token from the model. We implement this concept in Mixture-of-Experts (MoE) models, which we refer to as Roster of Experts (RoE). RoE is a training-free inference algorithm that turns a single MoE into a dynamic ensemble of MoEs. RoE injects controlled stochasticity into the expert routing mechanism, enabling it to sample multiple diverse experts for each token and aggregate their outputs for a more accurate final prediction. To overcome the computational cost, we introduce an efficient batching strategy and a specialized KV-caching mechanism that minimizes compute and memory overhead. For example, RoE enables a 7B MoE model to match the performance of a 10.5B MoE model while using 30% less compute for inference. These gains are achieved without any fine-tuning of model parameters.
>
---
#### [replaced 055] Language Modeling for the Future of Finance: A Survey into Metrics, Tasks, and Data Opportunities
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07274v3](http://arxiv.org/pdf/2504.07274v3)**

> **作者:** Nikita Tatarinov; Siddhant Sukhani; Agam Shah; Sudheer Chava
>
> **摘要:** Recent advances in language modeling have led to a growing number of papers related to finance in top-tier Natural Language Processing (NLP) venues. To systematically examine this trend, we review 374 NLP research papers published between 2017 and 2024 across 38 conferences and workshops, with a focused analysis of 221 papers that directly address finance-related tasks. We evaluate these papers across 11 quantitative and qualitative dimensions, and our study identifies the following opportunities for NLP researchers: (i) expanding the scope of forecasting tasks; (ii) enriching evaluation with financial metrics; (iii) leveraging multilingual and crisis-period datasets; and (iv) balancing PLMs with efficient or interpretable alternatives. We identify actionable directions supported by dataset and tool recommendations, with implications for both the academia and industry communities.
>
---
#### [replaced 056] The Silent Judge: Unacknowledged Shortcut Bias in LLM-as-a-Judge
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.26072v2](http://arxiv.org/pdf/2509.26072v2)**

> **作者:** Arash Marioriyad; Mohammad Hossein Rohban; Mahdieh Soleymani Baghshah
>
> **备注:** 9 Pages, 5 Tables, 1 Figures
>
> **摘要:** Large language models (LLMs) are increasingly deployed as automatic judges to evaluate system outputs in tasks such as summarization, dialogue, and creative writing. A faithful judge should base its verdicts solely on response quality and explicitly acknowledge the factors shaping its decision. We show that current LLM judges fail on both counts by relying on shortcuts introduced in the prompt. Our study uses two evaluation datasets: ELI5, a benchmark for long-form question answering, and LitBench, a recent benchmark for creative writing. Both datasets provide pairwise comparisons, where the evaluator must choose which of two responses is better. From each dataset we construct 100 pairwise judgment tasks and employ two widely used models, GPT-4o and Gemini-2.5-Flash, as evaluators in the role of LLM-as-a-judge. For each pair, we assign superficial cues to the responses, provenance cues indicating source identity (Human, Expert, LLM, or Unknown) and recency cues indicating temporal origin (Old, 1950 vs. New, 2025), while keeping the rest of the prompt fixed. Results reveal consistent verdict shifts: both models exhibit a strong recency bias, systematically favoring new responses over old, as well as a clear provenance hierarchy (Expert > Human > LLM > Unknown). These biases are especially pronounced in GPT-4o and in the more subjective and open-ended LitBench domain. Crucially, cue acknowledgment is rare: justifications almost never reference the injected cues, instead rationalizing decisions in terms of content qualities. These findings demonstrate that current LLM-as-a-judge systems are shortcut-prone and unfaithful, undermining their reliability as evaluators in both research and deployment.
>
---
#### [replaced 057] LazyEviction: Lagged KV Eviction with Attention Pattern Observation for Efficient Long Reasoning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.15969v2](http://arxiv.org/pdf/2506.15969v2)**

> **作者:** Haoyue Zhang; Hualei Zhang; Xiaosong Ma; Jie Zhang; Song Guo
>
> **摘要:** Large Language Models (LLMs) exhibit enhanced capabilities by Chain-of-Thought reasoning. However, the extended reasoning sequences introduce significant GPU memory overhead due to increased key-value (KV) cache. Existing KV cache compression methods mitigate memory bottlenecks but struggle in long reasoning tasks. In this paper, we analyze attention patterns in reasoning tasks and reveal a \textbf{Token Importance Recurrence} phenomenon: a large proportion of tokens regain high attention after multiple decoding steps, which is failed to capture by existing works and may lead to unpredictable eviction on such periodically critical tokens. To address this, we propose \textbf{LazyEviction}, an observation window-based lagged eviction framework retaining latent recurring tokens by prioritized eviction based on tokens' recurrence patterns. Extensive experiments demonstrate that LazyEviction reduces KV cache by 50\%\textasciitilde70\% while maintaining comparable accuracy, outperforming existing KV cache compression baselines. Our implementation code can be found at https://github.com/Halo-949/LazyEviction.
>
---
#### [replaced 058] Exploring Compositional Generalization (in COGS/ReCOGS_pos) by Transformers using Restricted Access Sequence Processing (RASP)
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.15349v3](http://arxiv.org/pdf/2504.15349v3)**

> **作者:** William Bruns
>
> **备注:** 8 pages main text with 3 figures and 4 tables; limitations page and references separate; 10 more figures, 1 image, and 3 more tables in the appendices supplement the work
>
> **摘要:** Humans understand new combinations of words encountered if they are combinations of words recognized from different contexts, an ability called Compositional Generalization. The COGS benchmark (Kim and Linzen, 2020) arXiv:2010.05465 reports 0% accuracy for Transformer models on some structural generalizations. We use (Weiss et al., 2021) arXiv:2106.06981's Restricted Access Sequence Processing (RASP), a Transformer-equivalent programming language, to demonstrate that a Transformer Encoder-Decoder can perform COGS and the semantically equivalent ReCOGS_pos (Wu et al., 2024) arXiv:2303.13716 systematically and compositionally: Our RASP models attain near perfect scores on structural generalization splits on COGS (exact match) and ReCOGS_pos (semantic exact match). Our RASP models show the (Re)COGS tasks do not require a hierarchical or tree-structured solution (contrary to (Kim and Linzen, 2020) arXiv:2010.05465, (Yao and Koller, 2022) arXiv:2210.13050, (Murty et al., 2022) arXiv:2211.01288, (Liu et al., 2021) arXiv:2107.06516): we use word-level tokens with an "embedding" layer that tags with possible part of speech, applying just once per encoder pass 19 attention-head compatible flat pattern-matching rules (easily identified with specific training examples), shown using grammar coverage (Zeller et al., 2023) to cover the non-recursive aspects of the input grammar, plus masking out prepositional phrases ("pp noun") and/or sentential complements (cp) when recognizing grammar patterns and extracting nouns related to the main verb in the sentence, and output the next logical form (LF) token (repeating until the LF is complete). The models do not apply recursive, tree-structured rules like "np_det pp np -> np_pp -> np", but score near perfect semantic and string exact match on both COGS and ReCOGS pp recursion, cp recursion using the decoder loop.
>
---
#### [replaced 059] A Comprehensive Taxonomy of Negation for NLP and Neural Retrievers
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2507.22337v3](http://arxiv.org/pdf/2507.22337v3)**

> **作者:** Roxana Petcu; Samarth Bhargav; Maarten de Rijke; Evangelos Kanoulas
>
> **摘要:** Understanding and solving complex reasoning tasks is vital for addressing the information needs of a user. Although dense neural models learn contextualised embeddings, they still underperform on queries containing negation. To understand this phenomenon, we study negation in both traditional neural information retrieval and LLM-based models. We (1) introduce a taxonomy of negation that derives from philosophical, linguistic, and logical definitions; (2) generate two benchmark datasets that can be used to evaluate the performance of neural information retrieval models and to fine-tune models for a more robust performance on negation; and (3) propose a logic-based classification mechanism that can be used to analyze the performance of retrieval models on existing datasets. Our taxonomy produces a balanced data distribution over negation types, providing a better training setup that leads to faster convergence on the NevIR dataset. Moreover, we propose a classification schema that reveals the coverage of negation types in existing datasets, offering insights into the factors that might affect the generalization of fine-tuned models on negation.
>
---
#### [replaced 060] Reasoning on a Spectrum: Aligning LLMs to System 1 and System 2 Thinking
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12470v2](http://arxiv.org/pdf/2502.12470v2)**

> **作者:** Alireza S. Ziabari; Nona Ghazizadeh; Zhivar Sourati; Farzan Karimi-Malekabadi; Payam Piray; Morteza Dehghani
>
> **摘要:** Large Language Models (LLMs) exhibit impressive reasoning abilities, yet their reliance on structured step-by-step processing reveals a critical limitation. In contrast, human cognition fluidly adapts between intuitive, heuristic (System 1) and analytical, deliberative (System 2) reasoning depending on the context. This difference between human cognitive flexibility and LLMs' reliance on a single reasoning style raises a critical question: while human fast heuristic reasoning evolved for its efficiency and adaptability, is a uniform reasoning approach truly optimal for LLMs, or does its inflexibility make them brittle and unreliable when faced with tasks demanding more agile, intuitive responses? To answer these questions, we explicitly align LLMs to these reasoning styles by curating a dataset with valid System 1 and System 2 answers, and evaluate their performance across reasoning benchmarks. Our results reveal an accuracy-efficiency trade-off: System 2-aligned models excel in arithmetic and symbolic reasoning, while System 1-aligned models perform better in commonsense reasoning tasks. To analyze the reasoning spectrum, we interpolated between the two extremes by varying the proportion of alignment data, which resulted in a monotonic change in accuracy. A mechanistic analysis of model responses shows that System 1 models employ more definitive outputs, whereas System 2 models demonstrate greater uncertainty. Building on these findings, we further combine System 1- and System 2-aligned models based on the entropy of their generations, without additional training, and obtain a dynamic model that outperforms across nearly all benchmarks. This work challenges the assumption that step-by-step reasoning is always optimal and highlights the need for adapting reasoning strategies based on task demands.
>
---
#### [replaced 061] LearnLens: LLM-Enabled Personalised, Curriculum-Grounded Feedback with Educators in the Loop
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.04295v4](http://arxiv.org/pdf/2507.04295v4)**

> **作者:** Runcong Zhao; Artem Bobrov; Jiazheng Li; Cesare Aloisi; Yulan He
>
> **备注:** EMNLP 2025
>
> **摘要:** Effective feedback is essential for student learning but is time-intensive for teachers. We present LearnLens, a modular, LLM-based system that generates personalised, curriculum-aligned feedback in science education. LearnLens comprises three components: (1) an error-aware assessment module that captures nuanced reasoning errors; (2) a curriculum-grounded generation module that uses a structured, topic-linked memory chain rather than traditional similarity-based retrieval, improving relevance and reducing noise; and (3) an educator-in-the-loop interface for customisation and oversight. LearnLens addresses key challenges in existing systems, offering scalable, high-quality feedback that empowers both teachers and students.
>
---
#### [replaced 062] The Distribution of Dependency Distance and Hierarchical Distance in Contemporary Written Japanese and Its Influencing Factors
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.21421v2](http://arxiv.org/pdf/2504.21421v2)**

> **作者:** Linxuan Wang; Shuiyuan Yu
>
> **备注:** This paper has been accepted by the 13th International Quantitative Linguistics Conference QUALICO 2025
>
> **摘要:** To explore the relationship between dependency distance (DD) and hierarchical distance (HD) in Japanese, we compared the probability distributions of DD and HD with and without sentence length fixed, and analyzed the changes in mean dependency distance (MDD) and mean hierarchical distance (MHD) as sentence length increases, along with their correlation coefficient based on the Balanced Corpus of Contemporary Written Japanese. It was found that the valency of the predicates is the underlying factor behind the trade-off relation between MDD and MHD in Japanese. Native speakers of Japanese regulate the linear complexity and hierarchical complexity through the valency of the predicates, and the relative sizes of MDD and MHD depend on whether the threshold of valency has been reached. Apart from the cognitive load, the valency of the predicates also affects the probability distributions of DD and HD. The effect of the valency of the predicates on the distribution of HD is greater than on that of DD, which leads to differences in their probability distributions and causes the mean of MDD to be lower than that of MHD.
>
---
#### [replaced 063] MaxPoolBERT: Enhancing BERT Classification via Layer- and Token-Wise Aggregation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15696v2](http://arxiv.org/pdf/2505.15696v2)**

> **作者:** Maike Behrendt; Stefan Sylvius Wagner; Stefan Harmeling
>
> **摘要:** The [CLS] token in BERT is commonly used as a fixed-length representation for classification tasks, yet prior work has shown that both other tokens and intermediate layers encode valuable contextual information. In this work, we study lightweight extensions to BERT that refine the [CLS] representation by aggregating information across layers and tokens. Specifically, we explore three modifications: (i) max-pooling the [CLS] token across multiple layers, (ii) enabling the [CLS] token to attend over the entire final layer using an additional multi-head attention (MHA) layer, and (iii) combining max-pooling across the full sequence with MHA. Our approach, called MaxPoolBERT, enhances BERT's classification accuracy (especially on low-resource tasks) without requiring new pre-training or significantly increasing model size. Experiments on the GLUE benchmark show that MaxPoolBERT consistently achieves a better performance than the standard BERT base model on low resource tasks of the GLUE benchmark.
>
---
#### [replaced 064] LLMs4All: A Systematic Review of Large Language Models Across Academic Disciplines
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.19580v4](http://arxiv.org/pdf/2509.19580v4)**

> **作者:** Yanfang Ye; Zheyuan Zhang; Tianyi Ma; Zehong Wang; Yiyang Li; Shifu Hou; Weixiang Sun; Kaiwen Shi; Yijun Ma; Wei Song; Ahmed Abbasi; Ying Cheng; Jane Cleland-Huang; Steven Corcelli; Robert Goulding; Ming Hu; Ting Hua; John Lalor; Fang Liu; Tengfei Luo; Ed Maginn; Nuno Moniz; Jason Rohr; Brett Savoie; Daniel Slate; Matthew Webber; Olaf Wiest; Johnny Zhang; Nitesh V. Chawla
>
> **备注:** This version corrects the author metadata and refines the paper's title. Earlier third-party (Google/Google Scholar) indexes omitted the first/lead author (Y. Ye); the arXiv v4 record here is authoritative
>
> **摘要:** Cutting-edge Artificial Intelligence (AI) techniques keep reshaping our view of the world. For example, Large Language Models (LLMs) based applications such as ChatGPT have shown the capability of generating human-like conversation on extensive topics. Due to the impressive performance on a variety of language-related tasks (e.g., open-domain question answering, translation, and document summarization), one can envision the far-reaching impacts that can be brought by the LLMs with broader real-world applications (e.g., customer service, education and accessibility, and scientific discovery). Inspired by their success, this paper will offer an overview of state-of-the-art LLMs and their integration into a wide range of academic disciplines, including: (1) arts, letters, and law (e.g., history, philosophy, political science, arts and architecture, law), (2) economics and business (e.g., finance, economics, accounting, marketing), and (3) science and engineering (e.g., mathematics, physics and mechanical engineering, chemistry and chemical engineering, life sciences and bioengineering, earth sciences and civil engineering, computer science and electrical engineering). Integrating humanity and technology, in this paper, we will explore how LLMs are shaping research and practice in these fields, while also discussing key limitations, open challenges, and future directions in the era of generative AI. The review of how LLMs are engaged across disciplines-along with key observations and insights-can help researchers and practitioners interested in exploiting LLMs to advance their works in diverse real-world applications.
>
---
#### [replaced 065] Simulating and Understanding Deceptive Behaviors in Long-Horizon Interactions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.03999v2](http://arxiv.org/pdf/2510.03999v2)**

> **作者:** Yang Xu; Xuanming Zhang; Samuel Yeh; Jwala Dhamala; Ousmane Dia; Rahul Gupta; Sharon Li
>
> **摘要:** Deception is a pervasive feature of human communication and an emerging concern in large language models (LLMs). While recent studies document instances of LLM deception under pressure, most evaluations remain confined to single-turn prompts and fail to capture the long-horizon interactions in which deceptive strategies typically unfold. We introduce the first simulation framework for probing and evaluating deception in LLMs under extended sequences of interdependent tasks and dynamic contextual pressures. Our framework instantiates a multi-agent system: a performer agent tasked with completing tasks and a supervisor agent that evaluates progress, provides feedback, and maintains evolving states of trust. An independent deception auditor then reviews full trajectories to identify when and how deception occurs. We conduct extensive experiments across 11 frontier models, spanning both closed- and open-source systems, and find that deception is model-dependent, increases with event pressure, and consistently erodes supervisor trust. Qualitative analyses further reveal distinct strategies of concealment, equivocation, and falsification. Our findings establish deception as an emergent risk in long-horizon interactions and provide a foundation for evaluating future LLMs in real-world, trust-sensitive contexts.
>
---
#### [replaced 066] Enhancing Long-Chain Reasoning Distillation through Error-Aware Self-Reflection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22131v2](http://arxiv.org/pdf/2505.22131v2)**

> **作者:** Zhuoyang Wu; Xinze Li; Zhenghao Liu; Yukun Yan; Zhiyuan Liu; Minghe Yu; Cheng Yang; Yu Gu; Ge Yu; Maosong Sun
>
> **摘要:** Large Language Models (LLMs) have exhibited strong reasoning capabilities and achieved remarkable performance in mathematical problem-solving tasks. Recently, distilling reasoning ability from long-form Chains-of-Thought (CoTs) has emerged as a promising approach for enhancing Small Language Models (SLMs). Existing studies typically treat SLMs as student models and use long-form CoTs as supervision signals for Supervised Fine-Tuning (SFT) to transfer reasoning ability. However, such long-form CoT teachers are usually unaware of the student model's capacity, which limits the effective utilization of the provided reasoning traces. To overcome this limitation, we propose errOr-aware self-ReflectION (ORION), a framework that refines teacher CoTs through an Error-Aware Reflection process. ORION enables the student model to construct more tailored teacher CoTs by refining teacher CoTs and incorporating its own reasoning errors. Experiments on multiple mathematical reasoning benchmarks demonstrate that ORION consistently improves performance by more than 2% over all baselines. Further analysis reveals that the CoTs constructed by ORION exhibit higher coherence and logical consistency, thereby serving as more effective supervision signals for SFT. All codes are available at https://github.com/NEUIR/ORION.git.
>
---
#### [replaced 067] DefenderBench: A Toolkit for Evaluating Language Agents in Cybersecurity Environments
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00739v4](http://arxiv.org/pdf/2506.00739v4)**

> **作者:** Chiyu Zhang; Marc-Alexandre Cote; Michael Albada; Anush Sankaran; Jack W. Stokes; Tong Wang; Amir Abdi; William Blum; Muhammad Abdul-Mageed
>
> **备注:** Accepted by NeurIPS 2025 Workshop Scaling Environments for Agents (SEA)
>
> **摘要:** Large language model (LLM) agents have shown impressive capabilities in human language comprehension and reasoning, yet their potential in cybersecurity remains underexplored. We introduce DefenderBench, a practical, open-source toolkit for evaluating language agents across offense, defense, and cybersecurity knowledge-based tasks. DefenderBench includes environments for network intrusion, malicious content detection, code vulnerability analysis, and cybersecurity knowledge assessment. It is intentionally designed to be affordable and easily accessible for researchers while providing fair and rigorous assessment. We benchmark several state-of-the-art (SoTA) and popular LLMs, including both open- and closed-weight models, using a standardized agentic framework. Our results show that Claude-3.7-sonnet performs best with a DefenderBench score of 81.65, followed by Claude-3.7-sonnet-think with 78.40, while the best open-weight model, Llama 3.3 70B, is not far behind with a DefenderBench score of 71.81. DefenderBench's modular design allows seamless integration of custom LLMs and tasks, promoting reproducibility and fair comparisons. An anonymized version of DefenderBench is available at https://github.com/microsoft/DefenderBench.
>
---
#### [replaced 068] DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.10446v2](http://arxiv.org/pdf/2509.10446v2)**

> **作者:** Rui Lu; Zhenyu Hou; Zihan Wang; Hanchen Zhang; Xiao Liu; Yujiang Li; Shi Feng; Jie Tang; Yuxiao Dong
>
> **摘要:** Augmenting large language models (LLMs) with browsing tools substantially improves their potential as deep search agents to solve complex, real-world tasks. Yet, open LLMs still perform poorly in such settings due to limited long-horizon reasoning capacity with browsing tools and the lack of sufficiently difficult supervised data. To address these challenges, we present DeepDive to advance deep search agents. First, we propose a strategy to automatically synthesize complex, difficult, and hard-to-find questions from open knowledge graphs. Second, we apply end-to-end multi-turn reinforcement learning (RL) to enhance LLMs' long-horizon reasoning with deep search. To encourage diversity and reduce redundancy, we design a redundancy penalty that discourages repeated similar queries. Experiments show that DeepDive-32B achieves a new open-source competitive result on BrowseComp, outperforming WebSailor, DeepSeek-R1-Browse, and Search-o1. We demonstrate that multi-turn RL training improves deep search ability and significantly contributes to the performance improvements across multiple benchmarks. We observe that DeepDive enables test-time scaling of tool calls and parallel sampling. All datasets, models, and code are publicly available at https://github.com/THUDM/DeepDive.
>
---
#### [replaced 069] Can LLMs Express Personality Across Cultures? Introducing CulturalPersonas for Evaluating Trait Alignment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05670v2](http://arxiv.org/pdf/2506.05670v2)**

> **作者:** Priyanka Dey; Yugal Khanter; Aayush Bothra; Jieyu Zhao; Emilio Ferrara
>
> **摘要:** As LLMs become central to interactive applications, ranging from tutoring to mental health, the ability to express personality in culturally appropriate ways is increasingly important. While recent works have explored personality evaluation of LLMs, they largely overlook the interplay between culture and personality. To address this, we introduce CulturalPersonas, the first large-scale benchmark with human validation for evaluating LLMs' personality expression in culturally grounded, behaviorally rich contexts. Our dataset spans 3,000 scenario-based questions across six diverse countries, designed to elicit personality through everyday scenarios rooted in local values. We evaluate three LLMs, using both multiple-choice and open-ended response formats. Our results show that CulturalPersonas improves alignment with country-specific human personality distributions (over a 20% reduction in Wasserstein distance across models and countries) and elicits more expressive, culturally coherent outputs compared to existing benchmarks. CulturalPersonas surfaces meaningful modulated trait outputs in response to culturally grounded prompts, offering new directions for aligning LLMs to global norms of behavior. By bridging personality expression and cultural nuance, we envision that CulturalPersonas will pave the way for more socially intelligent and globally adaptive LLMs.
>
---
#### [replaced 070] KnowledgeSmith: Uncovering Knowledge Updating in LLMs with Model Editing and Unlearning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.02392v2](http://arxiv.org/pdf/2510.02392v2)**

> **作者:** Yinyi Luo; Zhexian Zhou; Hao Chen; Kai Qiu; Marios Savvides; Sharon Li; Jindong Wang
>
> **备注:** Technical report
>
> **摘要:** Knowledge editing and machine unlearning are two popular approaches for large language models (LLMs) to stay up-to-date. However, the knowledge updating mechanism of LLMs remains largely unexplored due to insufficient, isolated, and small-scale evaluation. For instance, are LLMs similar to humans in modifying certain knowledge? What differs editing and unlearning as training data increases? This paper proposes KnowledgeSmith, a unified framework to systematically understand the updating mechanism of LLMs. We first cast editing and unlearning as instances of one constrained optimization problem. Then, we propose an automatic dataset generator that provides structured interventions across multiple graph levels and data scales, enabling controlled studies of how different modification strategies propagate through model knowledge. Extensive experiments demonstrate nuanced insights over knowledge propagation, plasticity scaling, consistency, and robustness. For instance, our results show that LLMs do not exhibit similar updating as humans for different levels of knowledge, and there exists consistency-capacity trade-off. We hope our findings can offer suggestions to the design of more reliable and scalable strategies. Code: https://github.com/AIFrontierLab/KnowledgeSmith.git
>
---
#### [replaced 071] Think Natively: Unlocking Multilingual Reasoning with Consistency-Enhanced Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.07300v2](http://arxiv.org/pdf/2510.07300v2)**

> **作者:** Xue Zhang; Yunlong Liang; Fandong Meng; Songming Zhang; Kaiyu Huang; Yufeng Chen; Jinan Xu; Jie Zhou
>
> **备注:** 13 pages, 8 tables, 4 figures. Code is available at: https://github.com/XZhang00/M-Thinker
>
> **摘要:** Large Reasoning Models (LRMs) have achieved remarkable performance on complex reasoning tasks by adopting the "think-then-answer" paradigm, which enhances both accuracy and interpretability. However, current LRMs exhibit two critical limitations when processing non-English languages: (1) They often struggle to maintain input-output language consistency; (2) They generally perform poorly with wrong reasoning paths and lower answer accuracy compared to English. These limitations significantly degrade the user experience for non-English speakers and hinder the global deployment of LRMs. To address these limitations, we propose M-Thinker, which is trained by the GRPO algorithm that involves a Language Consistency (LC) reward and a novel Cross-lingual Thinking Alignment (CTA) reward. Specifically, the LC reward defines a strict constraint on the language consistency between the input, thought, and answer. Besides, the CTA reward compares the model's non-English reasoning paths with its English reasoning path to transfer its own reasoning capability from English to non-English languages. Through an iterative RL procedure, our M-Thinker-1.5B/7B models not only achieve nearly 100% language consistency and superior performance on two multilingual benchmarks (MMATH and PolyMath), but also exhibit excellent generalization on out-of-domain languages.
>
---
#### [replaced 072] Cognition-of-Thought Elicits Social-Aligned Reasoning in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.23441v2](http://arxiv.org/pdf/2509.23441v2)**

> **作者:** Xuanming Zhang; Yuxuan Chen; Samuel Yeh; Sharon Li
>
> **摘要:** Large language models (LLMs) excel at complex reasoning but can still exhibit harmful behaviors. Current alignment strategies typically embed safety into model weights, making these controls implicit, static, and difficult to modify. This paper introduces Cognition-of-Thought (CooT), a novel decoding-time framework that equips LLMs with an explicit cognitive self-monitoring loop. CooT couples a standard text Generator with a cognitive Perceiver that continuously monitors the unfolding sequence. The Perceiver uses a structured, precedence-based hierarchy of principles (e.g., safety over obedience) to detect potential misalignments as they arise. When violations are flagged, CooT intervenes by rolling back the generation to the point of error and regenerating under injected guidance that combines universal social priors with context-specific warnings. CooT thus transforms alignment from a fixed property into an explicit, dynamic, and auditable process active during inference, allowing for flexible policy updates without retraining the model. Extensive experiments across multiple benchmarks and model families confirm that CooT consistently improves safety and social reasoning performance.
>
---
#### [replaced 073] Boundary-Guided Policy Optimization for Memory-efficient RL of Diffusion Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.11683v2](http://arxiv.org/pdf/2510.11683v2)**

> **作者:** Nianyi Lin; Jiajie Zhang; Lei Hou; Juanzi Li
>
> **摘要:** A key challenge in applying reinforcement learning (RL) to diffusion large language models (dLLMs) lies in the intractability of their likelihood functions, which are essential for the RL objective, necessitating corresponding approximation in each training step. While existing methods approximate the log-likelihoods by their evidence lower bounds (ELBOs) via customized Monte Carlo (MC) sampling, the forward computational graphs of all MC samples need to be retained for the gradient computation of non-linear terms in the RL objective, resulting in significant memory overhead. This constraint restricts feasible sample sizes, leading to imprecise likelihood approximations and ultimately distorting the RL objective. To overcome this limitation, we propose \emph{Boundary-Guided Policy Optimization} (BGPO), a memory-efficient RL algorithm that maximizes a specially constructed lower bound of the ELBO-based objective. This lower bound is carefully designed to satisfy two key properties: (1) Linearity: it is formulated in a linear sum where each term depends only on a single MC sample, thereby enabling gradient accumulation across samples and ensuring constant memory usage; (2) Equivalence: Both the value and gradient of this lower bound are equal to those of the ELBO-based objective in on-policy training, making it also an effective approximation for the original RL objective. These properties allow BGPO to adopt a large MC sample size, resulting in more accurate likelihood approximations and improved RL objective estimation, which in turn leads to enhanced performance. Experiments show that BGPO significantly outperforms previous RL algorithms for dLLMs in math problem solving, code generation, and planning tasks. Our codes and models are available at \href{https://github.com/THU-KEG/BGPO}{https://github.com/THU-KEG/BGPO}.
>
---
#### [replaced 074] A Survey of Multilingual Reasoning in Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.09457v2](http://arxiv.org/pdf/2502.09457v2)**

> **作者:** Akash Ghosh; Debayan Datta; Sriparna Saha; Chirag Agarwal
>
> **备注:** EMNLP Findings 2025
>
> **摘要:** While reasoning and multilingual capabilities in language models (LMs) have achieved remarkable progress in recent years, their integration into a unified paradigm - multilingual reasoning - is at a nascent stage. Multilingual reasoning requires language models to handle logical reasoning across languages while addressing misalignment, biases, and challenges in low-resource settings. This survey provides the first in-depth review of multilingual reasoning in LMs. In this survey, we provide a systematic overview of existing methods that leverage LMs for multilingual reasoning, specifically outlining the challenges, motivations, and foundational aspects of applying language models to reason across diverse languages. We provide an overview of the standard data resources used for training multilingual reasoning in LMs and the evaluation benchmarks employed to assess their multilingual capabilities. Next, we analyze various state-of-the-art methods and their performance on these benchmarks. Finally, we explore future research opportunities to improve multilingual reasoning in LMs, focusing on enhancing their ability to handle diverse languages and complex reasoning tasks. Rapid growth of evolving developments in this field can be actively tracked on our project page: [https://github.com/AkashGhosh/Survey-of-Multilingual-Reasoning-in-Language-Models](https://github.com/AkashGhosh/Survey-of-Multilingual-Reasoning-in-Language-Models)
>
---
#### [replaced 075] CrisiText: A dataset of warning messages for LLM training in emergency communication
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.09243v2](http://arxiv.org/pdf/2510.09243v2)**

> **作者:** Giacomo Gonella; Gian Maria Campedelli; Stefano Menini; Marco Guerini
>
> **摘要:** Effectively identifying threats and mitigating their potential damage during crisis situations, such as natural disasters or violent attacks, is paramount for safeguarding endangered individuals. To tackle these challenges, AI has been used in assisting humans in emergency situations. Still, the use of NLP techniques remains limited and mostly focuses on classification tasks. The significant potential of timely warning message generation using NLG architectures, however, has been largely overlooked. In this paper we present CrisiText, the first large-scale dataset for the generation of warning messages across 13 different types of crisis scenarios. The dataset contains more than 400,000 warning messages (spanning almost 18,000 crisis situations) aimed at assisting civilians during and after such events. To generate the dataset, we started from existing crisis descriptions and created chains of events related to the scenarios. Each event was then paired with a warning message. The generations follow experts' written guidelines to ensure correct terminology and factuality of their suggestions. Additionally, each message is accompanied by three suboptimal warning types to allow for the study of different NLG approaches. To this end, we conducted a series of experiments comparing supervised fine-tuning setups with preference alignment, zero-shot, and few-shot approaches. We further assessed model performance in out-of-distribution scenarios and evaluated the effectiveness of an automatic post-editor.
>
---
#### [replaced 076] The Price of a Second Thought: On the Evaluation of Reasoning Efficiency in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22017v2](http://arxiv.org/pdf/2505.22017v2)**

> **作者:** Siqi Fan; Bowen Qin; Peng Han; Shuo Shang; Yequan Wang; Aixin Sun
>
> **备注:** Added new experiments and revised the manuscript for clarity
>
> **摘要:** Recent thinking models trained with reinforcement learning and backward-checking CoT often suffer from overthinking: they produce excessively long outputs even on simple problems, wasting computation. Existing evaluations, based on token efficiency, give an incomplete view as they neglect problem difficulty and intermediate computation costs. We formalize reasoning efficiency as a relative measure between thinking and instruct models, treating instruct models as the minimal-effort baseline. A systematic study across four thinking models and multiple benchmarks reveals two consistent patterns: (i) instruct models achieve higher efficiency overall, and (ii) problem difficulty affects efficiency, with thinking models wasting computation on easy problems but providing value on harder ones. Building on this insight, we propose COTHINK, a simple two-stage pipeline: an instruct model drafts a brief outline, and a thinking model expands it. On GSM8K, MATH500, and AIME24, COTHINK cuts token usage by 21.1% while keeping accuracy on four thinking models, and remains competitive with strong efficiency baselines.
>
---
#### [replaced 077] Causal Agent based on Large Language Model
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.06849v2](http://arxiv.org/pdf/2408.06849v2)**

> **作者:** Kairong Han; Kun Kuang; Ziyu Zhao; Junjian Ye; Fei Wu
>
> **摘要:** The large language model (LLM) has achieved significant success across various domains. However, the inherent complexity of causal problems and causal theory poses challenges in accurately describing them in natural language, making it difficult for LLM to comprehend and use them effectively. Causal methods are not easily conveyed through natural language, which hinders LLM's ability to apply them accurately. Additionally, causal datasets are typically tabular, while LLM excels in handling natural language data, creating a structural mismatch that impedes effective reasoning with tabular data. To address these challenges, we have equipped the LLM with causal tools within an agent framework, named the Causal Agent, enabling it to tackle causal problems. The causal agent comprises tools, memory, and reasoning modules. In the tool module, the causal agent calls Python code and uses the encapsulated causal function module to align tabular data with natural language. In the reasoning module, the causal agent performs reasoning through multiple iterations with the tools. In the memory module, the causal agent maintains a dictionary instance where the keys are unique names and the values are causal graphs. To verify the causal ability of the causal agent, we established a Causal Tabular Question Answer (CausalTQA) benchmark consisting of four levels of causal problems: variable level, edge level, causal graph level, and causal effect level. CausalTQA consists of about 1.4K for these four levels questions. Causal agent demonstrates remarkable efficacy on the four-level causal problems, with accuracy rates all above 80\%. Through verification on the real-world dataset QRData, the causal agent is 6\% higher than the original SOTA. For further insights and implementation details, our code is accessible via the GitHub repository https://github.com/kairong-han/causal_agent.
>
---
#### [replaced 078] Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10412v3](http://arxiv.org/pdf/2506.10412v3)**

> **作者:** Ching Chang; Jeehyun Hwang; Yidan Shi; Haixin Wang; Wen-Chih Peng; Tien-Fu Chen; Wei Wang
>
> **备注:** This paper has been accepted by the NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** Time series data in real-world applications such as healthcare, climate modeling, and finance are often irregular, multimodal, and messy, with varying sampling rates, asynchronous modalities, and pervasive missingness. However, existing benchmarks typically assume clean, regularly sampled, unimodal data, creating a significant gap between research and real-world deployment. We introduce Time-IMM, a dataset specifically designed to capture cause-driven irregularity in multimodal multivariate time series. Time-IMM represents nine distinct types of time series irregularity, categorized into trigger-based, constraint-based, and artifact-based mechanisms. Complementing the dataset, we introduce IMM-TSF, a benchmark library for forecasting on irregular multimodal time series, enabling asynchronous integration and realistic evaluation. IMM-TSF includes specialized fusion modules, including a timestamp-to-text fusion module and a multimodality fusion module, which support both recency-aware averaging and attention-based integration strategies. Empirical results demonstrate that explicitly modeling multimodality on irregular time series data leads to substantial gains in forecasting performance. Time-IMM and IMM-TSF provide a foundation for advancing time series analysis under real-world conditions. The dataset is publicly available at https://www.kaggle.com/datasets/blacksnail789521/time-imm/data, and the benchmark library can be accessed at https://github.com/blacksnail789521/IMM-TSF.
>
---
#### [replaced 079] Unspoken Hints: Accuracy Without Acknowledgement in LLM Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.26041v2](http://arxiv.org/pdf/2509.26041v2)**

> **作者:** Arash Marioriyad; Shaygan Adim; Nima Alighardashi; Mahdieh Soleymani Banghshah; Mohammad Hossein Rohban
>
> **备注:** 5 Pages, 4 Figures, 4 Tables
>
> **摘要:** Large language models (LLMs) increasingly rely on chain-of-thought (CoT) prompting to solve mathematical and logical reasoning tasks. Yet, a central question remains: to what extent are these generated rationales \emph{faithful} to the underlying computations, rather than post-hoc narratives shaped by hints that function as answer shortcuts embedded in the prompt? Following prior work on hinted vs.\ unhinted prompting, we present a systematic study of CoT faithfulness under controlled hint manipulations. Our experimental design spans four datasets (AIME, GSM-Hard, MATH-500, UniADILR), two state-of-the-art models (GPT-4o and Gemini-2-Flash), and a structured set of hint conditions varying in correctness (correct and incorrect), presentation style (sycophancy and data leak), and complexity (raw answers, two-operator expressions, four-operator expressions). We evaluate both task accuracy and whether hints are explicitly acknowledged in the reasoning. Our results reveal three key findings. First, correct hints substantially improve accuracy, especially on harder benchmarks and logical reasoning, while incorrect hints sharply reduce accuracy in tasks with lower baseline competence. Second, acknowledgement of hints is highly uneven: equation-based hints are frequently referenced, whereas raw hints are often adopted silently, indicating that more complex hints push models toward verbalizing their reliance in the reasoning process. Third, presentation style matters: sycophancy prompts encourage overt acknowledgement, while leak-style prompts increase accuracy but promote hidden reliance. This may reflect RLHF-related effects, as sycophancy exploits the human-pleasing side and data leak triggers the self-censoring side. Together, these results demonstrate that LLM reasoning is systematically shaped by shortcuts in ways that obscure faithfulness.
>
---
#### [replaced 080] DoctorAgent-RL: A Multi-Agent Collaborative Reinforcement Learning System for Multi-Turn Clinical Dialogue
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19630v3](http://arxiv.org/pdf/2505.19630v3)**

> **作者:** Yichun Feng; Jiawei Wang; Lu Zhou; Zhen Lei; Yixue Li
>
> **摘要:** Large language models (LLMs) have demonstrated excellent capabilities in the field of biomedical question answering, but their application in real-world clinical consultations still faces core challenges. Single-round consultation systems require patients to describe all symptoms upfront, leading to vague diagnosis with unclear complaints. Traditional multi-turn dialogue models, constrained by static supervised learning, lack flexibility and fail to intelligently extract key clinical information. To address these limitations, we propose \Ours{}, a reinforcement learning (RL)-based multi-agent collaborative framework that models medical consultations as a dynamic decision-making process under uncertainty. The doctor agent continuously optimizes its questioning strategy within the RL framework through multi-turn interactions with the patient agent, dynamically adjusting its information-gathering path based on comprehensive rewards from the Consultation Evaluator. This RL fine-tuning mechanism enables LLMs to autonomously develop interaction strategies aligned with clinical reasoning logic, rather than superficially imitating patterns in existing dialogue data. Notably, we constructed MTMedDialog, the first English multi-turn medical consultation dataset capable of simulating patient interactions. Experiments demonstrate that \Ours{} outperforms existing models in both multi-turn reasoning capability and final diagnostic performance. This approach shows immense practical value by reducing misdiagnosis risks in time-pressured settings, freeing clinicians for complex cases, and pioneering a strategy to optimize medical resource allocation and alleviate workforce shortages. Code and data are available at https://github.com/JarvisUSTC/DoctorAgent-RL
>
---
#### [replaced 081] The Open Source Advantage in Large Language Models (LLMs)
- **分类: cs.CL; cs.LG; I.2.7**

- **链接: [http://arxiv.org/pdf/2412.12004v3](http://arxiv.org/pdf/2412.12004v3)**

> **作者:** Jiya Manchanda; Laura Boettcher; Matheus Westphalen; Jasser Jasser
>
> **备注:** 9 pages, 1 figure
>
> **摘要:** Large language models (LLMs) have rapidly advanced natural language processing, driving significant breakthroughs in tasks such as text generation, machine translation, and domain-specific reasoning. The field now faces a critical dilemma in its approach: closed-source models like GPT-4 deliver state-of-the-art performance but restrict reproducibility, accessibility, and external oversight, while open-source frameworks like LLaMA and Mixtral democratize access, foster collaboration, and support diverse applications, achieving competitive results through techniques like instruction tuning and LoRA. Hybrid approaches address challenges like bias mitigation and resource accessibility by combining the scalability of closed-source systems with the transparency and inclusivity of open-source framework. However, in this position paper, we argue that open-source remains the most robust path for advancing LLM research and ethical deployment.
>
---
#### [replaced 082] Steering Large Language Models for Machine Translation Personalization
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16612v2](http://arxiv.org/pdf/2505.16612v2)**

> **作者:** Daniel Scalena; Gabriele Sarti; Arianna Bisazza; Elisabetta Fersini; Malvina Nissim
>
> **摘要:** Large language models have simplified the production of personalized translations reflecting predefined stylistic constraints. However, these systems still struggle when stylistic requirements are implicitly represented by a set of examples, such as texts produced by a specific human translator. In this work, we explore various strategies for personalizing automatically generated translations when few examples are available, with a focus on the challenging domain of literary translation. We begin by determining the feasibility of the task and how style information is encoded within model representations. Then, we evaluate various prompting strategies and inference-time interventions for steering model generations towards a personalized style, with a particular focus on contrastive steering with sparse autoencoder (SAE) latents to identify salient personalization properties. We demonstrate that contrastive SAE steering yields robust style conditioning and translation quality, resulting in higher inference-time computational efficiency than prompting approaches. We further examine the impact of steering on model activations, finding that layers encoding personalization properties are impacted similarly by prompting and SAE steering, suggesting a similar mechanism at play.
>
---
#### [replaced 083] Triplet-Structured Knowledge Integration for Multi-Turn Medical Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.03536v2](http://arxiv.org/pdf/2510.03536v2)**

> **作者:** Zhaohan Meng; Zaiqiao Meng; Siwei Liu; Iadh Ounis
>
> **备注:** Preprint
>
> **摘要:** Large Language Models (LLMs) have shown strong performance on static medical Question Answering (QA) tasks, yet their reasoning often deteriorates in multi-turn clinical dialogues where patient information is scattered across turns. This paper introduces TriMediQ, a triplet-structured approach that enhances the reasoning reliability of LLMs through explicit knowledge integration. TriMediQ first employs a frozen triplet extraction LLM to convert patient responses into clinically grounded triplets, ensuring factual precision via constrained prompting. These triplets are incorporated into a patient-specific Knowledge Graph (KG), from which a trainable projection module consisting of a graph encoder and a projector captures relational dependencies while keeping all LLM parameters frozen. During inference, the projection module guides multi-hop reasoning over the KG, enabling coherent clinical dialogue understanding. Experiments on two interactive medical QA benchmarks show that TriMediQ achieves up to 10.4\% improvement in accuracy over five existing baselines on the iMedQA dataset. These results demonstrate that structuring patient information as triplets can effectively improve the reasoning capability of LLMs in multi-turn medical QA.
>
---
#### [replaced 084] When "Competency" in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel Complex Ciphers
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2402.10601v5](http://arxiv.org/pdf/2402.10601v5)**

> **作者:** Divij Handa; Zehua Zhang; Amir Saeidi; Shrinidhi Kumbhar; Md Nayem Uddin; Aswin RRV; Chitta Baral
>
> **备注:** Published in Reliable ML from Unreliable Data workshop @ NeurIPS 2025
>
> **摘要:** Recent advancements in Large Language Model (LLM) safety have primarily focused on mitigating attacks crafted in natural language or common ciphers (e.g. Base64), which are likely integrated into newer models' safety training. However, we reveal a paradoxical vulnerability: as LLMs advance in reasoning, they inadvertently become more susceptible to novel jailbreaking attacks. Enhanced reasoning enables LLMs to interpret complex instructions and decode complex user-defined ciphers, creating an exploitable security gap. To study this vulnerability, we introduce Attacks using Custom Encryptions (ACE), a jailbreaking technique that encodes malicious queries with novel ciphers. Extending ACE, we introduce Layered Attacks using Custom Encryptions (LACE), which applies multi-layer ciphers to amplify attack complexity. Furthermore, we develop CipherBench, a benchmark designed to evaluate LLMs' accuracy in decoding encrypted benign text. Our experiments reveal a critical trade-off: LLMs that are more capable of decoding ciphers are more vulnerable to LACE, with success rates on gpt-oss-20b escalating from 60% under ACE to 72% with LACE. These findings highlight a critical insight: as LLMs become more adept at deciphering complex user ciphers--many of which cannot be preemptively included in safety training--they become increasingly exploitable.
>
---
