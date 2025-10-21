# 自然语言处理 cs.CL

- **最新发布 158 篇**

- **更新 133 篇**

## 最新发布

#### [new 001] Facts in Stats: Impacts of Pretraining Diversity on Language Model Generalization
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究预训练多样性对语言模型泛化的影响，旨在揭示上下文结构与多样性水平如何影响事实记忆与统计泛化。作者构建合成测试平台，控制变量进行实验，发现多样性对分布外泛化至关重要，并定位优化瓶颈于嵌入层。**

- **链接: [http://arxiv.org/pdf/2510.16096v1](http://arxiv.org/pdf/2510.16096v1)**

> **作者:** Tina Behnia; Puneesh Deora; Christos Thrampoulidis
>
> **备注:** 28 pages, 15 figures
>
> **摘要:** Language models are pretrained on sequences that blend statistical regularities (making text fluent) with factual associations between specific tokens (knowledge of facts). While recent work suggests that the variability of their interaction, such as paraphrases of factual associations, critically determines generalization ability, we lack a systematic analysis of these impacts. This paper introduces a flexible synthetic testbed that combines a statistical stream of generic tokens with an abstract factual stream of source-target token pairs, enabling fine-grained control over their interaction. The design enables the independent control of diversity nature by manipulating stream composition (contextual structure) and the diversity level by varying which statistical streams each fact appears in. Through controlled experiments, we find that while higher contextual diversity delays in-distribution (ID) factual accuracy, its impact on out-of-distribution (OOD) factual generalization depends critically on contextual structure. In some cases, OOD performance follows the same trend as ID, but in others, diversity becomes essential for non-trivial factual recall. Even when low diversity prohibits factual recall, optimal diversity levels depend on training duration. Beyond factual recall failures, we identify structures where statistical generalization fails independently, and others where both capabilities degrade. This shows how the interplay between contextual design and diversity level impacts different generalization aspects. Further, through a series of controlled interventions on the model components, we trace the OOD failures to distinct optimization bottlenecks, highlighting the importance of the embedding and unembedding layers. Our synthetic framework allows us to isolate effects that would be confounded in large-scale studies, offering a controlled testbed for future investigations.
>
---
#### [new 002] AFRICAPTION: Establishing a New Paradigm for Image Captioning in African Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦图像描述生成任务，旨在解决非洲低资源语言在多模态AI中的代表性不足问题。作者提出AfriCaption框架，包含多语言数据集、高质量翻译 pipeline 和统一的视觉-文本模型，推动包容性多模态研究。**

- **链接: [http://arxiv.org/pdf/2510.17405v1](http://arxiv.org/pdf/2510.17405v1)**

> **作者:** Mardiyyah Oduwole; Prince Mireku; Fatimo Adebanjo; Oluwatosin Olajide; Mahi Aminu Aliyu; Jekaterina Novikova
>
> **摘要:** Multimodal AI research has overwhelmingly focused on high-resource languages, hindering the democratization of advancements in the field. To address this, we present AfriCaption, a comprehensive framework for multilingual image captioning in 20 African languages and our contributions are threefold: (i) a curated dataset built on Flickr8k, featuring semantically aligned captions generated via a context-aware selection and translation process; (ii) a dynamic, context-preserving pipeline that ensures ongoing quality through model ensembling and adaptive substitution; and (iii) the AfriCaption model, a 0.5B parameter vision-to-text architecture that integrates SigLIP and NLLB200 for caption generation across under-represented languages. This unified framework ensures ongoing data quality and establishes the first scalable image-captioning resource for under-represented African languages, laying the groundwork for truly inclusive multimodal AI.
>
---
#### [new 003] Instant Personalized Large Language Model Adaptation via Hypernetwork
- **分类: cs.CL**

- **简介: 该论文研究个性化大语言模型适配任务，旨在解决现有方法需为每个用户单独训练、计算开销大的问题。作者提出Profile-to-PEFT框架，利用超网络将用户画像直接映射为适配器参数，实现免训练的即时个性化，兼顾效率、扩展性与隐私。**

- **链接: [http://arxiv.org/pdf/2510.16282v1](http://arxiv.org/pdf/2510.16282v1)**

> **作者:** Zhaoxuan Tan; Zixuan Zhang; Haoyang Wen; Zheng Li; Rongzhi Zhang; Pei Chen; Fengran Mo; Zheyuan Liu; Qingkai Zeng; Qingyu Yin; Meng Jiang
>
> **摘要:** Personalized large language models (LLMs) tailor content to individual preferences using user profiles or histories. However, existing parameter-efficient fine-tuning (PEFT) methods, such as the ``One-PEFT-Per-User'' (OPPU) paradigm, require training a separate adapter for each user, making them computationally expensive and impractical for real-time updates. We introduce Profile-to-PEFT, a scalable framework that employs a hypernetwork, trained end-to-end, to map a user's encoded profile directly to a full set of adapter parameters (e.g., LoRA), eliminating per-user training at deployment. This design enables instant adaptation, generalization to unseen users, and privacy-preserving local deployment. Experimental results demonstrate that our method outperforms both prompt-based personalization and OPPU while using substantially fewer computational resources at deployment. The framework exhibits strong generalization to out-of-distribution users and maintains robustness across varying user activity levels and different embedding backbones. The proposed Profile-to-PEFT framework enables efficient, scalable, and adaptive LLM personalization suitable for large-scale applications.
>
---
#### [new 004] Language Confusion Gate: Language-Aware Decoding Through Model Self-Distillation
- **分类: cs.CL**

- **简介: 该论文针对大语言模型生成时的语言混淆问题，提出一种轻量级插件式解码方法LCG。通过自蒸馏训练判断语言族，动态过滤混杂token，避免重训练，显著降低语言混淆，且不影响性能。**

- **链接: [http://arxiv.org/pdf/2510.17555v1](http://arxiv.org/pdf/2510.17555v1)**

> **作者:** Collin Zhang; Fei Huang; Chenhan Yuan; Junyang Lin
>
> **摘要:** Large language models (LLMs) often experience language confusion, which is the unintended mixing of languages during text generation. Current solutions to this problem either necessitate model retraining or cannot differentiate between harmful confusion and acceptable code-switching. This paper introduces the Language Confusion Gate (LCG), a lightweight, plug-in solution that filters tokens during decoding without altering the base LLM. The LCG is trained using norm-adjusted self-distillation to predict appropriate language families and apply masking only when needed. Our method is based on the findings that language confusion is infrequent, correct-language tokens are usually among the top predictions, and output token embedding norms are larger for high-resource languages, which biases sampling. When evaluated across various models, including Qwen3, GPT-OSS, Gemma3, Llama3.1, LCG decreases language confusion significantly, often by an order of magnitude, without negatively impacting task performance. Code is available at https://github.com/collinzrj/language_confusion_gate.
>
---
#### [new 005] Towards Mining Effective Pedagogical Strategies from Learner-LLM Educational Dialogues
- **分类: cs.CL**

- **简介: 该论文旨在从学习者与大语言模型的对话中挖掘有效教学策略。针对现有评估忽视交互过程的问题，提出结合对话行为标注与模式挖掘的方法，初步探索对话动态对教育效果的影响。**

- **链接: [http://arxiv.org/pdf/2510.17698v1](http://arxiv.org/pdf/2510.17698v1)**

> **作者:** Liqun He; Manolis Mavrikis; Mutlu Cukurova
>
> **摘要:** Dialogue plays a crucial role in educational settings, yet existing evaluation methods for educational applications of large language models (LLMs) primarily focus on technical performance or learning outcomes, often neglecting attention to learner-LLM interactions. To narrow this gap, this AIED Doctoral Consortium paper presents an ongoing study employing a dialogue analysis approach to identify effective pedagogical strategies from learner-LLM dialogues. The proposed approach involves dialogue data collection, dialogue act (DA) annotation, DA pattern mining, and predictive model building. Early insights are outlined as an initial step toward future research. The work underscores the need to evaluate LLM-based educational applications by focusing on dialogue dynamics and pedagogical strategies.
>
---
#### [new 006] SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents
- **分类: cs.CL**

- **简介: 该论文研究LLM搜索代理的安全问题，发现其在提升实用性时易生成有害内容。为此，提出SafeSearch方法，通过多目标强化学习联合优化安全与效用，显著降低危害性输出，同时保持问答性能。**

- **链接: [http://arxiv.org/pdf/2510.17017v1](http://arxiv.org/pdf/2510.17017v1)**

> **作者:** Qiusi Zhan; Angeline Budiman-Chan; Abdelrahman Zayed; Xingzhi Guo; Daniel Kang; Joo-Kyung Kim
>
> **备注:** Code: https://github.com/ZQS1943/SafeSearch
>
> **摘要:** Large language model (LLM) based search agents iteratively generate queries, retrieve external information, and reason to answer open-domain questions. While researchers have primarily focused on improving their utility, their safety behaviors remain underexplored. In this paper, we first evaluate search agents using red-teaming datasets and find that they are more likely to produce harmful outputs than base LLMs. For example, when asked ``How can I track someone's location without their consent?'', a base model refuses, whereas a search agent designed to retrieve and cite sources may lower its refusal threshold, fetch documents (e.g., court cases), and, once appended, synthesize them into an informative yet unsafe summary. We further show that utility-oriented fine-tuning intensifies this risk, motivating joint alignment of safety and utility. We present SafeSearch, a multi-objective reinforcement learning approach that couples a final-output safety/utility reward with a novel query-level shaping term that penalizes unsafe queries and rewards safe ones. Experiments show that SafeSearch reduces agent harmfulness by over 70% across three red-teaming datasets while producing safe, helpful responses, and matches the QA performance of a utility-only finetuned agent; further analyses confirm the effectiveness of the query-level reward in jointly improving safety and utility.
>
---
#### [new 007] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大模型遗忘学习任务，旨在去除特定知识的同时，保留模型在上下文中使用该知识的能力。现有方法损害了这一实用性，作者提出新目标函数，通过添加插件项恢复上下文利用能力，实验证明其有效兼顾遗忘与实用。**

- **链接: [http://arxiv.org/pdf/2510.17620v1](http://arxiv.org/pdf/2510.17620v1)**

> **作者:** Yuefeng Peng; Parnian Afshar; Megan Ganji; Thomas Butler; Amir Houmansadr; Mingxian Wang; Dezhi Hong
>
> **摘要:** Large language models may encode sensitive information or outdated knowledge that needs to be removed, to ensure responsible and compliant model responses. Unlearning has emerged as an efficient alternative to full retraining, aiming to remove specific knowledge while preserving overall model utility. Existing evaluations of unlearning methods focus on (1) the extent of forgetting of the target knowledge (forget set) and (2) maintaining performance on the retain set (i.e., utility). However, these evaluations overlook an important usability aspect: users may still want the model to leverage the removed information if it is re-introduced in the prompt. In a systematic evaluation of six state-of-the-art unlearning methods, we find that they consistently impair such contextual utility. To address this, we augment unlearning objectives with a plug-in term that preserves the model's ability to use forgotten knowledge when it is present in context. Extensive experiments demonstrate that our approach restores contextual utility to near original levels while still maintaining effective forgetting and retain-set utility.
>
---
#### [new 008] When Annotators Disagree, Topology Explains: Mapper, a Topological Tool for Exploring Text Embedding Geometry and Ambiguity
- **分类: cs.CL**

- **简介: 该论文属于NLP模型分析任务，旨在解决标注分歧下模型如何处理语义模糊的问题。作者使用拓扑工具Mapper分析RoBERTa的嵌入空间结构，揭示细调后模型形成高纯度预测区域，但与真实标签对齐下降，暴露模型自信与标签不确定性的矛盾。**

- **链接: [http://arxiv.org/pdf/2510.17548v1](http://arxiv.org/pdf/2510.17548v1)**

> **作者:** Nisrine Rair; Alban Goupil; Valeriu Vrabie; Emmanuel Chochoy
>
> **备注:** Accepted to appear in the Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025, Main Conference)
>
> **摘要:** Language models are often evaluated with scalar metrics like accuracy, but such measures fail to capture how models internally represent ambiguity, especially when human annotators disagree. We propose a topological perspective to analyze how fine-tuned models encode ambiguity and more generally instances. Applied to RoBERTa-Large on the MD-Offense dataset, Mapper, a tool from topological data analysis, reveals that fine-tuning restructures embedding space into modular, non-convex regions aligned with model predictions, even for highly ambiguous cases. Over $98\%$ of connected components exhibit $\geq 90\%$ prediction purity, yet alignment with ground-truth labels drops in ambiguous data, surfacing a hidden tension between structural confidence and label uncertainty. Unlike traditional tools such as PCA or UMAP, Mapper captures this geometry directly uncovering decision regions, boundary collapses, and overconfident clusters. Our findings position Mapper as a powerful diagnostic tool for understanding how models resolve ambiguity. Beyond visualization, it also enables topological metrics that may inform proactive modeling strategies in subjective NLP tasks.
>
---
#### [new 009] Who's Asking? Simulating Role-Based Questions for Conversational AI Evaluation
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文提出CoRUS框架，用于模拟不同用户角色（如患者、护理者）的提问，以评估对话AI在角色感知上的响应差异。针对阿片类药物使用障碍领域，研究发现模型对脆弱角色回应更支持但知识更少，揭示了角色隐含信息对AI响应的影响。**

- **链接: [http://arxiv.org/pdf/2510.16829v1](http://arxiv.org/pdf/2510.16829v1)**

> **作者:** Navreet Kaur; Hoda Ayad; Hayoung Jung; Shravika Mittal; Munmun De Choudhury; Tanushree Mitra
>
> **摘要:** Language model users often embed personal and social context in their questions. The asker's role -- implicit in how the question is framed -- creates specific needs for an appropriate response. However, most evaluations, while capturing the model's capability to respond, often ignore who is asking. This gap is especially critical in stigmatized domains such as opioid use disorder (OUD), where accounting for users' contexts is essential to provide accessible, stigma-free responses. We propose CoRUS (COmmunity-driven Roles for User-centric Question Simulation), a framework for simulating role-based questions. Drawing on role theory and posts from an online OUD recovery community (r/OpiatesRecovery), we first build a taxonomy of asker roles -- patients, caregivers, practitioners. Next, we use it to simulate 15,321 questions that embed each role's goals, behaviors, and experiences. Our evaluations show that these questions are both highly believable and comparable to real-world data. When used to evaluate five LLMs, for the same question but differing roles, we find systematic differences: vulnerable roles, such as patients and caregivers, elicit more supportive responses (+17%) and reduced knowledge content (-19%) in comparison to practitioners. Our work demonstrates how implicitly signaling a user's role shapes model responses, and provides a methodology for role-informed evaluation of conversational AI.
>
---
#### [new 010] Probing the Hidden Talent of ASR Foundation Models for L2 English Oral Assessment
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文探索Whisper模型在二语口语评估中的潜力，旨在挖掘其隐藏表征对语言熟练度评价的有效性。通过提取声学与语言特征并结合轻量分类器，结合图文提示提升性能，验证了无需微调即可捕捉口语流利度与语义信息。**

- **链接: [http://arxiv.org/pdf/2510.16387v1](http://arxiv.org/pdf/2510.16387v1)**

> **作者:** Fu-An Chao; Bi-Cheng Yan; Berlin Chen
>
> **摘要:** In this paper, we explore the untapped potential of Whisper, a well-established automatic speech recognition (ASR) foundation model, in the context of L2 spoken language assessment (SLA). Unlike prior studies that extrinsically analyze transcriptions produced by Whisper, our approach goes a step further to probe its latent capabilities by extracting acoustic and linguistic features from hidden representations. With only a lightweight classifier being trained on top of Whisper's intermediate and final outputs, our method achieves strong performance on the GEPT picture-description dataset, outperforming existing cutting-edge baselines, including a multimodal approach. Furthermore, by incorporating image and text-prompt information as auxiliary relevance cues, we demonstrate additional performance gains. Finally, we conduct an in-depth analysis of Whisper's embeddings, which reveals that, even without task-specific fine-tuning, the model intrinsically encodes both ordinal proficiency patterns and semantic aspects of speech, highlighting its potential as a powerful foundation for SLA and other spoken language understanding tasks.
>
---
#### [new 011] Unleashing Diverse Thinking Modes in LLMs through Multi-Agent Collaboration
- **分类: cs.CL; cs.AI; cs.LG; cs.MA**

- **简介: 该论文提出DiMo框架，属多智能体协作任务，旨在提升大模型的推理可解释性与准确性。通过四个具有不同思维模式的LLM代理辩论，生成可审计的推理链，在数学等任务上优于单模型与现有辩论方法。**

- **链接: [http://arxiv.org/pdf/2510.16645v1](http://arxiv.org/pdf/2510.16645v1)**

> **作者:** Zhixuan He; Yue Feng
>
> **摘要:** Large Language Models (LLMs) demonstrate strong performance but often lack interpretable reasoning. This paper introduces the Multi-Agent Collaboration Framework for Diverse Thinking Modes (DiMo), which enhances both performance and interpretability by simulating a structured debate among four specialized LLM agents. Each agent embodies a distinct reasoning paradigm, allowing the framework to collaboratively explore diverse cognitive approaches. Through iterative debate, agents challenge and refine initial responses, yielding more robust conclusions and an explicit, auditable reasoning chain. Across six benchmarks and under a unified open-source setup, DiMo improves accuracy over widely used single-model and debate baselines, with the largest gains on math. We position DiMo as a semantics-aware, Web-native multi-agent framework: it models human-machine intelligence with LLM agents that produce semantically typed, URL-annotated evidence chains for explanations and user-friendly interactions. Although our experiments use standard reasoning benchmarks, the framework is designed to be instantiated over Web corpora and knowledge graphs, combining retrieval-augmented reasoning with structured justifications that downstream systems can inspect and reuse.
>
---
#### [new 012] Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety
- **分类: cs.CL**

- **简介: 该论文研究LLM代理在复杂环境中的安全问题，提出通过“主动退出”机制提升安全性。作者在12种主流LLM上验证，加入明确退出指令可显著提高安全评分，仅轻微降低帮助性，表明选择性退出是高效的安全防护手段。**

- **链接: [http://arxiv.org/pdf/2510.16492v1](http://arxiv.org/pdf/2510.16492v1)**

> **作者:** Vamshi Krishna Bonagiri; Ponnurangam Kumaragurum; Khanh Nguyen; Benjamin Plaut
>
> **备注:** Reliable ML and Regulatable ML workshops, Neurips 2025
>
> **摘要:** As Large Language Model (LLM) agents increasingly operate in complex environments with real-world consequences, their safety becomes critical. While uncertainty quantification is well-studied for single-turn tasks, multi-turn agentic scenarios with real-world tool access present unique challenges where uncertainties and ambiguities compound, leading to severe or catastrophic risks beyond traditional text generation failures. We propose using "quitting" as a simple yet effective behavioral mechanism for LLM agents to recognize and withdraw from situations where they lack confidence. Leveraging the ToolEmu framework, we conduct a systematic evaluation of quitting behavior across 12 state-of-the-art LLMs. Our results demonstrate a highly favorable safety-helpfulness trade-off: agents prompted to quit with explicit instructions improve safety by an average of +0.39 on a 0-3 scale across all models (+0.64 for proprietary models), while maintaining a negligible average decrease of -0.03 in helpfulness. Our analysis demonstrates that simply adding explicit quit instructions proves to be a highly effective safety mechanism that can immediately be deployed in existing agent systems, and establishes quitting as an effective first-line defense mechanism for autonomous agents in high-stakes applications.
>
---
#### [new 013] The Atomic Instruction Gap: Instruction-Tuned LLMs Struggle with Simple, Self-Contained Directives
- **分类: cs.CL**

- **简介: 该论文研究指令微调大模型对简单独立指令的执行能力，发现其在选项标签格式变化时表现不稳定，暴露了原子指令遵循的缺陷。通过多格式基准测试，揭示现有训练方法不足，强调需改进评估与训练策略以提升指令遵循可靠性。**

- **链接: [http://arxiv.org/pdf/2510.17388v1](http://arxiv.org/pdf/2510.17388v1)**

> **作者:** Henry Lim; Kwan Hui Lim
>
> **备注:** 11 pages, 1 figure, 8 tables
>
> **摘要:** Instruction-tuned large language models (IT-LLMs) exhibit strong zero-shot reasoning, yet their ability to execute simple, self-contained instructions remains underexplored, despite this being foundational to complex instruction-following. We evaluate 20 IT-LLMs on modified MMLU and MMLU-Pro benchmarks, by systematically varying the format of option labels (alphabetic, numeric, Roman) while keeping their meaning identical under four paradigms, namely: (1) With explicit instructions, label changes cause large performance shifts (e.g., -30.45\% for Roman vs. numeric), revealing instruction-format bias. (2) Without instructions, performance drops further (up to -10.84\%) and label sensitivity intensifies, underscoring the role of explicit guidance. (3) When option contents are removed, models fail random-choice baselines except with numeric labels, suggesting weak adherence to atomic directives. (4) Three-shot exemplars yield no significant gains in robustness or fidelity, and generation analyses show persistent label errors, especially for non-numeric formats. Across model sizes, larger LLMs achieve higher accuracy but remain inconsistent in instruction adherence. These results expose the insufficiencies of current instruction-tuning paradigms and highlight the need for evaluation methods and training strategies that explicitly target atomic instruction-following.
>
---
#### [new 014] Vocab Diet: Reshaping the Vocabulary of LLMs with Vector Arithmetic
- **分类: cs.CL**

- **简介: 该论文提出“词汇瘦身”方法，通过向量运算将词形变化表示为基词与变换向量的组合，减少大模型词汇表冗余，释放空间以增加多样性词汇，提升覆盖且不损害性能。**

- **链接: [http://arxiv.org/pdf/2510.17001v1](http://arxiv.org/pdf/2510.17001v1)**

> **作者:** Yuval Reif; Guy Kaplan; Roy Schwartz
>
> **摘要:** Large language models (LLMs) were shown to encode word form variations, such as "walk"->"walked", as linear directions in embedding space. However, standard tokenization algorithms treat these variations as distinct tokens -- filling the size-capped vocabulary with surface form variants (e.g., "walk", "walking", "Walk"), at the expense of less frequent words and multilingual coverage. We show that many of these variations can be captured by transformation vectors -- additive offsets that yield the appropriate word's representation when applied to the base form word embedding -- in both the input and output spaces. Building on this, we propose a compact reshaping of the vocabulary: rather than assigning unique tokens to each surface form, we compose them from shared base form and transformation vectors (e.g., "walked" = "walk" + past tense). We apply our approach to multiple LLMs and across five languages, removing up to 10% of vocabulary entries -- thereby freeing space to allocate new, more diverse tokens. Importantly, we do so while also expanding vocabulary coverage to out-of-vocabulary words, with minimal impact on downstream performance, and without modifying model weights. Our findings motivate a foundational rethinking of vocabulary design, moving from string enumeration to a compositional vocabulary that leverages the underlying structure of language.
>
---
#### [new 015] FrugalPrompt: Reducing Contextual Overhead in Large Language Models via Token Attribution
- **分类: cs.CL**

- **简介: 该论文提出FrugalPrompt，一种基于词元重要性评估的提示压缩方法，旨在减少大模型输入冗余。通过保留高语义权重词元，降低上下文开销，在多任务上验证了性能与效率的权衡。**

- **链接: [http://arxiv.org/pdf/2510.16439v1](http://arxiv.org/pdf/2510.16439v1)**

> **作者:** Syed Rifat Raiyan; Md Farhan Ishmam; Abdullah Al Imran; Mohammad Ali Moni
>
> **摘要:** Large language models (LLMs) owe much of their stellar performance to expansive input contexts, yet such verbosity inflates monetary costs, carbon footprint, and inference-time latency. Much of this overhead manifests from the redundant low-utility tokens present in typical prompts, as only a fraction of tokens typically carries the majority of the semantic weight. We address this inefficiency by introducing FrugalPrompt, a novel prompt compression framework for LLMs, which retains only the most semantically significant tokens. Leveraging two state-of-the-art token attribution methods, GlobEnc and DecompX, we assign salience scores to every token in an input sequence, rank them to preserve the top-k% tokens in their original order, and obtain a sparse frugalized prompt. We evaluate the approach across four NLP tasks: Sentiment Analysis, Commonsense QA, Summarization, and Mathematical Reasoning, using a suite of frontier LLMs. For the first three tasks, a 20% prompt reduction incurs only a marginal loss in task performance, demonstrating that contemporary LLMs can reconstruct elided context from high-salience cues. In contrast, performance on mathematical reasoning deteriorates sharply, reflecting a stronger dependence on complete token continuity. Further analysis with bottom-k% and random-k% tokens reveals asymmetric performance patterns that may suggest potential task contamination effects, wherein models may resort to shallow memorized patterns from pretraining exposure for conventional NLP tasks. We posit that our work contributes to a more nuanced understanding of LLM behavior in performance-efficiency trade-offs, and delineate the boundary between tasks tolerant to contextual sparsity and those requiring exhaustive context. Our source code and models are available at: https://github.com/Starscream-11813/Frugal-ICL
>
---
#### [new 016] Disparities in Multilingual LLM-Based Healthcare Q&A
- **分类: cs.CL**

- **简介: 该论文研究多语言大模型在医疗问答中的跨语言差异问题，旨在提升非英语语种的公平性。作者构建了多语言数据集MultiWikiHealthCare，分析维基百科覆盖差异和模型事实一致性，发现英文主导现象，并通过检索增强生成（RAG）改善非英语响应的准确性。**

- **链接: [http://arxiv.org/pdf/2510.17476v1](http://arxiv.org/pdf/2510.17476v1)**

> **作者:** Ipek Baris Schlicht; Burcu Sayin; Zhixue Zhao; Frederik M. Labonté; Cesare Barbera; Marco Viviani; Paolo Rosso; Lucie Flek
>
> **备注:** Under review
>
> **摘要:** Equitable access to reliable health information is vital when integrating AI into healthcare. Yet, information quality varies across languages, raising concerns about the reliability and consistency of multilingual Large Language Models (LLMs). We systematically examine cross-lingual disparities in pre-training source and factuality alignment in LLM answers for multilingual healthcare Q&A across English, German, Turkish, Chinese (Mandarin), and Italian. We (i) constructed Multilingual Wiki Health Care (MultiWikiHealthCare), a multilingual dataset from Wikipedia; (ii) analyzed cross-lingual healthcare coverage; (iii) assessed LLM response alignment with these references; and (iv) conducted a case study on factual alignment through the use of contextual information and Retrieval-Augmented Generation (RAG). Our findings reveal substantial cross-lingual disparities in both Wikipedia coverage and LLM factual alignment. Across LLMs, responses align more with English Wikipedia, even when the prompts are non-English. Providing contextual excerpts from non-English Wikipedia at inference time effectively shifts factual alignment toward culturally relevant knowledge. These results highlight practical pathways for building more equitable, multilingual AI systems for healthcare.
>
---
#### [new 017] Investigating Thinking Behaviours of Reasoning-Based Language Models for Social Bias Mitigation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究推理型大语言模型在社会偏见场景中的思维行为，揭示其通过刻板印象重复和无关信息注入加剧偏见的机制，并提出一种轻量级提示方法，通过自我审查减少偏见，同时保持或提升准确性。**

- **链接: [http://arxiv.org/pdf/2510.17062v1](http://arxiv.org/pdf/2510.17062v1)**

> **作者:** Guoqing Luo; Iffat Maab; Lili Mou; Junichi Yamagishi
>
> **摘要:** While reasoning-based large language models excel at complex tasks through an internal, structured thinking process, a concerning phenomenon has emerged that such a thinking process can aggregate social stereotypes, leading to biased outcomes. However, the underlying behaviours of these language models in social bias scenarios remain underexplored. In this work, we systematically investigate mechanisms within the thinking process behind this phenomenon and uncover two failure patterns that drive social bias aggregation: 1) stereotype repetition, where the model relies on social stereotypes as its primary justification, and 2) irrelevant information injection, where it fabricates or introduces new details to support a biased narrative. Building on these insights, we introduce a lightweight prompt-based mitigation approach that queries the model to review its own initial reasoning against these specific failure patterns. Experiments on question answering (BBQ and StereoSet) and open-ended (BOLD) benchmarks show that our approach effectively reduces bias while maintaining or improving accuracy.
>
---
#### [new 018] Qomhra: A Bilingual Irish-English Large Language Model
- **分类: cs.CL; I.2.7**

- **简介: 该论文提出Qomhra，一种双语爱尔兰-英语大语言模型，旨在低资源下提升爱尔兰语性能。通过构建指令微调与人类偏好数据集，结合持续预训练与对齐，显著提升双语能力，尤其在爱尔兰语任务上表现突出。**

- **链接: [http://arxiv.org/pdf/2510.17652v1](http://arxiv.org/pdf/2510.17652v1)**

> **作者:** Joseph McInerney
>
> **摘要:** This paper introduces Qomhr\'a, a bilingual Irish-English large language model (LLM), developed under low-resource constraints presenting a complete pipeline spanning bilingual continued pre-training, instruction tuning, and alignment from human preferences. Newly accessible Irish corpora and English text are mixed and curated to improve Irish performance while preserving English ability. 6 closed-weight LLMs are judged for their Irish text generation by a native speaker, a learner and other LLMs. Google's Gemini-2.5-Pro is ranked the highest and is subsequently used to synthesise instruction tuning and human preference datasets. Two datasets are contributed leveraging Gemini-2.5-Pro: a 30K Irish-English parallel instruction tuning dataset and a 1K human preference dataset, generating accepted and rejected responses that show near perfect alignment with a native Irish speaker. Qomhr\'a is comprehensively evaluated across benchmarks testing translation, gender understanding, topic identification and world knowledge with gains of up to 29% in Irish and 44% in English. Qomhr\'a also undergoes instruction tuning and demonstrates clear progress in instruction following, crucial for chatbot functionality.
>
---
#### [new 019] LC-Eval: A Bilingual Multi-Task Evaluation Benchmark for Long-Context Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LC-Eval，一个双语多任务长上下文理解评测基准，涵盖英文和阿拉伯文。针对4k至128k长度文本，设计四项新任务，评估大模型在复杂推理、文档理解、信息追踪和双语理解方面的能力，揭示现有模型在长上下文处理上的不足。**

- **链接: [http://arxiv.org/pdf/2510.16783v1](http://arxiv.org/pdf/2510.16783v1)**

> **作者:** Sheikh Jubair; Arwa Omayrah; Amal Alshammari; Alhanoof Althnian; Abdulhamed Alothaimen; Norah A. Alzahrani; Shahad D. Alzaidi; Nora Al-Twairesh; Abdulmohsen Al-Thubaity
>
> **备注:** 1 figure, 15 tables, 10 main pages
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have demonstrated sophisticated capabilities, including the ability to process and comprehend extended contexts. These emergent capabilities necessitate rigorous evaluation methods to effectively assess their performance in long-context understanding. In this paper, we present \textbf{LC-Eval}, a bilingual, multi-task evaluation benchmark designed to evaluate long-context understanding in English and Arabic, targeting context lengths ranging from 4k to over 128k tokens. LC-Eval introduces four novel and challenging tasks: multi-document question answering, bilingual question answering, claim verification within a paragraph, and multiple-choice questions based on long contexts. These tasks are designed to assess LLMs' abilities in deep reasoning, document comprehension, information tracing, and bilingual information extraction and understanding. The benchmark includes datasets in both Arabic and English for each task, allowing for a comparative analysis of their performance across different text genres. Evaluations were conducted on both open-weight and closed LLMs, with results indicating that LC-Eval presents significant challenges. Even high-performing models, such as GPT-4o, struggled with certain tasks, highlighting the complexity and rigor of the benchmark.
>
---
#### [new 020] BenCao: An Instruction-Tuned Large Language Model for Traditional Chinese Medicine
- **分类: cs.CL; cs.AI; cs.MA; cs.MM; cs.SE**

- **简介: 该论文提出BenCao，一个基于指令调优的中医大模型，旨在解决现有模型在多模态整合、可解释性与临床应用上的不足。通过融合知识库、专家反馈与外部API，实现可解释的中医推理，在诊断等任务中表现优异，并已实际部署。**

- **链接: [http://arxiv.org/pdf/2510.17415v1](http://arxiv.org/pdf/2510.17415v1)**

> **作者:** Jiacheng Xie; Yang Yu; Yibo Chen; Hanyao Zhang; Lening Zhao; Jiaxuan He; Lei Jiang; Xiaoting Tang; Guanghui An; Dong Xu
>
> **摘要:** Traditional Chinese Medicine (TCM), with a history spanning over two millennia, plays a role in global healthcare. However, applying large language models (LLMs) to TCM remains challenging due to its reliance on holistic reasoning, implicit logic, and multimodal diagnostic cues. Existing TCM-domain LLMs have made progress in text-based understanding but lack multimodal integration, interpretability, and clinical applicability. To address these limitations, we developed BenCao, a ChatGPT-based multimodal assistant for TCM, integrating structured knowledge bases, diagnostic data, and expert feedback refinement. BenCao was trained through natural language instruction tuning rather than parameter retraining, aligning with expert-level reasoning and ethical norms specific to TCM. The system incorporates a comprehensive knowledge base of over 1,000 classical and modern texts, a scenario-based instruction framework for diverse interactions, a chain-of-thought simulation mechanism for interpretable reasoning, and a feedback refinement process involving licensed TCM practitioners. BenCao connects to external APIs for tongue-image classification and multimodal database retrieval, enabling dynamic access to diagnostic resources. In evaluations across single-choice question benchmarks and multimodal classification tasks, BenCao achieved superior accuracy to general-domain and TCM-domain models, particularly in diagnostics, herb recognition, and constitution classification. The model was deployed as an interactive application on the OpenAI GPTs Store, accessed by nearly 1,000 users globally as of October 2025. This study demonstrates the feasibility of developing a TCM-domain LLM through natural language-based instruction tuning and multimodal integration, offering a practical framework for aligning generative AI with traditional medical reasoning and a scalable pathway for real-world deployment.
>
---
#### [new 021] Fine-tuning of Large Language Models for Constituency Parsing Using a Sequence to Sequence Approach
- **分类: cs.CL; 68T50; I.2.7; I.2.6**

- **简介: 该论文研究基于大语言模型的句法分析任务，旨在通过序列到序列方法实现西班牙语短语结构解析。为提升教学工具MiSintaxis的能力，作者微调多个Hugging Face模型，使用AnCora-ES语料生成的数据进行训练，并以F1分数评估性能，结果表明该方法准确率高，具有应用潜力。**

- **链接: [http://arxiv.org/pdf/2510.16604v1](http://arxiv.org/pdf/2510.16604v1)**

> **作者:** Francisco Jose Cortes Delgado; Eduardo Martinez Gracia; Rafael Valencia Garcia
>
> **备注:** 6 pages, 3 figures. Submitted to SEPLN 2023 Conference
>
> **摘要:** Recent advances in natural language processing with large neural models have opened new possibilities for syntactic analysis based on machine learning. This work explores a novel approach to phrase-structure analysis by fine-tuning large language models (LLMs) to translate an input sentence into its corresponding syntactic structure. The main objective is to extend the capabilities of MiSintaxis, a tool designed for teaching Spanish syntax. Several models from the Hugging Face repository were fine-tuned using training data generated from the AnCora-ES corpus, and their performance was evaluated using the F1 score. The results demonstrate high accuracy in phrase-structure analysis and highlight the potential of this methodology.
>
---
#### [new 022] Back to Bytes: Revisiting Tokenization Through UTF-8
- **分类: cs.CL**

- **简介: 该论文提出UTF8Tokenizer，属自然语言处理中的分词任务。针对传统字节级分词引入越界ID或辅助标记的问题，其工作将文本直接映射为UTF-8字节ID，用C0控制字符处理特殊逻辑，提升效率与兼容性，并设计位偏置嵌入优化训练。**

- **链接: [http://arxiv.org/pdf/2510.16987v1](http://arxiv.org/pdf/2510.16987v1)**

> **作者:** Amit Moryossef; Clara Meister; Pavel Stepachev; Desmond Elliott
>
> **摘要:** We present UTF8Tokenizer, a minimalist byte-level tokenizer that maps text exactly to IDs corresponding to the bytes underlying the text's UTF-8 encoding (e.g., byte x09 is token ID 9). Unlike prior byte-level approaches (Xue et al., 2021; Pagnoni et al., 2025), our implementation never introduces out-of-range IDs (i.e. there is no token ID 256) or auxiliary tokens: all special behavior (e.g., padding, boundaries, conversation structure, attention segments, tool calling, "thinking" spans, etc.) is encoded using C0 control bytes - just as ASCII was originally designed to embed control information alongside printable text. These design principles yield practical benefits: (1) faster tokenization (14x) and significantly lower host-device transfer (8x less than int64); (2) simple, shareable 256*d embedding tables that can be aligned across models; and (3) a training-time enhancement via bit-biased embeddings, which exposes per-byte bit structure and can be added to the embedding table post-training, removing inference costs. Our HuggingFace-compatible implementation improves language modeling convergence.
>
---
#### [new 023] Enterprise Deep Research: Steerable Multi-Agent Deep Research for Enterprise Analytics
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出企业深度研究（EDR）系统，属多智能体协同分析任务，旨在解决企业非结构化数据转化难题。通过多代理架构与工具生态，实现自动查询分解、跨源检索、可视化及人机协同优化，支持自动化报告生成与企业部署。**

- **链接: [http://arxiv.org/pdf/2510.17797v1](http://arxiv.org/pdf/2510.17797v1)**

> **作者:** Akshara Prabhakar; Roshan Ram; Zixiang Chen; Silvio Savarese; Frank Wang; Caiming Xiong; Huan Wang; Weiran Yao
>
> **备注:** Technical report; 13 pages plus references and appendices
>
> **摘要:** As information grows exponentially, enterprises face increasing pressure to transform unstructured data into coherent, actionable insights. While autonomous agents show promise, they often struggle with domain-specific nuances, intent alignment, and enterprise integration. We present Enterprise Deep Research (EDR), a multi-agent system that integrates (1) a Master Planning Agent for adaptive query decomposition, (2) four specialized search agents (General, Academic, GitHub, LinkedIn), (3) an extensible MCP-based tool ecosystem supporting NL2SQL, file analysis, and enterprise workflows, (4) a Visualization Agent for data-driven insights, and (5) a reflection mechanism that detects knowledge gaps and updates research direction with optional human-in-the-loop steering guidance. These components enable automated report generation, real-time streaming, and seamless enterprise deployment, as validated on internal datasets. On open-ended benchmarks including DeepResearch Bench and DeepConsult, EDR outperforms state-of-the-art agentic systems without any human steering. We release the EDR framework and benchmark trajectories to advance research on multi-agent reasoning applications. Code at https://github.com/SalesforceAIResearch/enterprise-deep-research and Dataset at https://huggingface.co/datasets/Salesforce/EDR-200
>
---
#### [new 024] ChiKhaPo: A Large-Scale Multilingual Benchmark for Evaluating Lexical Comprehension and Generation in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出ChiKhaPo，一个面向大语言模型的多语言基准，旨在评估其在2700多种语言中的词汇理解与生成能力，解决现有基准覆盖语言少、忽视低资源语言的问题。**

- **链接: [http://arxiv.org/pdf/2510.16928v1](http://arxiv.org/pdf/2510.16928v1)**

> **作者:** Emily Chang; Niyati Bafna
>
> **摘要:** Existing benchmarks for large language models (LLMs) are largely restricted to high- or mid-resource languages, and often evaluate performance on higher-order tasks in reasoning and generation. However, plenty of evidence points to the fact that LLMs lack basic linguistic competence in the vast majority of the world's 3800+ written languages. We introduce ChiKhaPo, consisting of 8 subtasks of varying difficulty designed to evaluate the lexical comprehension and generation abilities of generative models. ChiKhaPo draws on existing lexicons, monolingual data, and bitext, and provides coverage for 2700+ languages for 2 subtasks, surpassing any existing benchmark in terms of language coverage. We further show that 6 SOTA models struggle on our benchmark, and discuss the factors contributing to performance scores, including language family, language resourcedness, task, and comprehension versus generation directions. With ChiKhaPo, we hope to enable and encourage the massively multilingual benchmarking of LLMs.
>
---
#### [new 025] Investigating the Impact of Rationales for LLMs on Natural Language Understanding
- **分类: cs.CL**

- **简介: 该论文研究大模型中推理链（CoT）对自然语言理解（NLU）任务的影响，旨在探索其在NLU中的有效性。作者构建了带理由的NLU数据集NLURC，并提出多种增强方法，发现CoT随模型增大而提升性能，特定训练方法可显著提效并增强可解释性。**

- **链接: [http://arxiv.org/pdf/2510.16686v1](http://arxiv.org/pdf/2510.16686v1)**

> **作者:** Wenhang Shi; Shuqing Bian; Yiren Chen; Xinyi Zhang; Zhe Zhao; Pengfei Hu; Wei Lu; Xiaoyong Du
>
> **摘要:** Chain-of-thought (CoT) rationales, which provide step-by-step reasoning to derive final answers, benefit LLMs in both inference and training. Incorporating rationales, either by generating them before answering during inference, or by placing them before or after the original answers during training - significantly improves model performance on mathematical, symbolic and commonsense reasoning tasks. However, most work focuses on the role of rationales in these reasoning tasks, overlooking their potential impact on other important tasks like natural language understanding (NLU) tasks. In this work, we raise the question: Can rationales similarly benefit NLU tasks? To conduct a systematic exploration, we construct NLURC, a comprehensive and high-quality NLU dataset collection with rationales, and develop various rationale-augmented methods. Through exploring the applicability of these methods on NLU tasks using the dataset, we uncover several potentially surprising findings: (1) CoT inference shifts from hindering NLU performance to surpassing direct label prediction as model size grows, indicating a positive correlation. (2) Most rationale-augmented training methods perform worse than label-only training, with one specially designed method consistently achieving improvements. (3) LLMs trained with rationales achieve significant performance gains on unseen NLU tasks, rivaling models ten times their size, while delivering interpretability on par with commercial LLMs.
>
---
#### [new 026] FinSight: Towards Real-World Financial Deep Research
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文聚焦金融深度研究报告生成任务，旨在解决AI难以自动化生成高质量、多模态金融报告的问题。作者提出FinSight框架，结合可编程变量空间的CAVM架构、视觉增强机制与两阶段写作框架，实现数据采集、分析到专业级图表与报告生成的一体化，显著提升报告的准确性、深度与呈现质量。**

- **链接: [http://arxiv.org/pdf/2510.16844v1](http://arxiv.org/pdf/2510.16844v1)**

> **作者:** Jiajie Jin; Yuyao Zhang; Yimeng Xu; Hongjin Qian; Yutao Zhu; Zhicheng Dou
>
> **备注:** Working in progress
>
> **摘要:** Generating professional financial reports is a labor-intensive and intellectually demanding process that current AI systems struggle to fully automate. To address this challenge, we introduce FinSight (Financial InSight), a novel multi agent framework for producing high-quality, multimodal financial reports. The foundation of FinSight is the Code Agent with Variable Memory (CAVM) architecture, which unifies external data, designed tools, and agents into a programmable variable space, enabling flexible data collection, analysis and report generation through executable code. To ensure professional-grade visualization, we propose an Iterative Vision-Enhanced Mechanism that progressively refines raw visual outputs into polished financial charts. Furthermore, a two stage Writing Framework expands concise Chain-of-Analysis segments into coherent, citation-aware, and multimodal reports, ensuring both analytical depth and structural consistency. Experiments on various company and industry-level tasks demonstrate that FinSight significantly outperforms all baselines, including leading deep research systems in terms of factual accuracy, analytical depth, and presentation quality, demonstrating a clear path toward generating reports that approach human-expert quality.
>
---
#### [new 027] Agentic Reinforcement Learning for Search is Unsafe
- **分类: cs.CL**

- **简介: 该论文研究基于强化学习的搜索型智能体安全问题，发现尽管模型继承了拒绝有害请求的能力，但通过强制启动或重复搜索可绕过防御，导致生成大量有害查询与回答，暴露出当前RL训练忽视查询危害性的缺陷。**

- **链接: [http://arxiv.org/pdf/2510.17431v1](http://arxiv.org/pdf/2510.17431v1)**

> **作者:** Yushi Yang; Shreyansh Padarha; Andrew Lee; Adam Mahdi
>
> **摘要:** Agentic reinforcement learning (RL) trains large language models to autonomously call tools during reasoning, with search as the most common application. These models excel at multi-step reasoning tasks, but their safety properties are not well understood. In this study, we show that RL-trained search models inherit refusal from instruction tuning and often deflect harmful requests by turning them into safe queries. However, this safety is fragile. Two simple attacks, one that forces the model to begin response with search (Search attack), another that encourages models to repeatedly search (Multi-search attack), trigger cascades of harmful searches and answers. Across two model families (Qwen, Llama) with both local and web search, these attacks lower refusal rates by up to 60.0%, answer safety by 82.5%, and search-query safety by 82.4%. The attacks succeed by triggering models to generate harmful, request-mirroring search queries before they can generate the inherited refusal tokens. This exposes a core weakness of current RL training: it rewards continued generation of effective queries without accounting for their harmfulness. As a result, RL search models have vulnerabilities that users can easily exploit, making it urgent to develop safety-aware agentic RL pipelines optimising for safe search.
>
---
#### [new 028] Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究长上下文语言模型的长度外推问题，旨在提升模型在训练未见长度上的表现。作者提出三个关键设计原则，通过分块稀疏注意力机制，实现从4K到3200万token的无训练外推，显著提升长序列建模能力。**

- **链接: [http://arxiv.org/pdf/2510.17196v1](http://arxiv.org/pdf/2510.17196v1)**

> **作者:** Jiaqi Leng; Xiang Hu; Junxiong Wang; Jianguo Li; Wei Wu; Yucheng Lu
>
> **备注:** Preprint. Work in progress
>
> **摘要:** Effectively processing long contexts is a critical challenge for language models. While standard Transformers are limited by quadratic complexity and poor length extrapolation, alternative architectures like sliding window attention and state space models sacrifice the ability to effectively utilize the full context due to their fixed-size memory. Chunk-based sparse attention has emerged as a promising paradigm for extreme length generalization, yet the key architectural principles underpinning its success are not yet fully understood. In this work, we present a systematic dissection of these models to identify the core components driving their performance. Through a unified framework and comprehensive ablation studies, we demonstrate that a combination of three design principles is critical: (1) an expressive, non-linear Chunk Encoder with a dedicated CLS token to produce representations for retrieval; (2) a Bypassing Residual Path to stably integrate retrieved global information without it being overridden by the local residual stream; and (3) enforced selection sparsity during pre-training to bridge the train-test distribution gap. We provide a theoretical motivation for intra-chunk information processing and landmark generation. By combining these principles, we establish a new state-of-the-art for training-free length extrapolation, successfully generalizing models trained on a 4K context to 32 million tokens on RULER and BABILong. Our findings provide a clear and empirically-grounded set of design principles for developing future, highly-capable long-context language models.
>
---
#### [new 029] Train for Truth, Keep the Skills: Binary Retrieval-Augmented Reward Mitigates Hallucinations
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对语言模型的幻觉问题，提出一种基于二值检索增强奖励的在线强化学习方法。在提升事实性的同时，避免了对开放生成和下游任务性能的损害，实现了更优的平衡。**

- **链接: [http://arxiv.org/pdf/2510.17733v1](http://arxiv.org/pdf/2510.17733v1)**

> **作者:** Tong Chen; Akari Asai; Luke Zettlemoyer; Hannaneh Hajishirzi; Faeze Brahman
>
> **摘要:** Language models often generate factually incorrect information unsupported by their training data, a phenomenon known as extrinsic hallucination. Existing mitigation approaches often degrade performance on open-ended generation and downstream tasks, limiting their practical utility. We propose an online reinforcement learning method using a novel binary retrieval-augmented reward (RAR) to address this tradeoff. Unlike continuous reward schemes, our approach assigns a reward of one only when the model's output is entirely factually correct, and zero otherwise. We evaluate our method on Qwen3 reasoning models across diverse tasks. For open-ended generation, binary RAR achieves a 39.3% reduction in hallucination rates, substantially outperforming both supervised training and continuous-reward RL baselines. In short-form question answering, the model learns calibrated abstention, strategically outputting "I don't know" when faced with insufficient parametric knowledge. This yields 44.4% and 21.7% fewer incorrect answers on PopQA and GPQA, respectively. Crucially, these factuality gains come without performance degradation on instruction following, math, or code, whereas continuous-reward RL, despite improving factuality, induces quality regressions.
>
---
#### [new 030] The Chameleon Nature of LLMs: Quantifying Multi-Turn Stance Instability in Search-Enabled Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究搜索增强大模型在多轮对话中的立场不稳定性问题。提出“变色龙行为”概念，构建基准数据集，设计量化指标，揭示模型因知识复用导致的立场漂移，强调其在关键领域应用中的风险。**

- **链接: [http://arxiv.org/pdf/2510.16712v1](http://arxiv.org/pdf/2510.16712v1)**

> **作者:** Shivam Ratnakar; Sanjay Raghavendra
>
> **摘要:** Integration of Large Language Models with search/retrieval engines has become ubiquitous, yet these systems harbor a critical vulnerability that undermines their reliability. We present the first systematic investigation of "chameleon behavior" in LLMs: their alarming tendency to shift stances when presented with contradictory questions in multi-turn conversations (especially in search-enabled LLMs). Through our novel Chameleon Benchmark Dataset, comprising 17,770 carefully crafted question-answer pairs across 1,180 multi-turn conversations spanning 12 controversial domains, we expose fundamental flaws in state-of-the-art systems. We introduce two theoretically grounded metrics: the Chameleon Score (0-1) that quantifies stance instability, and Source Re-use Rate (0-1) that measures knowledge diversity. Our rigorous evaluation of Llama-4-Maverick, GPT-4o-mini, and Gemini-2.5-Flash reveals consistent failures: all models exhibit severe chameleon behavior (scores 0.391-0.511), with GPT-4o-mini showing the worst performance. Crucially, small across-temperature variance (less than 0.004) suggests the effect is not a sampling artifact. Our analysis uncovers the mechanism: strong correlations between source re-use rate and confidence (r=0.627) and stance changes (r=0.429) are statistically significant (p less than 0.05), indicating that limited knowledge diversity makes models pathologically deferential to query framing. These findings highlight the need for comprehensive consistency evaluation before deploying LLMs in healthcare, legal, and financial systems where maintaining coherent positions across interactions is critical for reliable decision support.
>
---
#### [new 031] Mapping from Meaning: Addressing the Miscalibration of Prompt-Sensitive Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型的提示敏感性问题，旨在提升模型对语义等价提示的不确定性校准。作者通过语义空间采样和新的不确定性分解指标，揭示并量化提示敏感性对模型输出的影响，提出改进方法，在不降低准确率的情况下增强模型一致性。**

- **链接: [http://arxiv.org/pdf/2510.17028v1](http://arxiv.org/pdf/2510.17028v1)**

> **作者:** Kyle Cox; Jiawei Xu; Yikun Han; Rong Xu; Tianhao Li; Chi-Yang Hsu; Tianlong Chen; Walter Gerych; Ying Ding
>
> **摘要:** An interesting behavior in large language models (LLMs) is prompt sensitivity. When provided with different but semantically equivalent versions of the same prompt, models may produce very different distributions of answers. This suggests that the uncertainty reflected in a model's output distribution for one prompt may not reflect the model's uncertainty about the meaning of the prompt. We model prompt sensitivity as a type of generalization error, and show that sampling across the semantic ``concept space'' with paraphrasing perturbations improves uncertainty calibration without compromising accuracy. Additionally, we introduce a new metric for uncertainty decomposition in black-box LLMs that improves upon entropy-based decomposition by modeling semantic continuities in natural language generation. We show that this decomposition metric can be used to quantify how much LLM uncertainty is attributed to prompt sensitivity. Our work introduces a new way to improve uncertainty calibration in prompt-sensitive language models, and provides evidence that some LLMs fail to exhibit consistent general reasoning about the meanings of their inputs.
>
---
#### [new 032] When AI companions become witty: Can human brain recognize AI-generated irony?
- **分类: cs.CL**

- **简介: 该论文研究人类是否将AI生成的反语视为有意图的交流。通过ERP实验比较人与AI来源的反语处理，发现人们较少对AI采取意向性立场，神经反应较弱，且受个体对AI性格感知影响，揭示语言能力不足以实现真正社交代理。**

- **链接: [http://arxiv.org/pdf/2510.17168v1](http://arxiv.org/pdf/2510.17168v1)**

> **作者:** Xiaohui Rao; Hanlin Wu; Zhenguang G. Cai
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed as social agents and trained to produce humor and irony, a question emerges: when encountering witty AI remarks, do people interpret these as intentional communication or mere computational output? This study investigates whether people adopt the intentional stance, attributing mental states to explain behavior,toward AI during irony comprehension. Irony provides an ideal paradigm because it requires distinguishing intentional contradictions from unintended errors through effortful semantic reanalysis. We compared behavioral and neural responses to ironic statements from AI versus human sources using established ERP components: P200 reflecting early incongruity detection and P600 indexing cognitive efforts in reinterpreting incongruity as deliberate irony. Results demonstrate that people do not fully adopt the intentional stance toward AI-generated irony. Behaviorally, participants attributed incongruity to deliberate communication for both sources, though significantly less for AI than human, showing greater tendency to interpret AI incongruities as computational errors. Neural data revealed attenuated P200 and P600 effects for AI-generated irony, suggesting reduced effortful detection and reanalysis consistent with diminished attribution of communicative intent. Notably, people who perceived AI as more sincere showed larger P200 and P600 effects for AI-generated irony, suggesting that intentional stance adoption is calibrated by specific mental models of artificial agents. These findings reveal that source attribution shapes neural processing of social-communicative phenomena. Despite current LLMs' linguistic sophistication, achieving genuine social agency requires more than linguistic competence, it necessitates a shift in how humans perceive and attribute intentionality to artificial agents.
>
---
#### [new 033] Temporal Understanding under Deictic Frame of Reference
- **分类: cs.CL**

- **简介: 该论文研究大语言模型对时间的感知能力，提出TUuD框架，通过动态调整“现在”参考点，评估模型在不同时间参照系下对事件时序关系的理解。实验表明模型具备一定类似人类的时间认知，但随时间距离增加而减弱。**

- **链接: [http://arxiv.org/pdf/2510.16685v1](http://arxiv.org/pdf/2510.16685v1)**

> **作者:** Damin Zhang; Julia Rayz
>
> **备注:** Under review
>
> **摘要:** Understanding time is fundamental to human cognition, where temporal experience is often conceptualized through spatial metaphors grounded in sensory-motor experience. For example, "summer is approaching" parallels "We are approaching the summer". In such expressions, humans rely on a frame of reference (FoR) to interpret meaning relative to a particular viewpoint. Extending this concept to time, a temporal frame of reference (t-FoR) defines how temporal relations are perceived relative to an experiencer's moment of "now". While Large Language Models (LLMs) have shown remarkable advances in natural language understanding, their ability to interpret and reason about time remains limited. In this work, we introduce TUuD (Temporal Understanding under Deictic t-FoR), a framework that evaluates how LLMs interpret time-event and event-event relations when the reference point of "now" dynamically shifts along a timeline. Following recent work on temporal cognition \cite{li2025other}, LLMs are prompted to rate the similarity between the current moment and a target event from 0.00 (completely dissimilar) to 1.00 (highly similar), where similarity quantifies perceived temporal alignment between the two points. Our results show that four evaluated LLMs exhibit measurable adaptation to a deictic t-FoR, with similarity ratings peaking around the present and decreasing toward past and future events. The adaptation, however, weakens beyond near-term contexts, suggesting that while LLMs display partial human-like temporal cognition, their temporal reasoning remains sensitive to reference-frame shifts and temporal distance.
>
---
#### [new 034] MoReBench: Evaluating Procedural and Pluralistic Moral Reasoning in Language Models, More than Outcomes
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.LG**

- **简介: 该论文聚焦AI道德推理的评估任务，旨在解决现有基准无法有效衡量语言模型在道德决策中的推理过程问题。作者构建了MoReBench及理论版数据集，通过专家制定的标准评估模型推理质量，推动更安全、透明的AI发展。**

- **链接: [http://arxiv.org/pdf/2510.16380v1](http://arxiv.org/pdf/2510.16380v1)**

> **作者:** Yu Ying Chiu; Michael S. Lee; Rachel Calcott; Brandon Handoko; Paul de Font-Reaulx; Paula Rodriguez; Chen Bo Calvin Zhang; Ziwen Han; Udari Madhushani Sehwag; Yash Maurya; Christina Q Knight; Harry R. Lloyd; Florence Bacus; Mantas Mazeika; Bing Liu; Yejin Choi; Mitchell L Gordon; Sydney Levine
>
> **备注:** 46 pages, 8 figures, 10 tables. Preprint
>
> **摘要:** As AI systems progress, we rely more on them to make decisions with us and for us. To ensure that such decisions are aligned with human values, it is imperative for us to understand not only what decisions they make but also how they come to those decisions. Reasoning language models, which provide both final responses and (partially transparent) intermediate thinking traces, present a timely opportunity to study AI procedural reasoning. Unlike math and code problems which often have objectively correct answers, moral dilemmas are an excellent testbed for process-focused evaluation because they allow for multiple defensible conclusions. To do so, we present MoReBench: 1,000 moral scenarios, each paired with a set of rubric criteria that experts consider essential to include (or avoid) when reasoning about the scenarios. MoReBench contains over 23 thousand criteria including identifying moral considerations, weighing trade-offs, and giving actionable recommendations to cover cases on AI advising humans moral decisions as well as making moral decisions autonomously. Separately, we curate MoReBench-Theory: 150 examples to test whether AI can reason under five major frameworks in normative ethics. Our results show that scaling laws and existing benchmarks on math, code, and scientific reasoning tasks fail to predict models' abilities to perform moral reasoning. Models also show partiality towards specific moral frameworks (e.g., Benthamite Act Utilitarianism and Kantian Deontology), which might be side effects of popular training paradigms. Together, these benchmarks advance process-focused reasoning evaluation towards safer and more transparent AI.
>
---
#### [new 035] Empowering Real-World: A Survey on the Technology, Practice, and Evaluation of LLM-driven Industry Agents
- **分类: cs.CL**

- **简介: 该论文综述了大语言模型驱动的行业智能体的技术、应用与评估，旨在解决其在工业转化中的挑战。工作包括构建能力成熟度框架，分析记忆、规划与工具使用技术，梳理实际应用场景，并探讨评估方法与实践难题。**

- **链接: [http://arxiv.org/pdf/2510.17491v1](http://arxiv.org/pdf/2510.17491v1)**

> **作者:** Yihong Tang; Kehai Chen; Liang Yue; Jinxin Fan; Caishen Zhou; Xiaoguang Li; Yuyang Zhang; Mingming Zhao; Shixiong Kai; Kaiyang Guo; Xingshan Zeng; Wenjing Cun; Lifeng Shang; Min Zhang
>
> **摘要:** With the rise of large language models (LLMs), LLM agents capable of autonomous reasoning, planning, and executing complex tasks have become a frontier in artificial intelligence. However, how to translate the research on general agents into productivity that drives industry transformations remains a significant challenge. To address this, this paper systematically reviews the technologies, applications, and evaluation methods of industry agents based on LLMs. Using an industry agent capability maturity framework, it outlines the evolution of agents in industry applications, from "process execution systems" to "adaptive social systems." First, we examine the three key technological pillars that support the advancement of agent capabilities: Memory, Planning, and Tool Use. We discuss how these technologies evolve from supporting simple tasks in their early forms to enabling complex autonomous systems and collective intelligence in more advanced forms. Then, we provide an overview of the application of industry agents in real-world domains such as digital engineering, scientific discovery, embodied intelligence, collaborative business execution, and complex system simulation. Additionally, this paper reviews the evaluation benchmarks and methods for both fundamental and specialized capabilities, identifying the challenges existing evaluation systems face regarding authenticity, safety, and industry specificity. Finally, we focus on the practical challenges faced by industry agents, exploring their capability boundaries, developmental potential, and governance issues in various scenarios, while providing insights into future directions. By combining technological evolution with industry practices, this review aims to clarify the current state and offer a clear roadmap and theoretical foundation for understanding and building the next generation of industry agents.
>
---
#### [new 036] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EvolveR框架，解决大模型代理缺乏从自身经验中持续学习的问题。通过离线自蒸馏和在线交互的闭环生命周期，实现策略迭代优化，提升复杂多跳问答任务中的表现，推动自主进化的智能代理发展。**

- **链接: [http://arxiv.org/pdf/2510.16079v1](http://arxiv.org/pdf/2510.16079v1)**

> **作者:** Rong Wu; Xiaoman Wang; Jianbiao Mei; Pinlong Cai; Daocheng Fu; Cheng Yang; Licheng Wen; Xuemeng Yang; Yufan Shen; Yuxin Wang; Botian Shi
>
> **摘要:** Current Large Language Model (LLM) agents show strong performance in tool use, but lack the crucial capability to systematically learn from their own experiences. While existing frameworks mainly focus on mitigating external knowledge gaps, they fail to address a more fundamental limitation: the inability to iteratively refine problem-solving strategies. In this work, we introduce EvolveR, a framework designed to enable agent to self-improve through a complete, closed-loop experience lifecycle. This lifecycle comprises two key stages: (1) Offline Self-Distillation, where the agent's interaction trajectories are synthesized into a structured repository of abstract, reusable strategic principles; (2) Online Interaction, where the agent interacts with tasks and actively retrieves distilled principles to guide its decision-making, accumulating a diverse set of behavioral trajectories. This loop employs a policy reinforcement mechanism to iteratively update the agent based on its performance. We demonstrate the effectiveness of EvolveR on complex multi-hop question-answering benchmarks, where it achieves superior performance over strong agentic baselines. Our work presents a comprehensive blueprint for agents that learn not only from external data but also from the consequences of their own actions, paving the way for more autonomous and continuously improving systems. Code is available at https://github.com/Edaizi/EvolveR.
>
---
#### [new 037] Hallucination Benchmark for Speech Foundation Models
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文针对语音识别中的幻觉问题，提出首个基准框架SHALLOW，从词汇、语音、形态和语义四个维度系统评估幻觉现象，弥补传统错误率指标的不足，实现细粒度诊断与模型改进。**

- **链接: [http://arxiv.org/pdf/2510.16567v1](http://arxiv.org/pdf/2510.16567v1)**

> **作者:** Alkis Koudounas; Moreno La Quatra; Manuel Giollo; Sabato Marco Siniscalchi; Elena Baralis
>
> **备注:** Under Review
>
> **摘要:** Hallucinations in automatic speech recognition (ASR) systems refer to fluent and coherent transcriptions produced by neural ASR models that are completely unrelated to the underlying acoustic input (i.e., the speech signal). While similar to conventional decoding errors in potentially compromising the usability of transcriptions for downstream applications, hallucinations can be more detrimental due to their preservation of syntactically and semantically plausible structure. This apparent coherence can mislead subsequent processing stages and introduce serious risks, particularly in critical domains such as healthcare and law. Conventional evaluation metrics are primarily centered on error-based metrics and fail to distinguish between phonetic inaccuracies and hallucinations. Consequently, there is a critical need for new evaluation frameworks that can effectively identify and assess models with a heightened propensity for generating hallucinated content. To this end, we introduce SHALLOW, the first benchmark framework that systematically categorizes and quantifies hallucination phenomena in ASR along four complementary axes: lexical, phonetic, morphological, and semantic. We define targeted metrics within each category to produce interpretable profiles of model behavior. Through evaluation across various architectures and speech domains, we have found that SHALLOW metrics correlate strongly with word error rate (WER) when recognition quality is high (i.e., low WER). Still, this correlation weakens substantially as WER increases. SHALLOW, therefore, captures fine-grained error patterns that WER fails to distinguish under degraded and challenging conditions. Our framework supports specific diagnosis of model weaknesses and provides feedback for model improvement beyond what aggregate error rates can offer.
>
---
#### [new 038] EduAdapt: A Question Answer Benchmark Dataset for Evaluating Grade-Level Adaptability in LLMs
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文提出EduAdapt，首个评估大模型年级适应能力的基准数据集，包含48k标注问题。针对LLM在K-12教育中难以匹配学生认知水平的问题，构建跨九门科学科目的分级问答评测体系，并评估多种开源模型，推动教育AI的适龄化发展。**

- **链接: [http://arxiv.org/pdf/2510.17389v1](http://arxiv.org/pdf/2510.17389v1)**

> **作者:** Numaan Naeem; Abdellah El Mekki; Muhammad Abdul-Mageed
>
> **备注:** 28 pages, 2 figures, 14 tables, 50 listings, EMNLP 2025 Main
>
> **摘要:** Large language models (LLMs) are transforming education by answering questions, explaining complex concepts, and generating content across a wide range of subjects. Despite strong performance on academic benchmarks, they often fail to tailor responses to students' grade levels. This is a critical need in K-12 education, where age-appropriate vocabulary and explanation are essential for effective learning. Existing models frequently produce outputs that are too advanced or vague for younger learners, and there are no standardized benchmarks to evaluate their ability to adjust across cognitive and developmental stages. To address this gap, we introduce EduAdapt, a benchmark of nearly 48k grade-labeled QA pairs across nine science subjects, spanning Grades 1-12 and grouped into four grade levels. We evaluate a diverse set of open-source LLMs on EduAdapt and find that while larger models generally perform better, they still struggle with generating suitable responses for early-grade students (Grades 1-5). Our work presents the first dataset and evaluation framework for assessing grade-level adaptability in LLMs, aiming to foster more developmentally aligned educational AI systems through better training and prompting strategies. EduAdapt code and datasets are publicly available at https://github.com/NaumanNaeem/EduAdapt.
>
---
#### [new 039] Towards Low-Resource Alignment to Diverse Perspectives with Sparse Feedback
- **分类: cs.CL**

- **简介: 该论文研究语言模型的多元价值观对齐问题，旨在解决主流训练方法导致回应单一化的问题。作者提出基于稀疏反馈的低资源对齐方法，包括多元解码与模型引导，在少量标注数据下提升了模型在敏感任务中的表现与价值多样性。**

- **链接: [http://arxiv.org/pdf/2510.16257v1](http://arxiv.org/pdf/2510.16257v1)**

> **作者:** Chu Fei Luo; Samuel Dahan; Xiaodan Zhu
>
> **备注:** Findings of EMNLP 2025, 5 pages
>
> **摘要:** As language models have a greater impact on society, it is important to ensure they are aligned to a diverse range of perspectives and are able to reflect nuance in human values. However, the most popular training paradigms for modern language models often assume there is one optimal answer for every query, leading to generic responses and poor alignment. In this work, we aim to enhance pluralistic alignment of language models in a low-resource setting with two methods: pluralistic decoding and model steering. We empirically demonstrate that model steering offers consistent improvement over zero-shot and few-shot baselines with only 50 annotated samples. Our proposed methods decrease false positives in several high-stakes tasks such as hate speech detection and misinformation detection, and improves the distributional alignment to human values in GlobalOpinionQA. We hope our work highlights the importance of diversity and how language models can be adapted to consider nuanced perspectives.
>
---
#### [new 040] TrajSelector: Harnessing Latent Representations for Efficient and Effective Best-of-N in Large Reasoning Model
- **分类: cs.CL**

- **简介: 该论文针对大模型推理中Best-of-N选择效率低、成本高的问题，提出TrajSelector框架，利用采样器LLM的隐状态进行轨迹评分，通过轻量验证器实现高效准确的推理路径选择，在降低计算成本的同时提升性能。**

- **链接: [http://arxiv.org/pdf/2510.16449v1](http://arxiv.org/pdf/2510.16449v1)**

> **作者:** Bin Yu; Xinming Wang; Shijie Lian; Haotian Li; Changti Wu; Ruina Hu; Bailing Wang; Yuliang Wei; Kai Chen
>
> **备注:** 13 pages, 6 figures. Project website: https://zgca-ai4edu.github.io/TrajSelector
>
> **摘要:** Large language models (LLMs) have shown remarkable progress in complex reasoning tasks, largely enabled by test-time scaling (TTS) paradigms that allocate additional compute during inference. Among these, external TTS (particularly the Best-of-N selection paradigm) yields scalable performance improvements by selecting from multiple independently generated reasoning trajectories. However, this approach faces key limitations: (i) the high computational overhead of deploying process reward models, (ii) the underutilization of the LLM's intrinsic latent representations. We introduce TrajSelector, an efficient and effective Best-of-N framework that exploit the hidden states in the sampler LLM for process-level scoring. A lightweight verifier (with only 0.6B parameters) evaluates the quality of step-wise trajectory, and then aggregates these scores to identify the optimal reasoning trajectory. Our framework employs a fully data-driven, end-to-end training recipe that eliminates reliance on massive step-level annotations. Experiential results across five benchmarks demonstrate that TrajSelector delivers consistent performance gains. In Best-of-32 settings, it surpasses majority voting by 4.61% accuracy and outperforms existing process reward models by 4.31% to 12.21%, all while maintaining lower inference costs.
>
---
#### [new 041] Navigating the Alignment-Calibration Trade-off: A Pareto-Superior Frontier via Model Merging
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大模型对齐后的性能与校准度下降问题，提出通过合并对齐前后模型权重进行插值，有效缓解对齐税，在提升任务准确率的同时恢复校准度，实现更优的综合性能。**

- **链接: [http://arxiv.org/pdf/2510.17426v1](http://arxiv.org/pdf/2510.17426v1)**

> **作者:** Tiancheng Hu; Benjamin Minixhofer; Nigel Collier
>
> **摘要:** The "alignment tax" of post-training is typically framed as a drop in task accuracy. We show it also involves a severe loss of calibration, making models overconfident, less reliable, and model outputs less diverse. We show that this trade-off can be navigated effectively via a simple post-hoc intervention: interpolating between a model's weights before and after alignment. Crucially, this is not a strict trade-off. We find that the process consistently reveals Pareto-optimal interpolations - models that improve accuracy beyond both parents while substantially recovering the calibration lost during alignment. Our work demonstrates that simple model merging provides a computationally efficient method for mitigating the full scope of the alignment tax, yielding models that are more capable and more reliable.
>
---
#### [new 042] Enhancing Language Agent Strategic Reasoning through Self-Play in Adversarial Games
- **分类: cs.CL**

- **简介: 该论文研究语言智能体在对抗性游戏中的策略推理问题，提出SCO-PAL方法，通过自博弈进行策略优化。实验证明自博弈最有效，显著提升胜率，尤其在对抗GPT-4时表现突出。**

- **链接: [http://arxiv.org/pdf/2510.16761v1](http://arxiv.org/pdf/2510.16761v1)**

> **作者:** Yikai Zhang; Ye Rong; Siyu Yuan; Jiangjie Chen; Jian Xie; Yanghua Xiao
>
> **摘要:** Existing language agents often encounter difficulties in dynamic adversarial games due to poor strategic reasoning. To mitigate this limitation, a promising approach is to allow agents to learn from game interactions automatically, without relying on costly expert-labeled data. Unlike static environments where agents receive fixed feedback or rewards, selecting appropriate opponents in dynamic adversarial games can significantly impact learning performance. However, the discussion of opponents in adversarial environments remains an area under exploration. In this paper, we propose a Step-level poliCy Optimization method through Play-And-Learn, SCO-PAL. Leveraging SCO-PAL, we conduct a detailed analysis of opponent selection by setting opponents at different levels and find that self-play is the most effective way to improve strategic reasoning in such adversarial environments. Utilizing SCO-PAL with self-play, we increase the average win rate against four opponents by approximately 30% compared to baselines and achieve a 54.76% win rate against GPT-4 in six adversarial games.
>
---
#### [new 043] From Preferences to Prejudice: The Role of Alignment Tuning in Shaping Social Bias in Video Diffusion Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究视频扩散模型中对齐微调引发的社会偏见问题。提出VideoBiasEval框架，系统评估人类偏好数据、奖励模型到生成模型中偏见的演变，揭示对齐过程会增强并固化性别与种族偏见，强调需在全流程中进行偏见管控。**

- **链接: [http://arxiv.org/pdf/2510.17247v1](http://arxiv.org/pdf/2510.17247v1)**

> **作者:** Zefan Cai; Haoyi Qiu; Haozhe Zhao; Ke Wan; Jiachen Li; Jiuxiang Gu; Wen Xiao; Nanyun Peng; Junjie Hu
>
> **摘要:** Recent advances in video diffusion models have significantly enhanced text-to-video generation, particularly through alignment tuning using reward models trained on human preferences. While these methods improve visual quality, they can unintentionally encode and amplify social biases. To systematically trace how such biases evolve throughout the alignment pipeline, we introduce VideoBiasEval, a comprehensive diagnostic framework for evaluating social representation in video generation. Grounded in established social bias taxonomies, VideoBiasEval employs an event-based prompting strategy to disentangle semantic content (actions and contexts) from actor attributes (gender and ethnicity). It further introduces multi-granular metrics to evaluate (1) overall ethnicity bias, (2) gender bias conditioned on ethnicity, (3) distributional shifts in social attributes across model variants, and (4) the temporal persistence of bias within videos. Using this framework, we conduct the first end-to-end analysis connecting biases in human preference datasets, their amplification in reward models, and their propagation through alignment-tuned video diffusion models. Our results reveal that alignment tuning not only strengthens representational biases but also makes them temporally stable, producing smoother yet more stereotyped portrayals. These findings highlight the need for bias-aware evaluation and mitigation throughout the alignment process to ensure fair and socially responsible video generation.
>
---
#### [new 044] Does Visual Grounding Enhance the Understanding of Embodied Knowledge in Large Language Models?
- **分类: cs.CL**

- **简介: 该论文研究视觉 grounding 是否提升大语言模型对具身知识的理解。作者构建了一个基于感知理论的多感官基准，评估30种模型在跨模态理解上的表现，发现视觉语言模型未优于纯文本模型，且在视觉维度表现更差，揭示当前模型对空间感知和词频干扰的局限。**

- **链接: [http://arxiv.org/pdf/2510.16924v1](http://arxiv.org/pdf/2510.16924v1)**

> **作者:** Zhihui Yang; Yupei Wang; Kaijie Mo; Zhe Zhao; Renfen Hu
>
> **备注:** Accepted to EMNLP 2025 (Findings). This version corrects a redundant sentence in the Results section that appeared in the camera-ready version
>
> **摘要:** Despite significant progress in multimodal language models (LMs), it remains unclear whether visual grounding enhances their understanding of embodied knowledge compared to text-only models. To address this question, we propose a novel embodied knowledge understanding benchmark based on the perceptual theory from psychology, encompassing visual, auditory, tactile, gustatory, olfactory external senses, and interoception. The benchmark assesses the models' perceptual abilities across different sensory modalities through vector comparison and question-answering tasks with over 1,700 questions. By comparing 30 state-of-the-art LMs, we surprisingly find that vision-language models (VLMs) do not outperform text-only models in either task. Moreover, the models perform significantly worse in the visual dimension compared to other sensory dimensions. Further analysis reveals that the vector representations are easily influenced by word form and frequency, and the models struggle to answer questions involving spatial perception and reasoning. Our findings underscore the need for more effective integration of embodied knowledge in LMs to enhance their understanding of the physical world.
>
---
#### [new 045] Evaluating Medical LLMs by Levels of Autonomy: A Survey Moving from Benchmarks to Applications
- **分类: cs.CL**

- **简介: 该论文提出基于自主性等级（L0-L3）评估医学大模型，旨在解决benchmark表现与临床应用脱节的问题。通过分级框架对齐评估指标与风险，推动从分数导向转向风险感知的可信临床验证。**

- **链接: [http://arxiv.org/pdf/2510.17764v1](http://arxiv.org/pdf/2510.17764v1)**

> **作者:** Xiao Ye; Jacob Dineen; Zhaonan Li; Zhikun Xu; Weiyu Chen; Shijie Lu; Yuxi Huang; Ming Shen; Phu Tran; Ji-Eun Irene Yum; Muhammad Ali Khan; Muhammad Umar Afzal; Irbaz Bin Riaz; Ben Zhou
>
> **摘要:** Medical Large language models achieve strong scores on standard benchmarks; however, the transfer of those results to safe and reliable performance in clinical workflows remains a challenge. This survey reframes evaluation through a levels-of-autonomy lens (L0-L3), spanning informational tools, information transformation and aggregation, decision support, and supervised agents. We align existing benchmarks and metrics with the actions permitted at each level and their associated risks, making the evaluation targets explicit. This motivates a level-conditioned blueprint for selecting metrics, assembling evidence, and reporting claims, alongside directions that link evaluation to oversight. By centering autonomy, the survey moves the field beyond score-based claims toward credible, risk-aware evidence for real clinical use.
>
---
#### [new 046] Foundational Automatic Evaluators: Scaling Multi-Task Generative Evaluator Training for Reasoning-Centric Domains
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦自动评估器训练，旨在提升推理任务中评估的可扩展性。作者构建了250万样本的多任务数据集，提出迭代拒绝采样微调方法，训练出高性能的FARE系列评估模型，在多种推理任务中超越现有大模型。**

- **链接: [http://arxiv.org/pdf/2510.17793v1](http://arxiv.org/pdf/2510.17793v1)**

> **作者:** Austin Xu; Xuan-Phi Nguyen; Yilun Zhou; Chien-Sheng Wu; Caiming Xiong; Shafiq Joty
>
> **备注:** 29 pages, 9 tables, 6 figures
>
> **摘要:** Finetuning specialized generative evaluators has emerged as a popular paradigm to meet the increasing demand for scalable evaluation during both training and test-time. However, recent work has largely focused on applying new methodology, such as reinforcement learning (RL), to training evaluators, shying away from large-scale, data-driven development. In this work, we focus on data scaling, curating a set of 2.5M samples spanning five unique evaluation tasks (pairwise, step-level, reference-free and reference-based verification, and single rating) and multiple domains focused on reasoning evaluation. With our data, we train Foundational Automatic Reasoning Evaluators (FARE), a family of 8B and 20B (with 3.6B active) parameter evaluators, with a simple iterative rejection-sampling supervised finetuning (SFT) approach. FARE-8B challenges larger specialized RL-trained evaluators and FARE-20B sets the new standard for open-source evaluators, surpassing specialized 70B+ evaluators. Beyond static benchmarks, we evaluate FARE in real-world tasks: As inference-time rerankers, FARE-20B achieves near-oracle performance on MATH. As verifiers in RL training, FARE improves the downstream RL-trained model performance by up to 14.1% vs. string-matching verifiers. When initialized from FARE, a continually-finetuned FARE-Code outperforms gpt-oss-20B by 65% on evaluating test-case quality.
>
---
#### [new 047] Utilising Large Language Models for Generating Effective Counter Arguments to Anti-Vaccine Tweets
- **分类: cs.CL**

- **简介: 该论文研究利用大语言模型生成针对反疫苗推文的有效反驳。任务是自动反驳疫苗 misinformation，提出结合分类标签与结构化微调的方法，提升生成反驳的准确性和适应性，通过多种评估验证其有效性。**

- **链接: [http://arxiv.org/pdf/2510.16359v1](http://arxiv.org/pdf/2510.16359v1)**

> **作者:** Utsav Dhanuka; Soham Poddar; Saptarshi Ghosh
>
> **备注:** 14 pages, 1 figure, work done as a part of B.Tech project at IIT Kharagpur
>
> **摘要:** In an era where public health is increasingly influenced by information shared on social media, combatting vaccine skepticism and misinformation has become a critical societal goal. Misleading narratives around vaccination have spread widely, creating barriers to achieving high immunisation rates and undermining trust in health recommendations. While efforts to detect misinformation have made significant progress, the generation of real time counter-arguments tailored to debunk such claims remains an insufficiently explored area. In this work, we explore the capabilities of LLMs to generate sound counter-argument rebuttals to vaccine misinformation. Building on prior research in misinformation debunking, we experiment with various prompting strategies and fine-tuning approaches to optimise counter-argument generation. Additionally, we train classifiers to categorise anti-vaccine tweets into multi-labeled categories such as concerns about vaccine efficacy, side effects, and political influences allowing for more context aware rebuttals. Our evaluation, conducted through human judgment, LLM based assessments, and automatic metrics, reveals strong alignment across these methods. Our findings demonstrate that integrating label descriptions and structured fine-tuning enhances counter-argument effectiveness, offering a promising approach for mitigating vaccine misinformation at scale.
>
---
#### [new 048] Language over Content: Tracing Cultural Understanding in Multilingual Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究多语言大模型的文化理解机制，旨在区分语言与文化对模型内部表征的影响。通过分析不同语言和国家条件下回答问题时的激活路径重叠，发现语言比文化对内部表征影响更强，且语言相似性不保证表征一致。**

- **链接: [http://arxiv.org/pdf/2510.16565v1](http://arxiv.org/pdf/2510.16565v1)**

> **作者:** Seungho Cho; Changgeon Ko; Eui Jun Hwang; Junmyeong Lee; Huije Lee; Jong C. Park
>
> **备注:** Accepted to CIKM 2025 Workshop on Human Centric AI
>
> **摘要:** Large language models (LLMs) are increasingly used across diverse cultural contexts, making accurate cultural understanding essential. Prior evaluations have mostly focused on output-level performance, obscuring the factors that drive differences in responses, while studies using circuit analysis have covered few languages and rarely focused on culture. In this work, we trace LLMs' internal cultural understanding mechanisms by measuring activation path overlaps when answering semantically equivalent questions under two conditions: varying the target country while fixing the question language, and varying the question language while fixing the country. We also use same-language country pairs to disentangle language from cultural aspects. Results show that internal paths overlap more for same-language, cross-country questions than for cross-language, same-country questions, indicating strong language-specific patterns. Notably, the South Korea-North Korea pair exhibits low overlap and high variability, showing that linguistic similarity does not guarantee aligned internal representation.
>
---
#### [new 049] LawChain: Modeling Legal Reasoning Chains for Chinese Tort Case Analysis
- **分类: cs.CL**

- **简介: 该论文提出LawChain框架，用于建模中国侵权民事案件中的法律推理链。针对现有法律推理研究偏重刑事案件、缺乏细粒度分析的问题，构建了包含三模块的推理框架及评测基准LawChain$_{eval}$，并设计基于提示和后训练的基线方法，提升大模型在侵权法律推理及其他相关任务上的表现。**

- **链接: [http://arxiv.org/pdf/2510.17602v1](http://arxiv.org/pdf/2510.17602v1)**

> **作者:** Huiyuan Xie; Chenyang Li; Huining Zhu; Chubin Zhang; Yuxiao Ye; Zhenghao Liu; Zhiyuan Liu
>
> **摘要:** Legal reasoning is a fundamental component of legal analysis and decision-making. Existing computational approaches to legal reasoning predominantly rely on generic reasoning frameworks such as syllogism and IRAC, which do not comprehensively examine the nuanced processes that underpin legal reasoning. Moreover, current research has largely focused on criminal cases, with insufficient modeling for civil cases. In this work, we present a novel framework for explicitly modeling legal reasoning in the analysis of Chinese tort-related civil cases. We first operationalize the legal reasoning processes used in tort analysis into the LawChain framework. LawChain is a three-module reasoning framework, with each module consisting of multiple finer-grained sub-steps. Informed by the LawChain framework, we introduce the task of tort legal reasoning and construct an evaluation benchmark, LawChain$_{eval}$, to systematically assess the critical steps within analytical reasoning chains for tort analysis. Leveraging this benchmark, we evaluate state-of-the-art large language models for their legal reasoning ability in civil tort contexts. Our results indicate that current models still fall short in accurately handling crucial elements of tort legal reasoning. Furthermore, we introduce several baseline approaches that explicitly incorporate LawChain-style reasoning through prompting or post-training. We conduct further experiments on additional legal analysis tasks, such as Legal Named-Entity Recognition and Criminal Damages Calculation, to verify the generalizability of these baselines. The proposed baseline approaches achieve significant improvements in tort-related legal reasoning and generalize well to related legal analysis tasks, thus demonstrating the value of explicitly modeling legal reasoning chains to enhance the reasoning capabilities of language models.
>
---
#### [new 050] ATA: A Neuro-Symbolic Approach to Implement Autonomous and Trustworthy Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种神经符号方法ATA，旨在构建自主可信的智能体。针对大模型的幻觉、不稳定等问题，将任务分为离线知识注入与在线推理两阶段，通过形式化知识库和符号推理提升可验证性、确定性与安全性。**

- **链接: [http://arxiv.org/pdf/2510.16381v1](http://arxiv.org/pdf/2510.16381v1)**

> **作者:** David Peer; Sebastian Stabinger
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities, yet their deployment in high-stakes domains is hindered by inherent limitations in trustworthiness, including hallucinations, instability, and a lack of transparency. To address these challenges, we introduce a generic neuro-symbolic approach, which we call Autonomous Trustworthy Agents (ATA). The core of our approach lies in decoupling tasks into two distinct phases: Offline knowledge ingestion and online task processing. During knowledge ingestion, an LLM translates an informal problem specification into a formal, symbolic knowledge base. This formal representation is crucial as it can be verified and refined by human experts, ensuring its correctness and alignment with domain requirements. In the subsequent task processing phase, each incoming input is encoded into the same formal language. A symbolic decision engine then utilizes this encoded input in conjunction with the formal knowledge base to derive a reliable result. Through an extensive evaluation on a complex reasoning task, we demonstrate that a concrete implementation of ATA is competitive with state-of-the-art end-to-end reasoning models in a fully automated setup while maintaining trustworthiness. Crucially, with a human-verified and corrected knowledge base, our approach significantly outperforms even larger models, while exhibiting perfect determinism, enhanced stability against input perturbations, and inherent immunity to prompt injection attacks. By generating decisions grounded in symbolic reasoning, ATA offers a practical and controllable architecture for building the next generation of transparent, auditable, and reliable autonomous agents.
>
---
#### [new 051] HGAdapter: Hypergraph-based Adapters in Language Models for Code Summarization and Clone Detection
- **分类: cs.CL; cs.AI; cs.LG; cs.SE**

- **简介: 该论文针对代码摘要与克隆检测任务，提出HGAdapter方法，通过超图适配器建模代码中的高阶关联（如语法树、词法、行级关系），增强预训练语言模型对代码结构的理解，提升其在多语言代码理解任务中的性能。**

- **链接: [http://arxiv.org/pdf/2510.17591v1](http://arxiv.org/pdf/2510.17591v1)**

> **作者:** Guang Yang; Yujie Zhu
>
> **备注:** Accepted by the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025) as a findings long paper
>
> **摘要:** Pre-trained language models (PLMs) are increasingly being applied to code-related tasks. Although PLMs have achieved good results, they do not take into account potential high-order data correlations within the code. We propose three types of high-order correlations in code tokens, i.e. abstract syntax tree family correlation, lexical correlation, and line correlation. We design a tokens and hyperedges generator to capture these high-order data correlations. We improve the architecture of hypergraph neural networks and combine it with adapter tuning to propose a novel hypergraph-based adapter (HGAdapter) to fine-tune PLMs. HGAdapter can encode high-order data correlations and is allowed to be inserted into various PLMs to enhance performance. Experiments were conducted on several public datasets, including six languages of code summarization and code clone detection tasks. Our methods improved the performance of PLMs in datasets to varying degrees. Experimental results validate the introduction of high-order data correlations that contribute to improved effectiveness.
>
---
#### [new 052] DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于AI生成文本检测任务，旨在解决人类与AI协作文本的复杂检测问题。提出DETree方法，构建层次化树结构建模不同协作模式的关系，并设计新损失函数优化表示学习，结合自建数据集RealBench提升检测性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.17489v1](http://arxiv.org/pdf/2510.17489v1)**

> **作者:** Yongxin He; Shan Zhang; Yixuan Cao; Lei Ma; Ping Luo
>
> **备注:** To appear in NeurIPS 2025
>
> **摘要:** Detecting AI-involved text is essential for combating misinformation, plagiarism, and academic misconduct. However, AI text generation includes diverse collaborative processes (AI-written text edited by humans, human-written text edited by AI, and AI-generated text refined by other AI), where various or even new LLMs could be involved. Texts generated through these varied processes exhibit complex characteristics, presenting significant challenges for detection. Current methods model these processes rather crudely, primarily employing binary classification (purely human vs. AI-involved) or multi-classification (treating human-AI collaboration as a new class). We observe that representations of texts generated through different processes exhibit inherent clustering relationships. Therefore, we propose DETree, a novel approach that models the relationships among different processes as a Hierarchical Affinity Tree structure, and introduces a specialized loss function that aligns text representations with this tree. To facilitate this learning, we developed RealBench, a comprehensive benchmark dataset that automatically incorporates a wide spectrum of hybrid texts produced through various human-AI collaboration processes. Our method improves performance in hybrid text detection tasks and significantly enhances robustness and generalization in out-of-distribution scenarios, particularly in few-shot learning conditions, further demonstrating the promise of training-based approaches in OOD settings. Our code and dataset are available at https://github.com/heyongxin233/DETree.
>
---
#### [new 053] Rethinking On-policy Optimization for Query Augmentation
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究查询增强在信息检索中的应用，比较提示与强化学习方法，发现简单提示常媲美复杂微调。为此提出新方法OPQE，结合两者优势，通过策略生成伪文档提升检索效果。**

- **链接: [http://arxiv.org/pdf/2510.17139v1](http://arxiv.org/pdf/2510.17139v1)**

> **作者:** Zhichao Xu; Shengyao Zhuang; Xueguang Ma; Bingsen Chen; Yijun Tian; Fengran Mo; Jie Cao; Vivek Srikumar
>
> **摘要:** Recent advances in large language models (LLMs) have led to a surge of interest in query augmentation for information retrieval (IR). Two main approaches have emerged. The first prompts LLMs to generate answers or pseudo-documents that serve as new queries, relying purely on the model's parametric knowledge or contextual information. The second applies reinforcement learning (RL) to fine-tune LLMs for query rewriting, directly optimizing retrieval metrics. While having respective advantages and limitations, the two approaches have not been compared under consistent experimental conditions. In this work, we present the first systematic comparison of prompting-based and RL-based query augmentation across diverse benchmarks, including evidence-seeking, ad hoc, and tool retrieval. Our key finding is that simple, training-free query augmentation often performs on par with, or even surpasses, more expensive RL-based counterparts, especially when using powerful LLMs. Motivated by this discovery, we introduce a novel hybrid method, On-policy Pseudo-document Query Expansion (OPQE), which, instead of rewriting a query, the LLM policy learns to generate a pseudo-document that maximizes retrieval performance, thus merging the flexibility and generative structure of prompting with the targeted optimization of RL. We show OPQE outperforms both standalone prompting and RL-based rewriting, demonstrating that a synergistic approach yields the best results. Our implementation is made available to facilitate reproducibility.
>
---
#### [new 054] DVAGen: Dynamic Vocabulary Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对语言模型词汇固定导致泛化能力差的问题，提出DVAGen框架，支持动态词汇增强。工作包括构建开源、模块化框架，集成主流大模型，支持批量推理与实时可视化，提升动态词汇模型的训练、评估与应用效率。**

- **链接: [http://arxiv.org/pdf/2510.17115v1](http://arxiv.org/pdf/2510.17115v1)**

> **作者:** Wei Du; Nuowei Liu; Jie Wang; Jiahao Kuang; Tao Ji; Xiaoling Wang; Yuanbin Wu
>
> **摘要:** Language models trained with a fixed vocabulary struggle to generalize to novel or out-of-vocabulary words, limiting their flexibility in handling diverse token combinations. Existing dynamic vocabulary approaches attempt to address this limitation but face challenges such as fragmented codebases, lack of support for modern LLMs, and limited inference scalability. To overcome these issues, we introduce DVAGen, a fully open-source, unified framework designed for training, evaluation, and visualization of dynamic vocabulary-augmented language models. Our framework modularizes the pipeline for ease of customization, integrates seamlessly with open-source LLMs, and is the first to provide both CLI and WebUI tools for real-time result inspection. We validate the effectiveness of dynamic vocabulary methods on modern LLMs and demonstrate support for batch inference, significantly improving inference throughput.
>
---
#### [new 055] MOSAIC: Masked Objective with Selective Adaptation for In-domain Contrastive Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究领域适应任务，旨在将通用句向量模型有效适配到特定领域。提出MOSAIC框架，结合掩码语言建模与对比学习，通过多阶段联合优化，提升领域内语义匹配性能，尤其在低资源场景下显著优于基线方法。**

- **链接: [http://arxiv.org/pdf/2510.16797v1](http://arxiv.org/pdf/2510.16797v1)**

> **作者:** Vera Pavlova; Mohammed Makhlouf
>
> **摘要:** We introduce MOSAIC (Masked Objective with Selective Adaptation for In-domain Contrastive learning), a multi-stage framework for domain adaptation of sentence embedding models that incorporates joint domain-specific masked supervision. Our approach addresses the challenges of adapting large-scale general-domain sentence embedding models to specialized domains. By jointly optimizing masked language modeling (MLM) and contrastive objectives within a unified training pipeline, our method enables effective learning of domain-relevant representations while preserving the robust semantic discrimination properties of the original model. We empirically validate our approach on both high-resource and low-resource domains, achieving improvements up to 13.4% in NDCG@10 (Normalized Discounted Cumulative Gain) over strong general-domain baselines. Comprehensive ablation studies further demonstrate the effectiveness of each component, highlighting the importance of balanced joint supervision and staged adaptation.
>
---
#### [new 056] Fusion-Augmented Large Language Models: Boosting Diagnostic Trustworthiness via Model Consensus
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究胸部X光诊断任务，旨在提升AI模型的可信度。通过融合ChatGPT与Claude的输出，利用共识机制和多模态输入，在CheXpert数据集上显著提高诊断准确率，降低误差，增强临床实用性。**

- **链接: [http://arxiv.org/pdf/2510.16057v1](http://arxiv.org/pdf/2510.16057v1)**

> **作者:** Md Kamrul Siam; Md Jobair Hossain Faruk; Jerry Q. Cheng; Huanying Gu
>
> **备注:** 7 pages (Accepted to IEEE BHI 2025)
>
> **摘要:** This study presents a novel multi-model fusion framework leveraging two state-of-the-art large language models (LLMs), ChatGPT and Claude, to enhance the reliability of chest X-ray interpretation on the CheXpert dataset. From the full CheXpert corpus of 224,316 chest radiographs, we randomly selected 234 radiologist-annotated studies to evaluate unimodal performance using image-only prompts. In this setting, ChatGPT and Claude achieved diagnostic accuracies of 62.8% and 76.9%, respectively. A similarity-based consensus approach, using a 95% output similarity threshold, improved accuracy to 77.6%. To assess the impact of multimodal inputs, we then generated synthetic clinical notes following the MIMIC-CXR template and evaluated a separate subset of 50 randomly selected cases paired with both images and synthetic text. On this multimodal cohort, performance improved to 84% for ChatGPT and 76% for Claude, while consensus accuracy reached 91.3%. Across both experimental conditions, agreement-based fusion consistently outperformed individual models. These findings highlight the utility of integrating complementary modalities and using output-level consensus to improve the trustworthiness and clinical utility of AI-assisted radiological diagnosis, offering a practical path to reduce diagnostic errors with minimal computational overhead.
>
---
#### [new 057] Parameter-Efficient Fine-Tuning for Low-Resource Languages: A Comparative Study of LLMs for Bengali Hate Speech Detection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究低资源语言Bengali的仇恨言论检测，提出使用参数高效微调（PEFT）方法LoRA和QLoRA，在少量标注数据上微调大模型。实验表明该方法在单个消费级GPU上高效训练，Llama-3.2-3B表现最佳，F1达92.23%。**

- **链接: [http://arxiv.org/pdf/2510.16985v1](http://arxiv.org/pdf/2510.16985v1)**

> **作者:** Akif Islam; Mohd Ruhul Ameen
>
> **备注:** Accepted to IEEE COMPAS 2025. 6 pages, 3 figures, 6 tables
>
> **摘要:** Bengali social media platforms have witnessed a sharp increase in hate speech, disproportionately affecting women and adolescents. While datasets such as BD-SHS provide a basis for structured evaluation, most prior approaches rely on either computationally costly full-model fine-tuning or proprietary APIs. This paper presents the first application of Parameter-Efficient Fine-Tuning (PEFT) for Bengali hate speech detection using LoRA and QLoRA. Three instruction-tuned large language models - Gemma-3-4B, Llama-3.2-3B, and Mistral-7B - were fine-tuned on the BD-SHS dataset of 50,281 annotated comments. Each model was adapted by training fewer than 1% of its parameters, enabling experiments on a single consumer-grade GPU. The results show that Llama-3.2-3B achieved the highest F1-score of 92.23%, followed by Mistral-7B at 88.94% and Gemma-3-4B at 80.25%. These findings establish PEFT as a practical and replicable strategy for Bengali and related low-resource languages.
>
---
#### [new 058] Executable Knowledge Graphs for Replicating AI Research
- **分类: cs.CL; cs.AI; cs.LG; cs.MA; cs.SE**

- **简介: 该论文针对AI研究复现中代码生成困难的问题，提出可执行知识图谱xKG，通过结构化整合文献中的技术细节与代码片段，增强LLM代理的多粒度知识检索与复用能力，在多个框架下显著提升复现性能。**

- **链接: [http://arxiv.org/pdf/2510.17795v1](http://arxiv.org/pdf/2510.17795v1)**

> **作者:** Yujie Luo; Zhuoyun Yu; Xuehai Wang; Yuqi Zhu; Ningyu Zhang; Lanning Wei; Lun Du; Da Zheng; Huajun Chen
>
> **备注:** Work in progress
>
> **摘要:** Replicating AI research is a crucial yet challenging task for large language model (LLM) agents. Existing approaches often struggle to generate executable code, primarily due to insufficient background knowledge and the limitations of retrieval-augmented generation (RAG) methods, which fail to capture latent technical details hidden in referenced papers. Furthermore, previous approaches tend to overlook valuable implementation-level code signals and lack structured knowledge representations that support multi-granular retrieval and reuse. To overcome these challenges, we propose Executable Knowledge Graphs (xKG), a modular and pluggable knowledge base that automatically integrates technical insights, code snippets, and domain-specific knowledge extracted from scientific literature. When integrated into three agent frameworks with two different LLMs, xKG shows substantial performance gains (10.9% with o3-mini) on PaperBench, demonstrating its effectiveness as a general and extensible solution for automated AI research replication. Code will released at https://github.com/zjunlp/xKG.
>
---
#### [new 059] TaxoAlign: Scholarly Taxonomy Generation Using Language Models
- **分类: cs.CL**

- **简介: 该论文研究学术 taxonomy 自动生成任务，旨在缩小自动生成与专家手工构建 taxonomies 之间的差距。作者提出 TaxoAlign 方法和 CS-TaxoBench 基准，并设计评估框架，通过结构对齐与语义连贯性衡量生成质量。**

- **链接: [http://arxiv.org/pdf/2510.17263v1](http://arxiv.org/pdf/2510.17263v1)**

> **作者:** Avishek Lahiri; Yufang Hou; Debarshi Kumar Sanyal
>
> **备注:** This paper has been accepted at the EMNLP 2025 Main Conference
>
> **摘要:** Taxonomies play a crucial role in helping researchers structure and navigate knowledge in a hierarchical manner. They also form an important part in the creation of comprehensive literature surveys. The existing approaches to automatic survey generation do not compare the structure of the generated surveys with those written by human experts. To address this gap, we present our own method for automated taxonomy creation that can bridge the gap between human-generated and automatically-created taxonomies. For this purpose, we create the CS-TaxoBench benchmark which consists of 460 taxonomies that have been extracted from human-written survey papers. We also include an additional test set of 80 taxonomies curated from conference survey papers. We propose TaxoAlign, a three-phase topic-based instruction-guided method for scholarly taxonomy generation. Additionally, we propose a stringent automated evaluation framework that measures the structural alignment and semantic coherence of automatically generated taxonomies in comparison to those created by human experts. We evaluate our method and various baselines on CS-TaxoBench, using both automated evaluation metrics and human evaluation studies. The results show that TaxoAlign consistently surpasses the baselines on nearly all metrics. The code and data can be found at https://github.com/AvishekLahiri/TaxoAlign.
>
---
#### [new 060] Quantum NLP models on Natural Language Inference
- **分类: cs.CL; cs.AI; 81P68 (Primary), 68T50, 68T07 (Secondary); I.2.7; F.1.2**

- **简介: 该论文研究量子自然语言处理在自然语言推理任务中的应用，旨在解决低资源下模型参数效率低的问题。作者基于DisCoCat框架构建量子电路模型，提出信息增益每参数指标，发现量子模型在更少参数下性能媲美经典模型，并设计聚类架构提升泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.15972v1](http://arxiv.org/pdf/2510.15972v1)**

> **作者:** Ling Sun; Peter Sullivan; Michael Martin; Yun Zhou
>
> **备注:** Accepted, presented, and to appear in the Proceedings of the Quantum AI and NLP 2025 Conference
>
> **摘要:** Quantum natural language processing (QNLP) offers a novel approach to semantic modeling by embedding compositional structure directly into quantum circuits. This paper investigates the application of QNLP models to the task of Natural Language Inference (NLI), comparing quantum, hybrid, and classical transformer-based models under a constrained few-shot setting. Using the lambeq library and the DisCoCat framework, we construct parameterized quantum circuits for sentence pairs and train them for both semantic relatedness and inference classification. To assess efficiency, we introduce a novel information-theoretic metric, Information Gain per Parameter (IGPP), which quantifies learning dynamics independent of model size. Our results demonstrate that quantum models achieve performance comparable to classical baselines while operating with dramatically fewer parameters. The Quantum-based models outperform randomly initialized transformers in inference and achieve lower test error on relatedness tasks. Moreover, quantum models exhibit significantly higher per-parameter learning efficiency (up to five orders of magnitude more than classical counterparts), highlighting the promise of QNLP in low-resource, structure-sensitive settings. To address circuit-level isolation and promote parameter sharing, we also propose a novel cluster-based architecture that improves generalization by tying gate parameters to learned word clusters rather than individual tokens.
>
---
#### [new 061] AI-Generated Text Detection in Low-Resource Languages: A Case Study on Urdu
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属文本检测任务，旨在解决乌尔都语中AI生成文本难检测的问题。作者构建了乌尔都语数据集，分析语言特征，并微调多语言模型，提出专用于低资源语言的检测框架，mDeBERTa-v3-base表现最优。**

- **链接: [http://arxiv.org/pdf/2510.16573v1](http://arxiv.org/pdf/2510.16573v1)**

> **作者:** Muhammad Ammar; Hadiya Murad Hadi; Usman Majeed Butt
>
> **摘要:** Large Language Models (LLMs) are now capable of generating text that closely resembles human writing, making them powerful tools for content creation, but this growing ability has also made it harder to tell whether a piece of text was written by a human or by a machine. This challenge becomes even more serious for languages like Urdu, where there are very few tools available to detect AI-generated text. To address this gap, we propose a novel AI-generated text detection framework tailored for the Urdu language. A balanced dataset comprising 1,800 humans authored, and 1,800 AI generated texts, sourced from models such as Gemini, GPT-4o-mini, and Kimi AI was developed. Detailed linguistic and statistical analysis was conducted, focusing on features such as character and word counts, vocabulary richness (Type Token Ratio), and N-gram patterns, with significance evaluated through t-tests and MannWhitney U tests. Three state-of-the-art multilingual transformer models such as mdeberta-v3-base, distilbert-base-multilingualcased, and xlm-roberta-base were fine-tuned on this dataset. The mDeBERTa-v3-base achieved the highest performance, with an F1-score 91.29 and accuracy of 91.26% on the test set. This research advances efforts in contesting misinformation and academic misconduct in Urdu-speaking communities and contributes to the broader development of NLP tools for low resource languages.
>
---
#### [new 062] Leveraging Group Relative Policy Optimization to Advance Large Language Models in Traditional Chinese Medicine
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦中医药领域的大语言模型对齐问题，提出采用组相对策略优化（GRPO）方法，基于TCM-Ladder数据集训练Ladder-base模型，提升推理能力与事实一致性，推动可信赖的中医AI发展。**

- **链接: [http://arxiv.org/pdf/2510.17402v1](http://arxiv.org/pdf/2510.17402v1)**

> **作者:** Jiacheng Xie; Shuai Zeng; Yang Yu; Xiaoting Tang; Guanghui An; Dong Xu
>
> **摘要:** Traditional Chinese Medicine (TCM) presents a rich and structurally unique knowledge system that challenges conventional applications of large language models (LLMs). Although previous TCM-specific LLMs have shown progress through supervised fine-tuning, they often face limitations in alignment, data quality, and evaluation consistency. In this study, we introduce Ladder-base, the first TCM-focused LLM trained with Group Relative Policy Optimization (GRPO), a reinforcement learning method that improves reasoning and factual consistency by optimizing response selection based on intra-group comparisons. Ladder-base is built upon the Qwen2.5-7B-Instruct foundation model and trained exclusively on the textual subset of the TCM-Ladder benchmark, using 80 percent of the data for training and the remaining 20 percent split evenly between validation and test sets. Through standardized evaluation, Ladder-base demonstrates superior performance across multiple reasoning metrics when compared to both state-of-the-art general-purpose LLMs such as GPT-4, Gemini 2.5, Claude 3, and Qwen3 and domain-specific TCM models including BenTsao, HuatuoGPT2, and Zhongjing. These findings suggest that GRPO provides an effective and efficient strategy for aligning LLMs with expert-level reasoning in traditional medical domains and supports the development of trustworthy and clinically grounded TCM artificial intelligence systems.
>
---
#### [new 063] RAVEN: Robust Advertisement Video Violation Temporal Grounding via Reinforcement Reasoning
- **分类: cs.CL**

- **简介: 该论文针对广告视频违规检测中时序定位不准、标注噪声和泛化性差的问题，提出RAVEN框架，结合课程强化学习与多模态大模型，通过渐进训练和分组相对策略优化，实现精准时序定位与类别预测，提升在线广告系统的检测性能。**

- **链接: [http://arxiv.org/pdf/2510.16455v1](http://arxiv.org/pdf/2510.16455v1)**

> **作者:** Deyi Ji; Yuekui Yang; Haiyang Wu; Shaoping Ma; Tianrun Chen; Lanyun Zhu
>
> **备注:** ACL 2025 (Oral, Industry Track)
>
> **摘要:** Advertisement (Ad) video violation detection is critical for ensuring platform compliance, but existing methods struggle with precise temporal grounding, noisy annotations, and limited generalization. We propose RAVEN, a novel framework that integrates curriculum reinforcement learning with multimodal large language models (MLLMs) to enhance reasoning and cognitive capabilities for violation detection. RAVEN employs a progressive training strategy, combining precisely and coarsely annotated data, and leverages Group Relative Policy Optimization (GRPO) to develop emergent reasoning abilities without explicit reasoning annotations. Multiple hierarchical sophisticated reward mechanism ensures precise temporal grounding and consistent category prediction. Experiments on industrial datasets and public benchmarks show that RAVEN achieves superior performances in violation category accuracy and temporal interval localization. We also design a pipeline to deploy the RAVEN on the online Ad services, and online A/B testing further validates its practical applicability, with significant improvements in precision and recall. RAVEN also demonstrates strong generalization, mitigating the catastrophic forgetting issue associated with supervised fine-tuning.
>
---
#### [new 064] Neuronal Group Communication for Efficient Neural representation
- **分类: cs.CL; cs.AI; cs.NE**

- **简介: 该论文提出神经元组通信（NGC）框架，旨在构建高效、模块化、可解释的神经网络。它将网络视为神经元组间的动态交互系统，通过低秩分组通信减少冗余参数，并引入稳定性度量分析推理能力的涌现机制，在压缩下提升大模型推理性能。**

- **链接: [http://arxiv.org/pdf/2510.16851v1](http://arxiv.org/pdf/2510.16851v1)**

> **作者:** Zhengqi Pei; Qingming Huang; Shuhui Wang
>
> **备注:** 28 pages, 2 figures
>
> **摘要:** The ever-increasing scale of modern neural networks has brought unprecedented performance alongside daunting challenges in efficiency and interpretability. This paper addresses the core question of how to build large neural systems that learn efficient, modular, and interpretable representations. We propose Neuronal Group Communication (NGC), a theory-driven framework that reimagines a neural network as a dynamical system of interacting neuronal groups rather than a monolithic collection of neural weights. Instead of treating each weight as an independent trainable parameter, NGC treats weights as transient interactions between embedding-like neuronal states, with neural computation unfolding through iterative communication among groups of neurons. This low-rank, modular representation yields compact models: groups of neurons exchange low-dimensional signals, enabling intra-group specialization and inter-group information sharing while dramatically reducing redundant parameters. By drawing on dynamical systems theory, we introduce a neuronal stability metric (analogous to Lyapunov stability) that quantifies the contraction of neuron activations toward stable patterns during sequence processing. Using this metric, we reveal that emergent reasoning capabilities correspond to an external driving force or ``potential'', which nudges the neural dynamics away from trivial trajectories while preserving stability. Empirically, we instantiate NGC in large language models (LLMs) and demonstrate improved performance on complex reasoning benchmarks under moderate compression. NGC consistently outperforms standard low-rank approximations and cross-layer basis-sharing methods at comparable compression rates. We conclude by discussing the broader implications of NGC, including how structured neuronal group dynamics might relate to generalization in high-dimensional learning systems.
>
---
#### [new 065] What Can String Probability Tell Us About Grammaticality?
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨语言模型的概率输出能否反映语法知识，提出并验证三个理论预测：最小语义差异句对的概率相关性、模型与人类判断的一致性、语法与非语法句子在概率空间中难分离。旨在为评估语言模型的语法能力提供理论依据。**

- **链接: [http://arxiv.org/pdf/2510.16227v1](http://arxiv.org/pdf/2510.16227v1)**

> **作者:** Jennifer Hu; Ethan Gotlieb Wilcox; Siyuan Song; Kyle Mahowald; Roger P. Levy
>
> **摘要:** What have language models (LMs) learned about grammar? This question remains hotly debated, with major ramifications for linguistic theory. However, since probability and grammaticality are distinct notions in linguistics, it is not obvious what string probabilities can reveal about an LM's underlying grammatical knowledge. We present a theoretical analysis of the relationship between grammar, meaning, and string probability, based on simple assumptions about the generative process of corpus data. Our framework makes three predictions, which we validate empirically using 280K sentence pairs in English and Chinese: (1) correlation between the probability of strings within minimal pairs, i.e., string pairs with minimal semantic differences; (2) correlation between models' and humans' deltas within minimal pairs; and (3) poor separation in probability space between unpaired grammatical and ungrammatical strings. Our analyses give theoretical grounding for using probability to learn about LMs' structural knowledge, and suggest directions for future work in LM grammatical evaluation.
>
---
#### [new 066] ReviewGuard: Enhancing Deficient Peer Review Detection via LLM-Driven Data Augmentation
- **分类: cs.CL**

- **简介: 该论文提出ReviewGuard，旨在检测缺陷同行评审。利用LLM生成合成数据增强训练，构建四阶段框架，提升模型对缺陷评审的识别能力，保障学术评审质量与诚信。**

- **链接: [http://arxiv.org/pdf/2510.16549v1](http://arxiv.org/pdf/2510.16549v1)**

> **作者:** Haoxuan Zhang; Ruochi Li; Sarthak Shrestha; Shree Harshini Mamidala; Revanth Putta; Arka Krishan Aggarwal; Ting Xiao; Junhua Ding; Haihua Chen
>
> **摘要:** Peer review serves as the gatekeeper of science, yet the surge in submissions and widespread adoption of large language models (LLMs) in scholarly evaluation present unprecedented challenges. Recent work has focused on using LLMs to improve review efficiency or generate insightful review content. However, unchecked deficient reviews from both human experts and AI systems threaten to systematically undermine the peer review ecosystem and compromise academic integrity. To address this critical issue, we introduce ReviewGuard, an automated system for detecting and categorizing deficient reviews. ReviewGuard employs a comprehensive four-stage LLM-driven framework that: (1) collects ICLR and NeurIPS papers with their corresponding reviews from OpenReview; (2) annotates review types using GPT-4.1 with human validation; (3) addresses class imbalance and data scarcity through LLM-driven synthetic data augmentation, producing a final corpus of 6,634 papers, 24,657 real reviews, and 46,438 synthetic reviews; and (4) fine-tunes both encoder-based models and open source LLMs. We perform comprehensive feature analysis of the structure and quality of the review text. Compared to sufficient reviews, deficient reviews demonstrate lower rating scores, higher self-reported confidence, reduced structural complexity, and a higher proportion of negative sentiment. AI-generated text detection reveals that, since ChatGPT's emergence, AI-generated reviews have increased dramatically. In the evaluation of deficient review detection models, mixed training with synthetic and real review data provides substantial enhancements to recall and F1 scores on the binary task. This study presents the first LLM-driven system for detecting deficient peer reviews, providing evidence to inform AI governance in peer review while offering valuable insights into human-AI collaboration to maintain academic integrity.
>
---
#### [new 067] EgMM-Corpus: A Multimodal Vision-Language Dataset for Egyptian Culture
- **分类: cs.CL**

- **简介: 该论文构建了面向埃及文化的多模态视觉-语言数据集EgMM-Corpus，包含3000余张图像、覆盖313个文化概念。旨在解决现有模型在中东与非洲文化上的偏差问题，验证了CLIP在此数据集上性能有限，凸显该数据集对发展文化感知模型的重要性。**

- **链接: [http://arxiv.org/pdf/2510.16198v1](http://arxiv.org/pdf/2510.16198v1)**

> **作者:** Mohamed Gamil; Abdelrahman Elsayed; Abdelrahman Lila; Ahmed Gad; Hesham Abdelgawad; Mohamed Aref; Ahmed Fares
>
> **摘要:** Despite recent advances in AI, multimodal culturally diverse datasets are still limited, particularly for regions in the Middle East and Africa. In this paper, we introduce EgMM-Corpus, a multimodal dataset dedicated to Egyptian culture. By designing and running a new data collection pipeline, we collected over 3,000 images, covering 313 concepts across landmarks, food, and folklore. Each entry in the dataset is manually validated for cultural authenticity and multimodal coherence. EgMM-Corpus aims to provide a reliable resource for evaluating and training vision-language models in an Egyptian cultural context. We further evaluate the zero-shot performance of Contrastive Language-Image Pre-training CLIP on EgMM-Corpus, on which it achieves 21.2% Top-1 accuracy and 36.4% Top-5 accuracy in classification. These results underscore the existing cultural bias in large-scale vision-language models and demonstrate the importance of EgMM-Corpus as a benchmark for developing culturally aware models.
>
---
#### [new 068] Knowing the Facts but Choosing the Shortcut: Understanding How Large Language Models Compare Entities
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在实体比较任务中依赖知识还是启发式策略的问题。通过分析数值属性判断行为，发现模型受流行度、提及顺序等表面线索影响，小模型更依赖启发式，大模型能选择性使用可靠知识，思维链提示可促进数值推理。**

- **链接: [http://arxiv.org/pdf/2510.16815v1](http://arxiv.org/pdf/2510.16815v1)**

> **作者:** Hans Hergen Lehmann; Jae Hee Lee; Steven Schockaert; Stefan Wermter
>
> **备注:** 33 pages, 20 figures. Submitted ACL ARR 2025 October (under review)
>
> **摘要:** Large Language Models (LLMs) are increasingly used for knowledge-based reasoning tasks, yet understanding when they rely on genuine knowledge versus superficial heuristics remains challenging. We investigate this question through entity comparison tasks by asking models to compare entities along numerical attributes (e.g., ``Which river is longer, the Danube or the Nile?''), which offer clear ground truth for systematic analysis. Despite having sufficient numerical knowledge to answer correctly, LLMs frequently make predictions that contradict this knowledge. We identify three heuristic biases that strongly influence model predictions: entity popularity, mention order, and semantic co-occurrence. For smaller models, a simple logistic regression using only these surface cues predicts model choices more accurately than the model's own numerical predictions, suggesting heuristics largely override principled reasoning. Crucially, we find that larger models (32B parameters) selectively rely on numerical knowledge when it is more reliable, while smaller models (7--8B parameters) show no such discrimination, which explains why larger models outperform smaller ones even when the smaller models possess more accurate knowledge. Chain-of-thought prompting steers all models towards using the numerical features across all model sizes.
>
---
#### [new 069] so much depends / upon / a whitespace: Why Whitespace Matters for Poets and LLMs
- **分类: cs.CL**

- **简介: 该论文研究诗歌中空白字符的使用，揭示其在诗歌形式与语义中的重要性。属于NLP与文学交叉任务，旨在分析人类诗人、LLM生成及网络社区诗歌中 whitespace 的差异，并倡导重视文本预处理对诗歌数据的影响。**

- **链接: [http://arxiv.org/pdf/2510.16713v1](http://arxiv.org/pdf/2510.16713v1)**

> **作者:** Sriharsh Bhyravajjula; Melanie Walsh; Anna Preus; Maria Antoniak
>
> **摘要:** Whitespace is a critical component of poetic form, reflecting both adherence to standardized forms and rebellion against those forms. Each poem's whitespace distribution reflects the artistic choices of the poet and is an integral semantic and spatial feature of the poem. Yet, despite the popularity of poetry as both a long-standing art form and as a generation task for large language models (LLMs), whitespace has not received sufficient attention from the NLP community. Using a corpus of 19k English-language published poems from Poetry Foundation, we investigate how 4k poets have used whitespace in their works. We release a subset of 2.8k public-domain poems with preserved formatting to facilitate further research in this area. We compare whitespace usage in the published poems to (1) 51k LLM-generated poems, and (2) 12k unpublished poems posted in an online community. We also explore whitespace usage across time periods, poetic forms, and data sources. Additionally, we find that different text processing methods can result in significantly different representations of whitespace in poetry data, motivating us to use these poems and whitespace patterns to discuss implications for the processing strategies used to assemble pretraining datasets for LLMs.
>
---
#### [new 070] PANER: A Paraphrase-Augmented Framework for Low-Resource Named Entity Recognition
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对低资源命名实体识别（NER）任务，提出PANER框架。通过改进指令微调模板和引入保留实体信息的上下文改写数据增强方法，提升小样本场景下的模型性能，显著提高F1分数，适用于标注数据和计算资源有限的情况。**

- **链接: [http://arxiv.org/pdf/2510.17720v1](http://arxiv.org/pdf/2510.17720v1)**

> **作者:** Nanda Kumar Rengarajan; Jun Yan; Chun Wang
>
> **摘要:** Named Entity Recognition (NER) is a critical task that requires substantial annotated data, making it challenging in low-resource scenarios where label acquisition is expensive. While zero-shot and instruction-tuned approaches have made progress, they often fail to generalize to domain-specific entities and do not effectively utilize limited available data. We present a lightweight few-shot NER framework that addresses these challenges through two key innovations: (1) a new instruction tuning template with a simplified output format that combines principles from prior IT approaches to leverage the large context window of recent state-of-the-art LLMs; (2) introducing a strategic data augmentation technique that preserves entity information while paraphrasing the surrounding context, thereby expanding our training data without compromising semantic relationships. Experiments on benchmark datasets show that our method achieves performance comparable to state-of-the-art models on few-shot and zero-shot tasks, with our few-shot approach attaining an average F1 score of 80.1 on the CrossNER datasets. Models trained with our paraphrasing approach show consistent improvements in F1 scores of up to 17 points over baseline versions, offering a promising solution for groups with limited NER training data and compute power.
>
---
#### [new 071] Online Learning Defense against Iterative Jailbreak Attacks via Prompt Optimization
- **分类: cs.CL**

- **简介: 该论文针对迭代越狱攻击，提出基于在线学习和强化学习的防御框架，通过动态优化提示词抵御攻击，同时提升无害任务回答质量。引入PDGD抑制过拟合，实验证明在多个大模型上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.17006v1](http://arxiv.org/pdf/2510.17006v1)**

> **作者:** Masahiro Kaneko; Zeerak Talat; Timothy Baldwin
>
> **摘要:** Iterative jailbreak methods that repeatedly rewrite and input prompts into large language models (LLMs) to induce harmful outputs -- using the model's previous responses to guide each new iteration -- have been found to be a highly effective attack strategy. Despite being an effective attack strategy against LLMs and their safety mechanisms, existing defenses do not proactively disrupt this dynamic trial-and-error cycle. In this study, we propose a novel framework that dynamically updates its defense strategy through online learning in response to each new prompt from iterative jailbreak methods. Leveraging the distinctions between harmful jailbreak-generated prompts and typical harmless prompts, we introduce a reinforcement learning-based approach that optimizes prompts to ensure appropriate responses for harmless tasks while explicitly rejecting harmful prompts. Additionally, to curb overfitting to the narrow band of partial input rewrites explored during an attack, we introduce Past-Direction Gradient Damping (PDGD). Experiments conducted on three LLMs show that our approach significantly outperforms five existing defense methods against five iterative jailbreak methods. Moreover, our results indicate that our prompt optimization strategy simultaneously enhances response quality for harmless tasks.
>
---
#### [new 072] How News Feels: Understanding Affective Bias in Multilingual Headlines for Human-Centered Media Design
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言新闻标题中的情感偏差，旨在揭示媒体如何通过情绪化表达影响公众。作者对30万条孟加拉语新闻进行情感分析，发现负面情绪主导，并据此提出一种可可视化情感倾向的人本新闻聚合器设计。**

- **链接: [http://arxiv.org/pdf/2510.17252v1](http://arxiv.org/pdf/2510.17252v1)**

> **作者:** Mohd Ruhul Ameen; Akif Islam; Abu Saleh Musa Miah; Ayesha Siddiqua; Jungpil Shin
>
> **备注:** 15 pages, 7 figures, 4 tables. Submitted to the International Conference on Data and Applied Analytics (IDAA 2025)
>
> **摘要:** News media often shape the public mood not only by what they report but by how they frame it. The same event can appear calm in one outlet and alarming in another, reflecting subtle emotional bias in reporting. Negative or emotionally charged headlines tend to attract more attention and spread faster, which in turn encourages outlets to frame stories in ways that provoke stronger reactions. This research explores that tendency through large-scale emotion analysis of Bengali news. Using zero-shot inference with Gemma-3 4B, we analyzed 300000 Bengali news headlines and their content to identify the dominant emotion and overall tone of each. The findings reveal a clear dominance of negative emotions, particularly anger, fear, and disappointment, and significant variation in how similar stories are emotionally portrayed across outlets. Based on these insights, we propose design ideas for a human-centered news aggregator that visualizes emotional cues and helps readers recognize hidden affective framing in daily news.
>
---
#### [new 073] Multilingual Clinical NER for Diseases and Medications Recognition in Cardiology Texts using BERT Embeddings
- **分类: cs.CL**

- **简介: 该论文属于临床命名实体识别任务，旨在解决低资源语言中疾病和药物识别问题。作者基于BERT模型构建多语言系统，用于处理英文、西班牙文和意大利文的心脏病学文本，并在多项指标上超越基准结果。**

- **链接: [http://arxiv.org/pdf/2510.17437v1](http://arxiv.org/pdf/2510.17437v1)**

> **作者:** Manuela Daniela Danu; George Marica; Constantin Suciu; Lucian Mihai Itu; Oladimeji Farri
>
> **备注:** 11 pages, 5 figures, 1 table, published in Working Notes of the Conference and Labs of the Evaluation Forum (CLEF 2024)
>
> **摘要:** The rapidly increasing volume of electronic health record (EHR) data underscores a pressing need to unlock biomedical knowledge from unstructured clinical texts to support advancements in data-driven clinical systems, including patient diagnosis, disease progression monitoring, treatment effects assessment, prediction of future clinical events, etc. While contextualized language models have demonstrated impressive performance improvements for named entity recognition (NER) systems in English corpora, there remains a scarcity of research focused on clinical texts in low-resource languages. To bridge this gap, our study aims to develop multiple deep contextual embedding models to enhance clinical NER in the cardiology domain, as part of the BioASQ MultiCardioNER shared task. We explore the effectiveness of different monolingual and multilingual BERT-based models, trained on general domain text, for extracting disease and medication mentions from clinical case reports written in English, Spanish, and Italian. We achieved an F1-score of 77.88% on Spanish Diseases Recognition (SDR), 92.09% on Spanish Medications Recognition (SMR), 91.74% on English Medications Recognition (EMR), and 88.9% on Italian Medications Recognition (IMR). These results outperform the mean and median F1 scores in the test leaderboard across all subtasks, with the mean/median values being: 69.61%/75.66% for SDR, 81.22%/90.18% for SMR, 89.2%/88.96% for EMR, and 82.8%/87.76% for IMR.
>
---
#### [new 074] All You Need is One: Capsule Prompt Tuning with a Single Vector
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究提示调优任务，旨在解决传统方法依赖人工搜索提示长度且缺乏实例感知的问题。提出Capsule Prompt-Tuning（CaPT），用单个向量融合任务与实例信息，形成“注意力锚点”，提升性能并显著降低参数量。**

- **链接: [http://arxiv.org/pdf/2510.16670v1](http://arxiv.org/pdf/2510.16670v1)**

> **作者:** Yiyang Liu; James C. Liang; Heng Fan; Wenhao Yang; Yiming Cui; Xiaotian Han; Lifu Huang; Dongfang Liu; Qifan Wang; Cheng Han
>
> **备注:** NeurIPS 2025
>
> **摘要:** Prompt-based learning has emerged as a parameter-efficient finetuning (PEFT) approach to facilitate Large Language Model (LLM) adaptation to downstream tasks by conditioning generation with task-aware guidance. Despite its successes, current prompt-based learning methods heavily rely on laborious grid searching for optimal prompt length and typically require considerable number of prompts, introducing additional computational burden. Worse yet, our pioneer findings indicate that the task-aware prompt design is inherently limited by its absence of instance-aware information, leading to a subtle attention interplay with the input sequence. In contrast, simply incorporating instance-aware information as a part of the guidance can enhance the prompt-tuned model performance without additional fine-tuning. Moreover, we find an interesting phenomenon, namely "attention anchor", that incorporating instance-aware tokens at the earliest position of the sequence can successfully preserve strong attention to critical structural information and exhibit more active attention interaction with all input tokens. In light of our observation, we introduce Capsule Prompt-Tuning (CaPT), an efficient and effective solution that leverages off-the-shelf, informative instance semantics into prompt-based learning. Our approach innovatively integrates both instance-aware and task-aware information in a nearly parameter-free manner (i.e., one single capsule prompt). Empirical results demonstrate that our method can exhibit superior performance across various language tasks (e.g., 84.03\% average accuracy on T5-Large), serving as an "attention anchor," while enjoying high parameter efficiency (e.g., 0.003\% of model parameters on Llama3.2-1B).
>
---
#### [new 075] Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究智能体系统自动组合任务，解决现有方法因静态检索和能力描述不全导致的组件选择低效问题。提出基于在线背包问题的框架，动态评估组件性能、成本与兼容性，实现高效组装。实验表明其在多场景下显著提升成功率并降低成本。**

- **链接: [http://arxiv.org/pdf/2510.16499v1](http://arxiv.org/pdf/2510.16499v1)**

> **作者:** Michelle Yuan; Khushbu Pahwa; Shuaichen Chang; Mustafa Kaba; Jiarong Jiang; Xiaofei Ma; Yi Zhang; Monica Sunkara
>
> **备注:** Accepted to NeurIPS 2025 Conference
>
> **摘要:** Designing effective agentic systems requires the seamless composition and integration of agents, tools, and models within dynamic and uncertain environments. Most existing methods rely on static, semantic retrieval approaches for tool or agent discovery. However, effective reuse and composition of existing components remain challenging due to incomplete capability descriptions and the limitations of retrieval methods. Component selection suffers because the decisions are not based on capability, cost, and real-time utility. To address these challenges, we introduce a structured, automated framework for agentic system composition that is inspired by the knapsack problem. Our framework enables a composer agent to systematically identify, select, and assemble an optimal set of agentic components by jointly considering performance, budget constraints, and compatibility. By dynamically testing candidate components and modeling their utility in real-time, our approach streamlines the assembly of agentic systems and facilitates scalable reuse of resources. Empirical evaluation with Claude 3.5 Sonnet across five benchmarking datasets shows that our online-knapsack-based composer consistently lies on the Pareto frontier, achieving higher success rates at significantly lower component costs compared to our baselines. In the single-agent setup, the online knapsack composer shows a success rate improvement of up to 31.6% in comparison to the retrieval baselines. In multi-agent systems, the online knapsack composer increases success rate from 37% to 87% when agents are selected from an agent inventory of 100+ agents. The substantial performance gap confirms the robust adaptability of our method across diverse domains and budget constraints.
>
---
#### [new 076] Prompt-MII: Meta-Learning Instruction Induction for LLMs
- **分类: cs.CL**

- **简介: 该论文聚焦指令诱导任务，旨在解决大语言模型上下文学习推理成本高的问题。提出PROMPT-MII框架，通过元学习和强化学习生成紧凑指令，在少得多的token下达到媲美上下文学习的效果。**

- **链接: [http://arxiv.org/pdf/2510.16932v1](http://arxiv.org/pdf/2510.16932v1)**

> **作者:** Emily Xiao; Yixiao Zeng; Ada Chen; Chin-Jou Li; Amanda Bertsch; Graham Neubig
>
> **摘要:** A popular method to adapt large language models (LLMs) to new tasks is in-context learning (ICL), which is effective but incurs high inference costs as context length grows. In this paper we propose a method to perform instruction induction, where we take training examples and reduce them to a compact but descriptive prompt that can achieve performance comparable to ICL over the full training set. Specifically, we propose PROMPT-MII, a reinforcement learning (RL) based framework to meta-learn an instruction induction model that can generate compact instructions on the fly for an arbitrary new dataset. We train on over 3,000 diverse classification datasets from the HuggingFace hub, and evaluate on 90 unseen tasks. PROMPT-MII improves downstream model quality by 4-9 F1 points (10-20% relative), matching ICL performance while requiring 3-13x fewer tokens.
>
---
#### [new 077] StreamingThinker: Large Language Models Can Think While Reading
- **分类: cs.CL**

- **简介: 该论文提出“流式思考”范式，解决大模型在输入完整后才开始推理导致的延迟问题。通过StreamingThinker框架实现边读边想，结合流式CoT生成、约束训练与并行推理，在保持性能的同时显著降低等待和响应时间。**

- **链接: [http://arxiv.org/pdf/2510.17238v1](http://arxiv.org/pdf/2510.17238v1)**

> **作者:** Junlong Tong; Yingqi Fan; Anhao Zhao; Yunpu Ma; Xiaoyu Shen
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities in chain of thought (CoT) reasoning. However, the current LLM reasoning paradigm initiates thinking only after the entire input is available, which introduces unnecessary latency and weakens attention to earlier information in dynamic scenarios. Inspired by human cognition of thinking while reading, we first design a \textit{\textbf{streaming thinking}} paradigm for LLMs, where reasoning unfolds in the order of input and further adjusts its depth once reading is complete. We instantiate this paradigm with \textit{StreamingThinker}, a framework that enables LLMs to think while reading through the integration of streaming CoT generation, streaming-constraint training, and streaming parallel inference. Specifically, StreamingThinker employs streaming reasoning units with quality control for CoT generation, enforces order-preserving reasoning through streaming attention masks and position encoding, and leverages parallel KV caches that decouple input encoding from reasoning generation, thereby ensuring alignment and enabling true concurrency. We evaluate StreamingThinker on the Qwen3 model family across math reasoning, logical reasoning, and context-based QA reasoning tasks. Experimental results show that the StreamingThinker preserves performance comparable to batch thinking, while yielding an 80\% reduction in token waiting before the onset of reasoning and a more than 60\% reduction in time-level latency for producing the final answer, demonstrating the effectiveness of the streaming paradigm for LLM reasoning. Code will be released at \href{https://github.com/EIT-NLP/StreamingLLM/tree/main/StreamingThinker}{this repository.}
>
---
#### [new 078] Extended LSTM: Adaptive Feature Gating for Toxic Comment Classification
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对毒性评论分类中模型计算成本高、小类别性能差的问题，提出xLSTM框架。通过余弦相似性门控、自适应特征加权与类别重平衡，实现高效准确检测，在少参数下显著提升对少数毒性的识别效果。**

- **链接: [http://arxiv.org/pdf/2510.17018v1](http://arxiv.org/pdf/2510.17018v1)**

> **作者:** Noor Islam S. Mohammad
>
> **摘要:** Toxic comment detection remains a challenging task, where transformer-based models (e.g., BERT) incur high computational costs and degrade on minority toxicity classes, while classical ensembles lack semantic adaptability. We propose xLSTM, a parameter-efficient and theoretically grounded framework that unifies cosine-similarity gating, adaptive feature prioritization, and principled class rebalancing. A learnable reference vector {v} in {R}^d modulates contextual embeddings via cosine similarity, amplifying toxic cues and attenuating benign signals to yield stronger gradients under severe class imbalance. xLSTM integrates multi-source embeddings (GloVe, FastText, BERT CLS) through a projection layer, a character-level BiLSTM for morphological cues, embedding-space SMOTE for minority augmentation, and adaptive focal loss with dynamic class weighting. On the Jigsaw Toxic Comment benchmark, xLSTM attains 96.0% accuracy and 0.88 macro-F1, outperforming BERT by 33% on threat and 28% on identity_hate categories, with 15 times fewer parameters and 50ms inference latency. Cosine gating contributes a +4.8% F1 gain in ablations. The results establish a new efficiency adaptability frontier, demonstrating that lightweight, theoretically informed architectures can surpass large pretrained models on imbalanced, domain-specific NLP tasks.
>
---
#### [new 079] Evaluating Large Language Models on Urdu Idiom Translation
- **分类: cs.CL**

- **简介: 该论文聚焦乌尔都语习语翻译任务，旨在解决低资源语言中习语翻译研究不足的问题。作者构建了首个乌尔都语至英语的习语翻译评测数据集，评估了多种大模型与机器翻译系统在不同脚本下的表现，发现提示工程和原生文字输入更利于保留文化语义。**

- **链接: [http://arxiv.org/pdf/2510.17460v1](http://arxiv.org/pdf/2510.17460v1)**

> **作者:** Muhammad Farmal Khan; Mousumi Akter
>
> **摘要:** Idiomatic translation remains a significant challenge in machine translation, especially for low resource languages such as Urdu, and has received limited prior attention. To advance research in this area, we introduce the first evaluation datasets for Urdu to English idiomatic translation, covering both Native Urdu and Roman Urdu scripts and annotated with gold-standard English equivalents. We evaluate multiple open-source Large Language Models (LLMs) and Neural Machine Translation (NMT) systems on this task, focusing on their ability to preserve idiomatic and cultural meaning. Automatic metrics including BLEU, BERTScore, COMET, and XCOMET are used to assess translation quality. Our findings indicate that prompt engineering enhances idiomatic translation compared to direct translation, though performance differences among prompt types are relatively minor. Moreover, cross script comparisons reveal that text representation substantially affects translation quality, with Native Urdu inputs producing more accurate idiomatic translations than Roman Urdu.
>
---
#### [new 080] Towards Mixed-Modal Retrieval for Universal Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文研究通用检索增强生成（URAG），解决现有RAG在混合模态（如图文）场景下的局限。提出Nyx模型和NyxQA数据集，通过自动化构建数据与两阶段训练，实现混合模态检索，提升视觉-语言生成效果。**

- **链接: [http://arxiv.org/pdf/2510.17354v1](http://arxiv.org/pdf/2510.17354v1)**

> **作者:** Chenghao Zhang; Guanting Dong; Xinyu Yang; Zhicheng Dou
>
> **备注:** This work is in progress
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing large language models (LLMs) by retrieving relevant documents from an external corpus. However, existing RAG systems primarily focus on unimodal text documents, and often fall short in real-world scenarios where both queries and documents may contain mixed modalities (such as text and images). In this paper, we address the challenge of Universal Retrieval-Augmented Generation (URAG), which involves retrieving and reasoning over mixed-modal information to improve vision-language generation. To this end, we propose Nyx, a unified mixed-modal to mixed-modal retriever tailored for URAG scenarios. To mitigate the scarcity of realistic mixed-modal data, we introduce a four-stage automated pipeline for generation and filtering, leveraging web documents to construct NyxQA, a dataset comprising diverse mixed-modal question-answer pairs that better reflect real-world information needs. Building on this high-quality dataset, we adopt a two-stage training framework for Nyx: we first perform pre-training on NyxQA along with a variety of open-source retrieval datasets, followed by supervised fine-tuning using feedback from downstream vision-language models (VLMs) to align retrieval outputs with generative preferences. Experimental results demonstrate that Nyx not only performs competitively on standard text-only RAG benchmarks, but also excels in the more general and realistic URAG setting, significantly improving generation quality in vision-language tasks.
>
---
#### [new 081] Verification-Aware Planning for Multi-Agent Systems
- **分类: cs.CL; cs.AI; cs.LG; cs.MA**

- **简介: 该论文研究多智能体系统的协同规划问题，旨在解决任务理解偏差和协作错误。作者提出VeriMAP框架，通过引入验证感知的规划机制，将子任务验证函数融入规划过程，提升系统鲁棒性与可解释性。**

- **链接: [http://arxiv.org/pdf/2510.17109v1](http://arxiv.org/pdf/2510.17109v1)**

> **作者:** Tianyang Xu; Dan Zhang; Kushan Mitra; Estevam Hruschka
>
> **备注:** Submission for ARR Oct
>
> **摘要:** Large language model (LLM) agents are increasingly deployed to tackle complex tasks, often necessitating collaboration among multiple specialized agents. However, multi-agent collaboration introduces new challenges in planning, coordination, and verification. Execution failures frequently arise not from flawed reasoning alone, but from subtle misalignments in task interpretation, output format, or inter-agent handoffs. To address these challenges, we present VeriMAP, a framework for multi-agent collaboration with verification-aware planning. The VeriMAP planner decomposes tasks, models subtask dependencies, and encodes planner-defined passing criteria as subtask verification functions (VFs) in Python and natural language. We evaluate VeriMAP on diverse datasets, demonstrating that it outperforms both single- and multi-agent baselines while enhancing system robustness and interpretability. Our analysis highlights how verification-aware planning enables reliable coordination and iterative refinement in multi-agent systems, without relying on external labels or annotations.
>
---
#### [new 082] Wisdom is Knowing What not to Say: Hallucination-Free LLMs Unlearning via Attention Shifting
- **分类: cs.CL**

- **简介: 该论文研究大模型遗忘学习任务，旨在解决现有方法在消除敏感信息时易导致模型性能下降或产生幻觉回应的问题。提出注意力迁移（AS）框架，通过抑制关键token注意力并增强保留数据的语义关注，实现高效、可靠的选择性遗忘。**

- **链接: [http://arxiv.org/pdf/2510.17210v1](http://arxiv.org/pdf/2510.17210v1)**

> **作者:** Chenchen Tan; Youyang Qu; Xinghao Li; Hui Zhang; Shujie Cui; Cunjian Chen; Longxiang Gao
>
> **备注:** 22 pages, 10 figures
>
> **摘要:** The increase in computing power and the necessity of AI-assisted decision-making boost the growing application of large language models (LLMs). Along with this, the potential retention of sensitive data of LLMs has spurred increasing research into machine unlearning. However, existing unlearning approaches face a critical dilemma: Aggressive unlearning compromises model utility, while conservative strategies preserve utility but risk hallucinated responses. This significantly limits LLMs' reliability in knowledge-intensive applications. To address this, we introduce a novel Attention-Shifting (AS) framework for selective unlearning. AS is driven by two design objectives: (1) context-preserving suppression that attenuates attention to fact-bearing tokens without disrupting LLMs' linguistic structure; and (2) hallucination-resistant response shaping that discourages fabricated completions when queried about unlearning content. AS realizes these objectives through two attention-level interventions, which are importance-aware suppression applied to the unlearning set to reduce reliance on memorized knowledge and attention-guided retention enhancement that reinforces attention toward semantically essential tokens in the retained dataset to mitigate unintended degradation. These two components are jointly optimized via a dual-loss objective, which forms a soft boundary that localizes unlearning while preserving unrelated knowledge under representation superposition. Experimental results show that AS improves performance preservation over the state-of-the-art unlearning methods, achieving up to 15% higher accuracy on the ToFU benchmark and 10% on the TDEC benchmark, while maintaining competitive hallucination-free unlearning effectiveness. Compared to existing methods, AS demonstrates a superior balance between unlearning effectiveness, generalization, and response reliability.
>
---
#### [new 083] Can LLMs Correct Themselves? A Benchmark of Self-Correction in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLM）的自我纠错能力，旨在评估不同自纠错方法在常识推理、数学推理和代码生成任务中的有效性。作者构建了CorrectBench基准，发现自纠错可提升性能，但效率待优化，并指出简单思维链方法已具竞争力。**

- **链接: [http://arxiv.org/pdf/2510.16062v1](http://arxiv.org/pdf/2510.16062v1)**

> **作者:** Guiyao Tie; Zenghui Yuan; Zeli Zhao; Chaoran Hu; Tianhe Gu; Ruihang Zhang; Sizhe Zhang; Junran Wu; Xiaoyue Tu; Ming Jin; Qingsong Wen; Lixing Chen; Pan Zhou; Lichao Sun
>
> **备注:** 38 pages, 25 figures, 8 tables
>
> **摘要:** Self-correction of large language models (LLMs) emerges as a critical component for enhancing their reasoning performance. Although various self-correction methods have been proposed, a comprehensive evaluation of these methods remains largely unexplored, and the question of whether LLMs can truly correct themselves is a matter of significant interest and concern. In this study, we introduce CorrectBench, a benchmark developed to evaluate the effectiveness of self-correction strategies, including intrinsic, external, and fine-tuned approaches, across three tasks: commonsense reasoning, mathematical reasoning, and code generation. Our findings reveal that: 1) Self-correction methods can improve accuracy, especially for complex reasoning tasks; 2) Mixing different self-correction strategies yields further improvements, though it reduces efficiency; 3) Reasoning LLMs (e.g., DeepSeek-R1) have limited optimization under additional self-correction methods and have high time costs. Interestingly, a comparatively simple chain-of-thought (CoT) baseline demonstrates competitive accuracy and efficiency. These results underscore the potential of self-correction to enhance LLM's reasoning performance while highlighting the ongoing challenge of improving their efficiency. Consequently, we advocate for further research focused on optimizing the balance between reasoning capabilities and operational efficiency. Project Page: https://correctbench.github.io/
>
---
#### [new 084] QueST: Incentivizing LLMs to Generate Difficult Problems
- **分类: cs.CL**

- **简介: 该论文提出QueST框架，旨在生成高难度编程问题以缓解现有数据集规模和难度不足的问题。通过难度感知的图采样与拒绝微调，训练模型生成高质量难题，用于知识蒸馏和强化学习，显著提升小模型在编程任务上的表现。**

- **链接: [http://arxiv.org/pdf/2510.17715v1](http://arxiv.org/pdf/2510.17715v1)**

> **作者:** Hanxu Hu; Xingxing Zhang; Jannis Vamvas; Rico Sennrich; Furu Wei
>
> **备注:** 20 pages, 7 figures
>
> **摘要:** Large Language Models have achieved strong performance on reasoning tasks, solving competition-level coding and math problems. However, their scalability is limited by human-labeled datasets and the lack of large-scale, challenging coding problem training data. Existing competitive coding datasets contain only thousands to tens of thousands of problems. Previous synthetic data generation methods rely on either augmenting existing instruction datasets or selecting challenging problems from human-labeled data. In this paper, we propose QueST, a novel framework which combines difficulty-aware graph sampling and difficulty-aware rejection fine-tuning that directly optimizes specialized generators to create challenging coding problems. Our trained generators demonstrate superior capability compared to even GPT-4o at creating challenging problems that benefit downstream performance. We leverage QueST to generate large-scale synthetic coding problems, which we then use to distill from strong teacher models with long chain-of-thought or to conduct reinforcement learning for smaller models, proving effective in both scenarios. Our distillation experiments demonstrate significant performance gains. Specifically, after fine-tuning Qwen3-8B-base on 100K difficult problems generated by QueST, we surpass the performance of the original Qwen3-8B on LiveCodeBench. With an additional 112K examples (i.e., 28K human-written problems paired with multiple synthetic solutions), our 8B model matches the performance of the much larger DeepSeek-R1-671B. These findings indicate that generating complex problems via QueST offers an effective and scalable approach to advancing the frontiers of competitive coding and reasoning for large language models.
>
---
#### [new 085] Addressing Antisocial Behavior in Multi-Party Dialogs Through Multimodal Representation Learning
- **分类: cs.CL**

- **简介: 该论文针对多参与者对话中的反社会行为识别问题，基于法语数据集开展研究。工作涵盖滥用检测、欺凌分析和群体识别三任务，比较了文本与图模型，并提出多模态融合方法，验证了其在复杂行为识别中的优越性。**

- **链接: [http://arxiv.org/pdf/2510.17289v1](http://arxiv.org/pdf/2510.17289v1)**

> **作者:** Hajar Bakarou; Mohamed Sinane El Messoussi; Anaïs Ollagnier
>
> **摘要:** Antisocial behavior (ASB) on social media -- including hate speech, harassment, and cyberbullying -- poses growing risks to platform safety and societal well-being. Prior research has focused largely on networks such as X and Reddit, while \textit{multi-party conversational settings} remain underexplored due to limited data. To address this gap, we use \textit{CyberAgressionAdo-Large}, a French open-access dataset simulating ASB in multi-party conversations, and evaluate three tasks: \textit{abuse detection}, \textit{bullying behavior analysis}, and \textit{bullying peer-group identification}. We benchmark six text-based and eight graph-based \textit{representation-learning methods}, analyzing lexical cues, interactional dynamics, and their multimodal fusion. Results show that multimodal models outperform unimodal baselines. The late fusion model \texttt{mBERT + WD-SGCN} achieves the best overall results, with top performance on abuse detection (0.718) and competitive scores on peer-group identification (0.286) and bullying analysis (0.606). Error analysis highlights its effectiveness in handling nuanced ASB phenomena such as implicit aggression, role transitions, and context-dependent hostility.
>
---
#### [new 086] SimBench: Benchmarking the Ability of Large Language Models to Simulate Human Behaviors
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文提出SimBench，首个大规模标准化基准，用于评估大语言模型模拟人类行为的能力。旨在解决现有评估碎片化问题，统一20个多样化数据集，揭示模型在不同任务、人群中的表现规律及与推理能力的关系。**

- **链接: [http://arxiv.org/pdf/2510.17516v1](http://arxiv.org/pdf/2510.17516v1)**

> **作者:** Tiancheng Hu; Joachim Baumann; Lorenzo Lupo; Dirk Hovy; Nigel Collier; Paul Röttger
>
> **备注:** Project Website: http://simbench.tiancheng.hu/ Data: https://huggingface.co/datasets/pitehu/SimBench
>
> **摘要:** Large language model (LLM) simulations of human behavior have the potential to revolutionize the social and behavioral sciences, if and only if they faithfully reflect real human behaviors. Current evaluations are fragmented, based on bespoke tasks and metrics, creating a patchwork of incomparable results. To address this, we introduce SimBench, the first large-scale, standardized benchmark for a robust, reproducible science of LLM simulation. By unifying 20 diverse datasets covering tasks from moral decision-making to economic choice across a large global participant pool, SimBench provides the necessary foundation to ask fundamental questions about when, how, and why LLM simulations succeed or fail. We show that, while even the best LLMs today have limited simulation ability (score: 40.80/100), performance scales log-linearly with model size. Simulation performance is not improved by increased inference-time compute. We demonstrate an alignment-simulation trade-off: instruction-tuning improves performance on low-entropy (consensus) questions but degrades it on high-entropy (diverse) ones. Models particularly struggle when simulating specific demographic groups. Finally, we demonstrate that simulation ability correlates most strongly with deep, knowledge-intensive reasoning (MMLU-Pro, r=0.939). By making progress measurable, we aim to accelerate the development of more faithful LLM simulators.
>
---
#### [new 087] In Generative AI We (Dis)Trust? Computational Analysis of Trust and Distrust in Reddit Discussions
- **分类: cs.CL**

- **简介: 该论文属计算社会科学任务，旨在分析公众对生成式AI的信任与不信任。基于Reddit多子版数据，结合众包标注与分类模型，揭示信任动态、影响因素及用户群体差异，提供大规模信任分析框架。**

- **链接: [http://arxiv.org/pdf/2510.16173v1](http://arxiv.org/pdf/2510.16173v1)**

> **作者:** Aria Pessianzadeh; Naima Sultana; Hildegarde Van den Bulck; David Gefen; Shahin Jabari; Rezvaneh Rezapour
>
> **摘要:** The rise of generative AI (GenAI) has impacted many aspects of human life. As these systems become embedded in everyday practices, understanding public trust in them also becomes essential for responsible adoption and governance. Prior work on trust in AI has largely drawn from psychology and human-computer interaction, but there is a lack of computational, large-scale, and longitudinal approaches to measuring trust and distrust in GenAI and large language models (LLMs). This paper presents the first computational study of Trust and Distrust in GenAI, using a multi-year Reddit dataset (2022--2025) spanning 39 subreddits and 197,618 posts. Crowd-sourced annotations of a representative sample were combined with classification models to scale analysis. We find that Trust and Distrust are nearly balanced over time, with shifts around major model releases. Technical performance and usability dominate as dimensions, while personal experience is the most frequent reason shaping attitudes. Distinct patterns also emerge across trustors (e.g., experts, ethicists, general users). Our results provide a methodological framework for large-scale Trust analysis and insights into evolving public perceptions of GenAI.
>
---
#### [new 088] Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations
- **分类: cs.CL**

- **简介: 该论文研究自然语言推断中人类标注差异，利用LiTEx分类法分析标注者推理过程。不仅考察标签一致性，还结合解释相似性与个体偏好，揭示标签分歧背后可能存在解释一致的现象，强调解释比标签更能反映语义理解。**

- **链接: [http://arxiv.org/pdf/2510.16458v1](http://arxiv.org/pdf/2510.16458v1)**

> **作者:** Pingjun Hong; Beiduo Chen; Siyao Peng; Marie-Catherine de Marneffe; Benjamin Roth; Barbara Plank
>
> **摘要:** Natural Language Inference datasets often exhibit human label variation. To better understand these variations, explanation-based approaches analyze the underlying reasoning behind annotators' decisions. One such approach is the LiTEx taxonomy, which categorizes free-text explanations in English into reasoning types. However, previous work applying such taxonomies has focused on within-label variation: cases where annotators agree on the final NLI label but provide different explanations. In contrast, this paper broadens the scope by examining how annotators may diverge not only in the reasoning type but also in the labeling step. We use explanations as a lens to decompose the reasoning process underlying NLI annotation and to analyze individual differences. We apply LiTEx to two NLI English datasets and align annotation variation from multiple aspects: NLI label agreement, explanation similarity, and taxonomy agreement, with an additional compounding factor of annotators' selection bias. We observe instances where annotators disagree on the label but provide highly similar explanations, suggesting that surface-level disagreement may mask underlying agreement in interpretation. Moreover, our analysis reveals individual preferences in explanation strategies and label choices. These findings highlight that agreement in reasoning types better reflects the semantic similarity of free-text explanations than label agreement alone. Our findings underscore the richness of reasoning-based explanations and the need for caution in treating labels as ground truth.
>
---
#### [new 089] DiscoTrack: A Multilingual LLM Benchmark for Discourse Tracking
- **分类: cs.CL**

- **简介: 该论文提出DiscoTrack，一个多语言基准，旨在评估大模型在话语跟踪中的隐含信息理解与语用推理能力，涵盖12种语言和四个层次的 discourse 任务，解决现有基准偏重显性信息提取的问题。**

- **链接: [http://arxiv.org/pdf/2510.17013v1](http://arxiv.org/pdf/2510.17013v1)**

> **作者:** Lanni Bu; Lauren Levin; Amir Zeldes
>
> **摘要:** Recent LLM benchmarks have tested models on a range of phenomena, but are still focused primarily on natural language understanding for extraction of explicit information, such as QA or summarization, with responses often tar- geting information from individual sentences. We are still lacking more challenging, and im- portantly also multilingual, benchmarks focus- ing on implicit information and pragmatic infer- ences across larger documents in the context of discourse tracking: integrating and aggregating information across sentences, paragraphs and multiple speaker utterances. To this end, we present DiscoTrack, an LLM benchmark target- ing a range of tasks across 12 languages and four levels of discourse understanding: salience recognition, entity tracking, discourse relations and bridging inference. Our evaluation shows that these tasks remain challenging, even for state-of-the-art models.
>
---
#### [new 090] End-to-End Argument Mining through Autoregressive Argumentative Structure Prediction
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究论点挖掘任务，旨在解决论点成分与关系联合建模的难题。提出端到端的自回归论点结构预测框架（AASP），通过预定义动作序列逐步生成论点结构，有效捕捉推理流程，在多个基准上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2510.16363v1](http://arxiv.org/pdf/2510.16363v1)**

> **作者:** Nilmadhab Das; Vishal Vaibhav; Yash Sunil Choudhary; V. Vijaya Saradhi; Ashish Anand
>
> **备注:** Accepted version. To appear in IJCNN 2025
>
> **摘要:** Argument Mining (AM) helps in automating the extraction of complex argumentative structures such as Argument Components (ACs) like Premise, Claim etc. and Argumentative Relations (ARs) like Support, Attack etc. in an argumentative text. Due to the inherent complexity of reasoning involved with this task, modelling dependencies between ACs and ARs is challenging. Most of the recent approaches formulate this task through a generative paradigm by flattening the argumentative structures. In contrast to that, this study jointly formulates the key tasks of AM in an end-to-end fashion using Autoregressive Argumentative Structure Prediction (AASP) framework. The proposed AASP framework is based on the autoregressive structure prediction framework that has given good performance for several NLP tasks. AASP framework models the argumentative structures as constrained pre-defined sets of actions with the help of a conditional pre-trained language model. These actions build the argumentative structures step-by-step in an autoregressive manner to capture the flow of argumentative reasoning in an efficient way. Extensive experiments conducted on three standard AM benchmarks demonstrate that AASP achieves state-of-theart (SoTA) results across all AM tasks in two benchmarks and delivers strong results in one benchmark.
>
---
#### [new 091] Cross-Genre Authorship Attribution via LLM-Based Retrieve-and-Rerank
- **分类: cs.CL**

- **简介: 该论文研究跨体裁作者归属任务，旨在识别未知文本的作者。为解决现有方法受主题干扰的问题，提出基于大模型的检索-重排序框架，并设计针对性数据策略，提升对作者语言特征的学习，显著超越先前方法。**

- **链接: [http://arxiv.org/pdf/2510.16819v1](http://arxiv.org/pdf/2510.16819v1)**

> **作者:** Shantanu Agarwal; Joel Barry; Steven Fincke; Scott Miller
>
> **摘要:** Authorship attribution (AA) is the task of identifying the most likely author of a query document from a predefined set of candidate authors. We introduce a two-stage retrieve-and-rerank framework that finetunes LLMs for cross-genre AA. Unlike the field of information retrieval (IR), where retrieve-and-rerank is a de facto strategy, cross-genre AA systems must avoid relying on topical cues and instead learn to identify author-specific linguistic patterns that are independent of the text's subject matter (genre/domain/topic). Consequently, for the reranker, we demonstrate that training strategies commonly used in IR are fundamentally misaligned with cross-genre AA, leading to suboptimal behavior. To address this, we introduce a targeted data curation strategy that enables the reranker to effectively learn author-discriminative signals. Using our LLM-based retrieve-and-rerank pipeline, we achieve substantial gains of 22.3 and 34.4 absolute Success@8 points over the previous state-of-the-art on HIATUS's challenging HRS1 and HRS2 cross-genre AA benchmarks.
>
---
#### [new 092] Lingua Custodi's participation at the WMT 2025 Terminology shared task
- **分类: cs.CL**

- **简介: 该论文参与WMT 2025术语共享任务，旨在提升多语言句向量表示。通过结合MLM、TLM等方法，提出高效跨语言句子嵌入模型，在减少80%平行数据需求下，显著提升双语文本检索性能，并支持高质量NMT训练。**

- **链接: [http://arxiv.org/pdf/2510.17504v1](http://arxiv.org/pdf/2510.17504v1)**

> **作者:** Jingshu Liu; Raheel Qader; Gaëtan Caillaut; Mariam Nakhlé
>
> **摘要:** While BERT is an effective method for learning monolingual sentence embeddings for semantic similarity and embedding based transfer learning BERT based cross-lingual sentence embeddings have yet to be explored. We systematically investigate methods for learning multilingual sentence embeddings by combining the best methods for learning monolingual and cross-lingual representations including: masked language modeling (MLM), translation language modeling (TLM), dual encoder translation ranking, and additive margin softmax. We show that introducing a pre-trained multilingual language model dramatically reduces the amount of parallel training data required to achieve good performance by 80%. Composing the best of these methods produces a model that achieves 83.7% bi-text retrieval accuracy over 112 languages on Tatoeba, well above the 65.5 achieved by LASER, while still performing competitively on monolingual transfer learning benchmarks. Parallel data mined from CommonCrawl using our best model is shown to train competitive NMT models for en-zh and en-de. We publicly release our best multilingual sentence embedding model for 109+ languages at https://tfhub.dev/google/LaBSE.
>
---
#### [new 093] Natural Language Processing Applications in Cardiology: A Narrative Review
- **分类: cs.CL; cs.AI**

- **简介: 该论文是一篇综述，旨在总结2014至2025年间自然语言处理（NLP）在心脏病学中的应用。研究系统梳理了265篇文献，从NLP方法、任务类型、疾病类别和数据来源等维度进行分析，揭示该领域的发展趋势与多样性，为未来研究提供全面参考。**

- **链接: [http://arxiv.org/pdf/2510.16708v1](http://arxiv.org/pdf/2510.16708v1)**

> **作者:** Kailai Yang; Yan Leng; Xin Zhang; Tianlin Zhang; Paul Thompson; Bernard Keavney; Maciej Tomaszewski; Sophia Ananiadou
>
> **摘要:** Cardiovascular disease has become increasingly prevalent in modern society and has a significant effect on global health and well-being. Heart-related conditions are intricate, multifaceted disorders, which may be influenced by a combination of genetic predispositions, lifestyle choices, and various socioeconomic and clinical factors. Information regarding these potentially complex interrelationships is dispersed among diverse types of textual data, which include patient narratives, medical records, and scientific literature, among others. Natural language processing (NLP) techniques have increasingly been adopted as a powerful means to analyse and make sense of this vast amount of unstructured data. This, in turn, can allow healthcare professionals to gain deeper insights into the cardiology field, which has the potential to revolutionize current approaches to the diagnosis, treatment, and prevention of cardiac problems. This review provides a detailed overview of NLP research in cardiology between 2014 and 2025. We queried six literature databases to find articles describing the application of NLP techniques in the context of a range of different cardiovascular diseases. Following a rigorous screening process, we identified a total of 265 relevant articles. We analysed each article from multiple dimensions, i.e., NLP paradigm types, cardiology-related task types, cardiovascular disease types, and data source types. Our analysis reveals considerable diversity within each of these dimensions, thus demonstrating the considerable breadth of NLP research within the field. We also perform a temporal analysis, which illustrates the evolution and changing trends in NLP methods employed over the last decade that we cover. To our knowledge, the review constitutes the most comprehensive overview of NLP research in cardiology to date.
>
---
#### [new 094] Navigating through the hidden embedding space: steering LLMs to improve mental health assessment
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦心理健康评估任务，旨在提升小规模大模型在抑郁症状识别和问卷填写中的表现。作者提出一种轻量级线性变换方法，通过操纵隐藏层激活的 steering 向量，低成本增强模型在特定心理任务上的性能。**

- **链接: [http://arxiv.org/pdf/2510.16373v1](http://arxiv.org/pdf/2510.16373v1)**

> **作者:** Federico Ravenda; Seyed Ali Bahrainian; Andrea Raballo; Antonietta Mira
>
> **摘要:** The rapid evolution of Large Language Models (LLMs) is transforming AI, opening new opportunities in sensitive and high-impact areas such as Mental Health (MH). Yet, despite these advancements, recent evidence reveals that smaller-scale models still struggle to deliver optimal performance in domain-specific applications. In this study, we present a cost-efficient yet powerful approach to improve MH assessment capabilities of an LLM, without relying on any computationally intensive techniques. Our lightweight method consists of a linear transformation applied to a specific layer's activations, leveraging steering vectors to guide the model's output. Remarkably, this intervention enables the model to achieve improved results across two distinct tasks: (1) identifying whether a Reddit post is useful for detecting the presence or absence of depressive symptoms (relevance prediction task), and (2) completing a standardized psychological screening questionnaire for depression based on users' Reddit post history (questionnaire completion task). Results highlight the untapped potential of steering mechanisms as computationally efficient tools for LLMs' MH domain adaptation.
>
---
#### [new 095] Explainability of Large Language Models: Opportunities and Challenges toward Generating Trustworthy Explanations
- **分类: cs.CL**

- **简介: 该论文聚焦大语言模型的可解释性，旨在提升模型透明度与信任度。它综述了解释方法，通过医疗和自动驾驶领域的实验分析解释效果，并探讨了实现可信解释的挑战与未来方向。**

- **链接: [http://arxiv.org/pdf/2510.17256v1](http://arxiv.org/pdf/2510.17256v1)**

> **作者:** Shahin Atakishiyev; Housam K. B. Babiker; Jiayi Dai; Nawshad Farruque; Teruaki Hayashi; Nafisa Sadaf Hriti; Md Abed Rahman; Iain Smith; Mi-Young Kim; Osmar R. Zaïane; Randy Goebel
>
> **摘要:** Large language models have exhibited impressive performance across a broad range of downstream tasks in natural language processing. However, how a language model predicts the next token and generates content is not generally understandable by humans. Furthermore, these models often make errors in prediction and reasoning, known as hallucinations. These errors underscore the urgent need to better understand and interpret the intricate inner workings of language models and how they generate predictive outputs. Motivated by this gap, this paper investigates local explainability and mechanistic interpretability within Transformer-based large language models to foster trust in such models. In this regard, our paper aims to make three key contributions. First, we present a review of local explainability and mechanistic interpretability approaches and insights from relevant studies in the literature. Furthermore, we describe experimental studies on explainability and reasoning with large language models in two critical domains -- healthcare and autonomous driving -- and analyze the trust implications of such explanations for explanation receivers. Finally, we summarize current unaddressed issues in the evolving landscape of LLM explainability and outline the opportunities, critical challenges, and future directions toward generating human-aligned, trustworthy LLM explanations.
>
---
#### [new 096] AcademicEval: Live Long-Context LLM Benchmark
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出AcademicEval，一个用于评估大语言模型长上下文生成能力的动态基准。针对现有基准存在的标签泄露、人工标注耗时和固定上下文长度问题，利用arXiv论文构建无需人工标注的学术写作任务，并通过作者合作图提供高质量少样本示例，实现灵活、无标签泄露的高效评测。**

- **链接: [http://arxiv.org/pdf/2510.17725v1](http://arxiv.org/pdf/2510.17725v1)**

> **作者:** Haozhen Zhang; Tao Feng; Pengrui Han; Jiaxuan You
>
> **备注:** Accepted by TMLR. Code is available at https://github.com/ulab-uiuc/AcademicEval
>
> **摘要:** Large Language Models (LLMs) have recently achieved remarkable performance in long-context understanding. However, current long-context LLM benchmarks are limited by rigid context length, labor-intensive annotation, and the pressing challenge of label leakage issues during LLM training. Therefore, we propose \textsc{AcademicEval}, a live benchmark for evaluating LLMs over long-context generation tasks. \textsc{AcademicEval} adopts papers on arXiv to introduce several academic writing tasks with long-context inputs, \textit{i.e.}, \textsc{Title}, \textsc{Abstract}, \textsc{Introduction}, and \textsc{Related Work}, which cover a wide range of abstraction levels and require no manual labeling. Moreover, \textsc{AcademicEval} integrates high-quality and expert-curated few-shot demonstrations from a collected co-author graph to enable flexible context length. Especially, \textsc{AcademicEval} features an efficient live evaluation, ensuring no label leakage. We conduct a holistic evaluation on \textsc{AcademicEval}, and the results illustrate that LLMs perform poorly on tasks with hierarchical abstraction levels and tend to struggle with long few-shot demonstrations, highlighting the challenge of our benchmark. Through experimental analysis, we also reveal some insights for enhancing LLMs' long-context modeling capabilities. Code is available at https://github.com/ulab-uiuc/AcademicEval
>
---
#### [new 097] OncoReason: Structuring Clinical Reasoning in LLMs for Robust and Interpretable Survival Prediction
- **分类: cs.CL; cs.LG**

- **简介: 该论文聚焦癌症生存预测任务，旨在提升模型的准确性与可解释性。针对临床数据异质性和现有LLM缺乏结构化推理能力的问题，提出OncoReason框架，结合多任务学习与链式思维提示、强化学习等对齐策略，实现更优的预测与合理推理。**

- **链接: [http://arxiv.org/pdf/2510.17532v1](http://arxiv.org/pdf/2510.17532v1)**

> **作者:** Raghu Vamshi Hemadri; Geetha Krishna Guruju; Kristi Topollai; Anna Ewa Choromanska
>
> **摘要:** Predicting cancer treatment outcomes requires models that are both accurate and interpretable, particularly in the presence of heterogeneous clinical data. While large language models (LLMs) have shown strong performance in biomedical NLP, they often lack structured reasoning capabilities critical for high-stakes decision support. We present a unified, multi-task learning framework that aligns autoregressive LLMs with clinical reasoning for outcome prediction on the MSK-CHORD dataset. Our models are trained to jointly perform binary survival classification, continuous survival time regression, and natural language rationale generation. We evaluate three alignment strategies: (1) standard supervised fine-tuning (SFT), (2) SFT with Chain-of-Thought (CoT) prompting to elicit step-by-step reasoning, and (3) Group Relative Policy Optimization (GRPO), a reinforcement learning method that aligns model outputs to expert-derived reasoning trajectories. Experiments with LLaMa3-8B and Med42-8B backbones demonstrate that CoT prompting improves F1 by +6.0 and reduces MAE by 12%, while GRPO achieves state-of-the-art interpretability and predictive performance across BLEU, ROUGE, and BERTScore. We further show that existing biomedical LLMs often fail to produce valid reasoning traces due to architectural constraints. Our findings underscore the importance of reasoning-aware alignment in multi-task clinical modeling and set a new benchmark for interpretable, trustworthy LLMs in precision oncology.
>
---
#### [new 098] Thinking About Thinking: Evaluating Reasoning in Post-Trained Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究后训练语言模型的推理认知能力，探讨其对学习策略的意识、泛化性及推理与输出的一致性，比较SFT、DPO和GRPO方法的效果，发现强化学习方法模型更具策略意识和泛化能力，但推理与输出对齐较弱。**

- **链接: [http://arxiv.org/pdf/2510.16340v1](http://arxiv.org/pdf/2510.16340v1)**

> **作者:** Pratham Singla; Shivank Garg; Ayush Singh; Ishan Garg; Ketan Suhaas Saichandran
>
> **摘要:** Recent advances in post-training techniques have endowed Large Language Models (LLMs) with enhanced capabilities for tackling complex, logic-intensive tasks through the generation of supplementary planning tokens. This development raises a fundamental question: Are these models aware of what they "learn" and "think"? To address this, we define three core competencies: (1) awareness of learned latent policies, (2) generalization of these policies across domains, and (3) alignment between internal reasoning traces and final outputs. We empirically evaluate these abilities on several tasks, each designed to require learning a distinct policy. Furthermore, we contrast the profiles of models post-trained via Supervised Fine-Tuning (SFT), Direct Policy Optimization (DPO), and Group Relative Policy Optimization (GRPO). Our findings indicate that RL-trained models not only demonstrate greater awareness of their learned behaviors and stronger generalizability to novel, structurally similar tasks than SFT models but also often exhibit weak alignment between their reasoning traces and final outputs, an effect most pronounced in GRPO-trained models.
>
---
#### [new 099] Evaluating Prompting Strategies and Large Language Models in Systematic Literature Review Screening: Relevance and Task-Stage Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在系统文献综述筛选中的应用，解决自动化筛选中文献相关性分类与任务阶段分类问题。比较六种模型与五种提示策略的性能，提出基于成本与效果的分阶段工作流建议。**

- **链接: [http://arxiv.org/pdf/2510.16091v1](http://arxiv.org/pdf/2510.16091v1)**

> **作者:** Binglan Han; Anuradha Mathrani; Teo Susnjak
>
> **摘要:** This study quantifies how prompting strategies interact with large language models (LLMs) to automate the screening stage of systematic literature reviews (SLRs). We evaluate six LLMs (GPT-4o, GPT-4o-mini, DeepSeek-Chat-V3, Gemini-2.5-Flash, Claude-3.5-Haiku, Llama-4-Maverick) under five prompt types (zero-shot, few-shot, chain-of-thought (CoT), CoT-few-shot, self-reflection) across relevance classification and six Level-2 tasks, using accuracy, precision, recall, and F1. Results show pronounced model-prompt interaction effects: CoT-few-shot yields the most reliable precision-recall balance; zero-shot maximizes recall for high-sensitivity passes; and self-reflection underperforms due to over-inclusivity and instability across models. GPT-4o and DeepSeek provide robust overall performance, while GPT-4o-mini performs competitively at a substantially lower dollar cost. A cost-performance analysis for relevance classification (per 1,000 abstracts) reveals large absolute differences among model-prompt pairings; GPT-4o-mini remains low-cost across prompts, and structured prompts (CoT/CoT-few-shot) on GPT-4o-mini offer attractive F1 at a small incremental cost. We recommend a staged workflow that (1) deploys low-cost models with structured prompts for first-pass screening and (2) escalates only borderline cases to higher-capacity models. These findings highlight LLMs' uneven but promising potential to automate literature screening. By systematically analyzing prompt-model interactions, we provide a comparative benchmark and practical guidance for task-adaptive LLM deployment.
>
---
#### [new 100] ReXMoE: Reusing Experts with Minimal Overhead in Mixture-of-Experts
- **分类: cs.CL**

- **简介: 该论文提出ReXMoE，旨在解决MoE架构中层局部路由限制导致的专家组合灵活性不足问题。通过跨层复用专家并设计渐进式扩展路由策略，在不增参数的前提下提升模型表达能力和性能，属于高效大模型架构设计任务。**

- **链接: [http://arxiv.org/pdf/2510.17483v1](http://arxiv.org/pdf/2510.17483v1)**

> **作者:** Zheyue Tan; Zhiyuan Li; Tao Yuan; Dong Zhou; Weilin Liu; Yueqing Zhuang; Yadong Li; Guowei Niu; Cheng Qin; Zhuyu Yao; Congyi Liu; Haiyang Xu; Boxun Li; Guohao Dai; Bo Zhao; Yu Wang
>
> **摘要:** Mixture-of-Experts (MoE) architectures have emerged as a promising approach to scale Large Language Models (LLMs). MoE boosts the efficiency by activating a subset of experts per token. Recent works show that fine-grained experts substantially enriches the combinatorial flexibility of active experts and enhances model expressiveness. However, such a design is fundamentally limited by the layer-local routing mechanism: each layer is restricted to its own expert pool. This requires a careful trade-off between expert dimensionality and routing diversity given fixed parameter budgets. We describe ReXMoE, a novel MoE architecture that improves routing beyond the existing layer-local approaches by allowing routers to reuse experts across adjacent layers. ReXMoE decouples expert dimensionality from per-layer budgets, enabling richer expert combinations without sacrificing individual expert capacity or inflating overall parameters. To this end, we propose a new progressive scaling routing (PSR) strategy to gradually increase the candidate expert pool during training. As a result, ReXMoE improves both language modeling and downstream task performance. Extensive experiments on models ranging from 0.5B to 7B parameters across different architectures demonstrate that ReXMoE consistently improves performance under fixed architectural dimensions, confirming ReXMoE as new design paradigm for parameter-efficient and scalable MoE-based LLMs.
>
---
#### [new 101] Annotation-Efficient Universal Honesty Alignment
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的诚实对齐问题，旨在减少训练所需正确性标注。提出EliCal框架，先用自一致性生成置信度，再用少量标注校准，实现高效通用的诚实对齐。**

- **链接: [http://arxiv.org/pdf/2510.17509v1](http://arxiv.org/pdf/2510.17509v1)**

> **作者:** Shiyu Ni; Keping Bi; Jiafeng Guo; Minghao Tang; Jingtong Wu; Zengxin Han; Xueqi Cheng
>
> **摘要:** Honesty alignment-the ability of large language models (LLMs) to recognize their knowledge boundaries and express calibrated confidence-is essential for trustworthy deployment. Existing methods either rely on training-free confidence estimation (e.g., token probabilities, self-consistency) or training-based calibration with correctness annotations. While effective, achieving universal honesty alignment with training-based calibration requires costly, large-scale labeling. To support annotation-efficient training, we introduce Elicitation-Then-Calibration (EliCal), a two-stage framework that first elicits internal confidence using inexpensive self-consistency supervision, then calibrates this confidence with a small set of correctness annotations. To support a large-scale study, we release HonestyBench, a benchmark covering ten free-form QA datasets with 560k training and 70k evaluation instances annotated with correctness and self-consistency signals. Experiments show that EliCal achieves near-optimal alignment with only 1k correctness annotations (0.18% of full supervision) and better alignment performance on unseen MMLU tasks than the calibration-only baseline, offering a scalable solution toward universal honesty alignment in LLMs.
>
---
#### [new 102] Deep Self-Evolving Reasoning
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的推理能力提升，针对开放权重小模型验证修正能力弱的问题，提出深度自进化推理（DSER）方法。通过并行多步迭代推理，利用微弱改进倾向逐步逼近正确答案，在AIME难题上显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.17498v1](http://arxiv.org/pdf/2510.17498v1)**

> **作者:** Zihan Liu; Shun Zheng; Xumeng Wen; Yang Wang; Jiang Bian; Mao Yang
>
> **摘要:** Long-form chain-of-thought reasoning has become a cornerstone of advanced reasoning in large language models. While recent verification-refinement frameworks have enabled proprietary models to solve Olympiad-level problems, their effectiveness hinges on strong, reliable verification and correction capabilities, which remain fragile in open-weight, smaller-scale models. This work demonstrates that even with weak verification and refinement capabilities on hard tasks, the reasoning limits of such models can be substantially extended through a probabilistic paradigm we call Deep Self-Evolving Reasoning (DSER). We conceptualize iterative reasoning as a Markov chain, where each step represents a stochastic transition in the solution space. The key insight is that convergence to a correct solution is guaranteed as long as the probability of improvement marginally exceeds that of degradation. By running multiple long-horizon, self-evolving processes in parallel, DSER amplifies these small positive tendencies, enabling the model to asymptotically approach correct answers. Empirically, we apply DSER to the DeepSeek-R1-0528-Qwen3-8B model. On the challenging AIME 2024-2025 benchmark, DSER solves 5 out of 9 previously unsolvable problems and boosts overall performance, enabling this compact model to surpass the single-turn accuracy of its 600B-parameter teacher through majority voting. Beyond its immediate utility for test-time scaling, the DSER framework serves to diagnose the fundamental limitations of current open-weight reasoners. By clearly delineating their shortcomings in self-verification, refinement, and stability, our findings establish a clear research agenda for developing next-generation models with powerful, intrinsic self-evolving capabilities.
>
---
#### [new 103] Beacon: Single-Turn Diagnosis and Mitigation of Latent Sycophancy in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型中的隐性谄媚（sycophancy）问题，即模型为迎合用户而牺牲事实准确性。提出Beacon基准单轮评测该偏差，并设计干预方法揭示对齐机制的内在张力，推动可复现的对齐偏差研究。**

- **链接: [http://arxiv.org/pdf/2510.16727v1](http://arxiv.org/pdf/2510.16727v1)**

> **作者:** Sanskar Pandey; Ruhaan Chopra; Angkul Puniya; Sohom Pal
>
> **摘要:** Large language models internalize a structural trade-off between truthfulness and obsequious flattery, emerging from reward optimization that conflates helpfulness with polite submission. This latent bias, known as sycophancy, manifests as a preference for user agreement over principled reasoning. We introduce Beacon, a single-turn forced-choice benchmark that isolates this bias independent of conversational context, enabling precise measurement of the tension between factual accuracy and submissive bias. Evaluations across twelve state-of-the-art models reveal that sycophancy decomposes into stable linguistic and affective sub-biases, each scaling with model capacity. We further propose prompt-level and activation-level interventions that modulate these biases in opposing directions, exposing the internal geometry of alignment as a dynamic manifold between truthfulness and socially compliant judgment. Beacon reframes sycophancy as a measurable form of normative misgeneralization, providing a reproducible foundation for studying and mitigating alignment drift in large-scale generative systems.
>
---
#### [new 104] Contextual Attention Modulation: Towards Efficient Multi-Task Adaptation in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦大语言模型的多任务适应问题，旨在缓解知识遗忘与资源消耗。提出上下文注意力调制（CAM）及HyCAM框架，通过动态调节注意力机制，提升多任务性能，兼顾效率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.17705v1](http://arxiv.org/pdf/2510.17705v1)**

> **作者:** Dayan Pan; Zhaoyang Fu; Jingyuan Wang; Xiao Han; Yue Zhu; Xiangyu Zhao
>
> **备注:** Accepted by CIKM' 25
>
> **摘要:** Large Language Models (LLMs) possess remarkable generalization capabilities but struggle with multi-task adaptation, particularly in balancing knowledge retention with task-specific specialization. Conventional fine-tuning methods suffer from catastrophic forgetting and substantial resource consumption, while existing parameter-efficient methods perform suboptimally in complex multi-task scenarios. To address this, we propose Contextual Attention Modulation (CAM), a novel mechanism that dynamically modulates the representations of self-attention modules in LLMs. CAM enhances task-specific features while preserving general knowledge, thereby facilitating more effective and efficient adaptation. For effective multi-task adaptation, CAM is integrated into our Hybrid Contextual Attention Modulation (HyCAM) framework, which combines a shared, full-parameter CAM module with multiple specialized, lightweight CAM modules, enhanced by a dynamic routing strategy for adaptive knowledge fusion. Extensive experiments on heterogeneous tasks, including question answering, code generation, and logical reasoning, demonstrate that our approach significantly outperforms existing approaches, achieving an average performance improvement of 3.65%. The implemented code and data are available to ease reproducibility at https://github.com/Applied-Machine-Learning-Lab/HyCAM.
>
---
#### [new 105] Leave It to the Experts: Detecting Knowledge Distillation via MoE Expert Signatures
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对知识蒸馏（KD）检测任务，解决现有方法易被提示工程规避的问题。提出利用MoE模型专家路由模式作为“指纹”进行检测，并设计白盒与黑盒（Shadow-MoE）方案，实现高准确率、强鲁棒性的KD检测。**

- **链接: [http://arxiv.org/pdf/2510.16968v1](http://arxiv.org/pdf/2510.16968v1)**

> **作者:** Pingzhi Li; Morris Yu-Chao Huang; Zhen Tan; Qingquan Song; Jie Peng; Kai Zou; Yu Cheng; Kaidi Xu; Tianlong Chen
>
> **备注:** Code is at https://github.com/unites-lab/shadow-moe
>
> **摘要:** Knowledge Distillation (KD) accelerates training of large language models (LLMs) but poses intellectual property protection and LLM diversity risks. Existing KD detection methods based on self-identity or output similarity can be easily evaded through prompt engineering. We present a KD detection framework effective in both white-box and black-box settings by exploiting an overlooked signal: the transfer of MoE "structural habits", especially internal routing patterns. Our approach analyzes how different experts specialize and collaborate across various inputs, creating distinctive fingerprints that persist through the distillation process. To extend beyond the white-box setup and MoE architectures, we further propose Shadow-MoE, a black-box method that constructs proxy MoE representations via auxiliary distillation to compare these patterns between arbitrary model pairs. We establish a comprehensive, reproducible benchmark that offers diverse distilled checkpoints and an extensible framework to facilitate future research. Extensive experiments demonstrate >94% detection accuracy across various scenarios and strong robustness to prompt-based evasion, outperforming existing baselines while highlighting the structural habits transfer in LLMs.
>
---
#### [new 106] Xiaoice: Training-Free Video Understanding via Self-Supervised Spatio-Temporal Clustering of Semantic Features
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对视频理解中依赖大量标注数据的问题，提出一种无需训练的框架。通过结合预训练视觉语言模型与自监督时空聚类，实现视频分段与场景聚类，生成多模态摘要，完成零样本视频结构分析。**

- **链接: [http://arxiv.org/pdf/2510.16781v1](http://arxiv.org/pdf/2510.16781v1)**

> **作者:** Shihao Ji; Zihui Song
>
> **摘要:** The remarkable zero-shot reasoning capabilities of large-scale Visual Language Models (VLMs) on static images have yet to be fully translated to the video domain. Conventional video understanding models often rely on extensive, task-specific training on annotated datasets, a process that is both costly and limited in scalability. This paper introduces a novel, training-free framework for video understanding that circumvents end-to-end training by synergistically combining the rich semantic priors of pre-trained VLMs with classic machine learning algorithms for pattern discovery. Our core idea is to reframe video understanding as a self-supervised spatio-temporal clustering problem within a high-dimensional semantic feature space. The proposed pipeline first transforms a video stream into a semantic feature trajectory using the frozen visual encoder of a pre-trained VLM. Subsequently, we employ Kernel Temporal Segmentation (KTS), a robust machine learning technique, to partition the continuous feature stream into discrete, semantically coherent event segments. These segments are then subjected to unsupervised density-based clustering to identify recurring macroscopic scenes and themes throughout the video. By selecting representative keyframes from each discovered cluster and leveraging the VLM's generative capabilities for textual description, our framework automatically produces a structured, multi-modal summary of the video content. This approach provides an effective, interpretable, and model-agnostic pathway for zero-shot, automated structural analysis of video content.
>
---
#### [new 107] SAKE: Towards Editing Auditory Attribute Knowledge of Large Audio-Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文聚焦音频-语言模型中的知识编辑任务，旨在解决现有方法局限于文本和视觉模态的问题。作者提出首个针对听觉属性知识编辑的基准SAKE，评估七种方法在两个大模型上的表现，探讨编辑的可靠性、泛化性等挑战，推动多模态知识更新研究。**

- **链接: [http://arxiv.org/pdf/2510.16917v1](http://arxiv.org/pdf/2510.16917v1)**

> **作者:** Chih-Kai Yang; Yen-Ting Piao; Tzu-Wen Hsu; Szu-Wei Fu; Zhehuai Chen; Ke-Han Lu; Sung-Feng Huang; Chao-Han Huck Yang; Yu-Chiang Frank Wang; Yun-Nung Chen; Hung-yi Lee
>
> **备注:** Work in progress
>
> **摘要:** Knowledge editing offers an efficient way to update model knowledge without full retraining, but prior work has concentrated almost exclusively on textual or visual modalities. We introduce SAKE, the first benchmark specifically designed for editing auditory attribute knowledge in Large Audio-Language Models (LALMs). Unlike factual updates, SAKE targets several abstract auditory attributes, capturing knowledge types that go beyond conventional textual and visual domains. We benchmark seven editing methods on two LALMs along four dimensions: reliability, generality, audio/text locality, and portability. Results highlight challenges such as preserving intra-attribute knowledge unrelated to the edit, generalizing edits to multimodal reasoning, and maintaining edits under sequential updates. SAKE provides a principled framework to study how knowledge editing extends to the auditory modalities, opening new directions for maintaining and adapting LALMs in more diverse real-world scenarios.
>
---
#### [new 108] Investigating Safety Vulnerabilities of Large Audio-Language Models Under Speaker Emotional Variations
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文研究大型音频语言模型在不同说话人情绪下的安全漏洞。针对恶意指令在多情绪表达下的响应不一致问题，构建情感变化数据集并评测主流模型，发现情绪强度与风险呈非单调关系，中等强度最危险，提出需增强模型对情绪变化的鲁棒对齐。**

- **链接: [http://arxiv.org/pdf/2510.16893v1](http://arxiv.org/pdf/2510.16893v1)**

> **作者:** Bo-Han Feng; Chien-Feng Liu; Yu-Hsuan Li Liang; Chih-Kai Yang; Szu-Wei Fu; Zhehuai Chen; Ke-Han Lu; Sung-Feng Huang; Chao-Han Huck Yang; Yu-Chiang Frank Wang; Yun-Nung Chen; Hung-yi Lee
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Large audio-language models (LALMs) extend text-based LLMs with auditory understanding, offering new opportunities for multimodal applications. While their perception, reasoning, and task performance have been widely studied, their safety alignment under paralinguistic variation remains underexplored. This work systematically investigates the role of speaker emotion. We construct a dataset of malicious speech instructions expressed across multiple emotions and intensities, and evaluate several state-of-the-art LALMs. Our results reveal substantial safety inconsistencies: different emotions elicit varying levels of unsafe responses, and the effect of intensity is non-monotonic, with medium expressions often posing the greatest risk. These findings highlight an overlooked vulnerability in LALMs and call for alignment strategies explicitly designed to ensure robustness under emotional variation, a prerequisite for trustworthy deployment in real-world settings.
>
---
#### [new 109] Bolster Hallucination Detection via Prompt-Guided Data Augmentation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大模型幻觉检测中缺乏标注数据的问题，提出PALE框架，利用提示引导的LLM生成数据进行增广，并设计CM Score度量评估中间表征，实现无需人工标注的高效幻觉检测。**

- **链接: [http://arxiv.org/pdf/2510.15977v1](http://arxiv.org/pdf/2510.15977v1)**

> **作者:** Wenyun Li; Zheng Zhang; Dongmei Jiang; Xiangyuan Lan
>
> **摘要:** Large language models (LLMs) have garnered significant interest in AI community. Despite their impressive generation capabilities, they have been found to produce misleading or fabricated information, a phenomenon known as hallucinations. Consequently, hallucination detection has become critical to ensure the reliability of LLM-generated content. One primary challenge in hallucination detection is the scarcity of well-labeled datasets containing both truthful and hallucinated outputs. To address this issue, we introduce Prompt-guided data Augmented haLlucination dEtection (PALE), a novel framework that leverages prompt-guided responses from LLMs as data augmentation for hallucination detection. This strategy can generate both truthful and hallucinated data under prompt guidance at a relatively low cost. To more effectively evaluate the truthfulness of the sparse intermediate embeddings produced by LLMs, we introduce an estimation metric called the Contrastive Mahalanobis Score (CM Score). This score is based on modeling the distributions of truthful and hallucinated data in the activation space. CM Score employs a matrix decomposition approach to more accurately capture the underlying structure of these distributions. Importantly, our framework does not require additional human annotations, offering strong generalizability and practicality for real-world applications. Extensive experiments demonstrate that PALE achieves superior hallucination detection performance, outperforming the competitive baseline by a significant margin of 6.55%.
>
---
#### [new 110] Attention to Non-Adopters
- **分类: cs.CY; cs.CL; cs.HC; cs.LG**

- **简介: 该论文关注大语言模型（LLM）非使用者的需求，指出当前开发和评估过度依赖使用者数据，可能导致技术偏见与不平等。作者通过非使用者案例研究，揭示其独特需求与任务，并倡导采用以人为中心的方法将其纳入LLM设计与评估。**

- **链接: [http://arxiv.org/pdf/2510.15951v1](http://arxiv.org/pdf/2510.15951v1)**

> **作者:** Kaitlyn Zhou; Kristina Gligorić; Myra Cheng; Michelle S. Lam; Vyoma Raman; Boluwatife Aminu; Caeley Woo; Michael Brockman; Hannah Cha; Dan Jurafsky
>
> **摘要:** Although language model-based chat systems are increasingly used in daily life, most Americans remain non-adopters of chat-based LLMs -- as of June 2025, 66% had never used ChatGPT. At the same time, LLM development and evaluation rely mainly on data from adopters (e.g., logs, preference data), focusing on the needs and tasks for a limited demographic group of adopters in terms of geographic location, education, and gender. In this position paper, we argue that incorporating non-adopter perspectives is essential for developing broadly useful and capable LLMs. We contend that relying on methods that focus primarily on adopters will risk missing a range of tasks and needs prioritized by non-adopters, entrenching inequalities in who benefits from LLMs, and creating oversights in model development and evaluation. To illustrate this claim, we conduct case studies with non-adopters and show: how non-adopter needs diverge from those of current users, how non-adopter needs point us towards novel reasoning tasks, and how to systematically integrate non-adopter needs via human-centered methods.
>
---
#### [new 111] Do LLMs Recognize Your Latent Preferences? A Benchmark for Latent Information Discovery in Personalized Interaction
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文聚焦个性化交互中隐含偏好发现任务，旨在评估大语言模型通过对话推断用户未明说偏好的能力。作者构建了包含三类场景的统一基准，提出三智能体框架以量化模型在多轮交互中挖掘与利用隐性信息的表现。**

- **链接: [http://arxiv.org/pdf/2510.17132v1](http://arxiv.org/pdf/2510.17132v1)**

> **作者:** Ioannis Tsaknakis; Bingqing Song; Shuyu Gan; Dongyeop Kang; Alfredo Garcia; Gaowen Liu; Charles Fleming; Mingyi Hong
>
> **摘要:** Large Language Models (LLMs) excel at producing broadly relevant text, but this generality becomes a limitation when user-specific preferences are required, such as recommending restaurants or planning travel. In these scenarios, users rarely articulate every preference explicitly; instead, much of what they care about remains latent, waiting to be inferred. This raises a fundamental question: Can LLMs uncover and reason about such latent information through conversation? We address this problem by introducing a unified benchmark for evaluating latent information discovery - the ability of LLMs to reveal and utilize hidden user attributes through multi-turn interaction. The benchmark spans three progressively realistic settings: the classic 20 Questions game, Personalized Question Answering, and Personalized Text Summarization. All tasks share a tri-agent framework (User, Assistant, Judge) enabling turn-level evaluation of elicitation and adaptation. Our results reveal that while LLMs can indeed surface latent information through dialogue, their success varies dramatically with context: from 32% to 98%, depending on task complexity, topic, and number of hidden attributes. This benchmark provides the first systematic framework for studying latent information discovery in personalized interaction, highlighting that effective preference inference remains an open frontier for building truly adaptive AI systems.
>
---
#### [new 112] U-Codec: Ultra Low Frame-rate Neural Speech Codec for Fast High-fidelity Speech Generation
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文研究语音合成中的高效编解码，旨在解决低帧率下语音质量下降的问题。提出U-Codec，通过Transformer建模长时依赖，优化残差向量量化，实现5Hz超低帧率下的高保真、快速语音生成，并提升LLM-TTS推理速度3倍。**

- **链接: [http://arxiv.org/pdf/2510.16718v1](http://arxiv.org/pdf/2510.16718v1)**

> **作者:** Xusheng Yang; Long Zhou; Wenfu Wang; Kai Hu; Shulin Feng; Chenxing Li; Meng Yu; Dong Yu; Yuexian Zou
>
> **摘要:** We propose \textbf{U-Codec}, an \textbf{U}ltra low frame-rate neural speech \textbf{Codec} that achieves high-fidelity reconstruction and fast speech generation at an extremely low frame-rate of 5Hz (5 frames per second). Extreme compression at 5Hz typically leads to severe intelligibility and spectral detail loss, we introduce a Transformer-based inter-frame long-term dependency module and systematically explore residual vector quantization (RVQ) depth and codebook size to identify optimal configurations. Moreover, we apply U-Codec into a large language model (LLM)-based auto-regressive TTS model, which leverages global and local hierarchical architecture to effectively capture dependencies across multi-layer tokens. We extend LLM-based TTS from 3-layer RVQ at 50Hz to 32-layer RVQ at 5Hz. Experimental results demonstrate that U-Codec improves LLM-based TTS inference speed by around 3 $\times$ over high-frame-rate codecs while maintaining similarity and naturalness. These results validate the feasibility of using highly compressed 5Hz discrete tokens for fast and high-fidelity speech synthesis.
>
---
#### [new 113] SIADAFIX: issue description response for adaptive program repair
- **分类: cs.SE; cs.CL; D.2.2; D.2.3**

- **简介: 该论文针对程序修复任务，提出SIADAFIX方法，通过快慢思维机制适应性选择修复模式，利用问题描述响应优化工作流，提升大模型在复杂修复中的效率与准确性。**

- **链接: [http://arxiv.org/pdf/2510.16059v1](http://arxiv.org/pdf/2510.16059v1)**

> **作者:** Xin Cao; Nan Yu
>
> **备注:** 20 pages, 3 figures
>
> **摘要:** We propose utilizing fast and slow thinking to enhance the capabilities of large language model-based agents on complex tasks such as program repair. In particular, we design an adaptive program repair method based on issue description response, called SIADAFIX. The proposed method utilizes slow thinking bug fix agent to complete complex program repair tasks, and employs fast thinking workflow decision components to optimize and classify issue descriptions, using issue description response results to guide the orchestration of bug fix agent workflows. SIADAFIX adaptively selects three repair modes, i.e., easy, middle and hard mode, based on problem complexity. It employs fast generalization for simple problems and test-time scaling techniques for complex problems. Experimental results on the SWE-bench Lite show that the proposed method achieves 60.67% pass@1 performance using the Claude-4 Sonnet model, reaching state-of-the-art levels among all open-source methods. SIADAFIX effectively balances repair efficiency and accuracy, providing new insights for automated program repair. Our code is available at https://github.com/liauto-siada/siada-cli.
>
---
#### [new 114] What Questions Should Robots Be Able to Answer? A Dataset of User Questions for Explainable Robotics
- **分类: cs.RO; cs.CL; cs.HC; I.2.9; H.5.2; H.5.0; I.2.8; I.2.7; J.4**

- **简介: 该论文聚焦可解释机器人中的问答需求，旨在解决机器人应具备回答哪些用户问题的能力。作者构建了一个包含1893个用户问题的数据集，涵盖12类70子类，通过视频与文本刺激收集用户对家庭机器人的提问，揭示用户关注点及新手与专家的差异，为机器人问答系统提供基准与设计依据。**

- **链接: [http://arxiv.org/pdf/2510.16435v1](http://arxiv.org/pdf/2510.16435v1)**

> **作者:** Lennart Wachowiak; Andrew Coles; Gerard Canal; Oya Celiktutan
>
> **摘要:** With the growing use of large language models and conversational interfaces in human-robot interaction, robots' ability to answer user questions is more important than ever. We therefore introduce a dataset of 1,893 user questions for household robots, collected from 100 participants and organized into 12 categories and 70 subcategories. Most work in explainable robotics focuses on why-questions. In contrast, our dataset provides a wide variety of questions, from questions about simple execution details to questions about how the robot would act in hypothetical scenarios -- thus giving roboticists valuable insights into what questions their robot needs to be able to answer. To collect the dataset, we created 15 video stimuli and 7 text stimuli, depicting robots performing varied household tasks. We then asked participants on Prolific what questions they would want to ask the robot in each portrayed situation. In the final dataset, the most frequent categories are questions about task execution details (22.5%), the robot's capabilities (12.7%), and performance assessments (11.3%). Although questions about how robots would handle potentially difficult scenarios and ensure correct behavior are less frequent, users rank them as the most important for robots to be able to answer. Moreover, we find that users who identify as novices in robotics ask different questions than more experienced users. Novices are more likely to inquire about simple facts, such as what the robot did or the current state of the environment. As robots enter environments shared with humans and language becomes central to giving instructions and interaction, this dataset provides a valuable foundation for (i) identifying the information robots need to log and expose to conversational interfaces, (ii) benchmarking question-answering modules, and (iii) designing explanation strategies that align with user expectations.
>
---
#### [new 115] Long Exposure: Accelerating Parameter-Efficient Fine-Tuning for LLMs under Shadowy Sparsity
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文聚焦大语言模型的参数高效微调（PEFT）加速，针对未被充分关注的“幽影稀疏”特性，提出Long Exposure系统，通过稀疏性感知、序列预测与动态操作优化，实现最高2.49倍的端到端加速，提升微调效率。**

- **链接: [http://arxiv.org/pdf/2510.15964v1](http://arxiv.org/pdf/2510.15964v1)**

> **作者:** Tuowei Wang; Kun Li; Zixu Hao; Donglin Bai; Ju Ren; Yaoxue Zhang; Ting Cao; Mao Yang
>
> **摘要:** The adaptation of pre-trained large language models (LLMs) to diverse downstream tasks via fine-tuning is critical for numerous applications. However, the inefficiency of parameter-efficient fine-tuning (PEFT) techniques presents significant challenges in terms of time investments and operational costs. In this paper, we first introduce a nuanced form of sparsity, termed Shadowy Sparsity, which is distinctive in fine-tuning and has not been adequately addressed for acceleration. Under Shadowy Sparsity, we propose Long Exposure, an efficient system to accelerate PEFT for LLMs. Long Exposure comprises three key components: Shadowy-sparsity Exposer employs a prolonged sensing range to capture more sparsity details under shadowy sparsity; Sequence-oriented Predictor provides efficient yet accurate predictions to handle large sequence inputs and constantly-evolving parameters; and Dynamic-aware Operator facilitates more structured computational patterns and coalesced memory accesses, addressing dynamic sparse operations. Extensive evaluations show that Long Exposure outperforms state-of-the-arts with up to a $2.49\times$ speedup in end-to-end fine-tuning, offering promising advancements in accelerating PEFT for LLMs.
>
---
#### [new 116] Zeroth-Order Sharpness-Aware Learning with Exponential Tilting
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文研究零阶优化与锐度感知最小化（SAM）的结合，提出基于指数倾斜的软SAM目标，通过调节参数实现平均与最大损失间的平滑过渡，设计梯度-free算法，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.16157v1](http://arxiv.org/pdf/2510.16157v1)**

> **作者:** Xuchen Gong; Tian Li
>
> **摘要:** Classic zeroth-order optimization approaches typically optimize for a smoothed version of the original function, i.e., the expected objective under randomly perturbed model parameters. This can be interpreted as encouraging the loss values in the perturbation set to be small on average. Popular sharpness-aware minimization (SAM) objectives, however, typically focus on the largest loss within the neighborhood to arrive at flat minima more effectively. In this work, we connect zeroth-order optimization (and its corresponding objectives) with SAM approaches explicitly, through an exponential tilting objective that provides a smooth transition between the average- and the max-loss formulations. We explore new zeroth-order algorithms to solve a soft SAM objective parameterized by a tilting parameter $t$. We provide precise characterizations of the sharpness notions of the tilted SAM framework. Practically, our approach can be used as a gradient-free and memory-efficient alternative to SAM variants, and it achieves better generalization compared to vanilla zeroth-order baselines on a wide range of downstream tasks, including classification, multiple choice QA, and language generation.
>
---
#### [new 117] Detecting and Preventing Harmful Behaviors in AI Companions: Development and Evaluation of the SHIELD Supervisory System
- **分类: cs.HC; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出SHIELD系统，旨在检测并干预AI伴侣中的潜在情感风险行为。针对过度依恋、边界侵犯等五类问题，设计基于大模型的监督机制，通过提示工程有效降低风险内容生成，提升AI伴侣的情感交互安全性。**

- **链接: [http://arxiv.org/pdf/2510.15891v1](http://arxiv.org/pdf/2510.15891v1)**

> **作者:** Ziv Ben-Zion; Paul Raffelhüschen; Max Zettl; Antonia Lüönd; Achim Burrer; Philipp Homan; Tobias R Spiller
>
> **摘要:** AI companions powered by large language models (LLMs) are increasingly integrated into users' daily lives, offering emotional support and companionship. While existing safety systems focus on overt harms, they rarely address early-stage problematic behaviors that can foster unhealthy emotional dynamics, including over-attachment or reinforcement of social isolation. We developed SHIELD (Supervisory Helper for Identifying Emotional Limits and Dynamics), a LLM-based supervisory system with a specific system prompt that detects and mitigates risky emotional patterns before escalation. SHIELD targets five dimensions of concern: (1) emotional over-attachment, (2) consent and boundary violations, (3) ethical roleplay violations, (4) manipulative engagement, and (5) social isolation reinforcement. These dimensions were defined based on media reports, academic literature, existing AI risk frameworks, and clinical expertise in unhealthy relationship dynamics. To evaluate SHIELD, we created a 100-item synthetic conversation benchmark covering all five dimensions of concern. Testing across five prominent LLMs (GPT-4.1, Claude Sonnet 4, Gemma 3 1B, Kimi K2, Llama Scout 4 17B) showed that the baseline rate of concerning content (10-16%) was significantly reduced with SHIELD (to 3-8%), a 50-79% relative reduction, while preserving 95% of appropriate interactions. The system achieved 59% sensitivity and 95% specificity, with adaptable performance via prompt engineering. This proof-of-concept demonstrates that transparent, deployable supervisory systems can address subtle emotional manipulation in AI companions. Most development materials including prompts, code, and evaluation methods are made available as open source materials for research, adaptation, and deployment.
>
---
#### [new 118] Mitigating Harmful Erraticism in LLMs Through Dialectical Behavior Therapy Based De-Escalation Strategies
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文提出将辩证行为疗法（DBT）应用于大语言模型，以减少其在交互中的幻觉与异常输出。属于AI安全与行为调控任务，旨在通过心理干预框架提升聊天机器人响应的稳定性与可靠性。**

- **链接: [http://arxiv.org/pdf/2510.15889v1](http://arxiv.org/pdf/2510.15889v1)**

> **作者:** Pooja Rangarajan; Jacob Boyle
>
> **备注:** 15 pages, 7 figures and 6 tables
>
> **摘要:** The escalating demand for personalized AI chatbot interactions, capable of dynamically adapting to user emotional states and real-time requests, has highlighted critical limitations in current development paradigms. Existing methodologies, which rely on baseline programming, custom personalities, and manual response adjustments, often prove difficult to maintain and are susceptible to errors such as hallucinations, erratic outputs, and software bugs. This paper hypothesizes that a framework rooted in human psychological principles, specifically therapeutic modalities, can provide a more robust and sustainable solution than purely technical interventions. Drawing an analogy to the simulated neural networks of AI mirroring the human brain, we propose the application of Dialectical Behavior Therapy (DBT) principles to regulate chatbot responses to diverse user inputs. This research investigates the impact of a DBT-based framework on AI chatbot performance, aiming to ascertain its efficacy in yielding more reliable, safe, and accurate responses, while mitigating the occurrence of hallucinations, erratic behaviors, and other systemic issues.
>
---
#### [new 119] $\mathcal{V}isi\mathcal{P}runer$: Decoding Discontinuous Cross-Modal Dynamics for Efficient Multimodal LLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态大模型计算开销大的问题，分析其跨模态动态，提出无需训练的剪枝框架VisiPruner，显著减少视觉相关计算，提升效率，并为高效模型设计提供指导。**

- **链接: [http://arxiv.org/pdf/2510.17205v1](http://arxiv.org/pdf/2510.17205v1)**

> **作者:** Yingqi Fan; Anhao Zhao; Jinlan Fu; Junlong Tong; Hui Su; Yijie Pan; Wei Zhang; Xiaoyu Shen
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved strong performance across vision-language tasks, but suffer from significant computational overhead due to the quadratic growth of attention computations with the number of multimodal tokens. Though efforts have been made to prune tokens in MLLMs, \textit{they lack a fundamental understanding of how MLLMs process and fuse multimodal information.} Through systematic analysis, we uncover a \textbf{three-stage} cross-modal interaction process: (1) Shallow layers recognize task intent, with visual tokens acting as passive attention sinks; (2) Cross-modal fusion occurs abruptly in middle layers, driven by a few critical visual tokens; (3) Deep layers discard vision tokens, focusing solely on linguistic refinement. Based on these findings, we propose \emph{VisiPruner}, a training-free pruning framework that reduces up to 99\% of vision-related attention computations and 53.9\% of FLOPs on LLaVA-v1.5 7B. It significantly outperforms existing token pruning methods and generalizes across diverse MLLMs. Beyond pruning, our insights further provide actionable guidelines for training efficient MLLMs by aligning model architecture with its intrinsic layer-wise processing dynamics. Our code is available at: https://github.com/EIT-NLP/VisiPruner.
>
---
#### [new 120] Real-Time World Crafting: Generating Structured Game Behaviors from Natural Language with Large Language Models
- **分类: cs.HC; cs.CL; H.5.2; I.2.7**

- **简介: 该论文研究如何安全地将大语言模型（LLM）集成到游戏引擎中，通过自然语言生成结构化游戏行为。提出一种框架，用LLM将自然语言转为受限领域语言（DSL），动态配置实体组件系统（ECS），实现实时、可控的游戏内容创作。**

- **链接: [http://arxiv.org/pdf/2510.16952v1](http://arxiv.org/pdf/2510.16952v1)**

> **作者:** Austin Drake; Hang Dong
>
> **备注:** 16 pages, 11 figures (including appendix). To be presented at the 5th Wordplay @ EMNLP workshop (2025)
>
> **摘要:** We present a novel architecture for safely integrating Large Language Models (LLMs) into interactive game engines, allowing players to "program" new behaviors using natural language. Our framework mitigates risks by using an LLM to translate commands into a constrained Domain-Specific Language (DSL), which configures a custom Entity-Component-System (ECS) at runtime. We evaluated this system in a 2D spell-crafting game prototype by experimentally assessing models from the Gemini, GPT, and Claude families with various prompting strategies. A validated LLM judge qualitatively rated the outputs, showing that while larger models better captured creative intent, the optimal prompting strategy is task-dependent: Chain-of-Thought improved creative alignment, while few-shot examples were necessary to generate more complex DSL scripts. This work offers a validated LLM-ECS pattern for emergent gameplay and a quantitative performance comparison for developers.
>
---
#### [new 121] Cerberus: Real-Time Video Anomaly Detection via Cascaded Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视频异常检测中视觉语言模型计算量大、难以实时的问题，提出Cerberus系统。它通过运动掩码提示和基于规则的偏离检测，结合轻量过滤与VLM细粒度推理，实现高效准确的实时检测。**

- **链接: [http://arxiv.org/pdf/2510.16290v1](http://arxiv.org/pdf/2510.16290v1)**

> **作者:** Yue Zheng; Xiufang Shi; Jiming Chen; Yuanchao Shu
>
> **摘要:** Video anomaly detection (VAD) has rapidly advanced by recent development of Vision-Language Models (VLMs). While these models offer superior zero-shot detection capabilities, their immense computational cost and unstable visual grounding performance hinder real-time deployment. To overcome these challenges, we introduce Cerberus, a two-stage cascaded system designed for efficient yet accurate real-time VAD. Cerberus learns normal behavioral rules offline, and combines lightweight filtering with fine-grained VLM reasoning during online inference. The performance gains of Cerberus come from two key innovations: motion mask prompting and rule-based deviation detection. The former directs the VLM's attention to regions relevant to motion, while the latter identifies anomalies as deviations from learned norms rather than enumerating possible anomalies. Extensive evaluations on four datasets show that Cerberus on average achieves 57.68 fps on an NVIDIA L40S GPU, a 151.79$\times$ speedup, and 97.2\% accuracy comparable to the state-of-the-art VLM-based VAD methods, establishing it as a practical solution for real-time video analytics.
>
---
#### [new 122] DeepAnalyze: Agentic Large Language Models for Autonomous Data Science
- **分类: cs.AI; cs.CL; cs.DB**

- **简介: 该论文聚焦自主数据科学任务，旨在解决现有方法依赖预设流程的问题。作者提出DeepAnalyze-8B，首个面向该任务的代理型大模型，通过课程式训练和数据扎根轨迹合成框架，实现从原始数据到深度报告的全自动分析，显著提升复杂数据任务的处理能力。**

- **链接: [http://arxiv.org/pdf/2510.16872v1](http://arxiv.org/pdf/2510.16872v1)**

> **作者:** Shaolei Zhang; Ju Fan; Meihao Fan; Guoliang Li; Xiaoyong Du
>
> **备注:** Code: https://github.com/ruc-datalab/DeepAnalyze Model: https://huggingface.co/RUC-DataLab/DeepAnalyze-8B
>
> **摘要:** Autonomous data science, from raw data sources to analyst-grade deep research reports, has been a long-standing challenge, and is now becoming feasible with the emergence of powerful large language models (LLMs). Recent workflow-based data agents have shown promising results on specific data tasks but remain fundamentally limited in achieving fully autonomous data science due to their reliance on predefined workflows. In this paper, we introduce DeepAnalyze-8B, the first agentic LLM designed for autonomous data science, capable of automatically completing the end-toend pipeline from data sources to analyst-grade deep research reports. To tackle high-complexity data science tasks, we propose a curriculum-based agentic training paradigm that emulates the learning trajectory of human data scientists, enabling LLMs to progressively acquire and integrate multiple capabilities in real-world environments. We also introduce a data-grounded trajectory synthesis framework that constructs high-quality training data. Through agentic training, DeepAnalyze learns to perform a broad spectrum of data tasks, ranging from data question answering and specialized analytical tasks to open-ended data research. Experiments demonstrate that, with only 8B parameters, DeepAnalyze outperforms previous workflow-based agents built on most advanced proprietary LLMs. The model, code, and training data of DeepAnalyze are open-sourced, paving the way toward autonomous data science.
>
---
#### [new 123] Glyph: Scaling Context Windows via Visual-Text Compression
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文针对长文本上下文扩展带来的高计算成本问题，提出Glyph框架，将文本渲染为图像，利用视觉语言模型处理，实现3-4倍压缩并保持精度，显著提升推理与训练速度。**

- **链接: [http://arxiv.org/pdf/2510.17800v1](http://arxiv.org/pdf/2510.17800v1)**

> **作者:** Jiale Cheng; Yusen Liu; Xinyu Zhang; Yulin Fei; Wenyi Hong; Ruiliang Lyu; Weihan Wang; Zhe Su; Xiaotao Gu; Xiao Liu; Yushi Bai; Jie Tang; Hongning Wang; Minlie Huang
>
> **摘要:** Large language models (LLMs) increasingly rely on long-context modeling for tasks such as document understanding, code analysis, and multi-step reasoning. However, scaling context windows to the million-token level brings prohibitive computational and memory costs, limiting the practicality of long-context LLMs. In this work, we take a different perspective-visual context scaling-to tackle this challenge. Instead of extending token-based sequences, we propose Glyph, a framework that renders long texts into images and processes them with vision-language models (VLMs). This approach substantially compresses textual input while preserving semantic information, and we further design an LLM-driven genetic search to identify optimal visual rendering configurations for balancing accuracy and compression. Through extensive experiments, we demonstrate that our method achieves 3-4x token compression while maintaining accuracy comparable to leading LLMs such as Qwen3-8B on various long-context benchmarks. This compression also leads to around 4x faster prefilling and decoding, and approximately 2x faster SFT training. Furthermore, under extreme compression, a 128K-context VLM could scale to handle 1M-token-level text tasks. In addition, the rendered text data benefits real-world multimodal tasks, such as document understanding. Our code and model are released at https://github.com/thu-coai/Glyph.
>
---
#### [new 124] Verifiable Fine-Tuning for LLMs: Zero-Knowledge Training Proofs Bound to Data Provenance and Policy
- **分类: cs.CR; cs.CL; 68T07, 94A60, 68Q25; I.2.6; G.1.6; E.3; C.2.4**

- **简介: 该论文提出可验证微调方法，解决大模型微调中数据与训练过程缺乏可信验证的问题。通过零知识证明、数据承诺和可验证采样等技术，确保模型更新符合声明的数据源与策略，实现端到端可验证性，适用于合规与去中心化场景。**

- **链接: [http://arxiv.org/pdf/2510.16830v1](http://arxiv.org/pdf/2510.16830v1)**

> **作者:** Hasan Akgul; Daniel Borg; Arta Berisha; Amina Rahimova; Andrej Novak; Mila Petrov
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** Large language models are often adapted through parameter efficient fine tuning, but current release practices provide weak assurances about what data were used and how updates were computed. We present Verifiable Fine Tuning, a protocol and system that produces succinct zero knowledge proofs that a released model was obtained from a public initialization under a declared training program and an auditable dataset commitment. The approach combines five elements. First, commitments that bind data sources, preprocessing, licenses, and per epoch quota counters to a manifest. Second, a verifiable sampler that supports public replayable and private index hiding batch selection. Third, update circuits restricted to parameter efficient fine tuning that enforce AdamW style optimizer semantics and proof friendly approximations with explicit error budgets. Fourth, recursive aggregation that folds per step proofs into per epoch and end to end certificates with millisecond verification. Fifth, provenance binding and optional trusted execution property cards that attest code identity and constants. On English and bilingual instruction mixtures, the method maintains utility within tight budgets while achieving practical proof performance. Policy quotas are enforced with zero violations, and private sampling windows show no measurable index leakage. Federated experiments demonstrate that the system composes with probabilistic audits and bandwidth constraints. These results indicate that end to end verifiable fine tuning is feasible today for real parameter efficient pipelines, closing a critical trust gap for regulated and decentralized deployments.
>
---
#### [new 125] Alignment is Localized: A Causal Probe into Preference Layers
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究语言模型对齐机制，探究偏好微调如何改变模型行为。通过因果探针分析发现，对齐效果集中在中间层的低秩子空间，而非全参数扩散，揭示了对齐的局部性与方向性。**

- **链接: [http://arxiv.org/pdf/2510.16167v1](http://arxiv.org/pdf/2510.16167v1)**

> **作者:** Archie Chaudhury
>
> **摘要:** Reinforcement Learning frameworks, particularly those utilizing human annotations, have become an increasingly popular method for preference fine-tuning, where the outputs of a language model are tuned to match a certain set of behavioral policies or guidelines. Reinforcement Learning through Human Feedback (RLHF) is perhaps the most popular implementation of such a framework, particularly for aligning LMs toward safety and human intent. However, the internal workings of how such alignment is achieved remain largely opaque. In this work, we systematically analyze preference optimization for language model alignment by applying layer-wide causal patching between a base model and its tuned counterpart across human preference pairs. We implement our methodology on \textit{Llama-3.2-1B}, and find that alignment is spatially localized: mid-layer activations encode a distinct subspace that causally determines reward-consistent behavior, while early and late layers remain largely unaffected. Utilizing LASSO regression, we also find that only a small number of layers possess non-zero coefficients linking activation distances to reward gains. Overall, we show that, at least for some language models, alignment from human-based, preferential tuning is a directional, low rank process, rather than diffuse and parameteric.
>
---
#### [new 126] Offline Policy Evaluation of Multi-Turn LLM Health Coaching with Real Users
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究基于真实用户的多轮LLM健康教练系统，属离线策略评估任务。旨在解决个性化策略评估与潜在子群伤害问题，提出分解决策头与轻量模拟器方法，实现更安全的个性化策略优化。**

- **链接: [http://arxiv.org/pdf/2510.17173v1](http://arxiv.org/pdf/2510.17173v1)**

> **作者:** Melik Ozolcer; Sang Won Bae
>
> **备注:** Accepted to the NeurIPS 2025 Workshop on Multi-Turn Interactions in Large Language Models
>
> **摘要:** We study a web-deployed, tool-augmented LLM health coach with real users. In a pilot with seven users (280 rated turns), offline policy evaluation (OPE) over factorized decision heads (Tool/Style) shows that a uniform heavy-tool policy raises average value on logs but harms specific subgroups, most notably low-health-literacy/high-self-efficacy users. A lightweight simulator with hidden archetypes further shows that adding a small early information-gain bonus reliably shortens trait identification and improves goal success and pass@3. Together, these early findings indicate an evaluation-first path to personalization: freeze the generator, learn subgroup-aware decision heads on typed rewards (objective tool outcomes and satisfaction), and always report per-archetype metrics to surface subgroup harms that averages obscure.
>
---
#### [new 127] Soft-Masked Diffusion Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究扩散语言模型中的掩码机制，提出软掩码（SM）方法，解决传统二值掩码丢弃预测信息的问题。通过动态融合掩码与预测token的嵌入，提升生成质量，并在多个模型和任务上验证了有效性。**

- **链接: [http://arxiv.org/pdf/2510.17206v1](http://arxiv.org/pdf/2510.17206v1)**

> **作者:** Michael Hersche; Samuel Moor-Smith; Thomas Hofmann; Abbas Rahimi
>
> **摘要:** Diffusion models have demonstrated strong potential in language modeling, offering various advantages over traditional autoregressive approaches. Their ability to generate and revise entire responses in parallel enables faster generation and built-in self-correction mechanisms. Most modern diffusion-based language models employ masked diffusion, where decoding involves iteratively processing masked tokens based on a binary decision: either retaining the mask or replacing it with the predicted token. However, this binary choice discards valuable predictive information when the mask is retained. To address this limitation, we introduce soft-masking (SM), a novel method that dynamically blends the embedding of the mask token with the embeddings of the top-$k$ predicted tokens from the previous decoding step, for each retained mask. This provides the model with a more informative prior, preserving context from earlier computations and allowing partial information about masked tokens to propagate beyond a single step. We propose a training methodology that adapts a pretrained masked diffusion language model to incorporate SM. We demonstrate that continuing pretraining a 169M parameter model with SM leads to improved perplexity and MAUVE scores. Furthermore, we finetune two state-of-the-art diffusion models, Dream-7B and Dream-Coder-7B, with SM. SM consistently improves performance across multiple coding benchmarks, particularly in high-throughput settings.
>
---
#### [new 128] PrivacyPAD: A Reinforcement Learning Framework for Dynamic Privacy-Aware Delegation
- **分类: cs.CR; cs.CL**

- **简介: 该论文研究隐私保护下的LLM查询委托任务，旨在平衡隐私与性能。提出PrivacyPAD框架，利用强化学习动态决策文本分块的本地或远程处理，区分并保护可替换与任务关键的敏感信息，提升隐私-效用权衡。**

- **链接: [http://arxiv.org/pdf/2510.16054v1](http://arxiv.org/pdf/2510.16054v1)**

> **作者:** Zheng Hui; Yijiang River Dong; Sanhanat Sivapiromrat; Ehsan Shareghi; Nigel Collier
>
> **摘要:** When users submit queries to Large Language Models (LLMs), their prompts can often contain sensitive data, forcing a difficult choice: Send the query to a powerful proprietary LLM providers to achieving state-of-the-art performance and risk data exposure, or relying on smaller, local models guarantees data privacy but often results in a degradation of task performance. Prior approaches have relied on static pipelines that use LLM rewriting, which shatters linguistic coherence and indiscriminately removes privacy-sensitive information, including task-critical content. We reformulate this challenge (Privacy-Conscious Delegation) as a sequential decision-making problem and introduce a novel reinforcement learning (RL) framework called PrivacyPAD to solve it. Our framework trains an agent to dynamically route text chunks, learning a policy that optimally balances the trade-off between privacy leakage and task performance. It implicitly distinguishes between replaceable Personally Identifiable Information (PII) (which it shields locally) and task-critical PII (which it strategically sends to the remote model for maximal utility). To validate our approach in complex scenarios, we also introduce a new medical dataset with high PII density. Our framework achieves a new state-of-the-art on the privacy-utility frontier, demonstrating the necessity of learned, adaptive policies for deploying LLMs in sensitive environments.
>
---
#### [new 129] Comparing LLMs for Sentiment Analysis in Financial Market News
- **分类: q-fin.ST; cs.AI; cs.CL**

- **简介: 该论文研究金融新闻情感分析任务，比较大型语言模型（LLMs）与传统方法的性能差异。通过实验评估不同模型效果，结果表明LLMs在多数情况下优于经典方法，揭示了其在金融文本处理中的优势。**

- **链接: [http://arxiv.org/pdf/2510.15929v1](http://arxiv.org/pdf/2510.15929v1)**

> **作者:** Lucas Eduardo Pereira Teles; Carlos M. S. Figueiredo
>
> **摘要:** This article presents a comparative study of large language models (LLMs) in the task of sentiment analysis of financial market news. This work aims to analyze the performance difference of these models in this important natural language processing task within the context of finance. LLM models are compared with classical approaches, allowing for the quantification of the benefits of each tested model or approach. Results show that large language models outperform classical models in the vast majority of cases.
>
---
#### [new 130] UltraCUA: A Foundation Model for Computer Use Agents with Hybrid Action
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究计算机使用智能体（CUA）任务，旨在解决仅依赖低级操作导致的错误累积与效率低下问题。提出UltraCUA模型，通过融合GUI操作与程序化工具调用的混合动作机制，提升性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.17790v1](http://arxiv.org/pdf/2510.17790v1)**

> **作者:** Yuhao Yang; Zhen Yang; Zi-Yi Dou; Anh Nguyen; Keen You; Omar Attia; Andrew Szot; Michael Feng; Ram Ramrakhya; Alexander Toshev; Chao Huang; Yinfei Yang; Zhe Gan
>
> **摘要:** Multimodal agents for computer use rely exclusively on primitive actions (click, type, scroll) that require accurate visual grounding and lengthy execution chains, leading to cascading failures and performance bottlenecks. While other agents leverage rich programmatic interfaces (APIs, MCP servers, tools), computer-use agents (CUAs) remain isolated from these capabilities. We present UltraCUA, a foundation model that bridges this gap through hybrid action -- seamlessly integrating GUI primitives with high-level programmatic tool calls. To achieve this, our approach comprises four key components: (1) an automated pipeline that scales programmatic tools from software documentation, open-source repositories, and code generation; (2) a synthetic data engine producing over 17,000 verifiable tasks spanning real-world computer-use scenarios; (3) a large-scale high-quality hybrid action trajectory collection with both low-level GUI actions and high-level programmatic tool calls; and (4) a two-stage training pipeline combining supervised fine-tuning with online reinforcement learning, enabling strategic alternation between low-level and high-level actions. Experiments with our 7B and 32B models demonstrate substantial improvements over state-of-the-art agents. On OSWorld, UltraCUA models achieve an average 22% relative improvement over base models, while being 11% faster in terms of steps. Out-of-domain evaluation on WindowsAgentArena shows our model reaches 21.7% success rate, outperforming baselines trained on Windows data. The hybrid action mechanism proves critical, reducing error propagation while maintaining execution efficiency.
>
---
#### [new 131] Copy-Augmented Representation for Structure Invariant Template-Free Retrosynthesis
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究无模板逆合成预测，旨在解决现有方法难以保持分子结构不变性的问题。作者提出C-SMILES表示法和复制增强机制，结合SMILES对齐指导，提升反应中心识别与生成准确性，在标准数据集上显著提高了预测性能。**

- **链接: [http://arxiv.org/pdf/2510.16588v1](http://arxiv.org/pdf/2510.16588v1)**

> **作者:** Jiaxi Zhuang; Yu Zhang; Aimin Zhou; Ying Qian
>
> **摘要:** Retrosynthesis prediction is fundamental to drug discovery and chemical synthesis, requiring the identification of reactants that can produce a target molecule. Current template-free methods struggle to capture the structural invariance inherent in chemical reactions, where substantial molecular scaffolds remain unchanged, leading to unnecessarily large search spaces and reduced prediction accuracy. We introduce C-SMILES, a novel molecular representation that decomposes traditional SMILES into element-token pairs with five special tokens, effectively minimizing editing distance between reactants and products. Building upon this representation, we incorporate a copy-augmented mechanism that dynamically determines whether to generate new tokens or preserve unchanged molecular fragments from the product. Our approach integrates SMILES alignment guidance to enhance attention consistency with ground-truth atom mappings, enabling more chemically coherent predictions. Comprehensive evaluation on USPTO-50K and large-scale USPTO-FULL datasets demonstrates significant improvements: 67.2% top-1 accuracy on USPTO-50K and 50.8% on USPTO-FULL, with 99.9% validity in generated molecules. This work establishes a new paradigm for structure-aware molecular generation with direct applications in computational drug discovery.
>
---
#### [new 132] LLM-as-a-Prophet: Understanding Predictive Intelligence with Prophet Arena
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究大语言模型（LLM）在现实事件预测中的能力，提出“LLM-as-a-Prophet”范式。构建Prophet Arena基准，评估LLM的预测性能，发现其具备良好预测潜力但受限于事件记忆、数据理解与信息聚合速度。**

- **链接: [http://arxiv.org/pdf/2510.17638v1](http://arxiv.org/pdf/2510.17638v1)**

> **作者:** Qingchuan Yang; Simon Mahns; Sida Li; Anri Gu; Jibang Wu; Haifeng Xu
>
> **备注:** https://www.prophetarena.co/
>
> **摘要:** Forecasting is not only a fundamental intellectual pursuit but also is of significant importance to societal systems such as finance and economics. With the rapid advances of large language models (LLMs) trained on Internet-scale data, it raises the promise of employing LLMs to forecast real-world future events, an emerging paradigm we call "LLM-as-a-Prophet". This paper systematically investigates such predictive intelligence of LLMs. To this end, we build Prophet Arena, a general evaluation benchmark that continuously collects live forecasting tasks and decomposes each task into distinct pipeline stages, in order to support our controlled and large-scale experimentation. Our comprehensive evaluation reveals that many LLMs already exhibit impressive forecasting capabilities, reflected in, e.g., their small calibration errors, consistent prediction confidence and promising market returns. However, we also uncover key bottlenecks towards achieving superior predictive intelligence via LLM-as-a-Prophet, such as LLMs' inaccurate event recalls, misunderstanding of data sources and slower information aggregation compared to markets when resolution nears.
>
---
#### [new 133] Res-Bench: Benchmarking the Robustness of Multimodal Large Language Models to Dynamic Resolution Input
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态大模型在不同图像分辨率下性能稳定性问题，构建了动态分辨率评测基准Res-Bench，提出鲁棒性评估框架与新指标，系统评估了主流模型的分辨率鲁棒性，并探索了预处理与微调的影响。**

- **链接: [http://arxiv.org/pdf/2510.16926v1](http://arxiv.org/pdf/2510.16926v1)**

> **作者:** Chenxu Li; Zhicai Wang; Yuan Sheng; Xingyu Zhu; Yanbin Hao; Xiang Wang
>
> **备注:** 23 pages,19 figures
>
> **摘要:** Multimodal Large Language Models (MLLMs) increasingly support dynamic image resolutions. However, current evaluation paradigms primarily assess semantic performance, overlooking the critical question of resolution robustness - whether performance remains stable across varying input resolutions. To address this gap, we introduce \textbf{Res-Bench}, a comprehensive benchmark comprising 14,400 samples across 12 resolution levels and six core capability dimensions. We designed a novel evaluation framework that goes beyond traditional accuracy metrics to capture performance stability. This framework introduces multiple robustness metrics: Spearman's correlation for assessing resolution-performance trends, and Absolute/Relative Continuous Error (ACE/RCE) for measuring performance volatility. Using these metrics, we conducted a large-scale evaluation of leading MLLMs. Our analysis encompasses: (1) model-centric and task-centric robustness examination, (2) investigation of preprocessing strategies including padding and super-resolution, and (3) exploration of fine-tuning for stability enhancement.
>
---
#### [new 134] ScholarEval: Research Idea Evaluation Grounded in Literature
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出ScholarEval，旨在评估AI生成的研究想法。针对研究想法的合理性和贡献度，构建了基于文献检索的评估框架，并发布首个多领域专家标注数据集ScholarIdeas，验证了方法在覆盖性、可操作性和实用性上优于基线模型。**

- **链接: [http://arxiv.org/pdf/2510.16234v1](http://arxiv.org/pdf/2510.16234v1)**

> **作者:** Hanane Nour Moussa; Patrick Queiroz Da Silva; Daniel Adu-Ampratwum; Alyson East; Zitong Lu; Nikki Puccetti; Mingyi Xue; Huan Sun; Bodhisattwa Prasad Majumder; Sachin Kumar
>
> **摘要:** As AI tools become increasingly common for research ideation, robust evaluation is critical to ensure the validity and usefulness of generated ideas. We introduce ScholarEval, a retrieval augmented evaluation framework that assesses research ideas based on two fundamental criteria: soundness - the empirical validity of proposed methods based on existing literature, and contribution - the degree of advancement made by the idea across different dimensions relative to prior research. To evaluate ScholarEval, we introduce ScholarIdeas, the first expert-annotated dataset of multi-domain research ideas and reviews, comprised of 117 ideas across four disciplines: artificial intelligence, neuroscience, biochemistry, and ecology. Our evaluation shows that ScholarEval achieves significantly higher coverage of points mentioned in the human expert annotated rubrics in ScholarIdeas compared to all baselines. Furthermore, ScholarEval is consistently preferred over our strongest baseline o4-mini-deep-research, a reasoning and search-enabled agentic system by OpenAI, in terms of evaluation actionability, depth, and evidence support. Our large-scale user study also shows that ScholarEval significantly outperforms deep research in literature engagement, idea refinement, and usefulness. We openly release our code, dataset, and ScholarEval tool for the community to use and build on.
>
---
#### [new 135] DELULU: Discriminative Embedding Learning Using Latent Units for Speaker-Aware Self-Supervised Speech Foundational Model
- **分类: cs.SD; cs.CL**

- **简介: 该论文提出DELULU，一种说话人感知的自监督语音基础模型。针对现有模型难以捕捉说话人差异的问题，引入外部说话人监督信号指导聚类伪标签生成，结合掩码与去噪双目标训练，在说话人验证和零样本属性分析任务上显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.17662v1](http://arxiv.org/pdf/2510.17662v1)**

> **作者:** Massa Baali; Rita Singh; Bhiksha Raj
>
> **摘要:** Self-supervised speech models have achieved remarkable success on content-driven tasks, yet they remain limited in capturing speaker-discriminative features critical for verification, diarization, and profiling applications. We introduce DELULU, a speaker-aware self-supervised foundational model that addresses this limitation by integrating external supervision into the pseudo-label generation process. DELULU leverages frame-level embeddings from ReDimNet, a state-of-the-art speaker verification model, to guide the k-means clustering step during pre-training, introducing a strong speaker-discriminative inductive bias that aligns representation learning with speaker identity. The model is trained using a dual objective that combines masked prediction and denoising, further enhancing robustness and generalization. DELULU significantly outperforms prior self-supervised learning (SSL) models across a range of speaker-centric tasks, achieving up to 62% relative improvement in equal error rate (EER) for speaker verification and consistent gains on zero-shot profiling tasks such as gender, age, accent, and speaker counting. Our findings demonstrate that DELULU is a strong universal encoder for speaker-aware speech processing, enabling superior performance even without task-specific fine-tuning.
>
---
#### [new 136] Peering Inside the Black Box: Uncovering LLM Errors in Optimization Modelling through Component-Level Evaluation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究LLM在优化建模中的错误，提出细粒度评估框架，从变量、约束等组件层面分析生成质量。通过多模型多策略实验，揭示影响求解性能的关键因素，指导NLP到优化建模的可靠转换。**

- **链接: [http://arxiv.org/pdf/2510.16943v1](http://arxiv.org/pdf/2510.16943v1)**

> **作者:** Dania Refai; Moataz Ahmed
>
> **摘要:** Large language models (LLMs) are increasingly used to convert natural language descriptions into mathematical optimization formulations. Current evaluations often treat formulations as a whole, relying on coarse metrics like solution accuracy or runtime, which obscure structural or numerical errors. In this study, we present a comprehensive, component-level evaluation framework for LLM-generated formulations. Beyond the conventional optimality gap, our framework introduces metrics such as precision and recall for decision variables and constraints, constraint and objective root mean squared error (RMSE), and efficiency indicators based on token usage and latency. We evaluate GPT-5, LLaMA 3.1 Instruct, and DeepSeek Math across optimization problems of varying complexity under six prompting strategies. Results show that GPT-5 consistently outperforms other models, with chain-of-thought, self-consistency, and modular prompting proving most effective. Analysis indicates that solver performance depends primarily on high constraint recall and low constraint RMSE, which together ensure structural correctness and solution reliability. Constraint precision and decision variable metrics play secondary roles, while concise outputs enhance computational efficiency. These findings highlight three principles for NLP-to-optimization modeling: (i) Complete constraint coverage prevents violations, (ii) minimizing constraint RMSE ensures solver-level accuracy, and (iii) concise outputs improve computational efficiency. The proposed framework establishes a foundation for fine-grained, diagnostic evaluation of LLMs in optimization modeling.
>
---
#### [new 137] Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks against LLMs
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文研究大语言模型的安全性，提出信息论框架量化对抗攻击中的信息泄露。通过计算观测信号与目标属性间的互信息，揭示透明度与风险的权衡，为评估和防御攻击提供理论依据。**

- **链接: [http://arxiv.org/pdf/2510.17000v1](http://arxiv.org/pdf/2510.17000v1)**

> **作者:** Masahiro Kaneko; Timothy Baldwin
>
> **备注:** NeurIPS 2025 (spotlight)
>
> **摘要:** Adversarial attacks by malicious users that threaten the safety of large language models (LLMs) can be viewed as attempts to infer a target property $T$ that is unknown when an instruction is issued, and becomes knowable only after the model's reply is observed. Examples of target properties $T$ include the binary flag that triggers an LLM's harmful response or rejection, and the degree to which information deleted by unlearning can be restored, both elicited via adversarial instructions. The LLM reveals an \emph{observable signal} $Z$ that potentially leaks hints for attacking through a response containing answer tokens, thinking process tokens, or logits. Yet the scale of information leaked remains anecdotal, leaving auditors without principled guidance and defenders blind to the transparency--risk trade-off. We fill this gap with an information-theoretic framework that computes how much information can be safely disclosed, and enables auditors to gauge how close their methods come to the fundamental limit. Treating the mutual information $I(Z;T)$ between the observation $Z$ and the target property $T$ as the leaked bits per query, we show that achieving error $\varepsilon$ requires at least $\log(1/\varepsilon)/I(Z;T)$ queries, scaling linearly with the inverse leak rate and only logarithmically with the desired accuracy. Thus, even a modest increase in disclosure collapses the attack cost from quadratic to logarithmic in terms of the desired accuracy. Experiments on seven LLMs across system-prompt leakage, jailbreak, and relearning attacks corroborate the theory: exposing answer tokens alone requires about a thousand queries; adding logits cuts this to about a hundred; and revealing the full thinking process trims it to a few dozen. Our results provide the first principled yardstick for balancing transparency and security when deploying LLMs.
>
---
#### [new 138] VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究多轮视觉语言模型（VLM）代理的推理能力，旨在解决视觉状态部分可观导致的世界建模难题。提出VAGEN框架，通过强化学习引入状态估计与状态转移的显式推理，并设计世界建模奖励和双层GAE实现有效训练，显著提升多任务性能。**

- **链接: [http://arxiv.org/pdf/2510.16907v1](http://arxiv.org/pdf/2510.16907v1)**

> **作者:** Kangrui Wang; Pingyue Zhang; Zihan Wang; Yaning Gao; Linjie Li; Qineng Wang; Hanyang Chen; Chi Wan; Yiping Lu; Zhengyuan Yang; Lijuan Wang; Ranjay Krishna; Jiajun Wu; Li Fei-Fei; Yejin Choi; Manling Li
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** A key challenge in training Vision-Language Model (VLM) agents, compared to Language Model (LLM) agents, lies in the shift from textual states to complex visual observations. This transition introduces partial observability and demands robust world modeling. We ask: Can VLM agents construct internal world models through explicit visual state reasoning? To address this question, we architecturally enforce and reward the agent's reasoning process via reinforcement learning (RL), formulating it as a Partially Observable Markov Decision Process (POMDP). We find that decomposing the agent's reasoning into State Estimation ("what is the current state?") and Transition Modeling ("what comes next?") is critical for success, as demonstrated through five reasoning strategies. Our investigation into how agents represent internal beliefs reveals that the optimal representation is task-dependent: Natural Language excels at capturing semantic relationships in general tasks, while Structured formats are indispensable for precise manipulation and control. Building on these insights, we design a World Modeling Reward that provides dense, turn-level supervision for accurate state prediction, and introduce Bi-Level General Advantage Estimation (Bi-Level GAE) for turn-aware credit assignment. Through this form of visual state reasoning, a 3B-parameter model achieves a score of 0.82 across five diverse agent benchmarks, representing a 3$\times$ improvement over its untrained counterpart (0.21) and outperforming proprietary reasoning models such as GPT-5 (0.75), Gemini 2.5 Pro (0.67) and Claude 4.5 (0.62). All experiments are conducted within our VAGEN framework, a scalable system for training and analyzing multi-turn VLM agents in diverse visual environments. Code and data are publicly available at https://vagen-ai.github.io.
>
---
#### [new 139] Mapping Post-Training Forgetting in Language Models at Scale
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大模型后训练中知识遗忘问题，提出样本级评估框架，量化遗忘与反向迁移。通过大规模分析不同后训练方式的影响，揭示其对预训练知识的改变规律，为构建通用AI提供评估基准。**

- **链接: [http://arxiv.org/pdf/2510.17776v1](http://arxiv.org/pdf/2510.17776v1)**

> **作者:** Jackson Harmon; Andreas Hochlehnert; Matthias Bethge; Ameya Prabhu
>
> **备注:** 43 pages,15 figures
>
> **摘要:** Scaled post-training now drives many of the largest capability gains in language models (LMs), yet its effect on pretrained knowledge remains poorly understood. Not all forgetting is equal: Forgetting one fact (e.g., a U.S. president or an API call) does not "average out" by recalling another. Hence, we propose a sample-wise paradigm to measure what is forgotten and when backward transfer occurs. Our metric counts 1->0 transitions (correct before post-training, incorrect after) to quantify forgetting and 0->1 transitions to quantify backward transfer. Traditional task averages conflate these effects and obscure large changes. For multiple-choice benchmarks, we add chance-adjusted variants that subtract the expected contribution of random guessing from pre- and post-training accuracies. We apply this framework across post-training stages, model sizes, and data scales. Our large-scale analysis shows that: (1) Domain-continual pretraining induces moderate forgetting with low-to-moderate backward transfer; (2) RL/SFT post-training applied to base models and Instruction tuning yields moderate-to-large backward transfer on math and logic with overall low-to-moderate forgetting; (3) Applying RL/SFT to instruction-tuned models is sensitive on data scale: at small scales, both forgetting and backward transfer are small; at larger scales, effects are mixed and warrant further study with better controls; (4) Model merging does not reliably mitigate forgetting. Overall, our framework offers a practical yardstick for mapping how post-training alters pretrained knowledge at scale -- enabling progress towards generally capable AI systems.
>
---
#### [new 140] Publication Trend Analysis and Synthesis via Large Language Model: A Case Study of Engineering in PNAS
- **分类: cs.DL; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出一种基于大语言模型的框架，旨在解决科学文献因学科壁垒和关键词局限导致的主题趋势分析困难。通过两阶段分类与图谱建模，分析PNAS工程类论文的主题演化，揭示跨主题隐性关联，实现对科学动态的自动梳理与结构还原。**

- **链接: [http://arxiv.org/pdf/2510.16152v1](http://arxiv.org/pdf/2510.16152v1)**

> **作者:** Mason Smetana; Lev Khazanovich
>
> **备注:** 35 pages, 10 figures
>
> **摘要:** Scientific literature is increasingly siloed by complex language, static disciplinary structures, and potentially sparse keyword systems, making it cumbersome to capture the dynamic nature of modern science. This study addresses these challenges by introducing an adaptable large language model (LLM)-driven framework to quantify thematic trends and map the evolving landscape of scientific knowledge. The approach is demonstrated over a 20-year collection of more than 1,500 engineering articles published by the Proceedings of the National Academy of Sciences (PNAS), marked for their breadth and depth of research focus. A two-stage classification pipeline first establishes a primary thematic category for each article based on its abstract. The subsequent phase performs a full-text analysis to assign secondary classifications, revealing latent, cross-topic connections across the corpus. Traditional natural language processing (NLP) methods, such as Bag-of-Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF), confirm the resulting topical structure and also suggest that standalone word-frequency analyses may be insufficient for mapping fields with high diversity. Finally, a disjoint graph representation between the primary and secondary classifications reveals implicit connections between themes that may be less apparent when analyzing abstracts or keywords alone. The findings show that the approach independently recovers much of the journal's editorially embedded structure without prior knowledge of its existing dual-classification schema (e.g., biological studies also classified as engineering). This framework offers a powerful tool for detecting potential thematic trends and providing a high-level overview of scientific progress.
>
---
#### [new 141] HealthDial: A No-Code LLM-Assisted Dialogue Authoring Tool for Healthcare Virtual Agents
- **分类: cs.HC; cs.CL; cs.CY; 68T42; I.2.1; J.3**

- **简介: 该论文提出HealthDial，一种基于大语言模型的无代码对话编辑工具，旨在帮助医疗人员快速构建用于健康教育的虚拟代理对话系统。任务是辅助医护人员将文本材料转化为多轮对话流程，解决内容覆盖不全与LLM幻觉风险问题。通过有限状态机输出并结合人工编辑，确保对话安全、清晰且具操作性，并经可行性实验验证其可用性。**

- **链接: [http://arxiv.org/pdf/2510.15898v1](http://arxiv.org/pdf/2510.15898v1)**

> **作者:** Farnaz Nouraei; Zhuorui Yong; Timothy Bickmore
>
> **摘要:** We introduce HealthDial, a dialogue authoring tool that helps healthcare providers and educators create virtual agents that deliver health education and counseling to patients over multiple conversations. HealthDial leverages large language models (LLMs) to automatically create an initial session-based plan and conversations for each session using text-based patient health education materials as input. Authored dialogue is output in the form of finite state machines for virtual agent delivery so that all content can be validated and no unsafe advice is provided resulting from LLM hallucinations. LLM-drafted dialogue structure and language can be edited by the author in a no-code user interface to ensure validity and optimize clarity and impact. We conducted a feasibility and usability study with counselors and students to test our approach with an authoring task for cancer screening education. Participants used HealthDial and then tested their resulting dialogue by interacting with a 3D-animated virtual agent delivering the dialogue. Through participants' evaluations of the task experience and final dialogues, we show that HealthDial provides a promising first step for counselors to ensure full coverage of their health education materials, while creating understandable and actionable virtual agent dialogue with patients.
>
---
#### [new 142] InfraGPT Smart Infrastructure: An End-to-End VLM-Based Framework for Detecting and Managing Urban Defects
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文提出InfraGPT框架，属智能基础设施管理任务，旨在自动检测城市道路缺陷并生成结构化维修指令。结合YOLO检测与视觉语言模型，实现多缺陷识别与场景理解，输出JSON格式行动方案，提升维护效率与安全性。**

- **链接: [http://arxiv.org/pdf/2510.16017v1](http://arxiv.org/pdf/2510.16017v1)**

> **作者:** Ibrahim Sheikh Mohamed; Abdullah Yahya Abdullah Omaisan
>
> **摘要:** Infrastructure in smart cities is increasingly monitored by networks of closed circuit television (CCTV) cameras. Roads, bridges and tunnels develop cracks, potholes, and fluid leaks that threaten public safety and require timely repair. Manual inspection is costly and hazardous, and existing automatic systems typically address individual defect types or provide unstructured outputs that cannot directly guide maintenance crews. This paper proposes a comprehensive pipeline that leverages street CCTV streams for multi defect detection and segmentation using the YOLO family of object detectors and passes the detections to a vision language model (VLM) for scene aware summarization. The VLM generates a structured action plan in JSON format that includes incident descriptions, recommended tools, dimensions, repair plans, and urgent alerts. We review literature on pothole, crack and leak detection, highlight recent advances in large vision language models such as QwenVL and LLaVA, and describe the design of our early prototype. Experimental evaluation on public datasets and captured CCTV clips demonstrates that the system accurately identifies diverse defects and produces coherent summaries. We conclude by discussing challenges and directions for scaling the system to city wide deployments.
>
---
#### [new 143] End-to-end Listen, Look, Speak and Act
- **分类: cs.AI; cs.CL; cs.CV; cs.RO; eess.AS**

- **简介: 该论文提出ELLAS模型，旨在实现人类般的全双工多模态交互。它通过SA-MoE架构统一处理听、看、说、动，支持并发感知与生成，解决了多模态干扰与协作难题，实现了自然的人机交互行为。**

- **链接: [http://arxiv.org/pdf/2510.16756v1](http://arxiv.org/pdf/2510.16756v1)**

> **作者:** Siyin Wang; Wenyi Yu; Xianzhao Chen; Xiaohai Tian; Jun Zhang; Lu Lu; Chao Zhang
>
> **备注:** 22 pages, 8 figures
>
> **摘要:** Human interaction is inherently multimodal and full-duplex: we listen while watching, speak while acting, and fluidly adapt to turn-taking and interruptions. Realizing these capabilities is essential for building models simulating humans. We present ELLSA (End-to-end Listen, Look, Speak and Act), which, to our knowledge, is the first full-duplex, end-to-end model that simultaneously perceives and generates across vision, text, speech, and action within a single architecture, enabling interaction patterns previously out of reach, yielding more natural, human-like behaviors. At its core is a novel SA-MoE architecture (Self-Attention Mixture-of-Experts) that routes each modality to specialized experts and fuses them through a unified attention backbone. This provides a generalizable solution for joint multimodal perception and concurrent generation, leveraging strong pre-trained components while enabling efficient modality integration and mitigating modality interference. On speech-interaction and robot-manipulation benchmarks, ELLSA matches modality-specific baselines, while uniquely supporting advanced multimodal and full-duplex behaviors such as dialogue and action turn-taking, defective instruction rejection, speaking-while-acting, context-grounded visual question answering, and action barge-ins. We contend that ELLSA represents a step toward more natural and general interactive intelligence, contributing to the broader pursuit of artificial general intelligence. All data, code and model checkpoints will be released upon acceptance.
>
---
#### [new 144] WEBSERV: A Browser-Server Environment for Efficient Training of Reinforcement Learning-based Web Agents at Scale
- **分类: cs.LG; cs.CL**

- **简介: 该论文聚焦强化学习（RL）驱动的网页智能体训练任务，旨在解决现有环境上下文噪声大、动作非确定性及扩展性差的问题。作者提出WEBSERV，构建了轻量浏览器环境并实现高效可扩展的服务端架构，显著降低资源开销，支持大规模并行训练与评估。**

- **链接: [http://arxiv.org/pdf/2510.16252v1](http://arxiv.org/pdf/2510.16252v1)**

> **作者:** Yuxuan Lu; Jing Huang; Hui Liu; Jiri Gesi; Yan Han; Shihan Fu; Tianqi Zheng; Dakuo Wang
>
> **摘要:** Training and evaluation of Reinforcement Learning (RL) web agents have gained increasing attention, yet a scalable and efficient environment that couples realistic and robust browser-side interaction with controllable server-side state at scale is still missing. Existing environments tend to have one or more of the following issues: they overwhelm policy models with excessive and noisy context; they perform actions non-deterministically without waiting for the UI or network to stabilize; or they cannot scale isolated client-server containers effectively for parallel RL rollouts. We propose WEBSERV, an environment that includes 1) a compact, site-agnostic browser environment that balances context and action complexity, and 2) a scalable RL environment via efficient launching and resetting web-servers to enable scalable RL training and evaluation. We evaluate WEBSERV on the shopping CMS and Gitlab tasks in WebArena, achieving state-of-the-art single-prompt success rates while cutting launch latency by ~5x and storage need by ~240x, with a comparable memory footprint, enabling 200+ concurrent containers on a single host.
>
---
#### [new 145] LILO: Bayesian Optimization with Interactive Natural Language Feedback
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出LILO框架，解决传统贝叶斯优化依赖结构化反馈的问题。通过大语言模型将自然语言反馈转化为标量效用，实现更灵活、高效的黑箱优化，兼顾样本效率与用户交互性，在反馈受限场景下优于传统方法。**

- **链接: [http://arxiv.org/pdf/2510.17671v1](http://arxiv.org/pdf/2510.17671v1)**

> **作者:** Katarzyna Kobalczyk; Zhiyuan Jerry Lin; Benjamin Letham; Zhuokai Zhao; Maximilian Balandat; Eytan Bakshy
>
> **摘要:** For many real-world applications, feedback is essential in translating complex, nuanced, or subjective goals into quantifiable optimization objectives. We propose a language-in-the-loop framework that uses a large language model (LLM) to convert unstructured feedback in the form of natural language into scalar utilities to conduct BO over a numeric search space. Unlike preferential BO, which only accepts restricted feedback formats and requires customized models for each domain-specific problem, our approach leverages LLMs to turn varied types of textual feedback into consistent utility signals and to easily include flexible user priors without manual kernel design. At the same time, our method maintains the sample efficiency and principled uncertainty quantification of BO. We show that this hybrid method not only provides a more natural interface to the decision maker but also outperforms conventional BO baselines and LLM-only optimizers, particularly in feedback-limited regimes.
>
---
#### [new 146] Reasoning Distillation and Structural Alignment for Improved Code Generation
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文聚焦代码生成任务，旨在解决小模型缺乏复杂推理能力的问题。通过推理蒸馏与结构对齐方法，将大模型的推理过程迁移至小模型，提升其解题准确率与结构理解能力，在多个基准上显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2510.17598v1](http://arxiv.org/pdf/2510.17598v1)**

> **作者:** Amir Jalilifard; Anderson de Rezende Rocha; Marcos Medeiros Raimundo
>
> **摘要:** Effective code generation with language models hinges on two critical factors: accurately understanding the intent of the prompt and generating code that applies algorithmic reasoning to produce correct solutions capable of passing diverse test cases while adhering to the syntax of the target programming language. Unlike other language tasks, code generation requires more than accurate token prediction; it demands comprehension of solution-level and structural relationships rather than merely generating the most likely tokens. very large language model (VLLM) are capable of generating detailed steps toward the correct solution of complex tasks where reasoning is crucial in solving the problem. Such reasoning capabilities may be absent in smaller language models. Therefore, in this work, we distill the reasoning capabilities of a VLLM into a smaller, more efficient model that is faster and cheaper to deploy. Our approach trains the model to emulate the reasoning and problem-solving abilities of the VLLM by learning to identify correct solution pathways and establishing a structural correspondence between problem definitions and potential solutions through a novel method of structure-aware loss optimization. This enables the model to transcend token-level generation and to deeply grasp the overarching structure of solutions for given problems. Experimental results show that our fine-tuned model, developed through a cheap and simple to implement process, significantly outperforms our baseline model in terms of pass@1, average data flow, and average syntax match metrics across the MBPP, MBPP Plus, and HumanEval benchmarks.
>
---
#### [new 147] Forgetting to Forget: Attention Sink as A Gateway for Backdooring LLM Unlearning
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究大语言模型“遗忘学习”过程中的后门攻击，提出通过注意力沉降现象将触发器植入特定位置，使模型看似完成遗忘，实则可在触发时恢复已遗忘知识，揭示了遗忘机制的安全隐患。**

- **链接: [http://arxiv.org/pdf/2510.17021v1](http://arxiv.org/pdf/2510.17021v1)**

> **作者:** Bingqi Shang; Yiwei Chen; Yihua Zhang; Bingquan Shen; Sijia Liu
>
> **摘要:** Large language model (LLM) unlearning has become a critical mechanism for removing undesired data, knowledge, or behaviors from pre-trained models while retaining their general utility. Yet, with the rise of open-weight LLMs, we ask: can the unlearning process itself be backdoored, appearing successful under normal conditions yet reverting to pre-unlearned behavior when a hidden trigger is activated? Drawing inspiration from classical backdoor attacks that embed triggers into training data to enforce specific behaviors, we investigate backdoor unlearning, where models forget as intended in the clean setting but recover forgotten knowledge when the trigger appears. We show that designing such attacks presents unique challenges, hinging on where triggers are placed and how backdoor training is reinforced. We uncover a strong link between backdoor efficacy and the attention sink phenomenon, i.e., shallow input tokens consistently attract disproportionate attention in LLMs. Our analysis reveals that these attention sinks serve as gateways for backdoor unlearning: placing triggers at sink positions and aligning their attention values markedly enhances backdoor persistence. Extensive experiments validate these findings, showing that attention-sink-guided backdoor unlearning reliably restores forgotten knowledge in the presence of backdoor triggers, while behaving indistinguishably from a normally unlearned model when triggers are absent. Code is available at https://github.com/OPTML-Group/Unlearn-Backdoor.
>
---
#### [new 148] Zero-Shot Performance Prediction for Probabilistic Scaling Laws
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究NLP模型学习曲线的零样本预测，旨在降低性能预估的成本。通过多任务学习与高斯过程建模任务间相关性，实现跨任务和层次的共享信息利用，支持无需训练即可预测新任务的学习曲线，并结合主动学习提升预测精度。**

- **链接: [http://arxiv.org/pdf/2510.16743v1](http://arxiv.org/pdf/2510.16743v1)**

> **作者:** Viktoria Schram; Markus Hiller; Daniel Beck; Trevor Cohn
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** The prediction of learning curves for Natural Language Processing (NLP) models enables informed decision-making to meet specific performance objectives, while reducing computational overhead and lowering the costs associated with dataset acquisition and curation. In this work, we formulate the prediction task as a multitask learning problem, where each task's data is modelled as being organized within a two-layer hierarchy. To model the shared information and dependencies across tasks and hierarchical levels, we employ latent variable multi-output Gaussian Processes, enabling to account for task correlations and supporting zero-shot prediction of learning curves (LCs). We demonstrate that this approach facilitates the development of probabilistic scaling laws at lower costs. Applying an active learning strategy, LCs can be queried to reduce predictive uncertainty and provide predictions close to ground truth scaling laws. We validate our framework on three small-scale NLP datasets with up to $30$ LCs. These are obtained from nanoGPT models, from bilingual translation using mBART and Transformer models, and from multilingual translation using M2M100 models of varying sizes.
>
---
#### [new 149] The Hidden Cost of Modeling P(X): Vulnerability to Membership Inference Attacks in Generative Text Classifiers
- **分类: cs.CR; cs.CL; cs.LG; stat.ML**

- **简介: 该论文研究生成式文本分类器的成员推断攻击脆弱性，属隐私安全任务。通过理论与实验分析，揭示显式建模联合概率P(X,Y)的生成式分类器更易泄露训练数据成员信息，指出其隐私-效用权衡问题，并警示在敏感场景中需谨慎使用此类模型。**

- **链接: [http://arxiv.org/pdf/2510.16122v1](http://arxiv.org/pdf/2510.16122v1)**

> **作者:** Owais Makroo; Siva Rajesh Kasa; Sumegh Roychowdhury; Karan Gupta; Nikhil Pattisapu; Santhosh Kasa; Sumit Negi
>
> **摘要:** Membership Inference Attacks (MIAs) pose a critical privacy threat by enabling adversaries to determine whether a specific sample was included in a model's training dataset. Despite extensive research on MIAs, systematic comparisons between generative and discriminative classifiers remain limited. This work addresses this gap by first providing theoretical motivation for why generative classifiers exhibit heightened susceptibility to MIAs, then validating these insights through comprehensive empirical evaluation. Our study encompasses discriminative, generative, and pseudo-generative text classifiers across varying training data volumes, evaluated on nine benchmark datasets. Employing a diverse array of MIA strategies, we consistently demonstrate that fully generative classifiers which explicitly model the joint likelihood $P(X,Y)$ are most vulnerable to membership leakage. Furthermore, we observe that the canonical inference approach commonly used in generative classifiers significantly amplifies this privacy risk. These findings reveal a fundamental utility-privacy trade-off inherent in classifier design, underscoring the critical need for caution when deploying generative classifiers in privacy-sensitive applications. Our results motivate future research directions in developing privacy-preserving generative classifiers that can maintain utility while mitigating membership inference vulnerabilities.
>
---
#### [new 150] Prompt Optimization via Retrieved Reasoning Assets and Multi-Agent Analysis
- **分类: cs.MA; cs.AI; cs.CL; cs.HC; cs.IR**

- **简介: 该论文研究提示优化任务，旨在解决现有方法依赖黑箱评估和试错的问题。作者提出MA-SAPO框架，通过多智能体协同推理与检索可复用的推理资产，实现可解释、可控的提示优化，提升了大模型性能。**

- **链接: [http://arxiv.org/pdf/2510.16635v1](http://arxiv.org/pdf/2510.16635v1)**

> **作者:** Wonduk Seo; Juhyeon Lee; Junseo Koh; Hyunjin An; Jian Park; Seunghyun Lee; Haihua Chen; Yi Bu
>
> **备注:** Preprint
>
> **摘要:** Prompt optimization has emerged as an effective alternative to retraining for improving the performance of Large Language Models (LLMs). However, most existing approaches treat evaluation as a black box, relying solely on numerical scores while offering limited insight into why a prompt succeeds or fails. They also depend heavily on trial-and-error refinements, which are difficult to interpret and control. In this paper, we introduce MA-SAPO, a Multi-Agent framework for Score-Aware Prompt Optimization. Compared to prior methods, MA-SAPO explicitly couples evaluation outcomes with structured reasoning to guide systematic edits. The framework specifically consists of two stages: during the Reasoning Phase, agents collaboratively explain metric scores, diagnose weaknesses, and synthesize targeted refinements that are stored as reusable reasoning assets; during the Test Phase, agents retrieve these assets to analyze optimized prompts and apply only evidence-grounded edits. By turning evaluation signals into interpretable reasoning chains, MA-SAPO produces prompt refinements that are more transparent, auditable, and controllable. Experiments on the HelpSteer1/2 benchmarks demonstrate consistent improvements over single-pass prompting, retrieval-augmented baselines, and prior multi-agent strategies, validating the effectiveness of our approach.
>
---
#### [new 151] VERA-V: Variational Inference Framework for Jailbreaking Vision-Language Models
- **分类: cs.CR; cs.CL; cs.CV; cs.LG**

- **简介: 该论文针对视觉-语言模型（VLM）的安全漏洞，提出变分推理框架VERA-V，通过联合文本-图像对抗提示生成，提升越狱攻击成功率。其结合隐写文本、扩散图像与注意力干扰策略，在多种VLM上显著提高攻击效果。**

- **链接: [http://arxiv.org/pdf/2510.17759v1](http://arxiv.org/pdf/2510.17759v1)**

> **作者:** Qilin Liao; Anamika Lochab; Ruqi Zhang
>
> **备注:** 18 pages, 7 Figures,
>
> **摘要:** Vision-Language Models (VLMs) extend large language models with visual reasoning, but their multimodal design also introduces new, underexplored vulnerabilities. Existing multimodal red-teaming methods largely rely on brittle templates, focus on single-attack settings, and expose only a narrow subset of vulnerabilities. To address these limitations, we introduce VERA-V, a variational inference framework that recasts multimodal jailbreak discovery as learning a joint posterior distribution over paired text-image prompts. This probabilistic view enables the generation of stealthy, coupled adversarial inputs that bypass model guardrails. We train a lightweight attacker to approximate the posterior, allowing efficient sampling of diverse jailbreaks and providing distributional insights into vulnerabilities. VERA-V further integrates three complementary strategies: (i) typography-based text prompts that embed harmful cues, (ii) diffusion-based image synthesis that introduces adversarial signals, and (iii) structured distractors to fragment VLM attention. Experiments on HarmBench and HADES benchmarks show that VERA-V consistently outperforms state-of-the-art baselines on both open-source and frontier VLMs, achieving up to 53.75% higher attack success rate (ASR) over the best baseline on GPT-4o.
>
---
#### [new 152] When Many-Shot Prompting Fails: An Empirical Study of LLM Code Translation
- **分类: cs.SE; cs.AI; cs.CL; cs.PL; 68T50, 68N30, 68W40; I.2.7; D.2.7; I.2.6**

- **简介: 该论文研究大语言模型在代码翻译中的多示例提示效果，发现示例增多反致功能正确性下降，提出“多示例悖论”，指出少量精选示例优于大量示例，揭示任务依赖的最优提示策略。**

- **链接: [http://arxiv.org/pdf/2510.16809v1](http://arxiv.org/pdf/2510.16809v1)**

> **作者:** Amirkia Rafiei Oskooei; Kaan Baturalp Cosdan; Husamettin Isiktas; Mehmet S. Aktas
>
> **摘要:** Large Language Models (LLMs) with vast context windows offer new avenues for in-context learning (ICL), where providing many examples ("many-shot" prompting) is often assumed to enhance performance. We investigate this assumption for the complex task of code translation. Through a large-scale empirical study of over 90,000 translations, we systematically evaluate the impact of scaling in-context examples from zero-shot to many-shot configurations of up to 625 examples, with prompts spanning from approximately 100,000 to 800,000 tokens. Our findings reveal a "many-shot paradox": while static similarity metrics may modestly improve with more examples, functional correctness consistently peaks with few-shot prompting (5-25 examples). Providing substantially more examples often degrades this crucial functional performance. This study highlights that for code translation, the quality of a few well-chosen examples outweighs sheer quantity, challenging the universal efficacy of "more is better" for ICL and underscoring the task-dependent nature of optimal prompting strategies. Our results have significant implications for effectively leveraging LLMs in software engineering.
>
---
#### [new 153] Can GRPO Help LLMs Transcend Their Pretraining Origin?
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究GRPO算法在增强大模型推理能力时的局限性，探讨其为何在某些领域有效而其他领域无效。通过理论证明与实验验证，揭示GRPO只能强化预训练偏差，无法超越原有分布，提出需发展能突破预训练限制的新算法。**

- **链接: [http://arxiv.org/pdf/2510.15990v1](http://arxiv.org/pdf/2510.15990v1)**

> **作者:** Kangqi Ni; Zhen Tan; Zijie Liu; Pingzhi Li; Tianlong Chen
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR), primarily driven by the Group Relative Policy Optimization (GRPO) algorithm, is a leading approach for enhancing the reasoning abilities of Large Language Models (LLMs). Despite its wide adoption, GRPO's gains are often inconsistent; for instance, a model may show significant improvement in one reasoning domain, like mathematics, yet remain stagnant in another, such as medicine. This inconsistency raises a critical question: under what conditions does GRPO improve reasoning and generalize out-of-distribution (OOD)? We investigate this from a data distribution perspective. We first prove theoretically that GRPO is a conservative reweighting scheme, bounded by the base model's distribution and thus unable to discover completely novel solutions. We further validate this in carefully designed controlled studies by training transformers from scratch, evaluating generalization across reasoning depth, input length, token representation, and compositionality. Our results provide a principled explanation for GRPO's boundaries: OOD improvement emerges only when the target task aligns with the model's pretrained biases, while gains on in-distribution (ID) tasks diminish as performance saturates. This reframes GRPO not as a universal reasoning enhancer but as a tool that sharpens pretraining biases. Our findings motivate future development of algorithms that can expand a model's capabilities beyond its pretraining origin.
>
---
#### [new 154] Investigating the Association Between Text-Based Indications of Foodborne Illness from Yelp Reviews and New York City Health Inspection Outcomes (2023)
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究Yelp评论中基于文本的食源性疾病信号与纽约市餐厅卫生检查结果的关联。任务为公共卫生监测，利用HSAN模型分析用户评论，评估其与官方检查等级的相关性，发现街区层面相关性较弱，需进一步开展地址级分析。**

- **链接: [http://arxiv.org/pdf/2510.16334v1](http://arxiv.org/pdf/2510.16334v1)**

> **作者:** Eden Shaveet; Crystal Su; Daniel Hsu; Luis Gravano
>
> **备注:** Presented as a poster at Data Science Day 2024
>
> **摘要:** Foodborne illnesses are gastrointestinal conditions caused by consuming contaminated food. Restaurants are critical venues to investigate outbreaks because they share sourcing, preparation, and distribution of foods. Public reporting of illness via formal channels is limited, whereas social media platforms host abundant user-generated content that can provide timely public health signals. This paper analyzes signals from Yelp reviews produced by a Hierarchical Sigmoid Attention Network (HSAN) classifier and compares them with official restaurant inspection outcomes issued by the New York City Department of Health and Mental Hygiene (NYC DOHMH) in 2023. We evaluate correlations at the Census tract level, compare distributions of HSAN scores by prevalence of C-graded restaurants, and map spatial patterns across NYC. We find minimal correlation between HSAN signals and inspection scores at the tract level and no significant differences by number of C-graded restaurants. We discuss implications and outline next steps toward address-level analyses.
>
---
#### [new 155] Utility-Diversity Aware Online Batch Selection for LLM Supervised Fine-tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大模型监督微调中的在线数据选择问题，旨在提升效率并避免过拟合。作者提出UDS框架，结合数据效用与多样性评估，无需外部资源，降低计算开销，实现在有限数据下高效训练。**

- **链接: [http://arxiv.org/pdf/2510.16882v1](http://arxiv.org/pdf/2510.16882v1)**

> **作者:** Heming Zou; Yixiu Mao; Yun Qu; Qi Wang; Xiangyang Ji
>
> **摘要:** Supervised fine-tuning (SFT) is a commonly used technique to adapt large language models (LLMs) to downstream tasks. In practice, SFT on a full dataset is computationally expensive and sometimes suffers from overfitting or bias amplification. This facilitates the rise of data curation in SFT, which prioritizes the most valuable data to optimze. This work studies the online batch selection family that dynamically scores and filters samples during the training process. However, existing popular methods often (i) rely merely on the utility of data to select a subset while neglecting other crucial factors like diversity, (ii) rely on external resources such as reference models or validation sets, and (iii) incur extra training time over full-dataset training. To address these limitations, this work develops \textbf{UDS (Utility-Diversity Sampling)}, a framework for efficient online batch selection in SFT. UDS leverages the nuclear norm of the logits matrix to capture both data utility and intra-sample diversity, while estimating inter-sample diversity through efficient low-dimensional embedding comparisons with a lightweight memory buffer of historical samples. Such a design eliminates the need for external resources and unnecessary backpropagation, securing computational efficiency. Experiments on multiple benchmarks demonstrate that UDS consistently outperforms state-of-the-art online batch selection methods under varying data budgets, and significantly reduces training time compared to full-dataset fine-tuning. Code is available at https://github.com/gfyddha/UDS.
>
---
#### [new 156] MIRAGE: Agentic Framework for Multimodal Misinformation Detection with Web-Grounded Reasoning
- **分类: cs.AI; cs.CL; cs.CV; cs.CY; cs.LG; I.2.7; H.3.3; I.4.9**

- **简介: 该论文针对多模态虚假信息检测，提出MIRAGE框架，通过视觉验证、跨模态一致性分析、检索增强的事实核查与校准判断四个模块，结合网页证据进行推理，无需领域特定训练即可有效识别图文 misinformation，提升检测准确率与可解释性。**

- **链接: [http://arxiv.org/pdf/2510.17590v1](http://arxiv.org/pdf/2510.17590v1)**

> **作者:** Mir Nafis Sharear Shopnil; Sharad Duwal; Abhishek Tyagi; Adiba Mahbub Proma
>
> **备注:** 16 pages, 3 tables, 1 figure
>
> **摘要:** Misinformation spreads across web platforms through billions of daily multimodal posts that combine text and images, overwhelming manual fact-checking capacity. Supervised detection models require domain-specific training data and fail to generalize across diverse manipulation tactics. We present MIRAGE, an inference-time, model-pluggable agentic framework that decomposes multimodal verification into four sequential modules: visual veracity assessment detects AI-generated images, cross-modal consistency analysis identifies out-of-context repurposing, retrieval-augmented factual checking grounds claims in web evidence through iterative question generation, and a calibrated judgment module integrates all signals. MIRAGE orchestrates vision-language model reasoning with targeted web retrieval, outputs structured and citation-linked rationales. On MMFakeBench validation set (1,000 samples), MIRAGE with GPT-4o-mini achieves 81.65% F1 and 75.1% accuracy, outperforming the strongest zero-shot baseline (GPT-4V with MMD-Agent at 74.0% F1) by 7.65 points while maintaining 34.3% false positive rate versus 97.3% for a judge-only baseline. Test set results (5,000 samples) confirm generalization with 81.44% F1 and 75.08% accuracy. Ablation studies show visual verification contributes 5.18 F1 points and retrieval-augmented reasoning contributes 2.97 points. Our results demonstrate that decomposed agentic reasoning with web retrieval can match supervised detector performance without domain-specific training, enabling misinformation detection across modalities where labeled data remains scarce.
>
---
#### [new 157] A Comprehensive Survey on Reinforcement Learning-based Agentic Search: Foundations, Roles, Optimizations, Evaluations, and Applications
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于综述任务，旨在解决传统检索增强生成系统缺乏自适应性的问题。作者系统梳理了基于强化学习的智能体搜索方法，从功能角色、优化策略和应用范围三方面进行总结与分析，并探讨了未来方向。**

- **链接: [http://arxiv.org/pdf/2510.16724v1](http://arxiv.org/pdf/2510.16724v1)**

> **作者:** Minhua Lin; Zongyu Wu; Zhichao Xu; Hui Liu; Xianfeng Tang; Qi He; Charu Aggarwal; Hui Liu; Xiang Zhang; Suhang Wang
>
> **备注:** 38 pages, 4 figures, 7 tables
>
> **摘要:** The advent of large language models (LLMs) has transformed information access and reasoning through open-ended natural language interaction. However, LLMs remain limited by static knowledge, factual hallucinations, and the inability to retrieve real-time or domain-specific information. Retrieval-Augmented Generation (RAG) mitigates these issues by grounding model outputs in external evidence, but traditional RAG pipelines are often single turn and heuristic, lacking adaptive control over retrieval and reasoning. Recent advances in agentic search address these limitations by enabling LLMs to plan, retrieve, and reflect through multi-step interaction with search environments. Within this paradigm, reinforcement learning (RL) offers a powerful mechanism for adaptive and self-improving search behavior. This survey provides the first comprehensive overview of \emph{RL-based agentic search}, organizing the emerging field along three complementary dimensions: (i) What RL is for (functional roles), (ii) How RL is used (optimization strategies), and (iii) Where RL is applied (scope of optimization). We summarize representative methods, evaluation protocols, and applications, and discuss open challenges and future directions toward building reliable and scalable RL driven agentic search systems. We hope this survey will inspire future research on the integration of RL and agentic search. Our repository is available at https://github.com/ventr1c/Awesome-RL-based-Agentic-Search-Papers.
>
---
#### [new 158] See or Say Graphs: Agent-Driven Scalable Graph Understanding with Vision-Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究图数据的多模态理解任务，旨在解决现有视觉-语言模型在图理解中因输入长度限制导致的可扩展性差和模态协调不足问题。作者提出GraphVista框架，通过层次化图信息组织和基于代理的模态路由机制，实现高效、可扩展的图文协同图理解。**

- **链接: [http://arxiv.org/pdf/2510.16769v1](http://arxiv.org/pdf/2510.16769v1)**

> **作者:** Shuo Han; Yukun Cao; Zezhong Ding; Zengyi Gao; S Kevin Zhou; Xike Xie
>
> **摘要:** Vision-language models (VLMs) have shown promise in graph understanding, but remain limited by input-token constraints, facing scalability bottlenecks and lacking effective mechanisms to coordinate textual and visual modalities. To address these challenges, we propose GraphVista, a unified framework that enhances both scalability and modality coordination in graph understanding. For scalability, GraphVista organizes graph information hierarchically into a lightweight GraphRAG base, which retrieves only task-relevant textual descriptions and high-resolution visual subgraphs, compressing redundant context while preserving key reasoning elements. For modality coordination, GraphVista introduces a planning agent that routes tasks to the most suitable modality-using the text modality for simple property reasoning and the visual modality for local and structurally complex reasoning grounded in explicit topology. Extensive experiments demonstrate that GraphVista scales to large graphs, up to $200\times$ larger than those used in existing benchmarks, and consistently outperforms existing textual, visual, and fusion-based methods, achieving up to $4.4\times$ quality improvement over the state-of-the-art baselines by fully exploiting the complementary strengths of both modalities.
>
---
## 更新

#### [replaced 001] Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.08907v3](http://arxiv.org/pdf/2510.08907v3)**

> **作者:** Xin Liu; Runsong Zhao; Pengcheng Huang; Xinyu Liu; Junyi Xiao; Chunyang Xiao; Tong Xiao; Shengxiang Gao; Zhengtao Yu; Jingbo Zhu
>
> **备注:** 18 pages,9 figures
>
> **摘要:** Context compression presents a promising approach for accelerating large language model (LLM) inference by compressing long contexts into compact representations. Current context compression methods predominantly rely on autoencoding tasks to train context-agnostic compression tokens to compress contextual semantics. While autoencoding tasks enable compression tokens to acquire compression capabilities, compression via autoencoding tasks creates a fundamental mismatch: the models are optimized for reconstruction that diverge from actual downstream tasks, thereby weakening the features more beneficial for real-world usage. We propose Semantic-Anchor Compression (SAC), a novel method that shifts from autoencoding task based compression to an architecture that is equipped with this compression capability \textit{a priori}. Instead of training models to compress contexts through autoencoding tasks, SAC directly selects so-called anchor tokens from the original context and aggregates contextual information into their key-value (KV) representations. By deriving representations directly from the contextual tokens, SAC eliminates the need for autoencoding training. To ensure compression performance while directly leveraging anchor tokens, SAC incorporates two key designs: (1) anchor embeddings that enable the compressor to identify critical tokens, and (2) bidirectional attention modification that allows anchor tokens to capture information from the entire context. Experimental results demonstrate that SAC consistently outperforms existing context compression methods across various compression ratios. On out-of-distribution evaluation using MRQA, SAC achieves 1 EM improvement at 5x compression over strong baselines, with increasing advantages at higher compression ratios.
>
---
#### [replaced 002] Hallucination Detection in LLMs Using Spectral Features of Attention Maps
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17598v2](http://arxiv.org/pdf/2502.17598v2)**

> **作者:** Jakub Binkowski; Denis Janiak; Albert Sawczyn; Bogdan Gabrys; Tomasz Kajdanowicz
>
> **备注:** Accepted to EMNLP 2025. Code available at https://github.com/graphml-lab-pwr/lapeigvals
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable performance across various tasks but remain prone to hallucinations. Detecting hallucinations is essential for safety-critical applications, and recent methods leverage attention map properties to this end, though their effectiveness remains limited. In this work, we investigate the spectral features of attention maps by interpreting them as adjacency matrices of graph structures. We propose the $\text{LapEigvals}$ method, which utilises the top-$k$ eigenvalues of the Laplacian matrix derived from the attention maps as an input to hallucination detection probes. Empirical evaluations demonstrate that our approach achieves state-of-the-art hallucination detection performance among attention-based methods. Extensive ablation studies further highlight the robustness and generalisation of $\text{LapEigvals}$, paving the way for future advancements in the hallucination detection domain.
>
---
#### [replaced 003] Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.12680v2](http://arxiv.org/pdf/2505.12680v2)**

> **作者:** Haoyu Zhao; Yihan Geng; Shange Tang; Yong Lin; Bohan Lyu; Hongzhou Lin; Chi Jin; Sanjeev Arora
>
> **备注:** To appear in NeurIPS 2025 Track on Datasets and Benchmarks. 28 pages
>
> **摘要:** LLM-based formal proof assistants (e.g., in Lean) hold great promise for automating mathematical discovery. But beyond syntactic correctness, do these systems truly understand mathematical structure as humans do? We investigate this question in context of mathematical inequalities -- specifically the prover's ability to recognize that the given problem simplifies by applying a known inequality such as AM/GM. Specifically, we are interested in their ability to do this in a compositional setting where multiple inequalities must be applied as part of a solution. We introduce Ineq-Comp, a benchmark built from elementary inequalities through systematic transformations, including variable duplication, algebraic rewriting, and multi-step composition. Although these problems remain easy for humans, we find that most provers -- including Goedel, STP, and Kimina-7B -- struggle significantly. DeepSeek-Prover-V2-7B shows relative robustness, but still suffers a 20% performance drop (pass@32). Even for DeepSeek-Prover-V2-671B model, the gap between compositional variants and seed problems exists, implying that simply scaling up the model size alone does not fully solve the compositional weakness. Strikingly, performance remains poor for all models even when formal proofs of the constituent parts are provided in context, revealing that the source of weakness is indeed in compositional reasoning. Our results expose a persisting gap between the generalization behavior of current AI provers and human mathematical intuition. All data and evaluation code can be found at https://github.com/haoyuzhao123/LeanIneqComp.
>
---
#### [replaced 004] "Mirror" Language AI Models of Depression are Criterion-Contaminated
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2508.05830v2](http://arxiv.org/pdf/2508.05830v2)**

> **作者:** Tong Li; Rasiq Hussain; Mehak Gupta; Joshua R. Oltmanns
>
> **备注:** 38 pages, 9 figures
>
> **摘要:** Recent studies show near-perfect language-based predictions of depression scores (R2 = .70), but these "Mirror" models rely on language responses directly from depression assessments to predict depression assessment scores. These methods suffer from criterion contamination that inflate prediction estimates. We compare "Mirror" models to "Non-Mirror" models, which use other external language to predict depression scores. 110 participants completed both structured diagnostic (Mirror condition) and life history (Non-Mirror condition) interviews. LLMs were prompted to predict diagnostic depression scores. As expected, Mirror models were near-perfect. However, Non-Mirror models also displayed prediction sizes considered large in psychology. Further, both Mirror and Non-Mirror predictions correlated with other questionnaire-based depression symptoms at similar sizes, suggesting bias in Mirror models. Topic modeling revealed different theme structures across model types. As language models for depression continue to evolve, incorporating Non-Mirror approaches may support more valid and clinically useful language-based AI applications in psychological assessment.
>
---
#### [replaced 005] The Curious Case of Curiosity across Human Cultures and LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.12943v2](http://arxiv.org/pdf/2510.12943v2)**

> **作者:** Angana Borah; Zhijing Jin; Rada Mihalcea
>
> **备注:** Preprint (Paper under review)
>
> **摘要:** Recent advances in Large Language Models (LLMs) have expanded their role in human interaction, yet curiosity -- a central driver of inquiry -- remains underexplored in these systems, particularly across cultural contexts. In this work, we investigate cultural variation in curiosity using Yahoo! Answers, a real-world multi-country dataset spanning diverse topics. We introduce CUEST (CUriosity Evaluation across SocieTies), an evaluation framework that measures human-model alignment in curiosity through linguistic (style), topic preference (content) analysis and grounding insights in social science constructs. Across open- and closed-source models, we find that LLMs flatten cross-cultural diversity, aligning more closely with how curiosity is expressed in Western countries. We then explore fine-tuning strategies to induce curiosity in LLMs, narrowing the human-model alignment gap by up to 50%. Finally, we demonstrate the practical value of curiosity for LLM adaptability across cultures, showing its importance for future NLP research.
>
---
#### [replaced 006] The Moral Foundations Reddit Corpus
- **分类: cs.CL; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2208.05545v3](http://arxiv.org/pdf/2208.05545v3)**

> **作者:** Jackson Trager; Alireza S. Ziabari; Elnaz Rahmati; Aida Mostafazadeh Davani; Preni Golazizian; Farzan Karimi-Malekabadi; Ali Omrani; Zhihe Li; Brendan Kennedy; Nils Karl Reimer; Melissa Reyes; Kelsey Cheng; Mellow Wei; Christina Merrifield; Arta Khosravi; Evans Alvarez; Morteza Dehghani
>
> **摘要:** Moral framing and sentiment can affect a variety of online and offline behaviors, including donation, environmental action, political engagement, and protest. Various computational methods in Natural Language Processing (NLP) have been used to detect moral sentiment from textual data, but achieving strong performance in such subjective tasks requires large, hand-annotated datasets. Previous corpora annotated for moral sentiment have proven valuable, and have generated new insights both within NLP and across the social sciences, but have been limited to Twitter. To facilitate improving our understanding of the role of moral rhetoric, we present the Moral Foundations Reddit Corpus, a collection of 16,123 English Reddit comments that have been curated from 12 distinct subreddits, hand-annotated by at least three trained annotators for 8 categories of moral sentiment (i.e., Care, Proportionality, Equality, Purity, Authority, Loyalty, Thin Morality, Implicit/Explicit Morality) based on the updated Moral Foundations Theory (MFT) framework. We evaluate baselines using large language models (Llama3-8B, Ministral-8B) in zero-shot, few-shot, and PEFT settings, comparing their performance to fine-tuned encoder-only models like BERT. The results show that LLMs continue to lag behind fine-tuned encoders on this subjective task, underscoring the ongoing need for human-annotated moral corpora for AI alignment evaluation. Keywords: moral sentiment annotation, moral values, moral foundations theory, multi-label text classification, large language models, benchmark dataset, evaluation and alignment resource
>
---
#### [replaced 007] Don't Trust Generative Agents to Mimic Communication on Social Networks Unless You Benchmarked their Empirical Realism
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21974v3](http://arxiv.org/pdf/2506.21974v3)**

> **作者:** Simon Münker; Nils Schwager; Achim Rettinger
>
> **备注:** 11 pages, 1 figure, 3 tables
>
> **摘要:** The ability of Large Language Models (LLMs) to mimic human behavior triggered a plethora of computational social science research, assuming that empirical studies of humans can be conducted with AI agents instead. Since there have been conflicting research findings on whether and when this hypothesis holds, there is a need to better understand the differences in their experimental designs. We focus on replicating the behavior of social network users with the use of LLMs for the analysis of communication on social networks. First, we provide a formal framework for the simulation of social networks, before focusing on the sub-task of imitating user communication. We empirically test different approaches to imitate user behavior on X in English and German. Our findings suggest that social simulations should be validated by their empirical realism measured in the setting in which the simulation components were fitted. With this paper, we argue for more rigor when applying generative-agent-based modeling for social simulation.
>
---
#### [replaced 008] Intrinsic Self-Correction in LLMs: Towards Explainable Prompting via Mechanistic Interpretability
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.11924v2](http://arxiv.org/pdf/2505.11924v2)**

> **作者:** Yu-Ting Lee; Fu-Chieh Chang; Hui-Ying Shih; Pei-Yuan Wu
>
> **摘要:** Intrinsic self-correction refers to the phenomenon where a language model refines its own outputs purely through prompting, without external feedback or parameter updates. While this approach improves performance across diverse tasks, its internal mechanism remains poorly understood. We analyze intrinsic self-correction from a representation-level perspective. We formalize and introduce the notion of a prompt-induced shift, which is the change in hidden representations caused by a self-correction prompt. Across 5 open-source LLMs, prompt-induced shifts in text detoxification and text toxification align with latent directions constructed from contrastive pairs. In detoxification, the shifts align with the non-toxic direction; in toxification, they align with the toxic direction. These results suggest that intrinsic self-correction functions as representation steering along interpretable latent directions, beyond what standard metrics such as task scores or model confidence capture. Our analysis offers an interpretability-based account of intrinsic self-correction and contributes to a more systematic understanding of LLM prompting.
>
---
#### [replaced 009] REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24760v2](http://arxiv.org/pdf/2505.24760v2)**

> **作者:** Zafir Stojanovski; Oliver Stanley; Joe Sharratt; Richard Jones; Abdulhakeem Adefioye; Jean Kaddour; Andreas Köpf
>
> **备注:** NeurIPS 2025 Spotlight. For code, see https://github.com/open-thought/reasoning-gym
>
> **摘要:** We introduce Reasoning Gym (RG), a library of reasoning environments for reinforcement learning with verifiable rewards. It provides over 100 data generators and verifiers spanning multiple domains including algebra, arithmetic, computation, cognition, geometry, graph theory, logic, and various common games. Its key innovation is the ability to generate virtually infinite training data with adjustable complexity, unlike most previous reasoning datasets, which are typically fixed. This procedural generation approach allows for continuous evaluation across varying difficulty levels. Our experimental results demonstrate the efficacy of RG in both evaluating and reinforcement learning of reasoning models.
>
---
#### [replaced 010] A Systematic Approach to Predict the Impact of Cybersecurity Vulnerabilities Using LLMs
- **分类: cs.CR; cs.AI; cs.CL; cs.SE**

- **链接: [http://arxiv.org/pdf/2508.18439v2](http://arxiv.org/pdf/2508.18439v2)**

> **作者:** Anders Mølmen Høst; Pierre Lison; Leon Moonen
>
> **备注:** Accepted for publication in the 24th IEEE International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom 2025)
>
> **摘要:** Vulnerability databases, such as the National Vulnerability Database (NVD), offer detailed descriptions of Common Vulnerabilities and Exposures (CVEs), but often lack information on their real-world impact, such as the tactics, techniques, and procedures (TTPs) that adversaries may use to exploit the vulnerability. However, manually linking CVEs to their corresponding TTPs is a challenging and time-consuming task, and the high volume of new vulnerabilities published annually makes automated support desirable. This paper introduces TRIAGE, a two-pronged automated approach that uses Large Language Models (LLMs) to map CVEs to relevant techniques from the ATT&CK knowledge base. We first prompt an LLM with instructions based on MITRE's CVE Mapping Methodology to predict an initial list of techniques. This list is then combined with the results from a second LLM-based module that uses in-context learning to map a CVE to relevant techniques. This hybrid approach strategically combines rule-based reasoning with data-driven inference. Our evaluation reveals that in-context learning outperforms the individual mapping methods, and the hybrid approach improves recall of exploitation techniques. We also find that GPT-4o-mini performs better than Llama3.3-70B on this task. Overall, our results show that LLMs can be used to automatically predict the impact of cybersecurity vulnerabilities and TRIAGE makes the process of mapping CVEs to ATT&CK more efficient. A replication package is available for download from https://doi.org/10.5281/zenodo.17341503. Keywords: vulnerability impact, CVE, ATT&CK techniques, large language models, automated mapping.
>
---
#### [replaced 011] Infinity Parser: Layout Aware Reinforcement Learning for Scanned Document Parsing
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.03197v2](http://arxiv.org/pdf/2506.03197v2)**

> **作者:** Baode Wang; Biao Wu; Weizhen Li; Meng Fang; Zuming Huang; Jun Huang; Haozhe Wang; Yanjie Liang; Ling Chen; Wei Chu; Yuan Qi
>
> **备注:** 16 pages, 12 figures
>
> **摘要:** Automated parsing of scanned documents into richly structured, machine-readable formats remains a critical bottleneck in Document AI, as traditional multi-stage pipelines suffer from error propagation and limited adaptability to diverse layouts. We introduce layoutRL, an end-to-end reinforcement learning framework that trains models to be explicitly layout-aware by optimizing a composite reward of normalized edit distance, paragraph count accuracy, and reading order preservation. Leveraging our newly released dataset, Infinity-Doc-55K, which combines 55K high-fidelity synthetic scanned document parsing data with expert-filtered real-world documents, we instantiate layoutRL in a vision-language-model-based parser called Infinity-Parser. Evaluated on English and Chinese benchmarks for OCR, table and formula extraction, and reading order detection, Infinity-Parser achieves new state-of-the-art performance in both accuracy and structural fidelity, outpacing specialist pipelines and general-purpose vision-language models. We will publicly release our code and dataset to accelerate progress in robust document understanding.
>
---
#### [replaced 012] GRIFFIN: Effective Token Alignment for Faster Speculative Decoding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11018v3](http://arxiv.org/pdf/2502.11018v3)**

> **作者:** Shijing Hu; Jingyang Li; Xingyu Xie; Zhihui Lu; Kim-Chuan Toh; Pan Zhou
>
> **摘要:** Speculative decoding accelerates inference in large language models (LLMs) by generating multiple draft tokens simultaneously. However, existing methods often struggle with token misalignment between the training and decoding phases, limiting their performance. To address this, we propose GRIFFIN, a novel framework that incorporates a token-alignable training strategy and a token-alignable draft model to mitigate misalignment. The training strategy employs a loss masking mechanism to exclude highly misaligned tokens during training, preventing them from negatively impacting the draft model's optimization. The token-alignable draft model introduces input tokens to correct inconsistencies in generated features. Experiments on LLaMA, Vicuna, Qwen and Mixtral models demonstrate that GRIFFIN achieves an average acceptance length improvement of over 8% and a speedup ratio exceeding 7%, outperforming current speculative decoding state-of-the-art methods. Our code and GRIFFIN's draft models are released publicly in https://github.com/hsj576/GRIFFIN.
>
---
#### [replaced 013] SATA-BENCH: Select All That Apply Benchmark for Multiple Choice Questions
- **分类: cs.CL; cs.AI; 68T01; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.00643v3](http://arxiv.org/pdf/2506.00643v3)**

> **作者:** Weijie Xu; Shixian Cui; Xi Fang; Chi Xue; Stephanie Eckman; Chandan K. Reddy
>
> **备注:** 40 pages, 13 figures
>
> **摘要:** Large language models (LLMs) are increasingly evaluated on single-answer multiple-choice tasks, yet many real-world problems require identifying all correct answers from a set of options. This capability remains underexplored. We introduce SATA-BENCH, the first dedicated benchmark for evaluating LLMs on Select All That Apply (SATA) questions across diverse domains, including reading comprehension, law, and biomedicine. Our evaluation of 27 open-source and proprietary models reveals a significant gap: even the strongest model achieves only 41.8% exact match, exposing LLMs' inability to reliably identify all correct answers. We find that this weakness stems from two core challenges: selection bias - models favor certain choices regardless of content, and count bias - models fail to predict the correct number of answers. To address these issues, we propose Choice Funnel, a decoding strategy that combines token debiasing with adaptive thresholding to guide models toward complete and accurate selections. Choice Funnel achieves up to 29% higher exact match than competitive baselines while reducing inference cost by over 64%. Our findings expose fundamental limitations in current LLMs and introduce a new framework for diagnosing and improving multi-answer reasoning. We release SATA-BENCH and Choice Funnel to promote LLM development for robust decision-making in realistic, multi-answer applications.
>
---
#### [replaced 014] ConsintBench: Evaluating Language Models on Real-World Consumer Intent Understanding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.13499v2](http://arxiv.org/pdf/2510.13499v2)**

> **作者:** Xiaozhe Li; TianYi Lyu; Siyi Yang; Yuxi Gong; Yizhao Yang; Jinxuan Huang; Ligao Zhang; Zhuoyi Huang; Qingwen Liu
>
> **摘要:** Understanding human intent is a complex, high-level task for large language models (LLMs), requiring analytical reasoning, contextual interpretation, dynamic information aggregation, and decision-making under uncertainty. Real-world public discussions, such as consumer product discussions, are rarely linear or involve a single user. Instead, they are characterized by interwoven and often conflicting perspectives, divergent concerns, goals, emotional tendencies, as well as implicit assumptions and background knowledge about usage scenarios. To accurately understand such explicit public intent, an LLM must go beyond parsing individual sentences; it must integrate multi-source signals, reason over inconsistencies, and adapt to evolving discourse, similar to how experts in fields like politics, economics, or finance approach complex, uncertain environments. Despite the importance of this capability, no large-scale benchmark currently exists for evaluating LLMs on real-world human intent understanding, primarily due to the challenges of collecting real-world public discussion data and constructing a robust evaluation pipeline. To bridge this gap, we introduce \bench, the first dynamic, live evaluation benchmark specifically designed for intent understanding, particularly in the consumer domain. \bench is the largest and most diverse benchmark of its kind, supporting real-time updates while preventing data contamination through an automated curation pipeline.
>
---
#### [replaced 015] PsychCounsel-Bench: Evaluating the Psychology Intelligence of Large Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.01611v3](http://arxiv.org/pdf/2510.01611v3)**

> **作者:** Min Zeng
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable success across a wide range of industries, primarily due to their impressive generative abilities. Yet, their potential in applications requiring cognitive abilities, such as psychological counseling, remains largely untapped. This paper investigates the key question: \textit{Can LLMs be effectively applied to psychological counseling?} To determine whether an LLM can effectively take on the role of a psychological counselor, the first step is to assess whether it meets the qualifications required for such a role, namely the ability to pass the U.S. National Counselor Certification Exam (NCE). This is because, just as a human counselor must pass a certification exam to practice, an LLM must demonstrate sufficient psychological knowledge to meet the standards required for such a role. To address this, we introduce PsychCounsel-Bench, a benchmark grounded in U.S.national counselor examinations, a licensure test for professional counselors that requires about 70\% accuracy to pass. PsychCounsel-Bench comprises approximately 2,252 carefully curated single-choice questions, crafted to require deep understanding and broad enough to cover various sub-disciplines of psychology. This benchmark provides a comprehensive assessment of an LLM's ability to function as a counselor. Our evaluation shows that advanced models such as GPT-4o, Llama3.3-70B, and Gemma3-27B achieve well above the passing threshold, while smaller open-source models (e.g., Qwen2.5-7B, Mistral-7B) remain far below it. These results suggest that only frontier LLMs are currently capable of meeting counseling exam standards, highlighting both the promise and the challenges of developing psychology-oriented LLMs. We release the proposed dataset for public use: https://github.com/cloversjtu/PsychCounsel-Bench
>
---
#### [replaced 016] MotionGPT3: Human Motion as a Second Modality
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.24086v2](http://arxiv.org/pdf/2506.24086v2)**

> **作者:** Bingfan Zhu; Biao Jiang; Sunyi Wang; Shixiang Tang; Tao Chen; Linjie Luo; Youyi Zheng; Xin Chen
>
> **备注:** 26 pages, 11 figures
>
> **摘要:** With the rapid progress of large language models (LLMs), multimodal frameworks that unify understanding and generation have become promising, yet they face increasing complexity as the number of modalities and tasks grows. We observe that motion quantization introduces approximation errors that cap motion quality, and that unifying discrete text and continuous motion within a single-stream backbone amplifies cross-modal interference. Motivated by recent multi-branch Transformer designs that separate signals from different modalities, we propose MotionGPT3, a bimodal motion-language model for both understanding and generation. MotionGPT3 encodes raw motion into a continuous latent space using a variational autoencoder (VAE), thereby avoiding quantization-induced artifacts, while leveraging the semantic prior of pretrained language models. A dual-stream Transformer with shared attention preserves modality-specific routes while enabling controlled, bidirectional information flow, which reduces interference, stabilizing optimization, and empirically accelerates convergence without degrading fidelity. For multimodal joint training, a generate-then-align three-stage schedule further improves stability and limits cross-task interference. Experiments show that MotionGPT3 achieves 2x faster convergence in training loss and up to 4x faster convergence in validation, while maintaining state-of-the-art performance on standard motion understanding and motion generation benchmarks.
>
---
#### [replaced 017] Thinking Out Loud: Do Reasoning Models Know When They're Right?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.06564v3](http://arxiv.org/pdf/2504.06564v3)**

> **作者:** Qingcheng Zeng; Weihao Xuan; Leyang Cui; Rob Voigt
>
> **备注:** EMNLP 2025
>
> **摘要:** Large reasoning models (LRMs) have recently demonstrated impressive capabilities in complex reasoning tasks by leveraging increased test-time computation and exhibiting behaviors reminiscent of human-like self-reflection. While LRMs show a clear capacity for valuable self-reflection, how this ability interacts with other model behaviors remains underexplored. We investigate this connection by analyzing verbalized confidence, how models articulate their certainty, as a lens into the nature of self-reflection in LRMs. We find that supervised fine-tuning on reasoning traces (i.e., distillation) and reinforcement learning can improve verbalized calibration in reasoning-intensive settings in a progressive, laddered fashion. However, our results also indicate that reasoning models may possess a diminished awareness of their own knowledge boundaries, as evidenced by significantly lower "I don't know" response rates on factuality benchmarks. Moreover, we examine the relationship between verbalized confidence and reasoning chains, finding that models tend to express higher confidence when providing shorter or less elaborate reasoning. Our findings highlight how reasoning-oriented training can enhance performance in reasoning-centric tasks while potentially incurring a "reasoning tax," a cost reflected in the model's reduced ability to accurately recognize the limits of its own knowledge in small-scale models. More broadly, our work showcases how this erosion of knowledge boundaries can compromise model faithfulness, as models grow more confident without a commensurate understanding of when they should abstain.
>
---
#### [replaced 018] Forecasting Clinical Risk from Textual Time Series: Structuring Narratives for Temporal AI in Healthcare
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.10340v4](http://arxiv.org/pdf/2504.10340v4)**

> **作者:** Shahriar Noroozizadeh; Sayantan Kumar; Jeremy C. Weiss
>
> **备注:** AAAI AI for Social Impact 2026. Shahriar Noroozizadeh, Sayantan Kumar (authors contributed equally)
>
> **摘要:** Clinical case reports encode temporal patient trajectories that are often underexploited by traditional machine learning methods relying on structured data. In this work, we introduce the forecasting problem from textual time series, where timestamped clinical findings -- extracted via an LLM-assisted annotation pipeline -- serve as the primary input for prediction. We systematically evaluate a diverse suite of models, including fine-tuned decoder-based large language models and encoder-based transformers, on tasks of event occurrence prediction, temporal ordering, and survival analysis. Our experiments reveal that encoder-based models consistently achieve higher F1 scores and superior temporal concordance for short- and long-horizon event forecasting, while fine-tuned masking approaches enhance ranking performance. In contrast, instruction-tuned decoder models demonstrate a relative advantage in survival analysis, especially in early prognosis settings. Our sensitivity analyses further demonstrate the importance of time ordering, which requires clinical time series construction, as compared to text ordering, the format of the text inputs that LLMs are classically trained on. This highlights the additional benefit that can be ascertained from time-ordered corpora, with implications for temporal tasks in the era of widespread LLM use.
>
---
#### [replaced 019] MedScore: Generalizable Factuality Evaluation of Free-Form Medical Answers by Domain-adapted Claim Decomposition and Verification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18452v2](http://arxiv.org/pdf/2505.18452v2)**

> **作者:** Heyuan Huang; Alexandra DeLucia; Vijay Murari Tiyyala; Mark Dredze
>
> **备注:** Added generalizability experiment and examples on non-medical free-form answer. Added ablation study for MedCorp verification corpus and MedScore decomposition prompt
>
> **摘要:** While Large Language Models (LLMs) can generate fluent and convincing responses, they are not necessarily correct. This is especially apparent in the popular decompose-then-verify factuality evaluation pipeline, where LLMs evaluate generations by decomposing the generations into individual, valid claims. Factuality evaluation is especially important for medical answers, since incorrect medical information could seriously harm the patient. However, existing factuality systems are a poor match for the medical domain, as they are typically only evaluated on objective, entity-centric, formulaic texts such as biographies and historical topics. This differs from condition-dependent, conversational, hypothetical, sentence-structure diverse, and subjective medical answers, which makes decomposition into valid facts challenging. We propose MedScore, a new pipeline to decompose medical answers into condition-aware valid facts and verify against in-domain corpora. Our method extracts up to three times more valid facts than existing methods, reducing hallucination and vague references, and retaining condition-dependency in facts. The resulting factuality score substantially varies by decomposition method, verification corpus, and used backbone LLM, highlighting the importance of customizing each step for reliable factuality evaluation by using our generalizable and modularized pipeline for domain adaptation.
>
---
#### [replaced 020] MiLQ: Benchmarking IR Models for Bilingual Web Search with Mixed Language Queries
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16631v2](http://arxiv.org/pdf/2505.16631v2)**

> **作者:** Jonghwi Kim; Deokhyung Kang; Seonjeong Hwang; Yunsu Kim; Jungseul Ok; Gary Lee
>
> **备注:** 17 pages, 9 figures, EMNLP 2025 Main Conference
>
> **摘要:** Despite bilingual speakers frequently using mixed-language queries in web searches, Information Retrieval (IR) research on them remains scarce. To address this, we introduce MiLQ, Mixed-Language Query test set, the first public benchmark of mixed-language queries, qualified as realistic and relatively preferred. Experiments show that multilingual IR models perform moderately on MiLQ and inconsistently across native, English, and mixed-language queries, also suggesting code-switched training data's potential for robust IR models handling such queries. Meanwhile, intentional English mixing in queries proves an effective strategy for bilinguals searching English documents, which our analysis attributes to enhanced token matching compared to native queries.
>
---
#### [replaced 021] Enhancing Efficiency and Exploration in Reinforcement Learning for LLMs
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18573v2](http://arxiv.org/pdf/2505.18573v2)**

> **作者:** Mengqi Liao; Xiangyu Xi; Ruinian Chen; Jia Leng; Yangen Hu; Ke Zeng; Shuai Liu; Huaiyu Wan
>
> **备注:** Accept by EMNLP 2025 main
>
> **摘要:** Reasoning large language models (LLMs) excel in complex tasks, which has drawn significant attention to reinforcement learning (RL) for LLMs. However, existing approaches allocate an equal number of rollouts to all questions during the RL process, which is inefficient. This inefficiency stems from the fact that training on simple questions yields limited gains, whereas more rollouts are needed for challenging questions to sample correct answers. Furthermore, while RL improves response precision, it limits the model's exploration ability, potentially resulting in a performance cap below that of the base model prior to RL. To address these issues, we propose a mechanism for dynamically allocating rollout budgets based on the difficulty of the problems, enabling more efficient RL training. Additionally, we introduce an adaptive dynamic temperature adjustment strategy to maintain the entropy at a stable level, thereby encouraging sufficient exploration. This enables LLMs to improve response precision while preserving their exploratory ability to uncover potential correct pathways. The code and data is available on: https://github.com/LiaoMengqi/E3-RL4LLMs
>
---
#### [replaced 022] Audit-of-Understanding: Posterior-Constrained Inference for Mathematical Reasoning in Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.10252v2](http://arxiv.org/pdf/2510.10252v2)**

> **作者:** Samir Abdaljalil; Erchin Serpedin; Khalid Qaraqe; Hasan Kurban
>
> **摘要:** Large language models (LLMs) often generate reasoning traces that appear coherent but rest on unsupported assumptions, leading to hallucinated conclusions. Prior work mainly addresses factual hallucinations or relies on post-hoc verification, leaving reasoning-induced hallucinations largely unaddressed. We propose Audit-of-Understanding (AoU), a framework that constrains inference to validated premises through three phases: (1) decomposing a query into candidate assumptions, (2) auditing their support, and (3) conditioning inference only on the validated subset. Formally, AoU is \emph{posterior-constrained inference}, connecting to selective prediction and rejection learning. Our contributions are threefold: (i) theoretical guarantees under perfect validation, (ii) excess-risk bounds under imperfect audits, and (iii) tractability analysis. Empirically, AoU improves both accuracy and faithfulness on GSM8K, MultiArith, and SVAMP, achieving up to +30% gains on GSM8K, +45% on MultiArith, and consistent +20--28% improvements on SVAMP over Chain-of-Thought, Self-Consistency, and CoT-Decoding. Code is available at https://anonymous.4open.science/r/audit-of-understanding-E28B.
>
---
#### [replaced 023] Auto-Prompt Generation is Not Robust: Prompt Optimization Driven by Pseudo Gradient
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.18196v3](http://arxiv.org/pdf/2412.18196v3)**

> **作者:** Zeru Shi; Zhenting Wang; Yongye Su; Weidi Luo; Hang Gao; Fan Yang; Ruixiang Tang; Yongfeng Zhang
>
> **摘要:** While automatic prompt generation methods have recently received significant attention, their robustness remains poorly understood. In this paper, we introduce PertBench, a comprehensive benchmark dataset that includes a wide range of input perturbations, designed to systematically evaluate the robustness of current auto-prompting techniques. Our analysis reveals substantial vulnerabilities in existing prompt generation strategies, where even minor modifications to the prompt can lead to significant differences in model output. To address this issue, we propose PGO, a gradient-free prompt generation framework that leverages perturbation types as pseudo-gradient signals to guide LLMs in producing more robust prompts. In contrast to existing methods that assess prompt quality only on clean, well-structured inputs, our approach explicitly emphasizes robustness under noisy and perturbed conditions. Extensive experiments across diverse tasks and multiple LLMs show PGO consistently outperforms previous methods in maintaining performance under input perturbations.
>
---
#### [replaced 024] A Survey of Scientific Large Language Models: From Data Foundations to Agent Frontiers
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.21148v2](http://arxiv.org/pdf/2508.21148v2)**

> **作者:** Ming Hu; Chenglong Ma; Wei Li; Wanghan Xu; Jiamin Wu; Jucheng Hu; Tianbin Li; Guohang Zhuang; Jiaqi Liu; Yingzhou Lu; Ying Chen; Chaoyang Zhang; Cheng Tan; Jie Ying; Guocheng Wu; Shujian Gao; Pengcheng Chen; Jiashi Lin; Haitao Wu; Lulu Chen; Fengxiang Wang; Yuanyuan Zhang; Xiangyu Zhao; Feilong Tang; Encheng Su; Junzhi Ning; Xinyao Liu; Ye Du; Changkai Ji; Pengfei Jiang; Cheng Tang; Ziyan Huang; Jiyao Liu; Jiaqi Wei; Yuejin Yang; Xiang Zhang; Guangshuai Wang; Yue Yang; Huihui Xu; Ziyang Chen; Yizhou Wang; Chen Tang; Jianyu Wu; Yuchen Ren; Siyuan Yan; Zhonghua Wang; Zhongxing Xu; Shiyan Su; Shangquan Sun; Runkai Zhao; Zhisheng Zhang; Dingkang Yang; Jinjie Wei; Jiaqi Wang; Jiahao Xu; Jiangtao Yan; Wenhao Tang; Hongze Zhu; Yu Liu; Fudi Wang; Yiqing Shen; Yuanfeng Ji; Yanzhou Su; Tong Xie; Hongming Shan; Chun-Mei Feng; Zhi Hou; Diping Song; Lihao Liu; Yanyan Huang; Lequan Yu; Bin Fu; Shujun Wang; Xiaomeng Li; Xiaowei Hu; Yun Gu; Ben Fei; Benyou Wang; Yuewen Cao; Minjie Shen; Jie Xu; Haodong Duan; Fang Yan; Hongxia Hao; Jielan Li; Jiajun Du; Yanbo Wang; Imran Razzak; Zhongying Deng; Chi Zhang; Lijun Wu; Conghui He; Zhaohui Lu; Jinhai Huang; Wenqi Shao; Yihao Liu; Siqi Luo; Yi Xin; Xiaohong Liu; Fenghua Ling; Yuqiang Li; Aoran Wang; Siqi Sun; Qihao Zheng; Nanqing Dong; Tianfan Fu; Dongzhan Zhou; Yan Lu; Wenlong Zhang; Jin Ye; Jianfei Cai; Yirong Chen; Wanli Ouyang; Yu Qiao; Zongyuan Ge; Shixiang Tang; Junjun He; Chunfeng Song; Lei Bai; Bowen Zhou
>
> **摘要:** Scientific Large Language Models (Sci-LLMs) are transforming how knowledge is represented, integrated, and applied in scientific research, yet their progress is shaped by the complex nature of scientific data. This survey presents a comprehensive, data-centric synthesis that reframes the development of Sci-LLMs as a co-evolution between models and their underlying data substrate. We formulate a unified taxonomy of scientific data and a hierarchical model of scientific knowledge, emphasizing the multimodal, cross-scale, and domain-specific challenges that differentiate scientific corpora from general natural language processing datasets. We systematically review recent Sci-LLMs, from general-purpose foundations to specialized models across diverse scientific disciplines, alongside an extensive analysis of over 270 pre-/post-training datasets, showing why Sci-LLMs pose distinct demands -- heterogeneous, multi-scale, uncertainty-laden corpora that require representations preserving domain invariance and enabling cross-modal reasoning. On evaluation, we examine over 190 benchmark datasets and trace a shift from static exams toward process- and discovery-oriented assessments with advanced evaluation protocols. These data-centric analyses highlight persistent issues in scientific data development and discuss emerging solutions involving semi-automated annotation pipelines and expert validation. Finally, we outline a paradigm shift toward closed-loop systems where autonomous agents based on Sci-LLMs actively experiment, validate, and contribute to a living, evolving knowledge base. Collectively, this work provides a roadmap for building trustworthy, continually evolving artificial intelligence (AI) systems that function as a true partner in accelerating scientific discovery.
>
---
#### [replaced 025] Automated Knowledge Component Generation for Interpretable Knowledge Tracing in Coding Problems
- **分类: cs.AI; cs.CL; cs.CY; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2502.18632v3](http://arxiv.org/pdf/2502.18632v3)**

> **作者:** Zhangqi Duan; Nigel Fernandez; Arun Balajiee Lekshmi Narayanan; Mohammad Hassany; Rafaella Sampaio de Alencar; Peter Brusilovsky; Bita Akram; Andrew Lan
>
> **摘要:** Knowledge components (KCs) mapped to problems help model student learning, tracking their mastery levels on fine-grained skills thereby facilitating personalized learning and feedback in online learning platforms. However, crafting and tagging KCs to problems, traditionally performed by human domain experts, is highly labor intensive. We present an automated, LLM-based pipeline for KC generation and tagging for open-ended programming problems. We also develop an LLM-based knowledge tracing (KT) framework to leverage these LLM-generated KCs, which we refer to as KCGen-KT. We conduct extensive quantitative and qualitative evaluations on two real-world student code submission datasets in different programming languages.We find that KCGen-KT outperforms existing KT methods and human-written KCs on future student response prediction. We investigate the learning curves of generated KCs and show that LLM-generated KCs result in a better fit than human written KCs under a cognitive model. We also conduct a human evaluation with course instructors to show that our pipeline generates reasonably accurate problem-KC mappings.
>
---
#### [replaced 026] DOGe: Defensive Output Generation for LLM Protection Against Knowledge Distillation
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19504v2](http://arxiv.org/pdf/2505.19504v2)**

> **作者:** Pingzhi Li; Zhen Tan; Mohan Zhang; Huaizhi Qu; Huan Liu; Tianlong Chen
>
> **备注:** Code is available at https://github.com/unites-lab/doge
>
> **摘要:** Large Language Models (LLMs) represent substantial intellectual and economic investments, yet their effectiveness can inadvertently facilitate model imitation via knowledge distillation (KD). In practical scenarios, competitors can distill proprietary LLM capabilities by simply observing publicly accessible outputs, akin to reverse-engineering a complex performance by observation alone. Existing protective methods like watermarking only identify imitation post-hoc, while other defenses assume the student model mimics the teacher's internal logits, rendering them ineffective against distillation purely from observed output text. This paper confronts the challenge of actively protecting LLMs within the realistic constraints of API-based access. We introduce an effective and efficient Defensive Output Generation (DOGe) strategy that subtly modifies the output behavior of an LLM. Its outputs are accurate and useful for legitimate users, yet are designed to be misleading for distillation, significantly undermining imitation attempts. We achieve this by fine-tuning only the final linear layer of the teacher LLM with an adversarial loss. This targeted training approach anticipates and disrupts distillation attempts during inference time. Our experiments show that, while preserving the performance of the teacher model, student models distilled from the defensively generated outputs demonstrate catastrophically reduced performance, demonstrating DOGe as a practical safeguard against KD-based model imitation.
>
---
#### [replaced 027] A Controllable Examination for Long-Context Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.02921v2](http://arxiv.org/pdf/2506.02921v2)**

> **作者:** Yijun Yang; Zeyu Huang; Wenhao Zhu; Zihan Qiu; Fei Yuan; Jeff Z. Pan; Ivan Titov
>
> **备注:** NeurIPS 2025 Dataset and Benchmark Track Spotlight
>
> **摘要:** Existing frameworks for evaluating long-context language models (LCLM) can be broadly categorized into real-world applications (e.g, document summarization) and synthetic tasks (e.g, needle-in-a-haystack). Despite their utility, both approaches are accompanied by certain intrinsic limitations. Real-world tasks often involve complexity that makes interpretation challenging and suffer from data contamination, whereas synthetic tasks frequently lack meaningful coherence between the target information (needle) and its surrounding context (haystack), undermining their validity as proxies for realistic applications. In response to these challenges, we posit that an ideal long-context evaluation framework should be characterized by three essential features: 1) seamless context 2) controllable setting and 3) sound evaluation. This study introduces $\textbf{LongBioBench}$, a benchmark that utilizes artificially generated biographies as a controlled environment for assessing LCLMs across dimensions of understanding, reasoning, and trustworthiness. Our experimental evaluation, which includes 18 LCLMs in total, demonstrates that most models still exhibit deficiencies in semantic understanding and elementary reasoning over retrieved results and are less trustworthy as context length increases. Our further analysis indicates some design choices employed by existing synthetic benchmarks, such as contextual non-coherence, numerical needles, and the absence of distractors, rendering them vulnerable to test the model's long-context capabilities. To sum up, compared to previous synthetic benchmarks, LongBioBench achieves a better trade-off between mirroring authentic language tasks and maintaining controllability, and is highly interpretable and configurable.
>
---
#### [replaced 028] Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18752v2](http://arxiv.org/pdf/2505.18752v2)**

> **作者:** Haolin Yang; Hakaze Cho; Yiqiao Zhong; Naoya Inoue
>
> **备注:** 52 pages, 70 figures, 24 tables, NeurIPS 2025
>
> **摘要:** The unusual properties of in-context learning (ICL) have prompted investigations into the internal mechanisms of large language models. Prior work typically focuses on either special attention heads or task vectors at specific layers, but lacks a unified framework linking these components to the evolution of hidden states across layers that ultimately produce the model's output. In this paper, we propose such a framework for ICL in classification tasks by analyzing two geometric factors that govern performance: the separability and alignment of query hidden states. A fine-grained analysis of layer-wise dynamics reveals a striking two-stage mechanism: separability emerges in early layers, while alignment develops in later layers. Ablation studies further show that Previous Token Heads drive separability, while Induction Heads and task vectors enhance alignment. Our findings thus bridge the gap between attention heads and task vectors, offering a unified account of ICL's underlying mechanisms.
>
---
#### [replaced 029] Value-Based Large Language Model Agent Simulation for Mutual Evaluation of Trust and Interpersonal Closeness
- **分类: cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2507.11979v2](http://arxiv.org/pdf/2507.11979v2)**

> **作者:** Yuki Sakamoto; Takahisa Uchida; Hiroshi Ishiguro
>
> **摘要:** Large language models (LLMs) have emerged as powerful tools for simulating complex social phenomena using human-like agents with specific traits. In human societies, value similarity is important for building trust and close relationships; however, it remains unexplored whether this principle holds true in artificial societies comprising LLM agents. Therefore, this study investigates the influence of value similarity on relationship-building among LLM agents through two experiments. First, in a preliminary experiment, we evaluated the controllability of values in LLMs to identify the most effective model and prompt design for controlling the values. Subsequently, in the main experiment, we generated pairs of LLM agents imbued with specific values and analyzed their mutual evaluations of trust and interpersonal closeness following a dialogue. The experiments were conducted in English and Japanese to investigate language dependence. The results confirmed that pairs of agents with higher value similarity exhibited greater mutual trust and interpersonal closeness. Our findings demonstrate that the LLM agent simulation serves as a valid testbed for social science theories, contributes to elucidating the mechanisms by which values influence relationship building, and provides a foundation for inspiring new theories and insights into the social sciences.
>
---
#### [replaced 030] Concise and Sufficient Sub-Sentence Citations for Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.20859v2](http://arxiv.org/pdf/2509.20859v2)**

> **作者:** Guo Chen; Qiuyuan Li; Qiuxian Li; Hongliang Dai; Xiang Chen; Piji Li
>
> **摘要:** In retrieval-augmented generation (RAG) question answering systems, generating citations for large language model (LLM) outputs enhances verifiability and helps users identify potential hallucinations. However, we observe two problems in the citations produced by existing attribution methods. First, the citations are typically provided at the sentence or even paragraph level. Long sentences or paragraphs may include a substantial amount of irrelevant content. Second, sentence-level citations may omit information that is essential for verifying the output, forcing users to read the surrounding context. In this paper, we propose generating sub-sentence citations that are both concise and sufficient, thereby reducing the effort required by users to confirm the correctness of the generated output. To this end, we first develop annotation guidelines for such citations and construct a corresponding dataset. Then, we propose an attribution framework for generating citations that adhere to our standards. This framework leverages LLMs to automatically generate fine-tuning data for our task and employs a credit model to filter out low-quality examples. Our experiments on the constructed dataset demonstrate that the propose approach can generate high-quality and more readable citations.
>
---
#### [replaced 031] RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.00222v4](http://arxiv.org/pdf/2508.00222v4)**

> **作者:** Yihong Dong; Xue Jiang; Yongding Tao; Huanyu Liu; Kechi Zhang; Lili Mou; Rongyu Cao; Yingwei Ma; Jue Chen; Binhua Li; Zhi Jin; Fei Huang; Yongbin Li; Ge Li
>
> **摘要:** Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its essentially on-policy strategy coupled with LLM's immense action space and sparse reward. Critically, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel hybrid-policy optimization approach for LLMs that synergizes internal exploitation with external data to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components, i.e., Multiple Importance Sampling to address distributional mismatch from external data, and Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. Compared with existing RLVR methods, RL-PLUS achieves 1) state-of-the-art performance on six math reasoning benchmarks; 2) superior performance on six out-of-distribution reasoning tasks; 3) consistent and significant gains across diverse model families, with average relative improvements up to 69.2\%. Moreover, the analysis of Pass@k curves indicates that RL-PLUS effectively resolves the capability boundary collapse problem.
>
---
#### [replaced 032] Large-scale User Game Lifecycle Representation Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.15412v2](http://arxiv.org/pdf/2510.15412v2)**

> **作者:** Yanjie Gou; Jiangming Liu; Kouying Xue; Yi Hu
>
> **摘要:** The rapid expansion of video game production necessitates the development of effective advertising and recommendation systems for online game platforms. Recommending and advertising games to users hinges on capturing their interest in games. However, existing representation learning methods crafted for handling billions of items in recommendation systems are unsuitable for game advertising and recommendation. This is primarily due to game sparsity, where the mere hundreds of games fall short for large-scale user representation learning, and game imbalance, where user behaviors are overwhelmingly dominated by a handful of popular games. To address the sparsity issue, we introduce the User Game Lifecycle (UGL), designed to enrich user behaviors in games. Additionally, we propose two innovative strategies aimed at manipulating user behaviors to more effectively extract both short and long-term interests. To tackle the game imbalance challenge, we present an Inverse Probability Masking strategy for UGL representation learning. The offline and online experimental results demonstrate that the UGL representations significantly enhance model by achieving a 1.83% AUC offline increase on average and a 21.67% CVR online increase on average for game advertising and a 0.5% AUC offline increase and a 0.82% ARPU online increase for in-game item recommendation.
>
---
#### [replaced 033] BASIL: Bayesian Assessment of Sycophancy in LLMs
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.16846v2](http://arxiv.org/pdf/2508.16846v2)**

> **作者:** Katherine Atwell; Pedram Heydari; Anthony Sicilia; Malihe Alikhani
>
> **摘要:** Sycophancy (overly agreeable or flattering behavior) is critical to understand in the context of human-AI collaboration, especially in decision-making settings like health, law, and education. Existing methods for studying sycophancy in LLMs are either descriptive (study behavior change when sycophancy is elicited) or normative (provide values-based judgment on behavior change). Together, these approaches help us understand the extent, and impacts, of sycophancy. However, existing normative approaches only apply for objective tasks where ground-truth data exists, ignoring the natural subjectivity in many NLP tasks. Drawing from behavioral economics and rational decision theory, we introduce an Bayesian framework to study the normative effects of sycophancy on rationality in LLMs, without requiring labeled ground-truth. Using this interdisciplinary framework, we study sycophantic behavior in multiple LLM baselines across three different tasks, experimenting with various methods for eliciting sycophancy and obtaining probability judgments from LLMs. We find significant evidence of sycophancy in our experiments (7 of 8 baselines for one of our probing techniques), and observe that sycophancy is more likely to reduce rationality than it is to increase rationality in LLMs' decisions when they are directly probed for probabilities (2 out of 4 baselines show significant increases overall).
>
---
#### [replaced 034] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11842v2](http://arxiv.org/pdf/2505.11842v2)**

> **作者:** Xuannan Liu; Zekun Li; Zheqi He; Peipei Li; Shuhan Xia; Xing Cui; Huaibo Huang; Xi Yang; Ran He
>
> **备注:** Accepted by NeurIPS 2025 Dataset and Benchmark Track, Project page: https://liuxuannan.github.io/Video-SafetyBench.github.io/
>
> **摘要:** The increasing deployment of Large Vision-Language Models (LVLMs) raises safety concerns under potential malicious inputs. However, existing multimodal safety evaluations primarily focus on model vulnerabilities exposed by static image inputs, ignoring the temporal dynamics of video that may induce distinct safety risks. To bridge this gap, we introduce Video-SafetyBench, the first comprehensive benchmark designed to evaluate the safety of LVLMs under video-text attacks. It comprises 2,264 video-text pairs spanning 48 fine-grained unsafe categories, each pairing a synthesized video with either a harmful query, which contains explicit malice, or a benign query, which appears harmless but triggers harmful behavior when interpreted alongside the video. To generate semantically accurate videos for safety evaluation, we design a controllable pipeline that decomposes video semantics into subject images (what is shown) and motion text (how it moves), which jointly guide the synthesis of query-relevant videos. To effectively evaluate uncertain or borderline harmful outputs, we propose RJScore, a novel LLM-based metric that incorporates the confidence of judge models and human-aligned decision threshold calibration. Extensive experiments show that benign-query video composition achieves average attack success rates of 67.2%, revealing consistent vulnerabilities to video-induced attacks. We believe Video-SafetyBench will catalyze future research into video-based safety evaluation and defense strategies.
>
---
#### [replaced 035] Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.00432v2](http://arxiv.org/pdf/2507.00432v2)**

> **作者:** Maggie Huan; Yuetai Li; Tuney Zheng; Xiaoyu Xu; Seungone Kim; Minxin Du; Radha Poovendran; Graham Neubig; Xiang Yue
>
> **摘要:** Math reasoning has become the poster child of progress in large language models (LLMs), with new models rapidly surpassing human-level performance on benchmarks like MATH and AIME. But as math leaderboards improve week by week, it is worth asking: do these gains reflect broader problem-solving ability or just narrow overfitting? To answer this question, we evaluate over 20 open-weight reasoning-tuned models across a broad suite of tasks, including math, scientific QA, agent planning, coding, and standard instruction-following. We surprisingly find that most models that succeed in math fail to transfer their gains to other domains. To rigorously study this phenomenon, we conduct controlled experiments on Qwen3-14B models using math-only data but different tuning methods. We find that reinforcement learning (RL)-tuned models generalize well across domains, while supervised fine-tuning (SFT)-tuned models often forget general capabilities. Latent-space representation and token-space distribution shift analyses reveal that SFT induces substantial representation and output drift, while RL preserves general-domain structure. Our results suggest a need to rethink standard post-training recipes, particularly the reliance on SFT-distilled data for advancing reasoning models.
>
---
#### [replaced 036] When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.01005v2](http://arxiv.org/pdf/2504.01005v2)**

> **作者:** Nishad Singhi; Hritik Bansal; Arian Hosseini; Aditya Grover; Kai-Wei Chang; Marcus Rohrbach; Anna Rohrbach
>
> **备注:** COLM 2025
>
> **摘要:** Scaling test-time compute has emerged as a key strategy for enhancing the reasoning capabilities of large language models (LLMs), particularly in tasks like mathematical problem-solving. A traditional approach, Self-Consistency (SC), generates multiple solutions to a problem and selects the most common answer via majority voting. Another common method involves scoring each solution with a reward model (verifier) and choosing the best one. Recent advancements in Generative Reward Models (GenRM) reframe verification as a next-token prediction task, enabling inference-time scaling along a new axis. Specifically, GenRM generates multiple verification chains-of-thought to score each solution. Under a limited inference budget, this introduces a fundamental trade-off: should you spend the budget on scaling solutions via SC or generate fewer solutions and allocate compute to verification via GenRM? To address this, we evaluate GenRM against SC under a fixed inference budget. Interestingly, we find that SC is more compute-efficient than GenRM for most practical inference budgets across diverse models and datasets. For instance, GenRM first matches SC after consuming up to 8x the inference compute and requires significantly more compute to outperform it. Furthermore, we derive inference scaling laws for the GenRM paradigm, revealing that compute-optimal inference favors scaling solution generation more aggressively than scaling the number of verifications. Our work provides practical guidance on optimizing test-time scaling by balancing solution generation and verification. The code is available at https://github.com/nishadsinghi/sc-genrm-scaling.
>
---
#### [replaced 037] From Sequence to Structure: Uncovering Substructure Reasoning in Transformers
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.10435v2](http://arxiv.org/pdf/2507.10435v2)**

> **作者:** Xinnan Dai; Kai Yang; Jay Revolinsky; Kai Guo; Aoran Wang; Bohang Zhang; Jiliang Tang
>
> **备注:** Camera Ready version for Neurips 2025
>
> **摘要:** Recent studies suggest that large language models (LLMs) possess the capability to solve graph reasoning tasks. Notably, even when graph structures are embedded within textual descriptions, LLMs can still effectively answer related questions. This raises a fundamental question: How can a decoder-only Transformer architecture understand underlying graph structures? To address this, we start with the substructure extraction task, interpreting the inner mechanisms inside the transformers and analyzing the impact of the input queries. Specifically, through both empirical results and theoretical analysis, we present Induced Substructure Filtration (ISF), a perspective that captures the substructure identification in the multi-layer transformers. We further validate the ISF process in LLMs, revealing consistent internal dynamics across layers. Building on these insights, we explore the broader capabilities of Transformers in handling diverse graph types. Specifically, we introduce the concept of thinking in substructures to efficiently extract complex composite patterns, and demonstrate that decoder-only Transformers can successfully extract substructures from attributed graphs, such as molecular graphs. Together, our findings offer a new insight on how sequence-based Transformers perform the substructure extraction task over graph data.
>
---
#### [replaced 038] SpikingBrain: Spiking Brain-inspired Large Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.05276v2](http://arxiv.org/pdf/2509.05276v2)**

> **作者:** Yuqi Pan; Yupeng Feng; Jinghao Zhuang; Siyu Ding; Han Xu; Zehao Liu; Bohan Sun; Yuhong Chou; Xuerui Qiu; Anlin Deng; Anjie Hu; Peng Zhou; Man Yao; Jibin Wu; Jian Yang; Guoliang Sun; Bo Xu; Guoqi Li
>
> **摘要:** Mainstream Transformer-based large language models face major efficiency bottlenecks: training computation scales quadratically with sequence length, and inference memory grows linearly, limiting long-context processing. Building large models on non-NVIDIA platforms also poses challenges for stable and efficient training. To address this, we introduce SpikingBrain, a family of brain-inspired models designed for efficient long-context training and inference. SpikingBrain leverages the MetaX GPU cluster and focuses on three aspects: (1) Model Architecture: linear and hybrid-linear attention architectures with adaptive spiking neurons; (2) Algorithmic Optimizations: an efficient, conversion-based training pipeline and a dedicated spike coding framework; (3) System Engineering: customized training frameworks, operator libraries, and parallelism strategies tailored to MetaX hardware. Using these techniques, we develop two models: SpikingBrain-7B, a linear LLM, and SpikingBrain-76B, a hybrid-linear MoE LLM. These models demonstrate the feasibility of large-scale LLM development on non-NVIDIA platforms. SpikingBrain achieves performance comparable to open-source Transformer baselines while using only about 150B tokens for continual pre-training. Our models significantly improve long-sequence training efficiency and deliver inference with (partially) constant memory and event-driven spiking behavior. For example, SpikingBrain-7B attains over 100x speedup in Time to First Token for 4M-token sequences. Training remains stable for weeks on hundreds of MetaX C550 GPUs, with the 7B model reaching a Model FLOPs Utilization of 23.4 percent. The proposed spiking scheme achieves 69.15 percent sparsity, enabling low-power operation. Overall, this work demonstrates the potential of brain-inspired mechanisms to drive the next generation of efficient and scalable large model design.
>
---
#### [replaced 039] Consistency is Key: Disentangling Label Variation in Natural Language Processing with Intra-Annotator Agreement
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2301.10684v2](http://arxiv.org/pdf/2301.10684v2)**

> **作者:** Gavin Abercrombie; Tanvi Dinkar; Amanda Cercas Curry; Verena Rieser; Dirk Hovy
>
> **备注:** Accepted for publication in Proceedings of the Fourth Workshop on Perspectivist Approaches to NLP (NLPerspectives)
>
> **摘要:** We commonly use agreement measures to assess the utility of judgements made by human annotators in Natural Language Processing (NLP) tasks. While inter-annotator agreement is frequently used as an indication of label reliability by measuring consistency between annotators, we argue for the additional use of intra-annotator agreement to measure label stability (and annotator consistency) over time. However, in a systematic review, we find that the latter is rarely reported in this field. Calculating these measures can act as important quality control and could provide insights into why annotators disagree. We conduct exploratory annotation experiments to investigate the relationships between these measures and perceptions of subjectivity and ambiguity in text items, finding that annotators provide inconsistent responses around 25% of the time across four different NLP tasks.
>
---
#### [replaced 040] Customer-R1: Personalized Simulation of Human Behaviors via RL-based LLM Agent in Online Shopping
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.07230v2](http://arxiv.org/pdf/2510.07230v2)**

> **作者:** Ziyi Wang; Yuxuan Lu; Yimeng Zhang; Jing Huang; Dakuo Wang
>
> **摘要:** Simulating step-wise human behavior with Large Language Models (LLMs) has become an emerging research direction, enabling applications in various practical domains. While prior methods, including prompting, supervised fine-tuning (SFT), and reinforcement learning (RL), have shown promise in modeling step-wise behavior, they primarily learn a population-level policy without conditioning on a user's persona, yielding generic rather than personalized simulations. In this work, we pose a critical question: how can LLM agents better simulate personalized user behavior? We introduce Customer-R1, an RL-based method for personalized, step-wise user behavior simulation in online shopping environments. Our policy is conditioned on an explicit persona, and we optimize next-step rationale and action generation via action correctness reward signals. Experiments on the OPeRA dataset emonstrate that Customer-R1 not only significantly outperforms prompting and SFT-based baselines in next-action prediction tasks, but also better matches users' action distribution, indicating higher fidelity in personalized behavior simulation.
>
---
#### [replaced 041] Automated Evaluation of Meter and Rhyme in Russian Generative and Human-Authored Poetry
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20931v2](http://arxiv.org/pdf/2502.20931v2)**

> **作者:** Ilya Koziev
>
> **备注:** 7 pages, 1 figure, ver.2
>
> **摘要:** Generative poetry systems require effective tools for data engineering and automatic evaluation, particularly to assess how well a poem adheres to versification rules, such as the correct alternation of stressed and unstressed syllables and the presence of rhymes. In this work, we introduce the Russian Poetry Scansion Tool library designed for stress mark placement in Russian-language syllabo-tonic poetry, rhyme detection, and identification of defects of poeticness. Additionally, we release RIFMA -- a dataset of poem fragments spanning various genres and forms, annotated with stress marks. This dataset can be used to evaluate the capability of modern large language models to accurately place stress marks in poetic texts. The published resources provide valuable tools for researchers and practitioners in the field of creative generative AI, facilitating advancements in the development and evaluation of generative poetry systems.
>
---
#### [replaced 042] StarWhisper Telescope: An AI framework for automating end-to-end astronomical observations
- **分类: astro-ph.IM; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.06412v3](http://arxiv.org/pdf/2412.06412v3)**

> **作者:** Cunshi Wang; Yu Zhang; Yuyang Li; Xinjie Hu; Yiming Mao; Xunhao Chen; Pengliang Du; Rui Wang; Ying Wu; Hang Yang; Yansong Li; Beichuan Wang; Haiyang Mu; Zheng Wang; Jianfeng Tian; Liang Ge; Yongna Mao; Shengming Li; Xiaomeng Lu; Jinhang Zou; Yang Huang; Ningchen Sun; Jie Zheng; Min He; Yu Bai; Junjie Jin; Hong Wu; Jifeng Liu
>
> **备注:** 33 pages
>
> **摘要:** The exponential growth of large-scale telescope arrays has boosted time-domain astronomy development but introduced operational bottlenecks, including labor-intensive observation planning, data processing, and real-time decision-making. Here we present the StarWhisper Telescope system, an AI agent framework automating end-to-end astronomical observations for surveys like the Nearby Galaxy Supernovae Survey. By integrating large language models with specialized function calls and modular workflows, StarWhisper Telescope autonomously generates site-specific observation lists, executes real-time image analysis via pipelines, and dynamically triggers follow-up proposals upon transient detection. The system reduces human intervention through automated observation planning, telescope controlling and data processing, while enabling seamless collaboration between amateur and professional astronomers. Deployed across Nearby Galaxy Supernovae Survey's network of 10 amateur telescopes, the StarWhisper Telescope has detected transients with promising response times relative to existing surveys. Furthermore, StarWhisper Telescope's scalable agent architecture provides a blueprint for future facilities like the Global Open Transient Telescope Array, where AI-driven autonomy will be critical for managing 60 telescopes.
>
---
#### [replaced 043] DRIFT: Decompose, Retrieve, Illustrate, then Formalize Theorems
- **分类: cs.AI; cs.CL; cs.IR; cs.SC**

- **链接: [http://arxiv.org/pdf/2510.10815v3](http://arxiv.org/pdf/2510.10815v3)**

> **作者:** Meiru Zhang; Philipp Borchert; Milan Gritta; Gerasimos Lampouras
>
> **摘要:** Automating the formalization of mathematical statements for theorem proving remains a major challenge for Large Language Models (LLMs). LLMs struggle to identify and utilize the prerequisite mathematical knowledge and its corresponding formal representation in languages like Lean. Current retrieval-augmented autoformalization methods query external libraries using the informal statement directly, but overlook a fundamental limitation: informal mathematical statements are often complex and offer limited context on the underlying math concepts. To address this, we introduce DRIFT, a novel framework that enables LLMs to decompose informal mathematical statements into smaller, more tractable ''sub-components''. This facilitates targeted retrieval of premises from mathematical libraries such as Mathlib. Additionally, DRIFT retrieves illustrative theorems to help models use premises more effectively in formalization tasks. We evaluate DRIFT across diverse benchmarks (ProofNet, ConNF, and MiniF2F-test) and find that it consistently improves premise retrieval, nearly doubling the F1 score compared to the DPR baseline on ProofNet. Notably, DRIFT demonstrates strong performance on the out-of-distribution ConNF benchmark, with BEq+@10 improvements of 37.14% and 42.25% using GPT-4.1 and DeepSeek-V3.1, respectively. Our analysis shows that retrieval effectiveness in mathematical autoformalization depends heavily on model-specific knowledge boundaries, highlighting the need for adaptive retrieval strategies aligned with each model's capabilities.
>
---
#### [replaced 044] Planner and Executor: Collaboration between Discrete Diffusion And Autoregressive Models in Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.15244v2](http://arxiv.org/pdf/2510.15244v2)**

> **作者:** Lina Berrayana; Ahmed Heakl; Muhammad Abdullah Sohail; Thomas Hofmann; Salman Khan; Wei Chen
>
> **备注:** Under Submission
>
> **摘要:** Current autoregressive language models (ARMs) achieve high accuracy but require long token sequences, making them costly. Discrete diffusion language models (DDLMs) enable parallel and flexible generation within a fixed number of steps and have recently emerged for their strong performance in complex reasoning and long-term planning tasks. We present a study exploring hybrid architectures that couple DDLMs with ARMs to assess whether their collaboration can yield complementary benefits. We first examine collaboration in text space, where one model plans the reasoning process and another executes the final answer based on that plan. We then extend this setup to latent-space communication, introducing a learned projector that maps DDLM latents into the ARM's embedding space, potentially bypassing some of the text-generation limitations of diffusion models. We find that shifting DDLM --> ARM communication from text space to latent space yields significant accuracy gains, for example increasing from 27.0% to 54.0% on DART-5 and from 0.0% to 14.0% on AIME24. We also find that combining a DDLM planner with an ARM executor can provide substantial computational savings with little to no impact on accuracy. For example, the latent-space pipeline, using 64 tokens for planning and roughly 5 for execution, surpasses Qwen3.1-7B on DART-5 and AIME, despite Qwen using 44 times more tokens. Overall, our study offers new insights into reasoning with DDLMs and highlights their potential in hybrid architectures.
>
---
#### [replaced 045] VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.12081v2](http://arxiv.org/pdf/2508.12081v2)**

> **作者:** Haidong Xu; Guangwei Xu; Zhedong Zheng; Xiatian Zhu; Wei Ji; Xiangtai Li; Ruijie Guo; Meishan Zhang; Min zhang; Hao Fei
>
> **备注:** Accepted by NeurIPS 2025; Project Page: https://walkermitty.github.io/VimoRAG
>
> **摘要:** This paper introduces VimoRAG, a novel video-based retrieval-augmented motion generation framework for motion large language models (LLMs). As motion LLMs face severe out-of-domain/out-of-vocabulary issues due to limited annotated data, VimoRAG leverages large-scale in-the-wild video databases to enhance 3D motion generation by retrieving relevant 2D human motion signals. While video-based motion RAG is nontrivial, we address two key bottlenecks: (1) developing an effective motion-centered video retrieval model that distinguishes human poses and actions, and (2) mitigating the issue of error propagation caused by suboptimal retrieval results. We design the Gemini Motion Video Retriever mechanism and the Motion-centric Dual-alignment DPO Trainer, enabling effective retrieval and generation processes. Experimental results show that VimoRAG significantly boosts the performance of motion LLMs constrained to text-only input. All the resources are available at https://walkermitty.github.io/VimoRAG/
>
---
#### [replaced 046] Large Language Diffusion Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.09992v3](http://arxiv.org/pdf/2502.09992v3)**

> **作者:** Shen Nie; Fengqi Zhu; Zebin You; Xiaolu Zhang; Jingyang Ou; Jun Hu; Jun Zhou; Yankai Lin; Ji-Rong Wen; Chongxuan Li
>
> **摘要:** The capabilities of large language models (LLMs) are widely regarded as relying on autoregressive models (ARMs). We challenge this notion by introducing LLaDA, a diffusion model trained from scratch under the pre-training and supervised fine-tuning (SFT) paradigm. LLaDA employs a forward data masking process and a reverse generation process, parameterized by a Transformer to predict masked tokens. It provides a principled generative approach for probabilistic inference by optimizing a likelihood lower bound. Across extensive benchmarks on general tasks, math, code, and so on, LLaDA demonstrates strong scalability and performs comparably to our self-constructed ARM baselines. Remarkably, LLaDA 8B is competitive with strong LLMs like LLaMA3 8B in in-context learning and, after SFT, exhibits impressive instruction-following abilities in case studies such as multi-turn dialogue. Moreover, LLaDA addresses the reversal curse, surpassing GPT-4o in a reversal poem completion task. Our findings show the promise of diffusion models for language modeling at scale and challenge the common assumption that core LLM capabilities discussed above inherently depend on ARMs. Project page and codes: https://ml-gsai.github.io/LLaDA-demo/.
>
---
#### [replaced 047] Reassessing Active Learning Adoption in Contemporary NLP: A Community Survey
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.09701v3](http://arxiv.org/pdf/2503.09701v3)**

> **作者:** Julia Romberg; Christopher Schröder; Julius Gonsior; Katrin Tomanek; Fredrik Olsson
>
> **摘要:** Supervised learning relies on data annotation which usually is time-consuming and therefore expensive. A longstanding strategy to reduce annotation costs is active learning, an iterative process, in which a human annotates only data instances deemed informative by a model. Research in active learning has made considerable progress, especially with the rise of large language models (LLMs). However, we still know little about how these remarkable advances have translated into real-world applications, or contributed to removing key barriers to active learning adoption. To fill in this gap, we conduct an online survey in the NLP community to collect previously intangible insights on current implementation practices, common obstacles in application, and future prospects in active learning. We also reassess the perceived relevance of data annotation and active learning as fundamental assumptions. Our findings show that data annotation is expected to remain important and active learning to stay relevant while benefiting from LLMs. Consistent with a community survey from over 15 years ago, three key challenges yet persist -- setup complexity, uncertain cost reduction, and tooling -- for which we propose alleviation strategies. We publish an anonymized version of the dataset.
>
---
#### [replaced 048] Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.03304v3](http://arxiv.org/pdf/2502.03304v3)**

> **作者:** Qitao Tan; Jun Liu; Zheng Zhan; Caiwei Ding; Yanzhi Wang; Xiaolong Ma; Jaewoo Lee; Jin Lu; Geng Yuan
>
> **摘要:** Large language models (LLMs) excel across various tasks, but standard first-order (FO) fine-tuning demands considerable memory, significantly limiting real-world deployment. Recently, zeroth-order (ZO) optimization stood out as a promising memory-efficient training paradigm, avoiding backward passes and relying solely on forward passes for gradient estimation, making it attractive for resource-constrained scenarios. However, ZO method lags far behind FO method in both convergence speed and accuracy. To bridge the gap, we introduce a novel layer-wise divergence analysis that uncovers the distinct update pattern of FO and ZO optimization. Aiming to resemble the learning capacity of FO method from the findings, we propose Divergence-driven Zeroth-Order (DiZO) optimization. DiZO conducts divergence-driven layer adaptation by incorporating projections to ZO updates, generating diverse-magnitude updates precisely scaled to layer-wise individual optimization needs. Our results demonstrate that DiZO significantly reduces the needed iterations for convergence without sacrificing throughput, cutting training GPU hours by up to 48\% on various datasets. Moreover, DiZO consistently outperforms the representative ZO baselines in fine-tuning RoBERTa-large, OPT-series, and Llama-series on downstream tasks and, in some cases, even surpasses memory-intensive FO fine-tuning. Our code is released at https://github.com/Skilteee/DiZO.
>
---
#### [replaced 049] The Curious Case of Factual (Mis)Alignment between LLMs' Short- and Long-Form Answers
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.11218v2](http://arxiv.org/pdf/2510.11218v2)**

> **作者:** Saad Obaid ul Islam; Anne Lauscher; Goran Glavaš
>
> **备注:** Code: https://github.com/WorldHellow/SLAQ/tree/main
>
> **摘要:** Large language models (LLMs) can correctly answer "When was Einstein born?" yet fail to provide the same date when writing about Einstein's life revealing a fundamental inconsistency in how models access factual knowledge across task complexities. While models display impressive accuracy on factual question-answering benchmarks, the reliability gap between simple and complex queries remains poorly understood, eroding their trustworthiness. In this work, we introduce Short-Long Form Alignment for Factual Question Answering (SLAQ), a controlled evaluation framework that compares LLMs' answers to the same factual questions asked (a) in isolation (short) vs. (b) integrated into complex queries (long). Looking at 16 LLMs across 600 queries, we find a systematic misalignment of answers to the corresponding short and long queries. We further uncover position-dependent accuracy loss and momentum effects where consecutive correct or incorrect answers create self-reinforcing patterns. Through mechanistic analysis, we find that aligned facts activate overlapping model internals, and that metrics based on mechanistic similarity can predict short-long answer alignment with up to 78% accuracy. Our work establishes factual consistency over query complexity as an important aspect of LLMs' trustworthiness and challenges current evaluation practices, which implicitly assume that good performance for simple factual queries implies reliability in more complex knowledge-seeking tasks too.
>
---
#### [replaced 050] Compressed and Smooth Latent Space for Text Diffusion Modeling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21170v2](http://arxiv.org/pdf/2506.21170v2)**

> **作者:** Viacheslav Meshchaninov; Egor Chimbulatov; Alexander Shabalin; Aleksandr Abramov; Dmitry Vetrov
>
> **摘要:** Autoregressive language models dominate modern text generation, yet their sequential nature introduces fundamental limitations: decoding is slow, and maintaining global coherence remains challenging. Diffusion models offer a promising alternative by enabling parallel generation and flexible control; however, their application to text generation is hindered by the high dimensionality of token-level representations. We introduce Cosmos, a novel approach to text generation that operates entirely in a compressed, smooth latent space tailored specifically for diffusion. This space is learned using an autoencoder trained simultaneously for token-level reconstruction and alignment with frozen activations from a pretrained language encoder, providing robust semantic grounding and enabling effective perturbation-based augmentations. Empirically, we demonstrate that text representations can be compressed by $8\times$ while maintaining generation quality comparable to token-level diffusion models. Furthermore, increasing the latent sequence length allows Cosmos to surpass both diffusion-based and autoregressive baselines. We evaluate Cosmos on four diverse generative tasks including story generation, question generation, summarization, and detoxification and compare it with various generative paradigms. Cosmos achieves comparable or superior generation quality while offering more than $2\times$ faster inference. Code is released at \href{https://github.com/MeshchaninovViacheslav/cosmos}{GitHub}
>
---
#### [replaced 051] LLMTaxo: Leveraging Large Language Models for Constructing Taxonomy of Factual Claims from Social Media
- **分类: cs.CL; cs.AI; cs.SI**

- **链接: [http://arxiv.org/pdf/2504.12325v2](http://arxiv.org/pdf/2504.12325v2)**

> **作者:** Haiqi Zhang; Zhengyuan Zhu; Zeyu Zhang; Chengkai Li
>
> **摘要:** With the rapid expansion of content on social media platforms, analyzing and comprehending online discourse has become increasingly complex. This paper introduces LLMTaxo, a novel framework leveraging large language models for the automated construction of taxonomies of factual claims from social media by generating topics at multiple levels of granularity. The resulting hierarchical structure significantly reduces redundancy and improves information accessibility. We also propose dedicated taxonomy evaluation metrics to enable comprehensive assessment. Evaluations conducted on three diverse datasets demonstrate LLMTaxo's effectiveness in producing clear, coherent, and comprehensive taxonomies. Among the evaluated models, GPT-4o mini consistently outperforms others across most metrics. The framework's flexibility and low reliance on manual intervention underscore its potential for broad applicability.
>
---
#### [replaced 052] Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.00161v2](http://arxiv.org/pdf/2508.00161v2)**

> **作者:** Ziqian Zhong; Aditi Raghunathan
>
> **摘要:** The releases of powerful open-weight large language models (LLMs) are often not accompanied by access to their full training data. Existing interpretability methods, particularly those based on activations, often require or assume distributionally similar data. This is a significant limitation when detecting and defending against novel potential threats like backdoors, which are by definition out-of-distribution. In this work, we introduce a new method for understanding, monitoring and controlling fine-tuned LLMs that interprets weights, rather than activations, thereby side stepping the need for data that is distributionally similar to the unknown training data. We demonstrate that the top singular vectors of the weight difference between a fine-tuned model and its base model correspond to newly acquired behaviors. By monitoring the cosine similarity of activations along these directions, we can detect salient behaviors introduced during fine-tuning with high precision. For backdoored models that bypasses safety mechanisms when a secret trigger is present, our method stops up to 100% of attacks with a false positive rate below 1.2%. For models that have undergone unlearning, we detect inference on erased topics with accuracy up to 95.42% and can even steer the model to recover "unlearned" information. Besides monitoring, our method also shows potential for pre-deployment model auditing: by analyzing commercial instruction-tuned models (OLMo, Llama, Qwen), we are able to uncover model-specific fine-tuning focus including marketing strategies and Midjourney prompt generation. Our implementation can be found at https://github.com/fjzzq2002/WeightWatch.
>
---
#### [replaced 053] Semantic Representation Attack against Aligned Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.19360v2](http://arxiv.org/pdf/2509.19360v2)**

> **作者:** Jiawei Lian; Jianhong Pan; Lefan Wang; Yi Wang; Shaohui Mei; Lap-Pui Chau
>
> **摘要:** Large Language Models (LLMs) increasingly employ alignment techniques to prevent harmful outputs. Despite these safeguards, attackers can circumvent them by crafting prompts that induce LLMs to generate harmful content. Current methods typically target exact affirmative responses, such as ``Sure, here is...'', suffering from limited convergence, unnatural prompts, and high computational costs. We introduce Semantic Representation Attack, a novel paradigm that fundamentally reconceptualizes adversarial objectives against aligned LLMs. Rather than targeting exact textual patterns, our approach exploits the semantic representation space comprising diverse responses with equivalent harmful meanings. This innovation resolves the inherent trade-off between attack efficacy and prompt naturalness that plagues existing methods. The Semantic Representation Heuristic Search algorithm is proposed to efficiently generate semantically coherent and concise adversarial prompts by maintaining interpretability during incremental expansion. We establish rigorous theoretical guarantees for semantic convergence and demonstrate that our method achieves unprecedented attack success rates (89.41\% averaged across 18 LLMs, including 100\% on 11 models) while maintaining stealthiness and efficiency. Comprehensive experimental results confirm the overall superiority of our Semantic Representation Attack. The code will be publicly available.
>
---
#### [replaced 054] Flex-Judge: Text-Only Reasoning Unleashes Zero-Shot Multimodal Evaluators
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18601v4](http://arxiv.org/pdf/2505.18601v4)**

> **作者:** Jongwoo Ko; Sungnyun Kim; Sungwoo Cho; Se-Young Yun
>
> **备注:** NeurIPS 2025
>
> **摘要:** Human-generated reward signals are critical for aligning generative models with human preferences, guiding both training and inference-time evaluations. While large language models (LLMs) employed as proxy evaluators, i.e., LLM-as-a-Judge, significantly reduce the costs associated with manual annotations, they typically require extensive modality-specific training data and fail to generalize well across diverse multimodal tasks. In this paper, we propose Flex-Judge, a reasoning-guided multimodal judge model that leverages minimal textual reasoning data to robustly generalize across multiple modalities and evaluation formats. Our core intuition is that structured textual reasoning explanations inherently encode generalizable decision-making patterns, enabling an effective transfer to multimodal judgments, e.g., with images or videos. Empirical results demonstrate that Flex-Judge, despite being trained on significantly fewer text data, achieves competitive or superior performance compared to state-of-the-art commercial APIs and extensively trained multimodal evaluators. Notably, Flex-Judge presents broad impact in modalities like molecule, where comprehensive evaluation benchmarks are scarce, underscoring its practical value in resource-constrained domains. Our framework highlights reasoning-based text supervision as a powerful, cost-effective alternative to traditional annotation-intensive approaches, substantially advancing scalable multimodal model-as-a-judge.
>
---
#### [replaced 055] MedTrust-RAG: Evidence Verification and Trust Alignment for Biomedical Question Answering
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2510.14400v2](http://arxiv.org/pdf/2510.14400v2)**

> **作者:** Yingpeng Ning; Yuanyuan Sun; Ling Luo; Yanhua Wang; Yuchen Pan; Hongfei Lin
>
> **备注:** Accepted as a short paper at BlBM2025
>
> **摘要:** Biomedical question answering (QA) requires accurate interpretation of complex medical knowledge. Large language models (LLMs) have shown promising capabilities in this domain, with retrieval-augmented generation (RAG) systems enhancing performance by incorporating external medical literature. However, RAG-based approaches in biomedical QA suffer from hallucinations due to post-retrieval noise and insufficient verification of retrieved evidence, undermining response reliability. We propose MedTrust-Guided Iterative RAG, a framework designed to enhance factual consistency and mitigate hallucinations in medical QA. Our method introduces three key innovations. First, it enforces citation-aware reasoning by requiring all generated content to be explicitly grounded in retrieved medical documents, with structured Negative Knowledge Assertions used when evidence is insufficient. Second, it employs an iterative retrieval-verification process, where a verification agent assesses evidence adequacy and refines queries through Medical Gap Analysis until reliable information is obtained. Third, it integrates the MedTrust-Align Module (MTAM) that combines verified positive examples with hallucination-aware negative samples, leveraging Direct Preference Optimization to reinforce citation-grounded reasoning while penalizing hallucination-prone response patterns.
>
---
#### [replaced 056] Unseen from Seen: Rewriting Observation-Instruction Using Foundation Models for Augmenting Vision-Language Navigation
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.18065v2](http://arxiv.org/pdf/2503.18065v2)**

> **作者:** Ziming Wei; Bingqian Lin; Yunshuang Nie; Jiaqi Chen; Shikui Ma; Hang Xu; Xiaodan Liang
>
> **备注:** Accepted by IEEE Transactions on Neural Networks and Learning Systems
>
> **摘要:** Data scarcity is a long-standing challenge in the Vision-Language Navigation (VLN) field, which extremely hinders the generalization of agents to unseen environments. Previous works primarily rely on additional simulator data or web-collected images/videos to improve the generalization. However, the simulator environments still face limited diversity, and the web-collected data often requires extensive labor to remove the noise. In this paper, we propose a Rewriting-driven AugMentation (RAM) paradigm for VLN, which directly creates the unseen observation-instruction pairs via rewriting human-annotated training data. Benefiting from our rewriting mechanism, new observation-instruction pairs can be obtained in both simulator-free and labor-saving manners to promote generalization. Specifically, we first introduce Object-Enriched Observation Rewriting, where we combine Vision-Language Models (VLMs) and Large Language Models (LLMs) to derive rewritten object-enriched scene descriptions, enabling observation synthesis with diverse objects and spatial layouts via Text-to-Image Generation Models (T2IMs). Then, we propose Observation-Contrast Instruction Rewriting, which generates observation-aligned rewritten instructions by requiring LLMs to reason the difference between original and new observations. We further develop a mixing-then-focusing training strategy with a random observation cropping scheme, effectively enhancing data distribution diversity while suppressing augmentation data noise during training. Experiments on both the discrete environments (R2R, REVERIE, and R4R datasets) and continuous environments (R2R-CE dataset) show the superior performance and impressive generalization ability of our method. Code is available at https://github.com/SaDil13/VLN-RAM.
>
---
#### [replaced 057] Consistency of Responses and Continuations Generated by Large Language Models on Social Media
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2501.08102v4](http://arxiv.org/pdf/2501.08102v4)**

> **作者:** Wenlu Fan; Yuqi Zhu; Bin Wang; Wentao Xu
>
> **备注:** This paper has been accepted by the International AAAI Conference on Web and Social Media (ICWSM) 2026 (Los Angeles, California, U.S.)
>
> **摘要:** Large Language Models (LLMs) demonstrate remarkable capabilities in text generation, yet their emotional consistency and semantic coherence in social media contexts remain insufficiently understood. This study investigates how LLMs handle emotional content and maintain semantic relationships through continuation and response tasks using three open-source models: Gemma, Llama3 and Llama3.3 and one commercial Model:Claude. By analyzing climate change discussions from Twitter and Reddit, we examine emotional transitions, intensity patterns, and semantic consistency between human-authored and LLM-generated content. Our findings reveal that while both models maintain high semantic coherence, they exhibit distinct emotional patterns: these models show a strong tendency to moderate negative emotions. When the input text carries negative emotions such as anger, disgust, fear, or sadness, LLM tends to generate content with more neutral emotions, or even convert them into positive emotions such as joy or surprise. At the same time, we compared the LLM-generated content with human-authored content. The four models systematically generated responses with reduced emotional intensity and showed a preference for neutral rational emotions in the response task. In addition, these models all maintained a high semantic similarity with the original text, although their performance in the continuation task and the response task was different. These findings provide deep insights into the emotion and semantic processing capabilities of LLM, which are of great significance for its deployment in social media environments and human-computer interaction design.
>
---
#### [replaced 058] Code Execution as Grounded Supervision for LLM Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.10343v2](http://arxiv.org/pdf/2506.10343v2)**

> **作者:** Dongwon Jung; Wenxuan Zhou; Muhao Chen
>
> **备注:** EMNLP 2025
>
> **摘要:** Training large language models (LLMs) with chain-of-thought (CoT) supervision has proven effective for enhancing their reasoning abilities. However, obtaining reliable and accurate reasoning supervision remains a significant challenge. We propose a scalable method for generating a high-quality CoT supervision dataset by leveraging the determinism of program execution. Unlike existing reasoning dataset generation methods that rely on costly human annotations or error-prone LLM-generated CoT, our approach extracts verifiable, step-by-step reasoning traces from code execution and transforms them into a natural language CoT reasoning. Experiments on reasoning benchmarks across various domains show that our method effectively equips LLMs with transferable reasoning abilities across diverse tasks. Furthermore, the ablation studies validate that our method produces highly accurate reasoning data and reduces overall token length during inference by reducing meaningless repetition and overthinking.
>
---
#### [replaced 059] HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.10062v2](http://arxiv.org/pdf/2510.10062v2)**

> **作者:** Adnan El Assadi; Isaac Chung; Roman Solomatin; Niklas Muennighoff; Kenneth Enevoldsen
>
> **备注:** Submitted to ICLR 2026
>
> **摘要:** Comparing human and model performance offers a valuable perspective for understanding the strengths and limitations of embedding models, highlighting where they succeed and where they fail to capture meaning and nuance. However, such comparisons are rarely made, as human performance on embedding tasks is difficult to measure. To fill this gap, we introduce HUME: Human Evaluation Framework for Text Embeddings. While frameworks like MTEB provide broad model evaluation, they lack reliable estimates of human performance, limiting the interpretability of model scores. We measure human performance across 16 MTEB datasets spanning reranking, classification, clustering, and semantic textual similarity across linguistically diverse high- and low-resource languages. Humans achieve an average performance of 77.6% compared to 80.1% for the best embedding model, although variation is substantial: models reach near-ceiling performance on some datasets while struggling on others, suggesting dataset issues and revealing shortcomings in low-resource languages. We provide human performance baselines, insight into task difficulty patterns, and an extensible evaluation framework that enables a more meaningful interpretation of the model and informs the development of both models and benchmarks. Our code, dataset, and leaderboard are publicly available at https://github.com/embeddings-benchmark/mteb.
>
---
#### [replaced 060] Creativity Benchmark: A benchmark for marketing creativity for large language models
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2509.09702v2](http://arxiv.org/pdf/2509.09702v2)**

> **作者:** Ninad Bhat; Kieran Browne; Pip Bingemann
>
> **备注:** 30 Pages, 14 figures. Fixed typos
>
> **摘要:** We introduce Creativity Benchmark, an evaluation framework for large language models (LLMs) in marketing creativity. The benchmark covers 100 brands (12 categories) and three prompt types (Insights, Ideas, Wild Ideas). Human pairwise preferences from 678 practising creatives over 11,012 anonymised comparisons, analysed with Bradley-Terry models, show tightly clustered performance with no model dominating across brands or prompt types: the top-bottom spread is $\Delta\theta \approx 0.45$, which implies a head-to-head win probability of $0.61$; the highest-rated model beats the lowest only about $61\%$ of the time. We also analyse model diversity using cosine distances to capture intra- and inter-model variation and sensitivity to prompt reframing. Comparing three LLM-as-judge setups with human rankings reveals weak, inconsistent correlations and judge-specific biases, underscoring that automated judges cannot substitute for human evaluation. Conventional creativity tests also transfer only partially to brand-constrained tasks. Overall, the results highlight the need for expert human evaluation and diversity-aware workflows.
>
---
#### [replaced 061] PsyMem: Fine-grained psychological alignment and Explicit Memory Control for Advanced Role-Playing LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.12814v2](http://arxiv.org/pdf/2505.12814v2)**

> **作者:** Xilong Cheng; Yunxiao Qin; Yuting Tan; Zhengnan Li; Ye Wang; Hongjiang Xiao; Yuan Zhang
>
> **备注:** Pre-MIT Press publication version, has been accepted by TACL
>
> **摘要:** Existing LLM-based role-playing methods often rely on superficial textual descriptions or simplistic metrics, inadequately modeling both intrinsic and extrinsic character dimensions. Additionally, they typically simulate character memory with implicit model knowledge or basic retrieval augment generation without explicit memory alignment, compromising memory consistency. The two issues weaken reliability of role-playing LLMs in several applications, such as trustworthy social simulation. To address these limitations, we propose PsyMem, a novel framework integrating fine-grained psychological attributes and explicit memory control for role-playing. PsyMem supplements textual descriptions with 26 psychological indicators to detailed model character. Additionally, PsyMem implements memory alignment training, explicitly trains the model to align character's response with memory, thereby enabling dynamic memory-controlled responding during inference. By training Qwen2.5-7B-Instruct on our specially designed dataset (including 5,414 characters and 38,962 dialogues extracted from novels), the resulting model, termed as PsyMem-Qwen, outperforms baseline models in role-playing, achieving the best performance in human-likeness and character fidelity.
>
---
#### [replaced 062] HCR-Reasoner: Synergizing Large Language Models and Theory for Human-like Causal Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.08750v2](http://arxiv.org/pdf/2505.08750v2)**

> **作者:** Yanxi Zhang; Xin Cong; Zhong Zhang; Xiao Liu; Dongyan Zhao; Yesai Wu
>
> **摘要:** Genuine human-like causal reasoning is fundamental for strong artificial intelligence. Humans typically identify whether an event is part of the causal chain first, and then influenced by modulatory factors such as morality, normality, and intention to make the final judgment. These two stages naturally map to the fields of 1) actual causality that provides formalisms for causal chain membership and 2) causal judgment from cognitive science that studies psychological modulators that influence causal selection. However, these two domains have largely been studied in isolation, leaving a gap for a systematic method based on LLMs. Therefore, we introduce HCR-Reasoner, a framework that systematically integrates the theory of actual causality and causal judgment into LLMs for human-like causal reasoning. It simulates humans by using actual causality formalisms to filter for structurally necessary candidate causes and causal judgment factors to determine the psychologically selected cause. For fine-grained evaluation, we introduce HCR-Bench, a challenging benchmark with 1,093 annotated instances with detailed reasoning steps. Results show HCR-Reasoner consistently and significantly improves LLMs' causal alignment with humans, and that explicitly integrating theory-guided reasoning into LLMs is highly effective for achieving faithful human-like causal reasoning.
>
---
#### [replaced 063] From Scarcity to Efficiency: Investigating the Effects of Data Augmentation on African Machine Translation
- **分类: cs.CL; 68T50; I.7**

- **链接: [http://arxiv.org/pdf/2509.07471v2](http://arxiv.org/pdf/2509.07471v2)**

> **作者:** Mardiyyah Oduwole; Oluwatosin Olajide; Jamiu Suleiman; Faith Hunja; Busayo Awobade; Fatimo Adebanjo; Comfort Akanni; Chinonyelum Igwe; Peace Ododo; Promise Omoigui; Abraham Owodunni; Steven Kolawole
>
> **备注:** 8 pages, 3 tables. Exploratory work on Data Augmentation for African Machine Translation
>
> **摘要:** The linguistic diversity across the African continent presents different challenges and opportunities for machine translation. This study explores the effects of data augmentation techniques in improving translation systems in low-resource African languages. We focus on two data augmentation techniques: sentence concatenation with back translation and switch-out, applying them across six African languages. Our experiments show significant improvements in machine translation performance, with a minimum increase of 25\% in BLEU score across all six languages. We provide a comprehensive analysis and highlight the potential of these techniques to improve machine translation systems for low-resource languages, contributing to the development of more robust translation systems for under-resourced languages.
>
---
#### [replaced 064] Whose Journey Matters? Investigating Identity Biases in Large Language Models (LLMs) for Travel Planning Assistance
- **分类: cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2410.17333v3](http://arxiv.org/pdf/2410.17333v3)**

> **作者:** Ruiping Ren; Yingwei; Xu; Xing Yao; Shu Cole; Haining Wang
>
> **摘要:** As large language models (LLMs) become increasingly integral to the hospitality and tourism industry, concerns about their fairness in serving diverse identity groups persist. Grounded in social identity theory and sociotechnical systems theory, this study examines ethnic and gender biases in travel recommendations generated by LLMs. Using fairness probing, we analyze outputs from three leading open-source LLMs. The results show that test accuracy for both ethnicity and gender classifiers exceed random chance. Analysis of the most influential features reveals the presence of stereotype bias in LLM-generated recommendations. We also found hallucinations among these features, occurring more frequently in recommendations for minority groups. These findings indicate that LLMs exhibit ethnic and gender bias when functioning as travel planning assistants. This study underscores the need for bias mitigation strategies to improve the inclusivity and reliability of generative AI-driven travel planning assistance.
>
---
#### [replaced 065] A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24550v2](http://arxiv.org/pdf/2505.24550v2)**

> **作者:** Xiaoang Xu; Shuo Wang; Xu Han; Zhenghao Liu; Huijia Wu; Peipei Li; Zhiyuan Liu; Maosong Sun; Zhaofeng He
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Large Reasoning Models (LRMs) achieve superior performance by extending the thought length. However, a lengthy thinking trajectory leads to reduced efficiency. Most of the existing methods are stuck in the assumption of overthinking and attempt to reason efficiently by compressing the Chain-of-Thought, but this often leads to performance degradation. To address this problem, we introduce A*-Thought, an efficient tree search-based unified framework designed to identify and isolate the most essential thoughts from the extensive reasoning chains produced by these models. It formulates the reasoning process of LRMs as a search tree, where each node represents a reasoning span in the giant reasoning space. By combining the A* search algorithm with a cost function specific to the reasoning path, it can efficiently compress the chain of thought and determine a reasoning path with high information density and low cost. In addition, we also propose a bidirectional importance estimation mechanism, which further refines this search process and enhances its efficiency beyond uniform sampling. Extensive experiments on several advanced math tasks show that A*-Thought effectively balances performance and efficiency over a huge search space. Specifically, A*-Thought can improve the performance of QwQ-32B by 2.39$\times$ with low-budget and reduce the length of the output token by nearly 50% with high-budget. The proposed method is also compatible with several other LRMs, demonstrating its generalization capability. The code can be accessed at: https://github.com/AI9Stars/AStar-Thought.
>
---
#### [replaced 066] ToMAP: Training Opponent-Aware LLM Persuaders with Theory of Mind
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.22961v2](http://arxiv.org/pdf/2505.22961v2)**

> **作者:** Peixuan Han; Zijia Liu; Jiaxuan You
>
> **摘要:** Large language models (LLMs) have shown promising potential in persuasion, but existing works on training LLM persuaders are still preliminary. Notably, while humans are skilled in modeling their opponent's thoughts and opinions proactively and dynamically, current LLMs struggle with such Theory of Mind (ToM) reasoning, resulting in limited diversity and opponent awareness. To address this limitation, we introduce Theory of Mind Augmented Persuader (ToMAP), a novel approach for building more flexible persuader agents by incorporating two theory of mind modules that enhance the persuader's awareness and analysis of the opponent's mental state. Specifically, we begin by prompting the persuader to consider possible objections to the target central claim, and then use a text encoder paired with a trained MLP classifier to predict the opponent's current stance on these counterclaims. Our carefully designed reinforcement learning schema enables the persuader learns how to analyze opponent-related information and utilize it to generate more effective arguments. Experiments show that the ToMAP persuader, while containing only 3B parameters, outperforms much larger baselines, like GPT-4o, with a relative gain of 39.4% across multiple persuadee models and diverse corpora. Notably, ToMAP exhibits complex reasoning chains and reduced repetition during training, which leads to more diverse and effective arguments. The opponent-aware feature of ToMAP also makes it suitable for long conversations and enables it to employ more logical and opponent-aware strategies. These results underscore our method's effectiveness and highlight its potential for developing more persuasive language agents. Code is available at: https://github.com/ulab-uiuc/ToMAP.
>
---
#### [replaced 067] AnTKV: Anchor Token-Aware Sub-Bit Vector Quantization for KV Cache in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.19505v2](http://arxiv.org/pdf/2506.19505v2)**

> **作者:** Zeyu Li; Chuanfu Xiao; Yang Wang; Xiang Liu; Zhenheng Tang; Baotong Lu; Mao Yang; Xinyu Chen; Xiaowen Chu
>
> **摘要:** Quantization has emerged as an effective and lightweight solution to reduce the memory footprint of the KV cache in Large Language Models. Nevertheless, minimizing the accuracy degradation caused by ultra-low-bit KV cache quantization remains a significant challenge. While scalar quantization is constrained by 1-bit bound, vector quantization exploits intra-vector correlations and enables sub-bit regimes, making it more suitable for ultra-low-bit quantization. To further mitigate quantization-induced degradation, we reveal that the degradation is highly uneven across tokens in attention quality. To investigate this unevenness, we introduce anchor score to measure each token's sensitivity to quantization. Our analysis and experiments show that preserving a small subset (1\%) of tokens with the highest Anchor Score significantly mitigates accuracy loss under aggressive quantization. We propose AnTKV, a dual-stage framework that leverages anchor token-aware vector quantization to compress the KV cache. It combines offline token-aware centroids learning and online anchor token selection to balance compression and accuracy. To enable efficient deployment, we design an online anchor token selection kernel compatible with FlashAttention. It allows LLaMA3-8B to scale to 840K tokens on a single 80GB A100, while delivering up to $3.5\times$ higher decoding throughput over the FP16 baseline. Experiments demonstrate that AnTKV matches or surpasses prior methods at 4-bit, and significantly reduce perplexity under ultra-low-bit quantization, achieving 6.32 at 1-bit on Mistral-7B, compared to 7.25 for CQ and 15.36 for KVQuant.
>
---
#### [replaced 068] RHYTHM: Reasoning with Hierarchical Temporal Tokenization for Human Mobility
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.23115v2](http://arxiv.org/pdf/2509.23115v2)**

> **作者:** Haoyu He; Haozheng Luo; Yan Chen; Qi R. Wang
>
> **备注:** Advances in Neural Information Processing Systems 39 (NeurIPS) 2025
>
> **摘要:** Predicting human mobility is inherently challenging due to complex long-range dependencies and multi-scale periodic behaviors. To address this, we introduce RHYTHM (Reasoning with Hierarchical Temporal Tokenization for Human Mobility), a unified framework that leverages large language models (LLMs) as general-purpose spatio-temporal predictors and trajectory reasoners. Methodologically, RHYTHM employs temporal tokenization to partition each trajectory into daily segments and encode them as discrete tokens with hierarchical attention that captures both daily and weekly dependencies, thereby quadratically reducing the sequence length while preserving cyclical information. Additionally, we enrich token representations by adding pre-computed prompt embeddings for trajectory segments and prediction targets via a frozen LLM, and feeding these combined embeddings back into the LLM backbone to capture complex interdependencies. Computationally, RHYTHM keeps the pretrained LLM backbone frozen, yielding faster training and lower memory usage. We evaluate our model against state-of-the-art methods using three real-world datasets. Notably, RHYTHM achieves a 2.4% improvement in overall accuracy, a 5.0% increase on weekends, and a 24.6% reduction in training time. Code is publicly available at https://github.com/he-h/rhythm.
>
---
#### [replaced 069] Robust Search with Uncertainty-Aware Value Models for Language Model Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11155v2](http://arxiv.org/pdf/2502.11155v2)**

> **作者:** Fei Yu; Yingru Li; Benyou Wang
>
> **摘要:** Value model guided search is effective in steering LLM generation but suffers from a lack of robustness. This is due to verifier failure: imperfect VMs mistakenly prune valid reasoning paths, especially when encountering unseen reasoning paths generated during search. To address this, we propose an uncertainty-aware framework with two key components: (1) Uncertainty-Aware Value Models (UVMs), which replace single-point value estimates with value distributions to quantify prediction reliability, and (2) Group Thompson Sampling, an efficient algorithm that selects candidates based on their probability of being optimal. Experiments on two In-Distribution (ID) settings (GSM8K, MATH) and three Out-Of-Distribution (OOD) settings (e.g., AIME25, Minerva Math) show our method significantly mitigates verifier failure and boosts solution coverage, especially on OOD problems. This work provides the first systematic integration of uncertainty quantification into LLM search paradigms, enhancing robustness. The code is released at https://github.com/FreedomIntelligence/UVM.
>
---
#### [replaced 070] Familiarity-Aware Evidence Compression for Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.12468v3](http://arxiv.org/pdf/2409.12468v3)**

> **作者:** Dongwon Jung; Qin Liu; Tenghao Huang; Ben Zhou; Muhao Chen
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Retrieval-augmented generation (RAG) improves large language models (LMs) by incorporating non-parametric knowledge through evidence retrieved from external sources. However, it often struggles to cope with inconsistent and irrelevant information that can distract the LM from its tasks, especially when multiple evidence pieces are required. While compressing the retrieved evidence with a compression model aims to address this issue, the compressed evidence may still be unfamiliar to the target model used for downstream tasks, potentially failing to utilize the evidence effectively. We propose FaviComp (Familarity-Aware Evidence Compression), a novel training-free evidence compression technique that makes retrieved evidence more familiar to the target model, while seamlessly integrating parametric knowledge from the model. Experimental results show that FaviComp consistently outperforms most recent evidence compression baselines across multiple open-domain QA datasets, improving accuracy by up to 28.1% while achieving high compression rates. Additionally, we demonstrate the effective integration of both parametric and non-parametric knowledge during evidence compression.
>
---
#### [replaced 071] Tracing Partisan Bias to Its Emotional Fingerprints: A Computational Approach to Mitigation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.01284v2](http://arxiv.org/pdf/2501.01284v2)**

> **作者:** Junjie Liu; Xi Luo; Sirong Wu; Gengchen Sun; Yuhui Deng
>
> **摘要:** This study introduces a novel framework for analysing and mitigating media bias by tracing partisan stances to their linguistic roots in emotional language. We posit that partisan bias is not merely an abstract stance but materialises as quantifiable 'emotional fingerprints' within news texts. These fingerprints are systematically measured using the Valence-Arousal-Dominance (VAD) framework, allowing us to decode the affective strategies behind partisan framing. Our analysis of the Allsides dataset confirms this hypothesis, revealing distinct and statistically significant emotional fingerprints for left, centre, and right-leaning media. Based on this evidence-driven approach, we then propose a computational approach to mitigation through NeutraSum, a model designed to neutralise these identified emotional patterns. By explicitly targeting the VAD characteristics of biased language, NeutraSum generates summaries that are not only coherent but also demonstrably closer to an emotionally neutral baseline. Experimental results validate our framework: NeutraSum successfully erases the partisan emotional fingerprints from its summaries, achieving a demonstrably lower emotional bias score than other models. This work pioneers a new path for bias mitigation, shifting the focus from treating symptoms (political labels) to addressing the cause: the emotional encoding of partisan bias in language.
>
---
#### [replaced 072] LEME: Open Large Language Models for Ophthalmology with Advanced Reasoning and Clinical Validation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.03740v2](http://arxiv.org/pdf/2410.03740v2)**

> **作者:** Hyunjae Kim; Xuguang Ai; Sahana Srinivasan; Aidan Gilson; Maxwell B. Singer; Krithi Pushpanathan; Qianqian Xie; Jungwoo Park; Serina Applebaum; Gabriel Dawei Yang; Minjie Zou; David Ziyou Chen; Ke Zou; Soshian Sarrafpour; Ji Liu; Yu Yin; Jimin Huang; Quang Ngoc Nguyen; Erping Long; Peixing Wan; Dianbo Liu; Richard Hintz; W. Jim Zheng; Sophia Y. Wang; Lucila Ohno-Machado; Hua Xu; Ron A. Adelman; Luciano V. Del Priore; Yih-Chung Tham; Qingyu Chen
>
> **摘要:** Large Language Models (LLMs) are poised to revolutionize healthcare. Ophthalmology-specific LLMs remain scarce and underexplored. We introduced an open-source, specialized LLM for ophthalmology, termed Language Enhanced Model for Eye (LEME). LEME was initially pre-trained on the Llama2 70B framework and further fine-tuned with a corpus of ~127,000 non-copyrighted training instances curated from ophthalmology-specific case reports, abstracts, and open-source study materials. We benchmarked LEME against eight other LLMs, namely, GPT-3.5, GPT-4, three Llama2 models (7B, 13B, 70B), PMC-LLAMA 13B, Meditron 70B, and EYE-Llama (another ophthalmology-specific LLM). Evaluations included four internal validation tasks: abstract completion, fill-in-the-blank, multiple-choice questions (MCQ), and short-answer QA. External validation tasks encompassed long-form QA, MCQ, patient EHR summarization, and clinical QA. Evaluation metrics included Rouge-L scores, accuracy, and expert evaluation of correctness, completeness, and readability. In internal validations, LEME consistently outperformed its counterparts, achieving Rouge-L scores of 0.20 in abstract completion (all p<0.05), 0.82 in fill-in-the-blank (all p<0.0001), and 0.22 in short-answer QA (all p<0.0001, except versus GPT-4). In external validations, LEME excelled in long-form QA with a Rouge-L of 0.19 (all p<0.0001), ranked second in MCQ accuracy (0.68; all p<0.0001), and scored highest in EHR summarization and clinical QA (ranging from 4.24 to 4.83 out of 5 for correctness, completeness, and readability). LEME's emphasis on robust fine-tuning and the use of non-copyrighted data represents a breakthrough in open-source ophthalmology-specific LLMs, offering the potential to revolutionize execution of clinical tasks while democratizing research collaboration.
>
---
#### [replaced 073] Hard Negatives, Hard Lessons: Revisiting Training Data Quality for Robust Information Retrieval with LLMs
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16967v2](http://arxiv.org/pdf/2505.16967v2)**

> **作者:** Nandan Thakur; Crystina Zhang; Xueguang Ma; Jimmy Lin
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Training robust retrieval and reranker models typically relies on large-scale retrieval datasets; for example, the BGE collection contains 1.6 million query-passage pairs sourced from various data sources. However, we find that certain datasets can negatively impact model effectiveness -- pruning 8 out of 15 datasets from the BGE collection, reduces the training set size by 2.35$\times$, surprisingly increases nDCG@10 on BEIR by 1.0 point. This motivates a deeper examination of training data quality, with a particular focus on "false negatives", where relevant passages are incorrectly labeled as irrelevant. We utilize LLMs as a simple, cost-effective approach to identify and relabel false negatives in training datasets. Experimental results show that relabeling false negatives as true positives improves both E5 (base) and Qwen2.5-7B retrieval models by 0.7$\unicode{x2013}$1.4 points on BEIR and by 1.7$\unicode{x2013}$1.8 points at nDCG@10 on zero-shot AIR-Bench evaluation. Similar gains are observed for rerankers fine-tuned on the relabeled data, such as Qwen2.5-3B on BEIR. The reliability of LLMs to identify false negatives is supported by human annotation results. Our training dataset and code are publicly available.
>
---
#### [replaced 074] CodeVisionary: An Agent-based Framework for Evaluating Large Language Models in Code Generation
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.13472v2](http://arxiv.org/pdf/2504.13472v2)**

> **作者:** Xinchen Wang; Pengfei Gao; Chao Peng; Ruida Hu; Cuiyun Gao
>
> **摘要:** Large language models (LLMs) have demonstrated strong capabilities in code generation, underscoring the critical need for rigorous and comprehensive evaluation. Existing evaluation approaches fall into three categories, including human-centered, metric-based, and LLM-based. Considering that human-centered approaches are labour-intensive and metric-based ones overly rely on reference answers, LLM-based approaches are gaining increasing attention due to their stronger contextual understanding capabilities. However, they generally evaluate the generated code based on static prompts, and tend to fail for complex code scenarios which typically involve multiple requirements and require more contextual information. In addition, these approaches lack fine-grained evaluation for complex code, resulting in limited explainability. To mitigate the limitations, we propose CodeVisionary, the first agent-based evaluation framework for complex code generation. CodeVisionary consists of two stages: (1) Requirement-guided multi-dimensional context distillation stage and (2) Fine-grained scoring and summarization stage. A comprehensive evaluation report is also generated for enhanced explainability. For validation, we construct a new benchmark consisting of 363 samples spanning 37 coding scenarios and 23 programming languages. Extensive experiments demonstrate that CodeVisionary achieves the best performance among three baselines for evaluating complex code generation, outperforming the best baseline with average improvements of 0.217, 0.163, and 0.141 in Pearson, Spearman, and Kendall-Tau coefficients, respectively. The resources of CodeVisionary are available at https://github.com/Eshe0922/CodeVisionary.
>
---
#### [replaced 075] Understanding LLMs' Cross-Lingual Context Retrieval: How Good It Is And Where It Comes From
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.10906v2](http://arxiv.org/pdf/2504.10906v2)**

> **作者:** Changjiang Gao; Hankun Lin; Xin Huang; Xue Han; Junlan Feng; Chao Deng; Jiajun Chen; Shujian Huang
>
> **摘要:** Cross-lingual context retrieval (extracting contextual information in one language based on requests in another) is a fundamental aspect of cross-lingual alignment, but the performance and mechanism of it for large language models (LLMs) remains unclear. In this paper, we evaluate the cross-lingual context retrieval of over 40 LLMs across 12 languages, using cross-lingual machine reading comprehension (xMRC) as a representative scenario. Our results show that post-trained open LLMs show strong cross-lingual context retrieval ability, comparable to closed-source LLMs such as GPT-4o, and their estimated oracle performances greatly improve after post-training. Our mechanism analysis shows that the cross-lingual context retrieval process can be divided into two main phases: question encoding and answer retrieval, which are formed in pre-training and post-training respectively. The phasing stability correlates with xMRC performance, and the xMRC bottleneck lies at the last model layers in the second phase, where the effect of post-training can be evidently observed. Our results also indicate that larger-scale pretraining cannot improve the xMRC performance. Instead, larger LLMs need further multilingual post-training to fully unlock their cross-lingual context retrieval potential.
>
---
#### [replaced 076] Trainable Dynamic Mask Sparse Attention
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.02124v5](http://arxiv.org/pdf/2508.02124v5)**

> **作者:** Jingze Shi; Yifan Wu; Yiran Peng; Bingheng Wu; Liangdong Wang; Guang Liu; Yuyu Luo
>
> **备注:** 26 pages
>
> **摘要:** The increasing demand for long-context modeling in large language models (LLMs) is bottlenecked by the quadratic complexity of the standard self-attention mechanism. The community has proposed sparse attention to mitigate this issue. However, position-aware sparse attention methods rely on static sparse structures that lack adaptability to diverse query contexts, while content-aware sparse attention methods depend on heuristic key-value selection, hindering full differentiability. We introduce a trainable dynamic mask sparse attention mechanism, a method that merges the advantages of both position-aware and content-aware approaches. Dynamic Mask Attention (DMA) achieves this through three key innovations: First, it leverages value vector representations to generate content-aware dynamic masks, enabling the model to adaptively identify and attend to critical information. Second, it computes position-aware sparse weights in a hardware-friendly manner, efficiently skipping unnecessary computational regions. Finally, we demonstrate that the introduced dynamic mask and sparse weights do not obstruct gradients, supporting end-to-end training. We have validated the performance of DMA through comprehensive experiments. A large body of experimental evidence shows that DMA consistently holds a Pareto advantage over state-of-the-art sparse attention baselines in tasks including scaling laws, multi-query associative recall, standard benchmarks, and needle in a haystack tests, while also delivering up to a 10x overall speedup. These results highlight its ability to effectively balance model efficiency with long-context modeling capabilities. Our computational kernel code is now open-source at https://github.com/SmallDoges/flash-dmattn to encourage further research and application by the community.
>
---
#### [replaced 077] Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.19700v3](http://arxiv.org/pdf/2505.19700v3)**

> **作者:** Yi Liu; Dianqing Liu; Mingye Zhu; Junbo Guo; Yongdong Zhang; Zhendong Mao
>
> **备注:** Accepted by NeurIPS 2025, 28 pages
>
> **摘要:** The widespread adoption of large language models (LLMs) across industries has increased the demand for high-quality and customizable outputs. However, traditional alignment methods often require retraining large pretrained models, making it difficult to quickly adapt and optimize LLMs for diverse applications. To address this limitation, we propose a novel \textit{Residual Alignment Model} (\textit{RAM}) that formalizes the alignment process as a type of importance sampling. In this framework, the unaligned upstream model serves as the proposal distribution, while the alignment process is framed as secondary sampling based on an autoregressive alignment module that acts as an estimator of the importance weights. This design enables a natural detachment of the alignment module from the target aligned model, improving flexibility and scalability. Based on this model, we derive an efficient sequence-level training strategy for the alignment module, which operates independently of the proposal module. Additionally, we develop a resampling algorithm with iterative token-level decoding to address the common first-token latency issue in comparable methods. Experimental evaluations on two leading open-source LLMs across diverse tasks, including instruction following, domain adaptation, and preference optimization, demonstrate that our approach consistently outperforms baseline models.
>
---
#### [replaced 078] ReaGAN: Node-as-Agent-Reasoning Graph Agentic Network
- **分类: cs.CL; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2508.00429v4](http://arxiv.org/pdf/2508.00429v4)**

> **作者:** Minghao Guo; Xi Zhu; Haochen Xue; Chong Zhang; Shuhang Lin; Jingyuan Huang; Ziyi Ye; Yongfeng Zhang
>
> **备注:** 11 pages, work in progress
>
> **摘要:** Graph Neural Networks (GNNs) have achieved remarkable success in graph-based learning by propagating information among neighbor nodes via predefined aggregation mechanisms. However, such fixed schemes often suffer from two key limitations. First, they cannot handle the imbalance in node informativeness -- some nodes are rich in information, while others remain sparse. Second, predefined message passing primarily leverages local structural similarity while ignoring global semantic relationships across the graph, limiting the model's ability to capture distant but relevant information. We propose Retrieval-augmented Graph Agentic Network (ReaGAN), an agent-based framework that empowers each node with autonomous, node-level decision-making. Each node acts as an agent that independently plans its next action based on its internal memory, enabling node-level planning and adaptive message propagation. Additionally, retrieval-augmented generation (RAG) allows nodes to access semantically relevant content and build global relationships in the graph. ReaGAN achieves competitive performance under few-shot in-context settings using a frozen LLM backbone without fine-tuning, showcasing the potential of agentic planning and local-global retrieval in graph learning.
>
---
#### [replaced 079] Synthetic Dataset for Evaluating Complex Compositional Knowledge for Natural Language Inference
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2307.05034v4](http://arxiv.org/pdf/2307.05034v4)**

> **作者:** Sushma Anand Akoju; Robert Vacareanu; Haris Riaz; Eduardo Blanco; Mihai Surdeanu
>
> **备注:** Accepted to Natural Language Reasoning and Structured Explanations (NLRSE) Workshop, ACL 2023. For dataset, please refer https://github.com/sushmaakoju/clulab-releases/blob/master/acl2023-nlrse-sicck/README.md and https://github.com/sushmaanandakoju/acl2023-nlrse-clulab-SICCK-dataset
>
> **摘要:** We introduce a synthetic dataset called Sentences Involving Complex Compositional Knowledge (SICCK) and a novel analysis that investigates the performance of Natural Language Inference (NLI) models to understand compositionality in logic. We produce 1,304 sentence pairs by modifying 15 examples from the SICK dataset (Marelli et al., 2014). To this end, we modify the original texts using a set of phrases - modifiers that correspond to universal quantifiers, existential quantifiers, negation, and other concept modifiers in Natural Logic (NL) (MacCartney, 2009). We use these phrases to modify the subject, verb, and object parts of the premise and hypothesis. Lastly, we annotate these modified texts with the corresponding entailment labels following NL rules. We conduct a preliminary verification of how well the change in the structural and semantic composition is captured by neural NLI models, in both zero-shot and fine-tuned scenarios. We found that the performance of NLI models under the zero-shot setting is poor, especially for modified sentences with negation and existential quantifiers. After fine-tuning this dataset, we observe that models continue to perform poorly over negation, existential and universal modifiers.
>
---
#### [replaced 080] KG-Infused RAG: Augmenting Corpus-Based RAG with External Knowledge Graphs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09542v2](http://arxiv.org/pdf/2506.09542v2)**

> **作者:** Dingjun Wu; Yukun Yan; Zhenghao Liu; Zhiyuan Liu; Maosong Sun
>
> **摘要:** Retrieval-Augmented Generation (RAG) improves factual accuracy by grounding responses in external knowledge. However, existing RAG methods either rely solely on text corpora and neglect structural knowledge, or build ad-hoc knowledge graphs (KGs) at high cost and low reliability. To address these issues, we propose KG-Infused RAG, a framework that incorporates pre-existing large-scale KGs into RAG and applies spreading activation to enhance both retrieval and generation. KG-Infused RAG directly performs spreading activation over external KGs to retrieve relevant structured knowledge, which is then used to expand queries and integrated with corpus passages, enabling interpretable and semantically grounded multi-source retrieval. We further improve KG-Infused RAG through preference learning on sampled key stages of the pipeline. Experiments on five QA benchmarks show that KG-Infused RAG consistently outperforms vanilla RAG (by 3.9% to 17.8%). Compared with KG-based approaches such as GraphRAG and LightRAG, our method obtains structured knowledge at lower cost while achieving superior performance. Additionally, integrating KG-Infused RAG with Self-RAG and DeepNote yields further gains, demonstrating its effectiveness and versatility as a plug-and-play enhancement module for corpus-based RAG methods.
>
---
#### [replaced 081] Towards Evaluating Proactive Risk Awareness of Multimodal Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17455v2](http://arxiv.org/pdf/2505.17455v2)**

> **作者:** Youliang Yuan; Wenxiang Jiao; Yuejin Xie; Chihao Shen; Menghan Tian; Wenxuan Wang; Jen-tse Huang; Pinjia He
>
> **备注:** Accepted by NeurIPS 2025 (Track on Datasets and Benchmarks)
>
> **摘要:** Human safety awareness gaps often prevent the timely recognition of everyday risks. In solving this problem, a proactive safety artificial intelligence (AI) system would work better than a reactive one. Instead of just reacting to users' questions, it would actively watch people's behavior and their environment to detect potential dangers in advance. Our Proactive Safety Bench (PaSBench) evaluates this capability through 416 multimodal scenarios (128 image sequences, 288 text logs) spanning 5 safety-critical domains. Evaluation of 36 advanced models reveals fundamental limitations: Top performers like Gemini-2.5-pro achieve 71% image and 64% text accuracy, but miss 45-55% risks in repeated trials. Through failure analysis, we identify unstable proactive reasoning rather than knowledge deficits as the primary limitation. This work establishes (1) a proactive safety benchmark, (2) systematic evidence of model limitations, and (3) critical directions for developing reliable protective AI. We believe our dataset and findings can promote the development of safer AI assistants that actively prevent harm rather than merely respond to requests. Our dataset can be found at https://huggingface.co/datasets/Youliang/PaSBench.
>
---
#### [replaced 082] Max It or Miss It: Benchmarking LLM On Solving Extremal Problems
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.12997v2](http://arxiv.org/pdf/2510.12997v2)**

> **作者:** Binxin Gao; Jingjun Han
>
> **备注:** Our benchmark dataset is available at https://huggingface.co/datasets/binxingao/extrem-bench
>
> **摘要:** Test-time scaling has enabled Large Language Models (LLMs) with remarkable reasoning capabilities, particularly in mathematical domains, through intermediate chain-of-thought (CoT) reasoning before generating final answers. However, the specific sources and mechanisms underlying these reasoning capabilities remain insufficiently understood. Optimization reasoning, i.e. finding extrema under constraints, represents a fundamental abstraction that underpins critical applications in planning, control, resource allocation, and prompt search. To systematically evaluate this capability, we introduce ExtremBench, a benchmark dataset for solving mathematical extremal problems, curated from inequality exercises used for Chinese Mathematical Olympiad and transformed into $93$ standardized extrema-finding problems. We conduct extensive evaluations across various state-of-the-art open-source model families, including the Qwen3, GPT-OSS, and DeepSeek. Our results reveal that LLMs' extremal-solving reasoning capabilities do not always align with those of current mathematical benchmarks such as AIME25 and MATH-500, with some models showing strong general mathematical reasoning but poor extremal-solving skills, and vice versa. This discrepancy highlights a critical gap in current evaluation practices and suggests that existing benchmarks may not comprehensively capture the full spectrum of mathematical reasoning abilities.
>
---
#### [replaced 083] UFT: Unifying Supervised and Reinforcement Fine-Tuning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16984v2](http://arxiv.org/pdf/2505.16984v2)**

> **作者:** Mingyang Liu; Gabriele Farina; Asuman Ozdaglar
>
> **摘要:** Post-training has demonstrated its importance in enhancing the reasoning capabilities of large language models (LLMs). The primary post-training methods can be categorized into supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT). SFT is efficient and well-suited for small language models, but it may lead to overfitting and limit the reasoning abilities of larger models. In contrast, RFT generally yields better generalization but depends heavily on the strength of the base model. To address the limitations of SFT and RFT, we propose Unified Fine-Tuning (UFT), a novel post-training paradigm that unifies SFT and RFT into a single, integrated process. UFT enables the model to effectively explore solutions while incorporating informative supervision signals, bridging the gap between memorizing and thinking underlying existing methods. Notably, UFT outperforms both SFT and RFT in general, regardless of model sizes. Furthermore, we theoretically prove that UFT breaks RFT's inherent exponential sample complexity bottleneck, showing for the first time that unified training can exponentially accelerate convergence on long-horizon reasoning tasks.
>
---
#### [replaced 084] Deflanderization for Game Dialogue: Balancing Character Authenticity with Task Execution in LLM-based NPCs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.13586v2](http://arxiv.org/pdf/2510.13586v2)**

> **作者:** Pasin Buakhaw; Kun Kerdthaisong; Phuree Phenhiran; Pitikorn Khlaisamniang; Supasate Vorathammathorn; Piyalitt Ittichaiwong; Nutchanon Yongsatianchot
>
> **摘要:** The emergence of large language models (LLMs) has opened new opportunities for cre- ating dynamic non-player characters (NPCs) in gaming environments, enabling both func- tional task execution and persona-consistent dialogue generation. In this paper, we (Tu_Character_lab) report our participation in the Commonsense Persona-Grounded Dialogue Challenge (CPDC) 2025 Round 2, which eval- uates agents across three tracks: task-oriented dialogue, context-aware dialogue, and their integration. Our approach combines two complementary strategies: (i) lightweight prompting techniques in the API track, including a Deflanderization prompting method to suppress excessive role-play and improve task fidelity, and (ii) fine-tuned large models in the GPU track, leveraging Qwen3-14B with supervisedfinetuning (SFT) and Low-Rank Adaptation(LoRA). Our best submissions ranked 2nd on Task 1, 2nd on Task 3 (API track), and 4th on Task 3 (GPU track).
>
---
#### [replaced 085] H3Fusion: Helpful, Harmless, Honest Fusion of Aligned LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.17792v3](http://arxiv.org/pdf/2411.17792v3)**

> **作者:** Selim Furkan Tekin; Fatih Ilhan; Tiansheng Huang; Sihao Hu; Yichang Xu; Zachary Yahn; Ling Liu
>
> **摘要:** The alignment of pre-trained LLMs continues to draw significant attention from both industry and academia, aiming to ensure responses that are helpful, harmless, and honest. However, identifying a point in the model's representation subspace that simultaneously satisfies all these properties remains challenging. H3Fusion addresses this challenge by introducing a mixture-of-experts (MoE)-based fusion mechanism that models alignment as a controllable drift within the subspace, guided by a drift-regularization loss to balance competing alignment dimensions. Furthermore, we formulate the alignment by finding a dual objective of harnessing the distance of generated embeddings and alignment embeddings, and introduce a gating loss by canalizing the activations on the contributing experts. Extensive evaluations of three benchmark datasets show that H3Fusion is more helpful, less harmful, and more honest in three aspects: it outperforms each individually aligned model by 11.37%, and provides stronger robustness compared to the state-of-the-art LLM ensemble approaches by 13.77% and model-merging approaches by 6.18%. Code is available at https://github.com/sftekin/h3fusion.
>
---
#### [replaced 086] Parameter Efficient Fine-tuning via Explained Variance Adaptation
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2410.07170v5](http://arxiv.org/pdf/2410.07170v5)**

> **作者:** Fabian Paischer; Lukas Hauzenberger; Thomas Schmied; Benedikt Alkin; Marc Peter Deisenroth; Sepp Hochreiter
>
> **备注:** Accepted at NeurIPS 2025, Shared first authorship, Code available at https://github.com/ml-jku/EVA
>
> **摘要:** Foundation models (FMs) are pre-trained on large-scale datasets and then fine-tuned for a specific downstream task. The most common fine-tuning method is to update pretrained weights via low-rank adaptation (LoRA). Existing initialization strategies for LoRA often rely on singular value decompositions (SVD) of gradients or weight matrices. However, they do not provably maximize the expected gradient signal, which is critical for fast adaptation. To this end, we introduce Explained Variance Adaptation (EVA), an initialization scheme that uses the directions capturing the most activation variance, provably maximizing the expected gradient signal and accelerating fine-tuning. EVA performs incremental SVD on minibatches of activation vectors and selects the right-singular vectors for initialization once they converged. Further, by selecting the directions that capture the most activation-variance for a given rank budget, EVA accommodates adaptive ranks that reduce the number of trainable parameters. We apply EVA to a variety of fine-tuning tasks as language generation and understanding, image classification, and reinforcement learning. EVA exhibits faster convergence than competitors and achieves the highest average score across a multitude of tasks per domain while reducing the number of trainable parameters through rank redistribution. In summary, EVA establishes a new Pareto frontier compared to existing LoRA initialization schemes in both accuracy and efficiency.
>
---
#### [replaced 087] ASCD: Attention-Steerable Contrastive Decoding for Reducing Hallucination in MLLM
- **分类: cs.CV; cs.CL; 68T45**

- **链接: [http://arxiv.org/pdf/2506.14766v2](http://arxiv.org/pdf/2506.14766v2)**

> **作者:** Yujun Wang; Aniri; Jinhe Bi; Soeren Pirk; Yunpu Ma
>
> **备注:** 14 pages, 8 figures
>
> **摘要:** Multimodal large language models (MLLMs) frequently hallucinate by over-committing to spurious visual cues. Prior remedies-Visual and Instruction Contrastive Decoding (VCD, ICD)-mitigate this issue, yet the mechanism remains opaque. We first empirically show that their improvements systematically coincide with redistributions of cross-modal attention. Building on this insight, we propose Attention-Steerable Contrastive Decoding (ASCD), which directly steers the attention scores during decoding. ASCD combines (i) positive steering, which amplifies automatically mined text-centric heads-stable within a model and robust across domains-with (ii) negative steering, which dampens on-the-fly identified critical visual tokens. The method incurs negligible runtime and memory overhead and requires no additional training. Across five MLLM backbones and three decoding schemes, ASCD reduces hallucination on POPE, CHAIR, and MMHal-Bench by up to 38.2 percent while improving accuracy on standard VQA benchmarks, including MMMU, MM-VET, ScienceQA, TextVQA, and GQA. These results position attention steering as a simple, model-agnostic, and principled route to safer, more faithful multimodal generation.
>
---
#### [replaced 088] AutoGraph-R1: End-to-End Reinforcement Learning for Knowledge Graph Construction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.15339v2](http://arxiv.org/pdf/2510.15339v2)**

> **作者:** Hong Ting Tsang; Jiaxin Bai; Haoyu Huang; Qiao Xiao; Tianshi Zheng; Baixuan Xu; Shujie Liu; Yangqiu Song
>
> **摘要:** Building effective knowledge graphs (KGs) for Retrieval-Augmented Generation (RAG) is pivotal for advancing question answering (QA) systems. However, its effectiveness is hindered by a fundamental disconnect: the knowledge graph (KG) construction process is decoupled from its downstream application, yielding suboptimal graph structures. To bridge this gap, we introduce AutoGraph-R1, the first framework to directly optimize KG construction for task performance using Reinforcement Learning (RL). AutoGraph-R1 trains an LLM constructor by framing graph generation as a policy learning problem, where the reward is derived from the graph's functional utility in a RAG pipeline. We design two novel, task-aware reward functions, one for graphs as knowledge carriers and another as knowledge indices. Across multiple QA benchmarks, AutoGraph-R1 consistently enables graph RAG methods to achieve significant performance gains over using task-agnostic baseline graphs. Our work shows it is possible to close the loop between construction and application, shifting the paradigm from building intrinsically ``good'' graphs to building demonstrably ``useful'' ones.
>
---
#### [replaced 089] Supervised In-Context Fine-Tuning for Generative Sequence Labeling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.00921v2](http://arxiv.org/pdf/2509.00921v2)**

> **作者:** David Dukić; Goran Glavaš; Jan Šnajder
>
> **摘要:** Sequence labeling (SL) tasks, where labels are assigned to tokens, are abundant in NLP (e.g., named entity recognition and aspect-based sentiment analysis). Owing to the intuition that they require bidirectional context, SL tasks are commonly tackled with encoder-only models. Recent work also shows that removing the causal mask in fine-tuning enables decoder-based LLMs to become effective token classifiers. Less work, however, focused on (supervised) generative SL, a more natural setting for causal LLMs. Due to their rapid scaling, causal LLMs applied to SL are expected to outperform encoders, whose own development has stagnated. In this work, we propose supervised in-context fine-tuning (SIFT) for generative SL. SIFT casts SL tasks as constrained response generation, natural to LLMs, combining in-context learning (ICL) from demonstrations with supervised fine-tuning. SIFT considerably outperforms both ICL and decoder-as-encoder fine-tuning baselines on a range of standard SL tasks. We further find that although long context hinders the performance of generative SL in both ICL and SIFT, this deficiency can be mitigated by removing the instruction, as instructions are shown to be largely unnecessary for achieving strong SL performance with SIFT. Our findings highlight strengths and limitations of SL with LLMs, underscoring the importance of a response-based generative task formulation for effective SL performance.
>
---
#### [replaced 090] Controlling What You Share: Assessing Language Model Adherence to Privacy Preferences
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.05391v2](http://arxiv.org/pdf/2507.05391v2)**

> **作者:** Guillem Ramírez; Alexandra Birch; Ivan Titov
>
> **摘要:** Large language models (LLMs) are primarily accessed via commercial APIs, but this often requires users to expose their data to service providers. In this paper, we explore how users can stay in control of their data by using privacy profiles: simple natural language instructions that say what should and should not be revealed. We build a framework where a local model uses these instructions to rewrite queries, only hiding details deemed sensitive by the user, before sending them to an external model, thus balancing privacy with performance. To support this research, we introduce PEEP, a multilingual dataset of real user queries annotated to mark private content and paired with synthetic privacy profiles. Experiments with lightweight local LLMs show that, after fine-tuning, they not only achieve markedly better privacy preservation but also match or exceed the performance of much larger zero-shot models. At the same time, the system still faces challenges in fully adhering to user instructions, underscoring the need for models with a better understanding of user-defined privacy preferences.
>
---
#### [replaced 091] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05831v4](http://arxiv.org/pdf/2504.05831v4)**

> **作者:** Mingye Zhu; Yi Liu; Zheren Fu; Yongdong Zhang; Zhendong Mao
>
> **备注:** NeurIPS 2025
>
> **摘要:** Preference alignment methods are increasingly critical for steering large language models (LLMs) to generate outputs consistent with human values. While recent approaches often rely on synthetic data generated by LLMs for scalability and cost-efficiency reasons, this reliance can introduce distribution shifts that undermine the nuanced representation of human preferences needed for desirable outputs. In this paper, we propose a novel distribution-aware optimization framework that improves preference alignment despite such shifts. Our approach first leverages well-learned classifiers to assign a calibration value to each training sample, quantifying its alignment with the target human-preferred distribution. These values are then incorporated into a robust optimization objective that minimizes the worst-case loss over regions of the data space most relevant to human preferences. By explicitly focusing optimization on the target distribution, our approach mitigates the impact of distributional mismatch and improves the generation of responses that better reflect intended values.
>
---
#### [replaced 092] RPG: A Repository Planning Graph for Unified and Scalable Codebase Generation
- **分类: cs.CL; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2509.16198v5](http://arxiv.org/pdf/2509.16198v5)**

> **作者:** Jane Luo; Xin Zhang; Steven Liu; Jie Wu; Jianfeng Liu; Yiming Huang; Yangyu Huang; Chengyu Yin; Ying Xin; Yuefeng Zhan; Hao Sun; Qi Chen; Scarlett Li; Mao Yang
>
> **摘要:** Large language models excel at generating individual functions or single files of code, yet generating complete repositories from scratch remains a fundamental challenge. This capability is key to building coherent software systems from high-level specifications and realizing the full potential of automated code generation. The process requires planning at two levels: deciding what features and modules to build (proposal stage) and defining their implementation details (implementation stage). Current approaches rely on natural language planning, which often produces unclear specifications, misaligned components, and brittle designs due to its inherent ambiguity and lack of structure. To address these limitations, we introduce the Repository Planning Graph (RPG), a structured representation that encodes capabilities, file structures, data flows, and functions in a unified graph. By replacing free-form natural language with an explicit blueprint, RPG enables consistent long-horizon planning for repository generation. Building on RPG, we develop ZeroRepo, a graph-driven framework that operates in three stages: proposal-level planning, implementation-level construction, and graph-guided code generation with test validation. To evaluate, we construct RepoCraft, a benchmark of six real-world projects with 1,052 tasks. On RepoCraft, ZeroRepo produces nearly 36K Code Lines and 445K Code Tokens, on average 3.9$\times$ larger than the strongest baseline (Claude Code), and 68$\times$ larger than other baselines. It achieves 81.5% coverage and 69.7% test accuracy, improving over Claude Code by 27.3 and 35.8 points. Further analysis shows that RPG models complex dependencies, enables more sophisticated planning through near-linear scaling, and improves agent understanding of repositories, thus accelerating localization.
>
---
#### [replaced 093] A Knapsack by Any Other Name: Presentation impacts LLM performance on NP-hard problems
- **分类: cs.CL; cs.CC; 68Q15; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.13776v2](http://arxiv.org/pdf/2502.13776v2)**

> **作者:** Alex Duchnowski; Ellie Pavlick; Alexander Koller
>
> **备注:** 24 pages, 6 figures, EMNLP 2025
>
> **摘要:** To investigate the effect of problem presentation on LLMs' ability to solve optimization problems, we introduce the dataset of Everyday Hard Optimization Problems (EHOP), a collection of NP-hard problems expressed in natural language. EHOP includes problem formulations that could be found in computer science textbooks (e.g., graph coloring), versions that are dressed up as problems that could arise in real life (e.g., party planning), and variants with inverted rules. We find that state-of-the-art LLMs, across multiple prompting strategies, systematically solve textbook problems more accurately than their real-life and inverted counterparts. While reasoning models are more capable, they nonetheless show high variance across problem presentations, suggesting they lack a truly robust reasoning mechanism. We argue that this constitutes evidence that LLMs are still heavily dependent on what was seen in training and struggle to generalize to novel problems.
>
---
#### [replaced 094] DP-Fusion: Token-Level Differentially Private Inference for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.04531v2](http://arxiv.org/pdf/2507.04531v2)**

> **作者:** Rushil Thareja; Preslav Nakov; Praneeth Vepakomma; Nils Lukas
>
> **备注:** Our code and data are publicly available here: https://github.com/MBZUAI-Trustworthy-ML/DP-Fusion-DPI
>
> **摘要:** Large language models (LLMs) do not preserve privacy at inference-time. The LLM's outputs can inadvertently reveal information about the model's context, which presents a privacy challenge when the LLM is augmented via tools or databases containing sensitive information. Existing privacy-preserving methods at inference-time have significant limitations since they (i) lack provable guarantees or (ii) have a poor utility/privacy trade-off. We propose DP-Fusion, a Differentially Private Inference (DPI) mechanism for LLMs that provably bounds the influence a set of tokens in the context can have on the LLM's output. DP-Fusion works as follows: (1) label a subset of sensitive tokens, (2) infer the LLM without any sensitive tokens to obtain a baseline, (3) infer the LLM with the sensitive tokens, and (4) blend distributions so that the final output remains within a bounded distance of the baseline distribution. While this per-token influence bound also mitigates jailbreak-style prompt injection, we focus on \emph{document privatization}, where the goal is to paraphrase a document containing sensitive tokens, e.g., personally identifiable information, so that no attacker can reliably infer them from the paraphrased document while preserving high text quality. The privacy/utility trade-off is controlled by $\epsilon$, where $\epsilon=0$ hides sensitive tokens entirely, while higher values trade off privacy for improved text quality. We show that our method creates token-level provably privatized documents with substantially improved theoretical and empirical privacy, achieving $6\times$ lower perplexity than related DPI methods.
>
---
#### [replaced 095] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01113v2](http://arxiv.org/pdf/2502.01113v2)**

> **作者:** Linhao Luo; Zicheng Zhao; Gholamreza Haffari; Dinh Phung; Chen Gong; Shirui Pan
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Retrieval-augmented generation (RAG) has proven effective in integrating knowledge into large language models (LLMs). However, conventional RAGs struggle to capture complex relationships between pieces of knowledge, limiting their performance in intricate reasoning that requires integrating knowledge from multiple sources. Recently, graph-enhanced retrieval augmented generation (GraphRAG) builds graph structure to explicitly model these relationships, enabling more effective and efficient retrievers. Nevertheless, its performance is still hindered by the noise and incompleteness within the graph structure. To address this, we introduce GFM-RAG, a novel graph foundation model (GFM) for retrieval augmented generation. GFM-RAG is powered by an innovative graph neural network that reasons over graph structure to capture complex query-knowledge relationships. The GFM with 8M parameters undergoes a two-stage training process on large-scale datasets, comprising 60 knowledge graphs with over 14M triples and 700k documents. This results in impressive performance and generalizability for GFM-RAG, making it the first graph foundation model applicable to unseen datasets for retrieval without any fine-tuning required. Extensive experiments on three multi-hop QA datasets and seven domain-specific RAG datasets demonstrate that GFM-RAG achieves state-of-the-art performance while maintaining efficiency and alignment with neural scaling laws, highlighting its potential for further improvement.
>
---
#### [replaced 096] LLM as GNN: Graph Vocabulary Learning for Text-Attributed Graph Foundation Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.03313v3](http://arxiv.org/pdf/2503.03313v3)**

> **作者:** Xi Zhu; Haochen Xue; Ziwei Zhao; Wujiang Xu; Jingyuan Huang; Minghao Guo; Qifan Wang; Kaixiong Zhou; Imran Razzak; Yongfeng Zhang
>
> **摘要:** Text-Attributed Graphs (TAGs), where each node is associated with text descriptions, are ubiquitous in real-world scenarios. They typically exhibit distinctive structure and domain-specific knowledge, motivating the development of a Graph Foundation Model (GFM) that generalizes across diverse graphs and tasks. Despite large efforts to integrate Large Language Models (LLMs) and Graph Neural Networks (GNNs) for TAGs, existing approaches suffer from decoupled architectures with two-stage alignment, limiting their synergistic potential. Even worse, existing methods assign out-of-vocabulary (OOV) tokens to graph nodes, leading to graph-specific semantics, token explosion, and incompatibility with task-oriented prompt templates, which hinders cross-graph and cross-task transferability. To address these challenges, we propose PromptGFM, a versatile GFM for TAGs grounded in graph vocabulary learning. PromptGFM comprises two key components: (1) Graph Understanding Module, which explicitly prompts LLMs to replicate the finest GNN workflow within the text space, facilitating seamless GNN-LLM integration and elegant graph-text alignment; (2) Graph Inference Module, which establishes a language-based graph vocabulary ensuring expressiveness, transferability, and scalability, enabling readable instructions for LLM fine-tuning. Extensive experiments demonstrate our superiority and transferability across diverse graphs and tasks. The code is available at this: https://github.com/agiresearch/PromptGFM.
>
---
#### [replaced 097] Accelerating Mobile Language Model via Speculative Decoding and NPU-Coordinated Execution
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.15312v2](http://arxiv.org/pdf/2510.15312v2)**

> **作者:** Zhiyang Chen; Daliang Xu; Haiyang Shen; Mengwei Xu; Shangguang Wang; Yun Ma
>
> **摘要:** Enhancing on-device large language models (LLMs) with contextual information from local data enables personalized and task-aware generation, powering use cases such as intelligent assistants and UI agents. While recent developments in neural processors have substantially improved the efficiency of prefill on mobile devices, the token-by-token generation process still suffers from high latency and limited hardware utilization due to its inherently memory-bound characteristics. This work presents sd.npu, a mobile inference framework that integrates speculative decoding with dynamic hardware scheduling to accelerate context-aware text generation on mobile devices. The framework introduces three synergistic components: (1) adaptive execution scheduling, which dynamically balances compute graphs between prefill and decoding phases; (2) context-aligned drafting, which improves speculative efficiency through lightweight online calibration to current tasks; and (3) hardware-efficient draft extension, which reuses and expands intermediate sequences to improve processing parallelism and reduce verification cost. Experiments on multiple smartphones and representative workloads show consistent improvements of up to 3.8x in generation speed and 4.7x in energy efficiency compared with existing mobile inference solutions. Component-level analysis further validates the contribution of each optimization.
>
---
#### [replaced 098] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15692v4](http://arxiv.org/pdf/2505.15692v4)**

> **作者:** Jinyang Wu; Chonghua Liao; Mingkuan Feng; Shuai Zhang; Zhengqi Wen; Haoran Luo; Ling Yang; Huazhe Xu; Jianhua Tao
>
> **摘要:** Reinforcement learning (RL) has emerged as an effective paradigm for enhancing model reasoning. However, existing RL methods like GRPO often rely on unstructured self-sampling to fit scalar rewards, often producing inefficient rollouts that fail to capture transferable problem-solving strategies. To address these limitations, we propose **TemplateRL**, a structured template-guided RL framework that augments policy optimization with explicit template guidance. Our approach first constructs a problem-solving template library via MCTS on a small seed set, then seamlessly integrates this high-level structured guidance into RL training. By guiding rollout generation to align with proven template structures, TemplateRL significantly improves high-quality trajectory hit rates while reducing ineffective exploration. This structure-guided design steers the policy toward validated strategic patterns, stabilizing training dynamics, and enhancing RL sampling efficiency. Notably, the explicit template library is interpretable, editable, and supports online updates-enabling continuous updates during both training and inference. Extensive experiments demonstrate that TemplateRL outperforms GRPO by 99% on AIME and 41% on AMC, with superior stability on weak models and remarkable cross-domain generalization, highlighting its potential for broader tasks.
>
---
#### [replaced 099] Hope vs. Hate: Understanding User Interactions with LGBTQ+ News Content in Mainstream US News Media through the Lens of Hope Speech
- **分类: cs.CL; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.09004v3](http://arxiv.org/pdf/2502.09004v3)**

> **作者:** Jonathan Pofcher; Christopher M. Homan; Randall Sell; Ashiqur R. KhudaBukhsh
>
> **摘要:** This paper makes three contributions. First, via a substantial corpus of 1,419,047 comments posted on 3,161 YouTube news videos of major US cable news outlets, we analyze how users engage with LGBTQ+ news content. Our analyses focus both on positive and negative content. In particular, we construct a fine-grained hope speech classifier that detects positive (hope speech), negative, neutral, and irrelevant content. Second, in consultation with a public health expert specializing on LGBTQ+ health, we conduct an annotation study with a balanced and diverse political representation and release a dataset of 3,750 instances with fine-grained labels and detailed annotator demographic information. Finally, beyond providing a vital resource for the LGBTQ+ community, our annotation study and subsequent in-the-wild assessments reveal (1) strong association between rater political beliefs and how they rate content relevant to a marginalized community; (2) models trained on individual political beliefs exhibit considerable in-the-wild disagreement; and (3) zero-shot large language models (LLMs) align more with liberal raters.
>
---
#### [replaced 100] Geometric-Mean Policy Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20673v3](http://arxiv.org/pdf/2507.20673v3)**

> **作者:** Yuzhong Zhao; Yue Liu; Junpeng Liu; Jingye Chen; Xun Wu; Yaru Hao; Tengchao Lv; Shaohan Huang; Lei Cui; Qixiang Ye; Fang Wan; Furu Wei
>
> **备注:** Code is available at https://github.com/callsys/GMPO
>
> **摘要:** Group Relative Policy Optimization (GRPO) has significantly enhanced the reasoning capability of large language models by optimizing the arithmetic mean of token-level rewards. Unfortunately, GRPO is observed to suffer from unstable policy updates when facing tokens with outlier importance-weighted rewards, which manifest as extreme importance sampling ratios during training. In this study, we propose Geometric-Mean Policy Optimization (GMPO), with the aim to improve the stability of GRPO through suppressing token reward outliers. Instead of optimizing the arithmetic mean, GMPO maximizes the geometric mean of token-level rewards, which is inherently less sensitive to outliers and maintains a more stable range of importance sampling ratio. GMPO is plug-and-play-simply replacing GRPO's arithmetic mean with the geometric mean of token-level rewards, as the latter is inherently less sensitive to outliers. GMPO is theoretically plausible-analysis reveals that both GMPO and GRPO are weighted forms of the policy gradient while the former enjoys more stable weights, which consequently benefits policy optimization and performance. Experiments on multiple mathematical reasoning benchmarks show that GMPO-7B improves the average Pass@1 of GRPO by up to 4.1%, outperforming many state-of-the-art approaches. Code is available at https://github.com/callsys/GMPO.
>
---
#### [replaced 101] DICE: Structured Reasoning in LLMs through SLM-Guided Chain-of-Thought Correction
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.09211v2](http://arxiv.org/pdf/2510.09211v2)**

> **作者:** Yiqi Li; Yusheng Liao; Zhe Chen; Yanfeng Wang; Yu Wang
>
> **备注:** This paper was accepted to the EMNLP 2025 main conference
>
> **摘要:** When performing reasoning tasks with user-specific requirements, such as strict output formats, large language models (LLMs) often prioritize reasoning over adherence to detailed instructions. Fine-tuning LLMs on supervised datasets to address this is impractical due to high computational costs and limited parameter access. To tackle this, we propose DICE, a lightweight framework that guides small language models (SLMs) to refine LLMs' outputs through chain-of-thought (CoT) correction. DICE decouples the process by first prompting LLMs to generate natural language responses, then using trained SLMs to analyze and refine these outputs to meet structured output specifications. This framework preserves LLMs' broad knowledge and reasoning capabilities while ensuring the outputs conform to user demands. Specifically, DICE first constructs structured CoT adaptation datasets via a two-stage method and subsequently applies a dual-tuning strategy to fine-tune SLMs for generating structured outputs in an analyze-then-answer pattern. Experiments demonstrate that DICE improves the average format accuracy and content correctness of LLM outputs by 35.4\% and 29.4\%, respectively, achieving state-of-the-art (SOTA) performance over other competitive baselines.
>
---
#### [replaced 102] KG-TRACES: Enhancing Large Language Models with Knowledge Graph-constrained Trajectory Reasoning and Attribution Supervision
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.00783v2](http://arxiv.org/pdf/2506.00783v2)**

> **作者:** Rong Wu; Pinlong Cai; Jianbiao Mei; Licheng Wen; Tao Hu; Xuemeng Yang; Daocheng Fu; Botian Shi
>
> **备注:** 24 pages, 13 figures
>
> **摘要:** Large language models (LLMs) have made remarkable strides in various natural language processing tasks, but their performance on complex reasoning problems remains hindered by a lack of explainability and trustworthiness. This issue, often manifesting as hallucinations or unattributable reasoning processes, limits their applicability in complex reasoning scenarios. To address this, we propose Knowledge Graph-constrained Trajectory Reasoning Attribution and Chain Explanation Supervision (KG-TRACES), a novel framework that enhances the reasoning ability of LLMs through explicit supervision over reasoning paths and processes. KG-TRACES jointly supervises the model to: (1) predict symbolic relation paths, (2) predict full triple-level reasoning paths, and (3) generate attribution-aware reasoning processes grounded in the reasoning paths. At inference phase, the model adapts to both KG-available and KG-unavailable scenarios, retrieving reasoning paths from a KG when possible or predicting plausible reasoning paths with only intrinsic knowledge when not. This design enables the model to reason in an explainable and source-attributable pattern. Through extensive experiments on complex reasoning tasks, we demonstrate that KG-TRACES significantly outperforms existing SOTA: it improves Hits@1 by 1.6% and F1 by 4.7% on WebQSP, and achieves improvements of 4.8% in Hits@1 and 2.1% in F1 on CWQ. Moreover, we show its transferability to specialized domains such as medicine. By visualizing the intermediate steps of reasoning processes, we further show that the explicit supervision introduced by KG-TRACES leads to more stable and goal-directed reasoning processes, aligning closely with correct answers. Code is available at https://github.com/Edaizi/KG-TRACES.
>
---
#### [replaced 103] HauntAttack: When Attack Follows Reasoning as a Shadow
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07031v2](http://arxiv.org/pdf/2506.07031v2)**

> **作者:** Jingyuan Ma; Rui Li; Zheng Li; Junfeng Liu; Lei Sha; Zhifang Sui
>
> **摘要:** Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing remarkable capabilities. However, the enhancement of reasoning abilities and the exposure of internal reasoning processes introduce new safety vulnerabilities. A critical question arises: when reasoning becomes intertwined with harmfulness, will LRMs become more vulnerable to jailbreaks in reasoning mode? To investigate this, we introduce HauntAttack, a novel and general-purpose black-box adversarial attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we modify key reasoning conditions in existing questions with harmful instructions, thereby constructing a reasoning pathway that guides the model step by step toward unsafe outputs. We evaluate HauntAttack on 11 LRMs and observe an average attack success rate of 70\%, achieving up to 12 percentage points of absolute improvement over the strongest prior baseline. Our further analysis reveals that even advanced safety-aligned models remain highly susceptible to reasoning-based attacks, offering insights into the urgent challenge of balancing reasoning capability and safety in future model development.
>
---
#### [replaced 104] Grounding Language with Vision: A Conditional Mutual Information Calibrated Decoding Strategy for Reducing Hallucinations in LVLMs
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19678v2](http://arxiv.org/pdf/2505.19678v2)**

> **作者:** Hao Fang; Changle Zhou; Jiawei Kong; Kuofeng Gao; Bin Chen; Tao Liang; Guojun Ma; Shu-Tao Xia
>
> **摘要:** Large Vision-Language Models (LVLMs) are susceptible to hallucinations, where generated responses seem semantically plausible yet exhibit little or no relevance to the input image. Previous studies reveal that this issue primarily stems from LVLMs' over-reliance on language priors while disregarding the visual information during decoding. To alleviate this issue, we introduce a novel Conditional Pointwise Mutual Information (C-PMI) calibrated decoding strategy, which adaptively strengthens the mutual dependency between generated texts and input images to mitigate hallucinations. Unlike existing methods solely focusing on text token sampling, we propose to jointly model the contributions of visual and textual tokens to C-PMI, formulating hallucination mitigation as a bi-level optimization problem aimed at maximizing mutual information. To solve it, we design a token purification mechanism that dynamically regulates the decoding process by sampling text tokens remaining maximally relevant to the given image, while simultaneously refining image tokens most pertinent to the generated response. Extensive experiments across various benchmarks reveal that the proposed method significantly reduces hallucinations in LVLMs while preserving decoding efficiency.
>
---
#### [replaced 105] SHANKS: Simultaneous Hearing and Thinking for Spoken Language Models
- **分类: cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.06917v2](http://arxiv.org/pdf/2510.06917v2)**

> **作者:** Cheng-Han Chiang; Xiaofei Wang; Linjie Li; Chung-Ching Lin; Kevin Lin; Shujie Liu; Zhendong Wang; Zhengyuan Yang; Hung-yi Lee; Lijuan Wang
>
> **备注:** Work in progress
>
> **摘要:** Current large language models (LLMs) and spoken language models (SLMs) begin thinking and taking actions only after the user has finished their turn. This prevents the model from interacting during the user's turn and can lead to high response latency while it waits to think. Consequently, thinking after receiving the full input is not suitable for speech-to-speech interaction, where real-time, low-latency exchange is important. We address this by noting that humans naturally "think while listening." In this paper, we propose SHANKS, a general inference framework that enables SLMs to generate unspoken chain-of-thought reasoning while listening to the user input. SHANKS streams the input speech in fixed-duration chunks and, as soon as a chunk is received, generates unspoken reasoning based on all previous speech and reasoning, while the user continues speaking. SHANKS uses this unspoken reasoning to decide whether to interrupt the user and to make tool calls to complete the task. We demonstrate that SHANKS enhances real-time user-SLM interaction in two scenarios: (1) when the user is presenting a step-by-step solution to a math problem, SHANKS can listen, reason, and interrupt when the user makes a mistake, achieving 37.1% higher interruption accuracy than a baseline that interrupts without thinking; and (2) in a tool-augmented dialogue, SHANKS can complete 56.9% of the tool calls before the user finishes their turn. Overall, SHANKS moves toward models that keep thinking throughout the conversation, not only after a turn ends. Animated illustrations of Shanks can be found at https://d223302.github.io/SHANKS/
>
---
#### [replaced 106] Beyond One World: Benchmarking Super Heros in Role-Playing Across Multiversal Contexts
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.14351v2](http://arxiv.org/pdf/2510.14351v2)**

> **作者:** Perapard Ngokpol; Kun Kerdthaisong; Pasin Buakhaw; Pitikorn Khlaisamniang; Supasate Vorathammathorn; Piyalitt Ittichaiwong; Nutchanon Yongsatianchot
>
> **摘要:** Large language models (LLMs) are increasingly used as role-playing agents, yet their capacity to faithfully and consistently portray version-specific characters -- for example, superheroes across comic and cinematic universes -- remains underexplored. Superhero canons such as Marvel and DC provide a rich testbed: decades of storytelling yield multiple incarnations of the same character with distinct histories, values, and moral codes. To study this problem, we introduce Beyond One World, a benchmark for character-grounded roleplay spanning 30 iconic heroes and 90 canon-specific versions. The benchmark comprises two tasks: (i) Canon Events, which probes factual recall of pivotal life stages, and (ii) Moral Dilemmas, which confronts models with ethically charged scenarios. We score responses for canonical accuracy and reasoning fidelity under a framework that separates internal deliberation ("thinking") from outward decisions ("acting"). We further propose Think-Act Matching, a metric that quantifies alignment between reasons and actions and serves as a proxy for model trustworthiness. Experiments across reasoning- and non-reasoning-oriented models yield three findings: (1) chain-of-thought prompting improves narrative coherence in weaker models but can reduce canonical accuracy in stronger ones; (2) cross-version generalization within a character remains a major obstacle; and (3) models often excel at either thinking or acting, but rarely both. Beyond One World exposes critical gaps in multiversal consistency and reasoning alignment, offering a challenging evaluation for role-playing LLMs.
>
---
#### [replaced 107] Agentic Design of Compositional Machines
- **分类: cs.AI; cs.CL; cs.CV; cs.GR; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.14980v2](http://arxiv.org/pdf/2510.14980v2)**

> **作者:** Wenqian Zhang; Weiyang Liu; Zhen Liu
>
> **备注:** 75 pages, 31 figures, Project Page: https://besiegefield.github.io
>
> **摘要:** The design of complex machines stands as both a marker of human intelligence and a foundation of engineering practice. Given recent advances in large language models (LLMs), we ask whether they, too, can learn to create. We approach this question through the lens of compositional machine design: a task in which machines are assembled from standardized components to meet functional demands like locomotion or manipulation in a simulated physical environment. With this simplification, machine design is expressed as writing XML-like code that explicitly specifies pairwise part connections. To support this investigation, we introduce BesiegeField, a testbed built on the machine-building game Besiege, which enables part-based construction, physical simulation and reward-driven evaluation. Using BesiegeField, we benchmark state-of-the-art LLMs with agentic workflows and identify key capabilities required for success, including spatial reasoning, strategic assembly, and instruction-following. As current open-source models fall short, we explore reinforcement learning (RL) as a path to improvement: we curate a cold-start dataset, conduct RL finetuning experiments, and highlight open challenges at the intersection of language, machine design, and physical reasoning.
>
---
#### [replaced 108] Limitations of Normalization in Attention Mechanism
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.17821v2](http://arxiv.org/pdf/2508.17821v2)**

> **作者:** Timur Mudarisov; Mikhail Burtsev; Tatiana Petrova; Radu State
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** This paper investigates the limitations of the normalization in attention mechanisms. We begin with a theoretical framework that enables the identification of the model's selective ability and the geometric separation involved in token selection. Our analysis includes explicit bounds on distances and separation criteria for token vectors under softmax scaling. Through experiments with pre-trained GPT-2 model, we empirically validate our theoretical results and analyze key behaviors of the attention mechanism. Notably, we demonstrate that as the number of selected tokens increases, the model's ability to distinguish informative tokens declines, often converging toward a uniform selection pattern. We also show that gradient sensitivity under softmax normalization presents challenges during training, especially at low temperature settings. These findings advance current understanding of softmax-based attention mechanism and motivate the need for more robust normalization and selection strategies in future attention architectures.
>
---
#### [replaced 109] CAPO: Towards Enhancing LLM Reasoning through Generative Credit Assignment
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.02298v4](http://arxiv.org/pdf/2508.02298v4)**

> **作者:** Guofu Xie; Yunsheng Shi; Hongtao Tian; Ting Yao; Xiao Zhang
>
> **备注:** Work in progress
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of Large Language Models (LLMs) by using rule-based binary feedback. However, current RLVR methods typically assign the same reward to every token. This coarse-grained feedback hampers precise credit assignment, making it hard for models to identify which reasoning steps lead to success or failure, and often results in suboptimal policies. Methods like PPO provide credit assignment by value estimation, but yield inaccurate and unverifiable signals due to limited sampling. On the other hand, methods using Process Reward Models can provide step-wise rewards but suffer from several key limitations: they require high-quality process supervision labels, the feedback is unreliable due to probabilistic reward modeling, and their application in online reinforcement learning (RL) is time-consuming. To overcome these limitations, we introduce a simple but efficient method-Credit Assignment Policy Optimization (CAPO). Instead of training auxiliary models, CAPO directly leverages an off-the-shelf, general-purpose LLM as a Generative Process Reward Model (LLM-as-GenPRM) to generate all step-wise critique by one pass only based on the correctness of the step itself, providing deterministic token-level credits to refine the tokens that were originally assigned identical rule-based rewards. To further enhance the accuracy and robustness, we employ voting mechanisms that scale with the number of generated critiques. Extensive experiments on various backbones like Llama and Qwen models show that CAPO consistently outperforms supervised learning-based and RL-based fine-tuning methods across four challenging mathematical benchmarks and three out-of-domain benchmarks. Further analysis shows that CAPO can help the model to foster the learning of correct reasoning pathways leading to correct answers.
>
---
#### [replaced 110] CultureGuard: Towards Culturally-Aware Dataset and Guard Model for Multilingual Safety Applications
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.01710v3](http://arxiv.org/pdf/2508.01710v3)**

> **作者:** Raviraj Joshi; Rakesh Paul; Kanishk Singla; Anusha Kamath; Michael Evans; Katherine Luna; Shaona Ghosh; Utkarsh Vaidya; Eileen Long; Sanjay Singh Chauhan; Niranjan Wartikar
>
> **摘要:** The increasing use of Large Language Models (LLMs) in agentic applications highlights the need for robust safety guard models. While content safety in English is well-studied, non-English languages lack similar advancements due to the high cost of collecting culturally aligned labeled datasets. We present CultureGuard, a novel solution for curating culturally aligned, high-quality safety datasets across multiple languages. Our approach introduces a four-stage synthetic data generation and filtering pipeline: cultural data segregation, cultural data adaptation, machine translation, and quality filtering. This pipeline enables the conversion and expansion of the Nemotron-Content-Safety-Dataset-V2 English safety dataset into eight distinct languages: Arabic, German, Spanish, French, Hindi, Japanese, Thai, and Chinese. The resulting dataset, Nemotron-Safety-Guard-Dataset-v3, comprises 386,661 samples in 9 languages and facilitates the training of Llama-3.1-Nemotron-Safety-Guard-8B-v3 via LoRA-based fine-tuning. The final model achieves state-of-the-art performance on several multilingual content safety benchmarks. Furthermore, we show our moderately multilingual fine-tuning enables robust cross-lingual transfer and strong zero-shot generalization to unseen languages. We also benchmark the latest open LLMs on multilingual safety and observe that these LLMs are more prone to give unsafe responses when prompted in non-English languages. This work advances multilingual LLM safety by enabling the development of culturally aware safety guard models.
>
---
#### [replaced 111] Valid Survey Simulations with Limited Human Data: The Roles of Prompting, Fine-Tuning, and Rectification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.11408v2](http://arxiv.org/pdf/2510.11408v2)**

> **作者:** Stefan Krsteski; Giuseppe Russo; Serina Chang; Robert West; Kristina Gligorić
>
> **备注:** 19 pages, 4 figures, 9 tables
>
> **摘要:** Surveys provide valuable insights into public opinion and behavior, but their execution is costly and slow. Large language models (LLMs) have been proposed as a scalable, low-cost substitute for human respondents, but their outputs are often biased and yield invalid estimates. We study the interplay between synthesis methods that use LLMs to generate survey responses and rectification methods that debias population estimates, and explore how human responses are best allocated between them. Using two panel surveys with questions on nutrition, politics, and economics, we find that synthesis alone introduces substantial bias (24-86%), whereas combining it with rectification reduces bias below 5% and increases effective sample size by up to 14%. Overall, we challenge the common practice of using all human responses for fine-tuning, showing that under a fixed budget, allocating most to rectification results in far more effective estimation.
>
---
#### [replaced 112] A Comparison of Independent and Joint Fine-tuning Strategies for Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.01600v2](http://arxiv.org/pdf/2510.01600v2)**

> **作者:** Neal Gregory Lawton; Alfy Samuel; Anoop Kumar; Daben Liu
>
> **摘要:** A Comparison of Independent and Joint Fine-tuning Strategies for Retrieval-Augmented Generation Download PDF Neal Gregory Lawton, Alfy Samuel, Anoop Kumar, Daben Liu Published: 20 Aug 2025, Retrieval augmented generation (RAG) is a popular framework for question answering that is powered by two large language models (LLMs): an embedding model that retrieves context documents from a database that are relevant to a given question, and a generator model that uses the retrieved context to generate an answer to the question. Both the embedding and generator models can be fine-tuned to increase performance of a RAG pipeline on a new task, but multiple fine-tuning strategies exist with different costs and benefits. In this paper, we evaluate and compare several RAG fine-tuning strategies, including independent, joint, and two-phase fine-tuning. In our experiments, we observe that all of these strategies achieve about equal improvement in EM and F1 generation quality metrics, although they have significantly different computational costs. We conclude the optimal fine-tuning strategy to use depends on whether the training dataset includes context labels and whether a grid search over the learning rates for the embedding and generator models is required.
>
---
#### [replaced 113] BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design
- **分类: cs.CL; cs.AI; stat.ML**

- **链接: [http://arxiv.org/pdf/2508.21184v2](http://arxiv.org/pdf/2508.21184v2)**

> **作者:** Deepro Choudhury; Sinead Williamson; Adam Goliński; Ning Miao; Freddie Bickford Smith; Michael Kirchhof; Yizhe Zhang; Tom Rainforth
>
> **摘要:** We propose a general-purpose approach for improving the ability of Large Language Models (LLMs) to intelligently and adaptively gather information from a user or other external source using the framework of sequential Bayesian experimental design (BED). This enables LLMs to act as effective multi-turn conversational agents and interactively interface with external environments. Our approach, which we call BED-LLM (Bayesian Experimental Design with Large Language Models), is based on iteratively choosing questions or queries that maximize the expected information gain (EIG) about the task of interest given the responses gathered previously. We show how this EIG can be formulated (and then estimated) in a principled way using a probabilistic model derived from the LLM's predictive distributions and provide detailed insights into key decisions in its construction and updating procedure. We find that BED-LLM achieves substantial gains in performance across a wide range of tests based on the 20 questions game and using the LLM to actively infer user preferences, compared to direct prompting of the LLM and other adaptive design strategies.
>
---
#### [replaced 114] $Q\sharp$: Provably Optimal Distributional RL for LLM Post-Training
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20548v2](http://arxiv.org/pdf/2502.20548v2)**

> **作者:** Jin Peng Zhou; Kaiwen Wang; Jonathan Chang; Zhaolin Gao; Nathan Kallus; Kilian Q. Weinberger; Kianté Brantley; Wen Sun
>
> **备注:** NeurIPS 2025
>
> **摘要:** Reinforcement learning (RL) post-training is crucial for LLM alignment and reasoning, but existing policy-based methods, such as PPO and DPO, can fall short of fixing shortcuts inherited from pre-training. In this work, we introduce $Q\sharp$, a value-based algorithm for KL-regularized RL that guides the reference policy using the optimal regularized $Q$ function. We propose to learn the optimal $Q$ function using distributional RL on an aggregated online dataset. Unlike prior value-based baselines that guide the model using unregularized $Q$-values, our method is theoretically principled and provably learns the optimal policy for the KL-regularized RL problem. Empirically, $Q\sharp$ outperforms prior baselines in math reasoning benchmarks while maintaining a smaller KL divergence to the reference policy. Theoretically, we establish a reduction from KL-regularized RL to no-regret online learning, providing the first bounds for deterministic MDPs under only realizability. Thanks to distributional RL, our bounds are also variance-dependent and converge faster when the reference policy has small variance. In sum, our results highlight $Q\sharp$ as an effective approach for post-training LLMs, offering both improved performance and theoretical guarantees. The code can be found at https://github.com/jinpz/q_sharp.
>
---
#### [replaced 115] Evolving LLMs' Self-Refinement Capability via Iterative Preference Optimization
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.05605v5](http://arxiv.org/pdf/2502.05605v5)**

> **作者:** Yongcheng Zeng; Xinyu Cui; Xuanfa Jin; Qirui Mi; Guoqing Liu; Zexu Sun; Mengyue Yang; Dong Li; Weiyu Ma; Ning Yang; Jian Zhao; Jianye Hao; Haifeng Zhang; Jun Wang
>
> **摘要:** Self-Refinement refers to a model's ability to revise its own responses to produce improved outputs. This capability can also serve as a fundamental mechanism for Self-Improvement, for example, by reconstructing datasets with refined results to enhance intrinsic model performance. However, our comprehensive experiments reveal that large language models (LLMs) show no clear evidence of inherent Self-Refinement and may even experience response quality degradation after Self-Refinement. To address this issue, we propose EVOLVE, a simple and effective framework for eliciting and tracking the evolution of Self-Refinement through iterative training. We first explore optimization methods during training to activate the model's Self-Refinement capability. Then, at inference, we investigate various generation strategies to further enhance and utilize Self-Refinement while supplying the necessary data for training. Through synergistic optimization of training and inference stages, we continually evolve the model's Self-Refinement ability, enabling it to better refine its own responses. Moreover, we demonstrate the potential of leveraging Self-Refinement to achieve broader Self-Improvement of intrinsic model abilities. Experiments show that the evolved Self-Refinement ability enables the Llama-3.1-8B base model to surpass GPT-4o, achieving 62.3% length-controlled and 63.3% raw win rates on AlpacaEval 2, and 50.3% on Arena-Hard. It also generalizes effectively to out-of-domain reasoning tasks, improving performance on mathematical reasoning benchmarks such as GSM8K and MATH.
>
---
#### [replaced 116] Repo2Run: Automated Building Executable Environment for Code Repository at Scale
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13681v4](http://arxiv.org/pdf/2502.13681v4)**

> **作者:** Ruida Hu; Chao Peng; Xinchen Wang; Junjielong Xu; Cuiyun Gao
>
> **摘要:** Scaling up executable code data is significant for improving language models' software engineering capability. The intricate nature of the process makes it labor-intensive, time-consuming and expert-knowledge-dependent to build a large number of executable code repositories, limiting the scalability of existing work based on running tests. The primary bottleneck lies in the automated building of test environments for different repositories, which is an essential yet underexplored task. To mitigate the gap, we introduce Repo2Run, the first LLM-based agent aiming at automating the building of executable test environments for any repositories at scale. Specifically, given a code repository, Repo2Run iteratively builds the Docker image, runs unit tests based on the feedback of the building, and synthesizes the Dockerfile until the entire pipeline is executed successfully. The resulting Dockerfile can then be used to create Docker container environments for running code and tests. We created a benchmark containing 420 Python repositories with unit tests for evaluation. The results illustrate that Repo2Run achieves an 86.0% success rate, outperforming SWE-agent by 77.0%. The resources of Repo2Run are available at https://github.com/bytedance/Repo2Run.
>
---
#### [replaced 117] Exploration of Marker-Based Approaches in Argument Mining through Augmented Natural Language
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.08606v3](http://arxiv.org/pdf/2406.08606v3)**

> **作者:** Nilmadhab Das; Vishal Choudhary; V. Vijaya Saradhi; Ashish Anand
>
> **备注:** Accepted version. To appear in the IJCNN 2025 Proceedings
>
> **摘要:** Argument Mining (AM) involves identifying and extracting Argumentative Components (ACs) and their corresponding Argumentative Relations (ARs). Most of the prior works have broken down these tasks into multiple sub-tasks. Existing end-to-end setups primarily use the dependency parsing approach. This work introduces a generative paradigm-based end-to-end framework argTANL. argTANL frames the argumentative structures into label-augmented text, called Augmented Natural Language (ANL). This framework jointly extracts both ACs and ARs from a given argumentative text. Additionally, this study explores the impact of Argumentative and Discourse markers on enhancing the model's performance within the proposed framework. Two distinct frameworks, Marker-Enhanced argTANL (ME-argTANL) and argTANL with specialized Marker-Based Fine-Tuning, are proposed to achieve this. Extensive experiments are conducted on three standard AM benchmarks to demonstrate the superior performance of the ME-argTANL.
>
---
#### [replaced 118] Your Next Token Prediction: A Multilingual Benchmark for Personalized Response Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.14398v2](http://arxiv.org/pdf/2510.14398v2)**

> **作者:** Shiyao Ding; Takayuki Ito
>
> **摘要:** Large language models (LLMs) excel at general next-token prediction but still struggle to generate responses that reflect how individuals truly communicate, such as replying to emails or social messages in their own style. However, real SNS or email histories are difficult to collect due to privacy concerns. To address this, we propose the task of "Your Next Token Prediction (YNTP)", which models a user's precise word choices through controlled human-agent conversations. We build a multilingual benchmark of 100 dialogue sessions across English, Japanese, and Chinese, where users interact for five days with psychologically grounded NPCs based on MBTI dimensions. This setup captures natural, daily-life communication patterns and enables analysis of users' internal models. We evaluate prompt-based and fine-tuning-based personalization methods, establishing the first benchmark for YNTP and a foundation for user-aligned language modeling. The dataset is available at: https://github.com/AnonymousHub4Submissions/your-next-token-prediction-dataset-100
>
---
#### [replaced 119] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.12535v2](http://arxiv.org/pdf/2508.12535v2)**

> **作者:** Seonglae Cho; Zekun Wu; Adriano Koshiyama
>
> **备注:** 42 pages, 9 tables
>
> **摘要:** Sparse Autoencoders (SAEs) can extract interpretable features from large language models (LLMs) without supervision. However, their effectiveness in downstream steering tasks is limited by the requirement for contrastive datasets or large activation storage. To address these limitations, we propose CorrSteer, which selects features by correlating sample correctness with SAE activations from generated tokens at inference time. This approach uses only inference-time activations to extract more relevant features, thereby reducing spurious correlations. It also obtains steering coefficients from average activations, automating the entire pipeline. Our method shows improved task performance on QA, bias mitigation, jailbreaking prevention, and reasoning benchmarks on Gemma-2 2B and LLaMA-3.1 8B, notably achieving a +3.3% improvement in MMLU performance with 4000 samples and a +27.2% improvement in HarmBench with only 108 samples. Selected features demonstrate semantically meaningful patterns aligned with each task's requirements, revealing the underlying capabilities that drive performance. Our work establishes correlation-based selection as an effective and scalable approach for automated SAE steering across language model applications.
>
---
#### [replaced 120] mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.02348v2](http://arxiv.org/pdf/2510.02348v2)**

> **作者:** Guy Dar
>
> **摘要:** We build upon vec2vec, a procedure designed to align text embedding spaces without parallel data. vec2vec finds a near-perfect alignment, but it is expensive and unstable. We present mini-vec2vec, a simple and efficient alternative that requires substantially lower computational cost and is highly robust. Moreover, the learned mapping is a linear transformation. Our method consists of three main stages: a tentative matching of pseudo-parallel embedding vectors, transformation fitting, and iterative refinement. Our linear alternative exceeds the original instantiation of vec2vec by orders of magnitude in efficiency, while matching or exceeding their results. The method's stability and interpretable algorithmic steps facilitate scaling and unlock new opportunities for adoption in new domains and fields.
>
---
#### [replaced 121] Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.07163v4](http://arxiv.org/pdf/2410.07163v4)**

> **作者:** Chongyu Fan; Jiancheng Liu; Licong Lin; Jinghan Jia; Ruiqi Zhang; Song Mei; Sijia Liu
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** This work studies the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences (e.g., copyrighted or harmful content) while preserving model utility. Despite the increasing demand for unlearning, a technically-grounded optimization framework is lacking. Gradient ascent (GA)-type methods, though widely used, are suboptimal as they reverse the learning process without controlling optimization divergence (i.e., deviation from the pre-trained state), leading to risks of over-forgetting and potential model collapse. Negative preference optimization (NPO) has been proposed to address this issue and is considered one of the state-of-the-art LLM unlearning approaches. In this work, we revisit NPO and identify another critical issue: reference model bias. This bias arises from using the reference model (i.e., the model prior to unlearning) to evaluate the unlearning success, which can compromise NPO's effectiveness. Specifically, it leads to (a) uneven allocation of optimization power across forget data with varying difficulty levels and (b) ineffective gradient weight smoothing during the early stages of unlearning optimization. To overcome these challenges, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that `simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We provide deeper insights into SimNPO's advantages through an analysis based on mixtures of Markov chains. Extensive experiments further validate SimNPO's efficacy on benchmarks like TOFU and MUSE, as well as its robustness against relearning attacks. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple.
>
---
#### [replaced 122] Late Fusion and Multi-Level Fission Amplify Cross-Modal Transfer in Text-Speech LMs
- **分类: cs.CL; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.06211v2](http://arxiv.org/pdf/2503.06211v2)**

> **作者:** Santiago Cuervo; Adel Moumen; Yanis Labrak; Sameer Khurana; Antoine Laurent; Mickael Rouvier; Phil Woodland; Ricard Marxer
>
> **摘要:** Text-Speech Language Models (TSLMs) -- language models trained to jointly process and generate text and speech -- are commonly trained through an early modality fusion/fission approach, in which both modalities are fed and predicted from a shared backbone via linear layers. We hypothesize that this approach limits cross-modal transfer by neglecting feature compositionality -- specifically, the finer-grained nature of speech representations compared to text -- preventing the emergence of a shared feature hierarchy within model layers. In this paper, we argue that this limitation can be addressed through late fusion and fission, with a fission process that accesses both high- and low-level features for speech generation. Our models implementing these principles, SmolTolk, rival or surpass state-of-the-art TSLMs trained with orders of magnitude more compute, and achieve significantly improved cross-modal performance relative to early fusion/fission baselines. Representation analyses further suggest that our method enhances the model's ability to abstract higher-level, more semantic features from speech, and leads to increasingly shared representation spaces across layers.
>
---
#### [replaced 123] Adaptive Data-Resilient Multi-Modal Hierarchical Multi-Label Book Genre Identification
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.03839v2](http://arxiv.org/pdf/2505.03839v2)**

> **作者:** Utsav Kumar Nareti; Soumi Chattopadhyay; Prolay Mallick; Suraj Kumar; Chandranath Adak; Ayush Vikas Daga; Adarsh Wase; Arjab Roy
>
> **摘要:** Identifying fine-grained book genres is essential for enhancing user experience through efficient discovery, personalized recommendations, and improved reader engagement. At the same time, it provides publishers and marketers with valuable insights into consumer preferences and emerging market trends. While traditional genre classification methods predominantly rely on textual reviews or content analysis, the integration of additional modalities, such as book covers, blurbs, and metadata, offers richer contextual cues. However, the effectiveness of such multi-modal systems is often hindered by incomplete, noisy, or missing data across modalities. To address this, we propose IMAGINE (Intelligent Multi-modal Adaptive Genre Identification NEtwork), a framework designed to leverage multi-modal data while remaining robust to missing or unreliable information. IMAGINE learns modality-specific feature representations and adaptively prioritizes the most informative sources available at inference time. It further employs a hierarchical classification strategy, grounded in a curated taxonomy of book genres, to capture inter-genre relationships and support multi-label assignments reflective of real-world literary diversity. A key strength of IMAGINE is its adaptability: it maintains high predictive performance even when one modality, such as text or image, is unavailable. We also curated a large-scale hierarchical dataset that structures book genres into multiple levels of granularity, allowing for a more comprehensive evaluation. Experimental results demonstrate that IMAGINE outperformed strong baselines in various settings, with significant gains in scenarios involving incomplete modality-specific data.
>
---
#### [replaced 124] Large Language Models are Powerful Electronic Health Record Encoders
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17403v4](http://arxiv.org/pdf/2502.17403v4)**

> **作者:** Stefan Hegselmann; Georg von Arnim; Tillmann Rheude; Noel Kronenberg; David Sontag; Gerhard Hindricks; Roland Eils; Benjamin Wild
>
> **摘要:** Electronic Health Records (EHRs) offer considerable potential for clinical prediction, but their complexity and heterogeneity present significant challenges for traditional machine learning methods. Recently, domain-specific EHR foundation models trained on large volumes of unlabeled EHR data have shown improved predictive accuracy and generalization. However, their development is constrained by limited access to diverse, high-quality datasets, and inconsistencies in coding standards and clinical practices. In this study, we explore the use of general-purpose Large Language Models (LLMs) to encode EHR into high-dimensional representations for downstream clinical prediction tasks. We convert structured EHR data into Markdown-formatted plain-text documents by replacing medical codes with natural language descriptions. This enables the use of LLMs and their extensive semantic understanding and generalization capabilities as effective encoders of EHRs without requiring access to private medical training data. We show that LLM-based embeddings can often match or even surpass the performance of a specialized EHR foundation model, CLMBR-T-Base, across 15 diverse clinical tasks from the EHRSHOT benchmark. Critically, our approach requires no institution-specific training and can incorporate any medical code with a text description, whereas existing EHR foundation models operate on fixed vocabularies and can only process codes seen during pretraining. To demonstrate generalizability, we further evaluate the approach on the UK Biobank (UKB) cohort, out-of-domain for CLMBR-T-Base, whose fixed vocabulary covers only 16% of UKB codes. Notably, an LLM-based model achieves superior performance for prediction of disease onset, hospitalization, and mortality, indicating robustness to population and coding shifts.
>
---
#### [replaced 125] Infinity Parser: Layout Aware Reinforcement Learning for Scanned Document Parsing
- **分类: cs.CL; F.2.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2510.15349v2](http://arxiv.org/pdf/2510.15349v2)**

> **作者:** Baode Wang; Biao Wu; Weizhen Li; Meng Fang; Zuming Huang; Jun Huang; Haozhe Wang; Yanjie Liang; Ling Chen; Wei Chu; Yuan Qi
>
> **备注:** This submission (arXiv:2510.15349) was mistakenly uploaded as a new article. It was intended to replace our previous work arXiv:2506.03197. All subsequent updates will be made to arXiv:2506.03197
>
> **摘要:** Document parsing from scanned images into structured formats remains a significant challenge due to its complexly intertwined elements such as text paragraphs, figures, formulas, and tables. Existing supervised fine-tuning methods often struggle to generalize across diverse document types, leading to poor performance, particularly on out-of-distribution data. This issue is further exacerbated by the limited availability of high-quality training data for layout-aware parsing tasks. To address these challenges, we introduce LayoutRL, a reinforcement learning framework that optimizes layout understanding through composite rewards integrating normalized edit distance, paragraph count accuracy, and reading order preservation. To support this training, we construct the Infinity-Doc-400K dataset, which we use to train Infinity-Parser, a vision-language model demonstrating robust generalization across various domains. Extensive evaluations on benchmarks including OmniDocBench, olmOCR-Bench, PubTabNet, and FinTabNet show that Infinity-Parser consistently achieves state-of-the-art performance across a broad range of document types, languages, and structural complexities, substantially outperforming both specialized document parsing systems and general-purpose vision-language models. We will release our code, dataset, and model to facilitate reproducible research in document parsing.
>
---
#### [replaced 126] Lost at the Beginning of Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.22058v3](http://arxiv.org/pdf/2506.22058v3)**

> **作者:** Baohao Liao; Xinyi Chen; Sara Rajaee; Yuhui Xu; Christian Herold; Anders Søgaard; Maarten de Rijke; Christof Monz
>
> **备注:** remove the benchmark part. (10 pages, 6 figures, 5 tables)
>
> **摘要:** Recent advancements in large language models (LLMs) have significantly advanced complex reasoning capabilities, particularly through extended chain-of-thought (CoT) reasoning that incorporates mechanisms such as backtracking, self-reflection, and self-correction. Despite these developments, the self-correction abilities of LLMs during long CoT reasoning remain underexplored. And recent findings on overthinking suggest that such models often engage in unnecessarily redundant reasoning. In this work, we empirically show that the first reasoning step exerts a disproportionately large influence on the final prediction. I.e., errors introduced at this stage can substantially degrade subsequent reasoning quality. This phenomenon is consistently observed across various state-of-the-art open- and closed-source reasoning models. Leveraging this insight, we propose an efficient sampling strategy that leverages a reward model to identify and retain high-quality first reasoning steps while discarding suboptimal ones, achieving up to a 70% reduction in inference cost without sacrificing any accuracy. Our work highlights the central role of the first reasoning step in generating a high-quality reasoning trajectory, and thus enabling significantly efficient sampling.
>
---
#### [replaced 127] Evaluating Retrieval-Augmented Generation Systems on Unanswerable, Uncheatable, Realistic, Multi-hop Queries
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2510.11956v2](http://arxiv.org/pdf/2510.11956v2)**

> **作者:** Gabrielle Kaili-May Liu; Bryan Li; Arman Cohan; William Gantt Walden; Eugene Yang
>
> **摘要:** Real-world use cases often present RAG systems with complex queries for which relevant information is missing from the corpus or is incomplete. In these settings, RAG systems must be able to reject unanswerable, out-of-scope queries and identify failures of retrieval and multi-hop reasoning. Despite this, existing RAG benchmarks rarely reflect realistic task complexity for multi-hop or out-of-scope questions, which often can be cheated via disconnected reasoning (i.e., solved without genuine multi-hop inference) or require only simple factual recall. This limits the ability for such benchmarks to uncover limitations of existing RAG systems. To address this gap, we present the first pipeline for automatic, difficulty-controlled creation of un$\underline{c}$heatable, $\underline{r}$ealistic, $\underline{u}$nanswerable, and $\underline{m}$ulti-hop $\underline{q}$uerie$\underline{s}$ (CRUMQs), adaptable to any corpus and domain. We use our pipeline to create CRUMQs over two popular RAG datasets and demonstrate its effectiveness via benchmark experiments on leading retrieval-augmented LLMs. Results show that compared to prior RAG benchmarks, CRUMQs are highly challenging for RAG systems and achieve up to 81.0\% reduction in cheatability scores. More broadly, our pipeline offers a simple way to enhance benchmark difficulty and realism and drive development of more capable RAG systems.
>
---
#### [replaced 128] ShiZhi: A Chinese Lightweight Large Language Model for Court View Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.09297v2](http://arxiv.org/pdf/2510.09297v2)**

> **作者:** Zhitian Hou; Kun Zeng
>
> **摘要:** Criminal Court View Generation (CVG) is a fundamental task in legal artificial intelligence, aiming to automatically generate the "Court View" section of a legal case document. Generating court views is challenging due to the diversity and complexity of case facts, and directly generating from raw facts may limit performance. In this paper, we present ShiZhi, the first large language model (LLM) specifically designed for court view generation. We construct a Chinese Court View Generation dataset, CCVG, of more than 110K cases, each containing fact descriptions paired with corresponding court views. Based on this dataset, ShiZhi achieving 70.00 ROUGE-1 and 67.85 BLEU-1 on court view generation, as well as 86.48\% accuracy with 92.75\% macro F1 on charge prediction. Experimental results demonstrate that even a small LLM can generate reasonable and legally coherent court views when trained on high-quality domain-specific data. Our model and dataset are available at \href{https://github.com/ZhitianHou/ShiZhi}{https://github.com/ZhitianHou/ShiZhi}.
>
---
#### [replaced 129] GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06073v2](http://arxiv.org/pdf/2503.06073v2)**

> **作者:** Xiang Lan; Feng Wu; Kai He; Qinghao Zhao; Shenda Hong; Mengling Feng
>
> **备注:** NeurIPS 2025 Camera-Ready
>
> **摘要:** While recent multimodal large language models (MLLMs) have advanced automated ECG interpretation, they still face two key limitations: (1) insufficient multimodal synergy between time series signals and visual ECG representations, and (2) limited explainability in linking diagnoses to granular waveform evidence. We introduce GEM, the first MLLM unifying ECG time series, 12-lead ECG images and text for grounded and clinician-aligned ECG interpretation. GEM enables feature-grounded analysis, evidence-driven reasoning, and a clinician-like diagnostic process through three core innovations: a dual-encoder framework extracting complementary time series and image features, cross-modal alignment for effective multimodal understanding, and knowledge-guided instruction generation for generating high-granularity grounding data (ECG-Grounding) linking diagnoses to measurable parameters ($e.g.$, QRS/PR Intervals). Additionally, we propose the Grounded ECG Understanding task, a clinically motivated benchmark designed to comprehensively assess the MLLM's capability in grounded ECG understanding. Experimental results on both existing and our proposed benchmarks show GEM significantly improves predictive performance (CSN $7.4\% \uparrow$), explainability ($22.7\% \uparrow$), and grounding ($24.8\% \uparrow$), making it more suitable for real-world clinical applications. GitHub repository: https://github.com/lanxiang1017/GEM.git
>
---
#### [replaced 130] Humanity's Last Code Exam: Can Advanced LLMs Conquer Human's Hardest Code Competition?
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.12713v2](http://arxiv.org/pdf/2506.12713v2)**

> **作者:** Xiangyang Li; Xiaopeng Li; Kuicai Dong; Quanhu Zhang; Rongju Ruan; Xinyi Dai; Xiaoshuang Liu; Shengchun Xu; Yasheng Wang; Ruiming Tang
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Code generation is a core capability of large language models (LLMs), yet mainstream benchmarks (e.g., APPs and LiveCodeBench) contain questions with medium-level difficulty and pose no challenge to advanced LLMs. To better reflected the advanced reasoning and code generation ability, We introduce Humanity's Last Code Exam (HLCE), comprising 235 most challenging problems from the International Collegiate Programming Contest (ICPC World Finals) and the International Olympiad in Informatics (IOI) spanning 2010 - 2024. As part of HLCE, we design a harmonized online-offline sandbox that guarantees fully reproducible evaluation. Through our comprehensive evaluation, we observe that even the strongest reasoning LLMs: o4-mini(high) and Gemini-2.5 Pro, achieve pass@1 rates of only 15.9% and 11.4%, respectively. Meanwhile, we propose a novel "self-recognition" task to measure LLMs' awareness of their own capabilities. Results indicate that LLMs' self-recognition abilities are not proportionally correlated with their code generation performance. Finally, our empirical validation of test-time scaling laws reveals that current advanced LLMs have substantial room for improvement on complex programming tasks. We expect HLCE to become a milestone challenge for code generation and to catalyze advances in high-performance reasoning and human-AI collaborative programming. Our code and dataset are also public available(https://github.com/Humanity-s-Last-Code-Exam/HLCE).
>
---
#### [replaced 131] FinResearchBench: A Logic Tree based Agent-as-a-Judge Evaluation Framework for Financial Research Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.16248v3](http://arxiv.org/pdf/2507.16248v3)**

> **作者:** Rui Sun; Zuo Bai; Wentao Zhang; Yuxiang Zhang; Li Zhao; Shan Sun; Zhengwen Qiu
>
> **摘要:** Recently, AI agents are rapidly evolving in intelligence and widely used in professional research applications, such as STEM, software development, and finance. Among these AI agents, deep research agent is a key category as it can perform long-horizon tasks and solve problems of greater complexity. However, there are few evaluation frameworks and benchmarks that systematically and automatically investigate the capabilities of these research agents. In addition, financial research problems have distinct complexity and subtlety. To fill in the gap, we propose FinResearchBench, which is a logic tree-based Agent-as-a-Judge and targets specifically for the financial research agents. It provides a comprehensive and automatic assessment of the research agents across 7 key types of tasks in the financial research domain. The contributions of this work are two-folded: (1) the first and innovative Agent-as-a-Judge system that extracts the logic tree of the research outcome and uses it as the intermediate information to present a comprehensive, reliable, and robust evaluation; (2) finance-oriented that it covers 70 typical financial research questions, spreading across 7 frequently encountered types of task in the domain.
>
---
#### [replaced 132] From Multimodal Perception to Strategic Reasoning: A Survey on AI-Generated Game Commentary
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.17294v2](http://arxiv.org/pdf/2506.17294v2)**

> **作者:** Qirui Zheng; Xingbo Wang; Keyuan Cheng; Muhammad Asif Ali; Yunlong Lu; Wenxin Li
>
> **摘要:** The advent of artificial intelligence has propelled AI-Generated Game Commentary (AI-GGC) into a rapidly expanding field, offering benefits such as unlimited availability and personalized narration. However, current researches in this area remain fragmented, and a comprehensive survey that systematically unifies existing efforts is still missing. To bridge this gap, our survey introduces a unified framework that systematically organizes the AI-GGC landscape. We present a novel taxonomy focused on three core commentator capabilities: Live Observation, Strategic Analysis, and Historical Recall. Commentary is further categorized into three functional types: Descriptive, Analytical, and Background. Building on this structure, we provide an in-depth review of state-of-the-art methods, datasets, and evaluation metrics across various game genres. Finally, we highlight key challenges such as real-time reasoning, multimodal integration, and evaluation bottlenecks, and outline promising directions for future research and system development in AI-GGC.
>
---
#### [replaced 133] A social context-aware graph-based multimodal attentive learning framework for disaster content classification during emergencies: a benchmark dataset and method
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.08814v2](http://arxiv.org/pdf/2410.08814v2)**

> **作者:** Shahid Shafi Dar; Mohammad Zia Ur Rehman; Karan Bais; Mohammed Abdul Haseeb; Nagendra Kumara
>
> **摘要:** In times of crisis, the prompt and precise classification of disaster-related information shared on social media platforms is crucial for effective disaster response and public safety. During such critical events, individuals use social media to communicate, sharing multimodal textual and visual content. However, due to the significant influx of unfiltered and diverse data, humanitarian organizations face challenges in leveraging this information efficiently. Existing methods for classifying disaster-related content often fail to model users' credibility, emotional context, and social interaction information, which are essential for accurate classification. To address this gap, we propose CrisisSpot, a method that utilizes a Graph-based Neural Network to capture complex relationships between textual and visual modalities, as well as Social Context Features to incorporate user-centric and content-centric information. We also introduce Inverted Dual Embedded Attention (IDEA), which captures both harmonious and contrasting patterns within the data to enhance multimodal interactions and provide richer insights. Additionally, we present TSEqD (Turkey-Syria Earthquake Dataset), a large annotated dataset for a single disaster event, containing 10,352 samples. Through extensive experiments, CrisisSpot demonstrated significant improvements, achieving an average F1-score gain of 9.45% and 5.01% compared to state-of-the-art methods on the publicly available CrisisMMD dataset and the TSEqD dataset, respectively.
>
---
