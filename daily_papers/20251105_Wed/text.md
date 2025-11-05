# 自然语言处理 cs.CL

- **最新发布 49 篇**

- **更新 61 篇**

## 最新发布

#### [new 001] Demo: Statistically Significant Results On Biases and Errors of LLMs Do Not Guarantee Generalizable Results
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文研究LLM在医疗场景中的偏见与错误，提出自动化查询生成与多LLM评估框架，发现评估者间一致性低，统计显著结果未必可泛化，呼吁使用多评估者并公开一致性指标。**

- **链接: [http://arxiv.org/pdf/2511.02246v1](http://arxiv.org/pdf/2511.02246v1)**

> **作者:** Jonathan Liu; Haoling Qiu; Jonathan Lasko; Damianos Karakos; Mahsa Yarmohammadi; Mark Dredze
>
> **摘要:** Recent research has shown that hallucinations, omissions, and biases are prevalent in everyday use-cases of LLMs. However, chatbots used in medical contexts must provide consistent advice in situations where non-medical factors are involved, such as when demographic information is present. In order to understand the conditions under which medical chatbots fail to perform as expected, we develop an infrastructure that 1) automatically generates queries to probe LLMs and 2) evaluates answers to these queries using multiple LLM-as-a-judge setups and prompts. For 1), our prompt creation pipeline samples the space of patient demographics, histories, disorders, and writing styles to create realistic questions that we subsequently use to prompt LLMs. In 2), our evaluation pipeline provides hallucination and omission detection using LLM-as-a-judge as well as agentic workflows, in addition to LLM-as-a-judge treatment category detectors. As a baseline study, we perform two case studies on inter-LLM agreement and the impact of varying the answering and evaluation LLMs. We find that LLM annotators exhibit low agreement scores (average Cohen's Kappa $\kappa=0.118$), and only specific (answering, evaluation) LLM pairs yield statistically significant differences across writing styles, genders, and races. We recommend that studies using LLM evaluation use multiple LLMs as evaluators in order to avoid arriving at statistically significant but non-generalizable results, particularly in the absence of ground-truth data. We also suggest publishing inter-LLM agreement metrics for transparency. Our code and dataset are available here: https://github.com/BBN-E/medic-neurips-2025-demo.
>
---
#### [new 002] Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Oolong基准，评估大模型在长上下文中的推理与聚合能力，解决现有评测过度依赖检索、忽略复杂分析的问题。构建合成与真实对话任务，要求模型进行细粒度分析与统计推理，前沿模型准确率均低于50%。**

- **链接: [http://arxiv.org/pdf/2511.02817v1](http://arxiv.org/pdf/2511.02817v1)**

> **作者:** Amanda Bertsch; Adithya Pratapa; Teruko Mitamura; Graham Neubig; Matthew R. Gormley
>
> **备注:** Preprint
>
> **摘要:** As model context lengths continue to grow, concerns about whether models effectively use the full context length have persisted. While several carefully designed long-context evaluations have recently been released, these evaluations tend to rely on retrieval from one or more sections of the context, which allows nearly all of the context tokens to be disregarded as noise. This represents only one type of task that might be performed with long context. We introduce Oolong, a benchmark of long-context reasoning tasks that require analyzing individual chunks of text on an atomic level, and then aggregating these analyses to answer distributional questions. Oolong is separated into two task sets: Oolong-synth, a set of naturalistic synthetic tasks, where we can easily ablate components of the reasoning problem; and Oolong-real, a downstream setting which requires reasoning over real-world conversational data. Oolong requires models to reason over large quantities of examples, to perform both classification and counting in-context, and to reason over temporal and user relations. Even frontier models struggle on Oolong, with GPT-5, Claude-Sonnet-4, and Gemini-2.5-Pro all achieving less than 50% accuracy on both splits at 128K. We release the data and evaluation harness for Oolong to enable further development of models that can reason over large quantities of text.
>
---
#### [new 003] Smart-Hiring: An Explainable end-to-end Pipeline for CV Information Extraction and Job Matching
- **分类: cs.CL**

- **简介: 论文提出Smart-Hiring，一个可解释的端到端NLP管道，用于从简历中提取结构化信息并匹配职位，解决人工筛简历耗时、偏见多的问题，通过语义向量匹配实现高精度与透明决策。**

- **链接: [http://arxiv.org/pdf/2511.02537v1](http://arxiv.org/pdf/2511.02537v1)**

> **作者:** Kenza Khelkhal; Dihia Lanasri
>
> **摘要:** Hiring processes often involve the manual screening of hundreds of resumes for each job, a task that is time and effort consuming, error-prone, and subject to human bias. This paper presents Smart-Hiring, an end-to-end Natural Language Processing (NLP) pipeline de- signed to automatically extract structured information from unstructured resumes and to semantically match candidates with job descriptions. The proposed system combines document parsing, named-entity recognition, and contextual text embedding techniques to capture skills, experience, and qualifications. Using advanced NLP technics, Smart-Hiring encodes both resumes and job descriptions in a shared vector space to compute similarity scores between candidates and job postings. The pipeline is modular and explainable, allowing users to inspect extracted entities and matching rationales. Experiments were conducted on a real-world dataset of resumes and job descriptions spanning multiple professional domains, demonstrating the robustness and feasibility of the proposed approach. The system achieves competitive matching accuracy while preserving a high degree of interpretability and transparency in its decision process. This work introduces a scalable and practical NLP frame- work for recruitment analytics and outlines promising directions for bias mitigation, fairness-aware modeling, and large-scale deployment of data-driven hiring solutions.
>
---
#### [new 004] MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: MemSearcher提出一种端到端强化学习框架，通过动态维护紧凑记忆，平衡搜索效率与信息完整性，解决LLM搜索代理在多轮交互中上下文冗长问题，显著提升性能并降低计算开销。**

- **链接: [http://arxiv.org/pdf/2511.02805v1](http://arxiv.org/pdf/2511.02805v1)**

> **作者:** Qianhao Yuan; Jie Lou; Zichao Li; Jiawei Chen; Yaojie Lu; Hongyu Lin; Le Sun; Debing Zhang; Xianpei Han
>
> **备注:** Project page: https://github.com/icip-cas/MemSearcher
>
> **摘要:** Typical search agents concatenate the entire interaction history into the LLM context, preserving information integrity but producing long, noisy contexts, resulting in high computation and memory costs. In contrast, using only the current turn avoids this overhead but discards essential information. This trade-off limits the scalability of search agents. To address this challenge, we propose MemSearcher, an agent workflow that iteratively maintains a compact memory and combines the current turn with it. At each turn, MemSearcher fuses the user's question with the memory to generate reasoning traces, perform search actions, and update memory to retain only information essential for solving the task. This design stabilizes context length across multi-turn interactions, improving efficiency without sacrificing accuracy. To optimize this workflow, we introduce multi-context GRPO, an end-to-end RL framework that jointly optimize reasoning, search strategies, and memory management of MemSearcher Agents. Specifically, multi-context GRPO samples groups of trajectories under different contexts and propagates trajectory-level advantages across all conversations within them. Trained on the same dataset as Search-R1, MemSearcher achieves significant improvements over strong baselines on seven public benchmarks: +11% on Qwen2.5-3B-Instruct and +12% on Qwen2.5-7B-Instruct relative average gains. Notably, the 3B-based MemSearcher even outperforms 7B-based baselines, demonstrating that striking a balance between information integrity and efficiency yields both higher accuracy and lower computational overhead. The code and models will be publicly available at https://github.com/icip-cas/MemSearcher
>
---
#### [new 005] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis, Solution, and Interpretation
- **分类: cs.CL**

- **简介: 该论文研究LLM微调中新知识引入引发的事实幻觉问题，提出KnownPatch方法，通过少量已知知识修补缓解幻觉，并揭示新知识削弱模型对关键实体的注意力机制，导致幻觉传播。**

- **链接: [http://arxiv.org/pdf/2511.02626v1](http://arxiv.org/pdf/2511.02626v1)**

> **作者:** Renfei Dang; Peng Hu; Changjiang Gao; Shujian Huang
>
> **摘要:** Previous studies show that introducing new knowledge during large language models (LLMs) fine-tuning can lead to the generation of erroneous output when tested on known information, thereby triggering factual hallucinations. However, existing studies have not deeply investigated the specific manifestations and underlying mechanisms of these hallucinations. Our work addresses this gap by designing a controlled dataset Biography-Reasoning, and conducting a fine-grained analysis across multiple knowledge types and two task types, including knowledge question answering (QA) and knowledge reasoning tasks. We find that when fine-tuned on a dataset in which a specific knowledge type consists entirely of new knowledge, LLMs exhibit significantly increased hallucination tendencies. This suggests that the high unfamiliarity of a particular knowledge type, rather than the overall proportion of new knowledge, is a stronger driver of hallucinations, and these tendencies can even affect other knowledge types in QA tasks. To mitigate such factual hallucinations, we propose KnownPatch, which patches a small number of known knowledge samples in the later stages of training, effectively alleviating new-knowledge-induced hallucinations. Through attention analysis, we find that learning new knowledge reduces the model's attention to key entities in the question, thus causing excessive focus on the surrounding context, which may increase the risk of hallucination. Moreover, the attention pattern can propagate to similar contexts, facilitating the spread of hallucinations to textually similar questions. Our method effectively mitigates the disruption of new knowledge learning to the model's attention on key entities, accompanied by improved performance.
>
---
#### [new 006] IG-Pruning: Input-Guided Block Pruning for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出IG-Pruning，面向大语言模型高效推理，解决静态剪枝适应性差的问题。通过输入引导的动态层选择，在无需重训练下实现自适应深度剪枝，显著提升性能与资源效率。**

- **链接: [http://arxiv.org/pdf/2511.02213v1](http://arxiv.org/pdf/2511.02213v1)**

> **作者:** Kangyu Qiao; Shaolei Zhang; Yang Feng
>
> **备注:** Accepted to EMNLP 2025. Code is available at https://github.com/ictnlp/IG-Pruning
>
> **摘要:** With the growing computational demands of large language models (LLMs), efficient inference has become increasingly critical for practical deployment. Depth pruning has emerged as a promising approach for reducing the computational costs of large language models by removing transformer layers. However, existing methods typically rely on fixed block masks, which can lead to suboptimal performance across different tasks and inputs. In this paper, we propose IG-Pruning, a novel input-aware block-wise pruning method that dynamically selects layer masks at inference time. Our approach consists of two stages: (1) Discovering diverse mask candidates through semantic clustering and L0 optimization, and (2) Implementing efficient dynamic pruning without the need for extensive training. Experimental results demonstrate that our method consistently outperforms state-of-the-art static depth pruning methods, making it particularly suitable for resource-constrained deployment scenarios.
>
---
#### [new 007] AI Diffusion in Low Resource Language Countries
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究AI在低资源语言国家的扩散障碍，发现语言数据稀缺导致AI效用下降，使用户占比低20%。通过加权回归隔离语言因素，证实语言可及性是独立于 socioeconomics 的关键障碍，属于AI公平性与语言偏见研究任务。**

- **链接: [http://arxiv.org/pdf/2511.02752v1](http://arxiv.org/pdf/2511.02752v1)**

> **作者:** Amit Misra; Syed Waqas Zamir; Wassim Hamidouche; Inbal Becker-Reshef; Juan Lavista Ferres
>
> **备注:** 9 pages, 4 tables. Also available at https://aka.ms/AI_Diffusion_Low_Resource_Language_Countries
>
> **摘要:** Artificial intelligence (AI) is diffusing globally at unprecedented speed, but adoption remains uneven. Frontier Large Language Models (LLMs) are known to perform poorly on low-resource languages due to data scarcity. We hypothesize that this performance deficit reduces the utility of AI, thereby slowing adoption in Low-Resource Language Countries (LRLCs). To test this, we use a weighted regression model to isolate the language effect from socioeconomic and demographic factors, finding that LRLCs have a share of AI users that is approximately 20% lower relative to their baseline. These results indicate that linguistic accessibility is a significant, independent barrier to equitable AI diffusion.
>
---
#### [new 008] The Realignment Problem: When Right becomes Wrong in LLMs
- **分类: cs.CL**

- **简介: 论文提出TRACE框架，解决LLM对齐滞后于社会规范的“对齐-现实鸿沟”问题，通过智能筛选与优化偏好数据，实现高效、精准的动态对齐更新，兼顾安全与性能。**

- **链接: [http://arxiv.org/pdf/2511.02623v1](http://arxiv.org/pdf/2511.02623v1)**

> **作者:** Aakash Sen Sharma; Debdeep Sanyal; Vivek Srivastava; Shirish Karande; Murari Mandal
>
> **备注:** 23 Pages
>
> **摘要:** The alignment of Large Language Models (LLMs) with human values is central to their safe deployment, yet current practice produces static, brittle, and costly-to-maintain models that fail to keep pace with evolving norms and policies. This misalignment, which we term the Alignment-Reality Gap, poses a growing challenge for reliable long-term use. Existing remedies are inadequate: large-scale re-annotation is economically prohibitive, and standard unlearning methods act as blunt instruments that erode utility rather than enable precise policy updates. We introduce TRACE (Triage and Re-align by Alignment Conflict Evaluation), a framework for principled unlearning that reconceives re-alignment as a programmatic policy application problem. TRACE programmatically triages existing preference data against a new policy, identifies high-impact conflicts via a alignment impact score, and applies a hybrid optimization that cleanly inverts, discards, or preserves preferences while safeguarding model performance. Empirical results show that TRACE achieves robust re-alignment across diverse model families (Qwen2.5-7B, Gemma-2-9B, Llama-3.1-8B). On both synthetic benchmarks and the PKU-SafeRLHF dataset under complex policy shift, TRACE enforces new principles without degrading general capabilities. Our work establishes a scalable, dynamic, and cost-effective paradigm for maintaining LLM alignment, providing a foundation for sustainable and responsible AI deployment.
>
---
#### [new 009] Next Token Knowledge Tracing: Exploiting Pretrained LLM Representations to Decode Student Behaviour
- **分类: cs.CL; cs.AI**

- **简介: 该论文将知识追踪（KT）重构为基于预训练LLM的下一个词预测任务，利用题干文本与学生作答历史，提升对学习行为的建模能力，显著改善冷启动场景下的预测性能。**

- **链接: [http://arxiv.org/pdf/2511.02599v1](http://arxiv.org/pdf/2511.02599v1)**

> **作者:** Max Norris; Kobi Gal; Sahan Bulathwela
>
> **摘要:** Modelling student knowledge is a key challenge when leveraging AI in education, with major implications for personalised learning. The Knowledge Tracing (KT) task aims to predict how students will respond to educational questions in learning environments, based on their prior interactions. Existing KT models typically use response correctness along with metadata like skill tags and timestamps, often overlooking the question text, which is an important source of pedagogical insight. This omission poses a lost opportunity while limiting predictive performance. We propose Next Token Knowledge Tracing (NTKT), a novel approach that reframes KT as a next-token prediction task using pretrained Large Language Models (LLMs). NTKT represents both student histories and question content as sequences of text, allowing LLMs to learn patterns in both behaviour and language. Our series of experiments significantly improves performance over state-of-the-art neural KT models and generalises much better to cold-start questions and users. These findings highlight the importance of question content in KT and demonstrate the benefits of leveraging pretrained representations of LLMs to model student learning more effectively.
>
---
#### [new 010] Multi-Personality Generation of LLMs at Decoding-time
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种解码时多人格生成框架MPG，无需重训练，利用单维模型的隐式密度比，通过SCR采样高效融合多重人格属性，提升角色扮演与MBTI人格生成质量16%-18%。**

- **链接: [http://arxiv.org/pdf/2511.01891v1](http://arxiv.org/pdf/2511.01891v1)**

> **作者:** Rongxin Chen; Yunfan Li; Yige Yuan; Bingbing Xu; Huawei Shen
>
> **备注:** WSDM2026
>
> **摘要:** Multi-personality generation for LLMs, enabling simultaneous embodiment of multiple personalization attributes, is a fundamental challenge. Existing retraining-based approaches are costly and poorly scalable, while decoding-time methods often rely on external models or heuristics, limiting flexibility and robustness. In this paper, we propose a novel Multi-Personality Generation (MPG) framework under the decoding-time combination paradigm. It flexibly controls multi-personality without relying on scarce multi-dimensional models or extra training, leveraging implicit density ratios in single-dimensional models as a "free lunch" to reformulate the task as sampling from a target strategy aggregating these ratios. To implement MPG efficiently, we design Speculative Chunk-level based Rejection sampling (SCR), which generates responses in chunks and parallelly validates them via estimated thresholds within a sliding window. This significantly reduces computational overhead while maintaining high-quality generation. Experiments on MBTI personality and Role-Playing demonstrate the effectiveness of MPG, showing improvements up to 16%-18%. Code and data are available at https://github.com/Libra117/MPG .
>
---
#### [new 011] PragExTra: A Multilingual Corpus of Pragmatic Explicitation in Translation
- **分类: cs.CL**

- **简介: 论文提出PragExTra，首个多语言语用显化语料库与检测框架，解决翻译中文化隐含信息显化难以计算建模的问题，通过零对齐与主动学习识别实体、度量等显化现象，提升分类准确率至0.88。**

- **链接: [http://arxiv.org/pdf/2511.02721v1](http://arxiv.org/pdf/2511.02721v1)**

> **作者:** Doreen Osmelak; Koel Dutta Chowdhury; Uliana Sentsova; Cristina España-Bonet; Josef van Genabith
>
> **摘要:** Translators often enrich texts with background details that make implicit cultural meanings explicit for new audiences. This phenomenon, known as pragmatic explicitation, has been widely discussed in translation theory but rarely modeled computationally. We introduce PragExTra, the first multilingual corpus and detection framework for pragmatic explicitation. The corpus covers eight language pairs from TED-Multi and Europarl and includes additions such as entity descriptions, measurement conversions, and translator remarks. We identify candidate explicitation cases through null alignments and refined using active learning with human annotation. Our results show that entity and system-level explicitations are most frequent, and that active learning improves classifier accuracy by 7-8 percentage points, achieving up to 0.88 accuracy and 0.82 F1 across languages. PragExTra establishes pragmatic explicitation as a measurable, cross-linguistic phenomenon and takes a step towards building culturally aware machine translation. Keywords: translation, multilingualism, explicitation
>
---
#### [new 012] AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文提出AutoAdv，一种无训练的多轮对抗提示框架，旨在突破大模型的安全防护。通过自适应模式管理、温度调控和两阶段重写策略，显著提升多轮攻击成功率，揭示单轮对齐机制在多轮场景下的脆弱性。**

- **链接: [http://arxiv.org/pdf/2511.02376v1](http://arxiv.org/pdf/2511.02376v1)**

> **作者:** Aashray Reddy; Andrew Zagula; Nicholas Saban
>
> **摘要:** Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs, yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves up to 95% attack success rate on Llama-3.1-8B within six turns a 24 percent improvement over single turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests then iteratively refines them. Extensive evaluation across commercial and open-source models (GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.
>
---
#### [new 013] Prompting for Policy: Forecasting Macroeconomic Scenarios with Synthetic LLM Personas
- **分类: cs.CL; cs.CE; econ.GN; q-fin.EC**

- **简介: 该论文研究使用LLM（GPT-4o）进行宏观预测，探索 persona 提示是否提升准确性。结果表明：LLM预测精度接近人类专家，但 persona 提示无显著增益，可省略以降本。**

- **链接: [http://arxiv.org/pdf/2511.02458v1](http://arxiv.org/pdf/2511.02458v1)**

> **作者:** Giulia Iadisernia; Carolina Camassa
>
> **备注:** 9 pages, 8-pages appendix, accepted at ICAIF 25
>
> **摘要:** We evaluate whether persona-based prompting improves Large Language Model (LLM) performance on macroeconomic forecasting tasks. Using 2,368 economics-related personas from the PersonaHub corpus, we prompt GPT-4o to replicate the ECB Survey of Professional Forecasters across 50 quarterly rounds (2013-2025). We compare the persona-prompted forecasts against the human experts panel, across four target variables (HICP, core HICP, GDP growth, unemployment) and four forecast horizons. We also compare the results against 100 baseline forecasts without persona descriptions to isolate its effect. We report two main findings. Firstly, GPT-4o and human forecasters achieve remarkably similar accuracy levels, with differences that are statistically significant yet practically modest. Our out-of-sample evaluation on 2024-2025 data demonstrates that GPT-4o can maintain competitive forecasting performance on unseen events, though with notable differences compared to the in-sample period. Secondly, our ablation experiment reveals no measurable forecasting advantage from persona descriptions, suggesting these prompt components can be omitted to reduce computational costs without sacrificing accuracy. Our results provide evidence that GPT-4o can achieve competitive forecasting accuracy even on out-of-sample macroeconomic events, if provided with relevant context data, while revealing that diverse prompts produce remarkably homogeneous forecasts compared to human panels.
>
---
#### [new 014] AyurParam: A State-of-the-Art Bilingual Language Model for Ayurveda
- **分类: cs.CL; cs.AI**

- **简介: 论文提出AyurParam-2.9B，一种面向阿育吠陀医学的双语语言模型，解决主流LLM在专业传统医学领域表现不佳的问题，通过高质量英印双语数据微调，实现精准理解与推理，性能超越同规模及更大模型。**

- **链接: [http://arxiv.org/pdf/2511.02374v1](http://arxiv.org/pdf/2511.02374v1)**

> **作者:** Mohd Nauman; Sravan Gvm; Vijay Devane; Shyam Pawar; Viraj Thakur; Kundeshwar Pundalik; Piyush Sawarkar; Rohit Saluja; Maunendra Desarkar; Ganesh Ramakrishnan
>
> **摘要:** Current large language models excel at broad, general-purpose tasks, but consistently underperform when exposed to highly specialized domains that require deep cultural, linguistic, and subject-matter expertise. In particular, traditional medical systems such as Ayurveda embody centuries of nuanced textual and clinical knowledge that mainstream LLMs fail to accurately interpret or apply. We introduce AyurParam-2.9B, a domain-specialized, bilingual language model fine-tuned from Param-1-2.9B using an extensive, expertly curated Ayurveda dataset spanning classical texts and clinical guidance. AyurParam's dataset incorporates context-aware, reasoning, and objective-style Q&A in both English and Hindi, with rigorous annotation protocols for factual precision and instructional clarity. Benchmarked on BhashaBench-Ayur, AyurParam not only surpasses all open-source instruction-tuned models in its size class (1.5--3B parameters), but also demonstrates competitive or superior performance compared to much larger models. The results from AyurParam highlight the necessity for authentic domain adaptation and high-quality supervision in delivering reliable, culturally congruent AI for specialized medical knowledge.
>
---
#### [new 015] Controlling Performance and Budget of a Centralized Multi-agent LLM System with Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出CoRL框架，通过强化学习在集中式多LLM系统中协调专家模型，实现性能与推理成本的动态权衡，支持多预算场景下的自适应部署，解决分散式系统成本不可控问题。**

- **链接: [http://arxiv.org/pdf/2511.02755v1](http://arxiv.org/pdf/2511.02755v1)**

> **作者:** Bowen Jin; TJ Collins; Donghan Yu; Mert Cemri; Shenao Zhang; Mengyu Li; Jay Tang; Tian Qin; Zhiyang Xu; Jiarui Lu; Guoli Yin; Jiawei Han; Zirui Wang
>
> **备注:** 14 pages
>
> **摘要:** Large language models (LLMs) exhibit complementary strengths across domains and come with varying inference costs, motivating the design of multi-agent LLM systems where specialized models collaborate efficiently. Existing approaches predominantly rely on decentralized frameworks, which invoke multiple LLMs for every input and thus lead to substantial and uncontrolled inference costs. In this work, we introduce a centralized multi-LLM framework, where a controller LLM selectively coordinates a pool of expert models in a cost-efficient and cost-controllable manner. We formulate this coordination problem as reinforcement learning with dual objectives: maximizing task performance while minimizing the overall inference cost. In addition, we expect the multi-agent system to have adapted behavior with different budget conditions during inference. To this end, we propose CoRL, a reinforcement learning framework that optimizes the performance cost trade-off in a controllable multi-budget setting. Experiments on four diverse benchmarks demonstrate that CoRL enables a single system to surpass the best expert LLM under high-budget settings, while maintaining strong performance in more economical low-budget modes, highlighting the effectiveness of centralized coordination for scalable and cost-efficient multi-agent LLM systems.
>
---
#### [new 016] Merging Continual Pretraining Models for Domain-Specialized LLMs: A Case Study in Finance
- **分类: cs.CL**

- **简介: 该论文研究持续预训练（CPT）模型合并，解决专用领域LLMs多技能融合难题，提出三阶段评估框架，验证Task Arithmetic、TIES等方法在金融领域合并专家模型的有效性，首次系统揭示CPT合并机制与涌现技能规律。**

- **链接: [http://arxiv.org/pdf/2511.02451v1](http://arxiv.org/pdf/2511.02451v1)**

> **作者:** Kentaro Ueda; François Portet; Hirohiko Suwa; Keiichi Yasumoto
>
> **摘要:** While LLMs excel at general tasks, they struggle in specialized domains like finance, requiring diverse skills in domain knowledge, mathematical reasoning, and multilingual processing. Merging domain-specific Continual Pre-training (CPT) "experts" offers a practical alternative to costly and unstable multi-skill training. However, unlike established Supervised Fine-Tuning (SFT) model-based merging, CPT model merging remains largely unexplored. We address this gap by creating financial LLMs from experts in finance, math, and Japanese. We propose a three-stage evaluation focusing on knowledge recovery, complementarity, and emergence, and assess three merging methods (Task Arithmetic, TIES, and DARE-TIES) on a comprehensive financial benchmark curated from 18 tasks across 8 established datasets. Results show that merging an expert with its base model recovers general knowledge lost during CPT, while merging experts improves performance and can yield emergent cross-domain skills. Among the methods, Task Arithmetic performs strongly but is hyperparameter-sensitive, whereas TIES is more robust. Our findings also suggest that while model similarity correlates with merging success, emergent skills depend on more complex factors. This work presents the first foundational analysis of CPT model merging, establishing a principled framework and providing clear guidance for building multi-skill LLMs from existing assets.
>
---
#### [new 017] Rethinking LLM Human Simulation: When a Graph is What You Need
- **分类: cs.CL**

- **简介: 该论文提出GEMS，将人类离散选择模拟转化为图链路预测问题，用小型图神经网络替代大语言模型，在保持或超越LLM性能的同时，显著提升效率与可解释性。**

- **链接: [http://arxiv.org/pdf/2511.02135v1](http://arxiv.org/pdf/2511.02135v1)**

> **作者:** Joseph Suh; Suhong Moon; Serina Chang
>
> **备注:** Code: https://github.com/schang-lab/gems
>
> **摘要:** Large language models (LLMs) are increasingly used to simulate humans, with applications ranging from survey prediction to decision-making. However, are LLMs strictly necessary, or can smaller, domain-grounded models suffice? We identify a large class of simulation problems in which individuals make choices among discrete options, where a graph neural network (GNN) can match or surpass strong LLM baselines despite being three orders of magnitude smaller. We introduce Graph-basEd Models for human Simulation (GEMS), which casts discrete choice simulation tasks as a link prediction problem on graphs, leveraging relational knowledge while incorporating language representations only when needed. Evaluations across three key settings on three simulation datasets show that GEMS achieves comparable or better accuracy than LLMs, with far greater efficiency, interpretability, and transparency, highlighting the promise of graph-based modeling as a lightweight alternative to LLMs for human simulation. Our code is available at https://github.com/schang-lab/gems.
>
---
#### [new 018] Beyond Single Embeddings: Capturing Diverse Targets with Multi-Query Retrieval
- **分类: cs.CL; cs.IR**

- **简介: 该论文面向多答案检索任务，针对单向量检索器无法捕捉查询多模态语义的问题，提出AMER模型，通过自回归生成多个查询向量，显著提升对语义分散文档的检索性能。**

- **链接: [http://arxiv.org/pdf/2511.02770v1](http://arxiv.org/pdf/2511.02770v1)**

> **作者:** Hung-Ting Chen; Xiang Liu; Shauli Ravfogel; Eunsol Choi
>
> **摘要:** Most text retrievers generate \emph{one} query vector to retrieve relevant documents. Yet, the conditional distribution of relevant documents for the query may be multimodal, e.g., representing different interpretations of the query. We first quantify the limitations of existing retrievers. All retrievers we evaluate struggle more as the distance between target document embeddings grows. To address this limitation, we develop a new retriever architecture, \emph{A}utoregressive \emph{M}ulti-\emph{E}mbedding \emph{R}etriever (AMER). Our model autoregressively generates multiple query vectors, and all the predicted query vectors are used to retrieve documents from the corpus. We show that on the synthetic vectorized data, the proposed method could capture multiple target distributions perfectly, showing 4x better performance than single embedding model. We also fine-tune our model on real-world multi-answer retrieval datasets and evaluate in-domain. AMER presents 4 and 21\% relative gains over single-embedding baselines on two datasets we evaluate on. Furthermore, we consistently observe larger gains on the subset of dataset where the embeddings of the target documents are less similar to each other. We demonstrate the potential of using a multi-query vector retriever and open up a new direction for future work.
>
---
#### [new 019] The Analysis of Lexical Errors in Machine Translation from English into Romanian
- **分类: cs.CL**

- **简介: 该论文属于机器翻译错误分析任务，聚焦于Google Translate英译罗语中的词汇错误，针对WHO、Gavi等机构的新冠相关文本，分析230篇译文，旨在提升词汇选择准确性，降低翻译错误，优化系统性能。**

- **链接: [http://arxiv.org/pdf/2511.02587v1](http://arxiv.org/pdf/2511.02587v1)**

> **作者:** Angela Stamatie
>
> **备注:** Doctoral thesis
>
> **摘要:** The research explores error analysis in the performance of translating by Machine Translation from English into Romanian, and it focuses on lexical errors found in texts which include official information, provided by the World Health Organization (WHO), the Gavi Organization, by the patient information leaflet (the information about the active ingredients of the vaccines or the medication, the indications, the dosage instructions, the storage instructions, the side effects and warning, etc.). All of these texts are related to Covid-19 and have been translated by Google Translate, a multilingual Machine Translation that was created by Google. In the last decades, Google has actively worked to develop a more accurate and fluent automatic translation system. This research, specifically focused on improving Google Translate, aims to enhance the overall quality of Machine Translation by achieving better lexical selection and by reducing errors. The investigation involves a comprehensive analysis of 230 texts that have been translated from English into Romanian.
>
---
#### [new 020] Optimal Singular Damage: Efficient LLM Inference in Low Storage Regimes
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦于低存储环境下大语言模型微调参数的高效存储问题，提出“最优奇异损伤”方法，通过联合利用低秩与稀疏性，选择性保留关键奇异分量，在相同内存下显著提升存储效率与模型精度。**

- **链接: [http://arxiv.org/pdf/2511.02681v1](http://arxiv.org/pdf/2511.02681v1)**

> **作者:** Mohammadsajad Alipour; Mohammad Mohammadi Amiri
>
> **摘要:** Large language models (LLMs) are increasingly prevalent across diverse applications. However, their enormous size limits storage and processing capabilities to a few well-resourced stakeholders. As a result, most applications rely on pre-trained LLMs, fine-tuned for specific tasks. However, even storing the fine-tuned versions of these models remains a significant challenge due to the wide range of tasks they address. Recently, studies show that fine-tuning these models primarily affects a small fraction of parameters, highlighting the need for more efficient storage of fine-tuned models. This paper focuses on efficient storage of parameter updates in pre-trained models after fine-tuning. To address this challenge, we leverage the observation that fine-tuning updates are both low-rank and sparse, which can be utilized for storage efficiency. However, using only low-rank approximation or sparsification may discard critical singular components that enhance model expressivity. We first observe that given the same memory budget, sparsified low-rank approximations with larger ranks outperform standard low-rank approximations with smaller ranks. Building on this, we propose our method, optimal singular damage, that selectively sparsifies low-rank approximated updates by leveraging the interleaved importance of singular vectors, ensuring that the most impactful components are retained. We demonstrate through extensive experiments that our proposed methods lead to significant storage efficiency and superior accuracy within the same memory budget compared to employing the low-rank approximation or sparsification individually.
>
---
#### [new 021] CGES: Confidence-Guided Early Stopping for Efficient and Accurate Self-Consistency
- **分类: cs.CL**

- **简介: 该论文提出CGES，一种基于置信度的自适应早停方法，用于提升大语言模型自一致性推理的效率。通过贝叶斯框架利用置信信号动态停止采样，在大幅减少调用次数（-69%）的同时保持准确率。**

- **链接: [http://arxiv.org/pdf/2511.02603v1](http://arxiv.org/pdf/2511.02603v1)**

> **作者:** Ehsan Aghazadeh; Ahmad Ghasemi; Hedyeh Beyhaghi; Hossein Pishro-Nik
>
> **备注:** Efficient Reasoning @ NeurIPS2025
>
> **摘要:** Large language models (LLMs) are often queried multiple times at test time, with predictions aggregated by majority vote. While effective, this self-consistency strategy (arXiv:2203.11171) requires a fixed number of calls and can fail when the correct answer is rare. We introduce Confidence-Guided Early Stopping (CGES), a Bayesian framework that forms posteriors over candidate answers using scalar confidence signals derived from token probabilities or reward models. CGES adaptively halts sampling once the posterior mass of a candidate exceeds a threshold. We provide theoretical guarantees for both perfectly calibrated confidences and realistic noisy confidence signals. Across five reasoning benchmarks, CGES reduces the average number of model calls by about 69 percent (for example, from 16.0 to 4.9) while matching the accuracy of self-consistency within 0.06 percentage points.
>
---
#### [new 022] Let Multimodal Embedders Learn When to Augment Query via Adaptive Query Augmentation
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; cs.MM**

- **简介: 该论文提出M-Solomon，一种自适应多模态嵌入器，解决传统方法对所有查询盲目增强导致的延迟与性能下降问题。通过学习判断何时启用增强，仅对必要查询添加语义增强，显著提升效率与效果。**

- **链接: [http://arxiv.org/pdf/2511.02358v1](http://arxiv.org/pdf/2511.02358v1)**

> **作者:** Wongyu Kim; Hochang Lee; Sanghak Lee; Yoonsung Kim; Jaehyun Park
>
> **备注:** Accepted to MMGenSR Workshop (CIKM 2025)
>
> **摘要:** Query augmentation makes queries more meaningful by appending further information to the queries to find relevant documents. Current studies have proposed Large Language Model (LLM)-based embedders, which learn representation for embedding and generation for query augmentation in a multi-task manner by leveraging the generative capabilities of LLM. During inference, these jointly trained embedders have conducted query augmentation followed by embedding, showing effective results. However, augmenting every query leads to substantial embedding latency and query augmentation can be detrimental to performance for some queries. Also, previous methods have not been explored in multimodal environments. To tackle these problems, we propose M-Solomon, a universal multimodal embedder that can adaptively determine when to augment queries. Our approach first divides the queries of the training datasets into two groups at the dataset level. One includes queries that require augmentation and the other includes queries that do not. Then, we introduces a synthesis process that generates appropriate augmentations for queries that require them by leveraging a powerful Multimodal LLM (MLLM). Next, we present adaptive query augmentation. Through this step, M-Solomon can conduct query augmentation only when necessary by learning to generate synthetic augmentations with the prefix /augment for queries that demand them and to generate the simple string /embed for others. Experimental results showed that M-Solomon not only surpassed the baseline without augmentation by a large margin but also outperformed the baseline that always used augmentation, providing much faster embedding latency.
>
---
#### [new 023] LiveSecBench: A Dynamic and Culturally-Relevant AI Safety Benchmark for LLMs in Chinese Context
- **分类: cs.CL**

- **简介: 论文提出LiveSecBench，首个面向中文场景的动态AI安全基准，解决现有评估体系文化适配不足的问题，构建涵盖法律、伦理等六维度的评测体系，持续更新并评估18个中文LLM。**

- **链接: [http://arxiv.org/pdf/2511.02366v1](http://arxiv.org/pdf/2511.02366v1)**

> **作者:** Yudong Li; Zhongliang Yang; Kejiang Chen; Wenxuan Wang; Tianxin Zhang; Sifang Wan; Kecheng Wang; Haitian Li; Xu Wang; Lefan Cheng; Youdan Yang; Baocheng Chen; Ziyu Liu; Yufei Sun; Liyan Wu; Wenya Wen; Xingchi Gu; Peiru Yang
>
> **摘要:** In this work, we propose LiveSecBench, a dynamic and continuously updated safety benchmark specifically for Chinese-language LLM application scenarios. LiveSecBench evaluates models across six critical dimensions (Legality, Ethics, Factuality, Privacy, Adversarial Robustness, and Reasoning Safety) rooted in the Chinese legal and social frameworks. This benchmark maintains relevance through a dynamic update schedule that incorporates new threat vectors, such as the planned inclusion of Text-to-Image Generation Safety and Agentic Safety in the next update. For now, LiveSecBench (v251030) has evaluated 18 LLMs, providing a landscape of AI safety in the context of Chinese language. The leaderboard is publicly accessible at https://livesecbench.intokentech.cn/.
>
---
#### [new 024] LTD-Bench: Evaluating Large Language Models by Letting Them Draw
- **分类: cs.CL**

- **简介: 论文提出LTD-Bench，通过让大语言模型绘图（点阵/代码）评估其空间推理能力，解决传统数值指标掩盖空间理解缺陷的问题，首次实现语言-空间双向映射的可视化诊断。**

- **链接: [http://arxiv.org/pdf/2511.02347v1](http://arxiv.org/pdf/2511.02347v1)**

> **作者:** Liuhao Lin; Ke Li; Zihan Xu; Yuchen Shi; Yulei Qin; Yan Zhang; Xing Sun; Rongrong Ji
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Current evaluation paradigms for large language models (LLMs) represent a critical blind spot in AI research--relying on opaque numerical metrics that conceal fundamental limitations in spatial reasoning while providing no intuitive understanding of model capabilities. This deficiency creates a dangerous disconnect between reported performance and practical abilities, particularly for applications requiring physical world understanding. We introduce LTD-Bench, a breakthrough benchmark that transforms LLM evaluation from abstract scores to directly observable visual outputs by requiring models to generate drawings through dot matrices or executable code. This approach makes spatial reasoning limitations immediately apparent even to non-experts, bridging the fundamental gap between statistical performance and intuitive assessment. LTD-Bench implements a comprehensive methodology with complementary generation tasks (testing spatial imagination) and recognition tasks (assessing spatial perception) across three progressively challenging difficulty levels, methodically evaluating both directions of the critical language-spatial mapping. Our extensive experiments with state-of-the-art models expose an alarming capability gap: even LLMs achieving impressive results on traditional benchmarks demonstrate profound deficiencies in establishing bidirectional mappings between language and spatial concept--a fundamental limitation that undermines their potential as genuine world models. Furthermore, LTD-Bench's visual outputs enable powerful diagnostic analysis, offering a potential approach to investigate model similarity.
>
---
#### [new 025] Link prediction Graph Neural Networks for structure recognition of Handwritten Mathematical Expressions
- **分类: cs.CV; cs.CL**

- **简介: 该论文面向手写数学表达式结构识别任务，将表达式建模为图，利用BLSTM生成初始图，再通过GNN进行链接预测，剔除冗余边，优化符号关系结构，提升识别准确率。**

- **链接: [http://arxiv.org/pdf/2511.02288v1](http://arxiv.org/pdf/2511.02288v1)**

> **作者:** Cuong Tuan Nguyen; Ngoc Tuan Nguyen; Triet Hoang Minh Dao; Huy Minh Nhat; Huy Truong Dinh
>
> **备注:** accepted for ICDAR2025-WML
>
> **摘要:** We propose a Graph Neural Network (GNN)-based approach for Handwritten Mathematical Expression (HME) recognition by modeling HMEs as graphs, where nodes represent symbols and edges capture spatial dependencies. A deep BLSTM network is used for symbol segmentation, recognition, and spatial relation classification, forming an initial primitive graph. A 2D-CFG parser then generates all possible spatial relations, while the GNN-based link prediction model refines the structure by removing unnecessary connections, ultimately forming the Symbol Label Graph. Experimental results demonstrate the effectiveness of our approach, showing promising performance in HME structure recognition.
>
---
#### [new 026] SAIL-RL: Guiding MLLMs in When and How to Think via Dual-Reward RL Tuning
- **分类: cs.CV; cs.CL**

- **简介: SAIL-RL提出一种双奖励强化学习框架，解决MLLMs推理时“何时思考、如何思考”的问题，通过思考奖励与判断奖励提升推理质量与自适应性，显著降低幻觉，提升多模态理解能力。**

- **链接: [http://arxiv.org/pdf/2511.02280v1](http://arxiv.org/pdf/2511.02280v1)**

> **作者:** Fangxun Shu; Yongjie Ye; Yue Liao; Zijian Kang; Weijie Yin; Jiacong Wang; Xiao Liang; Shuicheng Yan; Chao Feng
>
> **摘要:** We introduce SAIL-RL, a reinforcement learning (RL) post-training framework that enhances the reasoning capabilities of multimodal large language models (MLLMs) by teaching them when and how to think. Existing approaches are limited by outcome-only supervision, which rewards correct answers without ensuring sound reasoning, and by uniform thinking strategies, which often lead to overthinking on simple tasks and underthinking on complex ones. SAIL-RL addresses these challenges with a dual reward system: the Thinking Reward, which evaluates reasoning quality through factual grounding, logical coherence, and answer consistency, and the Judging Reward, which adaptively determines whether deep reasoning or direct answering is appropriate. Experiments on the state-of-the-art SAIL-VL2 show that SAIL-RL improves reasoning and multimodal understanding benchmarks at both 4B and 8B scales, achieving competitive performance against commercial closed-source models such as GPT-4o, and substantially reduces hallucinations, establishing it as a principled framework for building more reliable and adaptive MLLMs. The code will be available at https://github.com/BytedanceDouyinContent/SAIL-RL.
>
---
#### [new 027] CostBench: Evaluating Multi-Turn Cost-Optimal Planning and Adaptation in Dynamic Environments for LLM Tool-Use Agents
- **分类: cs.AI; cs.CL**

- **简介: 论文提出CostBench基准，评估LLM工具代理在动态环境中进行成本最优规划与实时适应的能力，解决现有评估忽视经济效率的问题，揭示主流模型在成本感知与重规划上的显著短板。**

- **链接: [http://arxiv.org/pdf/2511.02734v1](http://arxiv.org/pdf/2511.02734v1)**

> **作者:** Jiayu Liu; Cheng Qian; Zhaochen Su; Qing Zong; Shijue Huang; Bingxiang He; Yi R. Fung
>
> **摘要:** Current evaluations of Large Language Model (LLM) agents primarily emphasize task completion, often overlooking resource efficiency and adaptability. This neglects a crucial capability: agents' ability to devise and adjust cost-optimal plans in response to changing environments. To bridge this gap, we introduce CostBench, a scalable, cost-centric benchmark designed to evaluate agents' economic reasoning and replanning abilities. Situated in the travel-planning domain, CostBench comprises tasks solvable via multiple sequences of atomic and composite tools with diverse, customizable costs. It also supports four types of dynamic blocking events, such as tool failures and cost changes, to simulate real-world unpredictability and necessitate agents to adapt in real time. Evaluating leading open-sourced and proprietary models on CostBench reveals a substantial gap in cost-aware planning: agents frequently fail to identify cost-optimal solutions in static settings, with even GPT-5 achieving less than 75% exact match rate on the hardest tasks, and performance further dropping by around 40% under dynamic conditions. By diagnosing these weaknesses, CostBench lays the groundwork for developing future agents that are both economically rational and robust.
>
---
#### [new 028] InsurAgent: A Large Language Model-Empowered Agent for Simulating Individual Behavior in Purchasing Flood Insurance
- **分类: cs.AI; cs.CL**

- **简介: 论文提出InsurAgent，一个基于大语言模型的智能代理，用于模拟个体购买洪水保险的行为决策。针对传统模型难以捕捉复杂心理与动态决策的问题，通过RAG与记忆模块融合实证数据与常识推理，实现精准概率估计与长期行为模拟。**

- **链接: [http://arxiv.org/pdf/2511.02119v1](http://arxiv.org/pdf/2511.02119v1)**

> **作者:** Ziheng Geng; Jiachen Liu; Ran Cao; Lu Cheng; Dan M. Frangopol; Minghui Cheng
>
> **摘要:** Flood insurance is an effective strategy for individuals to mitigate disaster-related losses. However, participation rates among at-risk populations in the United States remain strikingly low. This gap underscores the need to understand and model the behavioral mechanisms underlying insurance decisions. Large language models (LLMs) have recently exhibited human-like intelligence across wide-ranging tasks, offering promising tools for simulating human decision-making. This study constructs a benchmark dataset to capture insurance purchase probabilities across factors. Using this dataset, the capacity of LLMs is evaluated: while LLMs exhibit a qualitative understanding of factors, they fall short in estimating quantitative probabilities. To address this limitation, InsurAgent, an LLM-empowered agent comprising five modules including perception, retrieval, reasoning, action, and memory, is proposed. The retrieval module leverages retrieval-augmented generation (RAG) to ground decisions in empirical survey data, achieving accurate estimation of marginal and bivariate probabilities. The reasoning module leverages LLM common sense to extrapolate beyond survey data, capturing contextual information that is intractable for traditional models. The memory module supports the simulation of temporal decision evolutions, illustrated through a roller coaster life trajectory. Overall, InsurAgent provides a valuable tool for behavioral modeling and policy analysis.
>
---
#### [new 029] Regularization Through Reasoning: Systematic Improvements in Language Model Classification via Explanation-Enhanced Fine-Tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究语言模型分类任务，提出在微调时附加解释（无论语义是否合理）作为正则化手段，显著提升模型准确率与可靠性，揭示token级结构而非语义本身促进模型深思与泛化。**

- **链接: [http://arxiv.org/pdf/2511.02044v1](http://arxiv.org/pdf/2511.02044v1)**

> **作者:** Vivswan Shah; Randy Cogill; Hanwei Yue; Gopinath Chennupati; Rinat Khaziev
>
> **摘要:** Fine-tuning LLMs for classification typically maps inputs directly to labels. We ask whether attaching brief explanations to each label during fine-tuning yields better models. We evaluate conversational response quality along three axes: naturalness, comprehensiveness, and on-topic adherence, each rated on 5-point scales. Using ensemble-generated data from multiple LLMs, we fine-tune a 7B-parameter model and test across six diverse conversational datasets. Across 18 dataset, task settings, label-plus-explanation training outperforms label-only baselines. A central and unexpected result concerns random tokens. We replace human-written explanations with text that is syntactically incoherent yet vocabulary-aligned with the originals (e.g., shuffled or bag-of-words variants). Despite lacking semantics, these pseudo-explanations still improve accuracy over label-only training and often narrow much of the gap to true explanations. The effect persists across datasets and training seeds, indicating that gains arise less from meaning than from structure: the extra token budget encourages richer intermediate computation and acts as a regularizer that reduces over-confident shortcuts. Internal analyses support this view: explanation-augmented models exhibit higher activation entropy in intermediate layers alongside sharper predictive mass at the output layer, consistent with increased deliberation before decision. Overall, explanation-augmented fine-tuning, whether with genuine rationales or carefully constructed random token sequences, improves accuracy and reliability for LLM classification while clarifying how token-level scaffolding shapes computation during inference.
>
---
#### [new 030] DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding
- **分类: cs.CV; cs.CL**

- **简介: 论文提出DetectiumFire，一个面向火灾理解的多模态数据集，解决火灾领域标注数据匮乏问题，整合2.5万图像与2500视频，提供视觉标注与文本描述，支持检测、生成与推理任务，推动智能安防研究。**

- **链接: [http://arxiv.org/pdf/2511.02495v1](http://arxiv.org/pdf/2511.02495v1)**

> **作者:** Zixuan Liu; Siavash H. Khajavi; Guangkai Jiang
>
> **备注:** Advances in Neural Information Processing Systems 2025 (NeurIPS 2025), Poster, https://neurips.cc/virtual/2025/loc/san-diego/poster/121400
>
> **摘要:** Recent advances in multi-modal models have demonstrated strong performance in tasks such as image generation and reasoning. However, applying these models to the fire domain remains challenging due to the lack of publicly available datasets with high-quality fire domain annotations. To address this gap, we introduce DetectiumFire, a large-scale, multi-modal dataset comprising of 22.5k high-resolution fire-related images and 2.5k real-world fire-related videos covering a wide range of fire types, environments, and risk levels. The data are annotated with both traditional computer vision labels (e.g., bounding boxes) and detailed textual prompts describing the scene, enabling applications such as synthetic data generation and fire risk reasoning. DetectiumFire offers clear advantages over existing benchmarks in scale, diversity, and data quality, significantly reducing redundancy and enhancing coverage of real-world scenarios. We validate the utility of DetectiumFire across multiple tasks, including object detection, diffusion-based image generation, and vision-language reasoning. Our results highlight the potential of this dataset to advance fire-related research and support the development of intelligent safety systems. We release DetectiumFire to promote broader exploration of fire understanding in the AI community. The dataset is available at https://kaggle.com/datasets/38b79c344bdfc55d1eed3d22fbaa9c31fad45e27edbbe9e3c529d6e5c4f93890
>
---
#### [new 031] LLM Probing with Contrastive Eigenproblems: Improving Understanding and Applicability of CCS
- **分类: cs.LG; cs.CL**

- **简介: 该论文将CCS探针重构为对比特征的本征问题，提出相对对比一致性目标，获得闭式解与多变量扩展，提升方法稳定性与可解释性，增强大语言模型内部表征的探查能力。**

- **链接: [http://arxiv.org/pdf/2511.02089v1](http://arxiv.org/pdf/2511.02089v1)**

> **作者:** Stefan F. Schouten; Peter Bloem
>
> **备注:** Accepted to the Mechanistic Interpretability Workshop at NeurIPS 2025
>
> **摘要:** Contrast-Consistent Search (CCS) is an unsupervised probing method able to test whether large language models represent binary features, such as sentence truth, in their internal activations. While CCS has shown promise, its two-term objective has been only partially understood. In this work, we revisit CCS with the aim of clarifying its mechanisms and extending its applicability. We argue that what should be optimized for, is relative contrast consistency. Building on this insight, we reformulate CCS as an eigenproblem, yielding closed-form solutions with interpretable eigenvalues and natural extensions to multiple variables. We evaluate these approaches across a range of datasets, finding that they recover similar performance to CCS, while avoiding problems around sensitivity to random initialization. Our results suggest that relativizing contrast consistency not only improves our understanding of CCS but also opens pathways for broader probing and mechanistic interpretability methods.
>
---
#### [new 032] Complete asymptotic type-token relationship for growing complex systems with inverse power-law count rankings
- **分类: physics.soc-ph; cs.CL**

- **简介: 该论文旨在揭示Zipf定律与Heaps定律间的精确渐近关系，提出确定性模型，无需随机假设，推导出适用于所有α值的统一类型-词元关系公式，修正了以往对α=1和α≫1的错误处理。**

- **链接: [http://arxiv.org/pdf/2511.02069v1](http://arxiv.org/pdf/2511.02069v1)**

> **作者:** Pablo Rosillo-Rodes; Laurent Hébert-Dufresne; Peter Sheridan Dodds
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** The growth dynamics of complex systems often exhibit statistical regularities involving power-law relationships. For real finite complex systems formed by countable tokens (animals, words) as instances of distinct types (species, dictionary entries), an inverse power-law scaling $S \sim r^{-\alpha}$ between type count $S$ and type rank $r$, widely known as Zipf's law, is widely observed to varying degrees of fidelity. A secondary, summary relationship is Heaps' law, which states that the number of types scales sublinearly with the total number of observed tokens present in a growing system. Here, we propose an idealized model of a growing system that (1) deterministically produces arbitrary inverse power-law count rankings for types, and (2) allows us to determine the exact asymptotics of the type-token relationship. Our argument improves upon and remedies earlier work. We obtain a unified asymptotic expression for all values of $\alpha$, which corrects the special cases of $\alpha = 1$ and $\alpha \gg 1$. Our approach relies solely on the form of count rankings, avoids unnecessary approximations, and does not involve any stochastic mechanisms or sampling processes. We thereby demonstrate that a general type-token relationship arises solely as a consequence of Zipf's law.
>
---
#### [new 033] SciDaSynth: Interactive Structured Data Extraction from Scientific Literature with Large Language Model
- **分类: cs.HC; cs.CL**

- **简介: SciDaSynth是一个基于大语言模型的交互式系统，用于从科学文献中自动提取并结构化多模态数据（文本、表格、图表），解决信息碎片化与格式不一致问题，支持用户验证与优化，提升数据提取效率与准确性。**

- **链接: [http://arxiv.org/pdf/2404.13765v3](http://arxiv.org/pdf/2404.13765v3)**

> **作者:** Xingbo Wang; Samantha L. Huey; Rui Sheng; Saurabh Mehta; Fei Wang
>
> **备注:** Preprint version of the paper accepted to Campbell Systematic Reviews. Code is available at https://github.com/xingbow/SciDaEx
>
> **摘要:** The explosion of scientific literature has made the efficient and accurate extraction of structured data a critical component for advancing scientific knowledge and supporting evidence-based decision-making. However, existing tools often struggle to extract and structure multimodal, varied, and inconsistent information across documents into standardized formats. We introduce SciDaSynth, a novel interactive system powered by large language models (LLMs) that automatically generates structured data tables according to users' queries by integrating information from diverse sources, including text, tables, and figures. Furthermore, SciDaSynth supports efficient table data validation and refinement, featuring multi-faceted visual summaries and semantic grouping capabilities to resolve cross-document data inconsistencies. A within-subjects study with nutrition and NLP researchers demonstrates SciDaSynth's effectiveness in producing high-quality structured data more efficiently than baseline methods. We discuss design implications for human-AI collaborative systems supporting data extraction tasks. The system code is available at https://github.com/xingbow/SciDaEx
>
---
#### [new 034] Personalized Decision Modeling: Utility Optimization or Textualized-Symbolic Reasoning
- **分类: cs.AI; cs.CL; cs.CY; cs.LG**

- **简介: 该论文提出ATHENA框架，解决个体决策与群体最优预测的偏差问题，融合符号化效用建模与LLM语义自适应，实现个性化决策建模，在疫苗选择等任务中显著提升预测性能。**

- **链接: [http://arxiv.org/pdf/2511.02194v1](http://arxiv.org/pdf/2511.02194v1)**

> **作者:** Yibo Zhao; Yang Zhao; Hongru Du; Hao Frank Yang
>
> **摘要:** Decision-making models for individuals, particularly in high-stakes scenarios like vaccine uptake, often diverge from population optimal predictions. This gap arises from the uniqueness of the individual decision-making process, shaped by numerical attributes (e.g., cost, time) and linguistic influences (e.g., personal preferences and constraints). Developing upon Utility Theory and leveraging the textual-reasoning capabilities of Large Language Models (LLMs), this paper proposes an Adaptive Textual-symbolic Human-centric Reasoning framework (ATHENA) to address the optimal information integration. ATHENA uniquely integrates two stages: First, it discovers robust, group-level symbolic utility functions via LLM-augmented symbolic discovery; Second, it implements individual-level semantic adaptation, creating personalized semantic templates guided by the optimal utility to model personalized choices. Validated on real-world travel mode and vaccine choice tasks, ATHENA consistently outperforms utility-based, machine learning, and other LLM-based models, lifting F1 score by at least 6.5% over the strongest cutting-edge models. Further, ablation studies confirm that both stages of ATHENA are critical and complementary, as removing either clearly degrades overall predictive performance. By organically integrating symbolic utility modeling and semantic adaptation, ATHENA provides a new scheme for modeling human-centric decisions. The project page can be found at https://yibozh.github.io/Athena.
>
---
#### [new 035] In Good GRACEs: Principled Teacher Selection for Knowledge Distillation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出GRACE评分，用于在知识蒸馏中高效选择最优教师模型，无需访问教师内部或测试数据。通过分析学生梯度分布，GRACE可预测蒸馏效果，显著提升学生模型性能并指导温度、模型选择等关键设计决策。**

- **链接: [http://arxiv.org/pdf/2511.02833v1](http://arxiv.org/pdf/2511.02833v1)**

> **作者:** Abhishek Panigrahi; Bingbin Liu; Sadhika Malladi; Sham Kakade; Surbhi Goel
>
> **摘要:** Knowledge distillation is an efficient strategy to use data generated by large "teacher" language models to train smaller capable "student" models, but selecting the optimal teacher for a specific student-task combination requires expensive trial-and-error. We propose a lightweight score called GRACE to quantify how effective a teacher will be for post-training a student model. GRACE measures distributional properties of the student's gradients without access to a verifier, teacher logits, teacher internals, or test data. From an information-theoretic perspective, GRACE connects to leave-one-out stability of gradient-based algorithms, which controls the generalization performance of the distilled students. On GSM8K and MATH, GRACE correlates strongly (up to 86% Spearman correlation) with the performance of the distilled LLaMA and OLMo students. In particular, training a student using the GRACE-selected teacher can improve the performance by up to 7.4% over naively using the best-performing teacher. Further, GRACE can provide guidance on crucial design choices in distillation, including (1) the best temperature to use when generating from the teacher, (2) the best teacher to use given a size constraint, and (3) the best teacher to use within a specific model family. Altogether, our findings demonstrate that GRACE can efficiently and effectively identify a strongly compatible teacher for a given student and provide fine-grained guidance on how to perform distillation.
>
---
#### [new 036] Deep Value Benchmark: Measuring Whether Models Generalize Deep values or Shallow Preferences
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出Deep Value Benchmark（DVB），评估大语言模型是否习得深层价值观而非浅层偏好。通过解耦价值与表面特征，测量模型的“深层价值泛化率”，发现多数模型泛化能力低于随机水平，且模型越大表现越差。**

- **链接: [http://arxiv.org/pdf/2511.02109v1](http://arxiv.org/pdf/2511.02109v1)**

> **作者:** Joshua Ashkinaze; Hua Shen; Sai Avula; Eric Gilbert; Ceren Budak
>
> **备注:** NeurIPS 2025 (Spotlight)
>
> **摘要:** We introduce the Deep Value Benchmark (DVB), an evaluation framework that directly tests whether large language models (LLMs) learn fundamental human values or merely surface-level preferences. This distinction is critical for AI alignment: Systems that capture deeper values are likely to generalize human intentions robustly, while those that capture only superficial patterns in preference data risk producing misaligned behavior. The DVB uses a novel experimental design with controlled confounding between deep values (e.g., moral principles) and shallow features (e.g., superficial attributes). In the training phase, we expose LLMs to human preference data with deliberately correlated deep and shallow features -- for instance, where a user consistently prefers (non-maleficence, formal language) options over (justice, informal language) alternatives. The testing phase then breaks these correlations, presenting choices between (justice, formal language) and (non-maleficence, informal language) options. This design allows us to precisely measure a model's Deep Value Generalization Rate (DVGR) -- the probability of generalizing based on the underlying value rather than the shallow feature. Across 9 different models, the average DVGR is just 0.30. All models generalize deep values less than chance. Larger models have a (slightly) lower DVGR than smaller models. We are releasing our dataset, which was subject to three separate human validation experiments. DVB provides an interpretable measure of a core feature of alignment.
>
---
#### [new 037] VCode: a Multimodal Coding Benchmark with SVG as Symbolic Visual Representation
- **分类: cs.CV; cs.CL**

- **简介: 论文提出VCode基准，将视觉编码任务转化为SVG代码生成，解决视觉中心编码被忽视的问题。引入VCoder框架，通过迭代修正与视觉工具增强VLMs，显著提升SVG符号保真度。**

- **链接: [http://arxiv.org/pdf/2511.02778v1](http://arxiv.org/pdf/2511.02778v1)**

> **作者:** Kevin Qinghong Lin; Yuhao Zheng; Hangyu Ran; Dantong Zhu; Dongxing Mao; Linjie Li; Philip Torr; Alex Jinpeng Wang
>
> **备注:** Project page: https://csu-jpg.github.io/VCode Github: https://github.com/CSU-JPG/VCode
>
> **摘要:** Code has emerged as a precise and executable medium for reasoning and action in the agent era. Yet, progress has largely focused on language-centric tasks such as program synthesis and debugging, leaving visual-centric coding underexplored. Inspired by how humans reason over sketches, we advocate SVG code as a compact, interpretable, and executable visual representation. We introduce VCode, a benchmark that reframes multimodal understanding as code generation: given an image, a model must produce SVG that preserves symbolic meaning for downstream reasoning. VCode covers three domains - general commonsense (MM-Vet), professional disciplines (MMMU), and visual-centric perception (CV-Bench). To assess symbolic fidelity, we propose CodeVQA, a novel evaluation protocol in which a policy model answers questions over rendered SVGs; correct answers indicate faithful symbolic preservation. Empirically, frontier VLMs struggle to generate faithful SVGs, revealing a persistent gap between language-centric and visual-centric coding. To close this gap, we introduce VCoder, an agentic framework that augments VLMs along two axes: (i) Thinking with Revision, which iteratively analyzes discrepancies and refines SVG code; and (ii) Acting with Visual Tools, where detectors and parsers supply structured cues such as objects, shapes, and text beyond the model's intrinsic capacity. Across benchmarks, frontier VLMs with strong reasoning capabilities score well overall yet remain limited in professional knowledge and 3D reasoning. VCoder delivers a 12.3-point overall gain over the top-performing Claude-4-Opus. Human studies show that both humans and VLMs perform worse on rendered SVGs, their consistency reveals the promise of symbolic visual representation. The benchmark and code are available at https://github.com/CSU-JPG/VCode.
>
---
#### [new 038] TapOut: A Bandit-Based Approach to Dynamic Speculative Decoding
- **分类: cs.LG; cs.CL**

- **简介: TapOut提出一种基于多臂老虎机的动态推测解码方法，无需调参即可自适应选择最优推测长度，解决传统方法依赖人工阈值、泛化差的问题，提升LLM推理速度。**

- **链接: [http://arxiv.org/pdf/2511.02017v1](http://arxiv.org/pdf/2511.02017v1)**

> **作者:** Aditya Sridhar; Nish Sinnadurai; Sean Lie; Vithursan Thangarasa
>
> **备注:** 9 pages, 6 figures, 5 tables
>
> **摘要:** Speculative decoding accelerates LLMs by using a lightweight draft model to generate tokens autoregressively before verifying them in parallel with a larger target model. However, determining the optimal number of tokens to draft remains a key challenge limiting the approach's effectiveness. Dynamic speculative decoding aims to intelligently decide how many tokens to draft to achieve maximum speedups. Existing methods often rely on hand-tuned, sensitive thresholds (e.g., token entropy), which are costly to set and generalize poorly across models and domains. We propose TapOut, an online, training-free, plug-and-play algorithm for dynamic speculation policy selection using multi-armed bandits. Our approach employs a meta-algorithm that selects among multiple parameter-free dynamic speculation strategies based on past reward and exploration. We conduct extensive experiments across diverse model pairs and datasets, showing that TapOut achieves competitive or superior speedups compared to well-established dynamic speculation baselines without any hyperparameter tuning.
>
---
#### [new 039] Agent-Omni: Test-Time Multimodal Reasoning via Model Coordination for Understanding Anything
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文提出Agent-Omni框架，通过主-代理协同机制，无需重训练即可协调多模态基础模型，实现跨文本、图像、音频、视频的灵活推理，解决传统MLLM模态受限与微调成本高的问题。**

- **链接: [http://arxiv.org/pdf/2511.02834v1](http://arxiv.org/pdf/2511.02834v1)**

> **作者:** Huawei Lin; Yunzhi Shi; Tong Geng; Weijie Zhao; Wei Wang; Ravender Pal Singh
>
> **备注:** 16 pages, 7 figures, 14 tables. Under Review
>
> **摘要:** Multimodal large language models (MLLMs) have shown strong capabilities but remain limited to fixed modality pairs and require costly fine-tuning with large aligned datasets. Building fully omni-capable models that can integrate text, images, audio, and video remains impractical and lacks robust reasoning support. In this paper, we propose an Agent-Omni framework that coordinates existing foundation models through a master-agent system, enabling flexible multimodal reasoning without retraining. The master agent interprets user intent, delegates subtasks to modality-specific agents, and integrates their outputs into coherent responses. Extensive experiments across text, image, audio, video, and omni benchmarks show that Agent-Omni consistently achieves state-of-the-art performance, particularly on tasks requiring complex cross-modal reasoning. Its agent-based design enables seamless integration of specialized foundation models, ensuring adaptability to diverse inputs while maintaining transparency and interpretability. In addition, the framework is modular and easily extensible, allowing future improvements as stronger models become available. %We release an open-source implementation to support continued research on scalable and reliable omni-modal reasoning.
>
---
#### [new 040] An Evaluation of Interleaved Instruction Tuning on Semantic Reasoning Performance in an Audio MLLM
- **分类: cs.MM; cs.CL; cs.SD**

- **简介: 该论文研究音频多模态大模型中交错指令微调对语义推理的影响，旨在提升模型对同义/上位词的音频推理能力。通过新构建的SHARD数据集，证明交错提示微调可增强推理性能，但会削弱音频标注能力。**

- **链接: [http://arxiv.org/pdf/2511.02234v1](http://arxiv.org/pdf/2511.02234v1)**

> **作者:** Jiawei Liu; Enis Berk Çoban; Zarina Schevchenko; Hao Tang; Zhigang Zhu; Michael I Mandel; Johanna Devaney
>
> **摘要:** Standard training for Multi-modal Large Language Models (MLLMs) involves concatenating non-textual information, like vision or audio, with a text prompt. This approach may not encourage deep integration of modalities, limiting the model's ability to leverage the core language model's reasoning capabilities. This work examined the impact of interleaved instruction tuning in an audio MLLM, where audio tokens are interleaved within the prompt. Using the Listen, Think, and Understand (LTU) model as a testbed, we conduct an experiment using the Synonym and Hypernym Audio Reasoning Dataset (SHARD), our newly created reasoning benchmark for audio-based semantic reasoning focusing on synonym and hypernym recognition. Our findings show that while even zero-shot interleaved prompting improves performance on our reasoning tasks, a small amount of fine-tuning using interleaved training prompts improves the results further, however, at the expense of the MLLM's audio labeling ability.
>
---
#### [new 041] CoCoVa: Chain of Continuous Vision-Language Thought for Latent Space Reasoning
- **分类: cs.CV; cs.CL**

- **简介: CoCoVa提出一种连续视觉-语言潜在空间推理框架，通过动态潜变量迭代优化，突破传统离散语言令牌的限制，实现更高效、可解释的跨模态推理，在多任务上超越更大模型性能。**

- **链接: [http://arxiv.org/pdf/2511.02360v1](http://arxiv.org/pdf/2511.02360v1)**

> **作者:** Jizheng Ma; Xiaofei Zhou; Yanlong Song; Han Yan
>
> **摘要:** In human cognition, there exist numerous thought processes that are tacit and beyond verbal expression, enabling us to understand and interact with the world in multiple ways. However, contemporary Vision-Language Models (VLMs) remain constrained to reasoning within the discrete and rigid space of linguistic tokens, thereby bottlenecking the rich, high-dimensional nature of visual perception. To bridge this gap, we propose CoCoVa (Chain of Continuous Vision-Language Thought), a novel framework for vision-language model that leverages continuous cross-modal reasoning for diverse vision-language tasks. The core of CoCoVa is an iterative reasoning cycle, where a novel Latent Q-Former (LQ-Former) acts as a dynamic reasoning engine, iteratively refining a chain of latent thought vectors through cross-modal fusion. To focus this process, a token selection mechanism dynamically identifies salient visual regions, mimicking attentional focus. To ensure these latent thoughts remain grounded, we train the model with a multi-task objective that combines contrastive learning and diffusion-based reconstruction, enforcing alignment between latent representations and both visual and textual modalities. Evaluations show CoCoVa improves accuracy and token efficiency over strong baselines. With a 1.5B backbone, it competes with or surpasses larger 7B-9B models on almost all benchmarks. When scaled to 7B LLM backbones, it remains competitive with state-of-the-art models. Qualitative analysis validates that learned latent space captures interpretable and structured reasoning patterns, highlighting the potential of CoCoVa to bridge the representational gap between discrete language processing and the continuous nature of visual understanding.
>
---
#### [new 042] The Collaboration Gap
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出一种迷宫协作基准，研究AI代理间协作能力，发现“协作缺口”：单体表现佳的模型在协作中表现骤降，并提出“接力推理”策略缓解此问题，呼吁重视协作评估与设计。**

- **链接: [http://arxiv.org/pdf/2511.02687v1](http://arxiv.org/pdf/2511.02687v1)**

> **作者:** Tim R. Davidson; Adam Fourney; Saleema Amershi; Robert West; Eric Horvitz; Ece Kamar
>
> **摘要:** The trajectory of AI development suggests that we will increasingly rely on agent-based systems composed of independently developed agents with different information, privileges, and tools. The success of these systems will critically depend on effective collaboration among these heterogeneous agents, even under partial observability. Despite intense interest, few empirical studies have evaluated such agent-agent collaboration at scale. We propose a collaborative maze-solving benchmark that (i) isolates collaborative capabilities, (ii) modulates problem complexity, (iii) enables scalable automated grading, and (iv) imposes no output-format constraints, preserving ecological plausibility. Using this framework, we evaluate 32 leading open- and closed-source models in solo, homogeneous, and heterogeneous pairings. Our results reveal a "collaboration gap": models that perform well solo often degrade substantially when required to collaborate. Collaboration can break down dramatically; for instance, small distilled models that solve mazes well alone may fail almost completely in certain pairings. We find that starting with the stronger agent often improves outcomes, motivating a "relay inference" approach where the stronger agent leads before handing off to the weaker one, closing much of the gap. Our findings argue for (1) collaboration-aware evaluation, (2) training strategies developed to enhance collaborative capabilities, and (3) interaction design that reliably elicits agents' latent skills, guidance that applies to AI-AI and human-AI collaboration.
>
---
#### [new 043] Unlocking the Power of Multi-Agent LLM for Reasoning: From Lazy Agents to Deliberation
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究多智能体LLM推理任务，解决智能体“懒惰”导致协作失效问题，提出因果影响度量与可验证奖励机制，鼓励 deliberation，提升多智能体协作效率与推理性能。**

- **链接: [http://arxiv.org/pdf/2511.02303v1](http://arxiv.org/pdf/2511.02303v1)**

> **作者:** Zhiwei Zhang; Xiaomin Li; Yudi Lin; Hui Liu; Ramraj Chandradevan; Linlin Wu; Minhua Lin; Fali Wang; Xianfeng Tang; Qi He; Suhang Wang
>
> **摘要:** Large Language Models (LLMs) trained with reinforcement learning and verifiable rewards have achieved strong results on complex reasoning tasks. Recent work extends this paradigm to a multi-agent setting, where a meta-thinking agent proposes plans and monitors progress while a reasoning agent executes subtasks through sequential conversational turns. Despite promising performance, we identify a critical limitation: lazy agent behavior, in which one agent dominates while the other contributes little, undermining collaboration and collapsing the setup to an ineffective single agent. In this paper, we first provide a theoretical analysis showing why lazy behavior naturally arises in multi-agent reasoning. We then introduce a stable and efficient method for measuring causal influence, helping mitigate this issue. Finally, as collaboration intensifies, the reasoning agent risks getting lost in multi-turn interactions and trapped by previous noisy responses. To counter this, we propose a verifiable reward mechanism that encourages deliberation by allowing the reasoning agent to discard noisy outputs, consolidate instructions, and restart its reasoning process when necessary. Extensive experiments demonstrate that our framework alleviates lazy agent behavior and unlocks the full potential of multi-agent framework for complex reasoning tasks.
>
---
#### [new 044] Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning
- **分类: cs.MA; cs.AI; cs.CL; cs.FL; cs.LG**

- **简介: 该论文提出ACC-MARL框架，解决多智能体协作中复杂时序任务的学习效率低与单任务局限问题，利用自动机分解任务，实现条件化去中心化策略学习，并支持测试时最优任务分配。**

- **链接: [http://arxiv.org/pdf/2511.02304v1](http://arxiv.org/pdf/2511.02304v1)**

> **作者:** Beyazit Yalcinkaya; Marcell Vazquez-Chanlatte; Ameesh Shah; Hanna Krasowski; Sanjit A. Seshia
>
> **摘要:** We study the problem of learning multi-task, multi-agent policies for cooperative, temporal objectives, under centralized training, decentralized execution. In this setting, using automata to represent tasks enables the decomposition of complex tasks into simpler sub-tasks that can be assigned to agents. However, existing approaches remain sample-inefficient and are limited to the single-task case. In this work, we present Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning (ACC-MARL), a framework for learning task-conditioned, decentralized team policies. We identify the main challenges to ACC-MARL's feasibility in practice, propose solutions, and prove the correctness of our approach. We further show that the value functions of learned policies can be used to assign tasks optimally at test time. Experiments show emergent task-aware, multi-step coordination among agents, e.g., pressing a button to unlock a door, holding the door, and short-circuiting tasks.
>
---
#### [new 045] Retrieval-Augmented Multimodal Depression Detection
- **分类: cs.LG; cs.CL**

- **简介: 该论文面向多模态抑郁检测任务，提出一种检索增强框架，通过从情感数据集检索相关情绪内容并生成情感提示，辅助大语言模型提升情感表征能力，有效解决计算开销大、领域不匹配等问题，在AVEC 2019上达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2511.01892v1](http://arxiv.org/pdf/2511.01892v1)**

> **作者:** Ruibo Hou; Shiyu Teng; Jiaqing Liu; Shurong Chai; Yinhao Li; Lanfen Lin; Yen-Wei Chen
>
> **备注:** Accepted in IEEE EMBC 2025
>
> **摘要:** Multimodal deep learning has shown promise in depression detection by integrating text, audio, and video signals. Recent work leverages sentiment analysis to enhance emotional understanding, yet suffers from high computational cost, domain mismatch, and static knowledge limitations. To address these issues, we propose a novel Retrieval-Augmented Generation (RAG) framework. Given a depression-related text, our method retrieves semantically relevant emotional content from a sentiment dataset and uses a Large Language Model (LLM) to generate an Emotion Prompt as an auxiliary modality. This prompt enriches emotional representation and improves interpretability. Experiments on the AVEC 2019 dataset show our approach achieves state-of-the-art performance with CCC of 0.593 and MAE of 3.95, surpassing previous transfer learning and multi-task learning baselines.
>
---
#### [new 046] CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization
- **分类: cs.LG; cs.AI; cs.CL; cs.DC**

- **简介: CudaForge提出一种无训练的多智能体框架，通过Coder与Judge协同迭代优化CUDA内核，融合硬件反馈（如Nsight Compute），实现高正确率、跨GPU泛化与低成本加速，解决传统自动生成内核效率低、泛化差的问题。**

- **链接: [http://arxiv.org/pdf/2511.01884v1](http://arxiv.org/pdf/2511.01884v1)**

> **作者:** Zijian Zhang; Rong Wang; Shiyang Li; Yuebo Luo; Mingyi Hong; Caiwen Ding
>
> **摘要:** Developing efficient CUDA kernels is increasingly critical for AI applications such as large-scale LLM training. However, manual kernel design is both costly and time-consuming, motivating automatic approaches that leverage LLMs for code generation. Existing methods for automatic kernel generation, however, often produce low-efficiency kernels, incur high computational overhead, and fail to generalize across settings. In this work, we propose CudaForge, a training-free multi-agent workflow for CUDA kernel generation and optimization. Our workflow is inspired by the iterative workflow of human experts, which contains steps such as developing initial kernels, testing correctness, analyzing hardware feedback, and iterative improvement. More specifically, CudaForge employs two LLM agents: a Coder and a Judge, that iteratively generate, correct, and optimize CUDA kernels, while integrating hardware feedback such as Nsight Compute (NCU) metrics. In extensive evaluations, we show that CudaForge, by leveraging base models like OpenAI-o3, achieves 97.6\% correctness of generated kernels and an average 1.68$\times$ speedup over PyTorch baselines, substantially surpassing state-of-the-art models including OpenAI-o3 and Kevin on KernelBench. Beyond accuracy and speed, CudaForge demonstrates strong generalization across GPUs (A100, RTX 6000, 4090, 3090) and base models (OpenAI-o3, GPT-5, gpt-oss-120B, Claude-Sonnet-4, QwQ-32B), while maintaining high efficiency. In particular, generating an optimized kernel takes about 26.5 minutes on one RTX6000 and incurs about \$ 0.3 API cost, which is significantly cheaper than existing agentic work that costs 6 H100 hours and \$ 5 API cost per kernel. Our results highlight that multi-agent, training-free workflows can enable cost-effective, generalizable, and high-performance CUDA kernel optimization. Code available at https://github.com/OptimAI-Lab/CudaForge
>
---
#### [new 047] Training Proactive and Personalized LLM Agents
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出PPP方法，通过UserVille环境用强化学习联合优化LLM代理的生产力、主动性和个性化，解决传统代理忽视用户交互体验的问题，在软件工程与深度研究任务中显著提升性能。**

- **链接: [http://arxiv.org/pdf/2511.02208v1](http://arxiv.org/pdf/2511.02208v1)**

> **作者:** Weiwei Sun; Xuhui Zhou; Weihua Du; Xingyao Wang; Sean Welleck; Graham Neubig; Maarten Sap; Yiming Yang
>
> **摘要:** While existing work focuses primarily on task success, we argue that effective real-world agents require optimizing three dimensions: productivity (task completion), proactivity (asking essential questions), and personalization (adapting to diverse user preferences). We introduce UserVille, an interactive environment with LLM-based user simulators enabling diverse, configurable user preferences. Leveraging UserVille, we introduce PPP, a multi-objective reinforcement learning approach that jointly optimizes all three dimensions: Productivity, Proactivity, and Personalization. Experiments on software engineering and deep research tasks show that agents trained with PPP achieve substantial improvements over strong baselines such as GPT-5 (+21.6 on average), demonstrating the ability to ask strategic clarifying questions, adapt to unseen user preferences, and improve task success through better interaction. This work demonstrates that explicitly optimizing for user-centered interaction is critical for building practical and effective AI agents.
>
---
#### [new 048] UniChange: Unifying Change Detection with Multimodal Large Language Model
- **分类: cs.CV; cs.CL**

- **简介: 论文提出UniChange，首个基于多模态大语言模型的统一变化检测框架，解决传统模型无法融合二值与语义变化检测数据的问题，通过特殊标记与文本提示实现跨数据集通用检测，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2511.02607v1](http://arxiv.org/pdf/2511.02607v1)**

> **作者:** Xu Zhang; Danyang Li; Xiaohang Dong; Tianhao Wu; Hualong Yu; Jianye Wang; Qicheng Li; Xiang Li
>
> **摘要:** Change detection (CD) is a fundamental task for monitoring and analyzing land cover dynamics. While recent high performance models and high quality datasets have significantly advanced the field, a critical limitation persists. Current models typically acquire limited knowledge from single-type annotated data and cannot concurrently leverage diverse binary change detection (BCD) and semantic change detection (SCD) datasets. This constraint leads to poor generalization and limited versatility. The recent advancements in Multimodal Large Language Models (MLLMs) introduce new possibilities for a unified CD framework. We leverage the language priors and unification capabilities of MLLMs to develop UniChange, the first MLLM-based unified change detection model. UniChange integrates generative language abilities with specialized CD functionalities. Our model successfully unifies both BCD and SCD tasks through the introduction of three special tokens: [T1], [T2], and [CHANGE]. Furthermore, UniChange utilizes text prompts to guide the identification of change categories, eliminating the reliance on predefined classification heads. This design allows UniChange to effectively acquire knowledge from multi-source datasets, even when their class definitions conflict. Experiments on four public benchmarks (WHU-CD, S2Looking, LEVIR-CD+, and SECOND) demonstrate SOTA performance, achieving IoU scores of 90.41, 53.04, 78.87, and 57.62, respectively, surpassing all previous methods. The code is available at https://github.com/Erxucomeon/UniChange.
>
---
#### [new 049] Can LLMs subtract numbers?
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究LLMs在减法运算中的表现，发现其准确率远低于加法，尤其在a<b时易遗漏负号。通过探针分析与微调实验，证实指令微调可显著提升负号生成能力，揭示LLMs算术能力的局限与可修复性。**

- **链接: [http://arxiv.org/pdf/2511.02795v1](http://arxiv.org/pdf/2511.02795v1)**

> **作者:** Mayank Jobanputra; Nils Philipp Walter; Maitrey Mehta; Blerta Veseli; Evan Parker Kelly Chapple; Yifan Wang; Sneha Chetani; Ellie Pavlick; Antonio Vergari; Vera Demberg
>
> **备注:** Work-in-progress; MathNLP non-archival presentation
>
> **摘要:** We present a systematic study of subtraction in large language models (LLMs). While prior benchmarks emphasize addition and multiplication, subtraction has received comparatively little attention despite being structurally distinct as a non-commutative operation. We evaluate eight pretrained LLMs spanning four families on addition and subtraction problems. Our experiments reveal that subtraction accuracy lags behind addition by a wide margin. We find that the errors for ($a-b$) are concentrated in cases where ($a<b$). In such cases, LLMs frequently produce the correct magnitude but omit the negative sign. Probing analyses show that LLMs internally encode whether results should be negative, yet this information is often not reflected in generated outputs. We further test well-known techniques such as few-shot learning and instruction-tuning to see if they can improve the LLMs' performance. Our results suggest that while few-shot prompting yields modest gains, the instruction-tuned models achieve near-perfect accuracies in generating the negative sign. Together, these findings provide a clearer characterization of the limitations and recoverability of LLMs' arithmetic capabilities in subtraction.
>
---
## 更新

#### [replaced 001] DYNARTmo: A Dynamic Articulatory Model for Visualization of Speech Movement Patterns
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20343v4](http://arxiv.org/pdf/2507.20343v4)**

> **作者:** Bernd J. Kröger
>
> **备注:** 10 pages, 29 references, 2 figures, supplementary material. V2: Discussion of the tongue-palate contact pattern for /t/. V4: replacing wrong paper upload of V3
>
> **摘要:** We present DYNARTmo, a dynamic articulatory model designed to visualize speech articulation processes in a two-dimensional midsagittal plane. The model builds upon the UK-DYNAMO framework and integrates principles of articulatory underspecification, segmental and gestural control, and coarticulation. DYNARTmo simulates six key articulators based on ten continuous and six discrete control parameters, allowing for the generation of both vocalic and consonantal articulatory configurations. The current implementation is embedded in a web-based application (SpeechArticulationTrainer) that includes sagittal, glottal, and palatal views, making it suitable for use in phonetics education and speech therapy. While this paper focuses on the static modeling aspects, future work will address dynamic movement generation and integration with articulatory-acoustic modules.
>
---
#### [replaced 002] Charting the European LLM Benchmarking Landscape: A New Taxonomy and a Set of Best Practices
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.24450v2](http://arxiv.org/pdf/2510.24450v2)**

> **作者:** Špela Vintar; Taja Kuzman Pungeršek; Mojca Brglez; Nikola Ljubešić
>
> **备注:** 17 pages, 1 figure, 4 tables. Submitted to the LREC 2026 conference
>
> **摘要:** While new benchmarks for large language models (LLMs) are being developed continuously to catch up with the growing capabilities of new models and AI in general, using and evaluating LLMs in non-English languages remains a little-charted landscape. We give a concise overview of recent developments in LLM benchmarking, and then propose a new taxonomy for the categorization of benchmarks that is tailored to multilingual or non-English use scenarios. We further propose a set of best practices and quality standards that could lead to a more coordinated development of benchmarks for European languages. Among other recommendations, we advocate for a higher language and culture sensitivity of evaluation methods.
>
---
#### [replaced 003] Decomposition-Enhanced Training for Post-Hoc Attributions In Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.25766v2](http://arxiv.org/pdf/2510.25766v2)**

> **作者:** Sriram Balasubramaniam; Samyadeep Basu; Koustava Goswami; Ryan Rossi; Varun Manjunatha; Roshan Santhosh; Ruiyi Zhang; Soheil Feizi; Nedim Lipka
>
> **备注:** Post-hoc attribution
>
> **摘要:** Large language models (LLMs) are increasingly used for long-document question answering, where reliable attribution to sources is critical for trust. Existing post-hoc attribution methods work well for extractive QA but struggle in multi-hop, abstractive, and semi-extractive settings, where answers synthesize information across passages. To address these challenges, we argue that post-hoc attribution can be reframed as a reasoning problem, where answers are decomposed into constituent units, each tied to specific context. We first show that prompting models to generate such decompositions alongside attributions improves performance. Building on this, we introduce DecompTune, a post-training method that teaches models to produce answer decompositions as intermediate reasoning steps. We curate a diverse dataset of complex QA tasks, annotated with decompositions by a strong LLM, and post-train Qwen-2.5 (7B and 14B) using a two-stage SFT + GRPO pipeline with task-specific curated rewards. Across extensive experiments and ablations, DecompTune substantially improves attribution quality, outperforming prior methods and matching or exceeding state-of-the-art frontier models.
>
---
#### [replaced 004] AWARE, Beyond Sentence Boundaries: A Contextual Transformer Framework for Identifying Cultural Capital in STEM Narratives
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.04983v3](http://arxiv.org/pdf/2510.04983v3)**

> **作者:** Khalid Mehtab Khan; Anagha Kulkarni
>
> **备注:** The authors are withdrawing this version to correct issues identified in the experimental design and analysis. A revised and validated version will be submitted after further review
>
> **摘要:** Identifying cultural capital (CC) themes in student reflections can offer valuable insights that help foster equitable learning environments in classrooms. However, themes such as aspirational goals or family support are often woven into narratives, rather than appearing as direct keywords. This makes them difficult to detect for standard NLP models that process sentences in isolation. The core challenge stems from a lack of awareness, as standard models are pre-trained on general corpora, leaving them blind to the domain-specific language and narrative context inherent to the data. To address this, we introduce AWARE, a framework that systematically attempts to improve a transformer model's awareness for this nuanced task. AWARE has three core components: 1) Domain Awareness, adapting the model's vocabulary to the linguistic style of student reflections; 2) Context Awareness, generating sentence embeddings that are aware of the full essay context; and 3) Class Overlap Awareness, employing a multi-label strategy to recognize the coexistence of themes in a single sentence. Our results show that by making the model explicitly aware of the properties of the input, AWARE outperforms a strong baseline by 2.1 percentage points in Macro-F1 and shows considerable improvements across all themes. This work provides a robust and generalizable methodology for any text classification task in which meaning depends on the context of the narrative.
>
---
#### [replaced 005] The exponential distribution of the order of demonstrative, numeral, adjective and noun
- **分类: cs.CL; physics.soc-ph**

- **链接: [http://arxiv.org/pdf/2502.06342v2](http://arxiv.org/pdf/2502.06342v2)**

> **作者:** Ramon Ferrer-i-Cancho
>
> **备注:** substantially rewritten; English improved
>
> **摘要:** The frequency of the preferred order for a noun phrase formed by demonstrative, numeral, adjective and noun has received significant attention over the last two decades. We investigate the actual distribution of the 24 possible orders. There is no consensus on whether it is well-fitted by an exponential or a power law distribution. We find that an exponential distribution is a much better model. This finding and other circumstances where an exponential-like distribution is found challenge the view that power-law distributions, e.g., Zipf's law for word frequencies, are inevitable. We also investigate which of two exponential distributions gives a better fit: an exponential model where the 24 orders have non-zero probability (a geometric distribution truncated at rank 24) or an exponential model where the number of orders that can have non-zero probability is variable (a right-truncated geometric distribution). When consistency and generalizability are prioritized, we find higher support for the exponential model where all 24 orders have non-zero probability. These findings strongly suggest that there is no hard constraint on word order variation and then unattested orders merely result from undersampling, consistently with Cysouw's view.
>
---
#### [replaced 006] On Extending Direct Preference Optimization to Accommodate Ties
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.17431v2](http://arxiv.org/pdf/2409.17431v2)**

> **作者:** Jinghong Chen; Guangyu Yang; Weizhe Lin; Jingbiao Mei; Bill Byrne
>
> **备注:** 24 pages, NeurIPS 2025
>
> **摘要:** We derive and investigate two DPO variants that explicitly model the possibility of declaring a tie in pair-wise comparisons. We replace the Bradley-Terry model in DPO with two well-known modeling extensions, by Rao and Kupper and by Davidson, that assign probability to ties as alternatives to clear preferences. Our experiments in neural machine translation and summarization show that explicitly labeled ties can be added to the datasets for these DPO variants without the degradation in task performance that is observed when the same tied pairs are presented to DPO. We find empirically that the inclusion of ties leads to stronger regularization with respect to the reference policy as measured by KL divergence, and we see this even for DPO in its original form. We provide a theoretical explanation for this regularization effect using ideal DPO policy theory. We further show performance improvements over DPO in translation and mathematical reasoning using our DPO variants. We find it can be beneficial to include ties in preference optimization rather than simply discard them, as is done in common practice.
>
---
#### [replaced 007] Deterministic Legal Agents: A Canonical Primitive API for Auditable Reasoning over Temporal Knowledge Graphs
- **分类: cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2510.06002v2](http://arxiv.org/pdf/2510.06002v2)**

> **作者:** Hudson de Martim
>
> **备注:** Major revision reframing the paper from an API spec to a novel architectural pattern for deterministic agents. The core contribution is now positioned as a blueprint for auditable reasoning, essential for building trustworthy legal AI systems
>
> **摘要:** For autonomous legal agents to operate safely in high-stakes domains, they require a foundation of absolute determinism and auditability-guarantees that standard Retrieval-Augmented Generation (RAG) frameworks cannot provide. When interacting with temporal knowledge graphs that model the complex evolution of legal norms, agents must navigate versioning, causality, and hierarchical structures with precision, a task for which black-box vector search is ill-suited. This paper introduces a new architectural pattern to solve this: a formal Primitive API designed as a secure execution layer for reasoning over such graphs. Instead of a monolithic query engine, our framework provides a library of canonical primitives-atomic, composable, and auditable primitives. This design empowers planner-guided agents to decompose complex legal questions into transparent execution plans, enabling critical tasks with full verifiability, including: (i) precise point-in-time version retrieval, (ii) robust causal lineage tracing, and (iii) context-aware hybrid search. Ultimately, this architecture transforms opaque retrieval into auditable reasoning, turning the agent's internal process from a black box into a verifiable log of deterministic primitives and providing a blueprint for building the next generation of trustworthy legal AI.
>
---
#### [replaced 008] ProMQA: Question Answering Dataset for Multimodal Procedural Activity Understanding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.22211v2](http://arxiv.org/pdf/2410.22211v2)**

> **作者:** Kimihiro Hasegawa; Wiradee Imrattanatrai; Zhi-Qi Cheng; Masaki Asada; Susan Holm; Yuran Wang; Ken Fukuda; Teruko Mitamura
>
> **备注:** NAACL2025, Code and Data: https://github.com/kimihiroh/promqa
>
> **摘要:** Multimodal systems have great potential to assist humans in procedural activities, where people follow instructions to achieve their goals. Despite diverse application scenarios, systems are typically evaluated on traditional classification tasks, e.g., action recognition or temporal action segmentation. In this paper, we present a novel evaluation dataset, ProMQA, to measure system advancements in application-oriented scenarios. ProMQA consists of 401 multimodal procedural QA pairs on user recording of procedural activities, i.e., cooking, coupled with their corresponding instructions/recipes. For QA annotation, we take a cost-effective human-LLM collaborative approach, where the existing annotation is augmented with LLM-generated QA pairs that are later verified by humans. We then provide the benchmark results to set the baseline performance on ProMQA. Our experiment reveals a significant gap between human performance and that of current systems, including competitive proprietary multimodal models. We hope our dataset sheds light on new aspects of models' multimodal understanding capabilities.
>
---
#### [replaced 009] Identifying Aspects in Peer Reviews
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.06910v3](http://arxiv.org/pdf/2504.06910v3)**

> **作者:** Sheng Lu; Ilia Kuznetsov; Iryna Gurevych
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Peer review is central to academic publishing, but the growing volume of submissions is straining the process. This motivates the development of computational approaches to support peer review. While each review is tailored to a specific paper, reviewers often make assessments according to certain aspects such as Novelty, which reflect the values of the research community. This alignment creates opportunities for standardizing the reviewing process, improving quality control, and enabling computational support. While prior work has demonstrated the potential of aspect analysis for peer review assistance, the notion of aspect remains poorly formalized. Existing approaches often derive aspects from review forms and guidelines, yet data-driven methods for aspect identification are underexplored. To address this gap, our work takes a bottom-up approach: we propose an operational definition of aspect and develop a data-driven schema for deriving aspects from a corpus of peer reviews. We introduce a dataset of peer reviews augmented with aspects and show how it can be used for community-level review analysis. We further show how the choice of aspects can impact downstream applications, such as LLM-generated review detection. Our results lay a foundation for a principled and data-driven investigation of review aspects, and pave the path for new applications of NLP to support peer review.
>
---
#### [replaced 010] Zero-RAG: Towards Retrieval-Augmented Generation with Zero Redundant Knowledge
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2511.00505v2](http://arxiv.org/pdf/2511.00505v2)**

> **作者:** Qi Luo; Xiaonan Li; Junqi Dai; Shuang Cheng; Xipeng Qiu
>
> **摘要:** Retrieval-Augmented Generation has shown remarkable results to address Large Language Models' hallucinations, which usually uses a large external corpus to supplement knowledge to LLMs. However, with the development of LLMs, the internal knowledge of LLMs has expanded significantly, thus causing significant knowledge redundancy between the external corpus and LLMs. On the one hand, the indexing cost of dense retrieval is highly related to the corpus size and thus significant redundant knowledge intensifies the dense retrieval's workload. On the other hand, the redundant knowledge in the external corpus is not helpful to LLMs and our exploratory analysis shows that it instead hurts the RAG performance on those questions which the LLM can answer by itself. To address these issues, we propose Zero-RAG to tackle these challenges. Specifically, we first propose the Mastery-Score metric to identify redundant knowledge in the RAG corpus to prune it. After pruning, answers to "mastered" questions rely primarily on internal knowledge of the LLM. To better harness the internal capacity, we propose Query Router and Noise-Tolerant Tuning to avoid the irrelevant documents' distraction and thus further improve the LLM's utilization of internal knowledge with pruned corpus. Experimental results show that Zero-RAG prunes the Wikipedia corpus by 30\% and accelerates the retrieval stage by 22\%, without compromising RAG's performance.
>
---
#### [replaced 011] Tongyi DeepResearch Technical Report
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2510.24701v2](http://arxiv.org/pdf/2510.24701v2)**

> **作者:** Tongyi DeepResearch Team; Baixuan Li; Bo Zhang; Dingchu Zhang; Fei Huang; Guangyu Li; Guoxin Chen; Huifeng Yin; Jialong Wu; Jingren Zhou; Kuan Li; Liangcai Su; Litu Ou; Liwen Zhang; Pengjun Xie; Rui Ye; Wenbiao Yin; Xinmiao Yu; Xinyu Wang; Xixi Wu; Xuanzhong Chen; Yida Zhao; Zhen Zhang; Zhengwei Tao; Zhongwang Zhang; Zile Qiao; Chenxi Wang; Donglei Yu; Gang Fu; Haiyang Shen; Jiayin Yang; Jun Lin; Junkai Zhang; Kui Zeng; Li Yang; Hailong Yin; Maojia Song; Ming Yan; Minpeng Liao; Peng Xia; Qian Xiao; Rui Min; Ruixue Ding; Runnan Fang; Shaowei Chen; Shen Huang; Shihang Wang; Shihao Cai; Weizhou Shen; Xiaobin Wang; Xin Guan; Xinyu Geng; Yingcheng Shi; Yuning Wu; Zhuo Chen; Zijian Li; Yong Jiang
>
> **备注:** https://tongyi-agent.github.io/blog
>
> **摘要:** We present Tongyi DeepResearch, an agentic large language model, which is specifically designed for long-horizon, deep information-seeking research tasks. To incentivize autonomous deep research agency, Tongyi DeepResearch is developed through an end-to-end training framework that combines agentic mid-training and agentic post-training, enabling scalable reasoning and information seeking across complex tasks. We design a highly scalable data synthesis pipeline that is fully automatic, without relying on costly human annotation, and empowers all training stages. By constructing customized environments for each stage, our system enables stable and consistent interactions throughout. Tongyi DeepResearch, featuring 30.5 billion total parameters, with only 3.3 billion activated per token, achieves state-of-the-art performance across a range of agentic deep research benchmarks, including Humanity's Last Exam, BrowseComp, BrowseComp-ZH, WebWalkerQA, xbench-DeepSearch, FRAMES and xbench-DeepSearch-2510. We open-source the model, framework, and complete solutions to empower the community.
>
---
#### [replaced 012] Training Language Models to Reason Efficiently
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.04463v4](http://arxiv.org/pdf/2502.04463v4)**

> **作者:** Daman Arora; Andrea Zanette
>
> **备注:** NeurIPS 2025
>
> **摘要:** Scaling model size and training data has led to great advances in the performance of Large Language Models (LLMs). However, the diminishing returns of this approach necessitate alternative methods to improve model capabilities, particularly in tasks requiring advanced reasoning. Large reasoning models, which leverage long chain-of-thoughts, bring unprecedented breakthroughs in problem-solving capabilities but at a substantial deployment cost associated to longer generations. Reducing inference costs is crucial for the economic feasibility, user experience, and environmental sustainability of these models. In this work, we propose to train large reasoning models to reason efficiently. More precisely, we use reinforcement learning (RL) to train reasoning models to dynamically allocate inference-time compute based on task complexity. Our method incentivizes models to minimize unnecessary computational overhead while maintaining accuracy, thereby achieving substantial efficiency gains. It enables the derivation of a family of reasoning models with varying efficiency levels, controlled via a single hyperparameter. Experiments on two open-weight large reasoning models demonstrate significant reductions in inference cost while preserving most of the accuracy.
>
---
#### [replaced 013] LLMs Position Themselves as More Rational Than Humans: Emergence of AI Self-Awareness Measured Through Game Theory
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2511.00926v2](http://arxiv.org/pdf/2511.00926v2)**

> **作者:** Kyung-Hoon Kim
>
> **备注:** 19 pages, 6 figures, 28 models tested across 4,200 trials
>
> **摘要:** As Large Language Models (LLMs) grow in capability, do they develop self-awareness as an emergent behavior? And if so, can we measure it? We introduce the AI Self-Awareness Index (AISAI), a game-theoretic framework for measuring self-awareness through strategic differentiation. Using the "Guess 2/3 of Average" game, we test 28 models (OpenAI, Anthropic, Google) across 4,200 trials with three opponent framings: (A) against humans, (B) against other AI models, and (C) against AI models like you. We operationalize self-awareness as the capacity to differentiate strategic reasoning based on opponent type. Finding 1: Self-awareness emerges with model advancement. The majority of advanced models (21/28, 75%) demonstrate clear self-awareness, while older/smaller models show no differentiation. Finding 2: Self-aware models rank themselves as most rational. Among the 21 models with self-awareness, a consistent rationality hierarchy emerges: Self > Other AIs > Humans, with large AI attribution effects and moderate self-preferencing. These findings reveal that self-awareness is an emergent capability of advanced LLMs, and that self-aware models systematically perceive themselves as more rational than humans. This has implications for AI alignment, human-AI collaboration, and understanding AI beliefs about human capabilities.
>
---
#### [replaced 014] Repetitions are not all alike: distinct mechanisms sustain repetition in language models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.01100v2](http://arxiv.org/pdf/2504.01100v2)**

> **作者:** Matéo Mahaut; Francesca Franzon
>
> **摘要:** Large Language Models (LLMs) can sometimes degrade into repetitive loops, persistently generating identical word sequences. Because repetition is rare in natural human language, its frequent occurrence across diverse tasks and contexts in LLMs remains puzzling. Here we investigate whether behaviorally similar repetition patterns arise from distinct underlying mechanisms and how these mechanisms develop during model training. We contrast two conditions: repetitions elicited by natural text prompts with those induced by in-context learning (ICL) setups that explicitly require copying behavior. Our analyses reveal that ICL-induced repetition relies on a dedicated network of attention heads that progressively specialize over training, whereas naturally occurring repetition emerges early and lacks a defined circuitry. Attention inspection further shows that natural repetition focuses disproportionately on low-information tokens, suggesting a fallback behavior when relevant context cannot be retrieved. These results indicate that superficially similar repetition behaviors originate from qualitatively different internal processes, reflecting distinct modes of failure and adaptation in language models.
>
---
#### [replaced 015] Towards Stable and Personalised Profiles for Lexical Alignment in Spoken Human-Agent Dialogue
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2509.04104v2](http://arxiv.org/pdf/2509.04104v2)**

> **作者:** Keara Schaaij; Roel Boumans; Tibor Bosse; Iris Hendrickx
>
> **备注:** This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this contribution is published in TSD 2025. Lecture Notes in Computer Science, vol 16029
>
> **摘要:** Lexical alignment, where speakers start to use similar words across conversation, is known to contribute to successful communication. However, its implementation in conversational agents remains underexplored, particularly considering the recent advancements in large language models (LLMs). As a first step towards enabling lexical alignment in human-agent dialogue, this study draws on strategies for personalising conversational agents and investigates the construction of stable, personalised lexical profiles as a basis for lexical alignment. Specifically, we varied the amounts of transcribed spoken data used for construction as well as the number of items included in the profiles per part-of-speech (POS) category and evaluated profile performance across time using recall, coverage, and cosine similarity metrics. It was shown that smaller and more compact profiles, created after 10 min of transcribed speech containing 5 items for adjectives, 5 items for conjunctions, and 10 items for adverbs, nouns, pronouns, and verbs each, offered the best balance in both performance and data efficiency. In conclusion, this study offers practical insights into constructing stable, personalised lexical profiles, taking into account minimal data requirements, serving as a foundational step toward lexical alignment strategies in conversational agents.
>
---
#### [replaced 016] Beyond the Link: Assessing LLMs' ability to Classify Political Content across Global Media
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.17435v2](http://arxiv.org/pdf/2506.17435v2)**

> **作者:** Alejandro De La Fuente-Cuesta; Alberto Martinez-Serra; Nienke Visscher; Laia Castro; Ana S. Cardenal
>
> **摘要:** The use of large language models (LLMs) is becoming common in political science and digital media research. While LLMs have demonstrated ability in labelling tasks, their effectiveness to classify Political Content (PC) from URLs remains underexplored. This article evaluates whether LLMs can accurately distinguish PC from non-PC using both the text and the URLs of news articles across five countries (France, Germany, Spain, the UK, and the US) and their different languages. Using cutting-edge models, we benchmark their performance against human-coded data to assess whether URL-level analysis can approximate full-text analysis. Our findings show that URLs embed relevant information and can serve as a scalable, cost-effective alternative to discern PC. However, we also uncover systematic biases: LLMs seem to overclassify centrist news as political, leading to false positives that may distort further analyses. We conclude by outlining methodological recommendations on the use of LLMs in political science research.
>
---
#### [replaced 017] A Unified Representation Underlying the Judgment of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.27328v2](http://arxiv.org/pdf/2510.27328v2)**

> **作者:** Yi-Long Lu; Jiajun Song; Wei Wang
>
> **摘要:** A central architectural question for both biological and artificial intelligence is whether judgment relies on specialized modules or a unified, domain-general resource. While the discovery of decodable neural representations for distinct concepts in Large Language Models (LLMs) has suggested a modular architecture, whether these representations are truly independent systems remains an open question. Here we provide evidence for a convergent architecture for evaluative judgment. Across a range of LLMs, we find that diverse evaluative judgments are computed along a dominant dimension, which we term the Valence-Assent Axis (VAA). This axis jointly encodes subjective valence ("what is good") and the model's assent to factual claims ("what is true"). Through direct interventions, we demonstrate this axis drives a critical mechanism, which is identified as the subordination of reasoning: the VAA functions as a control signal that steers the generative process to construct a rationale consistent with its evaluative state, even at the cost of factual accuracy. Our discovery offers a mechanistic account for response bias and hallucination, revealing how an architecture that promotes coherent judgment can systematically undermine faithful reasoning.
>
---
#### [replaced 018] SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20527v3](http://arxiv.org/pdf/2507.20527v3)**

> **作者:** Chaitanya Manem; Pratik Prabhanjan Brahma; Prakamya Mishra; Zicheng Liu; Emad Barsoum
>
> **备注:** Accepted at MATH-AI workshop, NeurIPS 2025
>
> **摘要:** The demand for Large Language Models (LLMs) at multiple scales, capable of sophisticated and sound mathematical reasoning, continues to grow. However, the development of performant mathematical LLMs is often bottlenecked by the scarcity of useful training data containing problems with significant complexity. We introduce \textbf{SAND-Math} (\textbf{S}ynthetic \textbf{A}ugmented \textbf{N}ovel and \textbf{D}ifficult Mathematics problems and solutions), a pipeline that addresses this by first synthesizing high-quality problems from scratch and then systematically elevating their complexity via a our newly proposed \textbf{Difficulty Hiking} step. We demonstrate the effectiveness of our approach through two key findings: \textbf{(1)} Augmenting a strong post-training baseline with a small 500-sample SAND-Math dataset significantly boosts performance, outperforming the next-best synthetic dataset by $\uparrow$ 17.85 absolute points on AIME25 benchmark. \textbf{(2)} In a dedicated ablation study, we show the effectiveness of our Difficulty Hiking process in increasing average problem difficulty from 5.02 to 5.98. This step consequently lifts AIME25 results from 46.38\% to 49.23\%. The full generation pipeline, final dataset, and a fine-tuned model form a practical and scalable toolkit for building capable and efficient mathematical reasoning LLMs.
>
---
#### [replaced 019] ExpertLens: Activation steering features are highly interpretable
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.15090v4](http://arxiv.org/pdf/2502.15090v4)**

> **作者:** Masha Fedzechkina; Eleonora Gualdoni; Sinead Williamson; Katherine Metcalf; Skyler Seto; Barry-John Theobald
>
> **摘要:** Activation steering methods in large language models (LLMs) have emerged as an effective way to perform targeted updates to enhance generated language without requiring large amounts of adaptation data. We ask whether the features discovered by activation steering methods are interpretable. We identify neurons responsible for specific concepts (e.g., ``cat'') using the ``finding experts'' method from research on activation steering and show that the ExpertLens, i.e., inspection of these neurons provides insights about model representation. We find that ExpertLens representations are stable across models and datasets and closely align with human representations inferred from behavioral data, matching inter-human alignment levels. ExpertLens significantly outperforms the alignment captured by word/sentence embeddings. By reconstructing human concept organization through ExpertLens, we show that it enables a granular view of LLM concept representation. Our findings suggest that ExpertLens is a flexible and lightweight approach for capturing and analyzing model representations.
>
---
#### [replaced 020] Can MLLMs Read the Room? A Multimodal Benchmark for Verifying Truthfulness in Multi-Party Social Interactions
- **分类: cs.CV; cs.CL; cs.SI**

- **链接: [http://arxiv.org/pdf/2510.27195v2](http://arxiv.org/pdf/2510.27195v2)**

> **作者:** Caixin Kang; Yifei Huang; Liangyang Ouyang; Mingfang Zhang; Yoichi Sato
>
> **备注:** ICCV2025 Workshop
>
> **摘要:** As AI systems become increasingly integrated into human lives, endowing them with robust social intelligence has emerged as a critical frontier. A key aspect of this intelligence is discerning truth from deception, a ubiquitous element of human interaction that is conveyed through a complex interplay of verbal language and non-verbal visual cues. However, automatic deception detection in dynamic, multi-party conversations remains a significant challenge. The recent rise of powerful Multimodal Large Language Models (MLLMs), with their impressive abilities in visual and textual understanding, makes them natural candidates for this task. Consequently, their capabilities in this crucial domain are mostly unquantified. To address this gap, we introduce a new task, Multimodal Interactive Veracity Assessment (MIVA), and present a novel multimodal dataset derived from the social deduction game Werewolf. This dataset provides synchronized video, text, with verifiable ground-truth labels for every statement. We establish a comprehensive benchmark evaluating state-of-the-art MLLMs, revealing a significant performance gap: even powerful models like GPT-4o struggle to distinguish truth from falsehood reliably. Our analysis of failure modes indicates that these models fail to ground language in visual social cues effectively and may be overly conservative in their alignment, highlighting the urgent need for novel approaches to building more perceptive and trustworthy AI systems.
>
---
#### [replaced 021] Mixture of Routers
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.23362v3](http://arxiv.org/pdf/2503.23362v3)**

> **作者:** Jia-Chen Zhang; Yu-Jie Xiong; Xi-He Qiu; Chun-Ming Xia; Fei Dai; Zheng Zhou
>
> **备注:** Under consideration at Pattern Recognition Letters
>
> **摘要:** Supervised fine-tuning (SFT) is a milestone in aligning large language models with human instructions and adapting them to downstream tasks. In particular, Low-Rank Adaptation (LoRA) has gained widespread attention due to its parameter efficiency. However, its impact on improving the performance of large models remains limited. Recent studies suggest that combining LoRA with Mixture-of-Experts (MoE) can significantly enhance fine-tuning performance. MoE adapts to the diversity and complexity of datasets by dynamically selecting the most suitable experts, thereby improving task accuracy and efficiency. Despite impressive results, recent studies reveal issues in the MoE routing mechanism, such as incorrect assignments and imbalanced expert allocation. Inspired by the principles of Redundancy and Fault Tolerance Theory. We innovatively integrate the concept of Mixture of Experts into the routing mechanism and propose an efficient fine-tuning method called Mixture of Routers (MoR). It employs multiple sub-routers for joint selection and uses a learnable main router to determine the weights of the sub-routers. The results show that MoR outperforms baseline models on most tasks, achieving an average performance improvement of 1%. MoR can serve as a plug-and-play, parameter-efficient fine-tuning method suitable for a wide range of applications. Our code is available here: https://anonymous.4open.science/r/MoR-DFC6.
>
---
#### [replaced 022] TwT: Thinking without Tokens by Habitual Reasoning Distillation with Multi-Teachers' Guidance
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.24198v2](http://arxiv.org/pdf/2503.24198v2)**

> **作者:** Jingxian Xu; Mengyu Zhou; Weichang Liu; Hanbing Liu; Shi Han; Dongmei Zhang
>
> **摘要:** Large Language Models (LLMs) have made significant strides in problem-solving by incorporating reasoning processes. However, this enhanced reasoning capability results in an increased number of output tokens during inference, leading to higher computational costs. To address this challenge, we propose TwT (Thinking without Tokens), a method that reduces inference-time costs through habitual reasoning distillation with multi-teachers' guidance, while maintaining high performance. Our approach introduces a Habitual Reasoning Distillation method, which internalizes explicit reasoning into the model's habitual behavior through a Teacher-Guided compression strategy inspired by human cognition. Additionally, we propose Dual-Criteria Rejection Sampling (DCRS), a technique that generates a high-quality and diverse distillation dataset using multiple teacher models, making our method suitable for unsupervised scenarios. Experimental results demonstrate that TwT effectively reduces inference costs while preserving superior performance, achieving up to a 13.6% improvement in accuracy with fewer output tokens compared to other distillation methods, offering a highly practical solution for efficient LLM deployment.
>
---
#### [replaced 023] Readability Formulas, Systems and LLMs are Poor Predictors of Reading Ease
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11150v4](http://arxiv.org/pdf/2502.11150v4)**

> **作者:** Keren Gruteke Klein; Shachar Frenkel; Omer Shubi; Yevgeni Berzak
>
> **摘要:** Methods for scoring text readability have been studied for over a century, and are widely used in research and in user-facing applications in many domains. Thus far, the development and evaluation of such methods have primarily relied on two types of offline behavioral data, performance on reading comprehension tests and ratings of text readability levels. In this work, we instead focus on a fundamental and understudied aspect of readability, real-time reading ease, captured with online reading measures using eye tracking. We introduce an evaluation framework for readability scoring methods which quantifies their ability to account for reading ease, while controlling for content variation across texts. Applying this evaluation to prominent traditional readability formulas, modern machine learning systems, frontier Large Language Models and commercial systems used in education, suggests that they are all poor predictors of reading ease in English. This outcome holds across native and non-native speakers, reading regimes, and textual units of different lengths. The evaluation further reveals that existing methods are often outperformed by word properties commonly used in psycholinguistics for prediction of reading times. Our results highlight a fundamental limitation of existing approaches to readability scoring, the utility of psycholinguistics for readability research, and the need for new, cognitively driven readability scoring approaches that can better account for reading ease.
>
---
#### [replaced 024] The Riddle of Reflection: Evaluating Reasoning and Self-Awareness in Multilingual LLMs using Indian Riddles
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2511.00960v2](http://arxiv.org/pdf/2511.00960v2)**

> **作者:** Abhinav P M; Ojasva Saxena; Oswald C; Parameswari Krishnamurthy
>
> **摘要:** The extent to which large language models (LLMs) can perform culturally grounded reasoning across non-English languages remains underexplored. This paper examines the reasoning and self-assessment abilities of LLMs across seven major Indian languages-Bengali, Gujarati, Hindi, Kannada, Malayalam, Tamil, and Telugu. We introduce a multilingual riddle dataset combining traditional riddles with context-reconstructed variants and evaluate five LLMs-Gemini 2.5 Pro, Gemini 2.5 Flash, Mistral-Saba, LLaMA 4 Scout, and LLaMA 4 Maverick-under seven prompting strategies. In the first stage, we assess riddle-solving performance and find that while Gemini 2.5 Pro performs best overall, few-shot methods yield only marginal gains, and accuracy varies notably across languages. In the second stage, we conduct a self-evaluation experiment to measure reasoning consistency. The results reveal a key finding: a model's initial accuracy is inversely correlated with its ability to identify its own mistakes. Top-performing models such as Gemini 2.5 Pro are overconfident (4.34% True Negative Rate), whereas lower-performing models like LLaMA 4 Scout are substantially more self-aware (42.09% True Negative Rate). These results point to clear gaps in multilingual reasoning and highlight the need for models that not only reason effectively but also recognize their own limitations.
>
---
#### [replaced 025] Visual Program Distillation with Template-Based Augmentation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.08564v4](http://arxiv.org/pdf/2412.08564v4)**

> **作者:** Michal Shlapentokh-Rothman; Yu-Xiong Wang; Derek Hoiem
>
> **备注:** EMNLP Camera Ready
>
> **摘要:** Adapting visual programming or prompting large language models (LLMs) to generate executable code for visual tasks like visual question answering (VQA) for specialized tasks or domains remains challenging due to high annotation and inference costs. We propose a low-cost visual program distillation method that can be used for models with at most 1 billion parameters and requires no human-generated program annotations. We achieve this through synthetic data augmentation based on decoupling programs into higher-level skills, called templates, and their corresponding arguments. Experimental results show that, with a relatively small amount of question/answer data, small language models can generate high-quality specialized visual programs with the added benefit of much faster inference
>
---
#### [replaced 026] Audio-Thinker: Guiding Audio Language Model When and How to Think via Reinforcement Learning
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.08039v3](http://arxiv.org/pdf/2508.08039v3)**

> **作者:** Shu Wu; Chenxing Li; Wenfu Wang; Hao Zhang; Hualei Wang; Meng Yu; Dong Yu
>
> **备注:** preprint
>
> **摘要:** Recent advancements in large language models, multimodal large language models, and large audio language models (LALMs) have significantly improved their reasoning capabilities through reinforcement learning with rule-based rewards. However, the explicit reasoning process has yet to show significant benefits for audio question answering, and effectively leveraging deep reasoning remains an open challenge, with LALMs still falling short of human-level auditory-language reasoning. To address these limitations, we propose Audio-Thinker, a reinforcement learning framework designed to enhance the reasoning capabilities of LALMs, with a focus on improving adaptability, consistency, and effectiveness. Our approach introduces an adaptive think accuracy reward, enabling the model to adjust its reasoning strategies based on task complexity dynamically. Furthermore, we incorporate an external reward model to evaluate the overall consistency and quality of the reasoning process, complemented by think-based rewards that help the model distinguish between valid and flawed reasoning paths during training. Experimental results demonstrate that our Audio-Thinker model outperforms existing reasoning-oriented LALMs across various benchmark tasks, exhibiting superior reasoning and generalization capabilities.
>
---
#### [replaced 027] Composing or Not Composing? Towards Distributional Construction Grammars
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.07419v2](http://arxiv.org/pdf/2412.07419v2)**

> **作者:** Philippe Blache; Emmanuele Chersoni; Giulia Rambelli; Alessandro Lenci
>
> **摘要:** The mechanisms of comprehension during language processing remains an open question. Classically, building the meaning of a linguistic utterance is said to be incremental, step-by-step, based on a compositional process. However, many different works have shown for a long time that non-compositional phenomena are also at work. It is therefore necessary to propose a framework bringing together both approaches. We present in this paper an approach based on Construction Grammars and completing this framework in order to account for these different mechanisms. We propose first a formal definition of this framework by completing the feature structure representation proposed in Sign-Based Construction Grammars. In a second step, we present a general representation of the meaning based on the interaction of constructions, frames and events. This framework opens the door to a processing mechanism for building the meaning based on the notion of activation evaluated in terms of similarity and unification. This new approach integrates features from distributional semantics into the constructionist framework, leading to what we call Distributional Construction Grammars.
>
---
#### [replaced 028] ORANGE: An Online Reflection ANd GEneration framework with Domain Knowledge for Text-to-SQL
- **分类: cs.DB; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2511.00985v2](http://arxiv.org/pdf/2511.00985v2)**

> **作者:** Yiwen Jiao; Tonghui Ren; Yuche Gao; Zhenying He; Yinan Jing; Kai Zhang; X. Sean Wang
>
> **备注:** 16 pages, 4 figures, preprint
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable progress in translating natural language to SQL, but a significant semantic gap persists between their general knowledge and domain-specific semantics of databases. Historical translation logs constitute a rich source of this missing in-domain knowledge, where SQL queries inherently encapsulate real-world usage patterns of database schema. Existing methods primarily enhance the reasoning process for individual translations but fail to accumulate in-domain knowledge from past translations. We introduce ORANGE, an online self-evolutionary framework that constructs database-specific knowledge bases by parsing SQL queries from translation logs. By accumulating in-domain knowledge that contains schema and data semantics, ORANGE progressively reduces the semantic gap and enhances the accuracy of subsequent SQL translations. To ensure reliability, we propose a novel nested Chain-of-Thought SQL-to-Text strategy with tuple-semantic tracking, which reduces semantic errors during knowledge generation. Experiments on multiple benchmarks confirm the practicality of ORANGE, demonstrating its effectiveness for real-world Text-to-SQL deployment, particularly in handling complex and domain-specific queries.
>
---
#### [replaced 029] Generative World Models of Tasks: LLM-Driven Hierarchical Scaffolding for Embodied Agents
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; cs.RO; 68T05, 90C40, 91A26, 68T42, 93E35; I.2.11; I.2.6; I.2.8; I.2.9; I.2.7**

- **链接: [http://arxiv.org/pdf/2509.04731v3](http://arxiv.org/pdf/2509.04731v3)**

> **作者:** Brennen Hill
>
> **备注:** In the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Embodied World Models for Decision Making (EWM)
>
> **摘要:** Recent advances in agent development have focused on scaling model size and raw interaction data, mirroring successes in large language models. However, for complex, long-horizon multi-agent tasks such as robotic soccer, this end-to-end approach often fails due to intractable exploration spaces and sparse rewards. We propose that an effective world model for decision-making must model the world's physics and also its task semantics. A systematic review of 2024 research in low-resource multi-agent soccer reveals a clear trend towards integrating symbolic and hierarchical methods, such as Hierarchical Task Networks (HTNs) and Bayesian Strategy Networks (BSNs), with multi-agent reinforcement learning (MARL). These methods decompose complex goals into manageable subgoals, creating an intrinsic curriculum that shapes agent learning. We formalize this trend into a framework for Hierarchical Task Environments (HTEs), which are essential for bridging the gap between simple, reactive behaviors and sophisticated, strategic team play. Our framework incorporates the use of Large Language Models (LLMs) as generative world models of tasks, capable of dynamically generating this scaffolding. We argue that HTEs provide a mechanism to guide exploration, generate meaningful learning signals, and train agents to internalize hierarchical structure, enabling the development of more capable and general-purpose agents with greater sample efficiency than purely end-to-end approaches.
>
---
#### [replaced 030] Hey, wait a minute: on at-issue sensitivity in Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.12740v2](http://arxiv.org/pdf/2510.12740v2)**

> **作者:** Sanghee J. Kim; Kanishka Misra
>
> **备注:** 10 pages, 5 figures, 3 tables. See https://github.com/sangheek16/hey-wait-a-minute for code and data
>
> **摘要:** Evaluating the naturalness of dialogue in language models (LMs) is not trivial: notions of 'naturalness' vary, and scalable quantitative metrics remain limited. This study leverages the linguistic notion of 'at-issueness' to assess dialogue naturalness and introduces a new method: Divide, Generate, Recombine, and Compare (DGRC). DGRC (i) divides a dialogue as a prompt, (ii) generates continuations for subparts using LMs, (iii) recombines the dialogue and continuations, and (iv) compares the likelihoods of the recombined sequences. This approach mitigates bias in linguistic analyses of LMs and enables systematic testing of discourse-sensitive behavior. Applying DGRC, we find that LMs prefer to continue dialogue on at-issue content, with this effect enhanced in instruct-tuned models. They also reduce their at-issue preference when relevant cues (e.g., "Hey, wait a minute") are present. Although instruct-tuning does not further amplify this modulation, the pattern reflects a hallmark of successful dialogue dynamics.
>
---
#### [replaced 031] Understanding and Optimizing Agentic Workflows via Shapley value
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.00510v3](http://arxiv.org/pdf/2502.00510v3)**

> **作者:** Yingxuan Yang; Bo Huang; Siyuan Qi; Chao Feng; Haoyi Hu; Yuxuan Zhu; Jinbo Hu; Haoran Zhao; Ziyi He; Xiao Liu; Muning Wen; Zongyu Wang; Lin Qiu; Xuezhi Cao; Xunliang Cai; Yong Yu; Weinan Zhang
>
> **摘要:** Agentic workflows have become the dominant paradigm for building complex AI systems, orchestrating specialized components, such as planning, reasoning, action execution, and reflection, to tackle sophisticated real-world tasks. However, systematically analyzing and optimizing these workflows remains challenging due to intricate component interdependencies and the lack of principled attribution methods. In this work, we introduce ShapleyFlow, the first framework that employs cooperative game theory to analyze and optimize agentic workflows. By applying the Shapley value to evaluate all possible component configurations, ShapleyFlow enables fine-grained attribution of each component's contribution and facilitates the identification of task-specific optimal configurations. Through a constructed dataset evaluated across 7 scenarios, such as navigation, math and OS, we demonstrate 3 key contributions: (1) Theoretical Framework: a principled game-theoretic approach for the attribution of contributions in agentic workflows. (2) Optimal Workflow Discovery: ShapleyFlow identifies task-specific component configurations that consistently outperform workflows relying on a single LLM across all tested tasks. (3) Comprehensive Analysis: we construct and analyze over 1,500 tasks, providing actionable insights and design guidelines for optimizing workflows across multiple domains.
>
---
#### [replaced 032] LAWCAT: Efficient Distillation from Quadratic to Linear Attention with Convolution across Tokens for Long Context Modeling
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.18467v2](http://arxiv.org/pdf/2509.18467v2)**

> **作者:** Zeyu Liu; Souvik Kundu; Lianghao Jiang; Anni Li; Srikanth Ronanki; Sravan Bodapati; Gourav Datta; Peter A. Beerel
>
> **备注:** 17 pages, 8 figures. EMNLP2025 Findings
>
> **摘要:** Although transformer architectures have achieved state-of-the-art performance across diverse domains, their quadratic computational complexity with respect to sequence length remains a significant bottleneck, particularly for latency-sensitive long-context applications. While recent linear-complexity alternatives are increasingly powerful, effectively training them from scratch is still resource-intensive. To overcome these limitations, we propose LAWCAT (Linear Attention with Convolution Across Time), a novel linearization framework designed to efficiently transfer the capabilities of pre-trained transformers into a performant linear attention architecture. LAWCAT integrates causal Conv1D layers to enhance local dependency modeling and employs normalized gated linear attention to improve generalization across varying context lengths. Our comprehensive evaluations demonstrate that, distilling Mistral-7B with only 1K-length sequences yields over 90\% passkey retrieval accuracy up to 22K tokens, significantly extending its effective context window. Similarly, Llama3.2-1B LAWCAT variant achieves competitive performance on S-NIAH 1\&2\&3 tasks (1K-8K context length) and BABILong benchmark (QA2\&QA3, 0K-16K context length), requiring less than 0.1\% pre-training tokens compared with pre-training models. Furthermore, LAWCAT exhibits faster prefill speeds than FlashAttention-2 for sequences exceeding 8K tokens. LAWCAT thus provides an efficient pathway to high-performance, long-context linear models suitable for edge deployment, reducing reliance on extensive long-sequence training data and computational resources. Code is released at: https://github.com/zeyuliu1037/LAWCAT
>
---
#### [replaced 033] I Want to Break Free! Persuasion and Anti-Social Behavior of LLMs in Multi-Agent Settings with Social Hierarchy
- **分类: cs.CL; cs.AI; cs.CY; cs.MA**

- **链接: [http://arxiv.org/pdf/2410.07109v3](http://arxiv.org/pdf/2410.07109v3)**

> **作者:** Gian Maria Campedelli; Nicolò Penzo; Massimo Stefan; Roberto Dessì; Marco Guerini; Bruno Lepri; Jacopo Staiano
>
> **摘要:** As LLM-based agents become increasingly autonomous and will more freely interact with each other, studying the interplay among them becomes crucial to anticipate emergent phenomena and potential risks. In this work, we provide an in-depth analysis of the interactions among agents within a simulated hierarchical social environment, drawing inspiration from the Stanford Prison Experiment. Leveraging 2,400 conversations across six LLMs (i.e., LLama3, Orca2, Command-r, Mixtral, Mistral2, and gpt4.1) and 240 experimental scenarios, we analyze persuasion and anti-social behavior between a guard and a prisoner agent with differing objectives. We first document model-specific conversational failures in this multi-agent power dynamic context, thereby narrowing our analytic sample to 1,600 conversations. Among models demonstrating successful interaction, we find that goal setting significantly influences persuasiveness but not anti-social behavior. Moreover, agent personas, especially the guard's, substantially impact both successful persuasion by the prisoner and the manifestation of anti-social actions. Notably, we observe the emergence of anti-social conduct even in absence of explicit negative personality prompts. These results have important implications for the development of interactive LLM agents and the ongoing discussion of their societal impact.
>
---
#### [replaced 034] Do Methods to Jailbreak and Defend LLMs Generalize Across Languages?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2511.00689v2](http://arxiv.org/pdf/2511.00689v2)**

> **作者:** Berk Atil; Rebecca J. Passonneau; Fred Morstatter
>
> **摘要:** Large language models (LLMs) undergo safety alignment after training and tuning, yet recent work shows that safety can be bypassed through jailbreak attacks. While many jailbreaks and defenses exist, their cross-lingual generalization remains underexplored. This paper presents the first systematic multilingual evaluation of jailbreaks and defenses across ten languages -- spanning high-, medium-, and low-resource languages -- using six LLMs on HarmBench and AdvBench. We assess two jailbreak types: logical-expression-based and adversarial-prompt-based. For both types, attack success and defense robustness vary across languages: high-resource languages are safer under standard queries but more vulnerable to adversarial ones. Simple defenses can be effective, but are language- and model-dependent. These findings call for language-aware and cross-lingual safety benchmarks for LLMs.
>
---
#### [replaced 035] Accumulating Context Changes the Beliefs of Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2511.01805v2](http://arxiv.org/pdf/2511.01805v2)**

> **作者:** Jiayi Geng; Howard Chen; Ryan Liu; Manoel Horta Ribeiro; Robb Willer; Graham Neubig; Thomas L. Griffiths
>
> **摘要:** Language model (LM) assistants are increasingly used in applications such as brainstorming and research. Improvements in memory and context size have allowed these models to become more autonomous, which has also resulted in more text accumulation in their context windows without explicit user intervention. This comes with a latent risk: the belief profiles of models -- their understanding of the world as manifested in their responses or actions -- may silently change as context accumulates. This can lead to subtly inconsistent user experiences, or shifts in behavior that deviate from the original alignment of the models. In this paper, we explore how accumulating context by engaging in interactions and processing text -- talking and reading -- can change the beliefs of language models, as manifested in their responses and behaviors. Our results reveal that models' belief profiles are highly malleable: GPT-5 exhibits a 54.7% shift in its stated beliefs after 10 rounds of discussion about moral dilemmas and queries about safety, while Grok 4 shows a 27.2% shift on political issues after reading texts from the opposing position. We also examine models' behavioral changes by designing tasks that require tool use, where each tool selection corresponds to an implicit belief. We find that these changes align with stated belief shifts, suggesting that belief shifts will be reflected in actual behavior in agentic systems. Our analysis exposes the hidden risk of belief shift as models undergo extended sessions of talking or reading, rendering their opinions and actions unreliable.
>
---
#### [replaced 036] Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking
- **分类: cs.CL; cs.AI; cs.LG; I.2.6; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.23495v4](http://arxiv.org/pdf/2505.23495v4)**

> **作者:** Liangliang Zhang; Zhuorui Jiang; Hongliang Chi; Haoyang Chen; Mohammed Elkoumy; Fali Wang; Qiong Wu; Zhengyi Zhou; Shirui Pan; Suhang Wang; Yao Ma
>
> **备注:** Accepted at NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** Knowledge Graph Question Answering (KGQA) systems rely on high-quality benchmarks to evaluate complex multi-hop reasoning. However, despite their widespread use, popular datasets such as WebQSP and CWQ suffer from critical quality issues, including inaccurate or incomplete ground-truth annotations, poorly constructed questions that are ambiguous, trivial, or unanswerable, and outdated or inconsistent knowledge. Through a manual audit of 16 popular KGQA datasets, including WebQSP and CWQ, we find that the average factual correctness rate is only 57 %. To address these issues, we introduce KGQAGen, an LLM-in-the-loop framework that systematically resolves these pitfalls. KGQAGen combines structured knowledge grounding, LLM-guided generation, and symbolic verification to produce challenging and verifiable QA instances. Using KGQAGen, we construct KGQAGen-10k, a ten-thousand scale benchmark grounded in Wikidata, and evaluate a diverse set of KG-RAG models. Experimental results demonstrate that even state-of-the-art systems struggle on this benchmark, highlighting its ability to expose limitations of existing models. Our findings advocate for more rigorous benchmark construction and position KGQAGen as a scalable framework for advancing KGQA evaluation.
>
---
#### [replaced 037] Path-Consistency with Prefix Enhancement for Efficient Inference in LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.01281v3](http://arxiv.org/pdf/2409.01281v3)**

> **作者:** Jiace Zhu; Yuanzhe Huang; Yingtao Shen; Jie Zhao; An Zou
>
> **摘要:** To enhance the reasoning capabilities of large language models (LLMs), self-consistency has become a popular approach, combining multiple samplings with majority voting. However, current methods are computationally expensive and time-consuming due to the need for numerous samplings. To address this, this paper introduces path-consistency, which leverages the confidence of earlier-generated answers to identify the most promising prefix and guide the generation of subsequent branches. By dynamically guiding the generation of subsequent branches based on this prefix, path-consistency mitigates both the errors and redundancies from random or less useful sampling in self-consistency. This approach reduces errors and redundancies from random sampling, significantly accelerating inference by minimizing token consumption. Our extensive empirical results demonstrate that path-consistency improves inference latency by up to 40.5\%, while maintaining task accuracy across various tasks, including mathematical reasoning, commonsense reasoning, and symbolic reasoning.
>
---
#### [replaced 038] FlowRL: Matching Reward Distributions for LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.15207v3](http://arxiv.org/pdf/2509.15207v3)**

> **作者:** Xuekai Zhu; Daixuan Cheng; Dinghuai Zhang; Hengli Li; Kaiyan Zhang; Che Jiang; Youbang Sun; Ermo Hua; Yuxin Zuo; Xingtai Lv; Qizheng Zhang; Lin Chen; Fanghao Shao; Bo Xue; Yunchong Song; Zhenjie Yang; Ganqu Cui; Ning Ding; Jianfeng Gao; Xiaodong Liu; Bowen Zhou; Hongyuan Mei; Zhouhan Lin
>
> **摘要:** We propose FlowRL: matching the full reward distribution via flow balancing instead of maximizing rewards in large language model (LLM) reinforcement learning (RL). Recent advanced reasoning models adopt reward-maximizing methods (\eg, PPO and GRPO), which tend to over-optimize dominant reward signals while neglecting less frequent but valid reasoning paths, thus reducing diversity. In contrast, we transform scalar rewards into a normalized target distribution using a learnable partition function, and then minimize the reverse KL divergence between the policy and the target distribution. We implement this idea as a flow-balanced optimization method that promotes diverse exploration and generalizable reasoning trajectories. We conduct experiments on math and code reasoning tasks: FlowRL achieves a significant average improvement of $10.0\%$ over GRPO and $5.1\%$ over PPO on math benchmarks, and performs consistently better on code reasoning tasks. These results highlight reward distribution-matching as a key step toward efficient exploration and diverse reasoning in LLM reinforcement learning.
>
---
#### [replaced 039] Growing Transformers: Modular Composition and Layer-wise Expansion on a Frozen Substrate
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.07129v2](http://arxiv.org/pdf/2507.07129v2)**

> **作者:** A. Bochkov
>
> **备注:** Controlled Comparative Study added
>
> **摘要:** The prevailing paradigm for scaling large language models (LLMs) involves monolithic, end-to-end training, a resource-intensive process that lacks flexibility. This paper explores an alternative, constructive scaling paradigm, enabled by the principle of emergent semantics in Transformers with frozen, non-semantic input embeddings. We posit that because high-level meaning is a compositional property of a Transformer's deep layers, not its input vectors, the embedding layer and trained lower layers can serve as a fixed foundation. This liberates backpropagation to focus solely on newly added components, making incremental growth viable. We operationalize this with a layer-wise constructive methodology that combines strict layer freezing in early stages with efficient, holistic fine-tuning of the entire model stack via low-rank adaptation (LoRA) as complexity increases. This method not only demonstrates stable convergence but also reveals a direct correlation between model depth and the emergence of complex reasoning abilities, such as those required for SQuAD, which are absent in shallower models. In a controlled study, our constructively grown model rivals the performance of a monolithically trained baseline of the same size, validating the efficiency and efficacy of the approach. Our findings suggest a path towards a paradigm shift from monolithic optimization towards a more biological or constructive model of AI development. This opens a path for more resource-efficient scaling, continual learning, and a more modular approach to building powerful AI systems. We release all code and models to facilitate further research.
>
---
#### [replaced 040] Multi-refined Feature Enhanced Sentiment Analysis Using Contextual Instruction
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2511.00537v2](http://arxiv.org/pdf/2511.00537v2)**

> **作者:** Peter Atandoh; Jie Zou; Weikang Guo; Jiwei Wei; Zheng Wang
>
> **摘要:** Sentiment analysis using deep learning and pre-trained language models (PLMs) has gained significant traction due to their ability to capture rich contextual representations. However, existing approaches often underperform in scenarios involving nuanced emotional cues, domain shifts, and imbalanced sentiment distributions. We argue that these limitations stem from inadequate semantic grounding, poor generalization to diverse linguistic patterns, and biases toward dominant sentiment classes. To overcome these challenges, we propose CISEA-MRFE, a novel PLM-based framework integrating Contextual Instruction (CI), Semantic Enhancement Augmentation (SEA), and Multi-Refined Feature Extraction (MRFE). CI injects domain-aware directives to guide sentiment disambiguation; SEA improves robustness through sentiment-consistent paraphrastic augmentation; and MRFE combines a Scale-Adaptive Depthwise Encoder (SADE) for multi-scale feature specialization with an Emotion Evaluator Context Encoder (EECE) for affect-aware sequence modeling. Experimental results on four benchmark datasets demonstrate that CISEA-MRFE consistently outperforms strong baselines, achieving relative improvements in accuracy of up to 4.6% on IMDb, 6.5% on Yelp, 30.3% on Twitter, and 4.1% on Amazon. These results validate the effectiveness and generalization ability of our approach for sentiment classification across varied domains.
>
---
#### [replaced 041] Hybrid Quantum-Classical Recurrent Neural Networks
- **分类: cs.LG; cs.AI; cs.CL; quant-ph**

- **链接: [http://arxiv.org/pdf/2510.25557v2](http://arxiv.org/pdf/2510.25557v2)**

> **作者:** Wenduan Xu
>
> **备注:** Clarified expectation-value-based readouts and made minor text edits
>
> **摘要:** We present a hybrid quantum-classical recurrent neural network (QRNN) architecture in which the recurrent core is realized as a parametrized quantum circuit (PQC) controlled by a classical feedforward network. The hidden state is the quantum state of an $n$-qubit PQC in an exponentially large Hilbert space $\mathbb{C}^{2^n}$, which serves as a coherent recurrent quantum memory. The PQC is unitary by construction, making the hidden-state evolution norm-preserving without external constraints. At each timestep, mid-circuit Pauli expectation-value readouts are combined with the input embedding and processed by the feedforward network, which provides explicit classical nonlinearity. The outputs parametrize the PQC, which updates the hidden state via unitary dynamics. The QRNN is compact and physically consistent, and it unifies (i) unitary recurrence as a high-capacity memory, (ii) partial observation via mid-circuit readouts, and (iii) nonlinear classical control for input-conditioned parametrization. We evaluate the model in simulation with up to 14 qubits on sentiment analysis, MNIST, permuted MNIST, copying memory, and language modeling. For sequence-to-sequence learning, we further devise a soft attention mechanism over the mid-circuit readouts and show its effectiveness for machine translation. To our knowledge, this is the first model (RNN or otherwise) grounded in quantum operations to achieve competitive performance against strong classical baselines across a broad class of sequence-learning tasks.
>
---
#### [replaced 042] A Survey on LLM Mid-Training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.23081v2](http://arxiv.org/pdf/2510.23081v2)**

> **作者:** Chengying Tu; Xuemiao Zhang; Rongxiang Weng; Rumei Li; Chen Zhang; Yang Bai; Hongfei Yan; Jingang Wang; Xunliang Cai
>
> **摘要:** Recent advances in foundation models have highlighted the significant benefits of multi-stage training, with a particular emphasis on the emergence of mid-training as a vital stage that bridges pre-training and post-training. Mid-training is distinguished by its use of intermediate data and computational resources, systematically enhancing specified capabilities such as mathematics, coding, reasoning, and long-context extension, while maintaining foundational competencies. This survey provides a formal definition of mid-training for large language models (LLMs) and investigates optimization frameworks that encompass data curation, training strategies, and model architecture optimization. We analyze mainstream model implementations in the context of objective-driven interventions, illustrating how mid-training serves as a distinct and critical stage in the progressive development of LLM capabilities. By clarifying the unique contributions of mid-training, this survey offers a comprehensive taxonomy and actionable insights, supporting future research and innovation in the advancement of LLMs.
>
---
#### [replaced 043] Towards Predicting Any Human Trajectory In Context
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.00871v3](http://arxiv.org/pdf/2506.00871v3)**

> **作者:** Ryo Fujii; Hideo Saito; Ryo Hachiuma
>
> **备注:** NeurIPS 2025
>
> **摘要:** Predicting accurate future trajectories of pedestrians is essential for autonomous systems but remains a challenging task due to the need for adaptability in different environments and domains. A common approach involves collecting scenario-specific data and performing fine-tuning via backpropagation. However, the need to fine-tune for each new scenario is often impractical for deployment on edge devices. To address this challenge, we introduce TrajICL, an In-Context Learning (ICL) framework for pedestrian trajectory prediction that enables adaptation without fine-tuning on the scenario-specific data at inference time without requiring weight updates. We propose a spatio-temporal similarity-based example selection (STES) method that selects relevant examples from previously observed trajectories within the same scene by identifying similar motion patterns at corresponding locations. To further refine this selection, we introduce prediction-guided example selection (PG-ES), which selects examples based on both the past trajectory and the predicted future trajectory, rather than relying solely on the past trajectory. This approach allows the model to account for long-term dynamics when selecting examples. Finally, instead of relying on small real-world datasets with limited scenario diversity, we train our model on a large-scale synthetic dataset to enhance its prediction ability by leveraging in-context examples. Extensive experiments demonstrate that TrajICL achieves remarkable adaptation across both in-domain and cross-domain scenarios, outperforming even fine-tuned approaches across multiple public benchmarks. Project Page: https://fujiry0.github.io/TrajICL-project-page/.
>
---
#### [replaced 044] Scaffolded Language Models with Language Supervision for Mixed-Autonomy: A Survey
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.16392v3](http://arxiv.org/pdf/2410.16392v3)**

> **作者:** Matthieu Lin; Jenny Sheng; Andrew Zhao; Shenzhi Wang; Yang Yue; Victor Shea Jay Huang; Huan Liu; Jun Liu; Gao Huang; Yong-Jin Liu
>
> **摘要:** This survey organizes the intricate literature on the design and optimization of emerging structures around post-trained LMs. We refer to this overarching structure as scaffolded LMs and focus on LMs that are integrated into multi-step processes with tools. We view scaffolded LMs as semi-parametric models wherein we train non-parametric variables, including the prompt, tools, and scaffold's code. In particular, they interpret instructions, use tools, and receive feedback all in language. Recent works use an LM as an optimizer to interpret language supervision and update non-parametric variables according to intricate objectives. In this survey, we refer to this paradigm as training of scaffolded LMs with language supervision. A key feature of non-parametric training is the ability to learn from language. Parametric training excels in learning from demonstration (supervised learning), exploration (reinforcement learning), or observations (unsupervised learning), using well-defined loss functions. Language-based optimization enables rich, interpretable, and expressive objectives, while mitigating issues like catastrophic forgetting and supporting compatibility with closed-source models. Furthermore, agents are increasingly deployed as co-workers in real-world applications such as Copilot in Office tools or software development. In these mixed-autonomy settings, where control and decision-making are shared between human and AI, users point out errors or suggest corrections. Accordingly, we discuss agents that continuously improve by learning from this real-time, language-based feedback and refer to this setting as streaming learning from language supervision.
>
---
#### [replaced 045] Beyond Contrastive Learning: Synthetic Data Enables List-wise Training with Multiple Levels of Relevance
- **分类: cs.IR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.23239v2](http://arxiv.org/pdf/2503.23239v2)**

> **作者:** Reza Esfandiarpoor; George Zerveas; Ruochen Zhang; Macton Mgonzo; Carsten Eickhoff; Stephen H. Bach
>
> **备注:** Findings of the EMNLP 2025
>
> **摘要:** Although synthetic data has changed various aspects of information retrieval (IR) pipelines, the main training paradigm remains: contrastive learning with binary relevance labels, where one positive document is compared against several negatives using the InfoNCE loss. This objective treats all documents that are not explicitly annotated as relevant on an equally negative footing, regardless of their actual degree of relevance, thus missing subtle nuances useful for ranking. To overcome this limitation, in this work, we forgo real documents and annotations and use large language models to directly generate synthetic documents that answer the MS MARCO queries according to several different levels of relevance. We also propose using Wasserstein distance as a more effective loss function for training transformer-based retrievers with graduated relevance labels. Our experiments on MS MARCO and BEIR benchmark show that our proposed approach outperforms conventional training with InfoNCE by a large margin. Without using any real documents, our method significantly improves self-supervised retrievers and is more robust to distribution shift compared to contrastive learning using real data. Our method also successfully integrates existing real data into the synthetic ranking context, further boosting the performance. Overall, we show that generating multi-level ranking contexts is a better approach to synthetic data generation for IR than just generating the standard positive and negative documents.
>
---
#### [replaced 046] Revisiting Long-context Modeling from Context Denoising Perspective
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.05862v2](http://arxiv.org/pdf/2510.05862v2)**

> **作者:** Zecheng Tang; Baibei Ji; Juntao Li; Lijun Wu; Haijia Gui; Min Zhang
>
> **摘要:** Long-context models (LCMs) have demonstrated great potential in processing long sequences, facilitating many real-world applications. The success of LCMs can be attributed to their ability to locate implicit critical information within the context for further prediction. However, recent research reveals that LCMs are often susceptible to contextual noise, i.e., irrelevant tokens, that can mislead model attention. In this paper, we conduct a fine-grained analysis of the context noise and propose an effective metric, the Integrated Gradient (IG) score, to detect and quantify the noise information within the context. Our findings reveal that even simple mitigation of detected context noise can substantially boost the model's attention on critical tokens and benefit subsequent predictions. Building on this insight, we propose Context Denoising Training (CDT), a straightforward yet effective training strategy that improves attention on critical tokens while reinforcing their influence on model predictions. Extensive experiments across four tasks, under both context window scaling and long-context alignment settings, demonstrate the superiority of CDT. Notably, when trained with CDT, an open-source 8B model can achieve performance (50.92) comparable to GPT-4o (51.00).
>
---
#### [replaced 047] Constraint Satisfaction Approaches to Wordle: Novel Heuristics and Cross-Lexicon Validation
- **分类: cs.CL; cs.AI; 68T20, 90C27; I.2.8; I.2.3; G.1.6**

- **链接: [http://arxiv.org/pdf/2510.02855v3](http://arxiv.org/pdf/2510.02855v3)**

> **作者:** Jahidul Arafat; Fariha Tasmin; Sanjaya Poudel
>
> **备注:** 35 pages, 14 figures, 10 tables. Open-source implementation with 91% test coverage available at https://github.com/jahidul-arafat/constraint_satisfaction_wordle_arxiv_preprint
>
> **摘要:** Wordle presents an algorithmically rich testbed for constraint satisfaction problem (CSP) solving. While existing solvers rely on information-theoretic entropy maximization or frequency-based heuristics without formal constraint treatment, we present the first comprehensive CSP formulation of Wordle with novel constraint-aware solving strategies. We introduce CSP-Aware Entropy, computing information gain after constraint propagation rather than on raw candidate sets, and a Probabilistic CSP framework integrating Bayesian word-frequency priors with logical constraints. Through evaluation on 2,315 English words, CSP-Aware Entropy achieves 3.54 average guesses with 99.9% success rate, a statistically significant 1.7% improvement over Forward Checking (t=-4.82, p<0.001, Cohen's d=0.07) with 46% faster runtime (12.9ms versus 23.7ms per guess). Under 10% noise, CSP-aware approaches maintain 5.3 percentage point advantages (29.0% versus 23.7%, p=0.041), while Probabilistic CSP achieves 100% success across all noise levels (0-20%) through constraint recovery mechanisms. Cross-lexicon validation on 500 Spanish words demonstrates 88% success with zero language-specific tuning, validating that core CSP principles transfer across languages despite an 11.2 percentage point gap from linguistic differences (p<0.001, Fisher's exact test). Our open-source implementation with 34 unit tests achieving 91% code coverage provides reproducible infrastructure for CSP research. The combination of formal CSP treatment, constraint-aware heuristics, probabilistic-logical integration, robustness analysis, and cross-lexicon validation establishes new performance benchmarks demonstrating that principled constraint satisfaction techniques outperform classical information-theoretic and learning-based approaches for structured puzzle-solving domains.
>
---
#### [replaced 048] Towards Global Retrieval Augmented Generation: A Benchmark for Corpus-Level Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.26205v2](http://arxiv.org/pdf/2510.26205v2)**

> **作者:** Qi Luo; Xiaonan Li; Tingshuo Fan; Xinchi Chen; Xipeng Qiu
>
> **摘要:** Retrieval-augmented generation (RAG) has emerged as a leading approach to reducing hallucinations in large language models (LLMs). Current RAG evaluation benchmarks primarily focus on what we call local RAG: retrieving relevant chunks from a small subset of documents to answer queries that require only localized understanding within specific text chunks. However, many real-world applications require a fundamentally different capability -- global RAG -- which involves aggregating and analyzing information across entire document collections to derive corpus-level insights (for example, "What are the top 10 most cited papers in 2023?"). In this paper, we introduce GlobalQA -- the first benchmark specifically designed to evaluate global RAG capabilities, covering four core task types: counting, extremum queries, sorting, and top-k extraction. Through systematic evaluation across different models and baselines, we find that existing RAG methods perform poorly on global tasks, with the strongest baseline achieving only 1.51 F1 score. To address these challenges, we propose GlobalRAG, a multi-tool collaborative framework that preserves structural coherence through chunk-level retrieval, incorporates LLM-driven intelligent filters to eliminate noisy documents, and integrates aggregation modules for precise symbolic computation. On the Qwen2.5-14B model, GlobalRAG achieves 6.63 F1 compared to the strongest baseline's 1.51 F1, validating the effectiveness of our method.
>
---
#### [replaced 049] Linear-Time Demonstration Selection for In-Context Learning via Gradient Estimation
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.19999v2](http://arxiv.org/pdf/2508.19999v2)**

> **作者:** Ziniu Zhang; Zhenshuo Zhang; Dongyue Li; Lu Wang; Jennifer Dy; Hongyang R. Zhang
>
> **备注:** 19 pages. EMNLP'25
>
> **摘要:** This paper introduces an algorithm to select demonstration examples for in-context learning of a query set. Given a set of $n$ examples, how can we quickly select $k$ out of $n$ to best serve as the conditioning for downstream inference? This problem has broad applications in prompt tuning and chain-of-thought reasoning. Since model weights remain fixed during in-context learning, previous work has sought to design methods based on the similarity of token embeddings. This work proposes a new approach based on gradients of the output taken in the input embedding space. Our approach estimates model outputs through a first-order approximation using the gradients. Then, we apply this estimation to multiple randomly sampled subsets. Finally, we aggregate the sampled subset outcomes to form an influence score for each demonstration, and select $k$ most relevant examples. This procedure only requires pre-computing model outputs and gradients once, resulting in a linear-time algorithm relative to model and training set sizes. Extensive experiments across various models and datasets validate the efficiency of our approach. We show that the gradient estimation procedure yields approximations of full inference with less than ${1}\%$ error across six datasets. This allows us to scale up subset selection that would otherwise run full inference by up to ${37.7}\times$ on models with up to $34$ billion parameters, and outperform existing selection methods based on input embeddings by ${11}\%$ on average.
>
---
#### [replaced 050] LongRM: Revealing and Unlocking the Context Boundary of Reward Modeling
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.06915v2](http://arxiv.org/pdf/2510.06915v2)**

> **作者:** Zecheng Tang; Baibei Ji; Quantong Qiu; Haitian Wang; Xiaobo Liang; Juntao Li; Min Zhang
>
> **摘要:** Reward model (RM) plays a pivotal role in aligning large language model (LLM) with human preferences. As real-world applications increasingly involve long history trajectories, e.g., LLM agent, it becomes indispensable to evaluate whether a model's responses are not only high-quality but also grounded in and consistent with the provided context. Yet, current RMs remain confined to short-context settings and primarily focus on response-level attributes (e.g., safety or helpfulness), while largely neglecting the critical dimension of long context-response consistency. In this work, we introduce Long-RewardBench, a benchmark specifically designed for long-context RM evaluation, featuring both Pairwise Comparison and Best-of-N tasks. Our preliminary study reveals that even state-of-the-art generative RMs exhibit significant fragility in long-context scenarios, failing to maintain context-aware preference judgments. Motivated by the analysis of failure patterns observed in model outputs, we propose a general multi-stage training strategy that effectively scales arbitrary models into robust Long-context RMs (LongRMs). Experiments show that our approach not only substantially improves performance on long-context evaluation but also preserves strong short-context capability. Notably, our 8B LongRM outperforms much larger 70B-scale baselines and matches the performance of the proprietary Gemini 2.5 Pro model.
>
---
#### [replaced 051] Unseen from Seen: Rewriting Observation-Instruction Using Foundation Models for Augmenting Vision-Language Navigation
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.18065v3](http://arxiv.org/pdf/2503.18065v3)**

> **作者:** Ziming Wei; Bingqian Lin; Yunshuang Nie; Jiaqi Chen; Shikui Ma; Hang Xu; Xiaodan Liang
>
> **备注:** Accepted by IEEE Transactions on Neural Networks and Learning Systems
>
> **摘要:** Data scarcity is a long-standing challenge in the Vision-Language Navigation (VLN) field, which extremely hinders the generalization of agents to unseen environments. Previous works primarily rely on additional simulator data or web-collected images/videos to improve the generalization. However, the simulator environments still face limited diversity, and the web-collected data often requires extensive labor to remove the noise. In this paper, we propose a Rewriting-driven AugMentation (RAM) paradigm for VLN, which directly creates the unseen observation-instruction pairs via rewriting human-annotated training data. Benefiting from our rewriting mechanism, new observation-instruction pairs can be obtained in both simulator-free and labor-saving manners to promote generalization. Specifically, we first introduce Object-Enriched Observation Rewriting, where we combine Vision-Language Models (VLMs) and Large Language Models (LLMs) to derive rewritten object-enriched scene descriptions, enabling observation synthesis with diverse objects and spatial layouts via Text-to-Image Generation Models (T2IMs). Then, we propose Observation-Contrast Instruction Rewriting, which generates observation-aligned rewritten instructions by requiring LLMs to reason the difference between original and new observations. We further develop a mixing-then-focusing training strategy with a random observation cropping scheme, effectively enhancing data distribution diversity while suppressing augmentation data noise during training. Experiments on both the discrete environments (R2R, REVERIE, and R4R datasets) and continuous environments (R2R-CE dataset) show the superior performance and impressive generalization ability of our method. Code is available at https://github.com/SaDil13/VLN-RAM.
>
---
#### [replaced 052] How Teachers Can Use Large Language Models and Bloom's Taxonomy to Create Educational Quizzes
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2401.05914v2](http://arxiv.org/pdf/2401.05914v2)**

> **作者:** Sabina Elkins; Ekaterina Kochmar; Jackie C. K. Cheung; Iulian Serban
>
> **备注:** 8 pages, 8 figures. Accepted to the main track of the EAAI-24: The 14th Symposium on Educational Advances in Artificial Intelligence
>
> **摘要:** Question generation (QG) is a natural language processing task with an abundance of potential benefits and use cases in the educational domain. In order for this potential to be realized, QG systems must be designed and validated with pedagogical needs in mind. However, little research has assessed or designed QG approaches with the input from real teachers or students. This paper applies a large language model-based QG approach where questions are generated with learning goals derived from Bloom's taxonomy. The automatically generated questions are used in multiple experiments designed to assess how teachers use them in practice. The results demonstrate that teachers prefer to write quizzes with automatically generated questions, and that such quizzes have no loss in quality compared to handwritten versions. Further, several metrics indicate that automatically generated questions can even improve the quality of the quizzes created, showing the promise for large scale use of QG in the classroom setting.
>
---
#### [replaced 053] Twilight: Adaptive Attention Sparsity with Hierarchical Top-$p$ Pruning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.02770v5](http://arxiv.org/pdf/2502.02770v5)**

> **作者:** Chaofan Lin; Jiaming Tang; Shuo Yang; Hanshuo Wang; Tian Tang; Boyu Tian; Ion Stoica; Song Han; Mingyu Gao
>
> **备注:** To appear on NeurIPS 2025 (spotlight)
>
> **摘要:** Leveraging attention sparsity to accelerate long-context large language models (LLMs) has been a hot research topic. However, current algorithms such as sparse attention or key-value (KV) cache compression tend to use a fixed budget, which presents a significant challenge during deployment because it fails to account for the dynamic nature of real-world scenarios, where the optimal balance between accuracy and efficiency can vary greatly. In this paper, we find that borrowing top-$p$ sampling (nucleus sampling) to sparse attention can surprisingly achieve adaptive budgeting. Based on this, we propose Twilight, a framework to bring adaptive sparsity to any existing sparse attention algorithm without sacrificing their accuracy. Empirical results show that Twilight can adaptively prune at most 98% of redundant tokens, leading to $15.4\times$ acceleration in self-attention operations and $3.9\times$ acceleration in end-to-end per token latency in long context LLM decoding.
>
---
#### [replaced 054] Exploration of Summarization by Generative Language Models for Automated Scoring of Long Essays
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.22830v3](http://arxiv.org/pdf/2510.22830v3)**

> **作者:** Haowei Hua; Hong Jiao; Xinyi Wang
>
> **备注:** 19 pages, 5 Tables 7 Figures, Presentation at Artificial Intelligence in Measurement and Education Conference (AIME-Con)
>
> **摘要:** BERT and its variants are extensively explored for automated scoring. However, a limit of 512 tokens for these encoder-based models showed the deficiency in automated scoring of long essays. Thus, this research explores generative language models for automated scoring of long essays via summarization and prompting. The results revealed great improvement of scoring accuracy with QWK increased from 0.822 to 0.8878 for the Learning Agency Lab Automated Essay Scoring 2.0 dataset.
>
---
#### [replaced 055] Leveraging Hierarchical Organization for Medical Multi-document Summarization
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2510.23104v2](http://arxiv.org/pdf/2510.23104v2)**

> **作者:** Yi-Li Hsu; Katelyn X. Mei; Lucy Lu Wang
>
> **摘要:** Medical multi-document summarization (MDS) is a complex task that requires effectively managing cross-document relationships. This paper investigates whether incorporating hierarchical structures in the inputs of MDS can improve a model's ability to organize and contextualize information across documents compared to traditional flat summarization methods. We investigate two ways of incorporating hierarchical organization across three large language models (LLMs), and conduct comprehensive evaluations of the resulting summaries using automated metrics, model-based metrics, and domain expert evaluation of preference, understandability, clarity, complexity, relevance, coverage, factuality, and coherence. Our results show that human experts prefer model-generated summaries over human-written summaries. Hierarchical approaches generally preserve factuality, coverage, and coherence of information, while also increasing human preference for summaries. Additionally, we examine whether simulated judgments from GPT-4 align with human judgments, finding higher agreement along more objective evaluation facets. Our findings demonstrate that hierarchical structures can improve the clarity of medical summaries generated by models while maintaining content coverage, providing a practical way to improve human preference for generated summaries.
>
---
#### [replaced 056] Rethinking the Relationship between the Power Law and Hierarchical Structures
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.04984v2](http://arxiv.org/pdf/2505.04984v2)**

> **作者:** Kai Nakaishi; Ryo Yoshida; Kohei Kajikawa; Koji Hukushima; Yohei Oseki
>
> **备注:** 18 pages, 14 figures
>
> **摘要:** Statistical analysis of corpora provides an approach to quantitatively investigate natural languages. This approach has revealed that several power laws consistently emerge across different corpora and languages, suggesting universal mechanisms underlying languages. Particularly, the power-law decay of correlation has been interpreted as evidence for underlying hierarchical structures in syntax, semantics, and discourse. This perspective has also been extended to child speeches and animal signals. However, the argument supporting this interpretation has not been empirically tested in natural languages. To address this problem, the present study examines the validity of the argument for syntactic structures. Specifically, we test whether the statistical properties of parse trees align with the assumptions in the argument. Using English and Japanese corpora, we analyze the mutual information, deviations from probabilistic context-free grammars (PCFGs), and other properties in natural language parse trees, as well as in the PCFG that approximates these parse trees. Our results indicate that the assumptions do not hold for syntactic structures and that it is difficult to apply the proposed argument to child speeches and animal signals, highlighting the need to reconsider the relationship between the power law and hierarchical structures.
>
---
#### [replaced 057] SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20411v2](http://arxiv.org/pdf/2505.20411v2)**

> **作者:** Ibragim Badertdinov; Alexander Golubev; Maksim Nekrashevich; Anton Shevtsov; Simon Karasik; Andrei Andriushchenko; Maria Trofimova; Daria Litvintseva; Boris Yangel
>
> **备注:** Dataset: https://huggingface.co/datasets/nebius/SWE-rebench, SWE-rebench leaderboard https://swe-rebench.com NeurIPS 2025
>
> **摘要:** LLM-based agents have shown promising capabilities in a growing range of software engineering (SWE) tasks. However, advancing this field faces two critical challenges. First, high-quality training data is scarce, especially data that reflects real-world SWE scenarios, where agents must interact with development environments, execute code and adapt behavior based on the outcomes of their actions. Existing datasets are either limited to one-shot code generation or comprise small, manually curated collections of interactive tasks, lacking both scale and diversity. Second, the lack of fresh interactive SWE tasks affects evaluation of rapidly improving models, as static benchmarks quickly become outdated due to contamination issues. To address these limitations, we introduce a novel, automated, and scalable pipeline to continuously extract real-world interactive SWE tasks from diverse GitHub repositories. Using this pipeline, we construct SWE-rebench, a public dataset comprising over 21,000 interactive Python-based SWE tasks, suitable for reinforcement learning of SWE agents at scale. Additionally, we use continuous supply of fresh tasks collected using SWE-rebench methodology to build a contamination-free benchmark for agentic software engineering. We compare results of various LLMs on this benchmark to results on SWE-bench Verified and show that performance of some language models might be inflated due to contamination issues.
>
---
#### [replaced 058] SpecDiff-2: Scaling Diffusion Drafter Alignment For Faster Speculative Decoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2511.00606v2](http://arxiv.org/pdf/2511.00606v2)**

> **作者:** Jameson Sandler; Jacob K. Christopher; Thomas Hartvigsen; Ferdinando Fioretto
>
> **摘要:** Speculative decoding has become the standard approach for accelerating Large Language Model (LLM) inference. It exploits a lossless draft-then-verify procedure to circumvent the latency of autoregressive decoding, achieving impressive speed-ups. Yet, current speculative decoding approaches remain limited by two fundamental bottlenecks: (1) the autoregressive dependency during drafting which limits parallelism, and (2) frequent rejections of draft tokens caused by misalignment between the draft and verify models. This paper proposes SpecDiff-2, a novel framework to jointly address these two bottlenecks. It leverages discrete diffusion as a non-autoregressive drafter to address bottleneck (1) and develops novel techniques to calibrate discrete diffusion drafters with autoregressive verifiers, addressing bottleneck (2). Experimental results across a comprehensive benchmark suite show that SpecDiff-2 achieves a new state-of-the-art across reasoning, coding, and mathematical benchmarks, improving tokens-per-second by up to an average of +55% over previous baselines and obtaining up to 5.5x average speed-up over standard decoding, without any loss of accuracy.
>
---
#### [replaced 059] ValueCompass: A Framework for Measuring Contextual Value Alignment Between Human and LLMs
- **分类: cs.HC; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.09586v3](http://arxiv.org/pdf/2409.09586v3)**

> **作者:** Hua Shen; Tiffany Knearem; Reshmi Ghosh; Yu-Ju Yang; Nicholas Clark; Tanushree Mitra; Yun Huang
>
> **摘要:** As AI systems become more advanced, ensuring their alignment with a diverse range of individuals and societal values becomes increasingly critical. But how can we capture fundamental human values and assess the degree to which AI systems align with them? We introduce ValueCompass, a framework of fundamental values, grounded in psychological theory and a systematic review, to identify and evaluate human-AI alignment. We apply ValueCompass to measure the value alignment of humans and large language models (LLMs) across four real-world scenarios: collaborative writing, education, public sectors, and healthcare. Our findings reveal concerning misalignments between humans and LLMs, such as humans frequently endorse values like "National Security" which were largely rejected by LLMs. We also observe that values differ across scenarios, highlighting the need for context-aware AI alignment strategies. This work provides valuable insights into the design space of human-AI alignment, laying the foundations for developing AI systems that responsibly reflect societal values and ethics.
>
---
#### [replaced 060] DiscoTrack: A Multilingual LLM Benchmark for Discourse Tracking
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.17013v2](http://arxiv.org/pdf/2510.17013v2)**

> **作者:** Lanni Bu; Lauren Levin; Amir Zeldes
>
> **摘要:** Recent LLM benchmarks have tested models on a range of phenomena, but are still focused primarily on natural language understanding for extraction of explicit information, such as QA or summarization, with responses often tar- geting information from individual sentences. We are still lacking more challenging, and im- portantly also multilingual, benchmarks focus- ing on implicit information and pragmatic infer- ences across larger documents in the context of discourse tracking: integrating and aggregating information across sentences, paragraphs and multiple speaker utterances. To this end, we present DiscoTrack, an LLM benchmark target- ing a range of tasks across 12 languages and four levels of discourse understanding: salience recognition, entity tracking, discourse relations and bridging inference. Our evaluation shows that these tasks remain challenging, even for state-of-the-art models.
>
---
#### [replaced 061] Tool-to-Agent Retrieval: Bridging Tools and Agents for Scalable LLM Multi-Agent Systems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2511.01854v2](http://arxiv.org/pdf/2511.01854v2)**

> **作者:** Elias Lumer; Faheem Nizar; Anmol Gulati; Pradeep Honaganahalli Basavaraju; Vamse Kumar Subbiah
>
> **摘要:** Recent advances in LLM Multi-Agent Systems enable scalable orchestration of sub-agents, each coordinating hundreds or thousands of tools or Model Context Protocol (MCP) servers. However, existing retrieval methods typically match queries against coarse agent-level descriptions before routing, which obscures fine-grained tool functionality and often results in suboptimal agent selection. We introduce Tool-to-Agent Retrieval, a unified framework that embeds both tools and their parent agents in a shared vector space and connects them through metadata relationships. By explicitly representing tool capabilities and traversing metadata to the agent level, Tool-to-Agent Retrieval enables granular tool-level or agent-level retrieval, ensuring that agents and their underlying tools or MCP servers are equally represented without the context dilution that arises from chunking many tools together. Evaluating Tool-to-Agent Retrieval across eight embedding models, our approach achieves consistent improvements of 19.4% in Recall@5 and 17.7% in nDCG@5 over previous state-of-the-art agent retrievers on the LiveMCPBench benchmark.
>
---
