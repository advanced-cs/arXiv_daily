# 计算机与社会 cs.CY

- **最新发布 15 篇**

- **更新 4 篇**

## 最新发布

#### [new 001] Surgeons Are Indian Males and Speech Therapists Are White Females: Auditing Biases in Vision-Language Models for Healthcare Professionals
- **分类: cs.CY; cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言模型（VLM）中的偏见审计任务，旨在解决医疗领域中AI模型对医疗职业与人口属性间刻板印象的反映问题。作者构建了职业分类体系和职业感知提示集，评估模型在多个医疗角色上的性别、种族等人口统计偏见，揭示了关键领域AI应用的潜在公平性和合规风险。**

- **链接: [http://arxiv.org/pdf/2510.06280v1](http://arxiv.org/pdf/2510.06280v1)**

> **作者:** Zohaib Hasan Siddiqui; Dayam Nadeem; Mohammad Masudur Rahman; Mohammad Nadeem; Shahab Saquib Sohail; Beenish Moalla Chaudhry
>
> **摘要:** Vision language models (VLMs), such as CLIP and OpenCLIP, can encode and reflect stereotypical associations between medical professions and demographic attributes learned from web-scale data. We present an evaluation protocol for healthcare settings that quantifies associated biases and assesses their operational risk. Our methodology (i) defines a taxonomy spanning clinicians and allied healthcare roles (e.g., surgeon, cardiologist, dentist, nurse, pharmacist, technician), (ii) curates a profession-aware prompt suite to probe model behavior, and (iii) benchmarks demographic skew against a balanced face corpus. Empirically, we observe consistent demographic biases across multiple roles and vision models. Our work highlights the importance of bias identification in critical domains such as healthcare as AI-enabled hiring and workforce analytics can have downstream implications for equity, compliance, and patient trust.
>
---
#### [new 002] Asking For It: Question-Answering for Predicting Rule Infractions in Online Content Moderation
- **分类: cs.CY; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于内容审核任务，旨在解决在线社区规则多样且执行不一致的问题。作者提出了ModQ框架，通过问答方式识别评论中最相关的违规规则，支持不同社区和规则的泛化，提升审核的透明性与自动化水平。**

- **链接: [http://arxiv.org/pdf/2510.06350v1](http://arxiv.org/pdf/2510.06350v1)**

> **作者:** Mattia Samory; Diana Pamfile; Andrew To; Shruti Phadke
>
> **备注:** Accepted at ICWSM 2026
>
> **摘要:** Online communities rely on a mix of platform policies and community-authored rules to define acceptable behavior and maintain order. However, these rules vary widely across communities, evolve over time, and are enforced inconsistently, posing challenges for transparency, governance, and automation. In this paper, we model the relationship between rules and their enforcement at scale, introducing ModQ, a novel question-answering framework for rule-sensitive content moderation. Unlike prior classification or generation-based approaches, ModQ conditions on the full set of community rules at inference time and identifies which rule best applies to a given comment. We implement two model variants - extractive and multiple-choice QA - and train them on large-scale datasets from Reddit and Lemmy, the latter of which we construct from publicly available moderation logs and rule descriptions. Both models outperform state-of-the-art baselines in identifying moderation-relevant rule violations, while remaining lightweight and interpretable. Notably, ModQ models generalize effectively to unseen communities and rules, supporting low-resource moderation settings and dynamic governance environments.
>
---
#### [new 003] Towards an Efficient, Customizable, and Accessible AI Tutor
- **分类: cs.CY**

- **简介: 该论文旨在开发一种高效、可定制且易于访问的AI导师系统。任务是解决大型语言模型在低资源环境中因高计算需求导致的可用性问题。作者提出了一种离线的检索增强生成（RAG）流程，结合小型语言模型与高效检索机制，应用于生物学教育内容。他们分析了小型模型在处理扩展上下文时的挑战，并提出改进方向，为受限环境下的教育工具部署奠定了基础。**

- **链接: [http://arxiv.org/pdf/2510.06255v1](http://arxiv.org/pdf/2510.06255v1)**

> **作者:** Juan Segundo Hevia; Facundo Arredondo; Vishesh Kumar
>
> **备注:** AAAI - iRAISE 2025
>
> **摘要:** The integration of large language models (LLMs) into education offers significant potential to enhance accessibility and engagement, yet their high computational demands limit usability in low-resource settings, exacerbating educational inequities. To address this, we propose an offline Retrieval-Augmented Generation (RAG) pipeline that pairs a small language model (SLM) with a robust retrieval mechanism, enabling factual, contextually relevant responses without internet connectivity. We evaluate the efficacy of this pipeline using domain-specific educational content, focusing on biology coursework. Our analysis highlights key challenges: smaller models, such as SmolLM, struggle to effectively leverage extended contexts provided by the RAG pipeline, particularly when noisy or irrelevant chunks are included. To improve performance, we propose exploring advanced chunking techniques, alternative small or quantized versions of larger models, and moving beyond traditional metrics like MMLU to a holistic evaluation framework assessing free-form response. This work demonstrates the feasibility of deploying AI tutors in constrained environments, laying the groundwork for equitable, offline, and device-based educational tools.
>
---
#### [new 004] Beyond Static Knowledge Messengers: Towards Adaptive, Fair, and Scalable Federated Learning for Medical AI
- **分类: cs.CY; cs.LG; 68T05, 62P10, 68T07; I.2.11; K.4.1; J.3**

- **简介: 论文提出自适应公平联邦学习（AFFL）框架，解决医疗AI中隐私保护协作学习的公平性、收敛速度和扩展性问题。通过动态知识传递、公平聚合和课程加速三项创新，实现高效、公平、合规的多机构协作，支持百级机构规模，并设计了MedFedBench评估基准。**

- **链接: [http://arxiv.org/pdf/2510.06259v1](http://arxiv.org/pdf/2510.06259v1)**

> **作者:** Jahidul Arafat; Fariha Tasmin; Sanjaya Poudel; Ahsan Habib Tareq; Iftekhar Haider
>
> **备注:** 20 pages, 4 figures, 14 tables. Proposes Adaptive Fair Federated Learning (AFFL) algorithm and MedFedBench benchmark suite for healthcare federated learning
>
> **摘要:** Medical AI faces challenges in privacy-preserving collaborative learning while ensuring fairness across heterogeneous healthcare institutions. Current federated learning approaches suffer from static architectures, slow convergence (45-73 rounds), fairness gaps marginalizing smaller institutions, and scalability constraints (15-client limit). We propose Adaptive Fair Federated Learning (AFFL) through three innovations: (1) Adaptive Knowledge Messengers dynamically scaling capacity based on heterogeneity and task complexity, (2) Fairness-Aware Distillation using influence-weighted aggregation, and (3) Curriculum-Guided Acceleration reducing rounds by 60-70%. Our theoretical analysis provides convergence guarantees with epsilon-fairness bounds, achieving O(T^{-1/2}) + O(H_max/T^{3/4}) rates. Projected results show 55-75% communication reduction, 56-68% fairness improvement, 34-46% energy savings, and 100+ institution support. The framework enables multi-modal integration across imaging, genomics, EHR, and sensor data while maintaining HIPAA/GDPR compliance. We propose MedFedBench benchmark suite for standardized evaluation across six healthcare dimensions: convergence efficiency, institutional fairness, privacy preservation, multi-modal integration, scalability, and clinical deployment readiness. Economic projections indicate 400-800% ROI for rural hospitals and 15-25% performance gains for academic centers. This work presents a seven-question research agenda, 24-month implementation roadmap, and pathways toward democratizing healthcare AI.
>
---
#### [new 005] LLM-Driven Rubric-Based Assessment of Algebraic Competence in Multi-Stage Block Coding Tasks with Design and Field Evaluation
- **分类: cs.CY; cs.AI**

- **简介: 论文提出了一种基于大语言模型（LLM）的量规评估框架，用于在线教育平台中多阶段代数编程任务的学生表现评估。任务是测量学生的代数能力和问题解决过程，解决传统评估方法难以捕捉认知深度的问题。工作包括设计与量规对齐的题目、开发LLM评估系统，并通过实地研究验证其有效性。**

- **链接: [http://arxiv.org/pdf/2510.06253v1](http://arxiv.org/pdf/2510.06253v1)**

> **作者:** Yong Oh Lee; Byeonghun Bang; Sejun Oh
>
> **摘要:** As online education platforms continue to expand, there is a growing need for assessment methods that not only measure answer accuracy but also capture the depth of students' cognitive processes in alignment with curriculum objectives. This study proposes and evaluates a rubric-based assessment framework powered by a large language model (LLM) for measuring algebraic competence, real-world-context block coding tasks. The problem set, designed by mathematics education experts, aligns each problem segment with five predefined rubric dimensions, enabling the LLM to assess both correctness and quality of students' problem-solving processes. The system was implemented on an online platform that records all intermediate responses and employs the LLM for rubric-aligned achievement evaluation. To examine the practical effectiveness of the proposed framework, we conducted a field study involving 42 middle school students engaged in multi-stage quadratic equation tasks with block coding. The study integrated learner self-assessments and expert ratings to benchmark the system's outputs. The LLM-based rubric evaluation showed strong agreement with expert judgments and consistently produced rubric-aligned, process-oriented feedback. These results demonstrate both the validity and scalability of incorporating LLM-driven rubric assessment into online mathematics and STEM education platforms.
>
---
#### [new 006] The Limits of Goal-Setting Theory in LLM-Driven Assessment
- **分类: cs.CY; cs.AI**

- **简介: 该论文研究大语言模型（LLM）驱动评估中目标设定理论的适用性。任务是分析LLM是否像人类一样因目标明确而表现更一致。作者通过让ChatGPT用不同详细程度的提示评估学生作业，测量其一致性。结果发现提示越具体，性能未提升，挑战了LLM与人类行为一致的假设。**

- **链接: [http://arxiv.org/pdf/2510.06997v1](http://arxiv.org/pdf/2510.06997v1)**

> **作者:** Mrityunjay Kumar
>
> **备注:** Accepted at T4E 2025 for poster
>
> **摘要:** Many users interact with AI tools like ChatGPT using a mental model that treats the system as human-like, which we call Model H. According to goal-setting theory, increased specificity in goals should reduce performance variance. If Model H holds, then prompting a chatbot with more detailed instructions should lead to more consistent evaluation behavior. This paper tests that assumption through a controlled experiment in which ChatGPT evaluated 29 student submissions using four prompts with increasing specificity. We measured consistency using intra-rater reliability (Cohen's Kappa) across repeated runs. Contrary to expectations, performance did not improve consistently with increased prompt specificity, and performance variance remained largely unchanged. These findings challenge the assumption that LLMs behave like human evaluators and highlight the need for greater robustness and improved input integration in future model development.
>
---
#### [new 007] Early Results from Teaching Modelling for Software Comprehension in New-Hire Onboarding
- **分类: cs.CY; cs.SE**

- **简介: 该论文属于教育干预任务，旨在解决新员工理解复杂软件系统能力不足的问题。研究在入职培训中引入系统思维与LTS建模课程，评估其对35名新员工的影响，发现课程对基础较弱者提升明显，整体效果有限，但反馈良好，建议差异化培训路径。**

- **链接: [http://arxiv.org/pdf/2510.07010v1](http://arxiv.org/pdf/2510.07010v1)**

> **作者:** Mrityunjay Kumar; Venkatesh Choppella
>
> **备注:** Accepted at COMPUTE 2025 as short paper
>
> **摘要:** Working effectively with large, existing software systems requires strong comprehension skills, yet most graduates enter the industry with little preparation for this challenge. We report early results from a pilot intervention integrated into a SaaS company's onboarding program: a five-session course introducing systems thinking and Labelled Transition System (LTS) modelling. Participants articulated their understanding of product behaviour using a structured template and completed matched pre- and post-assessments. Of 35 new hires, 31 provided paired records for analysis. Across the full cohort, gains were small and not statistically significant. However, participants below the median on the pre-test improved by 15 percentage points on average (statistically significant), while those above the median regressed slightly (not statistically significant). Course feedback indicated high engagement and perceived applicability. These results suggest that short, modelling-focused onboarding interventions can accelerate comprehension for less-prepared new hires. At the same time, they point to the need for differentiated pathways for stronger participants, and to the potential for companies to adopt such interventions at scale as a low-cost complement to existing onboarding.
>
---
#### [new 008] Technical Overview of Safe3Step (S3S): Power Ratings and quality wins for selecting at-large teams to the NCAA Division I Men's Lacrosse Championship
- **分类: cs.CY**

- **简介: 论文提出了一种名为Safe3Step（S3S）的新系统，用于选拔NCAA男子一级长曲棍球锦标赛的外卡球队。该系统旨在改进现有的RPI排名体系，通过三步法：评估球队实力、依据胜负质量评分排序、并根据胜负关系调整排名，从而更公平地选择参赛队伍。**

- **链接: [http://arxiv.org/pdf/2510.06279v1](http://arxiv.org/pdf/2510.06279v1)**

> **作者:** Lawrence Feldman; Matthew Bomparola
>
> **备注:** Whitepaper; 5 pages
>
> **摘要:** This document describes a system for selecting teams to the NCAA Men's Division I Lacrosse Championship Tournament called "Safe3Step" (S3S) that was developed in conversation with the NCAA Lacrosse Selection Criteria and Ranking Committee (SCR) with the objective of improving on the Ratings Percentage Index (RPI). S3S employs three steps that: 1) evaluate the strength of each team based on score data, 2) award S3S points to each team based on the quality of its wins and losses, ranking teams accordingly, and 3) examine each pair of teams with adjacent rankings, swapping ranks if the lower-ranked team has a better head-to-head record against the higher-ranked team. Safe3Step is not entirely new, but it improves on other "quality win" methods by using Power Ratings to identify team strengths, respecting head-to-head records, and adhering to standards of simplicity, transparency, and objectivity. Empirical analysis is left to future work.
>
---
#### [new 009] On the false election between regulation and innovation. Ideas for regulation through the responsible use of artificial intelligence in research and education.[Spanish version]
- **分类: cs.CY; cs.AI; J.4; K.3; K.4; K.5**

- **简介: 该论文探讨在人工智能发展中的监管与创新关系，旨在解决如何优先保护基本权利、推动负责任创新、以及建立国际合作机制的问题。论文通过讨论会回答三个核心问题，反思其对教育和研究的意义。**

- **链接: [http://arxiv.org/pdf/2510.07268v1](http://arxiv.org/pdf/2510.07268v1)**

> **作者:** Pompeu Casanovas
>
> **备注:** 20 pages, in Spanish language, 1 figure, 1 table, AI Hub-CSIC / EduCaixa, Escuela de Verano, Auditorio CaixaForum, Zaragoza, Spain, 4 July 2025
>
> **摘要:** This short essay is a reworking of the answers offered by the author at the Debate Session of the AIHUB (CSIC) and EduCaixa Summer School, organized by Marta Garcia-Matos and Lissette Lemus, and coordinated by Albert Sabater (OEIAC, UG), with the participation of Vanina Martinez-Posse (IIIA-CSIC), Eulalia Soler (Eurecat) and Pompeu Casanovas (IIIA-CSIC) on July 4th 2025. Albert Sabater posed three questions: (1) How can regulatory frameworks priori-tise the protection of fundamental rights (privacy, non-discrimination, autonomy, etc.) in the development of AI, without falling into the false dichotomy between regulation and innova-tion? (2) Given the risks of AI (bias, mass surveillance, manipulation), what examples of regu-lations or policies have demonstrated that it is possible to foster responsible innovation, putting the public interest before profitability, without giving in to competitive pressure from actors such as China or the US? (3) In a scenario where the US prioritizes flexibility, what mecha-nisms could ensure that international cooperation in AI does not become a race to the bottom in rights, but rather a global standard of accountability? The article attempts to answer these three questions and concludes with some reflections on the relevance of the answers for education and research.
>
---
#### [new 010] A Mixed-Methods Analysis of Repression and Mobilization in Bangladesh's July Revolution Using Machine Learning and Statistical Modeling
- **分类: stat.AP; cs.CY; cs.LG; stat.ME; stat.ML**

- **简介: 该论文属于社会科学研究任务，旨在分析孟加拉国2024年七月革命中政府镇压与民众动员之间的复杂关系。论文探讨了为何国家暴力镇压反而促使学生领导的起义取得成功。研究结合定性叙事与定量分析，利用机器学习和统计模型验证镇压引发反作用的机制，揭示了镇压的“反扑效应”由道德冲击触发并通过暴力影像传播加速的非线性过程。**

- **链接: [http://arxiv.org/pdf/2510.06264v1](http://arxiv.org/pdf/2510.06264v1)**

> **作者:** Md. Saiful Bari Siddiqui; Anupam Debashis Roy
>
> **备注:** Submitted to Social Forces. Final version may vary from this preprint
>
> **摘要:** The 2024 July Revolution in Bangladesh represents a landmark event in the study of civil resistance. This study investigates the central paradox of the success of this student-led civilian uprising: how state violence, intended to quell dissent, ultimately fueled the movement's victory. We employ a mixed-methods approach. First, we develop a qualitative narrative of the conflict's timeline to generate specific, testable hypotheses. Then, using a disaggregated, event-level dataset, we employ a multi-method quantitative analysis to dissect the complex relationship between repression and mobilisation. We provide a framework to analyse explosive modern uprisings like the July Revolution. Initial pooled regression models highlight the crucial role of protest momentum in sustaining the movement. To isolate causal effects, we specify a Two-Way Fixed Effects panel model, which provides robust evidence for a direct and statistically significant local suppression backfire effect. Our Vector Autoregression (VAR) analysis provides clear visual evidence of an immediate, nationwide mobilisation in response to increased lethal violence. We further demonstrate that this effect was non-linear. A structural break analysis reveals that the backfire dynamic was statistically insignificant in the conflict's early phase but was triggered by the catalytic moral shock of the first wave of lethal violence, and its visuals circulated around July 16th. A complementary machine learning analysis (XGBoost, out-of-sample R$^{2}$=0.65) corroborates this from a predictive standpoint, identifying "excessive force against protesters" as the single most dominant predictor of nationwide escalation. We conclude that the July Revolution was driven by a contingent, non-linear backfire, triggered by specific catalytic moral shocks and accelerated by the viral reaction to the visual spectacle of state brutality.
>
---
#### [new 011] Unpacking Discourses on Childbirth and Parenthood in Popular Social Media Platforms Across China, Japan, and South Korea
- **分类: cs.SI; cs.CY**

- **简介: 该论文分析中、日、韩社交媒体上关于生育和育儿的讨论，探究网络话语对生育意愿的影响。使用BERTopic和QWen模型对大量评论进行主题与情感分析，发现关注焦点如育儿成本、子女效用及个人主义存在差异，且平台评论呈现反生育倾向。研究旨在理解低生育率地区家庭价值观的网络传播。**

- **链接: [http://arxiv.org/pdf/2510.06788v1](http://arxiv.org/pdf/2510.06788v1)**

> **作者:** Zheng Wei; Yunqi Li; Yucheng He; Yuelu Li; Xian Xu; Huamin Qu; Pan Hui; Muzhi Zhou
>
> **备注:** Accepted for publication at The International AAAI Conference on Web and Social Media (ICWSM 2026)
>
> **摘要:** Social media use has been shown to be associated with low fertility desires. However, we know little about the discourses surrounding childbirth and parenthood that people consume online. We analyze 219,127 comments on 668 short videos related to reproduction and parenthood from Douyin and Tiktok in China, South Korea, and Japan, a region famous for its extremely low fertility level, to examine the topics and sentiment expressed online. BERTopic model is used to assist thematic analysis, and a large language model QWen is applied to label sentiment. We find that comments focus on childrearing costs in all countries, utility of children, particularly in Japan and South Korea, and individualism, primarily in China. Comments from Douyin exhibit the strongest anti-natalist sentiments, while the Japanese and Korean comments are more neutral. Short video characteristics, such as their stances or account type, significantly influence the responses, alongside regional socioeconomic indicators, including GDP, urbanization, and population sex ratio. This work provides one of the first comprehensive analyses of online discourses on family formation via popular algorithm-fed video sharing platforms in regions experiencing low fertility rates, making a valuable contribution to our understanding of the spread of family values online.
>
---
#### [new 012] Emotionally Vulnerable Subtype of Internet Gaming Disorder: Measuring and Exploring the Pathology of Problematic Generative AI Use
- **分类: cs.HC; cs.AI; cs.CY**

- **简介: 该论文属于心理学与行为研究任务，旨在解决生成式AI使用成瘾的测量与分类问题。研究开发并验证了PUGenAIS-9量表，确认其反映情绪脆弱型网络游戏障碍特征，支持其跨文化和性别稳定性，提出应基于ICD模型重新思考数字成瘾研究，以避免过度病理化。**

- **链接: [http://arxiv.org/pdf/2510.06908v1](http://arxiv.org/pdf/2510.06908v1)**

> **作者:** Haocan Sun; Di Wua; Weizi Liu; Guoming Yua; Mike Yao
>
> **备注:** 27 pages, 5 figures, 5 tables
>
> **摘要:** Concerns over the potential over-pathologization of generative AI (GenAI) use and the lack of conceptual clarity surrounding GenAI addiction call for empirical tools and theoretical refinement. This study developed and validated the PUGenAIS-9 (Problematic Use of Generative Artificial Intelligence Scale-9 items) and examined whether PUGenAIS reflects addiction-like patterns under the Internet Gaming Disorder (IGD) framework. Using samples from China and the United States (N = 1,508), we conducted confirmatory factor analysis and identified a robust 31-item structure across nine IGD-based dimensions. We then derived the PUGenAIS-9 by selecting the highest-loading items from each dimension and validated its structure in an independent sample (N = 1,426). Measurement invariance tests confirmed its stability across nationality and gender. Person-centered (latent profile analysis) and variable-centered (network analysis) approaches found that PUGenAIS matches the traits of the emotionally vulnerable subtype of IGD, not the competence-based kind. These results support using PUGenAIS-9 to identify problematic GenAI use and show the need to rethink digital addiction with an ICD (infrastructures, content, and device) model. This keeps addiction research responsive to new media while avoiding over-pathologizing.
>
---
#### [new 013] Exploring Human-AI Collaboration Using Mental Models of Early Adopters of Multi-Agent Generative AI Tools
- **分类: cs.HC; cs.AI; cs.CY**

- **简介: 该论文研究早期采用者如何理解多智能体生成式AI工具，探索人类与AI协作的机制、协作动态及透明性问题。通过13名微软开发者的访谈，发现他们将多智能体系统视为角色分工明确的“团队”，提出需解决错误传播、交互透明性及协作控制等问题，旨在为CSCW领域提供设计协作机制与透明策略的参考。**

- **链接: [http://arxiv.org/pdf/2510.06224v1](http://arxiv.org/pdf/2510.06224v1)**

> **作者:** Suchismita Naik; Austin L. Toombs; Amanda Snellinger; Scott Saponas; Amanda K. Hall
>
> **备注:** 19 pages, 1 table, 2 figures
>
> **摘要:** With recent advancements in multi-agent generative AI (Gen AI), technology organizations like Microsoft are adopting these complex tools, redefining AI agents as active collaborators in complex workflows rather than as passive tools. In this study, we investigated how early adopters and developers conceptualize multi-agent Gen AI tools, focusing on how they understand human-AI collaboration mechanisms, general collaboration dynamics, and transparency in the context of AI tools. We conducted semi-structured interviews with 13 developers, all early adopters of multi-agent Gen AI technology who work at Microsoft. Our findings revealed that these early adopters conceptualize multi-agent systems as "teams" of specialized role-based and task-based agents, such as assistants or reviewers, structured similar to human collaboration models and ranging from AI-dominant to AI-assisted, user-controlled interactions. We identified key challenges, including error propagation, unpredictable and unproductive agent loop behavior, and the need for clear communication to mitigate the layered transparency issues. Early adopters' perspectives about the role of transparency underscored its importance as a way to build trust, verify and trace errors, and prevent misuse, errors, and leaks. The insights and design considerations we present contribute to CSCW research about collaborative mechanisms with capabilities ranging from AI-dominant to AI-assisted interactions, transparency and oversight strategies in human-agent and agent-agent interactions, and how humans make sense of these multi-agent systems as dynamic, role-diverse collaborators which are customizable for diverse needs and workflows. We conclude with future research directions that extend CSCW approaches to the design of inter-agent and human mediation interactions.
>
---
#### [new 014] Am I Productive? Exploring the Experience of Remote Workers with Task Management Tools
- **分类: cs.HC; cs.CY**

- **简介: 论文探讨远程办公者使用任务管理工具的效果，旨在解决远程办公中生产力提升的问题。研究通过两周的混合方法日记研究和半结构化访谈，发现数字任务管理工具相较于纸笔未显著提升感知生产力，强调需加强工具的个性化设计。**

- **链接: [http://arxiv.org/pdf/2510.06816v1](http://arxiv.org/pdf/2510.06816v1)**

> **作者:** Russell Beale
>
> **摘要:** As the world continues to change, more and more knowledge workers are embracing remote work. Yet this comes with its challenges for their productivity, and while many Task Management applications promise to improve the productivity of remote workers, it remains unclear how effective they are. Based on existing frameworks, this study investigated the productivity needs and challenges of remote knowledge workers and how they use Task Management tools. The research was conducted through a 2-week long, mixed-methods diary study and semi-structured interview. Perceptions of productivity, task management tool use and productivity challenges were observed. The findings show that using a digital Task Management application made no significant difference to using pen and paper for improving perceived productivity of remote workers and discuss the need for better personalization of Task Management applications.
>
---
#### [new 015] Machines in the Crowd? Measuring the Footprint of Machine-Generated Text on Reddit
- **分类: cs.SI; cs.CL; cs.CY; physics.soc-ph**

- **简介: 该论文研究机器生成文本（MGT）在Reddit上的影响，属于自然语言处理与社交数据分析任务。旨在了解MGT在社交平台的分布特征及其与人类文本的对比。通过大规模检测与分析，发现MGT虽整体占比较低，但在特定社区和时间段可达9%，且具有独特语言风格并能获得相近甚至更高的用户参与度。**

- **链接: [http://arxiv.org/pdf/2510.07226v1](http://arxiv.org/pdf/2510.07226v1)**

> **作者:** Lucio La Cava; Luca Maria Aiello; Andrea Tagarelli
>
> **摘要:** Generative Artificial Intelligence is reshaping online communication by enabling large-scale production of Machine-Generated Text (MGT) at low cost. While its presence is rapidly growing across the Web, little is known about how MGT integrates into social media environments. In this paper, we present the first large-scale characterization of MGT on Reddit. Using a state-of-the-art statistical method for detection of MGT, we analyze over two years of activity (2022-2024) across 51 subreddits representative of Reddit's main community types such as information seeking, social support, and discussion. We study the concentration of MGT across communities and over time, and compared MGT to human-authored text in terms of social signals it expresses and engagement it receives. Our very conservative estimate of MGT prevalence indicates that synthetic text is marginally present on Reddit, but it can reach peaks of up to 9% in some communities in some months. MGT is unevenly distributed across communities, more prevalent in subreddits focused on technical knowledge and social support, and often concentrated in the activity of a small fraction of users. MGT also conveys distinct social signals of warmth and status giving typical of language of AI assistants. Despite these stylistic differences, MGT achieves engagement levels comparable than human-authored content and in a few cases even higher, suggesting that AI-generated text is becoming an organic component of online social discourse. This work offers the first perspective on the MGT footprint on Reddit, paving the way for new investigations involving platform governance, detection strategies, and community dynamics.
>
---
## 更新

#### [replaced 001] A risk model and analysis method for the psychological safety of human and autonomous vehicles interaction
- **分类: cs.HC; cs.CY**

- **链接: [http://arxiv.org/pdf/2411.05732v3](http://arxiv.org/pdf/2411.05732v3)**

> **作者:** Yandika Sirgabsou; Benjamin Hardin; François Leblanc; Efi Raili; Pericle Salvini; David Jackson; Marina Jirotka; Lars Kunze
>
> **摘要:** The rapid advancement of artificial intelligence and autonomous driving technologies has significantly propelled the development of autonomous vehicles (AVs). However, psychological barriers continue to impede widespread AV adoption, despite technological progress. This paper addresses the critical yet often overlooked aspect of psychological safety in AV design and operation. While traditional safety standards focus primarily on physical safety, this paper emphasizes the psychological implications that arise from human interactions with autonomous vehicles, highlighting the importance of trust and perceived risk as significant factors influencing user acceptance. The paper makes a methodological proposal, a framework for addressing AVs psychological safety consisting of three key contributions. First, it introduces a definition of psychological safety in AVs context. Secondly, it proposes a risk model for identifying and assessing AVs psychological hazards and risks. PsySIL (Psychological Safety Integrity Level), a classification of AV psychological risk levels is developed. Thirdly, an adapted system-theoretic analysis method for AVs psychological safety is proposed. The paper illustrates the application of the framework for assessing potential psychological hazards using a scenario involving a family's experience with an autonomous vehicle, pioneering a systems approach towards evaluating situations that could lead to psychological harm. By establishing a framework that incorporates psychological safety alongside physical safety, the paper contributes to the broader discourse on the safe deployment of autonomous vehicle, aiming to guide future developments in user-centred design and regulatory practices, while acknowledging the limitations brought by the application of the proposals on a rather simple but pedagogical illustrative example.
>
---
#### [replaced 002] Beyond Monoliths: Expert Orchestration for More Capable, Democratic, and Safe Language Models
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2506.00051v2](http://arxiv.org/pdf/2506.00051v2)**

> **作者:** Philip Quirke; Narmeen Oozeer; Chaithanya Bandi; Amir Abdullah; Jason Hoelscher-Obermaier; Jeff M. Phillips; Joshua Greaves; Clement Neo; Michael Lan; Fazl Barez; Shriyash Upadhyay
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** This position paper argues that the prevailing trajectory toward ever larger, more expensive generalist foundation models controlled by a handful of companies limits innovation and constrains progress. We challenge this approach by advocating for an "Expert Orchestration" (EO) framework as a superior alternative that democratizes LLM advancement. Our proposed framework intelligently selects from many existing models based on query requirements and decomposition, focusing on identifying what models do well rather than how they work internally. Independent "judge" models assess various models' capabilities across dimensions that matter to users, while "router" systems direct queries to the most appropriate specialists within an approved set. This approach delivers superior performance by leveraging targeted expertise rather than forcing costly generalist models to address all user requirements. EO enhances transparency, control, alignment, performance, safety and democratic participation through intelligent model selection.
>
---
#### [replaced 003] Epistemic Diversity and Knowledge Collapse in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.04226v3](http://arxiv.org/pdf/2510.04226v3)**

> **作者:** Dustin Wright; Sarah Masud; Jared Moore; Srishti Yadav; Maria Antoniak; Chan Young Park; Isabelle Augenstein
>
> **备注:** 16 pages; 8 figures, 4 tables; v2 changelog: Fixed the modeling for table 3, random effect is the model version; v3 changelog: Fixed minor formatting issues in tables 2 and 3;
>
> **摘要:** Large language models (LLMs) tend to generate lexically, semantically, and stylistically homogenous texts. This poses a risk of knowledge collapse, where homogenous LLMs mediate a shrinking in the range of accessible information over time. Existing works on homogenization are limited by a focus on closed-ended multiple-choice setups or fuzzy semantic features, and do not look at trends across time and cultural contexts. To overcome this, we present a new methodology to measure epistemic diversity, i.e., variation in real-world claims in LLM outputs, which we use to perform a broad empirical study of LLM knowledge collapse. We test 27 LLMs, 155 topics covering 12 countries, and 200 prompt variations sourced from real user chats. For the topics in our study, we show that while newer models tend to generate more diverse claims, nearly all models are less epistemically diverse than a basic web search. We find that model size has a negative impact on epistemic diversity, while retrieval-augmented generation (RAG) has a positive impact, though the improvement from RAG varies by the cultural context. Finally, compared to a traditional knowledge source (Wikipedia), we find that country-specific claims reflect the English language more than the local one, highlighting a gap in epistemic representation
>
---
#### [replaced 004] Who Pays for Fairness? Rethinking Recourse under Social Burden
- **分类: cs.LG; cs.CY**

- **链接: [http://arxiv.org/pdf/2509.04128v2](http://arxiv.org/pdf/2509.04128v2)**

> **作者:** Ainhize Barrainkua; Giovanni De Toni; Jose Antonio Lozano; Novi Quadrianto
>
> **摘要:** Machine learning based predictions are increasingly used in sensitive decision-making applications that directly affect our lives. This has led to extensive research into ensuring the fairness of classifiers. Beyond just fair classification, emerging legislation now mandates that when a classifier delivers a negative decision, it must also offer actionable steps an individual can take to reverse that outcome. This concept is known as algorithmic recourse. Nevertheless, many researchers have expressed concerns about the fairness guarantees within the recourse process itself. In this work, we provide a holistic theoretical characterization of unfairness in algorithmic recourse, formally linking fairness guarantees in recourse and classification, and highlighting limitations of the standard equal cost paradigm. We then introduce a novel fairness framework based on social burden, along with a practical algorithm (MISOB), broadly applicable under real-world conditions. Empirical results on real-world datasets show that MISOB reduces the social burden across all groups without compromising overall classifier accuracy.
>
---
