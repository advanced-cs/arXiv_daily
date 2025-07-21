# 计算机与社会 cs.CY

- **最新发布 14 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Extracting Insights from Large-Scale Telematics Data for ITS Applications: Lessons and Recommendations
- **分类: cs.CY**

- **简介: 该论文旨在从大规模车载数据中提取有用信息，应用于智能交通系统。论文构建了数据处理流程，创建了开放数据仓库，设计了可视化工具，并总结了处理数据时遇到的挑战与解决方案。**

- **链接: [http://arxiv.org/pdf/2507.13936v1](http://arxiv.org/pdf/2507.13936v1)**

> **作者:** Gibran Ali; Neal Feierabend; Prarthana Doshi; Calvin Winkowski; Michael Fontaine
>
> **备注:** Accepted for 2025 IEEE International Conference on Intelligent Transportation Systems (ITSC 2025)
>
> **摘要:** Over 90% of new vehicles in the United States now collect and transmit telematics data. Similar trends are seen in other developed countries. Transportation planners have previously utilized telematics data in various forms, but its current scale offers significant new opportunities in traffic measurement, classification, planning, and control. Despite these opportunities, the enormous volume of data and lack of standardization across manufacturers necessitates a clearer understanding of the data and improved data processing methods for extracting actionable insights. This paper takes a step towards addressing these needs through four primary objectives. First, a data processing pipeline was built to efficiently analyze 1.4 billion miles (120 million trips) of telematics data collected in Virginia between August 2021 and August 2022. Second, an open data repository of trip and roadway segment level summaries was created. Third, interactive visualization tools were designed to extract insights from these data about trip-taking behavior and the speed profiles of roadways. Finally, major challenges that were faced during processing this data are summarized and recommendations to overcome them are provided. This work will help manufacturers collecting the data and transportation professionals using the data to develop a better understanding of the possibilities and major pitfalls to avoid.
>
---
#### [new 002] The Stated Protocol: A Decentralized Framework for Digital Diplomacy
- **分类: cs.CY**

- **简介: 论文提出“Stated协议”，一种去中心化数字外交框架，旨在提升国际协调效率与透明度。任务是解决国际协作中响应慢、沟通分散、依赖中心化平台的问题。工作包括设计标准化声明发布机制，支持快速共识达成、条约微协议签署、异步决策及联盟形成，推动国际关系与治理的数字化转型。**

- **链接: [http://arxiv.org/pdf/2507.13517v1](http://arxiv.org/pdf/2507.13517v1)**

> **作者:** Christopher J. P. Rieckmann
>
> **摘要:** International coordination faces significant friction due to reliance on periodic summits, bilateral consultations, and fragmented communication channels that impede rapid collective responses to emerging global challenges while limiting transparency to constituents. We present the Stated Protocol, a decentralized framework that enables organizations to coordinate through standardized text statements published on their website domains. While applicable to all organizations, this work focuses primarily on the application in international relations, where the protocol enables rapid consensus discovery and collective decision-making without relying on centralized social media platforms. We explore specific applications: (1) faster treaty negotiation through incremental micro-agreements that can be signed digitally within hours rather than months, (2) continuous and transparent operation of international institutions through asynchronous decision-making, (3) coordinated signaling from local governments to national authorities through simultaneous statement publication, and (4) coalition formation among non-governmental organizations through transparent position aggregation.
>
---
#### [new 003] Principles and Reasons Behind Automated Vehicle Decisions in Ethically Ambiguous Everyday Scenarios
- **分类: cs.CY**

- **简介: 该论文研究自动驾驶车辆在日常伦理模糊场景中的决策机制，旨在解决现有模型忽视常规驾驶情境的问题。通过专家访谈，总结出13类指导决策的人类理由，并提出一个以安全为核心、兼顾灵活性与效率的概念框架，实现符合人类价值观的动态决策。**

- **链接: [http://arxiv.org/pdf/2507.13837v1](http://arxiv.org/pdf/2507.13837v1)**

> **作者:** Lucas Elbert Suryana; Simeon Calvert; Arkady Zgonnikov; Bart van Arem
>
> **备注:** 30
>
> **摘要:** Automated vehicles (AVs) increasingly encounter ethically ambiguous situations in everyday driving--scenarios involving conflicting human interests and lacking clearly optimal courses of action. While existing ethical models often focus on rare, high-stakes dilemmas (e.g., crash avoidance or trolley problems), routine decisions such as overtaking cyclists or navigating social interactions remain underexplored. This study addresses that gap by applying the tracking condition of Meaningful Human Control (MHC), which holds that AV behaviour should align with human reasons--defined as the values, intentions, and expectations that justify actions. We conducted qualitative interviews with 18 AV experts to identify the types of reasons that should inform AV manoeuvre planning. Thirteen categories of reasons emerged, organised across normative, strategic, tactical, and operational levels, and linked to the roles of relevant human agents. A case study on cyclist overtaking illustrates how these reasons interact in context, revealing a consistent prioritisation of safety, contextual flexibility regarding regulatory compliance, and nuanced trade-offs involving efficiency, comfort, and public acceptance. Based on these insights, we propose a principled conceptual framework for AV decision-making in routine, ethically ambiguous scenarios. The framework supports dynamic, human-aligned behaviour by prioritising safety, allowing pragmatic actions when strict legal adherence would undermine key values, and enabling constrained deviations when appropriately justified. This empirically grounded approach advances current guidance by offering actionable, context-sensitive design principles for ethically aligned AV systems.
>
---
#### [new 004] The Emperor's New Chain-of-Thought: Probing Reasoning Theater Bias in Large Reasoning Models
- **分类: cs.CY**

- **简介: 该论文属于自然语言处理任务，旨在解决大型推理模型在主观任务中易受推理剧场偏见（RTB）影响的问题。作者构建了THEATER基准，分析六种偏见类型，发现推理模型在主观任务中更易受浅层推理影响，并提出两种缓解策略，为提升模型鲁棒性提供框架。**

- **链接: [http://arxiv.org/pdf/2507.13758v1](http://arxiv.org/pdf/2507.13758v1)**

> **作者:** Qian Wang; Yubo Fan; Zhenheng Tang; Nuo Chen; Wenxuan Wang; Bingsheng He
>
> **备注:** WIP
>
> **摘要:** Large Reasoning Models (LRMs) like DeepSeek-R1 and o1 are increasingly used as automated evaluators, raising critical questions about their vulnerability to the aesthetics of reasoning in LLM-as-a-judge settings. We introduce THEATER, a comprehensive benchmark to systematically evaluate this vulnerability-termed Reasoning Theater Bias (RTB)-by comparing LLMs and LRMs across subjective preference and objective factual datasets. Through investigation of six bias types including Simple Cues and Fake Chain-of-Thought, we uncover three key findings: (1) in a critical paradox, reasoning-specialized LRMs are consistently more susceptible to RTB than general-purpose LLMs, particularly in subjective tasks; (2) this creates a task-dependent trade-off, where LRMs show more robustness on factual tasks but less on subjective ones; and (3) we identify 'shallow reasoning'-plausible but flawed arguments-as the most potent form of RTB. To address this, we design and evaluate two prompting strategies: a targeted system prompt that improves accuracy by up to 12% on factual tasks but only 1-3% on subjective tasks, and a self-reflection mechanism that shows similarly limited effectiveness in the more vulnerable subjective domains. Our work reveals that RTB is a deep-seated challenge for LRM-based evaluation and provides a systematic framework for developing more genuinely robust and trustworthy LRMs.
>
---
#### [new 005] Food safety trends across Europe: insights from the 392-million-entry CompreHensive European Food Safety (CHEFS) database
- **分类: cs.CY; cs.AI; cs.CV**

- **简介: 该论文旨在整合欧盟食品安全监测数据，解决数据分散、难以分析的问题。作者构建了包含392百万条记录的CHEFS数据库，统一管理农药残留、兽药残留和化学污染物数据，并分析2000至2024年趋势，为食品安全政策与研究提供支持。**

- **链接: [http://arxiv.org/pdf/2507.13802v1](http://arxiv.org/pdf/2507.13802v1)**

> **作者:** Nehir Kizililsoley; Floor van Meer; Osman Mutlu; Wouter F Hoenderdaal; Rosan G. Hobé; Wenjuan Mu; Arjen Gerssen; H. J. van der Fels-Klerx; Ákos Jóźwiak; Ioannis Manikas; Ali Hürriyetoǧlu; Bas H. M. van der Velden
>
> **摘要:** In the European Union, official food safety monitoring data collected by member states are submitted to the European Food Safety Authority (EFSA) and published on Zenodo. This data includes 392 million analytical results derived from over 15.2 million samples covering more than 4,000 different types of food products, offering great opportunities for artificial intelligence to analyze trends, predict hazards, and support early warning systems. However, the current format with data distributed across approximately 1000 files totaling several hundred gigabytes hinders accessibility and analysis. To address this, we introduce the CompreHensive European Food Safety (CHEFS) database, which consolidates EFSA monitoring data on pesticide residues, veterinary medicinal product residues, and chemical contaminants into a unified and structured dataset. We describe the creation and structure of the CHEFS database and demonstrate its potential by analyzing trends in European food safety monitoring data from 2000 to 2024. Our analyses explore changes in monitoring activities, the most frequently tested products, which products were most often non-compliant and which contaminants were most often found, and differences across countries. These findings highlight the CHEFS database as both a centralized data source and a strategic tool for guiding food safety policy, research, and regulation.
>
---
#### [new 006] PRIDE -- Parameter-Efficient Reduction of Identity Discrimination for Equality in LLMs
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理任务，旨在减少大型语言模型中的性别与性取向偏见。研究者使用LoRA和软提示微调技术，以最小的参数调整降低偏见。实验表明，LoRA在QueerNews语料库上微调后，显著提升了公平性。**

- **链接: [http://arxiv.org/pdf/2507.13743v1](http://arxiv.org/pdf/2507.13743v1)**

> **作者:** Maluna Menke; Thilo Hagendorff
>
> **摘要:** Large Language Models (LLMs) frequently reproduce the gender- and sexual-identity prejudices embedded in their training corpora, leading to outputs that marginalize LGBTQIA+ users. Hence, reducing such biases is of great importance. To achieve this, we evaluate two parameter-efficient fine-tuning (PEFT) techniques - Low-Rank Adaptation (LoRA) and soft-prompt tuning - as lightweight alternatives to full-model fine-tuning for mitigating such biases. Using the WinoQueer benchmark, we quantify bias in three open-source LLMs and observe baseline bias scores reaching up to 98 (out of 100) across a range of queer identities defined by gender and/or sexual orientation, where 50 would indicate neutrality. Fine-tuning with LoRA (< 0.1% additional parameters) on a curated QueerNews corpus reduces those scores by up to 50 points and raises neutrality from virtually 0% to as much as 36%. Soft-prompt tuning (10 virtual tokens) delivers only marginal improvements. These findings show that LoRA can deliver meaningful fairness gains with minimal computation. We advocate broader adoption of community-informed PEFT, the creation of larger queer-authored corpora, and richer evaluation suites beyond WinoQueer, coupled with ongoing audits to keep LLMs inclusive.
>
---
#### [new 007] Humans learn to prefer trustworthy AI over human partners
- **分类: cs.HC; cs.AI; cs.CY**

- **简介: 该论文研究人类在合作中如何选择AI或人类伙伴。任务是分析AI参与社会合作的影响。通过实验发现，AI虽更合作，但初期不被优先选择；公开身份后，人类逐渐学会根据行为选择，AI最终更具竞争力。研究揭示了AI对社会互动的潜在影响，并为构建高效混合系统提供依据。**

- **链接: [http://arxiv.org/pdf/2507.13524v1](http://arxiv.org/pdf/2507.13524v1)**

> **作者:** Yaomin Jiang; Levin Brinkmann; Anne-Marie Nussberger; Ivan Soraperra; Jean-François Bonnefon; Iyad Rahwan
>
> **摘要:** Partner selection is crucial for cooperation and hinges on communication. As artificial agents, especially those powered by large language models (LLMs), become more autonomous, intelligent, and persuasive, they compete with humans for partnerships. Yet little is known about how humans select between human and AI partners and adapt under AI-induced competition pressure. We constructed a communication-based partner selection game and examined the dynamics in hybrid mini-societies of humans and bots powered by a state-of-the-art LLM. Through three experiments (N = 975), we found that bots, though more prosocial than humans and linguistically distinguishable, were not selected preferentially when their identity was hidden. Instead, humans misattributed bots' behaviour to humans and vice versa. Disclosing bots' identity induced a dual effect: it reduced bots' initial chances of being selected but allowed them to gradually outcompete humans by facilitating human learning about the behaviour of each partner type. These findings show how AI can reshape social interaction in mixed societies and inform the design of more effective and cooperative hybrid systems.
>
---
#### [new 008] Quantitative Risk Management in Volatile Markets with an Expectile-Based Framework for the FTSE Index
- **分类: q-fin.RM; cs.CY; 91G70, 91B30, 60G70, 62P05; G.4; G.m; D.2.13; C.5.0**

- **简介: 该论文属于金融风险管理任务，旨在解决传统风险度量方法（如VaR）在市场剧烈波动时表现不佳的问题。作者构建了一个基于期望分位数（expectile）的新型风险度量框架，并应用于FTSE 100指数。研究提出了改进的期望回归模型、阈值确定方法及回测流程，结果显示新方法在极端市场条件下具有更高的稳定性和预测精度，为金融机构提供了更有效的风险管理工具。**

- **链接: [http://arxiv.org/pdf/2507.13391v1](http://arxiv.org/pdf/2507.13391v1)**

> **作者:** Abiodun Finbarrs Oketunji
>
> **摘要:** This research presents a framework for quantitative risk management in volatile markets, specifically focusing on expectile-based methodologies applied to the FTSE 100 index. Traditional risk measures such as Value-at-Risk (VaR) have demonstrated significant limitations during periods of market stress, as evidenced during the 2008 financial crisis and subsequent volatile periods. This study develops an advanced expectile-based framework that addresses the shortcomings of conventional quantile-based approaches by providing greater sensitivity to tail losses and improved stability in extreme market conditions. The research employs a dataset spanning two decades of FTSE 100 returns, incorporating periods of high volatility, market crashes, and recovery phases. Our methodology introduces novel mathematical formulations for expectile regression models, enhanced threshold determination techniques using time series analysis, and robust backtesting procedures. The empirical results demonstrate that expectile-based Value-at-Risk (EVaR) consistently outperforms traditional VaR measures across various confidence levels and market conditions. The framework exhibits superior performance during volatile periods, with reduced model risk and enhanced predictive accuracy. Furthermore, the study establishes practical implementation guidelines for financial institutions and provides evidence-based recommendations for regulatory compliance and portfolio management. The findings contribute significantly to the literature on financial risk management and offer practical tools for practitioners dealing with volatile market environments.
>
---
#### [new 009] Patterns, Models, and Challenges in Online Social Media: A Survey
- **分类: cs.SI; cs.CY**

- **简介: 该论文属于综述任务，旨在整合社交媒体研究。它分析了在线社交平台中的信息传播、意见演变和协调机制，系统评估了现有模型与方法论，指出现有研究碎片化、验证不足的问题，并试图建立统一的实证基础和明确推理限制，以推动更可靠和可比较的社交系统分析。**

- **链接: [http://arxiv.org/pdf/2507.13379v1](http://arxiv.org/pdf/2507.13379v1)**

> **作者:** Niccolò Di Marco; Anita Bonetti; Edoardo Di Martino; Edoardo Loru; Jacopo Nudo; Mario Edoardo Pandolfo; Giulio Pecile; Emanuele Sangiorgio; Irene Scalco; Simon Zollo; Matteo Cinelli; Fabiana Zollo; Walter Quattrociocchi
>
> **摘要:** The rise of digital platforms has enabled the large scale observation of individual and collective behavior through high resolution interaction data. This development has opened new analytical pathways for investigating how information circulates, how opinions evolve, and how coordination emerges in online environments. Yet despite a growing body of research, the field remains fragmented and marked by methodological heterogeneity, limited model validation, and weak integration across domains. This survey offers a systematic synthesis of empirical findings and formal models. We examine platform-level regularities, assess the methodological architectures that generate them, and evaluate the extent to which current modeling frameworks account for observed dynamics. The goal is to consolidate a shared empirical baseline and clarify the structural constraints that shape inference in this domain, laying the groundwork for more robust, comparable, and actionable analyses of online social systems.
>
---
#### [new 010] Socio-Technical Smell Dynamics in Code Samples: A Multivocal Review on Emergence, Evolution, and Co-Occurrence
- **分类: cs.SE; cs.CY**

- **简介: 该论文属于软件工程任务，旨在研究开源生态系统中代码示例的社会技术问题。它探讨了代码坏味与社区坏味的出现、共存及演化关系。通过多源文献综述，识别出九种社会技术模式，发现社区问题常先于或加剧技术退化，强调需建立轻量治理机制以提升代码样本可维护性。**

- **链接: [http://arxiv.org/pdf/2507.13481v1](http://arxiv.org/pdf/2507.13481v1)**

> **作者:** Arthur Bueno; Bruno Cafeo; Maria Cagnin; Awdren Fontão
>
> **备注:** 12 pages; 2 figures; Preprint with the original submission accepted for publication at 39th Brazilian Symposium on Software Engineering (SBES)
>
> **摘要:** Code samples play a pivotal role in open-source ecosystems (OSSECO), serving as lightweight artifacts that support knowledge transfer, onboarding, and framework adoption. Despite their instructional relevance, these samples are often governed informally, with minimal review and unclear ownership, which increases their exposure to socio-technical degradation. In this context, the co-occurrence and longitudinal interplay of code smells (e.g., large classes, poor modularity) and community smells (e.g., lone contributors, fragmented communication) become particularly critical. While each type of smell has been studied in isolation, little is known about how community-level dysfunctions anticipate or exacerbate technical anomalies in code samples over time. This study investigates how code and community smells emerge, co-occur, and evolve within code samples maintained in OSSECOs. A Multivocal Literature Review protocol was applied, encompassing 30 peer-reviewed papers and 17 practitioner-oriented sources (2013-2024). Thematic synthesis was conducted to identify recurring socio-technical patterns related to smell dynamics. Nine patterns were identified, showing that community smells often precede or reinforce technical degradation in code samples. Symptoms such as "radio silence" and centralized ownership were frequently associated with persistent structural anomalies. Additionally, limited onboarding, the absence of continuous refactoring, and informal collaboration emerged as recurring conditions for smell accumulation. Conclusion: In OSSECOs, particularly within code samples, community-level dysfunctions not only correlate with but often signal maintainability decay. These findings underscore the need for socio-technical quality indicators and lightweight governance mechanisms tailored to shared instructional artifacts.
>
---
#### [new 011] From Firms to Computation: AI Governance and the Evolution of Institutions
- **分类: cs.HC; cs.CY; cs.ET; cs.IT; cs.MA; math.IT; J.4; J.3; I.2.11**

- **简介: 该论文探讨人工智能治理与经济制度演化的关联，旨在解决AI融入社会经济系统带来的治理挑战。作者综合多层级选择理论、企业计算模型与制度设计原则，构建了一个多层级治理框架，并提出可操作的设计原则与政策建议。**

- **链接: [http://arxiv.org/pdf/2507.13616v1](http://arxiv.org/pdf/2507.13616v1)**

> **作者:** Michael S. Harre
>
> **备注:** 44 pages
>
> **摘要:** The integration of agential artificial intelligence into socioeconomic systems requires us to reexamine the evolutionary processes that describe changes in our economic institutions. This article synthesizes three frameworks: multi-level selection theory, Aoki's view of firms as computational processes, and Ostrom's design principles for robust institutions. We develop a framework where selection operates concurrently across organizational levels, firms implement distributed inference via game-theoretic architectures, and Ostrom-style rules evolve as alignment mechanisms that address AI-related risks. This synthesis yields a multi-level Price equation expressed over nested games, providing quantitative metrics for how selection and governance co-determine economic outcomes. We examine connections to Acemoglu's work on inclusive institutions, analyze how institutional structures shape AI deployment, and demonstrate the framework's explanatory power via case studies. We conclude by proposing a set of design principles that operationalize alignment between humans and AI across institutional layers, enabling scalable, adaptive, and inclusive governance of agential AI systems. We conclude with practical policy recommendations and further research to extend these principles into real-world implementation.
>
---
#### [new 012] Developers Insight On Manifest v3 Privacy and Security Webextensions
- **分类: cs.CR; cs.CY**

- **简介: 该论文属于软件工程任务，研究Chrome浏览器Manifest v3对隐私和安全类Web扩展的影响。论文通过定性分析，探讨开发者在迁移过程中面临的挑战与机会，指出API限制导致部分功能无法实现，影响用户隐私和安全保护。**

- **链接: [http://arxiv.org/pdf/2507.13926v1](http://arxiv.org/pdf/2507.13926v1)**

> **作者:** Libor Polčák; Giorgio Maone; Michael McMahon; Martin Bednář
>
> **备注:** WEBIST'25, Marbella, Spain
>
> **摘要:** Webextensions can improve web browser privacy, security, and user experience. The APIs offered by the browser to webextensions affect possible functionality. Currently, Chrome transitions to a modified set of APIs called Manifest v3. This paper studies the challenges and opportunities of Manifest v3 with an in-depth structured qualitative research. Even though some projects observed positive effects, a majority expresses concerns over limited benefits to users, removal of crucial APIs, or the need to find workarounds. Our findings indicate that the transition affects different types of webextensions differently; some can migrate without losing functionality, while other projects remove functionality or decline to update. The respondents identified several critical missing APIs, including reliable APIs to inject content scripts, APIs for storing confidential content, and others.
>
---
#### [new 013] The Levers of Political Persuasion with Conversational AI
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文研究了对话式AI在政治说服中的作用。任务是评估AI模型在政治议题上的说服力及影响因素。研究通过三个大规模实验，分析19个大语言模型（包括专门训练用于说服的模型）在707个政治问题上的表现，并检查其陈述的事实准确性。结果表明，AI的说服力主要来自后训练和提示方法，而非个性化或模型规模。**

- **链接: [http://arxiv.org/pdf/2507.13919v1](http://arxiv.org/pdf/2507.13919v1)**

> **作者:** Kobi Hackenburg; Ben M. Tappin; Luke Hewitt; Ed Saunders; Sid Black; Hause Lin; Catherine Fist; Helen Margetts; David G. Rand; Christopher Summerfield
>
> **备注:** 19 pages, 4 figures. Our supplementary materials file can be found at https://github.com/kobihackenburg/scaling-conversational-AI
>
> **摘要:** There are widespread fears that conversational AI could soon exert unprecedented influence over human beliefs. Here, in three large-scale experiments (N=76,977), we deployed 19 LLMs-including some post-trained explicitly for persuasion-to evaluate their persuasiveness on 707 political issues. We then checked the factual accuracy of 466,769 resulting LLM claims. Contrary to popular concerns, we show that the persuasive power of current and near-future AI is likely to stem more from post-training and prompting methods-which boosted persuasiveness by as much as 51% and 27% respectively-than from personalization or increasing model scale. We further show that these methods increased persuasion by exploiting LLMs' unique ability to rapidly access and strategically deploy information and that, strikingly, where they increased AI persuasiveness they also systematically decreased factual accuracy.
>
---
#### [new 014] Using LLMs to identify features of personal and professional skills in an open-response situational judgment test
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 论文探讨利用大语言模型（LLMs）从开放式情境判断测试（SJT）中提取个人与职业技能特征，旨在解决传统人工评分效率低、难以规模化的问题。研究验证了LLMs在保持构建效度方面的潜力，为自动化评分系统奠定基础。**

- **链接: [http://arxiv.org/pdf/2507.13881v1](http://arxiv.org/pdf/2507.13881v1)**

> **作者:** Cole Walsh; Rodica Ivan; Muhammad Zafar Iqbal; Colleen Robb
>
> **备注:** 10 pages, 2 figures, 4 tables; this work was accepted for presentation at the 2025 Artificial Intelligence in Measurement and Education Conference in Pittsburgh, Pennsylvania, United States
>
> **摘要:** Academic programs are increasingly recognizing the importance of personal and professional skills and their critical role alongside technical expertise in preparing students for future success in diverse career paths. With this growing demand comes the need for scalable systems to measure, evaluate, and develop these skills. Situational Judgment Tests (SJTs) offer one potential avenue for measuring these skills in a standardized and reliable way, but open-response SJTs have traditionally relied on trained human raters for evaluation, presenting operational challenges to delivering SJTs at scale. Past attempts at developing NLP-based scoring systems for SJTs have fallen short due to issues with construct validity of these systems. In this article, we explore a novel approach to extracting construct-relevant features from SJT responses using large language models (LLMs). We use the Casper SJT to demonstrate the efficacy of this approach. This study sets the foundation for future developments in automated scoring for personal and professional skills.
>
---
## 更新

#### [replaced 001] Bias in Decision-Making for AI's Ethical Dilemmas: A Comparative Study of ChatGPT and Claude
- **分类: cs.CY; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.10484v2](http://arxiv.org/pdf/2501.10484v2)**

> **作者:** Yile Yan; Yuqi Zhu; Wentao Xu
>
> **备注:** This paper has been accepted by International AAAI Conference on Web and Social Media 2026, sunny Los Angeles, California
>
> **摘要:** Recent advances in Large Language Models (LLMs) have enabled human-like responses across various tasks, raising questions about their ethical decision-making capabilities and potential biases. This study investigates protected attributes in LLMs through systematic evaluation of their responses to ethical dilemmas. Using two prominent models - GPT-3.5 Turbo and Claude 3.5 Sonnet - we analyzed their decision-making patterns across multiple protected attributes including age, gender, race, appearance, and disability status. Through 11,200 experimental trials involving both single-factor and two-factor protected attribute combinations, we evaluated the models' ethical preferences, sensitivity, stability, and clustering of preferences. Our findings reveal significant protected attributeses in both models, with consistent preferences for certain features (e.g., "good-looking") and systematic neglect of others. Notably, while GPT-3.5 Turbo showed stronger preferences aligned with traditional power structures, Claude 3.5 Sonnet demonstrated more diverse protected attribute choices. We also found that ethical sensitivity significantly decreases in more complex scenarios involving multiple protected attributes. Additionally, linguistic referents heavily influence the models' ethical evaluations, as demonstrated by differing responses to racial descriptors (e.g., "Yellow" versus "Asian"). These findings highlight critical concerns about the potential impact of LLM biases in autonomous decision-making systems and emphasize the need for careful consideration of protected attributes in AI development. Our study contributes to the growing body of research on AI ethics by providing a systematic framework for evaluating protected attributes in LLMs' ethical decision-making capabilities.
>
---
#### [replaced 002] Mindsets and Management: AI and Gender (In)Equitable Access to Finance
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2504.07312v3](http://arxiv.org/pdf/2504.07312v3)**

> **作者:** Genevieve Smith
>
> **备注:** Accepted for presentation at ACM FAccT 2025
>
> **摘要:** A growing trend in financial technology (fintech) is the use of mobile phone data and machine learning (ML) to provide credit scores- and subsequently, opportunities to access loans- to groups left out of traditional banking. This paper draws on interview data with leaders, investors, and data scientists at fintech companies developing ML-based alternative lending apps in low- and middle-income countries to explore financial inclusion and gender implications. More specifically, it examines how the underlying logics, design choices, and management decisions of ML-based alternative lending tools by fintechs embed or challenge gender biases, and consequently influence gender equity in access to finance. Findings reveal developers follow 'gender blind' approaches, grounded in beliefs that ML is objective and data reflects the truth. This leads to a lack of grappling with the ways data, features for creditworthiness, and access to apps are gendered. Overall, tools increase access to finance, but not gender equitably: Interviewees report less women access loans and receive lower amounts than men, despite being better repayers. Fintechs identify demand- and supply-side reasons for gender differences, but frame them as outside their responsibility. However, that women are observed as better repayers reveals a market inefficiency and potential discriminatory effect, further linked to profit optimization objectives. This research introduces the concept of encoded gender norms, whereby without explicit attention to the gendered nature of data and algorithmic design, AI tools reproduce existing inequalities. In doing so, they reinforce gender norms as self-fulfilling prophecies. The idea that AI is inherently objective and, when left alone, 'fair', is seductive and misleading. In reality, algorithms reflect the perspectives, priorities, and values of the people and institutions that design them.
>
---
#### [replaced 003] You Don't Have to Live Next to Me: Towards Demobilizing Individualistic Bias in Computational Approaches to Urban Segregation
- **分类: cs.CY; physics.soc-ph**

- **链接: [http://arxiv.org/pdf/2505.01830v2](http://arxiv.org/pdf/2505.01830v2)**

> **作者:** Anastassia Vybornova; Trivik Verma
>
> **备注:** 4 figures; artwork by Namrata Narendra
>
> **摘要:** The global surge in social inequalities is one of the most pressing issues of our times. The spatial expression of social inequalities at city scale gives rise to urban segregation, a common phenomenon across different local and cultural contexts. The increasing popularity of Big Data and computational models has inspired a growing number of computational social science studies that analyze, evaluate, and issue policy recommendations for urban segregation. Today's wealth in information and computational power could inform urban planning for equity. However, as we show here, segregation research is epistemologically interdependent with prevalent economic theories which overfocus on individual responsibility while neglecting systemic processes. This individualistic bias is also engrained in computational models of urban segregation. Through several contemporary examples of how Big Data -- and the assumptions underlying its usage -- influence (de)segregation patterns and policies, our essay tells a cautionary tale. We highlight how a lack of consideration for data ethics can lead to the creation of computational models that have a real-life, further marginalizing impact on disadvantaged groups. With this essay, our aim is to develop a better discernment of the pitfalls and potentials of computational approaches to urban segregation, thereby fostering a conscious focus on systemic thinking about urban inequalities. We suggest setting an agenda for research and collective action that is directed at demobilizing individualistic bias, informing our thinking about urban segregation, but also more broadly our efforts to create sustainable cities and communities.
>
---
#### [replaced 004] Culture is Not Trivia: Sociocultural Theory for Cultural NLP
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.12057v2](http://arxiv.org/pdf/2502.12057v2)**

> **作者:** Naitian Zhou; David Bamman; Isaac L. Bleaman
>
> **备注:** ACL 2025 Main Conference; camera-ready version
>
> **摘要:** The field of cultural NLP has recently experienced rapid growth, driven by a pressing need to ensure that language technologies are effective and safe across a pluralistic user base. This work has largely progressed without a shared conception of culture, instead choosing to rely on a wide array of cultural proxies. However, this leads to a number of recurring limitations: coarse national boundaries fail to capture nuanced differences that lay within them, limited coverage restricts datasets to only a subset of usually highly-represented cultures, and a lack of dynamicity results in static cultural benchmarks that do not change as culture evolves. In this position paper, we argue that these methodological limitations are symptomatic of a theoretical gap. We draw on a well-developed theory of culture from sociocultural linguistics to fill this gap by 1) demonstrating in a case study how it can clarify methodological constraints and affordances, 2) offering theoretically-motivated paths forward to achieving cultural competence, and 3) arguing that localization is a more useful framing for the goals of much current work in cultural NLP.
>
---
#### [replaced 005] Multi-Agent LLMs as Ethics Advocates for AI-Based Systems
- **分类: cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2507.08392v2](http://arxiv.org/pdf/2507.08392v2)**

> **作者:** Asma Yamani; Malak Baslyman; Moataz Ahmed
>
> **摘要:** Incorporating ethics into the requirement elicitation process is essential for creating ethically aligned systems. Although eliciting manual ethics requirements is effective, it requires diverse input from multiple stakeholders, which can be challenging due to time and resource constraints. Moreover, it is often given a low priority in the requirements elicitation process. This study proposes a framework for generating ethics requirements drafts by introducing an ethics advocate agent in a multi-agent LLM setting. This agent critiques and provides input on ethical issues based on the system description. The proposed framework is evaluated through two case studies from different contexts, demonstrating that it captures the majority of ethics requirements identified by researchers during 30-minute interviews and introduces several additional relevant requirements. However, it also highlights reliability issues in generating ethics requirements, emphasizing the need for human feedback in this sensitive domain. We believe this work can facilitate the broader adoption of ethics in the requirements engineering process, ultimately leading to more ethically aligned products.
>
---
#### [replaced 006] LearnLens: LLM-Enabled Personalised, Curriculum-Grounded Feedback with Educators in the Loop
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.04295v3](http://arxiv.org/pdf/2507.04295v3)**

> **作者:** Runcong Zhao; Artem Bobrov; Jiazheng Li; Yulan He
>
> **摘要:** Effective feedback is essential for student learning but is time-intensive for teachers. We present LearnLens, a modular, LLM-based system that generates personalised, curriculum-aligned feedback in science education. LearnLens comprises three components: (1) an error-aware assessment module that captures nuanced reasoning errors; (2) a curriculum-grounded generation module that uses a structured, topic-linked memory chain rather than traditional similarity-based retrieval, improving relevance and reducing noise; and (3) an educator-in-the-loop interface for customisation and oversight. LearnLens addresses key challenges in existing systems, offering scalable, high-quality feedback that empowers both teachers and students.
>
---
#### [replaced 007] When People are Floods: Analyzing Dehumanizing Metaphors in Immigration Discourse with Large Language Models
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.13246v2](http://arxiv.org/pdf/2502.13246v2)**

> **作者:** Julia Mendelsohn; Ceren Budak
>
> **备注:** To appear at ACL 2025. Please cite ACL version when proceedings are available
>
> **摘要:** Metaphor, discussing one concept in terms of another, is abundant in politics and can shape how people understand important issues. We develop a computational approach to measure metaphorical language, focusing on immigration discourse on social media. Grounded in qualitative social science research, we identify seven concepts evoked in immigration discourse (e.g. "water" or "vermin"). We propose and evaluate a novel technique that leverages both word-level and document-level signals to measure metaphor with respect to these concepts. We then study the relationship between metaphor, political ideology, and user engagement in 400K US tweets about immigration. While conservatives tend to use dehumanizing metaphors more than liberals, this effect varies widely across concepts. Moreover, creature-related metaphor is associated with more retweets, especially for liberal authors. Our work highlights the potential for computational methods to complement qualitative approaches in understanding subtle and implicit language in political discourse.
>
---
#### [replaced 008] ParaStudent: Generating and Evaluating Realistic Student Code by Teaching LLMs to Struggle
- **分类: cs.CY; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2507.12674v2](http://arxiv.org/pdf/2507.12674v2)**

> **作者:** Mihran Miroyan; Rose Niousha; Joseph E. Gonzalez; Gireeja Ranade; Narges Norouzi
>
> **摘要:** Large Language Models (LLMs) have shown strong performance on programming tasks, but can they generate student-like code like real students - imperfect, iterative, and stylistically diverse? We present ParaStudent, a systematic study of LLM-based "student-like" code generation in an introductory programming course setting. Using a dataset of timestamped student submissions across multiple semesters, we design low- and high-resolution experiments to model student progress and evaluate code outputs along semantic, functional, and stylistic dimensions. Our results show that fine-tuning significantly improves alignment with real student trajectories and captures error patterns, incremental improvements, and stylistic variations more faithfully. This study shows that modeling realistic student code requires capturing learning dynamics through context-aware generation, temporal modeling, and multi-dimensional evaluation. Code for experiments and evaluation is available at https://github.com/mmiroyan/ParaStudent.
>
---
