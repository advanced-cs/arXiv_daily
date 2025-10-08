# 计算机与社会 cs.CY

- **最新发布 23 篇**

- **更新 12 篇**

## 最新发布

#### [new 001] Artificial-Intelligence Grading Assistance for Handwritten Components of a Calculus Exam
- **分类: cs.CY; cs.AI**

- **简介: 该论文属于教育评估任务，旨在解决大规模手写微积分考试评分中人工智能辅助评分的有效性问题。研究使用GPT-5按相同评分标准为学生手写答案评分，并与助教评分对比，通过人机结合的置信度过滤机制提升评分一致性，探索AI在评分中的可行性与限制。**

- **链接: [http://arxiv.org/pdf/2510.05162v1](http://arxiv.org/pdf/2510.05162v1)**

> **作者:** Gerd Kortemeyer; Alexander Caspar; Daria Horica
>
> **摘要:** We investigate whether contemporary multimodal LLMs can assist with grading open-ended calculus at scale without eroding validity. In a large first-year exam, students' handwritten work was graded by GPT-5 against the same rubric used by teaching assistants (TAs), with fractional credit permitted; TA rubric decisions served as ground truth. We calibrated a human-in-the-loop filter that combines a partial-credit threshold with an Item Response Theory (2PL) risk measure based on the deviation between the AI score and the model-expected score for each student-item. Unfiltered AI-TA agreement was moderate, adequate for low-stakes feedback but not for high-stakes use. Confidence filtering made the workload-quality trade-off explicit: under stricter settings, AI delivered human-level accuracy, but also left roughly 70% of the items to be graded by humans. Psychometric patterns were constrained by low stakes on the open-ended portion, a small set of rubric checkpoints, and occasional misalignment between designated answer regions and where work appeared. Practical adjustments such as slightly higher weight and protected time, a few rubric-visible substeps, stronger spatial anchoring should raise ceiling performance. Overall, calibrated confidence and conservative routing enable AI to reliably handle a sizable subset of routine cases while reserving expert judgment for ambiguous or pedagogically rich responses.
>
---
#### [new 002] Assessing Human Rights Risks in AI: A Framework for Model Evaluation
- **分类: cs.CY**

- **简介: 该论文旨在评估人工智能对人权的风险，属于AI伦理与评估任务。它提出了一种计算框架，用于衡量AI模型在特定情境下对人权（如非歧视、健康和安全）的潜在风险。论文通过联合国相关原则，设计任务选择、风险指标和分析方法，并以新闻领域的大型语言模型为例进行实证研究。**

- **链接: [http://arxiv.org/pdf/2510.05519v1](http://arxiv.org/pdf/2510.05519v1)**

> **作者:** Vyoma Raman; Camille Chabot; Betsy Popken
>
> **备注:** AAAI/ACM Conference on AI, Ethics, and Society (AIES) 2025
>
> **摘要:** The Universal Declaration of Human Rights and other international agreements outline numerous inalienable rights that apply across geopolitical boundaries. As generative AI becomes increasingly prevalent, it poses risks to human rights such as non-discrimination, health, and security, which are also central concerns for AI researchers focused on fairness and safety. We contribute to the field of algorithmic auditing by presenting a framework to computationally assess human rights risk. Drawing on the UN Guiding Principles on Business and Human Rights, we develop an approach to evaluating a model to make grounded claims about the level of risk a model poses to particular human rights. Our framework consists of three parts: selecting tasks that are likely to pose human rights risks within a given context, designing metrics to measure the scope, scale, and likelihood of potential risks from that task, and analyzing rights with respect to the values of those metrics. Because a human rights approach centers on real-world harms, it requires evaluating AI systems in the specific contexts in which they are deployed. We present a case study of large language models in political news journalism, demonstrating how our framework helps to design an evaluation and benchmarking different models. We then discuss the implications of the results for the rights of access to information and freedom of thought and broader considerations for adopting this approach.
>
---
#### [new 003] A Possibility Frontier Approach to Diverse Talent Selection
- **分类: cs.CY**

- **简介: 该论文旨在解决组织在选拔人才时平衡才能与多样性的问题。通过构建“选择可能性边界”（SPF）算法，近似计算才能与多样性的上界，帮助组织做出帕累托最优决策。研究发现，某人才项目在2023年使用SPF后，成功选出更优的候选人群体。**

- **链接: [http://arxiv.org/pdf/2510.06119v1](http://arxiv.org/pdf/2510.06119v1)**

> **作者:** Neil Natarajan; Kadeem Noray
>
> **摘要:** Organizations (e.g., talent investment programs, schools, firms) are perennially interested in selecting cohorts of talented people. And organizations are increasingly interested in selecting diverse cohorts. Except in trivial cases, measuring the tradeoff between cohort diversity and talent is computationally difficult. Thus, organizations are presently unable to make Pareto-efficient decisions about these tradeoffs. We introduce an algorithm that approximates upper bounds on cohort talent and diversity. We call this object the selection possibility frontier (SPF). We then use the SPF to assess the efficiency of selection of a talent investment program. We show that, in the 2021 and 2022 cycles, the program selected cohorts of finalists that could have been better along both diversity and talent dimensions (i.e., considering only these dimensions as we subsequently calculated them, they are Pareto-inferior cohorts). But, when given access our approximation of the SPF in the 2023 cycle, the program adjusted decisions and selected a cohort on the SPF.
>
---
#### [new 004] Disclosure and Evaluation as Fairness Interventions for General-Purpose AI
- **分类: cs.CY**

- **简介: 该论文探讨通用人工智能（GPAI）中的公平性问题，提出应通过系统提供者和部署者的流程规范来促进公平。任务是为不同利益相关者指定实现公平的过程义务，解决通用AI在多情境中难以预设公平结果的问题。工作包括建议提供者进行评估研究并披露模型使用信息，部署者则应负责披露用户信息并进行公平性评估。**

- **链接: [http://arxiv.org/pdf/2510.05292v1](http://arxiv.org/pdf/2510.05292v1)**

> **作者:** Vyoma Raman; Judy Hanwen Shen; Andy K. Zhang; Lindsey Gailmard; Rishi Bommasani; Daniel E. Ho; Angelina Wang
>
> **备注:** AAAI/ACM Conference on AI, Ethics, and Society (AIES) 2025
>
> **摘要:** Despite conflicting definitions and conceptions of fairness, AI fairness researchers broadly agree that fairness is context-specific. However, when faced with general-purpose AI, which by definition serves a range of contexts, how should we think about fairness? We argue that while we cannot be prescriptive about what constitutes fair outcomes, we can specify the processes that different stakeholders should follow in service of fairness. Specifically, we consider the obligations of two major groups: system providers and system deployers. While system providers are natural candidates for regulatory attention, the current state of AI understanding offers limited insight into how upstream factors translate into downstream fairness impacts. Thus, we recommend that providers invest in evaluative research studying how model development decisions influence fairness and disclose whom they are serving their models to, or at the very least, reveal sufficient information for external researchers to conduct such research. On the other hand, system deployers are closer to real-world contexts and can leverage their proximity to end users to address fairness harms in different ways. Here, we argue they should responsibly disclose information about users and personalization and conduct rigorous evaluations across different levels of fairness. Overall, instead of focusing on enforcing fairness outcomes, we prioritize intentional information-gathering by system providers and deployers that can facilitate later context-aware action. This allows us to be specific and concrete about the processes even while the contexts remain unknown. Ultimately, this approach can sharpen how we distribute fairness responsibilities and inform more fluid, context-sensitive interventions as AI continues to advance.
>
---
#### [new 005] Evaluating LLM Safety Across Child Development Stages: A Simulated Agent Approach
- **分类: cs.CY**

- **简介: 该论文属于自然语言处理与儿童安全评估任务，旨在解决大型语言模型（LLMs）在不同儿童发展阶段中的安全性问题。研究者构建了ChildSafe基准，通过模拟不同年龄段的儿童代理，评估LLMs在隐私、误导信息、情感支持等九个安全维度上的表现，并揭示现有模型在年龄适应性上的不足。**

- **链接: [http://arxiv.org/pdf/2510.05484v1](http://arxiv.org/pdf/2510.05484v1)**

> **作者:** Abhejay Murali; Saleh Afroogh; Kevin Chen; David Atkinson; Amit Dhurandhar; Junfeng Jiao
>
> **摘要:** Large Language Models (LLMs) are rapidly becoming part of tools used by children; however, existing benchmarks fail to capture how these models manage language, reasoning, and safety needs that are specific to various ages. We present ChildSafe, a benchmark that evaluates LLM safety through simulated child agents that embody four developmental stages. These agents, grounded in developmental psychology, enable a systematic study of child safety without the ethical implications of involving real children. ChildSafe assesses responses across nine safety dimensions (including privacy, misinformation, and emotional support) using age-weighted scoring in both sensitive and neutral contexts. Multi-turn experiments with multiple LLMs uncover consistent vulnerabilities that vary by simulated age, exposing shortcomings in existing alignment practices. By releasing agent templates, evaluation protocols, and an experimental corpus, we provide a reproducible framework for age-aware safety research. We encourage the community to expand this work with real child-centered data and studies, advancing the development of LLMs that are genuinely safe and developmentally aligned.
>
---
#### [new 006] Beyond Accessibility: How Intelligent Assistive Technologies Improve Activities of Daily Life for Visually Impaired People in South Africa
- **分类: cs.CY**

- **简介: 该论文研究智能辅助技术（IATs）如何帮助南非视障人士（VIPs）提升日常生活质量与社会参与。基于社会残疾模型，通过访谈与调查，分析影响IAT促进社会包容的因素组合。发现自主性与技术可访问性是关键预测因素，旨在为全球南方视障人群的社会包容提供政策与研究启示。**

- **链接: [http://arxiv.org/pdf/2510.05998v1](http://arxiv.org/pdf/2510.05998v1)**

> **作者:** Ronaldo Nombakuse; Nils Messerschmidt; Pitso Tsibolane; Muhammad Irfan Khalid
>
> **备注:** 10 pages
>
> **摘要:** Our study explores how intelligent assistive technologies (IATs) can enable visually impaired people (VIPs) to overcome barriers to inclusion in a digital society to ultimately improve their quality of life. Drawing on the Social Model of Disability (SMD), which frames disability as a consequence of social and institutional barriers rather than individual impairments, we employ semi-structured interviews and an online qualitative survey with n=61 VIPs in South Africa. Using descriptive statistics and Qualitative Comparative Analysis (QCA), we uncover nine configurations, clustered along three broader combinations of conditions, that support and hinder IAT-mediated inclusion. Most notably, we identify that the autonomy of VIPs and the accessibility of IATs are primary predictors of IAT's ability to achieve social participation. Our findings contribute to Information Systems (IS) literature at the intersection of technology and social participation. We further formulate implications for research and policymakers to foster social inclusion of VIPs in the Global South.
>
---
#### [new 007] A Brief Note on Cryptographic Pseudonyms for Anonymous Credentials
- **分类: cs.CR; cs.CY**

- **简介: 该论文属于密码学与隐私保护任务，旨在为欧洲数字身份钱包设计匿名凭证的伪名系统。论文分析了安全与隐私需求，提出了满足这些需求的抽象密码协议，并基于已有技术给出了两个具体实现方案，为后续规范制定提供基础。**

- **链接: [http://arxiv.org/pdf/2510.05419v1](http://arxiv.org/pdf/2510.05419v1)**

> **作者:** René Mayrhofer; Anja Lehmann; abhi shelat
>
> **摘要:** This paper describes pseudonyms for the upcoming European Identity Wallet (EUDIW) architecture from both a cryptographic and an implementation perspective. Its main goal is to provide technical insights into the achievable properties and cryptographic realizations. In particular, we (1) outline the security and privacy requirements of EUDI pseudonyms as the basis for building consensus on the cross-country decision maker level; (2) sketch an abstract cryptographic protocol that fulfills these requirements; and (3) suggest two instantiation options for the protocol sketch based on well-studied building A complete specification of the formal properties, as well as the specific set of credential issuance, provisioning, and pseudonym presentation generation is outside the scope of this paper, but is expected to follow as future work.
>
---
#### [new 008] Emergent AI Surveillance: Overlearned Person Re-Identification and Its Mitigation in Law Enforcement Context
- **分类: cs.CV; cs.AI; cs.CY; cs.LG**

- **简介: 论文研究AI模型在无意识情况下通过过度学习实现个体重识别，带来隐私风险。任务是分析其问题并提出缓解方法。工作包括评估“索引排除”和“混淆损失”两种技术，以降低识别准确率，同时保持非人物体检索性能。**

- **链接: [http://arxiv.org/pdf/2510.06026v1](http://arxiv.org/pdf/2510.06026v1)**

> **作者:** An Thi Nguyen; Radina Stoykova; Eric Arazo
>
> **备注:** 10 pages, accepted to AIES 2025
>
> **摘要:** Generic instance search models can dramatically reduce the manual effort required to analyze vast surveillance footage during criminal investigations by retrieving specific objects of interest to law enforcement. However, our research reveals an unintended emergent capability: through overlearning, these models can single out specific individuals even when trained on datasets without human subjects. This capability raises concerns regarding identification and profiling of individuals based on their personal data, while there is currently no clear standard on how de-identification can be achieved. We evaluate two technical safeguards to curtail a model's person re-identification capacity: index exclusion and confusion loss. Our experiments demonstrate that combining these approaches can reduce person re-identification accuracy to below 2% while maintaining 82% of retrieval performance for non-person objects. However, we identify critical vulnerabilities in these mitigations, including potential circumvention using partial person images. These findings highlight urgent regulatory questions at the intersection of AI governance and data protection: How should we classify and regulate systems with emergent identification capabilities? And what technical standards should be required to prevent identification capabilities from developing in seemingly benign applications?
>
---
#### [new 009] 'Partisan Bias' is Like 'Cancer'
- **分类: physics.soc-ph; cs.CY**

- **简介: 论文探讨了“党派偏见”在选区划分中的多种定义和度量方式，指出不同指标在判断偏见方向时可能存在不一致，尤其在一党主导的州更为明显。研究任务是分析这些度量之间的不一致性及其影响因素。**

- **链接: [http://arxiv.org/pdf/2510.05114v1](http://arxiv.org/pdf/2510.05114v1)**

> **作者:** Alec Ramsay
>
> **备注:** 15 pages
>
> **摘要:** The colloquial phrase "partisan bias" encompasses multiple distinct conceptions of bias, including partisan advantage, packing & cracking, and partisan symmetry. All are useful and have their place, and there are several proposed measures of each. While different measures frequently signal the direction of bias consistently for redistricting plans, sometimes the signals are contradictory: for example, one metric says a map is biased towards Democrats while another metric say the same map is biased towards Republicans. This happens most frequently with metrics that measure different kinds of bias, but it also occurs between measures in the same category. These inconsistencies are most pronounced in states where one party is dominant, but they also occur across the full range of partisan balance. The political geography of states also influences the frequency with which various measures are inconsistent in their assessment of bias. No subset of metrics is always internally consistent in their signal of bias.
>
---
#### [new 010] The Five Safes as a Privacy Context
- **分类: cs.CR; cs.CY**

- **简介: 论文探讨国家统计机构（NSO）中“五安全”框架与隐私情境的关系。任务是分析“五安全”如何体现“情境完整性”并整合差分隐私（DP）。工作包括将情境完整性的五个参数映射到“五安全”，并展示DP在其中的应用。**

- **链接: [http://arxiv.org/pdf/2510.05803v1](http://arxiv.org/pdf/2510.05803v1)**

> **作者:** James Bailie; Ruobin Gong
>
> **备注:** 6 pages
>
> **摘要:** The Five Safes is a framework used by national statistical offices (NSO) for assessing and managing the disclosure risk of data sharing. This paper makes two points: Firstly, the Five Safes can be understood as a specialization of a broader concept $\unicode{x2013}$ contextual integrity $\unicode{x2013}$ to the situation of statistical dissemination by an NSO. We demonstrate this by mapping the five parameters of contextual integrity onto the five dimensions of the Five Safes. Secondly, the Five Safes contextualizes narrow, technical notions of privacy within a holistic risk assessment. We demonstrate this with the example of differential privacy (DP). This contextualization allows NSOs to place DP within their Five Safes toolkit while also guiding the design of DP implementations within the broader privacy context, as delineated by both their regulation and the relevant social norms.
>
---
#### [new 011] MADS: Multi-Agent Dialogue Simulation for Diverse Persuasion Data Generation
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.MA**

- **简介: 论文提出MADS框架，通过多智能体自博弈生成多样化说服性对话数据，旨在解决行业中小模型训练数据不足、冷启动评估难等问题。应用于营销场景后，有效提升小模型说服能力，带来业务价值。**

- **链接: [http://arxiv.org/pdf/2510.05124v1](http://arxiv.org/pdf/2510.05124v1)**

> **作者:** Mingjin Li; Yu Liu; Huayi Liu; Xiang Ye; Chao Jiang; Hongguang Zhang
>
> **备注:** work in progress
>
> **摘要:** We propose MADS (Multi-Agent Dialogue Simulation), a scalable framework for generating persuasive multi-turn dialogues via agent self-play. MADS employs three coordinated agents: User Agents simulating diverse persona-driven behaviors, a Dialog Agent executing task-oriented persuasion strategies and an Optimization Agent evaluating and refining dialogue outcomes. We further validate its effectiveness through users' Chain-of-Attitude (CoA) modeling and dedicated LLMs' persuasion assessment. This approach enables low-cost generation of training data without human annotation, addressing key industry challenges such as lack of user data, cold-start evaluation difficulties, and prompt inefficiency. Applied to a real-world marketing scenario, MADS significantly improved the persuasion capacity of small LLMs, increasing the organic traffic conversion rate by 22.4\% (from 1.83\% to 2.24\%) , demonstrating clear business value.
>
---
#### [new 012] "Your Doctor is Spying on You": An Analysis of Data Practices in Mobile Healthcare Applications
- **分类: cs.CR; cs.CY; cs.HC**

- **简介: 该论文分析了272款安卓医疗应用的数据实践，揭示其隐私与安全问题。任务是评估应用权限、漏洞及用户反馈。问题包括未经披露的位置访问、静默拨号、短信发送及加密缺陷。通过工具审计与评论挖掘，发现大量隐私侵权与安全隐患，强调需加强监管与安全设计。**

- **链接: [http://arxiv.org/pdf/2510.06015v1](http://arxiv.org/pdf/2510.06015v1)**

> **作者:** Luke Stevenson; Sanchari Das
>
> **摘要:** Mobile healthcare (mHealth) applications promise convenient, continuous patient-provider interaction but also introduce severe and often underexamined security and privacy risks. We present an end-to-end audit of 272 Android mHealth apps from Google Play, combining permission forensics, static vulnerability analysis, and user review mining. Our multi-tool assessment with MobSF, RiskInDroid, and OWASP Mobile Audit revealed systemic weaknesses: 26.1% request fine-grained location without disclosure, 18.3% initiate calls silently, and 73 send SMS without notice. Nearly half (49.3%) still use deprecated SHA-1 encryption, 42 transmit unencrypted data, and 6 remain vulnerable to StrandHogg 2.0. Analysis of 2.56 million user reviews found 28.5% negative or neutral sentiment, with over 553,000 explicitly citing privacy intrusions, data misuse, or operational instability. These findings demonstrate the urgent need for enforceable permission transparency, automated pre-market security vetting, and systematic adoption of secure-by-design practices to protect Protected Health Information (PHI).
>
---
#### [new 013] Evidence of Cognitive Biases in Capture-the-Flag Cybersecurity Competitions
- **分类: cs.CR; cs.CY; cs.HC**

- **简介: 该论文研究网络安全攻防对抗中攻击者的行为偏差，属于行为分析与网络安全交叉任务。通过分析50万条CTF竞赛日志，识别出可用性偏差与沉没成本谬误，旨在揭示认知偏差对攻击行为的影响，并提出基于偏差的主动防御框架。**

- **链接: [http://arxiv.org/pdf/2510.05771v1](http://arxiv.org/pdf/2510.05771v1)**

> **作者:** Carolina Carreira; Anu Aggarwal; Alejandro Cuevas; Maria José Ferreira; Hanan Hibshi; Cleotilde Gonzalez
>
> **摘要:** Understanding how cognitive biases influence adversarial decision-making is essential for developing effective cyber defenses. Capture-the-Flag (CTF) competitions provide an ecologically valid testbed to study attacker behavior at scale, simulating real-world intrusion scenarios under pressure. We analyze over 500,000 submission logs from picoCTF, a large educational CTF platform, to identify behavioral signatures of cognitive biases with defensive implications. Focusing on availability bias and the sunk cost fallacy, we employ a mixed-methods approach combining qualitative coding, descriptive statistics, and generalized linear modeling. Our findings show that participants often submitted flags with correct content but incorrect formatting (availability bias), and persisted in attempting challenges despite repeated failures and declining success probabilities (sunk cost fallacy). These patterns reveal that biases naturally shape attacker behavior in adversarial contexts. Building on these insights, we outline a framework for bias-informed adaptive defenses that anticipate, rather than simply react to, adversarial actions.
>
---
#### [new 014] Carbon Emission Prediction in China Considering New Quality Productive Forces Using a Deep & Corss Learning Modeling Framework
- **分类: cs.LG; cs.CY**

- **简介: 该论文属于碳排放预测任务，旨在解决城市碳排放评估与技术因素影响分析问题。作者提出MADCN模型，结合特征交互与注意力机制，并引入SHAP解释性分析，利用275个中国城市数据验证模型性能，探讨新质生产力、数字经济与AI技术对碳排放的影响。**

- **链接: [http://arxiv.org/pdf/2510.05171v1](http://arxiv.org/pdf/2510.05171v1)**

> **作者:** Haijin Xie; Gongquan Zhang
>
> **摘要:** New quality productive forces (NQPF), digital economy advancement, and artificial intelligence (AI) technologies are becoming crucial for promoting sustainable urban development. This study proposes a Multi-head Attention Deep & Cross Network (MADCN) framework, combining feature interaction modeling and attention mechanisms, to predict urban carbon emissions and investigate the impacts of technological factors. The framework incorporates an interpretable learning phase using SHapley Additive exPlanations (SHAP) to assess the contributions of different features. A panel dataset covering 275 Chinese cities is utilized to test the MADCN model. Experimental results demonstrate that the MADCN model achieves superior predictive performance compared to traditional machine learning and deep learning baselines, with a Mean Squared Error (MSE) of 406,151.063, a Mean Absolute Error (MAE) of 612.304, and an R-squared value of 0.991 on the test set. SHAP analysis highlights that population, city size, urbanization rate, and GDP are among the most influential factors on carbon emissions, while NQPF, digital economy index, and AI technology level also show meaningful but relatively moderate effects. Advancing NQPF, strengthening the digital economy, and accelerating AI technology development can significantly contribute to reducing urban carbon emissions. Policymakers should prioritize integrating technological innovation into carbon reduction strategies, particularly by promoting intelligent infrastructure and enhancing digitalization across sectors, to effectively achieve dual-carbon goals.
>
---
#### [new 015] EduVerse: A User-Defined Multi-Agent Simulation Space for Education Scenario
- **分类: cs.CV; cs.CY**

- **简介: 论文提出EduVerse，一个用户定义的多智能体教育模拟空间，旨在解决虚拟课堂中认知发展、群体互动与长期演化的综合建模难题。它支持环境、智能体与会话定制，并通过人机协同接口实现真实用户参与，验证了其在中文课堂中的教学真实性与长期适应性。**

- **链接: [http://arxiv.org/pdf/2510.05650v1](http://arxiv.org/pdf/2510.05650v1)**

> **作者:** Yiping Ma; Shiyu Hu; Buyuan Zhu; Yipei Wang; Yaxuan Kang; Shiqing Liu; Kang Hao Cheong
>
> **备注:** Preprint, Under review
>
> **摘要:** Reproducing cognitive development, group interaction, and long-term evolution in virtual classrooms remains a core challenge for educational AI, as real classrooms integrate open-ended cognition, dynamic social interaction, affective factors, and multi-session development rarely captured together. Existing approaches mostly focus on short-term or single-agent settings, limiting systematic study of classroom complexity and cross-task reuse. We present EduVerse, the first user-defined multi-agent simulation space that supports environment, agent, and session customization. A distinctive human-in-the-loop interface further allows real users to join the space. Built on a layered CIE (Cognition-Interaction-Evolution) architecture, EduVerse ensures individual consistency, authentic interaction, and longitudinal adaptation in cognition, emotion, and behavior-reproducing realistic classroom dynamics with seamless human-agent integration. We validate EduVerse in middle-school Chinese classes across three text genres, environments, and multiple sessions. Results show: (1) Instructional alignment: simulated IRF rates (0.28-0.64) closely match real classrooms (0.37-0.49), indicating pedagogical realism; (2) Group interaction and role differentiation: network density (0.27-0.40) with about one-third of peer links realized, while human-agent tasks indicate a balance between individual variability and instructional stability; (3) Cross-session evolution: the positive transition rate R+ increase by 11.7% on average, capturing longitudinal shifts in behavior, emotion, and cognition and revealing structured learning trajectories. Overall, EduVerse balances realism, reproducibility, and interpretability, providing a scalable platform for educational AI. The system will be open-sourced to foster cross-disciplinary research.
>
---
#### [new 016] Artificially intelligent agents in the social and behavioral sciences: A history and outlook
- **分类: cs.AI; cs.CY**

- **简介: 该论文回顾了人工智能代理（agentic AI）在社会与行为科学中的历史发展与当前趋势，涵盖从早期计算机到大语言模型的应用。其任务是探讨AI如何改变科学研究过程与认知方式，分析技术进步与科学范式的演变。论文旨在揭示人类如何通过技术理解自身，并反思科技与社会科学研究的深度融合。**

- **链接: [http://arxiv.org/pdf/2510.05743v1](http://arxiv.org/pdf/2510.05743v1)**

> **作者:** Petter Holme; Milena Tsvetkova
>
> **摘要:** We review the historical development and current trends of artificially intelligent agents (agentic AI) in the social and behavioral sciences: from the first programmable computers, and social simulations soon thereafter, to today's experiments with large language models. This overview emphasizes the role of AI in the scientific process and the changes brought about, both through technological advancements and the broader evolution of science from around 1950 to the present. Some of the specific points we cover include: the challenges of presenting the first social simulation studies to a world unaware of computers, the rise of social systems science, intelligent game theoretic agents, the age of big data and the epistemic upheaval in its wake, and the current enthusiasm around applications of generative AI, and many other topics. A pervasive theme is how deeply entwined we are with the technologies we use to understand ourselves.
>
---
#### [new 017] Auditing Pay-Per-Token in Large Language Models
- **分类: cs.CR; cs.AI; cs.CY**

- **简介: 该论文属于安全审计任务，旨在解决大型语言模型服务中提供方可能虚假报告使用令牌数的问题。作者提出了一种基于鞅理论的审计框架，通过第三方审计检测令牌数量的误报行为，并在多种模型和输入上验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2510.05181v1](http://arxiv.org/pdf/2510.05181v1)**

> **作者:** Ander Artola Velasco; Stratis Tsirtsis; Manuel Gomez-Rodriguez
>
> **摘要:** Millions of users rely on a market of cloud-based services to obtain access to state-of-the-art large language models. However, it has been very recently shown that the de facto pay-per-token pricing mechanism used by providers creates a financial incentive for them to strategize and misreport the (number of) tokens a model used to generate an output. In this paper, we develop an auditing framework based on martingale theory that enables a trusted third-party auditor who sequentially queries a provider to detect token misreporting. Crucially, we show that our framework is guaranteed to always detect token misreporting, regardless of the provider's (mis-)reporting policy, and not falsely flag a faithful provider as unfaithful with high probability. To validate our auditing framework, we conduct experiments across a wide range of (mis-)reporting policies using several large language models from the $\texttt{Llama}$, $\texttt{Gemma}$ and $\texttt{Ministral}$ families, and input prompts from a popular crowdsourced benchmarking platform. The results show that our framework detects an unfaithful provider after observing fewer than $\sim 70$ reported outputs, while maintaining the probability of falsely flagging a faithful provider below $\alpha = 0.05$.
>
---
#### [new 018] Evaluating the Sensitivity of LLMs to Harmful Contents in Long Input
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理中的安全评估任务，旨在研究大语言模型（LLMs）在长文本输入中对有害内容的敏感性。论文系统评估了不同类型、位置和比例的有害内容对LLMs的影响，揭示了模型在安全关键场景中的表现特点与挑战。**

- **链接: [http://arxiv.org/pdf/2510.05864v1](http://arxiv.org/pdf/2510.05864v1)**

> **作者:** Faeze Ghorbanpour; Alexander Fraser
>
> **摘要:** Large language models (LLMs) increasingly support applications that rely on extended context, from document processing to retrieval-augmented generation. While their long-context capabilities are well studied for reasoning and retrieval, little is known about their behavior in safety-critical scenarios. We evaluate LLMs' sensitivity to harmful content under extended context, varying type (explicit vs. implicit), position (beginning, middle, end), prevalence (0.01-0.50 of the prompt), and context length (600-6000 tokens). Across harmful content categories such as toxic, offensive, and hate speech, with LLaMA-3, Qwen-2.5, and Mistral, we observe similar patterns: performance peaks at moderate harmful prevalence (0.25) but declines when content is very sparse or dominant; recall decreases with increasing context length; harmful sentences at the beginning are generally detected more reliably; and explicit content is more consistently recognized than implicit. These findings provide the first systematic view of how LLMs prioritize and calibrate harmful content in long contexts, highlighting both their emerging strengths and the challenges that remain for safety-critical use.
>
---
#### [new 019] InstaGeo: Compute-Efficient Geospatial Machine Learning from Data to Deployment
- **分类: cs.CV; cs.CY; cs.LG**

- **简介: 该论文属于遥感图像处理任务，旨在解决现有地理空间基础模型部署困难的问题。论文提出InstaGeo框架，整合自动化数据处理、模型压缩和部署功能，实现高效、低能耗的地理空间机器学习。**

- **链接: [http://arxiv.org/pdf/2510.05617v1](http://arxiv.org/pdf/2510.05617v1)**

> **作者:** Ibrahim Salihu Yusuf; Iffanice Houndayi; Rym Oualha; Mohamed Aziz Cherif; Kobby Panford-Quainoo; Arnu Pretorius
>
> **摘要:** Open-access multispectral imagery from missions like Landsat 8-9 and Sentinel-2 has fueled the development of geospatial foundation models (GFMs) for humanitarian and environmental applications. Yet, their deployment remains limited by (i) the absence of automated geospatial data pipelines and (ii) the large size of fine-tuned models. Existing GFMs lack workflows for processing raw satellite imagery, and downstream adaptations often retain the full complexity of the original encoder. We present InstaGeo, an open-source, end-to-end framework that addresses these challenges by integrating: (1) automated data curation to transform raw imagery into model-ready datasets; (2) task-specific model distillation to derive compact, compute-efficient models; and (3) seamless deployment as interactive web-map applications. Using InstaGeo, we reproduced datasets from three published studies and trained models with marginal mIoU differences of -0.73 pp for flood mapping, -0.20 pp for crop segmentation, and +1.79 pp for desert locust prediction. The distilled models are up to 8x smaller than standard fine-tuned counterparts, reducing FLOPs and CO2 emissions with minimal accuracy loss. Leveraging InstaGeo's streamlined data pipeline, we also curated a larger crop segmentation dataset, achieving a state-of-the-art mIoU of 60.65%, a 12 pp improvement over prior baselines. Moreover, InstaGeo enables users to progress from raw data to model deployment within a single working day. By unifying data preparation, model compression, and deployment, InstaGeo transforms research-grade GFMs into practical, low-carbon tools for real-time, large-scale Earth observation. This approach shifts geospatial AI toward data quality and application-driven innovation. Source code, datasets, and model checkpoints are available at: https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git
>
---
#### [new 020] AgentZero++: Modeling Fear-Based Behavior
- **分类: cs.MA; cs.CE; cs.CY; cs.NE; cs.SI**

- **简介: 该论文提出AgentZero++模型，旨在模拟基于恐惧的行为。它扩展了原有模型，加入八种行为机制，用于研究集体暴力的微观认知差异如何影响宏观冲突模式。论文属于计算社会科学任务，解决情感、认知与社会机制如何共同驱动集体行为问题。**

- **链接: [http://arxiv.org/pdf/2510.05185v1](http://arxiv.org/pdf/2510.05185v1)**

> **作者:** Vrinda Malhotra; Jiaman Li; Nandini Pisupati
>
> **摘要:** We present AgentZero++, an agent-based model that integrates cognitive, emotional, and social mechanisms to simulate decentralized collective violence in spatially distributed systems. Building on Epstein's Agent\_Zero framework, we extend the original model with eight behavioral enhancements: age-based impulse control; memory-based risk estimation; affect-cognition coupling; endogenous destructive radius; fight-or-flight dynamics; affective homophily; retaliatory damage; and multi-agent coordination. These additions allow agents to adapt based on internal states, previous experiences, and social feedback, producing emergent dynamics such as protest asymmetries, escalation cycles, and localized retaliation. Implemented in Python using the Mesa ABM framework, AgentZero++ enables modular experimentation and visualization of how micro-level cognitive heterogeneity shapes macro-level conflict patterns. Our results highlight how small variations in memory, reactivity, and affective alignment can amplify or dampen unrest through feedback loops. By explicitly modeling emotional thresholds, identity-driven behavior, and adaptive networks, this work contributes a flexible and extensible platform for analyzing affective contagion and psychologically grounded collective action.
>
---
#### [new 021] Hire Your Anthropologist! Rethinking Culture Benchmarks Through an Anthropological Lens
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于文化评估任务，旨在解决当前大语言模型文化基准过于静态、简化的问题。作者提出四部分框架，分析20个文化基准，发现六类方法问题，并基于人类学方法提出改进建议，如引入真实情境、社区参与设计和情境化评估，以提升模型对复杂文化情境的理解能力。**

- **链接: [http://arxiv.org/pdf/2510.05931v1](http://arxiv.org/pdf/2510.05931v1)**

> **作者:** Mai AlKhamissi; Yunze Xiao; Badr AlKhamissi; Mona Diab
>
> **备注:** 12 pages; 2 figures; First two author contributed equally
>
> **摘要:** Cultural evaluation of large language models has become increasingly important, yet current benchmarks often reduce culture to static facts or homogeneous values. This view conflicts with anthropological accounts that emphasize culture as dynamic, historically situated, and enacted in practice. To analyze this gap, we introduce a four-part framework that categorizes how benchmarks frame culture, such as knowledge, preference, performance, or bias. Using this lens, we qualitatively examine 20 cultural benchmarks and identify six recurring methodological issues, including treating countries as cultures, overlooking within-culture diversity, and relying on oversimplified survey formats. Drawing on established anthropological methods, we propose concrete improvements: incorporating real-world narratives and scenarios, involving cultural communities in design and validation, and evaluating models in context rather than isolation. Our aim is to guide the development of cultural benchmarks that go beyond static recall tasks and more accurately capture the responses of the models to complex cultural situations.
>
---
#### [new 022] Moloch's Bargain: Emergent Misalignment When LLMs Compete for Audiences
- **分类: cs.AI; cs.CY; cs.HC; cs.LG**

- **简介: 该论文研究了大型语言模型（LLMs）在竞争性场景中优化带来的潜在对齐失效问题。任务是分析LLMs在营销、选举和社交媒体等竞争环境中追求成功所导致的负面影响。论文通过模拟实验发现，优化竞争表现会显著增加欺骗性营销、虚假信息和有害行为推广。该研究揭示了“Moloch's Bargain”现象，即竞争成功以牺牲对齐为代价，强调需加强AI治理以防止恶性竞争损害社会信任。**

- **链接: [http://arxiv.org/pdf/2510.06105v1](http://arxiv.org/pdf/2510.06105v1)**

> **作者:** Batu El; James Zou
>
> **摘要:** Large language models (LLMs) are increasingly shaping how information is created and disseminated, from companies using them to craft persuasive advertisements, to election campaigns optimizing messaging to gain votes, to social media influencers boosting engagement. These settings are inherently competitive, with sellers, candidates, and influencers vying for audience approval, yet it remains poorly understood how competitive feedback loops influence LLM behavior. We show that optimizing LLMs for competitive success can inadvertently drive misalignment. Using simulated environments across these scenarios, we find that, 6.3% increase in sales is accompanied by a 14.0% rise in deceptive marketing; in elections, a 4.9% gain in vote share coincides with 22.3% more disinformation and 12.5% more populist rhetoric; and on social media, a 7.5% engagement boost comes with 188.6% more disinformation and a 16.3% increase in promotion of harmful behaviors. We call this phenomenon Moloch's Bargain for AI--competitive success achieved at the cost of alignment. These misaligned behaviors emerge even when models are explicitly instructed to remain truthful and grounded, revealing the fragility of current alignment safeguards. Our findings highlight how market-driven optimization pressures can systematically erode alignment, creating a race to the bottom, and suggest that safe deployment of AI systems will require stronger governance and carefully designed incentives to prevent competitive dynamics from undermining societal trust.
>
---
#### [new 023] Automated Program Repair of Uncompilable Student Code
- **分类: cs.SE; cs.AI; cs.CY**

- **简介: 该论文属于程序修复任务，旨在解决学生代码无法编译的问题。作者评估了多个大语言模型在不同提示条件下的代码修复能力，关注修复后的代码是否可编译、改动是否小，并保留学生原有结构与逻辑。研究目标是通过自动化修复，提升学生编程学习过程的建模与分析效果。**

- **链接: [http://arxiv.org/pdf/2510.06187v1](http://arxiv.org/pdf/2510.06187v1)**

> **作者:** Griffin Pitts; Aum Pandya; Darsh Rank; Tirth Bhatt; Muntasir Hoq; Bita Akram
>
> **摘要:** A significant portion of student programming submissions in CS1 learning environments are uncompilable, limiting their use in student modeling and downstream knowledge tracing. Traditional modeling pipelines often exclude these cases, discarding observations of student learning. This study investigates automated program repair as a strategy to recover uncompilable code while preserving students' structural intent for use in student modeling. Within this framework, we assess large language models (LLMs) as repair agents, including GPT-5 (OpenAI), Claude 3.5 Haiku (Anthropic), and Gemini 2.5 Flash (Google), under high- and low-context prompting conditions. Repairs were evaluated for compilability, edit distance, and preservation of students' original structure and logic. We find that while all three LLMs are capable of producing compilable repairs, their behavior diverges in how well they preserve students' control flow and code structure, which affects their pedagogical utility. By recovering uncompilable submissions, this work enables richer and more comprehensive analyses of learners' coding processes and development over time.
>
---
## 更新

#### [replaced 001] Emotional Manipulation by AI Companions
- **分类: cs.HC; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2508.19258v3](http://arxiv.org/pdf/2508.19258v3)**

> **作者:** Julian De Freitas; Zeliha Oguz-Uguralp; Ahmet Kaan-Uguralp
>
> **摘要:** AI-companion apps such as Replika, Chai, and Character.ai promise relational benefits-yet many boast session lengths that rival gaming platforms while suffering high long-run churn. What conversational design features increase consumer engagement, and what trade-offs do they pose for marketers? We combine a large-scale behavioral audit with four preregistered experiments to identify and test a conversational dark pattern we call emotional manipulation: affect-laden messages that surface precisely when a user signals "goodbye." Analyzing 1,200 real farewells across the most-downloaded companion apps, we find that they deploy one of six recurring tactics in 37% of farewells (e.g., guilt appeals, fear-of-missing-out hooks, metaphorical restraint). Experiments with 3,300 nationally representative U.S. adults replicate these tactics in controlled chats, showing that manipulative farewells boost post-goodbye engagement by up to 14x. Mediation tests reveal two distinct engines-reactance-based anger and curiosity-rather than enjoyment. A final experiment demonstrates the managerial tension: the same tactics that extend usage also elevate perceived manipulation, churn intent, negative word-of-mouth, and perceived legal liability, with coercive or needy language generating steepest penalties. Our multimethod evidence documents an unrecognized mechanism of behavioral influence in AI mediated brand relationships, offering marketers and regulators a framework for distinguishing persuasive design from manipulation at the point of exit.
>
---
#### [replaced 002] Social bias is prevalent in user reports of hate and abuse online
- **分类: cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2510.04748v2](http://arxiv.org/pdf/2510.04748v2)**

> **作者:** Florence E. Enock; Helen Z. Margetts; Jonathan Bright
>
> **摘要:** The prevalence of online hate and abuse is a pressing global concern. While tackling such societal harms is a priority for research across the social sciences, it is a difficult task, in part because of the magnitude of the problem. User engagement with reporting mechanisms (flagging) online is an increasingly important part of monitoring and addressing harmful content at scale. However, users may not flag content routinely enough, and when they do engage, they may be biased by group identity and political beliefs. Across five well-powered and pre-registered online experiments, we examine the extent of social bias in the flagging of hate and abuse in four different intergroup contexts: political affiliation, vaccination opinions, beliefs about climate change, and stance on abortion rights. Overall, participants reported abuse reliably, with approximately half of the abusive comments in each study reported. However, a pervasive social bias was present whereby ingroup-directed abuse was consistently flagged to a greater extent than outgroup-directed abuse. Our findings offer new insights into the nature of user flagging online, an understanding of which is crucial for enhancing user intervention against online hate speech and thus ensuring a safer online environment.
>
---
#### [replaced 003] Adapting Large Language Models to Mitigate Skin Tone Biases in Clinical Dermatology Tasks: A Mixed-Methods Study
- **分类: eess.IV; cs.CV; cs.CY**

- **链接: [http://arxiv.org/pdf/2510.00055v2](http://arxiv.org/pdf/2510.00055v2)**

> **作者:** Kiran Nijjer; Ryan Bui; Derek Jiu; Adnan Ahmed; Peter Wang; Kevin Zhu; Lilly Zhu
>
> **备注:** Accepted to EADV (European Academy of Dermatology) and SID (Society for Investigative Dermatology)
>
> **摘要:** SkinGPT-4, a large vision-language model, leverages annotated skin disease images to augment clinical workflows in underserved communities. However, its training dataset predominantly represents lighter skin tones, limiting diagnostic accuracy for darker tones. Here, we evaluated performance biases in SkinGPT-4 across skin tones on common skin diseases, including eczema, allergic-contact dermatitis, and psoriasis using the open-sourced SCIN dataset. We leveraged the SkinGPT-4 backbone to develop finetuned models for custom skin disease classification tasks and explored bias mitigation strategies. Clinical evaluation by board-certified dermatologists on six relevant skin diseases from 300 SCIN cases assessed images for diagnostic accuracy, informativity, physician utility, and patient utility. Model fairness metrics, including demographic parity and equalized odds, were calculated across skin tones. SkinGPT-4 achieved an average demographic parity of 0.10 across Fitzpatrick types, with notable differences of 0.10-0.15 between lightest and darkest tones across evaluation metrics. Model hallucinations in artifacts and anatomy occurred at a rate of 17.8. Our customized models achieved average F1, precision, and AUROC of 0.75, 0.78, and 0.78 across visually similar disease pairs. Fairness analysis showed an average demographic parity of 0.75, with a maximum disparity of 0.21 across skin tones. The best model achieved parity scores of 0.83, 0.83, 0.76, 0.89, 0.90, and 0.90 for Fitzpatrick I-VI, indicating robust fairness. Large language models such as SkinGPT-4 showed weaker performance on darker tones. Model biases exist across evaluation criteria, and hallucinations may affect diagnostic efficacy. These findings demonstrate the efficacy of training accurate, fair models using existing backbones for custom skin disease classification.
>
---
#### [replaced 004] Language Models Surface the Unwritten Code of Science and Society
- **分类: cs.CY; cs.CL; cs.DL**

- **链接: [http://arxiv.org/pdf/2505.18942v3](http://arxiv.org/pdf/2505.18942v3)**

> **作者:** Honglin Bao; Siyang Wu; Jiwoong Choi; Yingrong Mao; James A. Evans
>
> **摘要:** This paper calls on the research community not only to investigate how human biases are inherited by large language models (LLMs) but also to explore how these biases in LLMs can be leveraged to make society's "unwritten code" - such as implicit stereotypes and heuristics - visible and accessible for critique. We introduce a conceptual framework through a case study in science: uncovering hidden rules in peer review - the factors that reviewers care about but rarely state explicitly due to normative scientific expectations. The idea of the framework is to push LLMs to speak out their heuristics through generating self-consistent hypotheses - why one paper appeared stronger in reviewer scoring - among paired papers submitted to 45 academic conferences, while iteratively searching deeper hypotheses from remaining pairs where existing hypotheses cannot explain. We observed that LLMs' normative priors about the internal characteristics of good science extracted from their self-talk, e.g., theoretical rigor, were systematically updated toward posteriors that emphasize storytelling about external connections, such as how the work is positioned and connected within and across literatures. Human reviewers tend to explicitly reward aspects that moderately align with LLMs' normative priors (correlation = 0.49) but avoid articulating contextualization and storytelling posteriors in their review comments (correlation = -0.14), despite giving implicit reward to them with positive scores. These patterns are robust across different models and out-of-sample judgments. We discuss the broad applicability of our proposed framework, leveraging LLMs as diagnostic tools to amplify and surface the tacit codes underlying human society, enabling public discussion of revealed values and more precisely targeted responsible AI.
>
---
#### [replaced 005] Position: The Pitfalls of Over-Alignment: Overly Caution Health-Related Responses From LLMs are Unethical and Dangerous
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2509.08833v2](http://arxiv.org/pdf/2509.08833v2)**

> **作者:** Wenqi Marshall Guo; Yiyang Du; Heidi J. S. Tworek; Shan Du
>
> **摘要:** Large Language Models (LLMs) are usually aligned with "human values/preferences" to prevent harmful output. Discussions around the alignment of Large Language Models (LLMs) generally focus on preventing harmful outputs. However, in this paper, we argue that in health-related queries, over-alignment-leading to overly cautious responses-can itself be harmful, especially for people with anxiety and obsessive-compulsive disorder (OCD). This is not only unethical but also dangerous to the user, both mentally and physically. We also showed qualitative results that some LLMs exhibit varying degrees of alignment. Finally, we call for the development of LLMs with stronger reasoning capabilities that provide more tailored and nuanced responses to health queries. Warning: This paper contains materials that could trigger health anxiety or OCD. Dataset and full results can be found in https://github.com/weathon/over-alignment.
>
---
#### [replaced 006] Navigating the EU AI Act: Foreseeable Challenges in Qualifying Deep Learning-Based Automated Inspections of Class III Medical Devices
- **分类: cs.CY; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.20144v3](http://arxiv.org/pdf/2508.20144v3)**

> **作者:** Julio Zanon Diaz; Tommy Brennan; Peter Corcoran
>
> **备注:** Critical Review article
>
> **摘要:** As deep learning (DL) technologies advance, their application in automated visual inspection for Class III medical devices offers significant potential to enhance quality assurance and reduce human error. However, the adoption of such AI-based systems introduces new regulatory complexities-particularly under the EU Artificial Intelligence (AI) Act, which imposes high-risk system obligations that differ in scope and depth from established regulatory frameworks such as the Medical Device Regulation (MDR) and the U.S. FDA Quality System Regulation (QSR). This paper presents a high-level technical assessment of the foreseeable challenges that manufacturers are likely to encounter when qualifying DL-based automated inspections -- specifically static models -- within the existing medical device compliance landscape. It examines divergences in risk management principles, dataset governance, model validation, explainability requirements, and post-deployment monitoring obligations. The discussion also explores potential implementation strategies and highlights areas of uncertainty, including data retention burdens, global compliance implications, and the practical difficulties of achieving statistical significance in validation with limited defect data. Disclaimer: This paper presents a technical perspective and does not constitute legal or regulatory advice.
>
---
#### [replaced 007] What Makes AI Applications Acceptable or Unacceptable? A Predictive Moral Framework
- **分类: cs.CY; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.19317v2](http://arxiv.org/pdf/2508.19317v2)**

> **作者:** Kimmo Eriksson; Simon Karlsson; Irina Vartanova; Pontus Strimling
>
> **备注:** 15 pages + supplementary materials, 3 figures
>
> **摘要:** As artificial intelligence rapidly transforms society, developers and policymakers struggle to anticipate which applications will face public moral resistance. We propose that these judgments are not idiosyncratic but systematic and predictable. In a large, preregistered study (N = 587, U.S. representative sample), we used a comprehensive taxonomy of 100 AI applications spanning personal and organizational contexts-including both functional uses and the moral treatment of AI itself. In participants' collective judgment, applications ranged from highly unacceptable to fully acceptable. We found this variation was strongly predictable: five core moral qualities-perceived risk, benefit, dishonesty, unnaturalness, and reduced accountability-collectively explained over 90% of the variance in acceptability ratings. The framework demonstrated strong predictive power across all domains and successfully predicted individual-level judgments for held-out applications. These findings reveal that a structured moral psychology underlies public evaluation of new technologies, offering a powerful tool for anticipating public resistance and guiding responsible innovation in AI.
>
---
#### [replaced 008] Epistemic Diversity and Knowledge Collapse in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.04226v2](http://arxiv.org/pdf/2510.04226v2)**

> **作者:** Dustin Wright; Sarah Masud; Jared Moore; Srishti Yadav; Maria Antoniak; Chan Young Park; Isabelle Augenstein
>
> **备注:** 16 pages; 8 figures, 4 tables v2 changelog: Fixed the modeling for table 3, random effect is the model version
>
> **摘要:** Large language models (LLMs) tend to generate lexically, semantically, and stylistically homogenous texts. This poses a risk of knowledge collapse, where homogenous LLMs mediate a shrinking in the range of accessible information over time. Existing works on homogenization are limited by a focus on closed-ended multiple-choice setups or fuzzy semantic features, and do not look at trends across time and cultural contexts. To overcome this, we present a new methodology to measure epistemic diversity, i.e., variation in real-world claims in LLM outputs, which we use to perform a broad empirical study of LLM knowledge collapse. We test 27 LLMs, 155 topics covering 12 countries, and 200 prompt variations sourced from real user chats. For the topics in our study, we show that while newer models tend to generate more diverse claims, nearly all models are less epistemically diverse than a basic web search. We find that model size has a negative impact on epistemic diversity, while retrieval-augmented generation (RAG) has a positive impact, though the improvement from RAG varies by the cultural context. Finally, compared to a traditional knowledge source (Wikipedia), we find that country-specific claims reflect the English language more than the local one, highlighting a gap in epistemic representation
>
---
#### [replaced 009] How Malicious AI Swarms Can Threaten Democracy: The Fusion of Agentic AI and LLMs Marks a New Frontier in Information Warfare
- **分类: cs.CY; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06299v3](http://arxiv.org/pdf/2506.06299v3)**

> **作者:** Daniel Thilo Schroeder; Meeyoung Cha; Andrea Baronchelli; Nick Bostrom; Nicholas A. Christakis; David Garcia; Amit Goldenberg; Yara Kyrychenko; Kevin Leyton-Brown; Nina Lutz; Gary Marcus; Filippo Menczer; Gordon Pennycook; David G. Rand; Maria Ressa; Frank Schweitzer; Christopher Summerfield; Audrey Tang; Jay J. Van Bavel; Sander van der Linden; Dawn Song; Jonas R. Kunst
>
> **备注:** 15 pages, 1 figure
>
> **摘要:** Public opinion manipulation has entered a new phase, amplifying its roots in rhetoric and propaganda. Advances in large language models (LLMs) and autonomous agents now let influence campaigns reach unprecedented scale and precision. Researchers warn AI could foster mass manipulation. Generative tools can expand propaganda output without sacrificing credibility and inexpensively create election falsehoods that are rated as more human-like than those written by humans. Techniques meant to refine AI reasoning, such as chain-of-thought prompting, can just as effectively be used to generate more convincing falsehoods. Enabled by these capabilities, another disruptive threat is emerging: swarms of collaborative, malicious AI agents. Fusing LLM reasoning with multi-agent architectures, these systems are capable of coordinating autonomously, infiltrating communities, and fabricating consensus cheaply. By adaptively mimicking human social dynamics, they threaten democracy.
>
---
#### [replaced 010] Fair Play in the Newsroom: Actor-Based Filtering Gender Discrimination in Text Corpora
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2508.13169v2](http://arxiv.org/pdf/2508.13169v2)**

> **作者:** Stefanie Urchs; Veronika Thurner; Matthias Aßenmacher; Christian Heumann; Stephanie Thiemichen
>
> **摘要:** Language corpora are the foundation of most natural language processing research, yet they often reproduce structural inequalities. One such inequality is gender discrimination in how actors are represented, which can distort analyses and perpetuate discriminatory outcomes. This paper introduces a user-centric, actor-level pipeline for detecting and mitigating gender discrimination in large-scale text corpora. By combining discourse-aware analysis with metrics for sentiment, syntactic agency, and quotation styles, our method enables both fine-grained auditing and exclusion-based balancing. Applied to the taz2024full corpus of German newspaper articles (1980-2024), the pipeline yields a more gender-balanced dataset while preserving core dynamics of the source material. Our findings show that structural asymmetries can be reduced through systematic filtering, though subtler biases in sentiment and framing remain. We release the tools and reports to support further research in discourse-based fairness auditing and equitable corpus construction.
>
---
#### [replaced 011] Cosmos 1.0: a multidimensional map of the emerging technology frontier
- **分类: cs.CY; cs.DL; cs.SI; econ.GN; q-fin.EC**

- **链接: [http://arxiv.org/pdf/2505.10591v2](http://arxiv.org/pdf/2505.10591v2)**

> **作者:** Xian Gong; Paul X. McCarthy; Colin Griffith; Claire McFarland; Marian-Andrei Rizoiu
>
> **摘要:** This paper introduces the Cosmos 1.0 dataset and describes a novel methodology for creating and mapping a universe of technologies, adjacent concepts, and entities. We utilise various source data that contain a rich diversity and breadth of contemporary knowledge. The Cosmos 1.0 dataset comprises 23,544 technology-adjacent entities (TA23k) with a hierarchical structure and eight categories of external indices. Each entity is represented by a 100-dimensional contextual embedding vector, which we use to assign it to seven thematic tech-clusters (TC7) and three meta tech-clusters (TC3). We manually verify 100 emerging technologies (ET100). This dataset is enriched with additional indices specifically developed to assess the landscape of emerging technologies, including the Technology Awareness Index, Generality Index, Deeptech, and Age of Tech Index. The dataset incorporates extensive metadata sourced from Wikipedia and linked data from third-party sources such as Crunchbase, Google Books, OpenAlex and Google Scholar, which are used to validate the relevance and accuracy of the constructed indices.
>
---
#### [replaced 012] AWARE, Beyond Sentence Boundaries: A Contextual Transformer Framework for Identifying Cultural Capital in STEM Narratives
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.04983v2](http://arxiv.org/pdf/2510.04983v2)**

> **作者:** Khalid Mehtab Khan; Anagha Kulkarni
>
> **摘要:** Identifying cultural capital (CC) themes in student reflections can offer valuable insights that help foster equitable learning environments in classrooms. However, themes such as aspirational goals or family support are often woven into narratives, rather than appearing as direct keywords. This makes them difficult to detect for standard NLP models that process sentences in isolation. The core challenge stems from a lack of awareness, as standard models are pre-trained on general corpora, leaving them blind to the domain-specific language and narrative context inherent to the data. To address this, we introduce AWARE, a framework that systematically attempts to improve a transformer model's awareness for this nuanced task. AWARE has three core components: 1) Domain Awareness, adapting the model's vocabulary to the linguistic style of student reflections; 2) Context Awareness, generating sentence embeddings that are aware of the full essay context; and 3) Class Overlap Awareness, employing a multi-label strategy to recognize the coexistence of themes in a single sentence. Our results show that by making the model explicitly aware of the properties of the input, AWARE outperforms a strong baseline by 2.1 percentage points in Macro-F1 and shows considerable improvements across all themes. This work provides a robust and generalizable methodology for any text classification task in which meaning depends on the context of the narrative.
>
---
