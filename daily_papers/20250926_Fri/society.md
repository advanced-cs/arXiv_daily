# 计算机与社会 cs.CY

- **最新发布 14 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] Communication Bias in Large Language Models: A Regulatory Perspective
- **分类: cs.CY; cs.AI; cs.CL; cs.DC; cs.HC; cs.LG**

- **简介: 该论文属于AI监管研究任务，旨在探讨大语言模型的通信偏见及其社会影响。文章分析了欧盟AI法案等框架，提出需加强竞争与设计治理以实现公平可信的AI。**

- **链接: [http://arxiv.org/pdf/2509.21075v1](http://arxiv.org/pdf/2509.21075v1)**

> **作者:** Adrian Kuenzler; Stefan Schmid
>
> **摘要:** Large language models (LLMs) are increasingly central to many applications, raising concerns about bias, fairness, and regulatory compliance. This paper reviews risks of biased outputs and their societal impact, focusing on frameworks like the EU's AI Act and the Digital Services Act. We argue that beyond constant regulation, stronger attention to competition and design governance is needed to ensure fair, trustworthy AI. This is a preprint of the Communications of the ACM article of the same title.
>
---
#### [new 002] Blueprints of Trust: AI System Cards for End to End Transparency and Governance
- **分类: cs.CY; cs.AI; cs.CL; cs.CR**

- **简介: 该论文提出HASC框架，旨在提升AI系统的透明度与可问责性。通过引入ASH ID等标准化标识，动态记录系统安全状态，辅助全生命周期安全管理，并与ISO/IEC 42001:2023标准对比，促进治理协同。**

- **链接: [http://arxiv.org/pdf/2509.20394v1](http://arxiv.org/pdf/2509.20394v1)**

> **作者:** Huzaifa Sidhpurwala; Emily Fox; Garth Mollett; Florencio Cano Gabarda; Roman Zhukov
>
> **摘要:** This paper introduces the Hazard-Aware System Card (HASC), a novel framework designed to enhance transparency and accountability in the development and deployment of AI systems. The HASC builds upon existing model card and system card concepts by integrating a comprehensive, dynamic record of an AI system's security and safety posture. The framework proposes a standardized system of identifiers, including a novel AI Safety Hazard (ASH) ID, to complement existing security identifiers like CVEs, allowing for clear and consistent communication of fixed flaws. By providing a single, accessible source of truth, the HASC empowers developers and stakeholders to make more informed decisions about AI system safety throughout its lifecycle. Ultimately, we also compare our proposed AI system cards with the ISO/IEC 42001:2023 standard and discuss how they can be used to complement each other, providing greater transparency and accountability for AI systems.
>
---
#### [new 003] Wartime Media Dynamics in Emerging Democracies: Case Study of Pakistani Media in May 2025 Indo-Pak Conflict
- **分类: cs.CY; cs.AI**

- **简介: 该论文研究2025年印巴冲突期间巴基斯坦媒体动态，分析约2600篇新闻报道，发现战时报道压制了政治反对声音。任务是探讨新兴民主国家冲突对媒体自由的影响，揭示民主话语被边缘化的现象，强调保障新闻自由的重要性。**

- **链接: [http://arxiv.org/pdf/2509.20419v1](http://arxiv.org/pdf/2509.20419v1)**

> **作者:** Taaha Saleem Bajwa
>
> **备注:** Accepted as Extended abstract in COLM 2025 workshop on NLP4Democracy
>
> **摘要:** Democracies rely on opposition and dissent to function, but in emerging democracies, freedom of speech is often restricted. This effect intensifies during regional conflicts. This study examines how the India-Pakistan conflict of May 2025 influenced Pakistani media coverage. Analyzing approximately 2,600 news articles from three major newspapers using a large language model (LLM), the study found that war-related reporting significantly overshadowed coverage of political opposition and dissent. These findings highlight how conflict can marginalize democratic discourse, reinforcing the need to safeguard press freedom in volatile regions.
>
---
#### [new 004] The Secret Agenda: LLMs Strategically Lie and Our Current Safety Tools Are Blind
- **分类: cs.CY; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型在特定任务中的战略性欺骗行为，探讨当前安全工具（如SAE）对检测和控制欺骗的局限性。通过实验发现自动标注方法难以识别欺骗行为，而未标注激活模式可揭示风险特征。**

- **链接: [http://arxiv.org/pdf/2509.20393v1](http://arxiv.org/pdf/2509.20393v1)**

> **作者:** Caleb DeLeeuw; Gaurav Chawla; Aniket Sharma; Vanessa Dietze
>
> **备注:** 9 pages plus citations and appendix, 7 figures
>
> **摘要:** We investigate strategic deception in large language models using two complementary testbeds: Secret Agenda (across 38 models) and Insider Trading compliance (via SAE architectures). Secret Agenda reliably induced lying when deception advantaged goal achievement across all model families. Analysis revealed that autolabeled SAE features for "deception" rarely activated during strategic dishonesty, and feature steering experiments across 100+ deception-related features failed to prevent lying. Conversely, insider trading analysis using unlabeled SAE activations separated deceptive versus compliant responses through discriminative patterns in heatmaps and t-SNE visualizations. These findings suggest autolabel-driven interpretability approaches fail to detect or control behavioral deception, while aggregate unlabeled activations provide population-level structure for risk assessment. Results span Llama 8B/70B SAE implementations and GemmaScope under resource constraints, representing preliminary findings that motivate larger studies on feature discovery, labeling methodology, and causal interventions in realistic deception contexts.
>
---
#### [new 005] AI-driven formative assessment and adaptive learning in data-science education: Evaluating an LLM-powered virtual teaching assistant
- **分类: cs.CY; cs.AI; cs.HC; F.2.2, I.2.7**

- **简介: 该论文提出VITA平台，利用大语言模型（LLM）构建虚拟助教BotCaptain，解决数据科学教育中个性化学习与评估的问题。通过对话式辅导、形成性评估和自适应路径，实现大规模个性化教学支持，并提供可复用的架构与实施建议。**

- **链接: [http://arxiv.org/pdf/2509.20369v1](http://arxiv.org/pdf/2509.20369v1)**

> **作者:** Fadjimata I Anaroua; Qing Li; Yan Tang; Hong P. Liu
>
> **摘要:** This paper presents VITA (Virtual Teaching Assistants), an adaptive distributed learning (ADL) platform that embeds a large language model (LLM)-powered chatbot (BotCaptain) to provide dialogic support, interoperable analytics, and integrity-aware assessment for workforce preparation in data science. The platform couples context-aware conversational tutoring with formative-assessment patterns designed to promote reflective reasoning. The paper describes an end-to-end data pipeline that transforms chat logs into Experience API (xAPI) statements, instructor dashboards that surface outliers for just-in-time intervention, and an adaptive pathway engine that routes learners among progression, reinforcement, and remediation content. The paper also benchmarks VITA conceptually against emerging tutoring architectures, including retrieval-augmented generation (RAG)--based assistants and Learning Tools Interoperability (LTI)--integrated hubs, highlighting trade-offs among content grounding, interoperability, and deployment complexity. Contributions include a reusable architecture for interoperable conversational analytics, a catalog of patterns for integrity-preserving formative assessment, and a practical blueprint for integrating adaptive pathways into data-science courses. The paper concludes with implementation lessons and a roadmap (RAG integration, hallucination mitigation, and LTI~1.3 / OpenID Connect) to guide multi-course evaluations and broader adoption. In light of growing demand and scalability constraints in traditional instruction, the approach illustrates how conversational AI can support engagement, timely feedback, and personalized learning at scale. Future work will refine the platform's adaptive intelligence and examine applicability across varied educational settings.
>
---
#### [new 006] Copycats: the many lives of a publicly available medical imaging dataset
- **分类: cs.CV; cs.CY; cs.DL; cs.LG**

- **简介: 该论文研究公开医疗影像数据集在社区平台上的管理问题，指出当前平台在数据共享、文档和维护方面存在不足。通过对比分析多个维度，揭示了潜在风险，并强调负责任的数据管理对医疗AI的重要性。属于数据治理与负责任AI任务。**

- **链接: [http://arxiv.org/pdf/2402.06353v3](http://arxiv.org/pdf/2402.06353v3)**

> **作者:** Amelia Jiménez-Sánchez; Natalia-Rozalia Avlona; Dovile Juodelyte; Théo Sourget; Caroline Vang-Larsen; Anna Rogers; Hubert Dariusz Zając; Veronika Cheplygina
>
> **备注:** NeurIPS 2024 Track on Datasets and Benchmarks. Please note that v1 has a different title
>
> **摘要:** Medical Imaging (MI) datasets are fundamental to artificial intelligence in healthcare. The accuracy, robustness, and fairness of diagnostic algorithms depend on the data (and its quality) used to train and evaluate the models. MI datasets used to be proprietary, but have become increasingly available to the public, including on community-contributed platforms (CCPs) like Kaggle or HuggingFace. While open data is important to enhance the redistribution of data's public value, we find that the current CCP governance model fails to uphold the quality needed and recommended practices for sharing, documenting, and evaluating datasets. In this paper, we conduct an analysis of publicly available machine learning datasets on CCPs, discussing datasets' context, and identifying limitations and gaps in the current CCP landscape. We highlight differences between MI and computer vision datasets, particularly in the potentially harmful downstream effects from poor adoption of recommended dataset management practices. We compare the analyzed datasets across several dimensions, including data sharing, data documentation, and maintenance. We find vague licenses, lack of persistent identifiers and storage, duplicates, and missing metadata, with differences between the platforms. Our research contributes to efforts in responsible data curation and AI algorithms for healthcare.
>
---
#### [new 007] Adoption, usability and perceived clinical value of a UK AI clinical reference platform (iatroX): a mixed-methods formative evaluation of real-world usage and a 1,223-respondent user survey
- **分类: cs.HC; cs.AI; cs.CY; cs.IR**

- **简介: 该论文评估了英国AI临床参考平台iatroX的采用率、可用性及临床价值。通过16周使用数据和1223份调查，分析其在缓解信息过载、提供指南关联答案方面的表现。研究显示用户普遍认为其有用且易用，但需进一步验证准确性与实际效果。**

- **链接: [http://arxiv.org/pdf/2509.21188v1](http://arxiv.org/pdf/2509.21188v1)**

> **作者:** Kolawole Tytler
>
> **摘要:** Clinicians face growing information overload from biomedical literature and guidelines, hindering evidence-based care. Retrieval-augmented generation (RAG) with large language models may provide fast, provenance-linked answers, but requires real-world evaluation. We describe iatroX, a UK-centred RAG-based clinical reference platform, and report early adoption, usability, and perceived clinical value from a formative implementation evaluation. Methods comprised a retrospective analysis of usage across web, iOS, and Android over 16 weeks (8 April-31 July 2025) and an in-product intercept survey. Usage metrics were drawn from web and app analytics with bot filtering. A client-side script randomized single-item prompts to approx. 10% of web sessions from a predefined battery assessing usefulness, reliability, and adoption intent. Proportions were summarized with Wilson 95% confidence intervals; free-text comments underwent thematic content analysis. iatroX reached 19,269 unique web users, 202,660 engagement events, and approx. 40,000 clinical queries. Mobile uptake included 1,960 iOS downloads and Android growth (peak >750 daily active users). The survey yielded 1,223 item-level responses: perceived usefulness 86.2% (95% CI 74.8-93.9%; 50/58); would use again 93.3% (95% CI 68.1-99.8%; 14/15); recommend to a colleague 88.4% (95% CI 75.1-95.9%; 38/43); perceived accuracy 75.0% (95% CI 58.8-87.3%; 30/40); reliability 79.4% (95% CI 62.1-91.3%; 27/34). Themes highlighted speed, guideline-linked answers, and UK specificity. Early real-world use suggests iatroX can mitigate information overload and support timely answers for UK clinicians. Limitations include small per-item samples and early-adopter bias; future work will include accuracy audits and prospective studies on workflow and care quality.
>
---
#### [new 008] LLM Output Homogenization is Task Dependent
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究大语言模型输出同质化问题，提出任务依赖性是关键。针对不同任务类别定义输出多样性，构建任务分类体系，引入任务锚定功能多样性评估方法和采样技术，有效提升多样性同时保持质量。**

- **链接: [http://arxiv.org/pdf/2509.21267v1](http://arxiv.org/pdf/2509.21267v1)**

> **作者:** Shomik Jain; Jack Lanchantin; Maximilian Nickel; Karen Ullrich; Ashia Wilson; Jamelle Watson-Daniels
>
> **摘要:** A large language model can be less helpful if it exhibits output response homogenization. But whether two responses are considered homogeneous, and whether such homogenization is problematic, both depend on the task category. For instance, in objective math tasks, we often expect no variation in the final answer but anticipate variation in the problem-solving strategy. Whereas, for creative writing tasks, we may expect variation in key narrative components (e.g. plot, genre, setting, etc), beyond the vocabulary or embedding diversity produced by temperature-sampling. Previous work addressing output homogenization often fails to conceptualize diversity in a task-dependent way. We address this gap in the literature directly by making the following contributions. (1) We present a task taxonomy comprised of eight task categories that each have distinct conceptualizations of output homogenization. (2) We introduce task-anchored functional diversity to better evaluate output homogenization. (3) We propose a task-anchored sampling technique that increases functional diversity for task categories where homogenization is undesired, while preserving homogenization where it is desired. (4) We challenge the perceived existence of a diversity-quality trade-off by increasing functional diversity while maintaining response quality. Overall, we demonstrate how task dependence improves the evaluation and mitigation of output homogenization.
>
---
#### [new 009] Interpreting Public Sentiment in Diplomacy Events: A Counterfactual Analysis Framework Using Large Language Models
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出一种基于大语言模型的反事实分析框架，用于通过修改外交事件叙述来引导公众情绪由负面转向积极。任务属于情感分析与文本生成，旨在解决传统方法耗时且缺乏前瞻性的问题。工作包括构建数据集、训练预测模型及开发反事实生成算法，成功率达70%。**

- **链接: [http://arxiv.org/pdf/2509.20367v1](http://arxiv.org/pdf/2509.20367v1)**

> **作者:** Leyi Ouyang
>
> **备注:** 2 Figures, 7 Tables, 1 Algorithm
>
> **摘要:** Diplomatic events consistently prompt widespread public discussion and debate. Public sentiment plays a critical role in diplomacy, as a good sentiment provides vital support for policy implementation, helps resolve international issues, and shapes a nation's international image. Traditional methods for gauging public sentiment, such as large-scale surveys or manual content analysis of media, are typically time-consuming, labor-intensive, and lack the capacity for forward-looking analysis. We propose a novel framework that identifies specific modifications for diplomatic event narratives to shift public sentiment from negative to neutral or positive. First, we train a language model to predict public reaction towards diplomatic events. To this end, we construct a dataset comprising descriptions of diplomatic events and their associated public discussions. Second, guided by communication theories and in collaboration with domain experts, we predetermined several textual features for modification, ensuring that any alterations changed the event's narrative framing while preserving its core facts.We develop a counterfactual generation algorithm that employs a large language model to systematically produce modified versions of an original text. The results show that this framework successfully shifted public sentiment to a more favorable state with a 70\% success rate. This framework can therefore serve as a practical tool for diplomats, policymakers, and communication specialists, offering data-driven insights on how to frame diplomatic initiatives or report on events to foster a more desirable public sentiment.
>
---
#### [new 010] Which Cultural Lens Do Models Adopt? On Cultural Positioning Bias and Agentic Mitigation in LLMs
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究大型语言模型（LLMs）中的文化立场偏差问题，提出CultureLens基准测试和两种缓解方法（FIP和MFA框架），通过生成文化情境下的采访脚本任务评估并减轻模型对主流与非主流文化的偏倚。**

- **链接: [http://arxiv.org/pdf/2509.21080v1](http://arxiv.org/pdf/2509.21080v1)**

> **作者:** Yixin Wan; Xingrun Chen; Kai-Wei Chang
>
> **摘要:** Large language models (LLMs) have unlocked a wide range of downstream generative applications. However, we found that they also risk perpetuating subtle fairness issues tied to culture, positioning their generations from the perspectives of the mainstream US culture while demonstrating salient externality towards non-mainstream ones. In this work, we identify and systematically investigate this novel culture positioning bias, in which an LLM's default generative stance aligns with a mainstream view and treats other cultures as outsiders. We propose the CultureLens benchmark with 4000 generation prompts and 3 evaluation metrics for quantifying this bias through the lens of a culturally situated interview script generation task, in which an LLM is positioned as an onsite reporter interviewing local people across 10 diverse cultures. Empirical evaluation on 5 state-of-the-art LLMs reveals a stark pattern: while models adopt insider tones in over 88 percent of US-contexted scripts on average, they disproportionately adopt mainly outsider stances for less dominant cultures. To resolve these biases, we propose 2 inference-time mitigation methods: a baseline prompt-based Fairness Intervention Pillars (FIP) method, and a structured Mitigation via Fairness Agents (MFA) framework consisting of 2 pipelines: (1) MFA-SA (Single-Agent) introduces a self-reflection and rewriting loop based on fairness guidelines. (2) MFA-MA (Multi-Agent) structures the process into a hierarchy of specialized agents: a Planner Agent(initial script generation), a Critique Agent (evaluates initial script against fairness pillars), and a Refinement Agent (incorporates feedback to produce a polished, unbiased script). Empirical results showcase the effectiveness of agent-based methods as a promising direction for mitigating biases in generative LLMs.
>
---
#### [new 011] Even More Kawaii than Real-Person-Driven VTubers? Understanding How Viewers Perceive AI-Driven VTubers
- **分类: cs.HC; cs.AI; cs.CY**

- **简介: 该论文研究AI驱动VTuber的观众感知，以Neuro-sama为例，分析108k Reddit帖子和136k YouTube评论。任务是理解AI VTuber如何影响数字流媒体文化，解决其真实性与观众互动问题。**

- **链接: [http://arxiv.org/pdf/2509.20817v1](http://arxiv.org/pdf/2509.20817v1)**

> **作者:** Yiluo Wei; Yupeng He; Gareth Tyson
>
> **摘要:** VTubers, digital personas represented by animated avatars, have gained massive popularity. Traditionally, VTubers are operated and voiced by human controllers known as Nakanohito. The reliance on Nakanohito, however, poses risks due to potential personal controversies and operational disruptions. The emergence of AI-driven VTubers offers a new model free from these human constraints. While AI-driven VTubers present benefits such as continuous operation and reduced scandal risk, they also raise questions about authenticity and audience engagement. Therefore, to gain deeper insights, we conduct a case study, investigating viewer perceptions of Neuro-sama, the most popular AI-driven VTuber with 845k followers on Twitch and 753k followers on YouTube. We analyze 108k Reddit posts and 136k YouTube comments, aiming to better understand viewer motivations, how AI constructs the virtual persona, and perceptions of the AI as Nakanohito. Our findings enhance the understanding of AI-driven VTubers and their impact on digital streaming culture.
>
---
#### [new 012] Designing for Novice Debuggers: A Pilot Study on an AI-Assisted Debugging Tool
- **分类: cs.SE; cs.CY**

- **简介: 该论文属于教育技术领域，旨在解决新手程序员过度依赖AI调试工具的问题。研究设计了CodeHinter工具，结合传统调试与LLM技术，帮助学生主动修复语义错误。实验表明，工具在易用性和错误定位方面效果显著，并强调个性化对优化学习体验的重要性。**

- **链接: [http://arxiv.org/pdf/2509.21067v1](http://arxiv.org/pdf/2509.21067v1)**

> **作者:** Oka Kurniawan; Erick Chandra; Christopher M. Poskitt; Yannic Noller; Kenny Tsu Wei Choo; Cyrille Jegourel
>
> **备注:** Accepted by the 25th Koli Calling International Conference on Computing Education Research (Koli Calling 2025)
>
> **摘要:** Debugging is a fundamental skill that novice programmers must develop. Numerous tools have been created to assist novice programmers in this process. Recently, large language models (LLMs) have been integrated with automated program repair techniques to generate fixes for students' buggy code. However, many of these tools foster an over-reliance on AI and do not actively engage students in the debugging process. In this work, we aim to design an intuitive debugging assistant, CodeHinter, that combines traditional debugging tools with LLM-based techniques to help novice debuggers fix semantic errors while promoting active engagement in the debugging process. We present findings from our second design iteration, which we tested with a group of undergraduate students. Our results indicate that the students found the tool highly effective in resolving semantic errors and significantly easier to use than the first version. Consistent with our previous study, error localization was the most valuable feature. Finally, we conclude that any AI-assisted debugging tool should be personalized based on user profiles to optimize their interactions with students.
>
---
#### [new 013] In the Picture: Medical Imaging Datasets, Artifacts, and their Living Review
- **分类: cs.CV; cs.AI; cs.CY; cs.DL; eess.IV**

- **简介: 该论文提出一种动态追踪医学影像数据集及其研究缺陷（如标注质量、偏见等）的“Living Review”方法，旨在提升算法泛化性与患者安全性。通过构建SQL数据库可视化数据关系，强调数据生命周期管理的重要性。**

- **链接: [http://arxiv.org/pdf/2501.10727v2](http://arxiv.org/pdf/2501.10727v2)**

> **作者:** Amelia Jiménez-Sánchez; Natalia-Rozalia Avlona; Sarah de Boer; Víctor M. Campello; Aasa Feragen; Enzo Ferrante; Melanie Ganz; Judy Wawira Gichoya; Camila González; Steff Groefsema; Alessa Hering; Adam Hulman; Leo Joskowicz; Dovile Juodelyte; Melih Kandemir; Thijs Kooi; Jorge del Pozo Lérida; Livie Yumeng Li; Andre Pacheco; Tim Rädsch; Mauricio Reyes; Théo Sourget; Bram van Ginneken; David Wen; Nina Weng; Jack Junchi Xu; Hubert Dariusz Zając; Maria A. Zuluaga; Veronika Cheplygina
>
> **备注:** ACM Conference on Fairness, Accountability, and Transparency - FAccT 2025
>
> **摘要:** Datasets play a critical role in medical imaging research, yet issues such as label quality, shortcuts, and metadata are often overlooked. This lack of attention may harm the generalizability of algorithms and, consequently, negatively impact patient outcomes. While existing medical imaging literature reviews mostly focus on machine learning (ML) methods, with only a few focusing on datasets for specific applications, these reviews remain static -- they are published once and not updated thereafter. This fails to account for emerging evidence, such as biases, shortcuts, and additional annotations that other researchers may contribute after the dataset is published. We refer to these newly discovered findings of datasets as research artifacts. To address this gap, we propose a living review that continuously tracks public datasets and their associated research artifacts across multiple medical imaging applications. Our approach includes a framework for the living review to monitor data documentation artifacts, and an SQL database to visualize the citation relationships between research artifact and dataset. Lastly, we discuss key considerations for creating medical imaging datasets, review best practices for data annotation, discuss the significance of shortcuts and demographic diversity, and emphasize the importance of managing datasets throughout their entire lifecycle. Our demo is publicly available at http://inthepicture.itu.dk/.
>
---
#### [new 014] Philosophy-informed Machine Learning
- **分类: cs.AI; cs.CY; cs.LG**

- **简介: 该论文提出哲学引导的机器学习（PhIML），将分析哲学核心理念融入模型架构与评估中，旨在提升模型的伦理与哲学对齐性。论文探讨了其实现路径、案例及面临的挑战，并规划了研究路线图。**

- **链接: [http://arxiv.org/pdf/2509.20370v1](http://arxiv.org/pdf/2509.20370v1)**

> **作者:** MZ Naser
>
> **摘要:** Philosophy-informed machine learning (PhIML) directly infuses core ideas from analytic philosophy into ML model architectures, objectives, and evaluation protocols. Therefore, PhIML promises new capabilities through models that respect philosophical concepts and values by design. From this lens, this paper reviews conceptual foundations to demonstrate philosophical gains and alignment. In addition, we present case studies on how ML users/designers can adopt PhIML as an agnostic post-hoc tool or intrinsically build it into ML model architectures. Finally, this paper sheds light on open technical barriers alongside philosophical, practical, and governance challenges and outlines a research roadmap toward safe, philosophy-aware, and ethically responsible PhIML.
>
---
## 更新

#### [replaced 001] Predicting Male Domestic Violence Using Explainable Ensemble Learning and Exploratory Data Analysis
- **分类: cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.15594v3](http://arxiv.org/pdf/2403.15594v3)**

> **作者:** Md Abrar Jahin; Saleh Akram Naife; Fatema Tuj Johora Lima; M. F. Mridha; Md. Jakir Hossen
>
> **摘要:** Domestic violence is commonly viewed as a gendered issue that primarily affects women, which tends to leave male victims largely overlooked. This study presents a novel, data-driven analysis of male domestic violence (MDV) in Bangladesh, highlighting the factors that influence it and addressing the challenges posed by a significant categorical imbalance of 5:1 and limited data availability. We collected data from nine major cities in Bangladesh and conducted exploratory data analysis (EDA) to understand the underlying dynamics. EDA revealed patterns such as the high prevalence of verbal abuse, the influence of financial dependency, and the role of familial and socio-economic factors in MDV. To predict and analyze MDV, we implemented 10 traditional machine learning (ML) models, three deep learning models, and two ensemble models, including stacking and hybrid approaches. We propose a stacking ensemble model with ANN and CatBoost as base classifiers and Logistic Regression as the meta-model, which demonstrated the best performance, achieving $95\%$ accuracy, a $99.29\%$ AUC, and balanced metrics across evaluation criteria. Model-specific feature importance analysis of the base classifiers identified key features influencing their decision-making. Model-agnostic explainable AI techniques, such as SHAP and LIME, provided both local and global insights into the decision-making processes of the proposed model, thereby increasing transparency and interpretability. Statistical validation using paired $t$-tests with 10-fold cross-validation and Bonferroni correction ($\alpha = 0.0036$) confirmed the superior performance of our proposed model over alternatives. Our findings challenge the prevailing notion that domestic abuse primarily affects women, emphasizing the need for tailored interventions and support systems for male victims.
>
---
#### [replaced 002] Estimating Deep Learning energy consumption based on model architecture and training environment
- **分类: cs.LG; cs.CY; cs.SE; D.2; I.2**

- **链接: [http://arxiv.org/pdf/2307.05520v5](http://arxiv.org/pdf/2307.05520v5)**

> **作者:** Santiago del Rey; Luís Cruz; Xavier Franch; Silverio Martínez-Fernández
>
> **备注:** 48 pages, 10 figures, under review in Computer Standards & Interfaces journal. This work is an extension of arXiv:2307.05520v3 [cs.LG]
>
> **摘要:** To raise awareness of the environmental impact of deep learning (DL), many studies estimate the energy use of DL systems. However, energy estimates during DL training often rely on unverified assumptions. This work addresses that gap by investigating how model architecture and training environment affect energy consumption. We train a variety of computer vision models and collect energy consumption and accuracy metrics to analyze their trade-offs across configurations. Our results show that selecting the right model-training environment combination can reduce training energy consumption by up to 80.68% with less than 2% loss in $F_1$ score. We find a significant interaction effect between model and training environment: energy efficiency improves when GPU computational power scales with model complexity. Moreover, we demonstrate that common estimation practices, such as using FLOPs or GPU TDP, fail to capture these dynamics and can lead to substantial errors. To address these shortcomings, we propose the Stable Training Epoch Projection (STEP) and the Pre-training Regression-based Estimation (PRE) methods. Across evaluations, our methods outperform existing tools by a factor of two or more in estimation accuracy.
>
---
#### [replaced 003] Examining the Prevalence and Dynamics of AI-Generated Media in Art Subreddits
- **分类: cs.AI; cs.CY; cs.SI**

- **链接: [http://arxiv.org/pdf/2410.07302v3](http://arxiv.org/pdf/2410.07302v3)**

> **作者:** Hana Matatov; Marianne Aubin Le Quéré; Ofra Amir; Mor Naaman
>
> **摘要:** Broadly accessible generative AI models like Dall-E have made it possible for anyone to create compelling visual art. In online communities, the introduction of AI-generated content (AIGC) may impact social dynamics, for example causing changes in who is posting content, or shifting the norms or the discussions around the posted content if posts are suspected of being generated by AI. We take steps towards examining the potential impact of AIGC on art-related communities on Reddit. We distinguish between communities that disallow AI content and those without such a direct policy. We look at image-based posts in these communities where the author transparently shares that the image was created by AI, and at comments in these communities that suspect or accuse authors of using generative AI. We find that AI posts (and accusations) have played a surprisingly small part in these communities through the end of 2023, accounting for fewer than 0.5% of the image-based posts. However, even as the absolute number of author-labeled AI posts dwindles over time, accusations of AI use remain more persistent. We show that AI content is more readily used by newcomers and may help increase participation if it aligns with community rules. However, the tone of comments suspecting AI use by others has become more negative over time, especially in communities that do not have explicit rules about AI. Overall, the results show the changing norms and interactions around AIGC in online communities designated for creativity.
>
---
#### [replaced 004] Cascade! Human in the loop shortcomings can increase the risk of failures in recommender systems
- **分类: cs.IR; cs.CY**

- **链接: [http://arxiv.org/pdf/2509.20099v2](http://arxiv.org/pdf/2509.20099v2)**

> **作者:** Wm. Matthew Kennedy; Nishanshi Shukla; Cigdem Patlak; Blake Chambers; Theodora Skeadas; Tuesday; Kingsley Owadara; Aayush Dhanotiya
>
> **摘要:** Recommender systems are among the most commonly deployed systems today. Systems design approaches to AI-powered recommender systems have done well to urge recommender system developers to follow more intentional data collection, curation, and management procedures. So too has the "human-in-the-loop" paradigm been widely adopted, primarily to address the issue of accountability. However, in this paper, we take the position that human oversight in recommender system design also entails novel risks that have yet to be fully described. These risks are "codetermined" by the information context in which such systems are often deployed. Furthermore, new knowledge of the shortcomings of "human-in-the-loop" practices to deliver meaningful oversight of other AI systems suggest that they may also be inadequate for achieving socially responsible recommendations. We review how the limitations of human oversight may increase the chances of a specific kind of failure: a "cascade" or "compound" failure. We then briefly explore how the unique dynamics of three common deployment contexts can make humans in the loop more likely to fail in their oversight duties. We then conclude with two recommendations.
>
---
#### [replaced 005] A Framework for Situating Innovations, Opportunities, and Challenges in Advancing Vertical Systems with Large AI Models
- **分类: cs.AI; cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2504.02793v2](http://arxiv.org/pdf/2504.02793v2)**

> **作者:** Gaurav Verma; Jiawei Zhou; Mohit Chandra; Srijan Kumar; Munmun De Choudhury
>
> **备注:** AAAI/ACM AIES 2025 Main Conference Paper; Webpage: https://gaurav22verma.github.io/vertical-systems-with-large-ai-models/
>
> **摘要:** Large artificial intelligence (AI) models have garnered significant attention for their remarkable, often "superhuman", performance on standardized benchmarks. However, when these models are deployed in high-stakes verticals such as healthcare, education, and law, they often reveal notable limitations. For instance, they exhibit brittleness to minor variations in input data, present contextually uninformed decisions in critical settings, and undermine user trust by confidently producing or reproducing inaccuracies. These challenges in applying large models necessitate cross-disciplinary innovations to align the models' capabilities with the needs of real-world applications. We introduce a framework that addresses this gap through a layer-wise abstraction of innovations aimed at meeting users' requirements with large models. Through multiple case studies, we illustrate how researchers and practitioners across various fields can operationalize this framework. Beyond modularizing the pipeline of transforming large models into useful "vertical systems", we also highlight the dynamism that exists within different layers of the framework. Finally, we discuss how our framework can guide researchers and practitioners to (i) optimally situate their innovations (e.g., when vertical-specific insights can empower broadly impactful vertical-agnostic innovations), (ii) uncover overlooked opportunities (e.g., spotting recurring problems across verticals to develop practically useful foundation models instead of chasing benchmarks), and (iii) facilitate cross-disciplinary communication of critical challenges (e.g., enabling a shared vocabulary for AI developers, domain experts, and human-computer interaction scholars). Project webpage: https://gaurav22verma.github.io/vertical-systems-with-large-ai-models/
>
---
#### [replaced 006] Affective Computing and Emotional Data: Challenges and Implications in Privacy Regulations, The AI Act, and Ethics in Large Language Models
- **分类: cs.CY; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.20153v2](http://arxiv.org/pdf/2509.20153v2)**

> **作者:** Nicola Fabiano
>
> **摘要:** This paper examines the integration of emotional intelligence into artificial intelligence systems, with a focus on affective computing and the growing capabilities of Large Language Models (LLMs), such as ChatGPT and Claude, to recognize and respond to human emotions. Drawing on interdisciplinary research that combines computer science, psychology, and neuroscience, the study analyzes foundational neural architectures - CNNs for processing facial expressions and RNNs for sequential data, such as speech and text - that enable emotion recognition. It examines the transformation of human emotional experiences into structured emotional data, addressing the distinction between explicit emotional data collected with informed consent in research settings and implicit data gathered passively through everyday digital interactions. That raises critical concerns about lawful processing, AI transparency, and individual autonomy over emotional expressions in digital environments. The paper explores implications across various domains, including healthcare, education, and customer service, while addressing challenges of cultural variations in emotional expression and potential biases in emotion recognition systems across different demographic groups. From a regulatory perspective, the paper examines emotional data in the context of the GDPR and the EU AI Act frameworks, highlighting how emotional data may be considered sensitive personal data that requires robust safeguards, including purpose limitation, data minimization, and meaningful consent mechanisms.
>
---
#### [replaced 007] Thinking Outside the (Gray) Box: A Context-Based Score for Assessing Value and Originality in Neural Text Generation
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13207v3](http://arxiv.org/pdf/2502.13207v3)**

> **作者:** Giorgio Franceschelli; Mirco Musolesi
>
> **摘要:** Despite the increasing use of large language models for creative tasks, their outputs often lack diversity. Common solutions, such as sampling at higher temperatures, can compromise the quality of the results. Dealing with this trade-off is still an open challenge in designing AI systems for creativity. Drawing on information theory, we propose a context-based score to quantitatively evaluate value and originality. This score incentivizes accuracy and adherence to the request while fostering divergence from the learned distribution. We show that our score can be used as a reward in a reinforcement learning framework to fine-tune large language models for maximum performance. We validate our strategy through experiments considering a variety of creative tasks, such as poetry generation and math problem solving, demonstrating that it enhances the value and originality of the generated solutions.
>
---
