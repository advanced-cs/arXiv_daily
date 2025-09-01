# 计算机与社会 cs.CY

- **最新发布 11 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Operational Validation of Large-Language-Model Agent Social Simulation: Evidence from Voat v/technology
- **分类: cs.CY; cs.SI; physics.soc-ph**

- **简介: 该论文验证大语言模型代理在模拟在线社区互动的有效性，通过构建Voat技术社区模拟，分析毒性动态与治理策略。研究基于YSocial框架，校准参数并运行30天实验，对比真实数据评估模拟的真实性与局限性。**

- **链接: [http://arxiv.org/pdf/2508.21740v1](http://arxiv.org/pdf/2508.21740v1)**

> **作者:** Aleksandar Tomašević; Darja Cvetković; Sara Major; Slobodan Maletić; Miroslav Anđelković; Ana Vranić; Boris Stupovski; Dušan Vudragović; Aleksandar Bogojević; Marija Mitrović Dankulov
>
> **摘要:** Large Language Models (LLMs) enable generative social simulations that can capture culturally informed, norm-guided interaction on online social platforms. We build a technology community simulation modeled on Voat, a Reddit-like alt-right news aggregator and discussion platform active from 2014 to 2020. Using the YSocial framework, we seed the simulation with a fixed catalog of technology links sampled from Voat's shared URLs (covering 30+ domains) and calibrate parameters to Voat's v/technology using samples from the MADOC dataset. Agents use a base, uncensored model (Dolphin 3.0, based on Llama 3.1 8B) and concise personas (demographics, political leaning, interests, education, toxicity propensity) to generate posts, replies, and reactions under platform rules for link and text submissions, threaded replies and daily activity cycles. We run a 30-day simulation and evaluate operational validity by comparing distributions and structures with matched Voat data: activity patterns, interaction networks, toxicity, and topic coverage. Results indicate familiar online regularities: similar activity rhythms, heavy-tailed participation, sparse low-clustering interaction networks, core-periphery structure, topical alignment with Voat, and elevated toxicity. Limitations of the current study include the stateless agent design and evaluation based on a single 30-day run, which constrains external validity and variance estimates. The simulation generates realistic discussions, often featuring toxic language, primarily centered on technology topics such as Big Tech and AI. This approach offers a valuable method for examining toxicity dynamics and testing moderation strategies within a controlled environment.
>
---
#### [new 002] Synthetic CVs To Build and Test Fairness-Aware Hiring Tools
- **分类: cs.CY; cs.IR; cs.LG**

- **简介: 论文提出合成CV数据集，用于测试公平性招聘工具，解决现有数据不足问题，构建1730份多样化CV作为基准。**

- **链接: [http://arxiv.org/pdf/2508.21179v1](http://arxiv.org/pdf/2508.21179v1)**

> **作者:** Jorge Saldivar; Anna Gatzioura; Carlos Castillo
>
> **摘要:** Algorithmic hiring has become increasingly necessary in some sectors as it promises to deal with hundreds or even thousands of applicants. At the heart of these systems are algorithms designed to retrieve and rank candidate profiles, which are usually represented by Curricula Vitae (CVs). Research has shown, however, that such technologies can inadvertently introduce bias, leading to discrimination based on factors such as candidates' age, gender, or national origin. Developing methods to measure, mitigate, and explain bias in algorithmic hiring, as well as to evaluate and compare fairness techniques before deployment, requires sets of CVs that reflect the characteristics of people from diverse backgrounds. However, datasets of these characteristics that can be used to conduct this research do not exist. To address this limitation, this paper introduces an approach for building a synthetic dataset of CVs with features modeled on real materials collected through a data donation campaign. Additionally, the resulting dataset of 1,730 CVs is presented, which we envision as a potential benchmarking standard for research on algorithmic hiring discrimination.
>
---
#### [new 003] From Drone Imagery to Livability Mapping: AI-powered Environment Perception in Rural China
- **分类: cs.CY; cs.CV**

- **简介: 该论文提出基于无人机影像与多模态大语言模型的农村宜居性评估框架，解决传统问卷与城市视觉方法在农村应用的局限，分析中国农村宜居性空间异质性及财政支出等核心影响因素，为乡村振兴政策提供数据支持。**

- **链接: [http://arxiv.org/pdf/2508.21738v1](http://arxiv.org/pdf/2508.21738v1)**

> **作者:** Weihuan Deng; Yaofu Huang; Luan Chen; Xun Li; Yao Yao
>
> **摘要:** With the deepening of poverty alleviation and rural revitalization strategies, improving the rural living environment and enhancing the quality of life have become key priorities. Rural livability is a key indicator for measuring the effectiveness of these efforts. Current measurement approaches face significant limitations, as questionnaire-based methods are difficult to scale, while urban-oriented visual perception methods are poorly suited for rural contexts. In this paper, a rural-specific livability assessment framework was proposed based on drone imagery and multimodal large language models (MLLMs). To comprehensively assess village livability, this study first used a top-down approach to collect large-scale drone imagery of 1,766 villages in 146 counties across China. In terms of the model framework, an efficient image comparison mechanism was developed, incorporating binary search interpolation to determine effective image pairs while reducing comparison iterations. Building on expert knowledge, a chain-of-thought prompting suitable for nationwide rural livability measurement was constructed, considering both living quality and ecological habitability dimensions. This approach enhanced the rationality and reliability of the livability assessment. Finally, this study characterized the spatial heterogeneity of rural livability across China and thoroughly analyzed its influential factors. The results show that: (1) The rural livability in China demonstrates a dual-core-periphery spatial pattern, radiating outward from Sichuan and Zhejiang provinces with declining gradients; (2) Among various influential factors, government fiscal expenditure emerged as the core determinant, with each unit increase corresponding to a 3.9 - 4.9 unit enhancement in livability. The findings provide valuable insights for rural construction policy-making.
>
---
#### [new 004] Developer Insights into Designing AI-Based Computer Perception Tools
- **分类: cs.HC; cs.AI; cs.CY**

- **简介: 本研究通过访谈开发者，探讨AI临床感知工具的设计挑战，提出四大设计优先级及跨学科合作建议，以平衡临床效用、用户接受度与伦理责任。**

- **链接: [http://arxiv.org/pdf/2508.21733v1](http://arxiv.org/pdf/2508.21733v1)**

> **作者:** Maya Guhan; Meghan E. Hurley; Eric A. Storch; John Herrington; Casey Zampella; Julia Parish-Morris; Gabriel Lázaro-Muñoz; Kristin Kostick-Quenet
>
> **备注:** 15 pages
>
> **摘要:** Artificial intelligence (AI)-based computer perception (CP) technologies use mobile sensors to collect behavioral and physiological data for clinical decision-making. These tools can reshape how clinical knowledge is generated and interpreted. However, effective integration of these tools into clinical workflows depends on how developers balance clinical utility with user acceptability and trustworthiness. Our study presents findings from 20 in-depth interviews with developers of AI-based CP tools. Interviews were transcribed and inductive, thematic analysis was performed to identify 4 key design priorities: 1) to account for context and ensure explainability for both patients and clinicians; 2) align tools with existing clinical workflows; 3) appropriately customize to relevant stakeholders for usability and acceptability; and 4) push the boundaries of innovation while aligning with established paradigms. Our findings highlight that developers view themselves as not merely technical architects but also ethical stewards, designing tools that are both acceptable by users and epistemically responsible (prioritizing objectivity and pushing clinical knowledge forward). We offer the following suggestions to help achieve this balance: documenting how design choices around customization are made, defining limits for customization choices, transparently conveying information about outputs, and investing in user training. Achieving these goals will require interdisciplinary collaboration between developers, clinicians, and ethicists.
>
---
#### [new 005] Harnessing IoT and Generative AI for Weather-Adaptive Learning in Climate Resilience Education
- **分类: cs.HC; cs.AI; cs.CY; cs.LG; cs.SE**

- **简介: 该论文提出FACTS系统，整合物联网与生成式AI，通过实时气象数据与个性化反馈提升气候韧性教育效果，解决传统教育缺乏动态适应性的问题。**

- **链接: [http://arxiv.org/pdf/2508.21666v1](http://arxiv.org/pdf/2508.21666v1)**

> **作者:** Imran S. A. Khan; Emmanuel G. Blanchard; Sébastien George
>
> **摘要:** This paper introduces the Future Atmospheric Conditions Training System (FACTS), a novel platform that advances climate resilience education through place-based, adaptive learning experiences. FACTS combines real-time atmospheric data collected by IoT sensors with curated resources from a Knowledge Base to dynamically generate localized learning challenges. Learner responses are analyzed by a Generative AI powered server, which delivers personalized feedback and adaptive support. Results from a user evaluation indicate that participants found the system both easy to use and effective for building knowledge related to climate resilience. These findings suggest that integrating IoT and Generative AI into atmospherically adaptive learning technologies holds significant promise for enhancing educational engagement and fostering climate awareness.
>
---
#### [new 006] Uncertainties within Weather Regime definitions for the Euro-Atlantic sector in ERA5 and CMIP6
- **分类: physics.ao-ph; cs.CY; physics.soc-ph**

- **简介: 该论文评估欧洲大西洋天气型定义的不确定性及其对能源风险的影响，比较三种方法并分析CMIP6模型适用性，发现方法和训练期显著影响结果，部分模型存在不稳定性。**

- **链接: [http://arxiv.org/pdf/2508.21701v1](http://arxiv.org/pdf/2508.21701v1)**

> **作者:** Lotte Hompes; Swinda K. J. Falkena; Laurens P. Stoop
>
> **摘要:** Certain Weather Regimes (WR) are associated with a higher risk of energy shortages, i.e. Blocking regimes for European winters. However, there are many uncertainties tied to the implementation of WRs and associated risks in the energy sector. Especially the impact of climate change is unknown. We investigate these uncertainties by looking at three methodologically diverse Euro-Atlantic WR definitions. We carry out a thorough validation of these methods and analyse their methodological and spatio-temporal sensitivity using ERA5 data. Furthermore, we look into the suitability of CMIP6 models for WR based impact assessments. Our sensitivity assessment showed that the persistence and occurrence of regimes are sensitive to small changes in the methodology. We show that the training period used has a very significant impact on the persistence and occurrence of the regimes found. For both WR4 and WR7, this results in instability of the regime patterns. All CMIP6 models investigated show instability of the regimes. Meaning that the normalised distance between the CMIP6 model regimes and our baseline regimes exceeds 0.4 or are visually extremely dissimilar. Only the WR4 regimes clustered on historical CMIP6 model data consistently have a normalised distance to our baseline regimes smaller than 0.4 and are visually identifiable. The WR6 definition exceeds the normalised distance threshold for all investigated CMIP6 experiments. Though all CMIP6 model experiments clustered with the WR7 definition have a normalised distance to the baseline regimes below 0.4, visual inspection of the regimes indicates instability. Great caution should be taken when applying WR's in impact models for the energy sector, due to this large instability and uncertainties associated with WR definitions.
>
---
#### [new 007] Conflict in Community-Based Design: A Case Study of a Relationship Breakdown
- **分类: cs.HC; cs.CY**

- **简介: 该论文研究社区设计中的价值冲突处理，解决如何应对压迫性实践的问题，通过案例分析提出多路径解决策略。**

- **链接: [http://arxiv.org/pdf/2508.21308v1](http://arxiv.org/pdf/2508.21308v1)**

> **作者:** Alekhya Gandu; Aakash Gautam
>
> **备注:** 23 pages
>
> **摘要:** Community-based design efforts rightly seek to reduce the power differences between researchers and community participants by aligning with community values and furthering their priorities. However, what should designers do when key community members' practices seem to enact an oppressive and harmful structure? We reflect on our two-year-long engagement with a non-profit organization in southern India that supports women subjected to domestic abuse or facing mental health crises. We highlight the organizational gaps in knowledge management and transfer, which became an avenue for our design intervention. During design, we encountered practices that upheld caste hierarchies. These practices were expected to be incorporated into our technology. Anticipating harms to indirect stakeholders, we resisted this incorporation. It led to a breakdown in our relationship with the partner organization. Reflecting on this experience, we outline pluralistic pathways that community-based designers might inhabit when navigating value conflicts. These include making space for reflection before and during engagements, strategically repositioning through role reframing or appreciative inquiry, and exiting the engagement if necessary.
>
---
#### [new 008] Risks and Compliance with the EU's Core Cyber Security Legislation
- **分类: cs.CR; cs.CY; cs.SE**

- **简介: 该论文通过分析欧盟五部核心网络安全立法，研究其风险框架的趋同与分歧，识别风险术语及合规缺口，旨在解决立法间的风险定义差异与合规复杂性问题，提出实践应对策略。**

- **链接: [http://arxiv.org/pdf/2508.21386v1](http://arxiv.org/pdf/2508.21386v1)**

> **作者:** Jukka Ruohonen; Jesper Løffler Nielsen; Jakub Skórczynski
>
> **备注:** Submitted to IST (VSI:RegCompliance in SE)
>
> **摘要:** The European Union (EU) has long favored a risk-based approach to regulation. Such an approach is also used in recent cyber security legislation enacted in the EU. Risks are also inherently related to compliance with the new legislation. Objective: The paper investigates how risks are framed in the EU's five core cyber security legislative acts, whether the framings indicate convergence or divergence between the acts and their risk concepts, and what qualifying words and terms are used when describing the legal notions of risks. Method : The paper's methodology is based on qualitative legal interpretation and taxonomy-building. Results: The five acts have an encompassing coverage of different cyber security risks, including but not limited to risks related to technical, organizational, and human security as well as those not originating from man-made actions. Both technical aspects and assets are used to frame the legal risk notions in many of the legislative acts. A threat-centric viewpoint is also present in one of the acts. Notable gaps are related to acceptable risks, non-probabilistic risks, and residual risks. Conclusion: The EU's new cyber security legislation has significantly extended the risk-based approach to regulations. At the same time, complexity and compliance burden have increased. With this point in mind, the paper concludes with a few practical takeaways about means to deal with compliance and research it.
>
---
#### [new 009] Accept or Deny? Evaluating LLM Fairness and Performance in Loan Approval across Table-to-Text Serialization Approaches
- **分类: cs.LG; cs.CL; cs.CY**

- **简介: 该论文评估不同表格到文本序列化方法对LLM在贷款审批任务中的性能与公平性影响，比较零样本与上下文学习效果，发现特定格式提升性能但加剧公平差距，强调数据表示对模型可靠性的重要性。**

- **链接: [http://arxiv.org/pdf/2508.21512v1](http://arxiv.org/pdf/2508.21512v1)**

> **作者:** Israel Abebe Azime; Deborah D. Kanubala; Tejumade Afonja; Mario Fritz; Isabel Valera; Dietrich Klakow; Philipp Slusallek
>
> **摘要:** Large Language Models (LLMs) are increasingly employed in high-stakes decision-making tasks, such as loan approvals. While their applications expand across domains, LLMs struggle to process tabular data, ensuring fairness and delivering reliable predictions. In this work, we assess the performance and fairness of LLMs on serialized loan approval datasets from three geographically distinct regions: Ghana, Germany, and the United States. Our evaluation focuses on the model's zero-shot and in-context learning (ICL) capabilities. Our results reveal that the choice of serialization (Serialization refers to the process of converting tabular data into text formats suitable for processing by LLMs.) format significantly affects both performance and fairness in LLMs, with certain formats such as GReat and LIFT yielding higher F1 scores but exacerbating fairness disparities. Notably, while ICL improved model performance by 4.9-59.6% relative to zero-shot baselines, its effect on fairness varied considerably across datasets. Our work underscores the importance of effective tabular data representation methods and fairness-aware models to improve the reliability of LLMs in financial decision-making.
>
---
#### [new 010] Stairway to Fairness: Connecting Group and Individual Fairness
- **分类: cs.IR; cs.AI; cs.CL; cs.CY**

- **简介: 该论文研究推荐系统中群体公平与个体公平的关系，解决两者评估标准不统一导致的对比困难问题。通过对比分析评估指标，发现高群体公平可能损害个体公平，为实践提供平衡两者的参考。**

- **链接: [http://arxiv.org/pdf/2508.21334v1](http://arxiv.org/pdf/2508.21334v1)**

> **作者:** Theresia Veronika Rampisela; Maria Maistro; Tuukka Ruotsalo; Falk Scholer; Christina Lioma
>
> **备注:** Accepted to RecSys 2025 (short paper)
>
> **摘要:** Fairness in recommender systems (RSs) is commonly categorised into group fairness and individual fairness. However, there is no established scientific understanding of the relationship between the two fairness types, as prior work on both types has used different evaluation measures or evaluation objectives for each fairness type, thereby not allowing for a proper comparison of the two. As a result, it is currently not known how increasing one type of fairness may affect the other. To fill this gap, we study the relationship of group and individual fairness through a comprehensive comparison of evaluation measures that can be used for both fairness types. Our experiments with 8 runs across 3 datasets show that recommendations that are highly fair for groups can be very unfair for individuals. Our finding is novel and useful for RS practitioners aiming to improve the fairness of their systems. Our code is available at: https://github.com/theresiavr/stairway-to-fairness.
>
---
#### [new 011] Mapping Toxic Comments Across Demographics: A Dataset from German Public Broadcasting
- **分类: cs.CL; cs.CY**

- **简介: 该论文构建首个包含年龄标注的德国毒性评论数据集，解决现有数据缺乏人口统计信息的问题，通过人工与LLM标注分析年龄差异，支持年龄意识内容审核系统开发。**

- **链接: [http://arxiv.org/pdf/2508.21084v1](http://arxiv.org/pdf/2508.21084v1)**

> **作者:** Jan Fillies; Michael Peter Hoffmann; Rebecca Reichel; Roman Salzwedel; Sven Bodemer; Adrian Paschke
>
> **备注:** The paper has been accepted to the EMNLP 2025 main track
>
> **摘要:** A lack of demographic context in existing toxic speech datasets limits our understanding of how different age groups communicate online. In collaboration with funk, a German public service content network, this research introduces the first large-scale German dataset annotated for toxicity and enriched with platform-provided age estimates. The dataset includes 3,024 human-annotated and 30,024 LLM-annotated anonymized comments from Instagram, TikTok, and YouTube. To ensure relevance, comments were consolidated using predefined toxic keywords, resulting in 16.7\% labeled as problematic. The annotation pipeline combined human expertise with state-of-the-art language models, identifying key categories such as insults, disinformation, and criticism of broadcasting fees. The dataset reveals age-based differences in toxic speech patterns, with younger users favoring expressive language and older users more often engaging in disinformation and devaluation. This resource provides new opportunities for studying linguistic variation across demographics and supports the development of more equitable and age-aware content moderation systems.
>
---
## 更新

#### [replaced 001] Mobile Apps Prioritizing Privacy, Efficiency and Equity: A Decentralized Approach to COVID-19 Vaccination Coordination
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2102.09372v2](http://arxiv.org/pdf/2102.09372v2)**

> **作者:** Joseph Bae; Rohan Sukumaran; Sheshank Shankar; Anshuman Sharma; Ishaan Singh; Haris Nazir; Colin Kang; Saurish Srivastava; Parth Patwa; Priyanshi Katiyar; Vitor Pamplona
>
> **摘要:** In this early draft, we describe a decentralized, app-based approach to COVID-19 vaccine distribution that facilitates zero knowledge verification, dynamic vaccine scheduling, continuous symptoms reporting, access to aggregate analytics based on population trends and more. To ensure equity, our solution is developed to work with limited internet access as well. In addition, we describe the six critical functions that we believe last mile vaccination management platforms must perform, examine existing vaccine management systems, and present a model for privacy-focused, individual-centric solutions.
>
---
#### [replaced 002] Revealing Fine-Grained Values and Opinions in Large Language Models
- **分类: cs.CL; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.19238v3](http://arxiv.org/pdf/2406.19238v3)**

> **作者:** Dustin Wright; Arnav Arora; Nadav Borenstein; Srishti Yadav; Serge Belongie; Isabelle Augenstein
>
> **备注:** Findings of EMNLP 2024; 28 pages, 20 figures, 7 tables
>
> **摘要:** Uncovering latent values and opinions embedded in large language models (LLMs) can help identify biases and mitigate potential harm. Recently, this has been approached by prompting LLMs with survey questions and quantifying the stances in the outputs towards morally and politically charged statements. However, the stances generated by LLMs can vary greatly depending on how they are prompted, and there are many ways to argue for or against a given position. In this work, we propose to address this by analysing a large and robust dataset of 156k LLM responses to the 62 propositions of the Political Compass Test (PCT) generated by 6 LLMs using 420 prompt variations. We perform coarse-grained analysis of their generated stances and fine-grained analysis of the plain text justifications for those stances. For fine-grained analysis, we propose to identify tropes in the responses: semantically similar phrases that are recurrent and consistent across different prompts, revealing natural patterns in the text that a given LLM is prone to produce. We find that demographic features added to prompts significantly affect outcomes on the PCT, reflecting bias, as well as disparities between the results of tests when eliciting closed-form vs. open domain responses. Additionally, patterns in the plain text rationales via tropes show that similar justifications are repeatedly generated across models and prompts even with disparate stances.
>
---
#### [replaced 003] Documenting Deployment with Fabric: A Repository of Real-World AI Governance
- **分类: cs.CY; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2508.14119v4](http://arxiv.org/pdf/2508.14119v4)**

> **作者:** Mackenzie Jorgensen; Kendall Brogle; Katherine M. Collins; Lujain Ibrahim; Arina Shah; Petra Ivanovic; Noah Broestl; Gabriel Piles; Paul Dongha; Hatim Abdulhussein; Adrian Weller; Jillian Powers; Umang Bhatt
>
> **备注:** AIES 2025
>
> **摘要:** Artificial intelligence (AI) is increasingly integrated into society, from financial services and traffic management to creative writing. Academic literature on the deployment of AI has mostly focused on the risks and harms that result from the use of AI. We introduce Fabric, a publicly available repository of deployed AI use cases to outline their governance mechanisms. Through semi-structured interviews with practitioners, we collect an initial set of 20 AI use cases. In addition, we co-design diagrams of the AI workflow with the practitioners. We discuss the oversight mechanisms and guardrails used in practice to safeguard AI use. The Fabric repository includes visual diagrams of AI use cases and descriptions of the deployed systems. Using the repository, we surface gaps in governance and find common patterns in human oversight of deployed AI systems. We intend for Fabric to serve as an extendable, evolving tool for researchers to study the effectiveness of AI governance.
>
---
#### [replaced 004] Effects of Antivaccine Tweets on COVID-19 Vaccinations, Cases, and Deaths
- **分类: cs.SI; cs.CY**

- **链接: [http://arxiv.org/pdf/2406.09142v2](http://arxiv.org/pdf/2406.09142v2)**

> **作者:** John Bollenbacher; Filippo Menczer; John Bryden
>
> **摘要:** Despite the wide availability of COVID-19 vaccines in the United States and their effectiveness in reducing hospitalizations and mortality during the pandemic, a majority of Americans chose not to be vaccinated during 2021. Recent work shows that vaccine misinformation affects intentions in controlled settings, but does not link it to real-world vaccination rates. Here, we present observational evidence of a causal relationship between exposure to antivaccine content and vaccination rates, and estimate the size of this effect. We present a compartmental epidemic model that includes vaccination, vaccine hesitancy, and exposure to antivaccine content. We fit the model to data to determine that a geographical pattern of exposure to online antivaccine content across US counties explains reduced vaccine uptake in the same counties. We find observational evidence that exposure to antivaccine content on Twitter caused about 14,000 people to refuse vaccination between February and August 2021 in the US, resulting in at least 510 additional cases and 8 additional deaths. This work provides a methodology for linking online speech with offline epidemic outcomes. Our findings should inform social media moderation policy as well as public health interventions.
>
---
#### [replaced 005] Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective
- **分类: cs.CL; cs.AI; cs.CY; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.19028v4](http://arxiv.org/pdf/2506.19028v4)**

> **作者:** Weijie Xu; Yiwen Wang; Chi Xue; Xiangkun Hu; Xi Fang; Guimin Dong; Chandan K. Reddy
>
> **备注:** 29 pages, 9 figures, 15 tables
>
> **摘要:** Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo (Fine-grained Semantic Comparison), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSCo more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics.
>
---
#### [replaced 006] Proactive HIV Care: AI-Based Comorbidity Prediction from Routine EHR Data
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2508.20133v2](http://arxiv.org/pdf/2508.20133v2)**

> **作者:** Solomon Russom; Dimitrios Kollias; Qianni Zhang
>
> **备注:** accepted at ICCV 2025
>
> **摘要:** People living with HIV face a high burden of comorbidities, yet early detection is often limited by symptom-driven screening. We evaluate the potential of AI to predict multiple comorbidities from routinely collected Electronic Health Records. Using data from 2,200 HIV-positive patients in South East London, comprising 30 laboratory markers and 7 demographic/social attributes, we compare demographic-aware models (which use both laboratory/social variables and demographic information as input) against demographic-unaware models (which exclude all demographic information). Across all methods, demographic-aware models consistently outperformed unaware counterparts. Demographic recoverability experiments revealed that gender and age can be accurately inferred from laboratory data, underscoring both the predictive value and fairness considerations of demographic features. These findings show that combining demographic and laboratory data can improve automated, multi-label comorbidity prediction in HIV care, while raising important questions about bias and interpretability in clinical AI.
>
---
#### [replaced 007] Software is infrastructure: failures, successes, costs, and the case for formal verification
- **分类: cs.SE; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.13821v3](http://arxiv.org/pdf/2506.13821v3)**

> **作者:** Giovanni Bernardi; Adrian Francalanza; Marco Peressotti; Mohammad Reza Mousavi
>
> **摘要:** In this chapter we outline the role that software has in modern society, along with the staggering costs of poor software quality. To lay this bare, we recall the costs of some of the major software failures that happened during the last 40 years. We argue that these costs justify researching, studying and applying formal software verification and in particular program analysis. This position is supported by successful industrial experiences.
>
---
#### [replaced 008] Public support for misinformation interventions depends on perceived fairness, effectiveness, and intrusiveness
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2508.05849v2](http://arxiv.org/pdf/2508.05849v2)**

> **作者:** Catherine King; Samantha C. Phillips; Kathleen M. Carley
>
> **备注:** 23 pages, 3 figures
>
> **摘要:** The proliferation of misinformation on social media has concerning possible consequences, such as the degradation of democratic norms. While recent research on countering misinformation has largely focused on analyzing the effectiveness of interventions, the factors associated with public support for these interventions have received little attention. We asked 1,010 American social media users to rate their support for and perceptions of ten misinformation interventions implemented by the government or social media companies. Our results indicate that the perceived fairness of the intervention is the most important factor in determining support, followed by the perceived effectiveness of that intervention and then the intrusiveness. Interventions that supported user agency and transparency, such as labeling content or fact-checking ads, were more popular than those that involved moderating or removing content or accounts. We found some demographic differences in support levels, with Democrats and women supporting interventions more and finding them more fair, more effective, and less intrusive than Republicans and men, respectively. It is critical to understand which interventions are supported and why, as public opinion can play a key role in the rollout and effectiveness of policies.
>
---
