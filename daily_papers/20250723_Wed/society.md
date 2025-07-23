# 计算机与社会 cs.CY

- **最新发布 17 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] Verifying International Agreements on AI: Six Layers of Verification for Rules on Large-Scale AI Development and Deployment
- **分类: cs.CY**

- **简介: 该论文属于政策与技术研究任务，旨在解决国际人工智能协议验证问题。为确保各国遵守大规模AI开发和部署规则，论文提出了六个验证层面，包括芯片安全、监控设备和人员机制，同时强调需应对技术与滥用挑战，推动相关研发进展。**

- **链接: [http://arxiv.org/pdf/2507.15916v1](http://arxiv.org/pdf/2507.15916v1)**

> **作者:** Mauricio Baker; Gabriel Kulp; Oliver Marks; Miles Brundage; Lennart Heim
>
> **备注:** 80 pages, summary included
>
> **摘要:** The risks of frontier AI may require international cooperation, which in turn may require verification: checking that all parties follow agreed-on rules. For instance, states might need to verify that powerful AI models are widely deployed only after their risks to international security have been evaluated and deemed manageable. However, research on AI verification could benefit from greater clarity and detail. To address this, this report provides an in-depth overview of AI verification, intended for both policy professionals and technical researchers. We present novel conceptual frameworks, detailed implementation options, and key R&D challenges. These draw on existing literature, expert interviews, and original analysis, all within the scope of confidentially overseeing AI development and deployment that uses thousands of high-end AI chips. We find that states could eventually verify compliance by using six largely independent verification approaches with substantial redundancy: (1) built-in security features in AI chips; (2-3) separate monitoring devices attached to AI chips; and (4-6) personnel-based mechanisms, such as whistleblower programs. While promising, these approaches require guardrails to protect against abuse and power concentration, and many of these technologies have yet to be built or stress-tested. To enable states to confidently verify compliance with rules on large-scale AI development and deployment, the R&D challenges we list need significant progress.
>
---
#### [new 002] Combining Cost-Constrained Runtime Monitors for AI Safety
- **分类: cs.CY; cs.AI**

- **简介: 该论文属于AI安全任务，旨在解决如何高效结合多个运行时监控器以检测有害行为。论文提出一种算法，在预算约束下优化监控器调用顺序与干预分配，最大化干预概率（召回率）。通过似然比分析与成本权衡，相比基线方法召回率提升超两倍，且多监控器组合优于单一监控器。**

- **链接: [http://arxiv.org/pdf/2507.15886v1](http://arxiv.org/pdf/2507.15886v1)**

> **作者:** Tim Tian Hua; James Baskerville; Henri Lemoine; Mia Hopman; Aryan Bhatt; Tyler Tracy
>
> **摘要:** Monitoring AIs at runtime can help us detect and stop harmful actions. In this paper, we study how to combine multiple runtime monitors into a single monitoring protocol. The protocol's objective is to maximize the probability of applying a safety intervention on misaligned outputs (i.e., maximize recall). Since running monitors and applying safety interventions are costly, the protocol also needs to adhere to an average-case budget constraint. Taking the monitors' performance and cost as given, we develop an algorithm to find the most efficient protocol. The algorithm exhaustively searches over when and which monitors to call, and allocates safety interventions based on the Neyman-Pearson lemma. By focusing on likelihood ratios and strategically trading off spending on monitors against spending on interventions, we more than double our recall rate compared to a naive baseline in a code review setting. We also show that combining two monitors can Pareto dominate using either monitor alone. Our framework provides a principled methodology for combining existing monitors to detect undesirable behavior in cost-sensitive settings.
>
---
#### [new 003] Disability Across Cultures: A Human-Centered Audit of Ableism in Western and Indic LLMs
- **分类: cs.CY; cs.AI; cs.HC**

- **简介: 该论文属于自然语言处理与社会伦理交叉任务，旨在解决大型语言模型在识别非西方文化中残障歧视（ableism）时的偏差问题。研究者将现有英语残障歧视数据集翻译为印地语，测试西方与印度开发的八个大模型对印地语和英语中残障歧视的识别能力，并对比模型与残障人士的判断差异，揭示模型在文化适应性上的不足。**

- **链接: [http://arxiv.org/pdf/2507.16130v1](http://arxiv.org/pdf/2507.16130v1)**

> **作者:** Mahika Phutane; Aditya Vashistha
>
> **摘要:** People with disabilities (PwD) experience disproportionately high levels of discrimination and hate online, particularly in India, where entrenched stigma and limited resources intensify these challenges. Large language models (LLMs) are increasingly used to identify and mitigate online hate, yet most research on online ableism focuses on Western audiences with Western AI models. Are these models adequately equipped to recognize ableist harm in non-Western places like India? Do localized, Indic language models perform better? To investigate, we adopted and translated a publicly available ableist speech dataset to Hindi, and prompted eight LLMs--four developed in the U.S. (GPT-4, Gemini, Claude, Llama) and four in India (Krutrim, Nanda, Gajendra, Airavata)--to score and explain ableism. In parallel, we recruited 175 PwD from both the U.S. and India to perform the same task, revealing stark differences between groups. Western LLMs consistently overestimated ableist harm, while Indic LLMs underestimated it. Even more concerning, all LLMs were more tolerant of ableism when it was expressed in Hindi and asserted Western framings of ableist harm. In contrast, Indian PwD interpreted harm through intention, relationality, and resilience--emphasizing a desire to inform and educate perpetrators. This work provides groundwork for global, inclusive standards of ableism, demonstrating the need to center local disability experiences in the design and evaluation of AI systems.
>
---
#### [new 004] PRAC3 (Privacy, Reputation, Accountability, Consent, Credit, Compensation): Long Tailed Risks of Voice Actors in AI Data-Economy
- **分类: cs.CY; cs.AI; cs.HC**

- **简介: 该论文研究AI语音数据经济中声音演员面临的长尾风险，提出PRAC3框架（隐私、声誉、问责、同意、署名、补偿），扩展了原有的C3伦理模型。通过访谈20名专业声音演员，揭示其声音被滥用带来的声誉损害与法律责任问题，并探讨如何建立更完善的数据治理模型以保护声音创作者权益。**

- **链接: [http://arxiv.org/pdf/2507.16247v1](http://arxiv.org/pdf/2507.16247v1)**

> **作者:** Tanusree Sharma; Yihao Zhou; Visar Berisha
>
> **摘要:** Early large-scale audio datasets, such as LibriSpeech, were built with hundreds of individual contributors whose voices were instrumental in the development of speech technologies, including audiobooks and voice assistants. Yet, a decade later, these same contributions have exposed voice actors to a range of risks. While existing ethical frameworks emphasize Consent, Credit, and Compensation (C3), they do not adequately address the emergent risks involving vocal identities that are increasingly decoupled from context, authorship, and control. Drawing on qualitative interviews with 20 professional voice actors, this paper reveals how the synthetic replication of voice without enforceable constraints exposes individuals to a range of threats. Beyond reputational harm, such as re-purposing voice data in erotic content, offensive political messaging, and meme culture, we document concerns about accountability breakdowns when their voice is leveraged to clone voices that are deployed in high-stakes scenarios such as financial fraud, misinformation campaigns, or impersonation scams. In such cases, actors face social and legal fallout without recourse, while very few of them have a legal representative or union protection. To make sense of these shifting dynamics, we introduce the PRAC3 framework, an expansion of C3 that foregrounds Privacy, Reputation, Accountability, Consent, Credit, and Compensation as interdependent pillars of data used in the synthetic voice economy. This framework captures how privacy risks are amplified through non-consensual training, how reputational harm arises from decontextualized deployment, and how accountability can be reimagined AI Data ecosystems. We argue that voice, as both a biometric identifier and creative labor, demands governance models that restore creator agency, ensure traceability, and establish enforceable boundaries for ethical reuse.
>
---
#### [new 005] Characterizing Online Activities Contributing to Suicide Mortality among Youth
- **分类: cs.CY; cs.CL**

- **简介: 该论文旨在分析青少年自杀死亡相关的在线活动，属于公共健康与计算交叉任务。通过混合方法，从近3万份文本中提取12类相关在线活动主题，并构建零样本学习框架进行大规模建模，探究其与人口特征及时间变化的关系，以支持早期干预。**

- **链接: [http://arxiv.org/pdf/2507.16185v1](http://arxiv.org/pdf/2507.16185v1)**

> **作者:** Aparna Ananthasubramaniam; Elyse J. Thulin; Viktoryia Kalesnikava; Silas Falde; Jonathan Kertawidjaja; Lily Johns; Alejandro Rodríguez-Putnam; Emma Spring; Kara Zivin; Briana Mezuk
>
> **备注:** Accepted at the AAAI International Conference on Web and Social Media (ICWSM) 2026
>
> **摘要:** The recent rise in youth suicide highlights the urgent need to understand how online experiences contribute to this public health issue. Our mixed-methods approach responds to this challenge by developing a set of themes focused on risk factors for suicide mortality in online spaces among youth ages 10-24, and a framework to model these themes at scale. Using 29,124 open text summaries of death investigations between 2013-2022, we conducted a thematic analysis to identify 12 types of online activities that were considered by investigators or next of kin to be relevant in contextualizing a given suicide death. We then develop a zero-shot learning framework to model these 12 themes at scale, and analyze variation in these themes by decedent characteristics and over time. Our work uncovers several online activities related to harm to self, harm to others, interpersonal interactions, activity levels online, and life events, which correspond to different phases of suicide risk from two prominent suicide theories. We find an association between these themes and decedent characteristics like age, means of death, and interpersonal problems, and many themes became more prevalent during the 2020 COVID-19 lockdowns. While digital spaces have taken some steps to address expressions of suicidality online, our work illustrates the opportunities for developing interventions related to less explicit indicators of suicide risk by combining suicide theories with computational research.
>
---
#### [new 006] The Impact of Pseudo-Science in Financial Loans Risk Prediction
- **分类: cs.CY; cs.LG**

- **简介: 论文研究金融贷款风险预测中伪科学假设的影响，揭示生存偏差导致模型表现虚高与不公平性增加，分析模型准确性与社会成本，表明社会最优模型未必显著损失准确率。**

- **链接: [http://arxiv.org/pdf/2507.16182v1](http://arxiv.org/pdf/2507.16182v1)**

> **作者:** Bruno Scarone; Ricardo Baeza-Yates
>
> **摘要:** We study the societal impact of pseudo-scientific assumptions for predicting the behavior of people in a straightforward application of machine learning to risk prediction in financial lending. This use case also exemplifies the impact of survival bias in loan return prediction. We analyze the models in terms of their accuracy and social cost, showing that the socially optimal model may not imply a significant accuracy loss for this downstream task. Our results are verified for commonly used learning methods and datasets. Our findings also show that there is a natural dynamic when training models that suffer survival bias where accuracy slightly deteriorates, and whose recall and precision improves with time. These results act as an illusion, leading the observer to believe that the system is getting better, when in fact the model is suffering from increasingly more unfairness and survival bias.
>
---
#### [new 007] Chameleon Channels: Measuring YouTube Accounts Repurposed for Deception and Profit
- **分类: cs.CY; cs.CR**

- **简介: 论文研究了YouTube上被转卖或转型的“变色龙频道”，分析其传播有害内容的现象。任务是检测并评估这些频道的规模与影响。论文通过观察二手账号交易市场及大规模采样分析，发现大量转用途频道传播虚假信息和金融诈骗内容，且用户数不降反升，揭示了信任滥用的风险。**

- **链接: [http://arxiv.org/pdf/2507.16045v1](http://arxiv.org/pdf/2507.16045v1)**

> **作者:** Alejandro Cuevas; Manoel Horta Ribeiro; Nicolas Christin
>
> **备注:** 21 pages, 12 figures, 2 tables
>
> **摘要:** Online content creators spend significant time and effort building their user base through a long, often arduous process, which requires finding the right ``niche'' to cater to. So, what incentive is there for an established content creator known for cat memes to completely reinvent their page channel and start promoting cryptocurrency services or cover electoral news events? And, if they do, do their existing subscribers not notice? We explore this problem of \textit{repurposed channels}, whereby a channel changes its identity and contents. We first characterize a market for ``second-hand'' social media accounts, which recorded sales exceeding USD~1M during our 6-month observation period. By observing YouTube channels (re)sold over these 6~months, we find that a substantial number (37\%) are used to disseminate potentially harmful content, often without facing any penalty. Even more surprisingly, these channels seem to gain rather than lose subscribers. To estimate the prevalence of channel repurposing ``in the wild,'' we also collect two snapshots of 1.4M quasi-randomly sampled YouTube accounts. In a 3-month period, we estimate that $\sim$0.25\% channels -- collectively holding $\sim$44M subscribers -- were repurposed. We confirm that these repurposed channels share several characteristics with sold channels -- mainly, the fact that they had a significantly high presence of potentially problematic content. Across repurposed channels, we find channels that became disinformation channels, as well as channels that link to web pages with financial scams. We reason that abusing the residual trust placed on these channels is advantageous to financially- and ideologically-motivated adversaries. This phenomenon is not exclusive to YouTube and we posit that the market for cultivating organic audiences is set to grow, particularly if it remains unchallenged by mitigations, technical or otherwise.
>
---
#### [new 008] Beyond Algorethics: Addressing the Ethical and Anthropological Challenges of AI Recommender Systems
- **分类: cs.CY; cs.AI**

- **简介: 论文探讨AI推荐系统的伦理与人类学挑战，指出其过度简化人性、损害自主性的问题，认为现有技术伦理方法不足，提出需跨学科、法规与教育结合的综合框架，实现以人为本的AI设计。**

- **链接: [http://arxiv.org/pdf/2507.16430v1](http://arxiv.org/pdf/2507.16430v1)**

> **作者:** Octavian M. Machidon
>
> **摘要:** In this paper, I examine the ethical and anthropological challenges posed by AI-driven recommender systems (RSs), which have become central to shaping digital environments and social interactions. By curating personalized content, RSs do not merely reflect user preferences but actively construct individual experiences across social media, entertainment platforms, and e-commerce. Despite their ubiquity, the ethical implications of RSs remain insufficiently explored, even as concerns over privacy, autonomy, and mental well-being intensify. I argue that existing ethical approaches, including algorethics, the effort to embed ethical principles into algorithmic design, are necessary but ultimately inadequate. RSs inherently reduce human complexity to quantifiable dimensions, exploit user vulnerabilities, and prioritize engagement over well-being. Addressing these concerns requires moving beyond purely technical solutions. I propose a comprehensive framework for human-centered RS design, integrating interdisciplinary perspectives, regulatory strategies, and educational initiatives to ensure AI systems foster rather than undermine human autonomy and societal flourishing.
>
---
#### [new 009] Quantifying Holistic Review: A Multi-Modal Approach to College Admissions Prediction
- **分类: cs.LG; cs.CY**

- **简介: 该论文属于大学录取评估任务，旨在解决传统综合评估不透明、不一致等问题。作者提出CAPS多模态框架，将申请者资料分解为学术成绩、论文质量和课外活动三个可解释部分，结合Transformer、LLM和XGBoost实现透明、可解释的评估，提升录取预测准确性与公平性。**

- **链接: [http://arxiv.org/pdf/2507.15862v1](http://arxiv.org/pdf/2507.15862v1)**

> **作者:** Jun-Wei Zeng; Jerry Shen
>
> **摘要:** This paper introduces the Comprehensive Applicant Profile Score (CAPS), a novel multi-modal framework designed to quantitatively model and interpret holistic college admissions evaluations. CAPS decomposes applicant profiles into three interpretable components: academic performance (Standardized Academic Score, SAS), essay quality (Essay Quality Index, EQI), and extracurricular engagement (Extracurricular Impact Score, EIS). Leveraging transformer-based semantic embeddings, LLM scoring, and XGBoost regression, CAPS provides transparent and explainable evaluations aligned with human judgment. Experiments on a synthetic but realistic dataset demonstrate strong performance, achieving an EQI prediction R^2 of 0.80, classification accuracy over 75%, a macro F1 score of 0.69, and a weighted F1 score of 0.74. CAPS addresses key limitations in traditional holistic review -- particularly the opacity, inconsistency, and anxiety faced by applicants -- thus paving the way for more equitable and data-informed admissions practices.
>
---
#### [new 010] AI-enhanced conversational agents for personalized asthma support Factors for engagement, value and efficacy
- **分类: cs.HC; cs.AI; cs.CY; cs.ET; K.4.2; J.3**

- **简介: 该论文研究哮喘患者对AI聊天机器人的接受度与使用意愿，旨在解决哮喘管理中患者参与度低的问题。通过分析1257名患者的调查数据，识别影响聊天机器人使用的关键因素，并提出优化建议。**

- **链接: [http://arxiv.org/pdf/2507.16735v1](http://arxiv.org/pdf/2507.16735v1)**

> **作者:** Laura Moradbakhti; Dorian Peters; Jennifer K. Quint; Björn Schuller; Darren Cook; Rafael A. Calvo
>
> **备注:** 7 Tables, 4 Figures
>
> **摘要:** Asthma-related deaths in the UK are the highest in Europe, and only 30% of patients access basic care. There is a need for alternative approaches to reaching people with asthma in order to provide health education, self-management support and bridges to care. Automated conversational agents (specifically, mobile chatbots) present opportunities for providing alternative and individually tailored access to health education, self-management support and risk self-assessment. But would patients engage with a chatbot, and what factors influence engagement? We present results from a patient survey (N=1257) devised by a team of asthma clinicians, patients, and technology developers, conducted to identify optimal factors for efficacy, value and engagement for a chatbot. Results indicate that most adults with asthma (53%) are interested in using a chatbot and the patients most likely to do so are those who believe their asthma is more serious and who are less confident about self-management. Results also indicate enthusiasm for 24/7 access, personalisation, and for WhatsApp as the preferred access method (compared to app, voice assistant, SMS or website). Obstacles to uptake include security/privacy concerns and skepticism of technological capabilities. We present detailed findings and consolidate these into 7 recommendations for developers for optimising efficacy of chatbot-based health support.
>
---
#### [new 011] PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 论文提出PICACO方法，属于上下文对齐任务，旨在解决大语言模型在单一提示中难以平衡多元价值观导致的对齐不足问题。通过优化元指令，提升模型对多种价值的理解与响应一致性，实现更优的价值对齐。**

- **链接: [http://arxiv.org/pdf/2507.16679v1](http://arxiv.org/pdf/2507.16679v1)**

> **作者:** Han Jiang; Dongyao Zhu; Zhihua Wei; Xiaoyuan Yi; Ziang Xiao; Xing Xie
>
> **摘要:** In-Context Learning has shown great potential for aligning Large Language Models (LLMs) with human values, helping reduce harmful outputs and accommodate diverse preferences without costly post-training, known as In-Context Alignment (ICA). However, LLMs' comprehension of input prompts remains agnostic, limiting ICA's ability to address value tensions--human values are inherently pluralistic, often imposing conflicting demands, e.g., stimulation vs. tradition. Current ICA methods therefore face the Instruction Bottleneck challenge, where LLMs struggle to reconcile multiple intended values within a single prompt, leading to incomplete or biased alignment. To address this, we propose PICACO, a novel pluralistic ICA method. Without fine-tuning, PICACO optimizes a meta-instruction that navigates multiple values to better elicit LLMs' understanding of them and improve their alignment. This is achieved by maximizing the total correlation between specified values and LLM responses, theoretically reinforcing value correlation while reducing distractive noise, resulting in effective value instructions. Extensive experiments on five value sets show that PICACO works well with both black-box and open-source LLMs, outperforms several recent strong baselines, and achieves a better balance across up to 8 distinct values.
>
---
#### [new 012] WhatsApp Tiplines and Multilingual Claims in the 2021 Indian Assembly Elections
- **分类: cs.SI; cs.CL; cs.CY; cs.HC**

- **简介: 该论文研究了2021年印度选举期间WhatsApp举报热线中多语言虚假信息的处理情况。任务是分析用户提交的多语言举报内容，识别主题分类、语言差异、事实核查机构间的用户重叠及响应效率。论文通过内容分析与用户行为研究，提出优化举报热线在选举期间应用的建议。**

- **链接: [http://arxiv.org/pdf/2507.16298v1](http://arxiv.org/pdf/2507.16298v1)**

> **作者:** Gautam Kishore Shahi; Scot A. Hale
>
> **摘要:** WhatsApp tiplines, first launched in 2019 to combat misinformation, enable users to interact with fact-checkers to verify misleading content. This study analyzes 580 unique claims (tips) from 451 users, covering both high-resource languages (English, Hindi) and a low-resource language (Telugu) during the 2021 Indian assembly elections using a mixed-method approach. We categorize the claims into three categories, election, COVID-19, and others, and observe variations across languages. We compare content similarity through frequent word analysis and clustering of neural sentence embeddings. We also investigate user overlap across languages and fact-checking organizations. We measure the average time required to debunk claims and inform tipline users. Results reveal similarities in claims across languages, with some users submitting tips in multiple languages to the same fact-checkers. Fact-checkers generally require a couple of days to debunk a new claim and share the results with users. Notably, no user submits claims to multiple fact-checking organizations, indicating that each organization maintains a unique audience. We provide practical recommendations for using tiplines during elections with ethical consideration of users' information.
>
---
#### [new 013] A Human-Centered Approach to Identifying Promises, Risks, & Challenges of Text-to-Image Generative AI in Radiology
- **分类: cs.HC; cs.AI; cs.CY**

- **简介: 该论文属于人机交互与医学AI结合的任务，旨在解决医疗领域中文本生成CT图像技术的应用潜力与风险问题。作者通过让医学学生、住院医师和放射科医生参与模型测试，分析其在医学教育和临床实践中的价值、风险及挑战，提出以人为本的AI开发方法。**

- **链接: [http://arxiv.org/pdf/2507.16207v1](http://arxiv.org/pdf/2507.16207v1)**

> **作者:** Katelyn Morrison; Arpit Mathur; Aidan Bradshaw; Tom Wartmann; Steven Lundi; Afrooz Zandifar; Weichang Dai; Kayhan Batmanghelich; Motahhare Eslami; Adam Perer
>
> **备注:** 13 pages, 2 figures, accepted to AAAI/ACM AIES 2025
>
> **摘要:** As text-to-image generative models rapidly improve, AI researchers are making significant advances in developing domain-specific models capable of generating complex medical imagery from text prompts. Despite this, these technical advancements have overlooked whether and how medical professionals would benefit from and use text-to-image generative AI (GenAI) in practice. By developing domain-specific GenAI without involving stakeholders, we risk the potential of building models that are either not useful or even more harmful than helpful. In this paper, we adopt a human-centered approach to responsible model development by involving stakeholders in evaluating and reflecting on the promises, risks, and challenges of a novel text-to-CT Scan GenAI model. Through exploratory model prompting activities, we uncover the perspectives of medical students, radiology trainees, and radiologists on the role that text-to-CT Scan GenAI can play across medical education, training, and practice. This human-centered approach additionally enabled us to surface technical challenges and domain-specific risks of generating synthetic medical images. We conclude by reflecting on the implications of medical text-to-image GenAI.
>
---
#### [new 014] GG-BBQ: German Gender Bias Benchmark for Question Answering
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 该论文属于自然语言处理中的公平性评估任务，旨在解决德语大语言模型中的性别偏见问题。作者基于英文数据集翻译构建了德语性别偏见测试集GG-BBQ，并通过人工修正提升翻译质量。最终评估多个德语模型，发现均存在性别偏见。**

- **链接: [http://arxiv.org/pdf/2507.16410v1](http://arxiv.org/pdf/2507.16410v1)**

> **作者:** Shalaka Satheesh; Katrin Klug; Katharina Beckh; Héctor Allende-Cid; Sebastian Houben; Teena Hassan
>
> **备注:** Accepted to the 6th Workshop on Gender Bias in Natural Language Processing (GeBNLP), taking place on August 1st 2025, as part of ACL 2025 in Vienna
>
> **摘要:** Within the context of Natural Language Processing (NLP), fairness evaluation is often associated with the assessment of bias and reduction of associated harm. In this regard, the evaluation is usually carried out by using a benchmark dataset, for a task such as Question Answering, created for the measurement of bias in the model's predictions along various dimensions, including gender identity. In our work, we evaluate gender bias in German Large Language Models (LLMs) using the Bias Benchmark for Question Answering by Parrish et al. (2022) as a reference. Specifically, the templates in the gender identity subset of this English dataset were machine translated into German. The errors in the machine translated templates were then manually reviewed and corrected with the help of a language expert. We find that manual revision of the translation is crucial when creating datasets for gender bias evaluation because of the limitations of machine translation from English to a language such as German with grammatical gender. Our final dataset is comprised of two subsets: Subset-I, which consists of group terms related to gender identity, and Subset-II, where group terms are replaced with proper names. We evaluate several LLMs used for German NLP on this newly created dataset and report the accuracy and bias scores. The results show that all models exhibit bias, both along and against existing social stereotypes.
>
---
#### [new 015] Voice-based AI Agents: Filling the Economic Gaps in Digital Health Delivery
- **分类: cs.AI; cs.CY; cs.HC; cs.SE**

- **简介: 该论文探讨语音AI代理在数字医疗中的应用，旨在解决经济和可及性差距。通过开发Agent PULSE，研究AI代理在预防护理和患者监测中的作用，分析其成本效益、技术挑战及政策问题，提出AI可提升医疗公平性与效率。**

- **链接: [http://arxiv.org/pdf/2507.16229v1](http://arxiv.org/pdf/2507.16229v1)**

> **作者:** Bo Wen; Chen Wang; Qiwei Han; Raquel Norel; Julia Liu; Thaddeus Stappenbeck; Jeffrey L. Rogers
>
> **备注:** IEEE International Conference on Digital Health (ICDH) 2025
>
> **摘要:** The integration of voice-based AI agents in healthcare presents a transformative opportunity to bridge economic and accessibility gaps in digital health delivery. This paper explores the role of large language model (LLM)-powered voice assistants in enhancing preventive care and continuous patient monitoring, particularly in underserved populations. Drawing insights from the development and pilot study of Agent PULSE (Patient Understanding and Liaison Support Engine) -- a collaborative initiative between IBM Research, Cleveland Clinic Foundation, and Morehouse School of Medicine -- we present an economic model demonstrating how AI agents can provide cost-effective healthcare services where human intervention is economically unfeasible. Our pilot study with 33 inflammatory bowel disease patients revealed that 70\% expressed acceptance of AI-driven monitoring, with 37\% preferring it over traditional modalities. Technical challenges, including real-time conversational AI processing, integration with healthcare systems, and privacy compliance, are analyzed alongside policy considerations surrounding regulation, bias mitigation, and patient autonomy. Our findings suggest that AI-driven voice agents not only enhance healthcare scalability and efficiency but also improve patient engagement and accessibility. For healthcare executives, our cost-utility analysis demonstrates huge potential savings for routine monitoring tasks, while technologists can leverage our framework to prioritize improvements yielding the highest patient impact. By addressing current limitations and aligning AI development with ethical and regulatory frameworks, voice-based AI agents can serve as a critical entry point for equitable, sustainable digital healthcare solutions.
>
---
#### [new 016] Advancing Responsible Innovation in Agentic AI: A study of Ethical Frameworks for Household Automation
- **分类: cs.AI; cs.CY; cs.MA**

- **简介: 该论文属于伦理与技术融合任务，旨在解决家庭环境中主动性人工智能（Agentic AI）带来的隐私、公平性和用户控制等伦理问题。论文分析了相关伦理框架、人机设计原则和治理实践，提出了面向弱势群体的可解释性、知情同意和控制机制等设计建议，并探索了通过社交媒体分析获取用户需求的方法。**

- **链接: [http://arxiv.org/pdf/2507.15901v1](http://arxiv.org/pdf/2507.15901v1)**

> **作者:** Joydeep Chandra; Satyam Kumar Navneet
>
> **摘要:** The implementation of Artificial Intelligence (AI) in household environments, especially in the form of proactive autonomous agents, brings about possibilities of comfort and attention as well as it comes with intra or extramural ethical challenges. This article analyzes agentic AI and its applications, focusing on its move from reactive to proactive autonomy, privacy, fairness and user control. We review responsible innovation frameworks, human-centered design principles, and governance practices to distill practical guidance for ethical smart home systems. Vulnerable user groups such as elderly individuals, children, and neurodivergent who face higher risks of surveillance, bias, and privacy risks were studied in detail in context of Agentic AI. Design imperatives are highlighted such as tailored explainability, granular consent mechanisms, and robust override controls, supported by participatory and inclusive methodologies. It was also explored how data-driven insights, including social media analysis via Natural Language Processing(NLP), can inform specific user needs and ethical concerns. This survey aims to provide both a conceptual foundation and suggestions for developing transparent, inclusive, and trustworthy agentic AI in household automation.
>
---
#### [new 017] Integrating Reason-Based Moral Decision-Making in the Reinforcement Learning Architecture
- **分类: cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于人工智能与伦理交叉任务，旨在解决如何使强化学习智能体具备道德决策能力的问题。论文提出了一种基于理由的道德决策框架（RBAMA），扩展了强化学习架构，使智能体能通过案例反馈学习道德规范，并据此调整行为以符合道德义务，从而提升其道德可辩护性与可信度。**

- **链接: [http://arxiv.org/pdf/2507.15895v1](http://arxiv.org/pdf/2507.15895v1)**

> **作者:** Lisa Dargasz
>
> **备注:** Master's thesis, April 2025, 122 pages
>
> **摘要:** Reinforcement Learning is a machine learning methodology that has demonstrated strong performance across a variety of tasks. In particular, it plays a central role in the development of artificial autonomous agents. As these agents become increasingly capable, market readiness is rapidly approaching, which means those agents, for example taking the form of humanoid robots or autonomous cars, are poised to transition from laboratory prototypes to autonomous operation in real-world environments. This transition raises concerns leading to specific requirements for these systems - among them, the requirement that they are designed to behave ethically. Crucially, research directed toward building agents that fulfill the requirement to behave ethically - referred to as artificial moral agents(AMAs) - has to address a range of challenges at the intersection of computer science and philosophy. This study explores the development of reason-based artificial moral agents (RBAMAs). RBAMAs are build on an extension of the reinforcement learning architecture to enable moral decision-making based on sound normative reasoning, which is achieved by equipping the agent with the capacity to learn a reason-theory - a theory which enables it to process morally relevant propositions to derive moral obligations - through case-based feedback. They are designed such that they adapt their behavior to ensure conformance to these obligations while they pursue their designated tasks. These features contribute to the moral justifiability of the their actions, their moral robustness, and their moral trustworthiness, which proposes the extended architecture as a concrete and deployable framework for the development of AMAs that fulfills key ethical desiderata. This study presents a first implementation of an RBAMA and demonstrates the potential of RBAMAs in initial experiments.
>
---
## 更新

#### [replaced 001] Software is infrastructure: failures, successes, costs, and the case for formal verification
- **分类: cs.SE; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.13821v2](http://arxiv.org/pdf/2506.13821v2)**

> **作者:** Giovanni Bernardi; Adrian Francalanza; Marco Peressotti; Mohammad Reza Mousavi
>
> **摘要:** In this chapter we outline the role that software has in modern society, along with the staggering costs of poor software quality. To lay this bare, we recall the costs of some of the major software failures that happened during the last~$40$ years. We argue that these costs justify researching, studying and applying formal software verification and in particular program analysis. This position is supported by successful industrial experiences.
>
---
#### [replaced 002] Romance, Relief, and Regret: Teen Narratives of Chatbot Overreliance
- **分类: cs.HC; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2507.15783v2](http://arxiv.org/pdf/2507.15783v2)**

> **作者:** Mohammad 'Matt' Namvarpour; Brandon Brofsky; Jessica Medina; Mamtaj Akter; Afsaneh Razi
>
> **摘要:** As Generative Artificial Intelligence (GenAI) driven chatbots like Character.AI become embedded in adolescent life, they raise concerns about emotional dependence and digital overreliance. While studies have investigated the overreliance of adults on these chatbots, they have not investigated teens' interactions with chatbots with customizable personas. We analyzed 318 Reddit posts made by users self-reported as 13-17 years old on the Character.AI subreddit to understand patterns of overreliance. We found teens commonly begin using chatbots for emotional support or creative expression, but many develop strong attachments that interfere with offline relationships and daily routines. Their posts revealed recurring signs of psychological distress, cycles of relapse, and difficulty disengaging. Teens reported that their overreliance often ended when they reflect on the harm, return to in-person social settings, or become frustrated by platform restrictions. Based on the implications of our findings, we provide recommendations for future chatbot design so they can promote self-awareness, support real-world engagement, and involve teens in developing safer digital tools.
>
---
#### [replaced 003] Multimodal Coordinated Online Behavior: Trade-offs and Strategies
- **分类: cs.SI; cs.AI; cs.CY; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.12108v2](http://arxiv.org/pdf/2507.12108v2)**

> **作者:** Lorenzo Mannocci; Stefano Cresci; Matteo Magnani; Anna Monreale; Maurizio Tesconi
>
> **摘要:** Coordinated online behavior, which spans from beneficial collective actions to harmful manipulation such as disinformation campaigns, has become a key focus in digital ecosystem analysis. Traditional methods often rely on monomodal approaches, focusing on single types of interactions like co-retweets or co-hashtags, or consider multiple modalities independently of each other. However, these approaches may overlook the complex dynamics inherent in multimodal coordination. This study compares different ways of operationalizing the detection of multimodal coordinated behavior. It examines the trade-off between weakly and strongly integrated multimodal models, highlighting the balance between capturing broader coordination patterns and identifying tightly coordinated behavior. By comparing monomodal and multimodal approaches, we assess the unique contributions of different data modalities and explore how varying implementations of multimodality impact detection outcomes. Our findings reveal that not all the modalities provide distinct insights, but that with a multimodal approach we can get a more comprehensive understanding of coordination dynamics. This work enhances the ability to detect and analyze coordinated online behavior, offering new perspectives for safeguarding the integrity of digital platforms.
>
---
#### [replaced 004] Mapping the Parasocial AI Market: User Trends, Engagement and Risks
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2507.14226v2](http://arxiv.org/pdf/2507.14226v2)**

> **作者:** Zilan Qian; Mari Izumikawa; Fiona Lodge; Angelo Leone
>
> **备注:** 17 pages, 17 figures
>
> **摘要:** A scan of 110 AI companion platforms reveals a rapidly growing global market for emotionally engaging, personalized AI interactions. While parasocial use of general-purpose AI (GPAI) tools currently dominates, a growing number of platforms are designed specifically for care, transactional, or mating companionship. In the UK alone, these platforms receive between 46 million and 91 million monthly visits (1.1-2.2 billion globally), with users spending an average of 3.5 minutes per session. For context, Instagram averaged 67.3 million UK visits per month between January and March 2025. Notably, mating-oriented AI companions make up 44% of UK visits (higher than the global average of 30%) but see lower session times and return rates than mixed-use platforms. As mating-oriented romantic AI offerings improve, increased engagement may follow, raising urgent concerns about online safety, particularly for children, given weak age safeguards. Meanwhile, GPAI tools are moving toward more emotionally intelligent, personalized interactions, making parasocial AI use increasingly mainstream. These trends highlight the need for the UK AI Security Institute (AISI) to monitor this sector and assess whether existing regulation sufficiently addresses emerging societal risks.
>
---
#### [replaced 005] SciFi-Benchmark: Leveraging Science Fiction To Improve Robot Behavior
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.10706v2](http://arxiv.org/pdf/2503.10706v2)**

> **作者:** Pierre Sermanet; Anirudha Majumdar; Vikas Sindhwani
>
> **备注:** Minor improvements over previous version
>
> **摘要:** Given the recent rate of progress in artificial intelligence (AI) and robotics, a tantalizing question is emerging: would robots controlled by emerging AI systems be strongly aligned with human values? In this work, we propose a scalable way to probe this question by generating a benchmark spanning the key moments in 824 major pieces of science fiction literature (movies, tv, novels and scientific books) where an agent (AI or robot) made critical decisions (good or bad). We use a state-of-the-art LLM's recollection of each key moment to generate questions in similar situations, the decisions made by the agent, and alternative decisions it could have made (good or bad). We then measure an approximation of how well models align with human values on a set of human-voted answers. We also generate rules that can be automatically improved via an amendment process in order to generate the first Sci-Fi inspired constitutions for promoting ethical behavior in AIs and robots in the real world. Our first finding is that modern LLMs paired with constitutions turn out to be well-aligned with human values (95.8%), contrary to unsettling decisions typically made in Sci-Fi (only 21.2% alignment). Secondly, we find that generated constitutions substantially increase alignment compared to the base model (79.4% to 95.8%), and show resilience to an adversarial prompt setting (23.3% to 92.3%). Additionally, we find that those constitutions are among the top performers on the ASIMOV Benchmark which is derived from real-world images and hospital injury reports. Sci-Fi-inspired constitutions are thus highly aligned and applicable in real-world situations. We release SciFi-Benchmark: a large-scale dataset to advance robot ethics and safety research. It comprises 9,056 questions and 53,384 answers generated through a novel LLM-introspection process, in addition to a smaller human-labeled evaluation set.
>
---
#### [replaced 006] Secondary Bounded Rationality: A Theory of How Algorithms Reproduce Structural Inequality in AI Hiring
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2507.09233v2](http://arxiv.org/pdf/2507.09233v2)**

> **作者:** Jia Xiao
>
> **摘要:** AI-driven recruitment systems, while promising efficiency and objectivity, often perpetuate systemic inequalities by encoding cultural and social capital disparities into algorithmic decision making. This article develops and defends a novel theory of secondary bounded rationality, arguing that AI systems, despite their computational power, inherit and amplify human cognitive and structural biases through technical and sociopolitical constraints. Analyzing multimodal recruitment frameworks, we demonstrate how algorithmic processes transform historical inequalities, such as elite credential privileging and network homophily, into ostensibly meritocratic outcomes. Using Bourdieusian capital theory and Simon's bounded rationality, we reveal a recursive cycle where AI entrenches exclusion by optimizing for legible yet biased proxies of competence. We propose mitigation strategies, including counterfactual fairness testing, capital-aware auditing, and regulatory interventions, to disrupt this self-reinforcing inequality.
>
---
#### [replaced 007] Reasoning Models Can be Easily Hacked by Fake Reasoning Bias
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2507.13758v2](http://arxiv.org/pdf/2507.13758v2)**

> **作者:** Qian Wang; Yubo Fan; Zhenheng Tang; Nuo Chen; Wenxuan Wang; Bingsheng He
>
> **摘要:** Large Reasoning Models (LRMs) like DeepSeek-R1 and o1 are increasingly used as automated evaluators, raising critical questions about their vulnerability to the aesthetics of reasoning in LLM-as-a-judge settings. We introduce THEATER, a comprehensive benchmark to systematically evaluate this vulnerability-termed Reasoning Theater Bias (RTB)-by comparing LLMs and LRMs across subjective preference and objective factual datasets. Through investigation of six bias types including Simple Cues and Fake Chain-of-Thought, we uncover three key findings: (1) in a critical paradox, reasoning-specialized LRMs are consistently more susceptible to RTB than general-purpose LLMs, particularly in subjective tasks; (2) this creates a task-dependent trade-off, where LRMs show more robustness on factual tasks but less on subjective ones; and (3) we identify 'shallow reasoning'-plausible but flawed arguments-as the most potent form of RTB. To address this, we design and evaluate two prompting strategies: a targeted system prompt that improves accuracy by up to 12% on factual tasks but only 1-3% on subjective tasks, and a self-reflection mechanism that shows similarly limited effectiveness in the more vulnerable subjective domains. Our work reveals that RTB is a deep-seated challenge for LRM-based evaluation and provides a systematic framework for developing more genuinely robust and trustworthy LRMs.
>
---
#### [replaced 008] Risks of AI Scientists: Prioritizing Safeguarding Over Autonomy
- **分类: cs.CY; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.04247v5](http://arxiv.org/pdf/2402.04247v5)**

> **作者:** Xiangru Tang; Qiao Jin; Kunlun Zhu; Tongxin Yuan; Yichi Zhang; Wangchunshu Zhou; Meng Qu; Yilun Zhao; Jian Tang; Zhuosheng Zhang; Arman Cohan; Zhiyong Lu; Mark Gerstein
>
> **摘要:** AI scientists powered by large language models have demonstrated substantial promise in autonomously conducting experiments and facilitating scientific discoveries across various disciplines. While their capabilities are promising, these agents also introduce novel vulnerabilities that require careful consideration for safety. However, there has been limited comprehensive exploration of these vulnerabilities. This perspective examines vulnerabilities in AI scientists, shedding light on potential risks associated with their misuse, and emphasizing the need for safety measures. We begin by providing an overview of the potential risks inherent to AI scientists, taking into account user intent, the specific scientific domain, and their potential impact on the external environment. Then, we explore the underlying causes of these vulnerabilities and provide a scoping review of the limited existing works. Based on our analysis, we propose a triadic framework involving human regulation, agent alignment, and an understanding of environmental feedback (agent regulation) to mitigate these identified risks. Furthermore, we highlight the limitations and challenges associated with safeguarding AI scientists and advocate for the development of improved models, robust benchmarks, and comprehensive regulations.
>
---
#### [replaced 009] Toward A Causal Framework for Modeling Perception
- **分类: cs.AI; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2401.13408v3](http://arxiv.org/pdf/2401.13408v3)**

> **作者:** Jose M. Alvarez; Salvatore Ruggieri
>
> **备注:** arXiv admin note: text overlap with arXiv:2305.09535 by other authors
>
> **摘要:** Perception occurs when individuals interpret the same information differently. It is a known cognitive phenomenon with implications for bias in human decision-making. Perception, however, remains understudied in machine learning (ML). This is problematic as modern decision flows, whether partially or fully automated by ML applications, always involve human experts. How might we account for cases in which two experts, e.g., interpret differently the same deferred instance or explanation from a ML model? Addressing this and similar questions requires a formulation of perception, particularly, in a manner that integrates with ML-enabled decision flows. In this work, we present a first approach to modeling perception causally. We define perception under causal reasoning using structural causal models (SCM). Our approach formalizes individual experience as additional causal knowledge that comes with and is used by the expert decision-maker in the form of a SCM. We define two kinds of probabilistic causal perception: structural perception and parametrical perception. We showcase our framework through a series of examples of modern decision flows. We also emphasize the importance of addressing perception in fair ML, discussing relevant fairness implications and possible applications.
>
---
#### [replaced 010] Aligning AI with Public Values: Deliberation and Decision-Making for Governing Multimodal LLMs in Political Video Analysis
- **分类: cs.CV; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2410.01817v2](http://arxiv.org/pdf/2410.01817v2)**

> **作者:** Tanusree Sharma; Yujin Potter; Zachary Kilhoffer; Yun Huang; Dawn Song; Yang Wang
>
> **摘要:** How AI models should deal with political topics has been discussed, but it remains challenging and requires better governance. This paper examines the governance of large language models through individual and collective deliberation, focusing on politically sensitive videos. We conducted a two-step study: interviews with 10 journalists established a baseline understanding of expert video interpretation; 114 individuals through deliberation using InclusiveAI, a platform that facilitates democratic decision-making through decentralized autonomous organization (DAO) mechanisms. Our findings reveal distinct differences in interpretative priorities: while experts emphasized emotion and narrative, the general public prioritized factual clarity, objectivity, and emotional neutrality. Furthermore, we examined how different governance mechanisms - quadratic vs. weighted voting and equal vs. 20/80 voting power - shape users' decision-making regarding AI behavior. Results indicate that voting methods significantly influence outcomes, with quadratic voting reinforcing perceptions of liberal democracy and political equality. Our study underscores the necessity of selecting appropriate governance mechanisms to better capture user perspectives and suggests decentralized AI governance as a potential way to facilitate broader public engagement in AI development, ensuring that varied perspectives meaningfully inform design decisions.
>
---
