# 计算机与社会 cs.CY

- **最新发布 16 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] Preventing Another Tessa: Modular Safety Middleware For Health-Adjacent AI Assistants
- **分类: cs.CY; cs.AI**

- **简介: 论文提出一种轻量级安全中间件，用于防止健康相关AI助手输出有害内容。通过结合词法门限与LLM策略过滤，实现高效拦截有害提示，提升AI在医疗场景中的安全性。**

- **链接: [http://arxiv.org/pdf/2509.07022v1](http://arxiv.org/pdf/2509.07022v1)**

> **作者:** Pavan Reddy; Nithin Reddy
>
> **备注:** 7 pages content, 1 page reference, 1 figure, Accepted at AAAI Fall Symposium Series
>
> **摘要:** In 2023, the National Eating Disorders Association's (NEDA) chatbot Tessa was suspended after providing harmful weight-loss advice to vulnerable users-an avoidable failure that underscores the risks of unsafe AI in healthcare contexts. This paper examines Tessa as a case study in absent safety engineering and demonstrates how a lightweight, modular safeguard could have prevented the incident. We propose a hybrid safety middleware that combines deterministic lexical gates with an in-line large language model (LLM) policy filter, enforcing fail-closed verdicts and escalation pathways within a single model call. Using synthetic evaluations, we show that this design achieves perfect interception of unsafe prompts at baseline cost and latency, outperforming traditional multi-stage pipelines. Beyond technical remedies, we map Tessa's failure patterns to established frameworks (OWASP LLM Top10, NIST SP 800-53), connecting practical safeguards to actionable governance controls. The results highlight that robust, auditable safety in health-adjacent AI does not require heavyweight infrastructure: explicit, testable checks at the last mile are sufficient to prevent "another Tessa", while governance and escalation ensure sustainability in real-world deployment.
>
---
#### [new 002] ArGen: Auto-Regulation of Generative AI via GRPO and Policy-as-Code
- **分类: cs.CY; cs.AI; cs.CL; cs.LG; 68T07, 68T50; I.2.6; I.2.7; K.4.1**

- **简介: 该论文提出ArGen框架，用于对齐大语言模型与复杂规则，解决伦理、安全和合规问题。通过GRPO、自动奖励评分和OPA治理层实现政策合规，以医疗AI助手为例展示其有效性。**

- **链接: [http://arxiv.org/pdf/2509.07006v1](http://arxiv.org/pdf/2509.07006v1)**

> **作者:** Kapil Madan
>
> **备注:** 53 pages, 7 figures, 8 tables. Open-source implementation available at: https://github.com/Principled-Evolution/argen-demo. Work explores the integration of policy-as-code for AI alignment, with a case study in culturally-nuanced, ethical AI using Dharmic principles
>
> **摘要:** This paper introduces ArGen (Auto-Regulation of Generative AI systems), a framework for aligning Large Language Models (LLMs) with complex sets of configurable, machine-readable rules spanning ethical principles, operational safety protocols, and regulatory compliance standards. Moving beyond just preference-based alignment, ArGen is designed to ensure LLMs adhere to these multifaceted policies through a novel synthesis of principle-based automated reward scoring, Group Relative Policy Optimisation (GRPO), and an Open Policy Agent (OPA) inspired governance layer. This approach provides the technical foundation for achieving and demonstrating compliance with diverse and nuanced governance requirements. To showcase the framework's capability to operationalize a deeply nuanced and culturally-specific value system, we present an in-depth case study: the development of a medical AI assistant guided by principles from Dharmic ethics (such as Ahimsa and Dharma), as derived from texts like the Bhagavad Gita. This challenging application demonstrates ArGen's adaptability, achieving a 70.9% improvement in domain-scope adherence over the baseline. Through our open-source repository, we show that ArGen's methodology offers a path to 'Governable Al' systems that are technically proficient, ethically robust, and verifiably compliant for safe deployment in diverse global contexts.
>
---
#### [new 003] Develop-Fair Use for Artificial Intelligence: A Sino-U.S. Copyright Law Comparison Based on the Ultraman, Bartz v. Anthropic, and Kadrey v. Meta Cases
- **分类: cs.CY**

- **简介: 论文比较中美国版权法，提出“Develop-Fair Use”概念，以应对生成式AI对传统合理使用的挑战。通过分析典型案例，主张将AI合理使用视为动态司法平衡工具，解决AI与出版业在市场中的竞争张力问题。**

- **链接: [http://arxiv.org/pdf/2509.07365v1](http://arxiv.org/pdf/2509.07365v1)**

> **作者:** Chanhou Lou
>
> **备注:** 9 pages
>
> **摘要:** Traditional fair use can no longer respond to the challenges posed by generative AI. Drawing on a comparative analysis of China's Ultraman and the U.S. cases Bartz v. Anthropic and Kadrey v. Meta, this article proposes "Develop-Fair Use" (DFU). DFU treats AI fair use (AIFU) not as a fixed exception but as a dynamic tool of judicial balancing that shifts analysis from closed scenarios to an evaluative rule for open-ended contexts. The judicial focus moves from formal classification of facts to a substantive balancing of competition in relevant markets. Although China and the U.S. follow different paths, both reveal this logic: Ultraman, by articulating a "four-context analysis," creates institutional space for AI industry development; the debate over the fourth factor, market impact, in the two U.S. cases, especially Kadrey's "market dilution" claim, expands review from substitution in copyright markets to wider industrial competition. The core of DFU is to recognize and balance the tension in relevant markets between an emerging AI industry that invokes fair use to build its markets and a publishing industry that develops markets, including one for "training licenses," to resist fair use. The boundary of fair use is therefore not a product of pure legal deduction but a case-specific factual judgment grounded in evolving market realities. This approach aims both to trim excess copyright scope and to remedy shortfalls in market competition.
>
---
#### [new 004] The Impact of Artificial Intelligence on Traditional Art Forms: A Disruption or Enhancement
- **分类: cs.CY; cs.AI**

- **简介: 该论文探讨AI对传统艺术的影响，分析其作为破坏或增强的双重作用。研究通过案例与数据，评估AI在艺术创作中的利弊，并提出伦理指南与协作方法，以促进艺术创新与文化传承。**

- **链接: [http://arxiv.org/pdf/2509.07029v1](http://arxiv.org/pdf/2509.07029v1)**

> **作者:** Viswa Chaitanya Marella; Sai Teja Erukude; Suhasnadh Reddy Veluru
>
> **备注:** 13 pages
>
> **摘要:** The introduction of Artificial Intelligence (AI) into the domains of traditional art (visual arts, performing arts, and crafts) has sparked a complicated discussion about whether this might be an agent of disruption or an enhancement of our traditional art forms. This paper looks at the duality of AI, exploring the ways that recent technologies like Generative Adversarial Networks and Diffusion Models, and text-to-image generators are changing the fields of painting, sculpture, calligraphy, dance, music, and the arts of craft. Using examples and data, we illustrate the ways that AI can democratize creative expression, improve productivity, and preserve cultural heritage, while also examining the negative aspects, including: the threats to authenticity within art, ethical concerns around data, and issues including socio-economic factors such as job losses. While we argue for the context-dependence of the impact of AI (the potential for creative homogenization and the devaluation of human agency in artmaking), we also illustrate the potential for hybrid practices featuring AI in cuisine, etc. We advocate for the development of ethical guidelines, collaborative approaches, and inclusive technology development. In sum, we are articulating a vision of AI in which it amplifies our innate creativity while resisting the displacement of the cultural, nuanced, and emotional aspects of traditional art. The future will be determined by human choices about how to govern AI so that it becomes a mechanism for artistic evolution and not a substitute for the artist's soul.
>
---
#### [new 005] Towards Postmortem Data Management Principles for Generative AI
- **分类: cs.CY**

- **简介: 该论文探讨生成式AI中已故用户数据的管理问题，提出三项管理原则，以保护其数据权利。研究分析现有隐私政策与法规，为政策制定者和隐私从业者提供指导。**

- **链接: [http://arxiv.org/pdf/2509.07375v1](http://arxiv.org/pdf/2509.07375v1)**

> **作者:** Ismat Jarin; Elina Van Kempen; Chloe Georgiou
>
> **摘要:** Foundation models, large language models (LLMs), and agentic AI systems rely heavily on vast corpora of user data. The use of such data for training has raised persistent concerns around ownership, copyright, and potential harms. In this work, we explore a related but less examined dimension: the ownership rights of data belonging to deceased individuals. We examine the current landscape of post-mortem data management and privacy rights as defined by the privacy policies of major technology companies and regulations such as the EU AI Act. Based on this analysis, we propose three post-mortem data management principles to guide the protection of deceased individuals data rights. Finally, we discuss directions for future work and offer recommendations for policymakers and privacy practitioners on deploying these principles alongside technological solutions to operationalize and audit them in practice.
>
---
#### [new 006] A Maslow-Inspired Hierarchy of Engagement with AI Model
- **分类: cs.CY; cs.AI**

- **简介: 该论文提出一个受马斯洛需求层次理论启发的AI参与层次模型，用于评估和指导AI成熟度。模型包含八个层级，涵盖技术、组织和伦理维度，并通过四个案例验证其适用性，旨在为学者和实践者提供分析与规划工具，解决AI应用中多维成熟度评估的问题。**

- **链接: [http://arxiv.org/pdf/2509.07032v1](http://arxiv.org/pdf/2509.07032v1)**

> **作者:** Madara Ogot
>
> **备注:** 30 pages, 14 tables
>
> **摘要:** The rapid proliferation of artificial intelligence (AI) across industry, government, and education highlights the urgent need for robust frameworks to conceptualise and guide engagement. This paper introduces the Hierarchy of Engagement with AI model, a novel maturity framework inspired by Maslow's hierarchy of needs. The model conceptualises AI adoption as a progression through eight levels, beginning with initial exposure and basic understanding and culminating in ecosystem collaboration and societal impact. Each level integrates technical, organisational, and ethical dimensions, emphasising that AI maturity is not only a matter of infrastructure and capability but also of trust, governance, and responsibility. Initial validation of the model using four diverse case studies (General Motors, the Government of Estonia, the University of Texas System, and the African Union AI Strategy) demonstrate the model's contextual flexibility across various sectors. The model provides scholars with a framework for analysing AI maturity and offers practitioners and policymakers a diagnostic and strategic planning tool to guide responsible and sustainable AI engagement. The proposed model demonstrates that AI maturity progression is multi-dimensional, requiring technological capability, ethical integrity, organisational resilience, and ecosystem collaboration.
>
---
#### [new 007] From Passive to Participatory: How Liberating Structures Can Revolutionize Our Conferences
- **分类: cs.CY; cs.SE**

- **简介: 论文探讨如何通过“解放结构”改革会议模式，解决当前学术会议提交量过大、质量下降的问题。提出分设两个会议轨道，促进互动与创新，提升学术交流质量。属于会议组织与学术交流优化任务。**

- **链接: [http://arxiv.org/pdf/2509.07046v1](http://arxiv.org/pdf/2509.07046v1)**

> **作者:** Daniel Russo; Margaret-Anne Storey
>
> **摘要:** Our conferences face a growing crisis: an overwhelming flood of submissions, increased reviewing burdens, and diminished opportunities for meaningful engagement. With AI making paper generation easier than ever, we must ask whether the current model fosters real innovation or simply incentivizes more publications. This article advocates for a shift from passive paper presentations to interactive, participatory formats. We propose Liberating Structures, facilitation techniques that promote collaboration and deeper intellectual exchange. By restructuring conferences into two tracks, one for generating new ideas and another for discussing established work, we can prioritize quality over quantity and reinvigorate academic gatherings. Embracing this change will ensure conferences remain spaces for real insight, creativity, and impactful collaboration in the AI era.
>
---
#### [new 008] Water Demand Forecasting of District Metered Areas through Learned Consumer Representations
- **分类: cs.LG; cs.AI; cs.CY**

- **简介: 该论文属于短期用水需求预测任务，旨在解决DMA中因非确定性因素导致的预测难题。通过无监督对比学习识别用户行为特征，并结合波变换卷积网络与跨注意力机制提升预测精度，实验证明其在MAPE指标上表现更优。**

- **链接: [http://arxiv.org/pdf/2509.07515v1](http://arxiv.org/pdf/2509.07515v1)**

> **作者:** Adithya Ramachandran; Thorkil Flensmark B. Neergaard; Tomás Arias-Vergara; Andreas Maier; Siming Bayer
>
> **备注:** Presented at European Conference for Signal Procesing - EUSIPCO 2025
>
> **摘要:** Advancements in smart metering technologies have significantly improved the ability to monitor and manage water utilities. In the context of increasing uncertainty due to climate change, securing water resources and supply has emerged as an urgent global issue with extensive socioeconomic ramifications. Hourly consumption data from end-users have yielded substantial insights for projecting demand across regions characterized by diverse consumption patterns. Nevertheless, the prediction of water demand remains challenging due to influencing non-deterministic factors, such as meteorological conditions. This work introduces a novel method for short-term water demand forecasting for District Metered Areas (DMAs) which encompass commercial, agricultural, and residential consumers. Unsupervised contrastive learning is applied to categorize end-users according to distinct consumption behaviors present within a DMA. Subsequently, the distinct consumption behaviors are utilized as features in the ensuing demand forecasting task using wavelet-transformed convolutional networks that incorporate a cross-attention mechanism combining both historical data and the derived representations. The proposed approach is evaluated on real-world DMAs over a six-month period, demonstrating improved forecasting performance in terms of MAPE across different DMAs, with a maximum improvement of 4.9%. Additionally, it identifies consumers whose behavior is shaped by socioeconomic factors, enhancing prior knowledge about the deterministic patterns that influence demand.
>
---
#### [new 009] Individual utilities of life satisfaction reveal inequality aversion unrelated to political alignment
- **分类: econ.GN; cs.AI; cs.CY; q-fin.EC**

- **简介: 该论文通过实验研究人们在不确定条件下的幸福感偏好，发现多数人风险厌恶且更关注社会公平，与政治立场无关。研究挑战平均幸福感作为政策指标的合理性，支持非线性效用模型的应用。任务为分析幸福感优先级与公平权衡问题。**

- **链接: [http://arxiv.org/pdf/2509.07793v1](http://arxiv.org/pdf/2509.07793v1)**

> **作者:** Crispin Cooper; Ana Friedrich; Tommaso Reggiani; Wouter Poortinga
>
> **备注:** 28 pages, 4 figures
>
> **摘要:** How should well-being be prioritised in society, and what trade-offs are people willing to make between fairness and personal well-being? We investigate these questions using a stated preference experiment with a nationally representative UK sample (n = 300), in which participants evaluated life satisfaction outcomes for both themselves and others under conditions of uncertainty. Individual-level utility functions were estimated using an Expected Utility Maximisation (EUM) framework and tested for sensitivity to the overweighting of small probabilities, as characterised by Cumulative Prospect Theory (CPT). A majority of participants displayed concave (risk-averse) utility curves and showed stronger aversion to inequality in societal life satisfaction outcomes than to personal risk. These preferences were unrelated to political alignment, suggesting a shared normative stance on fairness in well-being that cuts across ideological boundaries. The results challenge use of average life satisfaction as a policy metric, and support the development of nonlinear utility-based alternatives that more accurately reflect collective human values. Implications for public policy, well-being measurement, and the design of value-aligned AI systems are discussed.
>
---
#### [new 010] VISION: Robust and Interpretable Code Vulnerability Detection Leveraging Counterfactual Augmentation
- **分类: cs.AI; cs.CR; cs.CY; cs.LG**

- **简介: 该论文提出VISION框架，用于提升代码漏洞检测的鲁棒性与可解释性。通过生成反事实样本增强训练数据，解决GNN因数据不平衡和标签噪声导致的虚假关联问题，显著提升检测准确率，并发布CWE-20-CFA基准数据集。**

- **链接: [http://arxiv.org/pdf/2508.18933v1](http://arxiv.org/pdf/2508.18933v1)**

> **作者:** David Egea; Barproda Halder; Sanghamitra Dutta
>
> **摘要:** Automated detection of vulnerabilities in source code is an essential cybersecurity challenge, underpinning trust in digital systems and services. Graph Neural Networks (GNNs) have emerged as a promising approach as they can learn structural and logical code relationships in a data-driven manner. However, their performance is severely constrained by training data imbalances and label noise. GNNs often learn 'spurious' correlations from superficial code similarities, producing detectors that fail to generalize well to unseen real-world data. In this work, we propose a unified framework for robust and interpretable vulnerability detection, called VISION, to mitigate spurious correlations by systematically augmenting a counterfactual training dataset. Counterfactuals are samples with minimal semantic modifications but opposite labels. Our framework includes: (i) generating counterfactuals by prompting a Large Language Model (LLM); (ii) targeted GNN training on paired code examples with opposite labels; and (iii) graph-based interpretability to identify the crucial code statements relevant for vulnerability predictions while ignoring spurious ones. We find that VISION reduces spurious learning and enables more robust, generalizable detection, improving overall accuracy (from 51.8% to 97.8%), pairwise contrast accuracy (from 4.5% to 95.8%), and worst-group accuracy (from 0.7% to 85.5%) on the Common Weakness Enumeration (CWE)-20 vulnerability. We further demonstrate gains using proposed metrics: intra-class attribution variance, inter-class attribution distance, and node score dependency. We also release CWE-20-CFA, a benchmark of 27,556 functions (real and counterfactual) from the high-impact CWE-20 category. Finally, VISION advances transparent and trustworthy AI-based cybersecurity systems through interactive visualization for human-in-the-loop analysis.
>
---
#### [new 011] Automated Evaluation of Gender Bias Across 13 Large Multimodal Models
- **分类: cs.CV; cs.AI; cs.CY; I.2.7; F.2.2**

- **简介: 该论文评估13种大 multimodal 模型中的性别偏见，通过生成职业图像并分析性别表现，揭示模型放大职业性别刻板印象的问题，并提出自动化评估工具，推动AI公平性研究。**

- **链接: [http://arxiv.org/pdf/2509.07050v1](http://arxiv.org/pdf/2509.07050v1)**

> **作者:** Juan Manuel Contreras
>
> **摘要:** Large multimodal models (LMMs) have revolutionized text-to-image generation, but they risk perpetuating the harmful social biases in their training data. Prior work has identified gender bias in these models, but methodological limitations prevented large-scale, comparable, cross-model analysis. To address this gap, we introduce the Aymara Image Fairness Evaluation, a benchmark for assessing social bias in AI-generated images. We test 13 commercially available LMMs using 75 procedurally-generated, gender-neutral prompts to generate people in stereotypically-male, stereotypically-female, and non-stereotypical professions. We then use a validated LLM-as-a-judge system to score the 965 resulting images for gender representation. Our results reveal (p < .001 for all): 1) LMMs systematically not only reproduce but actually amplify occupational gender stereotypes relative to real-world labor data, generating men in 93.0% of images for male-stereotyped professions but only 22.5% for female-stereotyped professions; 2) Models exhibit a strong default-male bias, generating men in 68.3% of the time for non-stereotyped professions; and 3) The extent of bias varies dramatically across models, with overall male representation ranging from 46.7% to 73.3%. Notably, the top-performing model de-amplified gender stereotypes and approached gender parity, achieving the highest fairness scores. This variation suggests high bias is not an inevitable outcome but a consequence of design choices. Our work provides the most comprehensive cross-model benchmark of gender bias to date and underscores the necessity of standardized, automated evaluation tools for promoting accountability and fairness in AI development.
>
---
#### [new 012] LLM Analysis of 150+ years of German Parliamentary Debates on Migration Reveals Shift from Post-War Solidarity to Anti-Solidarity in the Last Decade
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 论文利用大语言模型分析150多年德国议会关于移民的辩论，揭示从战后团结到近年反团结的转变。任务是自动化标注政治文本中的（反）团结倾向，解决传统人工标注效率低的问题，评估不同模型效果并分析社会趋势。**

- **链接: [http://arxiv.org/pdf/2509.07274v1](http://arxiv.org/pdf/2509.07274v1)**

> **作者:** Aida Kostikova; Ole Pütz; Steffen Eger; Olga Sabelfeld; Benjamin Paassen
>
> **摘要:** Migration has been a core topic in German political debate, from millions of expellees post World War II over labor migration to refugee movements in the recent past. Studying political speech regarding such wide-ranging phenomena in depth traditionally required extensive manual annotations, limiting the scope of analysis to small subsets of the data. Large language models (LLMs) have the potential to partially automate even complex annotation tasks. We provide an extensive evaluation of a multiple LLMs in annotating (anti-)solidarity subtypes in German parliamentary debates compared to a large set of thousands of human reference annotations (gathered over a year). We evaluate the influence of model size, prompting differences, fine-tuning, historical versus contemporary data; and we investigate systematic errors. Beyond methodological evaluation, we also interpret the resulting annotations from a social science lense, gaining deeper insight into (anti-)solidarity trends towards migrants in the German post-World War II period and recent past. Our data reveals a high degree of migrant-directed solidarity in the postwar period, as well as a strong trend towards anti-solidarity in the German parliament since 2015, motivating further research. These findings highlight the promise of LLMs for political text analysis and the importance of migration debates in Germany, where demographic decline and labor shortages coexist with rising polarization.
>
---
#### [new 013] The Signalgate Case is Waiving a Red Flag to All Organizational and Behavioral Cybersecurity Leaders, Practitioners, and Researchers: Are We Receiving the Signal Amidst the Noise?
- **分类: cs.CR; cs.CY; J.4; K.4.1; K.4.3; K.5.0; K.5.2; K.6.5**

- **简介: 论文分析2025年Signalgate事件，探讨组织安全中的人为失误与治理缺陷。通过案例研究和NIST框架，提出加强领导力、零信任架构及行为激励等建议，以提升组织与国家安全水平。**

- **链接: [http://arxiv.org/pdf/2509.07053v1](http://arxiv.org/pdf/2509.07053v1)**

> **作者:** Paul Benjamin Lowry; Gregory D. Moody; Robert Willison; Clay Posey
>
> **摘要:** The Signalgate incident of March 2025, wherein senior US national security officials inadvertently disclosed sensitive military operational details via the encrypted messaging platform Signal, highlights critical vulnerabilities in organizational security arising from human error, governance gaps, and the misuse of technology. Although smaller in scale when compared to historical breaches involving billions of records, Signalgate illustrates critical systemic issues often overshadowed by a focus on external cyber threats. Employing a case-study approach and systematic review grounded in the NIST Cybersecurity Framework, we analyze the incident to identify patterns of human-centric vulnerabilities and governance challenges common to organizational security failures. Findings emphasize three critical points. (1) Organizational security depends heavily on human behavior, with internal actors often serving as the weakest link despite advanced technical defenses; (2) Leadership tone strongly influences organizational security culture and efficacy, and (3) widespread reliance on technical solutions without sufficient investments in human and organizational factors leads to ineffective practices and wasted resources. From these observations, we propose actionable recommendations for enhancing organizational and national security, including strong leadership engagement, comprehensive adoption of zero-trust architectures, clearer accountability structures, incentivized security behaviors, and rigorous oversight. Particularly during periods of organizational transition, such as mergers or large-scale personnel changes, additional measures become particularly important. Signalgate underscores the need for leaders and policymakers to reorient cybersecurity strategies toward addressing governance, cultural, and behavioral risks.
>
---
#### [new 014] Explaining How Quantization Disparately Skews a Model
- **分类: cs.LG; cs.AI; cs.CY**

- **简介: 该论文研究量化对模型在不同群体上的不均衡影响，分析其对准确率、梯度和Hessian矩阵的影响，并提出混合精度QAT与采样方法结合的解决方案，以实现公平的量化部署。属于模型压缩与公平性优化任务。**

- **链接: [http://arxiv.org/pdf/2509.07222v1](http://arxiv.org/pdf/2509.07222v1)**

> **作者:** Abhimanyu Bellam; Jung-Eun Kim
>
> **摘要:** Post Training Quantization (PTQ) is widely adopted due to its high compression capacity and speed with minimal impact on accuracy. However, we observed that disparate impacts are exacerbated by quantization, especially for minority groups. Our analysis explains that in the course of quantization there is a chain of factors attributed to a disparate impact across groups during forward and backward passes. We explore how the changes in weights and activations induced by quantization cause cascaded impacts in the network, resulting in logits with lower variance, increased loss, and compromised group accuracies. We extend our study to verify the influence of these impacts on group gradient norms and eigenvalues of the Hessian matrix, providing insights into the state of the network from an optimization point of view. To mitigate these effects, we propose integrating mixed precision Quantization Aware Training (QAT) with dataset sampling methods and weighted loss functions, therefore providing fair deployment of quantized neural networks.
>
---
#### [new 015] Wellbeing-Centered UX: Supporting Content Moderators
- **分类: cs.HC; cs.CY; J.4; K.4.1**

- **简介: 该论文探讨如何通过UX设计提升内容审核员的福祉。它属于用户体验研究任务，旨在解决审核员面临的心理压力与工具不足问题，提出贯穿产品开发周期的框架与策略，以改善其工作体验。**

- **链接: [http://arxiv.org/pdf/2509.07187v1](http://arxiv.org/pdf/2509.07187v1)**

> **作者:** Diana Mihalache; Dalila Szostak
>
> **备注:** In M. L. Daniel, A. Menking, M. T. Savio, & J. Claffey (Eds.) (In Press, upcoming), Trust, Safety, and the Internet We Share: Multistakeholder Insights. Taylor & Francis
>
> **摘要:** This chapter focuses on the intersection of user experience (UX) and wellbeing in the context of content moderation. Human content moderators play a key role in protecting end users from harm by detecting, evaluating, and addressing content that may violate laws or product policies. They face numerous challenges, including exposure to sensitive content, monotonous tasks, and complex decisions, which are often exacerbated by inadequate tools. This chapter explains the importance of incorporating wellbeing considerations throughout the product development lifecycle, offering a framework and practical strategies for implementation across key UX disciplines: research, writing, and design. By examining these considerations, this chapter provides a roadmap for creating user experiences that support content moderators, benefiting both the user and the business.
>
---
#### [new 016] That's So FETCH: Fashioning Ensemble Techniques for LLM Classification in Civil Legal Intake and Referral
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 论文提出FETCH分类器，用于法律问题分类，解决法律援助中准确匹配用户问题与资源的难题。通过混合LLM/ML模型和生成追问问题提升分类精度，使用真实数据集实现97.37%的高准确率，降低成本。**

- **链接: [http://arxiv.org/pdf/2509.07170v1](http://arxiv.org/pdf/2509.07170v1)**

> **作者:** Quinten Steenhuis
>
> **备注:** Submission to JURIX 2025
>
> **摘要:** Each year millions of people seek help for their legal problems by calling a legal aid program hotline, walking into a legal aid office, or using a lawyer referral service. The first step to match them to the right help is to identify the legal problem the applicant is experiencing. Misdirection has consequences. Applicants may miss a deadline, experience physical abuse, lose housing or lose custody of children while waiting to connect to the right legal help. We introduce and evaluate the FETCH classifier for legal issue classification and describe two methods for improving accuracy: a hybrid LLM/ML ensemble classification method, and the automatic generation of follow-up questions to enrich the initial problem narrative. We employ a novel data set of 419 real-world queries to a nonprofit lawyer referral service. Ultimately, we show classification accuracy (hits@2) of 97.37\% using a mix of inexpensive models, exceeding the performance of the current state-of-the-art GPT-5 model. Our approach shows promise in significantly reducing the cost of guiding users of the legal system to the right resource for their problem while achieving high accuracy.
>
---
## 更新

#### [replaced 001] Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives
- **分类: cs.GT; cs.AI; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.21627v2](http://arxiv.org/pdf/2505.21627v2)**

> **作者:** Ander Artola Velasco; Stratis Tsirtsis; Nastaran Okati; Manuel Gomez-Rodriguez
>
> **摘要:** State-of-the-art large language models require specialized hardware and substantial energy to operate. As a consequence, cloud-based services that provide access to large language models have become very popular. In these services, the price users pay for an output provided by a model depends on the number of tokens the model uses to generate it -- they pay a fixed price per token. In this work, we show that this pricing mechanism creates a financial incentive for providers to strategize and misreport the (number of) tokens a model used to generate an output, and users cannot prove, or even know, whether a provider is overcharging them. However, we also show that, if an unfaithful provider is obliged to be transparent about the generative process used by the model, misreporting optimally without raising suspicion is hard. Nevertheless, as a proof-of-concept, we develop an efficient heuristic algorithm that allows providers to significantly overcharge users without raising suspicion. Crucially, we demonstrate that the cost of running the algorithm is lower than the additional revenue from overcharging users, highlighting the vulnerability of users under the current pay-per-token pricing mechanism. Further, we show that, to eliminate the financial incentive to strategize, a pricing mechanism must price tokens linearly on their character count. While this makes a provider's profit margin vary across tokens, we introduce a simple prescription under which the provider who adopts such an incentive-compatible pricing mechanism can maintain the average profit margin they had under the pay-per-token pricing mechanism. Along the way, to illustrate and complement our theoretical results, we conduct experiments with several large language models from the $\texttt{Llama}$, $\texttt{Gemma}$ and $\texttt{Ministral}$ families, and input prompts from the LMSYS Chatbot Arena platform.
>
---
#### [replaced 002] Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2508.17450v3](http://arxiv.org/pdf/2508.17450v3)**

> **作者:** Bryan Chen Zhengyu Tan; Daniel Wai Kit Chin; Zhengyuan Liu; Nancy F. Chen; Roy Ka-Wei Lee
>
> **备注:** To appear at EMNLP 2025
>
> **摘要:** Large Language Models (LLMs) can struggle to balance gullibility to misinformation and resistance to valid corrections in persuasive dialogues, a critical challenge for reliable deployment. We introduce DuET-PD (Dual Evaluation for Trust in Persuasive Dialogues), a framework evaluating multi-turn stance-change dynamics across dual dimensions: persuasion type (corrective/misleading) and domain (knowledge via MMLU-Pro, and safety via SALAD-Bench). We find that even a state-of-the-art model like GPT-4o achieves only 27.32% accuracy in MMLU-Pro under sustained misleading persuasions. Moreover, results reveal a concerning trend of increasing sycophancy in newer open-source models. To address this, we introduce Holistic DPO, a training approach balancing positive and negative persuasion examples. Unlike prompting or resist-only training, Holistic DPO enhances both robustness to misinformation and receptiveness to corrections, improving Llama-3.1-8B-Instruct's accuracy under misleading persuasion in safety contexts from 4.21% to 76.54%. These contributions offer a pathway to developing more reliable and adaptable LLMs for multi-turn dialogue. Code is available at https://github.com/Social-AI-Studio/DuET-PD.
>
---
#### [replaced 003] Pilot Study on Generative AI and Critical Thinking in Higher Education Classrooms
- **分类: cs.CY; cs.AI; cs.HC; stat.AP**

- **链接: [http://arxiv.org/pdf/2509.00167v3](http://arxiv.org/pdf/2509.00167v3)**

> **作者:** W. F. Lamberti; S. R. Lawrence; D. White; S. Kim; S. Abdullah
>
> **摘要:** Generative AI (GAI) tools have seen rapid adoption in educational settings, yet their role in fostering critical thinking remains underexplored. While previous studies have examined GAI as a tutor for specific lessons or as a tool for completing assignments, few have addressed how students critically evaluate the accuracy and appropriateness of GAI-generated responses. This pilot study investigates students' ability to apply structured critical thinking when assessing Generative AI outputs in introductory Computational and Data Science courses. Given that GAI tools often produce contextually flawed or factually incorrect answers, we designed learning activities that require students to analyze, critique, and revise AI-generated solutions. Our findings offer initial insights into students' ability to engage critically with GAI content and lay the groundwork for more comprehensive studies in future semesters.
>
---
#### [replaced 004] The Model Hears You: Audio Language Model Deployments Should Consider the Principle of Least Privilege
- **分类: cs.SD; cs.AI; cs.CL; cs.CY; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.16833v2](http://arxiv.org/pdf/2503.16833v2)**

> **作者:** Luxi He; Xiangyu Qi; Michel Liao; Inyoung Cheong; Prateek Mittal; Danqi Chen; Peter Henderson
>
> **备注:** Published at AIES 2025
>
> **摘要:** The latest Audio Language Models (Audio LMs) process speech directly instead of relying on a separate transcription step. This shift preserves detailed information, such as intonation or the presence of multiple speakers, that would otherwise be lost in transcription. However, it also introduces new safety risks, including the potential misuse of speaker identity cues and other sensitive vocal attributes, which could have legal implications. In this paper, we urge a closer examination of how these models are built and deployed. Our experiments show that end-to-end modeling, compared with cascaded pipelines, creates socio-technical safety risks such as identity inference, biased decision-making, and emotion detection. This raises concerns about whether Audio LMs store voiceprints and function in ways that create uncertainty under existing legal regimes. We then argue that the Principle of Least Privilege should be considered to guide the development and deployment of these models. Specifically, evaluations should assess (1) the privacy and safety risks associated with end-to-end modeling; and (2) the appropriate scope of information access. Finally, we highlight related gaps in current audio LM benchmarks and identify key open research questions, both technical and policy-related, that must be addressed to enable the responsible deployment of end-to-end Audio LMs.
>
---
#### [replaced 005] No Thoughts Just AI: Biased LLM Hiring Recommendations Alter Human Decision Making and Limit Human Autonomy
- **分类: cs.CY; cs.AI; cs.CL; cs.HC; K.4.2**

- **链接: [http://arxiv.org/pdf/2509.04404v2](http://arxiv.org/pdf/2509.04404v2)**

> **作者:** Kyra Wilson; Mattea Sim; Anna-Maria Gueorguieva; Aylin Caliskan
>
> **备注:** Published in Proceedings of the 2025 AAAI/ACM Conference on AI, Ethics, and Society; code available at https://github.com/kyrawilson/No-Thoughts-Just-AI
>
> **摘要:** In this study, we conduct a resume-screening experiment (N=528) where people collaborate with simulated AI models exhibiting race-based preferences (bias) to evaluate candidates for 16 high and low status occupations. Simulated AI bias approximates factual and counterfactual estimates of racial bias in real-world AI systems. We investigate people's preferences for White, Black, Hispanic, and Asian candidates (represented through names and affinity groups on quality-controlled resumes) across 1,526 scenarios and measure their unconscious associations between race and status using implicit association tests (IATs), which predict discriminatory hiring decisions but have not been investigated in human-AI collaboration. When making decisions without AI or with AI that exhibits no race-based preferences, people select all candidates at equal rates. However, when interacting with AI favoring a particular group, people also favor those candidates up to 90% of the time, indicating a significant behavioral shift. The likelihood of selecting candidates whose identities do not align with common race-status stereotypes can increase by 13% if people complete an IAT before conducting resume screening. Finally, even if people think AI recommendations are low quality or not important, their decisions are still vulnerable to AI bias under certain circumstances. This work has implications for people's autonomy in AI-HITL scenarios, AI and work, design and evaluation of AI hiring systems, and strategies for mitigating bias in collaborative decision-making tasks. In particular, organizational and regulatory policy should acknowledge the complex nature of AI-HITL decision making when implementing these systems, educating people who use them, and determining which are subject to oversight.
>
---
#### [replaced 006] Automatically Detecting Online Deceptive Patterns
- **分类: cs.HC; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2411.07441v3](http://arxiv.org/pdf/2411.07441v3)**

> **作者:** Asmit Nayak; Shirley Zhang; Yash Wani; Rishabh Khandelwal; Kassem Fawaz
>
> **摘要:** Deceptive patterns in digital interfaces manipulate users into making unintended decisions, exploiting cognitive biases and psychological vulnerabilities. These patterns have become ubiquitous on various digital platforms. While efforts to mitigate deceptive patterns have emerged from legal and technical perspectives, a significant gap remains in creating usable and scalable solutions. We introduce our AutoBot framework to address this gap and help web stakeholders navigate and mitigate online deceptive patterns. AutoBot accurately identifies and localizes deceptive patterns from a screenshot of a website without relying on the underlying HTML code. AutoBot employs a two-stage pipeline that leverages the capabilities of specialized vision models to analyze website screenshots, identify interactive elements, and extract textual features. Next, using a large language model, AutoBot understands the context surrounding these elements to determine the presence of deceptive patterns. We also use AutoBot, to create a synthetic dataset to distill knowledge from 'teacher' LLMs to smaller language models. Through extensive evaluation, we demonstrate AutoBot's effectiveness in detecting deceptive patterns on the web, achieving an F1-score of 0.93 when detecting deceptive patterns, underscoring its potential as an essential tool for mitigating online deceptive patterns. We implement AutoBot, across three downstream applications targeting different web stakeholders: (1) a local browser extension providing users with real-time feedback, (2) a Lighthouse audit to inform developers of potential deceptive patterns on their sites, and (3) as a measurement tool designed for researchers and regulators.
>
---
#### [replaced 007] SemCAFE: When Named Entities make the Difference Assessing Web Source Reliability through Entity-level Analytics
- **分类: cs.CL; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.08776v2](http://arxiv.org/pdf/2504.08776v2)**

> **作者:** Gautam Kishore Shahi; Oshani Seneviratne; Marc Spaniol
>
> **摘要:** With the shift from traditional to digital media, the online landscape now hosts not only reliable news articles but also a significant amount of unreliable content. Digital media has faster reachability by significantly influencing public opinion and advancing political agendas. While newspaper readers may be familiar with their preferred outlets political leanings or credibility, determining unreliable news articles is much more challenging. The credibility of many online sources is often opaque, with AI generated content being easily disseminated at minimal cost. Unreliable news articles, particularly those that followed the Russian invasion of Ukraine in 2022, closely mimic the topics and writing styles of credible sources, making them difficult to distinguish. To address this, we introduce SemCAFE, a system designed to detect news reliability by incorporating entity relatedness into its assessment. SemCAFE employs standard Natural Language Processing techniques, such as boilerplate removal and tokenization, alongside entity level semantic analysis using the YAGO knowledge base. By creating a semantic fingerprint for each news article, SemCAFE could assess the credibility of 46,020 reliable and 3,407 unreliable articles on the 2022 Russian invasion of Ukraine. Our approach improved the macro F1 score by 12% over state of the art methods. The sample data and code are available on GitHub
>
---
