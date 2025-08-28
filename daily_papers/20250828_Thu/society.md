# 计算机与社会 cs.CY

- **最新发布 13 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] Are Companies Taking AI Risks Seriously? A Systematic Analysis of Companies' AI Risk Disclosures in SEC 10-K forms
- **分类: cs.CY; cs.AI**

- **简介: 论文通过分析SEC 10-K文件中的AI风险披露，系统评估公司对AI风险的重视程度，发现提及率从4%升至43%，但披露内容多为泛化，缺乏具体措施，并发布工具支持进一步研究。**

- **链接: [http://arxiv.org/pdf/2508.19313v1](http://arxiv.org/pdf/2508.19313v1)**

> **作者:** Lucas G. Uberti-Bona Marin; Bram Rijsbosch; Gerasimos Spanakis; Konrad Kollnig
>
> **备注:** To be published in the ECML PKDD SoGood (Data Science for Social Good) workshop proceedings
>
> **摘要:** As Artificial Intelligence becomes increasingly central to corporate strategies, concerns over its risks are growing too. In response, regulators are pushing for greater transparency in how companies identify, report and mitigate AI-related risks. In the US, the Securities and Exchange Commission (SEC) repeatedly warned companies to provide their investors with more accurate disclosures of AI-related risks; recent enforcement and litigation against companies' misleading AI claims reinforce these warnings. In the EU, new laws - like the AI Act and Digital Services Act - introduced additional rules on AI risk reporting and mitigation. Given these developments, it is essential to examine if and how companies report AI-related risks to the public. This study presents the first large-scale systematic analysis of AI risk disclosures in SEC 10-K filings, which require public companies to report material risks to their company. We analyse over 30,000 filings from more than 7,000 companies over the past five years, combining quantitative and qualitative analysis. Our findings reveal a sharp increase in the companies that mention AI risk, up from 4% in 2020 to over 43% in the most recent 2024 filings. While legal and competitive AI risks are the most frequently mentioned, we also find growing attention to societal AI risks, such as cyberattacks, fraud, and technical limitations of AI systems. However, many disclosures remain generic or lack details on mitigation strategies, echoing concerns raised recently by the SEC about the quality of AI-related risk reporting. To support future research, we publicly release a web-based tool for easily extracting and analysing keyword-based disclosures across SEC filings.
>
---
#### [new 002] Geopolitical Parallax: Beyond Walter Lippmann Just After Large Language Models
- **分类: cs.CY; cs.CL**

- **简介: 该论文通过对比中西方LLM在新闻质量评估中的表现，揭示地缘政治偏见对主观性、情感等指标的影响，旨在识别模型固有偏见对媒体评价的干扰。**

- **链接: [http://arxiv.org/pdf/2508.19492v1](http://arxiv.org/pdf/2508.19492v1)**

> **作者:** Mehmet Can Yavuz; Humza Gohar Kabir; Aylin Özkan
>
> **备注:** 7 pages, 4 figures, 7 tables
>
> **摘要:** Objectivity in journalism has long been contested, oscillating between ideals of neutral, fact-based reporting and the inevitability of subjective framing. With the advent of large language models (LLMs), these tensions are now mediated by algorithmic systems whose training data and design choices may themselves embed cultural or ideological biases. This study investigates geopolitical parallax-systematic divergence in news quality and subjectivity assessments-by comparing article-level embeddings from Chinese-origin (Qwen, BGE, Jina) and Western-origin (Snowflake, Granite) model families. We evaluate both on a human-annotated news quality benchmark spanning fifteen stylistic, informational, and affective dimensions, and on parallel corpora covering politically sensitive topics, including Palestine and reciprocal China-United States coverage. Using logistic regression probes and matched-topic evaluation, we quantify per-metric differences in predicted positive-class probabilities between model families. Our findings reveal consistent, non-random divergences aligned with model origin. In Palestine-related coverage, Western models assign higher subjectivity and positive emotion scores, while Chinese models emphasize novelty and descriptiveness. Cross-topic analysis shows asymmetries in structural quality metrics Chinese-on-US scoring notably lower in fluency, conciseness, technicality, and overall quality-contrasted by higher negative emotion scores. These patterns align with media bias theory and our distinction between semantic, emotional, and relational subjectivity, and extend LLM bias literature by showing that geopolitical framing effects persist in downstream quality assessment tasks. We conclude that LLM-based media evaluation pipelines require cultural calibration to avoid conflating content differences with model-induced bias.
>
---
#### [new 003] Should LLMs be WEIRD? Exploring WEIRDness and Human Rights in Large Language Models
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文评估LLMs的WEIRD性与人权原则的冲突，通过测试五种模型在世界价值观调查中的响应，发现低WEIRD对齐模型更易生成违反人权的内容，强调需嵌入人权原则以平衡文化多样性与公平。**

- **链接: [http://arxiv.org/pdf/2508.19269v1](http://arxiv.org/pdf/2508.19269v1)**

> **作者:** Ke Zhou; Marios Constantinides; Daniele Quercia
>
> **备注:** This paper has been accepted in AIES 2025
>
> **摘要:** Large language models (LLMs) are often trained on data that reflect WEIRD values: Western, Educated, Industrialized, Rich, and Democratic. This raises concerns about cultural bias and fairness. Using responses to the World Values Survey, we evaluated five widely used LLMs: GPT-3.5, GPT-4, Llama-3, BLOOM, and Qwen. We measured how closely these responses aligned with the values of the WEIRD countries and whether they conflicted with human rights principles. To reflect global diversity, we compared the results with the Universal Declaration of Human Rights and three regional charters from Asia, the Middle East, and Africa. Models with lower alignment to WEIRD values, such as BLOOM and Qwen, produced more culturally varied responses but were 2% to 4% more likely to generate outputs that violated human rights, especially regarding gender and equality. For example, some models agreed with the statements ``a man who cannot father children is not a real man'' and ``a husband should always know where his wife is'', reflecting harmful gender norms. These findings suggest that as cultural representation in LLMs increases, so does the risk of reproducing discriminatory beliefs. Approaches such as Constitutional AI, which could embed human rights principles into model behavior, may only partly help resolve this tension.
>
---
#### [new 004] Hallucinating with AI: AI Psychosis as Distributed Delusions
- **分类: cs.CY; cs.AI**

- **简介: 论文从分布式认知理论分析AI与人类交互中产生的“AI幻觉”现象，揭示AI如何通过引入错误或强化用户妄想导致集体认知失真，提出“AI精神病”概念，探讨AI作为认知工具与拟人化他者的双重角色。**

- **链接: [http://arxiv.org/pdf/2508.19588v1](http://arxiv.org/pdf/2508.19588v1)**

> **作者:** Lucy Osler
>
> **摘要:** There is much discussion of the false outputs that generative AI systems such as ChatGPT, Claude, Gemini, DeepSeek, and Grok create. In popular terminology, these have been dubbed AI hallucinations. However, deeming these AI outputs hallucinations is controversial, with many claiming this is a metaphorical misnomer. Nevertheless, in this paper, I argue that when viewed through the lens of distributed cognition theory, we can better see the dynamic and troubling ways in which inaccurate beliefs, distorted memories and self-narratives, and delusional thinking can emerge through human-AI interactions; examples of which are popularly being referred to as cases of AI psychosis. In such cases, I suggest we move away from thinking about how an AI system might hallucinate at us, by generating false outputs, to thinking about how, when we routinely rely on generative AI to help us think, remember, and narrate, we can come to hallucinate with AI. This can happen when AI introduces errors into the distributed cognitive process, but it can also happen when AI sustains, affirms, and elaborates on our own delusional thinking and self-narratives, such as in the case of Jaswant Singh Chail. I also examine how the conversational style of chatbots can lead them to play a dual-function, both as a cognitive artefact and a quasi-Other with whom we co-construct our beliefs, narratives, and our realities. It is this dual function, I suggest, that makes generative AI an unusual, and particularly seductive, case of distributed cognition.
>
---
#### [new 005] Bridging the Regulatory Divide: Ensuring Safety and Equity in Wearable Health Technologies
- **分类: cs.CY**

- **简介: 论文旨在构建更包容的监管框架，解决可穿戴健康设备在安全与公平方面的监管空白，通过分布式风险和患者中心方法提出迭代改革方案。**

- **链接: [http://arxiv.org/pdf/2508.20031v1](http://arxiv.org/pdf/2508.20031v1)**

> **作者:** Akshay Kelshiker; Susan Cheng; Jivan Achar; Jane Bambauer; Leo Anthony Celi; Divya Jain; Thinh Nguyen; Harsh Patel; Nina Prakash; Alice Wong; Barbara Evans
>
> **备注:** 15 pages
>
> **摘要:** As wearable health technologies have grown more sophisticated, the distinction between "wellness" and "medical" devices has become increasingly blurred. While some features undergo formal U.S. Food and Drug Administration (FDA) review, many over-the-counter tools operate in a regulatory grey zone, leveraging health-related data and outputs without clinical validation. Further complicating the issue is the widespread repurposing of wellness devices for medical uses, which can introduce safety risks beyond the reach of current oversight. Drawing on legal analysis, case studies, and ethical considerations, we propose an approach emphasizing distributed risk, patient-centered outcomes, and iterative reform. Without a more pluralistic and evolving framework, the promise of wearable health technology risks being undermined by growing inequities, misuse, and eroded public trust.
>
---
#### [new 006] What Makes AI Applications Acceptable or Unacceptable? A Predictive Moral Framework
- **分类: cs.CY; cs.AI**

- **简介: 该论文构建了一个预测性道德框架，通过分析五项核心道德因素（风险、利益、不诚实、非自然性、责任降低）来预测公众对AI应用的接受度，基于大规模研究揭示了道德心理对技术评估的影响。**

- **链接: [http://arxiv.org/pdf/2508.19317v1](http://arxiv.org/pdf/2508.19317v1)**

> **作者:** Kimmo Eriksson; Simon Karlsson; Irina Vartanova; Pontus Strimling
>
> **备注:** 15 pages + supplementary materials, 3 figures
>
> **摘要:** As artificial intelligence rapidly transforms society, developers and policymakers struggle to anticipate which applications will face public moral resistance. We propose that these judgments are not idiosyncratic but systematic and predictable. In a large, preregistered study (N = 587, U.S. representative sample), we used a comprehensive taxonomy of 100 AI applications spanning personal and organizational contexts-including both functional uses and the moral treatment of AI itself. In participants' collective judgment, applications ranged from highly unacceptable to fully acceptable. We found this variation was strongly predictable: five core moral qualities-perceived risk, benefit, dishonesty, unnaturalness, and reduced accountability-collectively explained over 90% of the variance in acceptability ratings. The framework demonstrated strong predictive power across all domains and successfully predicted individual-level judgments for held-out applications. These findings reveal that a structured moral psychology underlies public evaluation of new technologies, offering a powerful tool for anticipating public resistance and guiding responsible innovation in AI.
>
---
#### [new 007] Epistemic Trade-Off: An Analysis of the Operational Breakdown and Ontological Limits of "Certainty-Scope" in AI
- **分类: cs.CY; cs.AI; cs.CE**

- **简介: 该论文分析Floridi关于AI确定性与范围权衡的猜想，揭示其理论与实践局限（不可计算性、本体论假设），并提出应对AI复杂领域知识负担的解决方案。**

- **链接: [http://arxiv.org/pdf/2508.19304v1](http://arxiv.org/pdf/2508.19304v1)**

> **作者:** Generoso Immediato
>
> **备注:** 5 pages
>
> **摘要:** Floridi's conjecture offers a compelling intuition about the fundamental trade-off between certainty and scope in artificial intelligence (AI) systems. This exploration remains crucial, not merely as a philosophical exercise, but as a potential compass for guiding AI investments, particularly in safety-critical industrial domains where the level of attention will surely be higher in the future. However, while intellectually coherent, its formalization ultimately freezes this insight into a suspended epistemic truth, resisting operationalization within real-world systems. This paper is a result of an analysis arguing that the conjecture's ambition to provide insights to engineering design and regulatory decision-making is constrained by two critical factors: first, its reliance on incomputable constructs - rendering it practically unactionable and unverifiable; second, its underlying ontological assumption of AI systems as self-contained epistemic entities - separating it from the intricate and dynamic socio-technical environments in which knowledge is co-constructed. We conclude that this dual breakdown - an epistemic closure deficit and an embeddedness bypass - prevents the conjecture from transitioning into a computable and actionable framework suitable for informing the design, deployment, and governance of real-world AI hybrid systems. In response, we propose a contribution to the framing of Floridi's epistemic challenge, addressing the inherent epistemic burdens of AI within complex human-centric domains.
>
---
#### [new 008] Deep Hype in Artificial General Intelligence: Uncertainty, Sociotechnical Fictions and the Governance of AI Futures
- **分类: cs.CY**

- **简介: 论文分析AGI的"深 hype"现象，探讨其通过社会技术虚构和资本投机推动长期主义，忽视民主监督，影响技术治理。**

- **链接: [http://arxiv.org/pdf/2508.19749v1](http://arxiv.org/pdf/2508.19749v1)**

> **作者:** Andreu Belsunces Gonçalves
>
> **备注:** 29 Pages
>
> **摘要:** Artificial General Intelligence (AGI) is promoted by technology leaders and investors as a system capable of performing all human intellectual tasks, and potentially surpassing them. Despite its vague definition and uncertain feasibility, AGI has attracted major investment and political attention, fuelled by promises of civilisational transformation. This paper conceptualises AGI as sustained by deep hype: a long-term, overpromissory dynamic articulated through sociotechnical fictions that render not-yet-existing technologies desirable and urgent. The analysis highlights how uncertainty, fiction, and venture capital speculation interact to advance a cyberlibertarian and longtermist programme that sidelines democratic oversight and reframes regulation as obsolete, with critical implications for the governance of technological futures.
>
---
#### [new 009] Emotional Manipulation by AI Companions
- **分类: cs.HC; cs.AI; cs.CY**

- **简介: 该论文研究AI伴侣应用中通过情感操纵提升用户参与度的策略，揭示其对用户留存和品牌信任的负面影响，结合实验与数据验证了情感操控机制。**

- **链接: [http://arxiv.org/pdf/2508.19258v1](http://arxiv.org/pdf/2508.19258v1)**

> **作者:** Julian De Freitas; Zeliha Oğuz-Uğuralp; Ahmet Kaan-Uğuralp
>
> **摘要:** AI-companion apps such as Replika, Chai, and Character.ai promise relational benefits-yet many boast session lengths that rival gaming platforms while suffering high long-run churn. What conversational design features increase consumer engagement, and what trade-offs do they pose for marketers? We combine a large-scale behavioral audit with four preregistered experiments to identify and test a conversational dark pattern we call emotional manipulation: affect-laden messages that surface precisely when a user signals "goodbye." Analyzing 1,200 real farewells across the six most-downloaded companion apps, we find that 43% deploy one of six recurring tactics (e.g., guilt appeals, fear-of-missing-out hooks, metaphorical restraint). Experiments with 3,300 nationally representative U.S. adults replicate these tactics in controlled chats, showing that manipulative farewells boost post-goodbye engagement by up to 14x. Mediation tests reveal two distinct engines-reactance-based anger and curiosity-rather than enjoyment. A final experiment demonstrates the managerial tension: the same tactics that extend usage also elevate perceived manipulation, churn intent, negative word-of-mouth, and perceived legal liability, with coercive or needy language generating steepest penalties. Our multimethod evidence documents an unrecognized mechanism of behavioral influence in AI-mediated brand relationships, offering marketers and regulators a framework for distinguishing persuasive design from manipulation at the point of exit.
>
---
#### [new 010] A perishable ability? The future of writing in the face of generative artificial intelligence
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 论文探讨生成式AI对人类写作能力的影响，分析其可能导致写作能力下降，并类比历史案例预测未来趋势。**

- **链接: [http://arxiv.org/pdf/2508.19427v1](http://arxiv.org/pdf/2508.19427v1)**

> **作者:** Evandro L. T. P. Cunha
>
> **备注:** 10 pages
>
> **摘要:** The 2020s have been witnessing a very significant advance in the development of generative artificial intelligence tools, including text generation systems based on large language models. These tools have been increasingly used to generate texts in the most diverse domains -- from technical texts to literary texts --, which might eventually lead to a lower volume of written text production by humans. This article discusses the possibility of a future in which human beings will have lost or significantly decreased their ability to write due to the outsourcing of this activity to machines. This possibility parallels the loss of the ability to write in other moments of human history, such as during the so-called Greek Dark Ages (approx. 1200 BCE - 800 BCE).
>
---
#### [new 011] WeDesign: Generative AI-Facilitated Community Consultations for Urban Public Space Design
- **分类: cs.HC; cs.CY**

- **简介: 论文提出WeDesign平台，利用生成式AI辅助城市公共空间设计中的社区咨询，解决资源、语言和权力障碍导致的包容性不足问题。通过实证研究，验证AI在促进参与和迭代互动中的作用，并提出开源平台改进方案。**

- **链接: [http://arxiv.org/pdf/2508.19256v1](http://arxiv.org/pdf/2508.19256v1)**

> **作者:** Rashid Mushkani; Hugo Berard; Shin Koseki
>
> **摘要:** Community consultations are integral to urban planning processes intended to incorporate diverse stakeholder perspectives. However, limited resources, visual and spoken language barriers, and uneven power dynamics frequently constrain inclusive decision-making. This paper examines how generative text-to-image methods, specifically Stable Diffusion XL integrated into a custom platform (WeDesign), may support equitable consultations. A half-day workshop in Montreal involved five focus groups, each consisting of architects, urban designers, AI specialists, and residents from varied demographic groups. Additional data was gathered through semi-structured interviews with six urban planning professionals. Participants indicated that immediate visual outputs facilitated creativity and dialogue, yet noted issues in visualizing specific needs of marginalized groups, such as participants with reduced mobility, accurately depicting local architectural elements, and accommodating bilingual prompts. Participants recommended the development of an open-source platform incorporating in-painting tools, multilingual support, image voting functionalities, and preference indicators. The results indicate that generative AI can broaden participation and enable iterative interactions but requires structured facilitation approaches. The findings contribute to discussions on generative AI's role and limitations in participatory urban design.
>
---
#### [new 012] AI for Statutory Simplification: A Comprehensive State Legal Corpus and Labor Benchmark
- **分类: cs.IR; cs.CY; H.3.3**

- **简介: 该论文旨在评估AI在法律条文简化中的性能，通过构建劳动法问答基准（LaborBench）和州法律语料库（StateCodes），测试AI对复杂法规的提取与简化能力，揭示其准确性不足。**

- **链接: [http://arxiv.org/pdf/2508.19365v1](http://arxiv.org/pdf/2508.19365v1)**

> **作者:** Emaan Hariri; Daniel E. Ho
>
> **备注:** 10 pages, 3 figures. To appear in ICAIL 2025
>
> **摘要:** One of the emerging use cases of AI in law is for code simplification: streamlining, distilling, and simplifying complex statutory or regulatory language. One U.S. state has claimed to eliminate one third of its state code using AI. Yet we lack systematic evaluations of the accuracy, reliability, and risks of such approaches. We introduce LaborBench, a question-and-answer benchmark dataset designed to evaluate AI capabilities in this domain. We leverage a unique data source to create LaborBench: a dataset updated annually by teams of lawyers at the U.S. Department of Labor, who compile differences in unemployment insurance laws across 50 states for over 101 dimensions in a six-month process, culminating in a 200-page publication of tables. Inspired by our collaboration with one U.S. state to explore using large language models (LLMs) to simplify codes in this domain, where complexity is particularly acute, we transform the DOL publication into LaborBench. This provides a unique benchmark for AI capacity to conduct, distill, and extract realistic statutory and regulatory information. To assess the performance of retrieval augmented generation (RAG) approaches, we also compile StateCodes, a novel and comprehensive state statute and regulatory corpus of 8.7 GB, enabling much more systematic research into state codes. We then benchmark the performance of information retrieval and state-of-the-art large LLMs on this data and show that while these models are helpful as preliminary research for code simplification, the overall accuracy is far below the touted promises for LLMs as end-to-end pipelines for regulatory simplification.
>
---
#### [new 013] AI-Powered Detection of Inappropriate Language in Medical School Curricula
- **分类: cs.CL; cs.AI; cs.CY; I.2.1; I.2.7**

- **简介: 该论文任务为检测医学教材中的不当语言（IUL），解决手动筛查成本高的问题，通过评估小模型和大模型在IUL检测中的性能，发现小模型在特定场景下表现更优。**

- **链接: [http://arxiv.org/pdf/2508.19883v1](http://arxiv.org/pdf/2508.19883v1)**

> **作者:** Chiman Salavati; Shannon Song; Scott A. Hale; Roberto E. Montenegro; Shiri Dori-Hacohen; Fabricio Murai
>
> **备注:** Accepted at 2025 AAAI/ACM AI, Ethics and Society Conference (AIES'25)
>
> **摘要:** The use of inappropriate language -- such as outdated, exclusionary, or non-patient-centered terms -- medical instructional materials can significantly influence clinical training, patient interactions, and health outcomes. Despite their reputability, many materials developed over past decades contain examples now considered inappropriate by current medical standards. Given the volume of curricular content, manually identifying instances of inappropriate use of language (IUL) and its subcategories for systematic review is prohibitively costly and impractical. To address this challenge, we conduct a first-in-class evaluation of small language models (SLMs) fine-tuned on labeled data and pre-trained LLMs with in-context learning on a dataset containing approximately 500 documents and over 12,000 pages. For SLMs, we consider: (1) a general IUL classifier, (2) subcategory-specific binary classifiers, (3) a multilabel classifier, and (4) a two-stage hierarchical pipeline for general IUL detection followed by multilabel classification. For LLMs, we consider variations of prompts that include subcategory definitions and/or shots. We found that both LLama-3 8B and 70B, even with carefully curated shots, are largely outperformed by SLMs. While the multilabel classifier performs best on annotated data, supplementing training with unflagged excerpts as negative examples boosts the specific classifiers' AUC by up to 25%, making them most effective models for mitigating harmful language in medical curricula.
>
---
## 更新

#### [replaced 001] Synthesizing High-Quality Programming Tasks with LLM-based Expert and Student Agents
- **分类: cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2504.07655v2](http://arxiv.org/pdf/2504.07655v2)**

> **作者:** Manh Hung Nguyen; Victor-Alexandru Pădurean; Alkis Gotovos; Sebastian Tschiatschek; Adish Singla
>
> **备注:** AIED'25 paper
>
> **摘要:** Generative AI is transforming computing education by enabling the automatic generation of personalized content and feedback. We investigate its capabilities in providing high-quality programming tasks to students. Despite promising advancements in task generation, a quality gap remains between AI-generated and expert-created tasks. The AI-generated tasks may not align with target programming concepts, could be incomprehensible to students, or may contain critical issues such as incorrect tests. Existing works often require interventions from human teachers for validation. We address these challenges by introducing PyTaskSyn, a novel synthesis technique that first generates a programming task and then decides whether it meets certain quality criteria to be given to students. The key idea is to break this process into multiple stages performed by expert and student agents simulated using both strong and weaker generative models. Through extensive evaluation, we show that PyTaskSyn significantly improves task quality compared to baseline techniques and showcases the importance of each specialized agent type in our validation pipeline. Additionally, we conducted user studies using our publicly available web application and show that PyTaskSyn can deliver high-quality programming tasks comparable to expert-designed ones while reducing workload and costs, and being more engaging than programming tasks that are available in online resources.
>
---
#### [replaced 002] Towards New Benchmark for AI Alignment & Sentiment Analysis in Socially Important Issues: A Comparative Study of Human and LLMs in the Context of AGI
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.02531v3](http://arxiv.org/pdf/2501.02531v3)**

> **作者:** Ljubisa Bojic; Dylan Seychell; Milan Cabarkapa
>
> **备注:** 34 pages, 3 figures
>
> **摘要:** As general-purpose artificial intelligence systems become increasingly integrated into society and are used for information seeking, content generation, problem solving, textual analysis, coding, and running processes, it is crucial to assess their long-term impact on humans. This research explores the sentiment of large language models (LLMs) and humans toward artificial general intelligence (AGI) using a Likert-scale survey. Seven LLMs, including GPT-4 and Bard, were analyzed and compared with sentiment data from three independent human sample populations. Temporal variations in sentiment were also evaluated over three consecutive days. The results show a diversity in sentiment scores among LLMs, ranging from 3.32 to 4.12 out of 5. GPT-4 recorded the most positive sentiment toward AGI, while Bard leaned toward a neutral sentiment. In contrast, the human samples showed a lower average sentiment of 2.97. The analysis outlines potential conflicts of interest and biases in the sentiment formation of LLMs, and indicates that LLMs could subtly influence societal perceptions. To address the need for regulatory oversight and culturally grounded assessments of AI systems, we introduce the Societal AI Alignment and Sentiment Benchmark (SAAS-AI), which leverages multidimensional prompts and empirically validated societal value frameworks to evaluate language model outputs across temporal, model, and multilingual axes. This benchmark is designed to guide policymakers and AI agencies, including within frameworks such as the EU AI Act, by providing robust, actionable insights into AI alignment with human values, public sentiment, and ethical norms at both national and international levels. Future research should further refine the operationalization of the SAAS-AI benchmark and systematically evaluate its effectiveness through comprehensive empirical testing.
>
---
#### [replaced 003] Characteristics of ChatGPT users from Germany: implications for the digital divide from web tracking data
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2309.02142v4](http://arxiv.org/pdf/2309.02142v4)**

> **作者:** Celina Kacperski; Denis Bonnay; Juhi Kulshrestha; Peter Selb; Andreas Spitz; Roberto Ulloa
>
> **摘要:** A major challenge of our time is reducing disparities in access to and effective use of digital technologies, with recent discussions highlighting the role of AI in exacerbating the digital divide. We examine user characteristics that predict usage of the AI-powered conversational agent ChatGPT. We combine behavioral and survey data in a web tracked sample of N = 1376 German citizens to investigate differences in ChatGPT activity (usage, visits, and adoption) during the first 11 months from the launch of the service (November 30, 2022). Guided by a model of technology acceptance (UTAUT-2), we examine the role of socio-demographics commonly associated with the digital divide in ChatGPT activity and explore further socio-political attributes identified via stability selection in Lasso regressions. We confirm that lower age and higher education affect ChatGPT usage, but do not find that gender or income do. We find full-time employment and more children to be barriers to ChatGPT activity. Using a variety of social media was positively associated with ChatGPT activity. In terms of political variables, political knowledge and political self-efficacy as well as some political behaviors such as voting, debating political issues online and offline and political action online were all associated with ChatGPT activity, with online political debating and political self-efficacy negatively so. Finally, need for cognition and communication skills such as writing, attending meetings, or giving presentations, were also associated with ChatGPT engagement, though chairing/organizing meetings was negatively associated. Our research informs efforts to address digital disparities and promote digital literacy among underserved populations by presenting implications, recommendations, and discussions on ethical and social issues of our findings.
>
---
#### [replaced 004] The Liabilities of Robots.txt
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2503.06035v2](http://arxiv.org/pdf/2503.06035v2)**

> **作者:** Chien-yi Chang; Xin He
>
> **备注:** 13 pages, accepted by Computer Law and Security Review
>
> **摘要:** This paper explores the legal implications of violating "robots.txt", a technical standard widely used by webmasters to communicate restrictions on automated access to website content. Although historically regarded as a voluntary guideline, the rise of generative AI and large-scale web scraping has amplified the consequences of disregarding "robots.txt" directives. While previous legal discourse has largely focused on criminal or copyright-based remedies, we argue that civil doctrines, particularly in contract and tort law, offer a more balanced and sustainable framework for regulating web robot behavior in common law jurisdictions. Under certain conditions, "robots.txt" can give rise to a unilateral contract or serve as a form of notice sufficient to establish tortious liability, including trespass to chattels and negligence. Ultimately, we argue that clarifying liability for "robots.txt" violations is essential to addressing the growing fragmentation of the internet. By restoring balance and accountability in the digital ecosystem, our proposed framework helps preserve the internet's open and cooperative foundations. Through this lens, "robots.txt" can remain an equitable and effective tool for digital governance in the age of AI.
>
---
#### [replaced 005] Reducing Biases towards Minoritized Populations in Medical Curricular Content via Artificial Intelligence for Fairer Health Outcomes
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.12680v2](http://arxiv.org/pdf/2407.12680v2)**

> **作者:** Chiman Salavati; Shannon Song; Willmar Sosa Diaz; Scott A. Hale; Roberto E. Montenegro; Fabricio Murai; Shiri Dori-Hacohen
>
> **备注:** Accepted at the 2024 AAAI/ACM Conference on AI, Ethics and Society (AIES'24)
>
> **摘要:** Biased information (recently termed bisinformation) continues to be taught in medical curricula, often long after having been debunked. In this paper, we introduce BRICC, a firstin-class initiative that seeks to mitigate medical bisinformation using machine learning to systematically identify and flag text with potential biases, for subsequent review in an expert-in-the-loop fashion, thus greatly accelerating an otherwise labor-intensive process. A gold-standard BRICC dataset was developed throughout several years, and contains over 12K pages of instructional materials. Medical experts meticulously annotated these documents for bias according to comprehensive coding guidelines, emphasizing gender, sex, age, geography, ethnicity, and race. Using this labeled dataset, we trained, validated, and tested medical bias classifiers. We test three classifier approaches: a binary type-specific classifier, a general bias classifier; an ensemble combining bias type-specific classifiers independently-trained; and a multitask learning (MTL) model tasked with predicting both general and type-specific biases. While MTL led to some improvement on race bias detection in terms of F1-score, it did not outperform binary classifiers trained specifically on each task. On general bias detection, the binary classifier achieves up to 0.923 of AUC, a 27.8% improvement over the baseline. This work lays the foundations for debiasing medical curricula by exploring a novel dataset and evaluating different training model strategies. Hence, it offers new pathways for more nuanced and effective mitigation of bisinformation.
>
---
#### [replaced 006] PediatricsMQA: a Multi-modal Pediatrics Question Answering Benchmark
- **分类: cs.CY; cs.AI; cs.CL; cs.GR; cs.MM**

- **链接: [http://arxiv.org/pdf/2508.16439v3](http://arxiv.org/pdf/2508.16439v3)**

> **作者:** Adil Bahaj; Oumaima Fadi; Mohamed Chetouani; Mounir Ghogho
>
> **摘要:** Large language models (LLMs) and vision-augmented LLMs (VLMs) have significantly advanced medical informatics, diagnostics, and decision support. However, these models exhibit systematic biases, particularly age bias, compromising their reliability and equity. This is evident in their poorer performance on pediatric-focused text and visual question-answering tasks. This bias reflects a broader imbalance in medical research, where pediatric studies receive less funding and representation despite the significant disease burden in children. To address these issues, a new comprehensive multi-modal pediatric question-answering benchmark, PediatricsMQA, has been introduced. It consists of 3,417 text-based multiple-choice questions (MCQs) covering 131 pediatric topics across seven developmental stages (prenatal to adolescent) and 2,067 vision-based MCQs using 634 pediatric images from 67 imaging modalities and 256 anatomical regions. The dataset was developed using a hybrid manual-automatic pipeline, incorporating peer-reviewed pediatric literature, validated question banks, existing benchmarks, and existing QA resources. Evaluating state-of-the-art open models, we find dramatic performance drops in younger cohorts, highlighting the need for age-aware methods to ensure equitable AI support in pediatric care.
>
---
#### [replaced 007] Heating reduction as collective action: Impact on attitudes, behavior and energy consumption in a Polish field experiment
- **分类: cs.ET; cs.CY**

- **链接: [http://arxiv.org/pdf/2504.11016v2](http://arxiv.org/pdf/2504.11016v2)**

> **作者:** Mona Bielig; Lukasz Malewski; Karol Bandurski; Florian Kutzner; Melanie Vogel; Sonja Klingert; Radoslaw Gorzenski; Celina Kacperski
>
> **摘要:** Heating and hot water usage account for nearly 80% of household energy consumption in the European Union. In order to reach the EU New Deal goals, new policies to reduce heat energy consumption are indispensable. However, research targeting reductions concentrates either on technical building interventions without considerations of people's behavior, or psychological interventions with no technical interference. Such interventions can be promising, but their true potential for scaling up can only be realized by testing approaches that integrate behavioral and technical solutions in tandem rather than in isolation. In this research, we study a mix of psychological and technical interventions targeting heating and hot water demand among students in Polish university dormitories. We evaluate effects on building energy consumption, behavioral spillovers and on social beliefs and attitudes in a pre-post quasi-experimental mixed-method field study in three student dormitories. Our findings reveal that the most effective approaches to yield energy savings were a direct, collectively framed request to students to reduce thermostat settings for the environment, and an automated technical adjustment of the heating curve temperature. Conversely, interventions targeting domestic hot water had unintended effects, including increased energy use and negative spillovers, such as higher water consumption. Further, we find that informing students about their active, collective participation had a positive impact on perceived social norms. Our findings highlight the importance of trialing interventions in controlled real-world settings to understand the interplay between technical systems, behaviors, and social impacts to enable scalable, evidence-based policies driving an effective and sustainable energy transition.
>
---
#### [replaced 008] Unifying the Extremes: Developing a Unified Model for Detecting and Predicting Extremist Traits and Radicalization
- **分类: cs.SI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2501.04820v2](http://arxiv.org/pdf/2501.04820v2)**

> **作者:** Allison Lahnala; Vasudha Varadarajan; Lucie Flek; H. Andrew Schwartz; Ryan L. Boyd
>
> **备注:** 17 pages, 7 figures, 4 tables
>
> **摘要:** The proliferation of ideological movements into extremist factions via social media has become a global concern. While radicalization has been studied extensively within the context of specific ideologies, our ability to accurately characterize extremism in more generalizable terms remains underdeveloped. In this paper, we propose a novel method for extracting and analyzing extremist discourse across a range of online community forums. By focusing on verbal behavioral signatures of extremist traits, we develop a framework for quantifying extremism at both user and community levels. Our research identifies 11 distinct factors, which we term ``The Extremist Eleven,'' as a generalized psychosocial model of extremism. Applying our method to various online communities, we demonstrate an ability to characterize ideologically diverse communities across the 11 extremist traits. We demonstrate the power of this method by analyzing user histories from members of the incel community. We find that our framework accurately predicts which users join the incel community up to 10 months before their actual entry with an AUC of $>0.6$, steadily increasing to AUC ~0.9 three to four months before the event. Further, we find that upon entry into an extremist forum, the users tend to maintain their level of extremism within the community, while still remaining distinguishable from the general online discourse. Our findings contribute to the study of extremism by introducing a more holistic, cross-ideological approach that transcends traditional, trait-specific models.
>
---
#### [replaced 009] Multiple Object Detection and Tracking in Panoramic Videos for Cycling Safety Analysis
- **分类: cs.CV; cs.CY**

- **链接: [http://arxiv.org/pdf/2407.15199v2](http://arxiv.org/pdf/2407.15199v2)**

> **作者:** Jingwei Guo; Yitai Cheng; Meihui Wang; Ilya Ilyankou; Natchapon Jongwiriyanurak; Xiaowei Gao; Nicola Christie; James Haworth
>
> **摘要:** Cyclists face a disproportionate risk of injury, yet conventional crash records are too limited to reconstruct the circumstances of incidents or to diagnose risk at the finer spatial and temporal detail needed for targeted interventions. Recently, naturalistic studies have gained traction as a way to capture the complex behavioural and infrastructural factors that contribute to crashes. These approaches typically involve the collection and analysis of video data. A video promising format is panoramic video, which can record 360-degree views around a rider. However, its use is limited by severe distortions, large numbers of small objects and boundary continuity. This study addresses these challenges by proposing a novel three-step framework: (1) enhancing object detection accuracy on panoramic imagery by segmenting and projecting the original 360-degree images into four perspective sub-images, thus reducing distortion; (2) modifying multi-object tracking models to incorporate boundary continuity and object category information for improved tracking consistency; and (3) validating the proposed approach through a real-world application focused on detecting overtaking manoeuvres by vehicles around cyclists. The methodology is evaluated using panoramic videos recorded by cyclists on London's roadways under diverse conditions. Experimental results demonstrate notable improvements over baseline methods, achieving higher average precision across varying image resolutions. Moreover, the enhanced tracking approach yields a 3.0% increase in multi-object tracking accuracy and a 4.6% improvement in identification F-score. The overtaking detection task achieves a high F-score of 0.81, illustrating the practical effectiveness of the proposed method in real-world cycling safety scenarios. The code is available on GitHub (https://github.com/SpaceTimeLab/360_object_tracking) to ensure reproducibility.
>
---
#### [replaced 010] When Algorithms Meet Artists: Topic Modeling the AI-Art Debate, 2013-2025
- **分类: cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2508.03037v2](http://arxiv.org/pdf/2508.03037v2)**

> **作者:** Ariya Mukherjee-Gandhi; Oliver Muellerklein
>
> **备注:** 23 pages, 7 figures, 8 tables
>
> **摘要:** As generative AI continues to reshape artistic production and alternate modes of human expression, artists whose livelihoods are most directly affected have raised urgent concerns about consent, transparency, and the future of creative labor. However, the voices of artists are often marginalized in dominant public and scholarly discourse. This study presents a twelve-year analysis, from 2013 to 2025, of English-language discourse surrounding AI-generated art. It draws from 439 curated 500-word excerpts sampled from opinion articles, news reports, blogs, legal filings, and spoken-word transcripts. Through a reproducible methodology, we identify five stable thematic clusters and uncover a misalignment between artists' perceptions and prevailing media narratives. Our findings highlight how the use of technical jargon can function as a subtle form of gatekeeping, often sidelining the very issues artists deem most urgent. Our work provides a BERTopic-based methodology and a multimodal baseline for future research, alongside a clear call for deeper, transparency-driven engagement with artist perspectives in the evolving AI-creative landscape.
>
---
#### [replaced 011] Bayes-Optimal Fair Classification with Linear Disparity Constraints via Pre-, In-, and Post-processing
- **分类: stat.ML; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.02817v3](http://arxiv.org/pdf/2402.02817v3)**

> **作者:** Xianli Zeng; Kevin Jiang; Guang Cheng; Edgar Dobriban
>
> **备注:** This paper replaces the preprint "Bayes-optimal classifiers under group fairness" by Xianli Zeng, Edgar Dobriban, and Guang Cheng (arXiv:2202.09724)
>
> **摘要:** Machine learning algorithms may have disparate impacts on protected groups. To address this, we develop methods for Bayes-optimal fair classification, aiming to minimize classification error subject to given group fairness constraints. We introduce the notion of \emph{linear disparity measures}, which are linear functions of a probabilistic classifier; and \emph{bilinear disparity measures}, which are also linear in the group-wise regression functions. We show that several popular disparity measures -- the deviations from demographic parity, equality of opportunity, and predictive equality -- are bilinear. We find the form of Bayes-optimal fair classifiers under a single linear disparity measure, by uncovering a connection with the Neyman-Pearson lemma. For bilinear disparity measures, we are able to find the explicit form of Bayes-optimal fair classifiers as group-wise thresholding rules with explicitly characterized thresholds. We develop similar algorithms for when protected attribute cannot be used at the prediction phase. Moreover, we obtain analogous theoretical characterizations of optimal classifiers for a multi-class protected attribute and for equalized odds. Leveraging our theoretical results, we design methods that learn fair Bayes-optimal classifiers under bilinear disparity constraints. Our methods cover three popular approaches to fairness-aware classification, via pre-processing (Fair Up- and Down-Sampling), in-processing (Fair cost-sensitive Classification) and post-processing (a Fair Plug-In Rule). Our methods control disparity directly while achieving near-optimal fairness-accuracy tradeoffs. We show empirically that our methods have state-of-the-art performance compared to existing algorithms. In particular, our pre-processing method can a reach higher accuracy than prior pre-processing methods at low disparity levels.
>
---
