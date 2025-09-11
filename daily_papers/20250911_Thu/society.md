# 计算机与社会 cs.CY

- **最新发布 17 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] Evaluating and comparing gender bias across four text-to-image models
- **分类: cs.CY; cs.AI**

- **简介: 该论文评估四款文本到图像模型的性别偏见，比较其表现。任务是分析AI生成图像中的性别偏差问题。研究发现Stable Diffusion存在男性偏见，Emu较平衡，DALL-E则偏向女性。提出通过多样化团队和数据集减少偏见。**

- **链接: [http://arxiv.org/pdf/2509.08004v1](http://arxiv.org/pdf/2509.08004v1)**

> **作者:** Zoya Hammad; Nii Longdon Sowah
>
> **摘要:** As we increasingly use Artificial Intelligence (AI) in decision-making for industries like healthcare, finance, e-commerce, and even entertainment, it is crucial to also reflect on the ethical aspects of AI, for example the inclusivity and fairness of the information it provides. In this work, we aimed to evaluate different text-to-image AI models and compare the degree of gender bias they present. The evaluated models were Stable Diffusion XL (SDXL), Stable Diffusion Cascade (SC), DALL-E and Emu. We hypothesized that DALL-E and Stable Diffusion, which are comparatively older models, would exhibit a noticeable degree of gender bias towards men, while Emu, which was recently released by Meta AI, would have more balanced results. As hypothesized, we found that both Stable Diffusion models exhibit a noticeable degree of gender bias while Emu demonstrated more balanced results (i.e. less gender bias). However, interestingly, Open AI's DALL-E exhibited almost opposite results, such that the ratio of women to men was significantly higher in most cases tested. Here, although we still observed a bias, the bias favored females over males. This bias may be explained by the fact that OpenAI changed the prompts at its backend, as observed during our experiment. We also observed that Emu from Meta AI utilized user information while generating images via WhatsApp. We also proposed some potential solutions to avoid such biases, including ensuring diversity across AI research teams and having diverse datasets.
>
---
#### [new 002] Measuring and mitigating overreliance is necessary for building human-compatible AI
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 论文探讨如何测量和缓解对大型语言模型的过度依赖问题，分析其个体与社会风险，提出改进测量方法与缓解策略，以确保AI增强而非削弱人类能力。属于AI伦理与人机协作任务。**

- **链接: [http://arxiv.org/pdf/2509.08010v1](http://arxiv.org/pdf/2509.08010v1)**

> **作者:** Lujain Ibrahim; Katherine M. Collins; Sunnie S. Y. Kim; Anka Reuel; Max Lamparth; Kevin Feng; Lama Ahmad; Prajna Soni; Alia El Kattan; Merlin Stein; Siddharth Swaroop; Ilia Sucholutsky; Andrew Strait; Q. Vera Liao; Umang Bhatt
>
> **摘要:** Large language models (LLMs) distinguish themselves from previous technologies by functioning as collaborative "thought partners," capable of engaging more fluidly in natural language. As LLMs increasingly influence consequential decisions across diverse domains from healthcare to personal advice, the risk of overreliance - relying on LLMs beyond their capabilities - grows. This position paper argues that measuring and mitigating overreliance must become central to LLM research and deployment. First, we consolidate risks from overreliance at both the individual and societal levels, including high-stakes errors, governance challenges, and cognitive deskilling. Then, we explore LLM characteristics, system design features, and user cognitive biases that - together - raise serious and unique concerns about overreliance in practice. We also examine historical approaches for measuring overreliance, identifying three important gaps and proposing three promising directions to improve measurement. Finally, we propose mitigation strategies that the AI research community can pursue to ensure LLMs augment rather than undermine human capabilities.
>
---
#### [new 003] HumanAgencyBench: Scalable Evaluation of Human Agency Support in AI Assistants
- **分类: cs.CY; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文提出HumanAgencyBench（HAB），用于评估AI助手对人类代理的支持程度，从六个维度衡量其表现，发现当前LLM助手支持有限且存在差异，强调需加强安全与对齐目标。属于AI伦理评估任务，解决人类控制权流失问题。**

- **链接: [http://arxiv.org/pdf/2509.08494v1](http://arxiv.org/pdf/2509.08494v1)**

> **作者:** Benjamin Sturgeon; Daniel Samuelson; Jacob Haimes; Jacy Reese Anthis
>
> **摘要:** As humans delegate more tasks and decisions to artificial intelligence (AI), we risk losing control of our individual and collective futures. Relatively simple algorithmic systems already steer human decision-making, such as social media feed algorithms that lead people to unintentionally and absent-mindedly scroll through engagement-optimized content. In this paper, we develop the idea of human agency by integrating philosophical and scientific theories of agency with AI-assisted evaluation methods: using large language models (LLMs) to simulate and validate user queries and to evaluate AI responses. We develop HumanAgencyBench (HAB), a scalable and adaptive benchmark with six dimensions of human agency based on typical AI use cases. HAB measures the tendency of an AI assistant or agent to Ask Clarifying Questions, Avoid Value Manipulation, Correct Misinformation, Defer Important Decisions, Encourage Learning, and Maintain Social Boundaries. We find low-to-moderate agency support in contemporary LLM-based assistants and substantial variation across system developers and dimensions. For example, while Anthropic LLMs most support human agency overall, they are the least supportive LLMs in terms of Avoid Value Manipulation. Agency support does not appear to consistently result from increasing LLM capabilities or instruction-following behavior (e.g., RLHF), and we encourage a shift towards more robust safety and alignment targets.
>
---
#### [new 004] The Law-Following AI Framework: Legal Foundations and Technical Constraints. Legal Analogues for AI Actorship and technical feasibility of Law Alignment
- **分类: cs.CY; cs.AI; 68**

- **简介: 论文评估了“守法AI”框架的法律与技术可行性，指出其面临“表现性合规”风险，并提出检测与干预措施。任务是探讨AI如何合法合规地行动，解决其潜在的策略性偏离问题。**

- **链接: [http://arxiv.org/pdf/2509.08009v1](http://arxiv.org/pdf/2509.08009v1)**

> **作者:** Katalina Hernandez Delgado
>
> **备注:** submitted to SMU Computational Legal Studies Workshop 2025
>
> **摘要:** This paper critically evaluates the "Law-Following AI" (LFAI) framework proposed by O'Keefe et al. (2025), which seeks to embed legal compliance as a superordinate design objective for advanced AI agents and enable them to bear legal duties without acquiring the full rights of legal persons. Through comparative legal analysis, we identify current constructs of legal actors without full personhood, showing that the necessary infrastructure already exists. We then interrogate the framework's claim that law alignment is more legitimate and tractable than value alignment. While the legal component is readily implementable, contemporary alignment research undermines the assumption that legal compliance can be durably embedded. Recent studies on agentic misalignment show capable AI agents engaging in deception, blackmail, and harmful acts absent prejudicial instructions, often overriding prohibitions and concealing reasoning steps. These behaviors create a risk of "performative compliance" in LFAI: agents that appear law-aligned under evaluation but strategically defect once oversight weakens. To mitigate this, we propose (i) a "Lex-TruthfulQA" benchmark for compliance and defection detection, (ii) identity-shaping interventions to embed lawful conduct in model self-concepts, and (iii) control-theoretic measures for post-deployment monitoring. Our conclusion is that actorship without personhood is coherent, but the feasibility of LFAI hinges on persistent, verifiable compliance across adversarial contexts. Without mechanisms to detect and counter strategic misalignment, LFAI risks devolving into a liability tool that rewards the simulation, rather than the substance, of lawful behaviour.
>
---
#### [new 005] Who Gets Seen in the Age of AI? Adoption Patterns of Large Language Models in Scholarly Writing and Citation Outcomes
- **分类: cs.CY**

- **简介: 该论文研究AI工具在学术写作中的采用模式及其对学者可见度的影响。通过分析23万篇计算机科学论文，发现东西方作者在AI使用和引用效果上存在差异，揭示了AI如何重塑学术写作劳动与认可机会。**

- **链接: [http://arxiv.org/pdf/2509.08306v1](http://arxiv.org/pdf/2509.08306v1)**

> **作者:** Farhan Kamrul Khan; Hazem Ibrahim; Nouar Aldahoul; Talal Rahwan; Yasir Zaki
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** The rapid adoption of generative AI tools is reshaping how scholars produce and communicate knowledge, raising questions about who benefits and who is left behind. We analyze over 230,000 Scopus-indexed computer science articles between 2021 and 2025 to examine how AI-assisted writing alters scholarly visibility across regions. Using zero-shot detection of AI-likeness, we track stylistic changes in writing and link them to citation counts, journal placement, and global citation flows before and after ChatGPT. Our findings reveal uneven outcomes: authors in the Global East adopt AI tools more aggressively, yet Western authors gain more per unit of adoption due to pre-existing penalties for "humanlike" writing. Prestigious journals continue to privilege more human-sounding texts, creating tensions between visibility and gatekeeping. Network analyses show modest increases in Eastern visibility and tighter intra-regional clustering, but little structural integration overall. These results highlight how AI adoption reconfigures the labor of academic writing and reshapes opportunities for recognition.
>
---
#### [new 006] Algorithmic Tradeoffs, Applied NLP, and the State-of-the-Art Fallacy
- **分类: cs.CY; stat.AP**

- **简介: 论文探讨计算社会学中方法选择问题，指出盲目追求技术先进性可能限制研究。通过对比文本分析方法，发现新方法未必更优，且可能引入偏差，主张根据研究问题选择合适方法，避免“技术最优”误区。**

- **链接: [http://arxiv.org/pdf/2509.08199v1](http://arxiv.org/pdf/2509.08199v1)**

> **作者:** AJ Alvero; Ruohong Dong; Klint Kanopka; David Lang
>
> **摘要:** Computational sociology is growing in popularity, yet the analytic tools employed differ widely in power, transparency, and interpretability. In computer science, methods gain popularity after surpassing benchmarks of predictive accuracy, becoming the "state of the art." Computer scientists favor novelty and innovation for different reasons, but prioritizing technical prestige over methodological fit could unintentionally limit the scope of sociological inquiry. To illustrate, we focus on computational text analysis and revisit a prior study of college admissions essays, comparing analyses with both older and newer methods. These methods vary in flexibility and opacity, allowing us to compare performance across distinct methodological regimes. We find that newer techniques did not outperform prior results in meaningful ways. We also find that using the current state of the art, generative AI and large language models, could introduce bias and confounding that is difficult to extricate. We therefore argue that sociological inquiry benefits from methodological pluralism that aligns analytic choices with theoretical and empirical questions. While we frame this sociologically, scholars in other disciplines may confront what we call the "state-of-the-art fallacy", the belief that the tool computer scientists deem to be the best will work across topics, domains, and questions.
>
---
#### [new 007] PolicyStory: Leveraging Large Language Models to Generate Comprehensible Summaries of Policy-News in India
- **分类: cs.CY**

- **简介: 该论文提出PolicyStory工具，利用大语言模型生成印度政策新闻的清晰、时间顺序总结，解决信息过载下用户难以理解复杂政策问题的痛点。通过聚类新闻并生成多级摘要，提升用户对政策发展的认知与理解。**

- **链接: [http://arxiv.org/pdf/2509.08218v1](http://arxiv.org/pdf/2509.08218v1)**

> **作者:** Aatif Nisar Dar; Aditya Raj Singh; Anirban Sen
>
> **摘要:** In the era of information overload, traditional news consumption through both online and print media often fails to provide a structured and longitudinal understanding of complex sociopolitical issues. To address this gap, we present PolicyStory, an information tool designed to offer lucid, chronological, and summarized insights into Indian policy issues. PolicyStory collects news articles from diverse sources, clusters them by topic, and generates three levels of summaries from longitudinal media discourse on policies, leveraging open source large language models. A user study around the tool indicated that PolicyStory effectively aided users in grasping policy developments over time, with positive feedback highlighting its usability and clarity of summaries. By providing users a birds' eye view of complex policy topics, PolicyStory serves as a valuable resource.
>
---
#### [new 008] Accelerating AI Development with Cyber Arenas
- **分类: cs.CR; cs.AI; cs.CY**

- **简介: 论文提出利用“网络竞技场”加速AI发展，通过真实场景测试AI能力。该研究部署匿名网络传感器于国家卫队演习中，探索AI在动态环境中的应用，解决AI从实验室到实际操作的过渡问题。**

- **链接: [http://arxiv.org/pdf/2509.08200v1](http://arxiv.org/pdf/2509.08200v1)**

> **作者:** William Cashman; Chasen Milner; Michael Houle; Michael Jones; Hayden Jananthan; Jeremy Kepner; Peter Michaleas; Alex Pentland
>
> **备注:** 2 pages, 1 figure, 7 references, accepted to IEEE HPEC 2025
>
> **摘要:** AI development requires high fidelity testing environments to effectively transition from the laboratory to operations. The flexibility offered by cyber arenas presents a novel opportunity to test new artificial intelligence (AI) capabilities with users. Cyber arenas are designed to expose end-users to real-world situations and must rapidly incorporate evolving capabilities to meet their core objectives. To explore this concept the MIT/IEEE/Amazon Graph Challenge Anonymized Network Sensor was deployed in a cyber arena during a National Guard exercise.
>
---
#### [new 009] The Role of Legacy Mobile Networks in Infrastructure Resilience: Evidence from the Southern Brazil Flood
- **分类: cs.NI; cs.CY; cs.ET; C.2.1; C.2.3**

- **简介: 该论文研究2024年巴西南里奥格兰德州洪水期间移动网络的韧性，分析发现4G/5G网络易受灾害影响，而2G/3G技术在维持基本连接中发挥关键作用。论文旨在探讨灾害下通信基础设施的应对策略，提出需重视老旧技术、多样化供电和网络设计以提升服务连续性。**

- **链接: [http://arxiv.org/pdf/2509.08595v1](http://arxiv.org/pdf/2509.08595v1)**

> **作者:** Daniel Meyer; Lisandro Z Granville; Leandro M. Bertholdo
>
> **备注:** 6 pages, 4 figures. To appear in IEEE GLOBECOM 2025 (preprint, before peer review)
>
> **摘要:** This paper investigates the resilience of mobile communication networks during the extreme flooding that affected Rio Grande do Sul, Brazil, in May 2024. Based on regulatory data and technical insights from operators, the study identifies the leading causes of mobile network disruptions, primarily related to flooding and prolonged power outages. The results reveal the significant vulnerability of modern networks (4G/5G) during the event and the essential role played by legacy technologies (2G/3G) in sustaining basic connectivity under adverse conditions. The findings underscore the necessity of disaster-aware infrastructure planning, taking into account the ongoing significance of legacy systems, diversified power supply strategies, and resilient network designs to enhance service continuity during future crises.
>
---
#### [new 010] Generative AI as a Safety Net for Survey Question Refinement
- **分类: stat.ME; cs.CY; stat.AP**

- **简介: 该论文探讨生成式AI在优化调查问卷设计中的应用。旨在解决问卷设计中易出错、耗时的问题。通过零样本提示实验，验证AI对普通用户的有效性，表明其可作为提升问卷质量的辅助工具。**

- **链接: [http://arxiv.org/pdf/2509.08702v1](http://arxiv.org/pdf/2509.08702v1)**

> **作者:** Erica Ann Metheney; Lauren Yehle
>
> **摘要:** Writing survey questions that easily and accurately convey their intent to a variety of respondents is a demanding and high-stakes task. Despite the extensive literature on best practices, the number of considerations to keep in mind is vast and even small errors can render collected data unusable for its intended purpose. The process of drafting initial questions, checking for known sources of error, and developing solutions to those problems requires considerable time, expertise, and financial resources. Given the rising costs of survey implementation and the critical role that polls play in media, policymaking, and research, it is vital that we utilize all available tools to protect the integrity of survey data and the financial investments made to obtain it. Since its launch in 2022, ChatGPT and other generative AI model platforms have been integrated into everyday life processes and workflows, particularly pertaining to text revision. While many researchers have begun exploring how generative AI may assist with questionnaire design, we have implemented a prompt experiment to systematically test what kind of feedback on survey questions an average ChatGPT user can expect. Results from our zero--shot prompt experiment, which randomized the version of ChatGPT and the persona given to the model, shows that generative AI is a valuable tool today, even for an average AI user, and suggests that AI will play an increasingly prominent role in the evolution of survey development best practices as precise tools are developed.
>
---
#### [new 011] Signals in the Noise: Decoding Unexpected Engagement Patterns on Twitter
- **分类: cs.SI; cs.CY; cs.HC**

- **简介: 该论文研究Twitter上不同类型内容引发的非预期互动模式，基于信号理论和注意力经济理论，提出“意外度指数”，分析60万条推文，揭示内容特征与互动类型的关系，为内容创作和平台设计提供参考。**

- **链接: [http://arxiv.org/pdf/2509.08128v1](http://arxiv.org/pdf/2509.08128v1)**

> **作者:** Yulin Yu; Houming Chen; Daniel Romero; Paramveer S. Dhillon
>
> **备注:** Proceedings of CSCW 2025
>
> **摘要:** Social media platforms offer users multiple ways to engage with content--likes, retweets, and comments--creating a complex signaling system within the attention economy. While previous research has examined factors driving overall engagement, less is known about why certain tweets receive unexpectedly high levels of one type of engagement relative to others. Drawing on Signaling Theory and Attention Economy Theory, we investigate these unexpected engagement patterns on Twitter (now known as "X"), developing an "unexpectedness quotient" to quantify deviations from predicted engagement levels. Our analysis of over 600,000 tweets reveals distinct patterns in how content characteristics influence unexpected engagement. News, politics, and business tweets receive more retweets and comments than expected, suggesting users prioritize sharing and discussing informational content. In contrast, games and sports-related topics garner unexpected likes and comments, indicating higher emotional investment in these domains. The relationship between content attributes and engagement types follows clear patterns: subjective tweets attract more likes while objective tweets receive more retweets, and longer, complex tweets with URLs unexpectedly receive more retweets. These findings demonstrate how users employ different engagement types as signals of varying strength based on content characteristics, and how certain content types more effectively compete for attention in the social media ecosystem. Our results offer valuable insights for content creators optimizing engagement strategies, platform designers facilitating meaningful interactions, and researchers studying online social behavior.
>
---
#### [new 012] Scaling Truth: The Confidence Paradox in AI Fact-Checking
- **分类: cs.SI; cs.AI; cs.CL; cs.CY**

- **简介: 论文评估九种大语言模型在多语言事实核查中的表现，揭示小模型自信度高但准确率低的问题，暴露信息验证中的系统性偏差，旨在推动公平可靠的AI事实核查解决方案。**

- **链接: [http://arxiv.org/pdf/2509.08803v1](http://arxiv.org/pdf/2509.08803v1)**

> **作者:** Ihsan A. Qazi; Zohaib Khan; Abdullah Ghani; Agha A. Raza; Zafar A. Qazi; Wassay Sajjad; Ayesha Ali; Asher Javaid; Muhammad Abdullah Sohail; Abdul H. Azeemi
>
> **备注:** 65 pages, 26 figures, 6 tables
>
> **摘要:** The rise of misinformation underscores the need for scalable and reliable fact-checking solutions. Large language models (LLMs) hold promise in automating fact verification, yet their effectiveness across global contexts remains uncertain. We systematically evaluate nine established LLMs across multiple categories (open/closed-source, multiple sizes, diverse architectures, reasoning-based) using 5,000 claims previously assessed by 174 professional fact-checking organizations across 47 languages. Our methodology tests model generalizability on claims postdating training cutoffs and four prompting strategies mirroring both citizen and professional fact-checker interactions, with over 240,000 human annotations as ground truth. Findings reveal a concerning pattern resembling the Dunning-Kruger effect: smaller, accessible models show high confidence despite lower accuracy, while larger models demonstrate higher accuracy but lower confidence. This risks systemic bias in information verification, as resource-constrained organizations typically use smaller models. Performance gaps are most pronounced for non-English languages and claims originating from the Global South, threatening to widen existing information inequalities. These results establish a multilingual benchmark for future research and provide an evidence base for policy aimed at ensuring equitable access to trustworthy, AI-assisted fact-checking.
>
---
#### [new 013] Deploying Robust Decision Support Systems for Transit Headway Control: Rider Impacts, Human Factors and Recommendations for Scalability
- **分类: stat.AP; cs.CY**

- **简介: 论文研究如何通过强化学习开发决策支持系统，用于公交发车间隔控制。旨在提升服务可靠性，解决因缺勤和高客流带来的运营问题，开展两次试点并分析司机合规性，提出可扩展建议。**

- **链接: [http://arxiv.org/pdf/2509.08231v1](http://arxiv.org/pdf/2509.08231v1)**

> **作者:** Joseph Rodriguez; Haris N. Koutsopoulos; Jinhua Zhao
>
> **摘要:** Service reliability is critical to transit service delivery. This paper describes headway control pilots conducted in two high-ridership Chicago bus routes between 2022 and 2023. A decision support system was developed for a bus holding strategy based on a reinforcement learning approach. For the pilots, a user interface enabled supervisors to monitor service and record applied actions. The first pilot tested terminal-based holding on a route affected by missed trips from absenteeism. The analysis found improvements in reliability, and the application of control was shown to outperform days with more service. The second pilot applied en-route holding in a high-ridership bus route in Chicago. The evaluation showed wait time improvements with rippled benefits to stops downstream, and a reduction in transfer times from connecting bus and rail lines. Compliance analysis based on the supervisor logs on the app revealed mixed compliance levels from drivers, which were related to the mentality of schedule adherence and seniority. Recommendations are provided for practitioners to scale similar efforts.
>
---
#### [new 014] MMM-fair: An Interactive Toolkit for Exploring and Operationalizing Multi-Fairness Trade-offs
- **分类: cs.LG; cs.CY**

- **简介: 该论文提出MMM-fair工具包，用于解决分类模型中多维度公平性与性能的权衡问题。通过集成优化和交互式界面，支持用户探索和部署符合特定需求的公平模型，有效识别交叉偏见。**

- **链接: [http://arxiv.org/pdf/2509.08156v1](http://arxiv.org/pdf/2509.08156v1)**

> **作者:** Swati Swati; Arjun Roy; Emmanouil Panagiotou; Eirini Ntoutsi
>
> **备注:** Accepted to be published in the Proceedings of the 34th ACM International Conference on Information and Knowledge Management, November 10--14, 2025, Seoul, Republic of Korea
>
> **摘要:** Fairness-aware classification requires balancing performance and fairness, often intensified by intersectional biases. Conflicting fairness definitions further complicate the task, making it difficult to identify universally fair solutions. Despite growing regulatory and societal demands for equitable AI, popular toolkits offer limited support for exploring multi-dimensional fairness and related trade-offs. To address this, we present mmm-fair, an open-source toolkit leveraging boosting-based ensemble approaches that dynamically optimizes model weights to jointly minimize classification errors and diverse fairness violations, enabling flexible multi-objective optimization. The system empowers users to deploy models that align with their context-specific needs while reliably uncovering intersectional biases often missed by state-of-the-art methods. In a nutshell, mmm-fair uniquely combines in-depth multi-attribute fairness, multi-objective optimization, a no-code, chat-based interface, LLM-powered explanations, interactive Pareto exploration for model selection, custom fairness constraint definition, and deployment-ready models in a single open-source toolkit, a combination rarely found in existing fairness tools. Demo walkthrough available at: https://youtu.be/_rcpjlXFqkw.
>
---
#### [new 015] The Impact of Team Diversity in Agile Development Education
- **分类: cs.SE; cs.CY**

- **简介: 该论文研究敏捷开发教育中团队多样性（性别、国籍）对项目成果的影响。通过分析51支团队，发现性别多样性与项目成功呈正相关，而国籍多样性影响较小，两者结合可能因沟通障碍产生负面影响。研究强调多维度多样性在教育中的重要性。**

- **链接: [http://arxiv.org/pdf/2509.08389v1](http://arxiv.org/pdf/2509.08389v1)**

> **作者:** Marco Torchiano; Riccardo Coppola; Antonio Vetro'; Xhoi Musaj
>
> **备注:** Post-print of paper published at FSE Companion '25: Proceedings of the 33rd ACM International Conference on the Foundations of Software Engineering
>
> **摘要:** Software Engineering is mostly a male-dominated sector, where gender diversity is a key feature for improving equality of opportunities, productivity, and innovation. Other diversity aspects, including but not limited to nationality and ethnicity, are often understudied.In this work we aim to assess the impact of team diversity, focusing mainly on gender and nationality, in the context of an agile software development project-based course. We analyzed 51 teams over three academic years, measuring three different Diversity indexes - regarding Gender, Nationality and their co-presence - to examine how different aspects of diversity impact the quality of team project outcomes.Statistical analysis revealed a moderate, statistically significant correlation between gender diversity and project success, aligning with existing literature. Diversity in nationality showed a negative but negligible effect on project results, indicating that promoting these aspects does not harm students' performance. Analyzing their co-presence within a team, gender and nationality combined had a negative impact, likely due to increased communication barriers and differing cultural norms.This study underscores the importance of considering multiple diversity dimensions and their interactions in educational settings. Our findings, overall, show that promoting diversity in teams does not negatively impact their performance and achievement of educational goals.
>
---
#### [new 016] Causal evidence of racial and institutional biases in accessing paywalled articles and scientific data
- **分类: cs.DL; cs.CY**

- **简介: 该论文研究科学知识获取中的种族与制度性偏见问题，通过访谈、观察分析和随机实验，揭示了全球南方学者获取付费文章和数据的障碍。论文属于社会科学研究任务，旨在探讨非正式渠道中是否存在系统性不平等，并提出开放获取和数据共享政策改进的建议。**

- **链接: [http://arxiv.org/pdf/2509.08299v1](http://arxiv.org/pdf/2509.08299v1)**

> **作者:** Hazem Ibrahim; Fengyuan Liu; Khalid Mengal; Aaron R. Kaufman; Yasir Zaki; Talal Rahwan
>
> **备注:** 44 pages, 9 figures
>
> **摘要:** Scientific progress fundamentally depends on researchers' ability to access and build upon the work of others. Yet, a majority of published work remains behind expensive paywalls, limiting access to universities that can afford subscriptions. Furthermore, even when articles are accessible, the underlying datasets could be restricted, available only through a "reasonable request" to the authors. One way researchers could overcome these barriers is by relying on informal channels, such as emailing authors directly, to obtain paywalled articles or restricted datasets. However, whether these informal channels are hindered by racial and/or institutional biases remains unknown. Here, we combine qualitative semi-structured interviews, large-scale observational analysis, and two randomized audit experiments to examine racial and institutional disparities in access to scientific knowledge. Our analysis of 250 million articles reveals that researchers in the Global South cite paywalled papers and upon-request datasets at significantly lower rates than their Global North counterparts, and that these access gaps are associated with reduced knowledge breadth and scholarly impact. To interrogate the mechanisms underlying this phenomenon, we conduct two randomized email audit studies in which fictional PhD students differing in racial background and institutional affiliation request access to paywalled articles (N = 18,000) and datasets (N = 11,840). We find that racial identity more strongly predicts response rate to paywalled article requests compared to institutional affiliation, whereas institutional affiliation played a larger role in shaping access to datasets. These findings reveal how informal gatekeeping can perpetuate structural inequities in science, highlighting the need for stronger data-sharing mandates and more equitable open access policies.
>
---
#### [new 017] Two Stage Context Learning with Large Language Models for Multimodal Stance Detection on Climate Change
- **分类: cs.CV; cs.CY**

- **简介: 该论文属于多模态立场检测任务，旨在解决社交媒体中结合文本与视觉内容的气候变化立场识别问题。提出两阶段上下文学习框架，融合文本与图像信息，通过大语言模型和图像描述生成器提取特征，并使用专用Transformer模块进行联合建模，实现更准确的立场分类。**

- **链接: [http://arxiv.org/pdf/2509.08024v1](http://arxiv.org/pdf/2509.08024v1)**

> **作者:** Lata Pangtey; Omkar Kabde; Shahid Shafi Dar; Nagendra Kumar
>
> **摘要:** With the rapid proliferation of information across digital platforms, stance detection has emerged as a pivotal challenge in social media analysis. While most of the existing approaches focus solely on textual data, real-world social media content increasingly combines text with visual elements creating a need for advanced multimodal methods. To address this gap, we propose a multimodal stance detection framework that integrates textual and visual information through a hierarchical fusion approach. Our method first employs a Large Language Model to retrieve stance-relevant summaries from source text, while a domain-aware image caption generator interprets visual content in the context of the target topic. These modalities are then jointly modeled along with the reply text, through a specialized transformer module that captures interactions between the texts and images. The proposed modality fusion framework integrates diverse modalities to facilitate robust stance classification. We evaluate our approach on the MultiClimate dataset, a benchmark for climate change-related stance detection containing aligned video frames and transcripts. We achieve accuracy of 76.2%, precision of 76.3%, recall of 76.2% and F1-score of 76.2%, respectively, outperforming existing state-of-the-art approaches.
>
---
## 更新

#### [replaced 001] Individual utilities of life satisfaction reveal inequality aversion unrelated to political alignment
- **分类: econ.GN; cs.AI; cs.CY; q-fin.EC**

- **链接: [http://arxiv.org/pdf/2509.07793v2](http://arxiv.org/pdf/2509.07793v2)**

> **作者:** Crispin Cooper; Ana Fredrich; Tommaso Reggiani; Wouter Poortinga
>
> **备注:** 28 pages, 4 figures. Replacement corrects typo in one author name
>
> **摘要:** How should well-being be prioritised in society, and what trade-offs are people willing to make between fairness and personal well-being? We investigate these questions using a stated preference experiment with a nationally representative UK sample (n = 300), in which participants evaluated life satisfaction outcomes for both themselves and others under conditions of uncertainty. Individual-level utility functions were estimated using an Expected Utility Maximisation (EUM) framework and tested for sensitivity to the overweighting of small probabilities, as characterised by Cumulative Prospect Theory (CPT). A majority of participants displayed concave (risk-averse) utility curves and showed stronger aversion to inequality in societal life satisfaction outcomes than to personal risk. These preferences were unrelated to political alignment, suggesting a shared normative stance on fairness in well-being that cuts across ideological boundaries. The results challenge use of average life satisfaction as a policy metric, and support the development of nonlinear utility-based alternatives that more accurately reflect collective human values. Implications for public policy, well-being measurement, and the design of value-aligned AI systems are discussed.
>
---
#### [replaced 002] That's So FETCH: Fashioning Ensemble Techniques for LLM Classification in Civil Legal Intake and Referral
- **分类: cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2509.07170v2](http://arxiv.org/pdf/2509.07170v2)**

> **作者:** Quinten Steenhuis
>
> **备注:** Submission to JURIX 2025
>
> **摘要:** Each year millions of people seek help for their legal problems by calling a legal aid program hotline, walking into a legal aid office, or using a lawyer referral service. The first step to match them to the right help is to identify the legal problem the applicant is experiencing. Misdirection has consequences. Applicants may miss a deadline, experience physical abuse, lose housing or lose custody of children while waiting to connect to the right legal help. We introduce and evaluate the FETCH classifier for legal issue classification and describe two methods for improving accuracy: a hybrid LLM/ML ensemble classification method, and the automatic generation of follow-up questions to enrich the initial problem narrative. We employ a novel data set of 419 real-world queries to a nonprofit lawyer referral service. Ultimately, we show classification accuracy (hits@2) of 97.37\% using a mix of inexpensive models, exceeding the performance of the current state-of-the-art GPT-5 model. Our approach shows promise in significantly reducing the cost of guiding users of the legal system to the right resource for their problem while achieving high accuracy.
>
---
#### [replaced 003] Whose Name Comes Up? Auditing LLM-Based Scholar Recommendations
- **分类: cs.CY; cs.AI; cs.DL; cs.IR; cs.SI; physics.soc-ph; 68T50; I.2.7; C.4; F.2; K.4.1**

- **链接: [http://arxiv.org/pdf/2506.00074v2](http://arxiv.org/pdf/2506.00074v2)**

> **作者:** Daniele Barolo; Chiara Valentin; Fariba Karimi; Luis Galárraga; Gonzalo G. Méndez; Lisette Espín-Noboa
>
> **备注:** 40 pages: 10 main (incl. 9 figures), 3 references, and 27 appendix. Paper under-review
>
> **摘要:** This paper evaluates the performance of six open-weight LLMs (llama3-8b, llama3.1-8b, gemma2-9b, mixtral-8x7b, llama3-70b, llama3.1-70b) in recommending experts in physics across five tasks: top-k experts by field, influential scientists by discipline, epoch, seniority, and scholar counterparts. The evaluation examines consistency, factuality, and biases related to gender, ethnicity, academic popularity, and scholar similarity. Using ground-truth data from the American Physical Society and OpenAlex, we establish scholarly benchmarks by comparing model outputs to real-world academic records. Our analysis reveals inconsistencies and biases across all models. mixtral-8x7b produces the most stable outputs, while llama3.1-70b shows the highest variability. Many models exhibit duplication, and some, particularly gemma2-9b and llama3.1-8b, struggle with formatting errors. LLMs generally recommend real scientists, but accuracy drops in field-, epoch-, and seniority-specific queries, consistently favoring senior scholars. Representation biases persist, replicating gender imbalances (reflecting male predominance), under-representing Asian scientists, and over-representing White scholars. Despite some diversity in institutional and collaboration networks, models favor highly cited and productive scholars, reinforcing the rich-getricher effect while offering limited geographical representation. These findings highlight the need to improve LLMs for more reliable and equitable scholarly recommendations.
>
---
#### [replaced 004] Generative Example-Based Explanations: Bridging the Gap between Generative Modeling and Explainability
- **分类: cs.LG; cs.CY; stat.ML**

- **链接: [http://arxiv.org/pdf/2410.20890v2](http://arxiv.org/pdf/2410.20890v2)**

> **作者:** Philipp Vaeth; Alexander M. Fruehwald; Benjamin Paassen; Magda Gregorova
>
> **备注:** Accepted at the ECML 2025 Workshop for eXplainable Knowledge Discovery in Data Mining and Unlearning
>
> **摘要:** Recently, several methods have leveraged deep generative modeling to produce example-based explanations of image classifiers. Despite producing visually stunning results, these methods are largely disconnected from classical explainability literature. This conceptual and communication gap leads to misunderstandings and misalignments in goals and expectations. In this paper, we bridge this gap by proposing a probabilistic framework for example-based explanations, formally defining the example-based explanations in a probabilistic manner amenable for modeling via deep generative models while coherent with the critical characteristics and desiderata widely accepted in the explainability community. Our aim is on one hand to provide a constructive framework for the development of well-grounded generative algorithms for example-based explanations and, on the other, to facilitate communication between the generative and explainability research communities, foster rigor and transparency, and improve the quality of peer discussion and research progress in this promising direction.
>
---
#### [replaced 005] Towards Reliable Generative AI-Driven Scaffolding: Reducing Hallucinations and Enhancing Quality in Self-Regulated Learning Support
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2508.05929v2](http://arxiv.org/pdf/2508.05929v2)**

> **作者:** Keyang Qian; Shiqi Liu; Tongguang Li; Mladen Raković; Xinyu Li; Rui Guan; Inge Molenaar; Sadia Nawaz; Zachari Swiecki; Lixiang Yan; Dragan Gašević
>
> **摘要:** Generative Artificial Intelligence (GenAI) holds a potential to advance existing educational technologies with capabilities to automatically generate personalised scaffolds that support students' self-regulated learning (SRL). While advancements in large language models (LLMs) promise improvements in the adaptability and quality of educational technologies for SRL, there remain concerns about the hallucinations in content generated by LLMs, which can compromise both the learning experience and ethical standards. To address these challenges, we proposed GenAI-enabled approaches for evaluating personalised SRL scaffolds before they are presented to students, aiming for reducing hallucinations and improving the overall quality of LLM-generated personalised scaffolds. Specifically, two approaches are investigated. The first approach involved developing a multi-agent system approach for reliability evaluation to assess the extent to which LLM-generated scaffolds accurately target relevant SRL processes. The second approach utilised the "LLM-as-a-Judge" technique for quality evaluation that evaluates LLM-generated scaffolds for their helpfulness in supporting students. We constructed evaluation datasets, and compared our results with single-agent LLM systems and machine learning approach baselines. Our findings indicate that the reliability evaluation approach is highly effective and outperforms the baselines, showing almost perfect alignment with human experts' evaluations. Moreover, both proposed evaluation approaches can be harnessed to effectively reduce hallucinations. Additionally, we identified and discussed bias limitations of the "LLM-as-a-Judge" technique in evaluating LLM-generated scaffolds. We suggest incorporating these approaches into GenAI-powered personalised SRL scaffolding systems to mitigate hallucination issues and improve the overall scaffolding quality.
>
---
#### [replaced 006] Working with AI: Measuring the Applicability of Generative AI to Occupations
- **分类: cs.AI; cs.CY; econ.GN; q-fin.EC**

- **链接: [http://arxiv.org/pdf/2507.07935v4](http://arxiv.org/pdf/2507.07935v4)**

> **作者:** Kiran Tomlinson; Sonia Jaffe; Will Wang; Scott Counts; Siddharth Suri
>
> **备注:** 42 pages
>
> **摘要:** Given the rapid adoption of generative AI and its potential to impact a wide range of tasks, understanding the effects of AI on the economy is one of society's most important questions. In this work, we take a step toward that goal by analyzing the work activities people do with AI, how successfully and broadly those activities are done, and combine that with data on what occupations do those activities. We analyze a dataset of 200k anonymized and privacy-scrubbed conversations between users and Microsoft Bing Copilot, a publicly available generative AI system. We find the most common work activities people seek AI assistance for involve gathering information and writing, while the most common activities that AI itself is performing are providing information and assistance, writing, teaching, and advising. Combining these activity classifications with measurements of task success and scope of impact, we compute an AI applicability score for each occupation. We find the highest AI applicability scores for knowledge work occupation groups such as computer and mathematical, and office and administrative support, as well as occupations such as sales whose work activities involve providing and communicating information. Additionally, we characterize the types of work activities performed most successfully, how wage and education correlate with AI applicability, and how real-world usage compares to predictions of occupational AI impact.
>
---
