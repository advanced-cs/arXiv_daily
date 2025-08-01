# 计算机与社会 cs.CY

- **最新发布 12 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Architectural practice process and artificial intelligence -- an evolving practice
- **分类: cs.CY**

- **简介: 该论文探讨人工智能在建筑设计中的应用，分析其提升创意与效率的潜力及局限性。研究采用文献综述方法，梳理AI在建筑流程中的角色演变，指出其在空间感官与体验维度的不足。核心任务是明确AI作为协作伙伴的角色，促进建筑设计的未来发展。**

- **链接: [http://arxiv.org/pdf/2507.23653v1](http://arxiv.org/pdf/2507.23653v1)**

> **作者:** Mustapha El Moussaoui
>
> **备注:** 15 pages, 7 figures. De Gruyter Brill - Open Engineering 2025
>
> **摘要:** In an era of exponential technological advancement, artificial intelligence (AI) has emerged as a transformative force in architecture, reshaping traditional design and construction practices. This article explores the multifaceted roles of AI in the architectural process, emphasizing its potential to enhance creativity and efficiency while addressing its limitations in capturing multisensory and experiential dimensions of space. Historically, architectural innovation has paralleled technological progress, from basic tools to advanced computer-aided design systems. However, the integration of AI presents unique challenges, requiring architects to critically evaluate its role in design. A narrative review methodology was adopted, focusing on academic sources selected for their relevance, recency, and credibility. The findings reveal that AI is increasingly integrated across various stages of the architectural process, from early conceptualization and site analysis to generative design and construction detailing. AI tools excel at automating repetitive tasks and generating innovative design solutions, freeing architects to focus on creativity and problem-solving. Additionally, AI's (text- toimage) visual representation strength challenges the ocularcentric approaches in architecture, which should push future architects to address the holistic sensory and experiential qualities of space or the critical thinking inherent to architectural design. While AI offers transformative potential, architects must view it as a collaborative partner rather than a passive tool.
>
---
#### [new 002] SmartCourse: A Contextual AI-Powered Course Advising System for Undergraduates
- **分类: cs.CY; cs.AI**

- **简介: 论文提出SmartCourse，一个面向本科生（计算机专业）的AI课程推荐系统。它结合学生成绩单与学习计划，利用本地大模型提供个性化选课建议。系统包含管理功能与多界面支持，通过综合上下文信息提升推荐准确性，验证了上下文对学术规划的重要性。**

- **链接: [http://arxiv.org/pdf/2507.22946v1](http://arxiv.org/pdf/2507.22946v1)**

> **作者:** Yixuan Mi; Yiduo Yu; Yiyi Zhao
>
> **备注:** 7 pages, 6 figures, 1 table. *Corresponding author: Yixuan Mi. Code: https://github.com/EthanYixuanMi/Smartcourse-Contextual-Advising
>
> **摘要:** We present SmartCourse, an integrated course management and AI-driven advising system for undergraduate students (specifically tailored to the Computer Science (CPS) major). SmartCourse addresses the limitations of traditional advising tools by integrating transcript and plan information for student-specific context. The system combines a command-line interface (CLI) and a Gradio web GUI for instructors and students, manages user accounts, course enrollment, grading, and four-year degree plans, and integrates a locally hosted large language model (via Ollama) for personalized course recommendations. It leverages transcript and major plan to offer contextual advice (e.g., prioritizing requirements or retakes). We evaluated the system on 25 representative advising queries and introduced custom metrics: PlanScore, PersonalScore, Lift, and Recall to assess recommendation quality across different context conditions. Experiments show that using full context yields substantially more relevant recommendations than context-omitted modes, confirming the necessity of transcript and plan information for personalized academic advising. SmartCourse thus demonstrates how transcript-aware AI can enhance academic planning.
>
---
#### [new 003] Automating AI Failure Tracking: Semantic Association of Reports in AI Incident Database
- **分类: cs.CY; cs.AI; cs.IR**

- **简介: 该论文属于信息检索与自然语言处理任务，旨在解决AI事故数据库中新增报告与已有事件自动关联的问题。现有方法依赖人工，效率低。作者提出基于语义相似度的检索框架，使用标题和描述结合提升关联准确性，并验证了模型性能随训练数据增加而提升。**

- **链接: [http://arxiv.org/pdf/2507.23669v1](http://arxiv.org/pdf/2507.23669v1)**

> **作者:** Diego Russo; Gian Marco Orlando; Valerio La Gatta; Vincenzo Moscato
>
> **备注:** Accepted at the 28th European Conference on Artificial Intelligence (ECAI 2025)
>
> **摘要:** Artificial Intelligence (AI) systems are transforming critical sectors such as healthcare, finance, and transportation, enhancing operational efficiency and decision-making processes. However, their deployment in high-stakes domains has exposed vulnerabilities that can result in significant societal harm. To systematically study and mitigate these risk, initiatives like the AI Incident Database (AIID) have emerged, cataloging over 3,000 real-world AI failure reports. Currently, associating a new report with the appropriate AI Incident relies on manual expert intervention, limiting scalability and delaying the identification of emerging failure patterns. To address this limitation, we propose a retrieval-based framework that automates the association of new reports with existing AI Incidents through semantic similarity modeling. We formalize the task as a ranking problem, where each report-comprising a title and a full textual description-is compared to previously documented AI Incidents based on embedding cosine similarity. Benchmarking traditional lexical methods, cross-encoder architectures, and transformer-based sentence embedding models, we find that the latter consistently achieve superior performance. Our analysis further shows that combining titles and descriptions yields substantial improvements in ranking accuracy compared to using titles alone. Moreover, retrieval performance remains stable across variations in description length, highlighting the robustness of the framework. Finally, we find that retrieval performance consistently improves as the training set expands. Our approach provides a scalable and efficient solution for supporting the maintenance of the AIID.
>
---
#### [new 004] ELMES: An Automated Framework for Evaluating Large Language Models in Educational Scenarios
- **分类: cs.CY; cs.CL; cs.LG**

- **简介: 该论文提出了ELMES，一个用于评估教育场景中大语言模型（LLMs）的自动化框架。任务是解决当前LLMs在教育应用中缺乏适配评估指标的问题。工作包括构建模块化框架、设计多智能体对话机制、开发LLM-as-a-Judge评估方法，并对多个教育场景进行系统评测。**

- **链接: [http://arxiv.org/pdf/2507.22947v1](http://arxiv.org/pdf/2507.22947v1)**

> **作者:** Shou'ang Wei; Xinyun Wang; Shuzhen Bi; Jian Chen; Ruijia Li; Bo Jiang; Xin Lin; Min Zhang; Yu Song; BingDong Li; Aimin Zhou; Hao Hao
>
> **摘要:** The emergence of Large Language Models (LLMs) presents transformative opportunities for education, generating numerous novel application scenarios. However, significant challenges remain: evaluation metrics vary substantially across different educational scenarios, while many emerging scenarios lack appropriate assessment metrics. Current benchmarks predominantly measure general intelligence rather than pedagogical capabilities. To address this gap, we introduce ELMES, an open-source automated evaluation framework specifically designed for assessing LLMs in educational settings. ELMES features a modular architecture that enables researchers to create dynamic, multi-agent dialogues through simple configuration files, facilitating flexible scenario design without requiring extensive programming expertise. The framework incorporates a hybrid evaluation engine that objectively quantifies traditionally subjective pedagogical metrics using an LLM-as-a-Judge methodology. We conduct systematic benchmarking of state-of-the-art LLMs across four critical educational scenarios: Knowledge Point Explanation, Guided Problem-Solving Teaching, Interdisciplinary Lesson Plan Generation, and Contextualized Question Generation, employing fine-grained metrics developed in collaboration with education specialists. Our results demonstrate distinct capability distributions among models, revealing context-specific strengths and limitations. ELMES provides educators and researchers with an accessible evaluation framework that significantly reduces adaptation barriers for diverse educational applications while advancing the practical implementation of LLMs in pedagogy. The framework is publicly available at \emph{https://github.com/sii-research/elmes.git}.
>
---
#### [new 005] Informing AI Risk Assessment with News Media: Analyzing National and Political Variation in the Coverage of AI Risks
- **分类: cs.CY**

- **简介: 该论文属于社会科学研究任务，旨在分析不同国家和政治倾向媒体对AI风险报道的差异。论文比较了六国媒体数据，揭示了报道中AI风险优先级及政治化语言使用的不同，为AI风险评估提供社会视角。**

- **链接: [http://arxiv.org/pdf/2507.23718v1](http://arxiv.org/pdf/2507.23718v1)**

> **作者:** Mowafak Allaham; Kimon Kieslich; Nicholas Diakopoulos
>
> **备注:** Accepted to 8th AAAI/ACM Conference on AI, Ethics, and Society (2025)
>
> **摘要:** Risk-based approaches to AI governance often center the technological artifact as the primary focus of risk assessments, overlooking systemic risks that emerge from the complex interaction between AI systems and society. One potential source to incorporate more societal context into these approaches is the news media, as it embeds and reflects complex interactions between AI systems, human stakeholders, and the larger society. News media is influential in terms of which AI risks are emphasized and discussed in the public sphere, and thus which risks are deemed important. Yet, variations in the news media between countries and across different value systems (e.g. political orientations) may differentially shape the prioritization of risks through the media's agenda setting and framing processes. To better understand these variations, this work presents a comparative analysis of a cross-national sample of news media spanning 6 countries (the U.S., the U.K., India, Australia, Israel, and South Africa). Our findings show that AI risks are prioritized differently across nations and shed light on how left vs. right leaning U.S. based outlets not only differ in the prioritization of AI risks in their coverage, but also use politicized language in the reporting of these risks. These findings can inform risk assessors and policy-makers about the nuances they should account for when considering news media as a supplementary source for risk-based governance approaches.
>
---
#### [new 006] Future Illiteracies -- Architectural Epistemology and Artificial Intelligence
- **分类: cs.CY**

- **简介: 该论文探讨人工智能时代建筑实践面临的重复与创新问题，分析数据在AI系统中的作用及其对建筑创作的影响。任务是研究建筑认知论与AI的关系，提出应发挥人类创造力与主体性，使AI成为推动建筑实践垂直与水平发展的工具，避免陷入被动标准化设计。**

- **链接: [http://arxiv.org/pdf/2507.23434v1](http://arxiv.org/pdf/2507.23434v1)**

> **作者:** Mustapha El Moussaoui
>
> **备注:** 14 pages, 7 figures. MDPI - Architecture 2025
>
> **摘要:** In the age of artificial intelligence, architectural practice faces a paradox of immense potential and creeping standardization. As humans are increasingly relying on AI-generated outputs, architecture risks becoming a spectacle of repetition- a shuffling of data that neither truly innovates nor progresses vertically in creative depth. This paper explores the critical role of data in AI systems, scrutinizing the training datasets that form the basis of AI's generative capabilities and the implications for architectural practice. We argue that when architects approach AI passively, without actively engaging their own creative and critical faculties, they risk becoming passive users locked in an endless loop of horizontal expansion without meaningful vertical growth. By examining the epistemology of architecture in the AI age, this paper calls for a paradigm where AI serves as a tool for vertical and horizontal growth, contingent on human creativity and agency. Only by mastering this dynamic relationship can architects avoid the trap of passive, standardized design and unlock the true potential of AI.
>
---
#### [new 007] SigBERT: Combining Narrative Medical Reports and Rough Path Signature Theory for Survival Risk Estimation in Oncology
- **分类: cs.CL; cs.CY; cs.LG; stat.AP**

- **简介: 论文提出SigBERT，用于肿瘤学中的生存风险估计任务。它结合医学报告与粗糙路径签名理论，处理时序文本数据，提取几何特征并融合至Cox模型，提升风险预测性能。**

- **链接: [http://arxiv.org/pdf/2507.22941v1](http://arxiv.org/pdf/2507.22941v1)**

> **作者:** Paul Minchella; Loïc Verlingue; Stéphane Chrétien; Rémi Vaucher; Guillaume Metzler
>
> **备注:** 12 pages, 2 figures, accepted for ECML PKDD 2025
>
> **摘要:** Electronic medical reports (EHR) contain a vast amount of information that can be leveraged for machine learning applications in healthcare. However, existing survival analysis methods often struggle to effectively handle the complexity of textual data, particularly in its sequential form. Here, we propose SigBERT, an innovative temporal survival analysis framework designed to efficiently process a large number of clinical reports per patient. SigBERT processes timestamped medical reports by extracting and averaging word embeddings into sentence embeddings. To capture temporal dynamics from the time series of sentence embedding coordinates, we apply signature extraction from rough path theory to derive geometric features for each patient, which significantly enhance survival model performance by capturing complex temporal dynamics. These features are then integrated into a LASSO-penalized Cox model to estimate patient-specific risk scores. The model was trained and evaluated on a real-world oncology dataset from the L\'eon B\'erard Center corpus, with a C-index score of 0.75 (sd 0.014) on the independent test cohort. SigBERT integrates sequential medical data to enhance risk estimation, advancing narrative-based survival analysis.
>
---
#### [new 008] Knowledge Is More Than Performance: How Knowledge Diversity Drives Human-Human and Human-AI Interaction Synergy and Reveals Pure-AI Interaction Shortfalls
- **分类: cs.HC; cs.CY**

- **简介: 该论文研究人与AI及纯AI组在协作解决问题中的表现差异。任务是分析对话协作效果，解决为何纯AI组缺乏协同提升的问题。工作包括对比人类与LLM不同组合的问答准确性、信心及知识多样性，发现知识多样性是协同关键，提出需重视AI多样性以提升协作效果。**

- **链接: [http://arxiv.org/pdf/2507.22889v1](http://arxiv.org/pdf/2507.22889v1)**

> **作者:** Tom Sheffer; Alon Miron; Yaniv Dover; Ariel Goldstein
>
> **摘要:** Conversations transform individual knowledge into collective insight, allowing groups of humans and increasingly groups of artificial intelligence (AI) agents to collaboratively solve complex problems. Whether interactions between AI agents can replicate the synergy observed in human discussions remains an open question. To investigate this, we systematically compared four conversational configurations: pairs of large language models (LLM-LLM), trios of LLMs, trios of humans, and mixed human-LLM pairs. After agents answered questions individually, they engaged in open-ended discussions and then reconsidered their initial answers. Interactions involving humans consistently led to accuracy improvements after the conversations, benefiting both stronger and weaker participants. By contrast, purely LLM-based pairs and trios exhibited declines in accuracy, demonstrating limited conversational synergy. Analysis of participants' confidence and answer-switching behavior revealed that knowledge diversity is a critical factor enabling collaborative improvement. Crucially, the lack of gains in LLM-LLM interactions did not stem from a fundamental limitation of the models' ability to collaborate, but from highly similar knowledge states that left little room for productive exchange. Our findings argue for a paradigm shift in AI development: rather than optimizing individual models solely for standalone performance, explicitly cultivating diversity across agents, even at the cost of slightly lower individual accuracy, may yield AI collaborators that are more effective in group settings with humans or other AI systems.
>
---
#### [new 009] Breaking the mould of Social Mixed Reality -- State-of-the-Art and Glossary
- **分类: cs.HC; cs.CY; cs.ET; cs.GR; q-bio.NC; I.3.0; I.2; J.4; K.4**

- **简介: 该论文旨在解决混合现实（MR）技术在真实还原人类具身性与社会互动方面的不足。任务是推动MR技术向以人为本的方向发展，实现更丰富的社交体验。工作包括分析现状、提出术语表，并探讨虚拟角色、自主化、伦理设计及神经科学等关键议题。**

- **链接: [http://arxiv.org/pdf/2507.23454v1](http://arxiv.org/pdf/2507.23454v1)**

> **作者:** Marta Bieńkiewicz; Julia Ayache; Panayiotis Charalambous; Cristina Becchio; Marco Corragio; Bertram Taetz; Francesco De Lellis; Antonio Grotta; Anna Server; Daniel Rammer; Richard Kulpa; Franck Multon; Azucena Garcia-Palacios; Jessica Sutherland; Kathleen Bryson; Stéphane Donikian; Didier Stricker; Benoît Bardy
>
> **备注:** pre-print
>
> **摘要:** This article explores a critical gap in Mixed Reality (MR) technology: while advances have been made, MR still struggles to authentically replicate human embodiment and socio-motor interaction. For MR to enable truly meaningful social experiences, it needs to incorporate multi-modal data streams and multi-agent interaction capabilities. To address this challenge, we present a comprehensive glossary covering key topics such as Virtual Characters and Autonomisation, Responsible AI, Ethics by Design, and the Scientific Challenges of Social MR within Neuroscience, Embodiment, and Technology. Our aim is to drive the transformative evolution of MR technologies that prioritize human-centric innovation, fostering richer digital connections. We advocate for MR systems that enhance social interaction and collaboration between humans and virtual autonomous agents, ensuring inclusivity, ethical design and psychological safety in the process.
>
---
#### [new 010] Opacity as Authority: Arbitrariness and the Preclusion of Contestation
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文探讨任意性作为人类系统的基础功能机制，而非缺陷。它提出“动机→可验证性→可争议性”链条，分析任意性如何通过隐藏逻辑削弱问责，构建权威。论文旨在重新定义任意性，适用于法律、社会及AI系统。**

- **链接: [http://arxiv.org/pdf/2507.22944v1](http://arxiv.org/pdf/2507.22944v1)**

> **作者:** Naomi Omeonga wa Kayembe
>
> **摘要:** This article redefines arbitrariness not as a normative flaw or a symptom of domination, but as a foundational functional mechanism structuring human systems and interactions. Diverging from critical traditions that conflate arbitrariness with injustice, it posits arbitrariness as a semiotic trait: a property enabling systems - linguistic, legal, or social - to operate effectively while withholding their internal rationale. Building on Ferdinand de Saussure's concept of l'arbitraire du signe, the analysis extends this principle beyond language to demonstrate its cross-domain applicability, particularly in law and social dynamics. The paper introduces the "Motivation -> Constatability -> Contestability" chain, arguing that motivation functions as a crucial interface rendering an act's logic vulnerable to intersubjective contestation. When this chain is broken through mechanisms like "immotivization" or "Conflict Lateralization" (exemplified by "the blur of the wolf drowned in the fish"), acts produce binding effects without exposing their rationale, thus precluding justiciability. This structural opacity, while appearing illogical, is a deliberate design protecting authority from accountability. Drawing on Shannon's entropy model, the paper formalizes arbitrariness as A = H(L|M) (conditional entropy). It thereby proposes a modern theory of arbitrariness as a neutral operator central to control as well as care, an overlooked dimension of interpersonal relations. While primarily developed through human social systems, this framework also illuminates a new pathway for analyzing explainability in advanced artificial intelligence systems.
>
---
#### [new 011] Transparent AI: The Case for Interpretability and Explainability
- **分类: cs.LG; cs.AI; cs.CY**

- **简介: 该论文探讨人工智能透明性问题，旨在提升AI系统的可解释性与可解释性。属于AI伦理与可解释性研究任务。论文总结了跨领域实践经验，为不同成熟度的组织提供实施策略，强调将可解释性作为AI设计核心原则。**

- **链接: [http://arxiv.org/pdf/2507.23535v1](http://arxiv.org/pdf/2507.23535v1)**

> **作者:** Dhanesh Ramachandram; Himanshu Joshi; Judy Zhu; Dhari Gandhi; Lucas Hartman; Ananya Raval
>
> **摘要:** As artificial intelligence systems increasingly inform high-stakes decisions across sectors, transparency has become foundational to responsible and trustworthy AI implementation. Leveraging our role as a leading institute in advancing AI research and enabling industry adoption, we present key insights and lessons learned from practical interpretability applications across diverse domains. This paper offers actionable strategies and implementation guidance tailored to organizations at varying stages of AI maturity, emphasizing the integration of interpretability as a core design principle rather than a retrospective add-on.
>
---
#### [new 012] Beyond the Cloud: Assessing the Benefits and Drawbacks of Local LLM Deployment for Translators
- **分类: cs.CL; cs.CY; I.2.7; K.4.3**

- **简介: 该论文属于翻译技术任务，探讨本地部署大语言模型（LLM）对译者的影响。论文旨在解决数据隐私、安全与访问公平问题，评估了三种开源本地模型与云端商业模型的性能。研究重点是比较功能表现，而非翻译质量，强调本地部署的可行性与优势。**

- **链接: [http://arxiv.org/pdf/2507.23399v1](http://arxiv.org/pdf/2507.23399v1)**

> **作者:** Peter Sandrini
>
> **摘要:** The rapid proliferation of Large Language Models presents both opportunities and challenges for the translation field. While commercial, cloud-based AI chatbots have garnered significant attention in translation studies, concerns regarding data privacy, security, and equitable access necessitate exploration of alternative deployment models. This paper investigates the feasibility and performance of locally deployable, free language models as a viable alternative to proprietary, cloud-based AI solutions. This study evaluates three open-source models installed on CPU-based platforms and compared against commercially available online chat-bots. The evaluation focuses on functional performance rather than a comparative analysis of human-machine translation quality, an area already subject to extensive research. The platforms assessed were chosen for their accessibility and ease of use across various operating systems. While local deployment introduces its own challenges, the benefits of enhanced data control, improved privacy, and reduced dependency on cloud services are compelling. The findings of this study contribute to a growing body of knowledge concerning the democratization of AI technology and inform future research and development efforts aimed at making LLMs more accessible and practical for a wider range of users, specifically focusing on the needs of individual translators and small businesses.
>
---
## 更新

#### [replaced 001] EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework
- **分类: cs.AI; cs.CE; cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2504.14928v3](http://arxiv.org/pdf/2504.14928v3)**

> **作者:** Yao Shi; Rongkeng Liang; Yong Xu
>
> **备注:** Paper URL: https://aclanthology.org/2025.acl-long.1576 ;Presentation Video: https://www.youtube.com/watch?v=j63ooKE50I0
>
> **摘要:** Large language models (LLMs) increasingly serve as educational tools, yet evaluating their teaching capabilities remains challenging due to the resource-intensive, context-dependent, and methodologically complex nature of teacher-student interactions. We introduce EducationQ, a multi-agent dialogue framework that efficiently assesses teaching capabilities through simulated dynamic educational scenarios, featuring specialized agents for teaching, learning, and evaluation. Testing 14 LLMs across major AI Organizations (OpenAI, Meta, Google, Anthropic, and others) on 1,498 questions spanning 13 disciplines and 10 difficulty levels reveals that teaching effectiveness does not correlate linearly with model scale or general reasoning capabilities - with some smaller open-source models outperforming larger commercial counterparts in teaching contexts. This finding highlights a critical gap in current evaluations that prioritize knowledge recall over interactive pedagogy. Our mixed-methods evaluation, combining quantitative metrics with qualitative analysis and expert case studies, identifies distinct pedagogical strengths employed by top-performing models (e.g., sophisticated questioning strategies, adaptive feedback mechanisms). Human expert evaluations show 78% agreement with our automated qualitative analysis of effective teaching behaviors, validating our methodology. EducationQ demonstrates that LLMs-as-teachers require specialized optimization beyond simple scaling, suggesting next-generation educational AI prioritize targeted enhancement of specific pedagogical effectiveness.
>
---
#### [replaced 002] Prompt Engineering Techniques for Mitigating Cultural Bias Against Arabs and Muslims in Large Language Models: A Systematic Review
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.18199v2](http://arxiv.org/pdf/2506.18199v2)**

> **作者:** Bushra Asseri; Estabrag Abdelaziz; Areej Al-Wabil
>
> **备注:** Research is incomplete
>
> **摘要:** Large language models have demonstrated remarkable capabilities across various domains, yet concerns about cultural bias - particularly towards Arabs and Muslims - pose significant ethical challenges by perpetuating harmful stereotypes and marginalization. Despite growing recognition of bias in LLMs, prompt engineering strategies specifically addressing Arab and Muslim representation remain understudied. This mixed-methods systematic review examines such techniques, offering evidence-based guidance for researchers and practitioners. Following PRISMA guidelines and Kitchenham's systematic review methodology, we analyzed 8 empirical studies published between 2021-2024 investigating bias mitigation strategies. Our findings reveal five primary prompt engineering approaches: cultural prompting, affective priming, self-debiasing techniques, structured multi-step pipelines, and parameter-optimized continuous prompts. Although all approaches show potential for reducing bias, effectiveness varied substantially across studies and bias types. Evidence suggests that certain bias types may be more resistant to prompt-based mitigation than others. Structured multi-step pipelines demonstrated the highest overall effectiveness, achieving up to 87.7% reduction in bias, though they require greater technical expertise. Cultural prompting offers broader accessibility with substantial effectiveness. These results underscore the accessibility of prompt engineering for mitigating cultural bias without requiring access to model parameters. The limited number of studies identified highlights a significant research gap in this critical area. Future research should focus on developing culturally adaptive prompting techniques, creating Arab and Muslim-specific evaluation resources, and integrating prompt engineering with complementary debiasing methods to address deeper stereotypes while maintaining model utility.
>
---
#### [replaced 003] Toward Integrated Solutions: A Systematic Interdisciplinary Review of Cybergrooming Research
- **分类: cs.CY; cs.CR**

- **链接: [http://arxiv.org/pdf/2503.05727v2](http://arxiv.org/pdf/2503.05727v2)**

> **作者:** Heajun An; Marcos Silva; Qi Zhang; Arav Singh; Minqian Liu; Xinyi Zhang; Sarvech Qadir; Sang Won Lee; Lifu Huang; Pamela J. Wisniewski; Jin-Hee Cho
>
> **摘要:** Cybergrooming exploits minors through online trust-building, yet research remains fragmented, limiting holistic prevention. Social sciences focus on behavioral insights, while computational methods emphasize detection, but their integration remains insufficient. This review systematically synthesizes both fields using the PRISMA framework to enhance clarity, reproducibility, and cross-disciplinary collaboration. Findings show that qualitative methods offer deep insights but are resource-intensive, machine learning models depend on data quality, and standard metrics struggle with imbalance and cultural nuances. By bridging these gaps, this review advances interdisciplinary cybergrooming research, guiding future efforts toward more effective prevention and detection strategies.
>
---
#### [replaced 004] DeepShade: Enable Shade Simulation by Text-conditioned Image Generation
- **分类: cs.CV; cs.CY; 68T45, 68U10, 62H35; I.2.10; I.4.8; I.5.1**

- **链接: [http://arxiv.org/pdf/2507.12103v3](http://arxiv.org/pdf/2507.12103v3)**

> **作者:** Longchao Da; Xiangrui Liu; Mithun Shivakoti; Thirulogasankar Pranav Kutralingam; Yezhou Yang; Hua Wei
>
> **备注:** 7pages, 4 figures
>
> **摘要:** Heatwaves pose a significant threat to public health, especially as global warming intensifies. However, current routing systems (e.g., online maps) fail to incorporate shade information due to the difficulty of estimating shades directly from noisy satellite imagery and the limited availability of training data for generative models. In this paper, we address these challenges through two main contributions. First, we build an extensive dataset covering diverse longitude-latitude regions, varying levels of building density, and different urban layouts. Leveraging Blender-based 3D simulations alongside building outlines, we capture building shadows under various solar zenith angles throughout the year and at different times of day. These simulated shadows are aligned with satellite images, providing a rich resource for learning shade patterns. Second, we propose the DeepShade, a diffusion-based model designed to learn and synthesize shade variations over time. It emphasizes the nuance of edge features by jointly considering RGB with the Canny edge layer, and incorporates contrastive learning to capture the temporal change rules of shade. Then, by conditioning on textual descriptions of known conditions (e.g., time of day, solar angles), our framework provides improved performance in generating shade images. We demonstrate the utility of our approach by using our shade predictions to calculate shade ratios for real-world route planning in Tempe, Arizona. We believe this work will benefit society by providing a reference for urban planning in extreme heat weather and its potential practical applications in the environment.
>
---
#### [replaced 005] Leveraging LLMs to Create Content Corpora for Niche Domains
- **分类: cs.CL; cs.AI; cs.CY; I.2.7; H.3.1; H.3.3**

- **链接: [http://arxiv.org/pdf/2505.02851v2](http://arxiv.org/pdf/2505.02851v2)**

> **作者:** Franklin Zhang; Sonya Zhang; Alon Halevy
>
> **备注:** 9 pages (main content), 5 figures. Supplementary materials can be found at https://github.com/pigfyy/30DayGen-Supplementary-Materials
>
> **摘要:** Constructing specialized content corpora from vast, unstructured web sources for domain-specific applications poses substantial data curation challenges. In this paper, we introduce a streamlined approach for generating high-quality, domain-specific corpora by efficiently acquiring, filtering, structuring, and cleaning web-based data. We showcase how Large Language Models (LLMs) can be leveraged to address complex data curation at scale, and propose a strategical framework incorporating LLM-enhanced techniques for structured content extraction and semantic deduplication. We validate our approach in the behavior education domain through its integration into 30 Day Me, a habit formation application. Our data pipeline, named 30DayGen, enabled the extraction and synthesis of 3,531 unique 30-day challenges from over 15K webpages. A user survey reports a satisfaction score of 4.3 out of 5, with 91% of respondents indicating willingness to use the curated content for their habit-formation goals.
>
---
#### [replaced 006] Disparate Conditional Prediction in Multiclass Classifiers
- **分类: cs.LG; cs.CY; stat.ML**

- **链接: [http://arxiv.org/pdf/2206.03234v3](http://arxiv.org/pdf/2206.03234v3)**

> **作者:** Sivan Sabato; Eran Treister; Elad Yom-Tov
>
> **备注:** Published at ICML 2025
>
> **摘要:** We propose methods for auditing multiclass classifiers for fairness under multiclass equalized odds,by estimating the deviation from equalized odds when the classifier is not completely fair. We generalize to multiclass classifiers the measure of Disparate Conditional Prediction (DCP), originally suggested by Sabato & Yom-Tov (2020) for binary classifiers. DCP is defined as the fraction of the population for which the classifier predicts with conditional prediction probabilities that differ from the closest common baseline. We provide new local-optimization methods for estimating the multiclass DCPunder two different regimes,one in which the conditional confusion matrices for each protected sub-population are known, and one in which these cannot be estimated, for instance, because the classifier is inaccessible or because good-quality individual-level data is not available. These methods can be used to detect classifiers that likely treat a significant fraction of the population unfairly. Experiments demonstrate the accuracy of the methods. Code is provided at https://github.com/sivansabato/ DCPmulticlass.
>
---
#### [replaced 007] "I will never pay for this" Perception of fairness and factors affecting behaviour on 'pay-or-ok' models
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2505.12892v4](http://arxiv.org/pdf/2505.12892v4)**

> **作者:** Victor Morel; Farzaneh Karegar; Cristiana Santos
>
> **备注:** Accepted for publication at APF2025
>
> **摘要:** The rise of cookie paywalls ('pay-or-ok' models) has prompted growing debates around the right to privacy and data protection, monetisation, and the legitimacy of user consent. Despite their increasing use across sectors, limited research has explored how users perceive these models or what shapes their decisions to either consent to tracking or pay. To address this gap, we conducted four focus groups (n= 14) to examine users' perceptions of cookie paywalls, their judgments of fairness, and the conditions under which they might consider paying, alongside a legal analysis within the EU data protection legal framework. Participants primarily viewed cookie paywalls as profit-driven, with fairness perceptions varying depending on factors such as the presence of a third option beyond consent or payment, transparency of data practices, and the authenticity or exclusivity of the paid content. Participants voiced expectations for greater transparency, meaningful control over data collection, and less coercive alternatives, such as contextual advertising or "reject all" buttons. Although some conditions, including trusted providers, exclusive content, and reasonable pricing, could make participants consider paying, most expressed reluctance or unwillingness to do so. Crucially, our findings raise concerns about economic exclusion, where privacy and data protection might end up becoming a privilege rather than fundamental rights. Consent given under financial pressure may not meet the standard of being freely given, as required by the GDPR. To address these concerns, we recommend user-centred approaches that enhance transparency, reduce coercion, ensure the value of paid content, and explore inclusive alternatives. These measures are essential for supporting fairness, meaningful choice, and user autonomy in consent-driven digital environments.
>
---
#### [replaced 008] How AI Ideas Affect the Creativity, Diversity, and Evolution of Human Ideas: Evidence From a Large, Dynamic Experiment
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2401.13481v3](http://arxiv.org/pdf/2401.13481v3)**

> **作者:** Joshua Ashkinaze; Julia Mendelsohn; Li Qiwei; Ceren Budak; Eric Gilbert
>
> **备注:** Accepted at ACM Collective Intelligence 2025. Originally posted 2024
>
> **摘要:** Exposure to large language model output is rapidly increasing. How will seeing AI-generated ideas affect human ideas? We conducted an experiment (800+ participants, 40+ countries) where participants viewed creative ideas that were from ChatGPT or prior experimental participants and then brainstormed their own idea. We varied the number of AI-generated examples (none, low, or high exposure) and if the examples were labeled as 'AI' (disclosure). Our dynamic experiment design -- ideas from prior participants in an experimental condition are used as stimuli for future participants in the same experimental condition -- speaks to the interdependent process of cultural creation: creative ideas are built upon prior ideas. Hence, we capture the compounding effects of having LLMs 'in the culture loop'. We find that high AI exposure (but not low AI exposure) did not affect the creativity of individual ideas but did increase the average amount and rate of change of collective idea diversity. AI made ideas different, not better. There were no main effects of disclosure. We also found that self-reported creative people were less influenced by knowing an idea was from AI and that participants may knowingly adopt AI ideas when the task is difficult. Our findings suggest that introducing AI ideas may increase collective diversity but not individual creativity.
>
---
