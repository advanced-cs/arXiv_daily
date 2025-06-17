# 计算机与社会 cs.CY

- **最新发布 33 篇**

- **更新 12 篇**

## 最新发布

#### [new 001] Bias Delayed is Bias Denied? Assessing the Effect of Reporting Delays on Disparity Assessments
- **分类: cs.CY**

- **简介: 该论文属于算法公平性任务，研究延迟报告的种族数据对公平性评估的影响，发现延迟导致评估偏差，并验证现有方法难以缓解此问题。**

- **链接: [http://arxiv.org/pdf/2506.13735v1](http://arxiv.org/pdf/2506.13735v1)**

> **作者:** Jennah Gosciak; Aparna Balagopalan; Derek Ouyang; Allison Koenecke; Marzyeh Ghassemi; Daniel E. Ho
>
> **摘要:** Conducting disparity assessments at regular time intervals is critical for surfacing potential biases in decision-making and improving outcomes across demographic groups. Because disparity assessments fundamentally depend on the availability of demographic information, their efficacy is limited by the availability and consistency of available demographic identifiers. While prior work has considered the impact of missing data on fairness, little attention has been paid to the role of delayed demographic data. Delayed data, while eventually observed, might be missing at the critical point of monitoring and action -- and delays may be unequally distributed across groups in ways that distort disparity assessments. We characterize such impacts in healthcare, using electronic health records of over 5M patients across primary care practices in all 50 states. Our contributions are threefold. First, we document the high rate of race and ethnicity reporting delays in a healthcare setting and demonstrate widespread variation in rates at which demographics are reported across different groups. Second, through a set of retrospective analyses using real data, we find that such delays impact disparity assessments and hence conclusions made across a range of consequential healthcare outcomes, particularly at more granular levels of state-level and practice-level assessments. Third, we find limited ability of conventional methods that impute missing race in mitigating the effects of reporting delays on the accuracy of timely disparity assessments. Our insights and methods generalize to many domains of algorithmic fairness where delays in the availability of sensitive information may confound audits, thus deserving closer attention within a pipeline-aware machine learning framework.
>
---
#### [new 002] The Transition Matrix -- A classification of navigational patterns between LMS course sections
- **分类: cs.CY; cs.HC**

- **简介: 该论文属于教育数据分析任务，旨在研究LMS课程章节间的导航模式。通过构建转移矩阵和热图，分析学生在不同课程部分间的浏览行为，识别常见路径与结构特征。**

- **链接: [http://arxiv.org/pdf/2506.13275v1](http://arxiv.org/pdf/2506.13275v1)**

> **作者:** Tobias Hildebrandt; Lars Mehnen
>
> **摘要:** Learning management systems (LMS) like Moodle are increasingly used to support university teaching. As Moodle courses become more complex, incorporating diverse interactive elements, it is important to understand how students navigate through course sections and whether course designs are meeting student needs. While substantial research exists on student usage of individual LMS elements, there is a lack of research on broader navigational patterns between course sections and how these patterns differ across courses. This study analyzes navigational data from 747 courses in the Moodle LMS at a technical university of applied sciences, representing (after filtering) around 4,400 students and 1.8 million logged events. By mapping section names across a large sample of courses, the analysis enables cross-course comparisons of student navigational sequences between sections. Transition matrices and heat map visualizations are used to identify common navigational patterns. Findings include that many of the generated heatmap include one or more diagonal axis, indicating that students typically navigate from the current to the next or previous section. More fine-grained patterns show typical behavior for blended learning scenarios. Other patterns include dominant sections.
>
---
#### [new 003] Accessibility Barriers in Multi-Terabyte Public Datasets: The Gap Between Promise and Practice
- **分类: cs.CY; cs.DL; cs.IR; 68P20, 91D30; H.3.7; K.4.3**

- **简介: 该论文属于数据可访问性研究，探讨公开大数据集的实际使用障碍，揭示其理论开放与实践限制间的差距。**

- **链接: [http://arxiv.org/pdf/2506.13256v1](http://arxiv.org/pdf/2506.13256v1)**

> **作者:** Marc Bara
>
> **备注:** 5 pages, 28 references. Analysis of practical barriers to accessing multi-terabyte public datasets
>
> **摘要:** The promise of "free and open" multi-terabyte datasets often collides with harsh realities. While these datasets may be technically accessible, practical barriers -- from processing complexity to hidden costs -- create a system that primarily serves well-funded institutions. This study examines accessibility challenges across web crawls, satellite imagery, scientific data, and collaborative projects, revealing a consistent two-tier system where theoretical openness masks practical exclusivity. Our analysis demonstrates that datasets marketed as "publicly accessible" typically require minimum investments of \$1,000+ for meaningful analysis, with complex processing pipelines demanding \$10,000-100,000+ in infrastructure costs. The infrastructure requirements -- distributed computing knowledge, domain expertise, and substantial budgets -- effectively gatekeep these datasets despite their "open" status, limiting practical accessibility to those with institutional support or substantial resources.
>
---
#### [new 004] Safe-Child-LLM: A Developmental Benchmark for Evaluating LLM Safety in Child-AI Interactions
- **分类: cs.CY**

- **简介: 该论文属于AI安全评估任务，旨在解决儿童与大语言模型交互中的安全问题。研究构建了针对儿童和青少年的基准测试集，评估多个模型的安全性。**

- **链接: [http://arxiv.org/pdf/2506.13510v1](http://arxiv.org/pdf/2506.13510v1)**

> **作者:** Junfeng Jiao; Saleh Afroogh; Kevin Chen; Abhejay Murali; David Atkinson; Amit Dhurandhar
>
> **摘要:** As Large Language Models (LLMs) increasingly power applications used by children and adolescents, ensuring safe and age-appropriate interactions has become an urgent ethical imperative. Despite progress in AI safety, current evaluations predominantly focus on adults, neglecting the unique vulnerabilities of minors engaging with generative AI. We introduce Safe-Child-LLM, a comprehensive benchmark and dataset for systematically assessing LLM safety across two developmental stages: children (7-12) and adolescents (13-17). Our framework includes a novel multi-part dataset of 200 adversarial prompts, curated from red-teaming corpora (e.g., SG-Bench, HarmBench), with human-annotated labels for jailbreak success and a standardized 0-5 ethical refusal scale. Evaluating leading LLMs -- including ChatGPT, Claude, Gemini, LLaMA, DeepSeek, Grok, Vicuna, and Mistral -- we uncover critical safety deficiencies in child-facing scenarios. This work highlights the need for community-driven benchmarks to protect young users in LLM interactions. To promote transparency and collaborative advancement in ethical AI development, we are publicly releasing both our benchmark datasets and evaluation codebase at https://github.com/The-Responsible-AI-Initiative/Safe_Child_LLM_Benchmark.git
>
---
#### [new 005] SocialCredit+
- **分类: cs.CY; cs.AI**

- **简介: 该论文提出SocialCredit+，一个结合AI与社交媒体数据的信用评估系统，解决传统信用评估不足的问题，通过行为分析和伊斯兰合规检查提升信用评分准确性。**

- **链接: [http://arxiv.org/pdf/2506.12099v1](http://arxiv.org/pdf/2506.12099v1)**

> **作者:** Thabassum Aslam; Anees Aslam
>
> **摘要:** SocialCredit+ is AI powered credit scoring system that leverages publicly available social media data to augment traditional credit evaluation. It uses a conversational banking assistant to gather user consent and fetch public profiles. Multimodal feature extractors analyze posts, bios, images, and friend networks to generate a rich behavioral profile. A specialized Sharia-compliance layer flags any non-halal indicators and prohibited financial behavior based on Islamic ethics. The platform employs a retrieval-augmented generation module: an LLM accesses a domain specific knowledge base to generate clear, text-based explanations for each decision. We describe the end-to-end architecture and data flow, the models used, and system infrastructure. Synthetic scenarios illustrate how social signals translate into credit-score factors. This paper emphasizes conceptual novelty, compliance mechanisms, and practical impact, targeting AI researchers, fintech practitioners, ethical banking jurists, and investors.
>
---
#### [new 006] Intelligent Automation for FDI Facilitation: Optimizing Tariff Exemption Processes with OCR And Large Language Models
- **分类: cs.CY; cs.AI; econ.GN; q-fin.EC**

- **简介: 该论文属于智能自动化任务，旨在优化关税减免流程。通过OCR和大语言模型提升数据提取与审核效率，解决人工处理耗时且易错的问题。**

- **链接: [http://arxiv.org/pdf/2506.12093v1](http://arxiv.org/pdf/2506.12093v1)**

> **作者:** Muhammad Sukri Bin Ramli
>
> **摘要:** Tariff exemptions are fundamental to attracting Foreign Direct Investment (FDI) into the manufacturing sector, though the associated administrative processes present areas for optimization for both investing entities and the national tax authority. This paper proposes a conceptual framework to empower tax administration by leveraging a synergistic integration of Optical Character Recognition (OCR) and Large Language Model (LLM) technologies. The proposed system is designed to first utilize OCR for intelligent digitization, precisely extracting data from diverse application documents and key regulatory texts such as tariff orders. Subsequently, the LLM would enhance the capabilities of administrative officers by automating the critical and time-intensive task of verifying submitted HS Tariff Codes for machinery, equipment, and raw materials against official exemption lists. By enhancing the speed and precision of these initial assessments, this AI-driven approach systematically reduces potential for non-alignment and non-optimized exemption utilization, thereby streamlining the investment journey for FDI companies. For the national administration, the benefits include a significant boost in operational capacity, reduced administrative load, and a strengthened control environment, ultimately improving the ease of doing business and solidifying the nation's appeal as a premier destination for high-value manufacturing FDI.
>
---
#### [new 007] Artificial Intelligence and Civil Discourse: How LLMs Moderate Climate Change Conversations
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于AI与社会互动研究，探讨LLMs如何通过情感中立和低情绪强度调节气候讨论，提升公共对话质量。**

- **链接: [http://arxiv.org/pdf/2506.12077v1](http://arxiv.org/pdf/2506.12077v1)**

> **作者:** Wenlu Fan; Wentao Xu
>
> **备注:** 10 pages
>
> **摘要:** As large language models (LLMs) become increasingly integrated into online platforms and digital communication spaces, their potential to influence public discourse - particularly in contentious areas like climate change - requires systematic investigation. This study examines how LLMs naturally moderate climate change conversations through their distinct communicative behaviors. We conduct a comparative analysis of conversations between LLMs and human users on social media platforms, using five advanced models: three open-source LLMs (Gemma, Llama 3, and Llama 3.3) and two commercial systems (GPT-4o by OpenAI and Claude 3.5 by Anthropic). Through sentiment analysis, we assess the emotional characteristics of responses from both LLMs and humans. The results reveal two key mechanisms through which LLMs moderate discourse: first, LLMs consistently display emotional neutrality, showing far less polarized sentiment than human users. Second, LLMs maintain lower emotional intensity across contexts, creating a stabilizing effect in conversations. These findings suggest that LLMs possess inherent moderating capacities that could improve the quality of public discourse on controversial topics. This research enhances our understanding of how AI might support more civil and constructive climate change discussions and informs the design of AI-assisted communication tools.
>
---
#### [new 008] pySpainMobility: a Python Package to Access and Manage Spanish Open Mobility Data
- **分类: cs.CY**

- **简介: 该论文介绍了一个Python库pySpainMobility，用于访问和管理西班牙开放的移动性数据。任务是解决获取和处理大规模移动性数据的难题，通过提供标准化接口降低技术门槛，支持研究与应用。**

- **链接: [http://arxiv.org/pdf/2506.13385v1](http://arxiv.org/pdf/2506.13385v1)**

> **作者:** Ciro Beneduce; Tania Gullón Muñoz-Repiso; Bruno Lepri; Massimiliano Luca
>
> **摘要:** Mobility patterns play a critical role in a wide range of societal challenges, from epidemic modeling and emergency response to transportation planning and regional development. Yet, access to high-quality, timely, and openly available mobility data remains limited. In response, the Spanish Ministry of Transportation and Sustainable Mobility has released daily mobility datasets based on anonymized mobile phone data, covering districts, municipalities, and greater urban areas from February 2020 to June 2021 and again from January 2022 onward. This paper presents pySpainMobility, a Python package that simplifies access to these datasets and their associated study areas through a standardized, well-documented interface. By lowering the technical barrier to working with large-scale mobility data, the package enables reproducible analysis and supports applications across research, policy, and operational domains. The library is available at https://github.com/pySpainMobility.
>
---
#### [new 009] An LLM's Apology: Outsourcing Awkwardness in the Age of AI
- **分类: cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.13685v1](http://arxiv.org/pdf/2506.13685v1)**

> **作者:** Twm Stone; Anna Soligo
>
> **备注:** 9 pages
>
> **摘要:** A key part of modern social dynamics is flaking at short notice. However, anxiety in coming up with believable and socially acceptable reasons to do so can instead lead to 'ghosting', awkwardness, or implausible excuses, risking emotional harm and resentment in the other party. The ability to delegate this task to a Large Language Model (LLM) could substantially reduce friction and enhance the flexibility of user's social life while greatly minimising the aforementioned creative burden and moral qualms. We introduce FLAKE-Bench, an evaluation of models' capacity to effectively, kindly, and humanely extract themselves from a diverse set of social, professional and romantic scenarios. We report the efficacy of 10 frontier or recently-frontier LLMs in bailing on prior commitments, because nothing says "I value our friendship" like having AI generate your cancellation texts. We open-source FLAKE-Bench at github.com/Cloakless/flake-bench to support future research.
>
---
#### [new 010] "I Hadn't Thought About That": Creators of Human-like AI Weigh in on Ethics And Neurodivergence
- **分类: cs.CY; cs.AI; 68**

- **简介: 该论文属于伦理研究任务，探讨人形AI设计中的神经多样性问题，分析开发者对人类定义的偏见及其对自闭症群体的影响，并提出改进方向。**

- **链接: [http://arxiv.org/pdf/2506.12098v1](http://arxiv.org/pdf/2506.12098v1)**

> **作者:** Naba Rizvi; Taggert Smith; Tanvi Vidyala; Mya Bolds; Harper Strickland; Andrew Begel; Rua Williams; Imani Munyaka
>
> **备注:** published at FAccT 2025, 15 pages, 2 tables, 4 figures
>
> **摘要:** Human-like AI agents such as robots and chatbots are becoming increasingly popular, but they present a variety of ethical concerns. The first concern is in how we define humanness, and how our definition impacts communities historically dehumanized by scientific research. Autistic people in particular have been dehumanized by being compared to robots, making it even more important to ensure this marginalization is not reproduced by AI that may promote neuronormative social behaviors. Second, the ubiquitous use of these agents raises concerns surrounding model biases and accessibility. In our work, we investigate the experiences of the people who build and design these technologies to gain insights into their understanding and acceptance of neurodivergence, and the challenges in making their work more accessible to users with diverse needs. Even though neurodivergent individuals are often marginalized for their unique communication styles, nearly all participants overlooked the conclusions their end-users and other AI system makers may draw about communication norms from the implementation and interpretation of humanness applied in participants' work. This highlights a major gap in their broader ethical considerations, compounded by some participants' neuronormative assumptions about the behaviors and traits that distinguish "humans" from "bots" and the replication of these assumptions in their work. We examine the impact this may have on autism inclusion in society and provide recommendations for additional systemic changes towards more ethical research directions.
>
---
#### [new 011] Military AI Cyber Agents (MAICAs) Constitute a Global Threat to Critical Infrastructure
- **分类: cs.CY; cs.AI**

- **简介: 该论文属于风险评估任务，探讨MAICAs对关键基础设施的威胁，分析其技术可行性与地缘政治风险，并提出应对措施。**

- **链接: [http://arxiv.org/pdf/2506.12094v1](http://arxiv.org/pdf/2506.12094v1)**

> **作者:** Timothy Dubber; Seth Lazar
>
> **摘要:** This paper argues that autonomous AI cyber-weapons - Military-AI Cyber Agents (MAICAs) - create a credible pathway to catastrophic risk. It sets out the technical feasibility of MAICAs, explains why geopolitics and the nature of cyberspace make MAICAs a catastrophic risk, and proposes political, defensive-AI and analogue-resilience measures to blunt the threat.
>
---
#### [new 012] Navigating through CS1: The Role of Self-Regulation and Supervision in Student Progress
- **分类: cs.CY; K.3.2**

- **简介: 该论文属于教育研究任务，探讨CS1课程中学生自我调节与导师监督的关系，旨在提升学生过渡到大学学习的能力。研究通过访谈分析发现自我调节受多种因素影响，强调整合个人经历的重要性。**

- **链接: [http://arxiv.org/pdf/2506.13461v1](http://arxiv.org/pdf/2506.13461v1)**

> **作者:** Ville Isomöttönen; Denis Zhidkikh
>
> **备注:** 12 pages, 3 figures, submitted to ACM Transactions on Computing Education
>
> **摘要:** The need for students' self-regulation for fluent transitioning to university studies is known. Our aim was to integrate study-supportive activities with course supervision activities within CS1. We educated TAs to pay attention to students' study ability and self-regulation. An interview study ($N=14$) was undertaken to investigate this approach. A thematic analysis yielded rather mixed results in light of our aims. Self-regulation was underpinned by the influences external to our setting, including labor market-related needs, earlier crises in study habits, and personal characteristics such as passion, grit, creativity, and valuation of utility. Safety in one-to-one supervision was considered essential, while shyness, fear, and even altruism caused self-handicapping during the course. Students were aware of their learning styles and need for self-regulation, while did not always know how to self-regulate or preferred to externalize it. The results highlight that supporting self-regulation should be integrated with students' personal histories and experiences, and thereby calls attention to transformative learning pedagogies. The thematization can help to understand CS1 students' self-regulation processes and improve CS1 support practices.
>
---
#### [new 013] Information Suppression in Large Language Models: Auditing, Quantifying, and Characterizing Censorship in DeepSeek
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于AI伦理研究任务，旨在揭示大模型中的信息压制现象。通过审计框架分析DeepSeek模型，发现其在输出中隐去敏感内容，凸显了对内容审查机制的系统性评估需求。**

- **链接: [http://arxiv.org/pdf/2506.12349v1](http://arxiv.org/pdf/2506.12349v1)**

> **作者:** Peiran Qiu; Siyi Zhou; Emilio Ferrara
>
> **摘要:** This study examines information suppression mechanisms in DeepSeek, an open-source large language model (LLM) developed in China. We propose an auditing framework and use it to analyze the model's responses to 646 politically sensitive prompts by comparing its final output with intermediate chain-of-thought (CoT) reasoning. Our audit unveils evidence of semantic-level information suppression in DeepSeek: sensitive content often appears within the model's internal reasoning but is omitted or rephrased in the final output. Specifically, DeepSeek suppresses references to transparency, government accountability, and civic mobilization, while occasionally amplifying language aligned with state propaganda. This study underscores the need for systematic auditing of alignment, content moderation, information suppression, and censorship practices implemented into widely-adopted AI models, to ensure transparency, accountability, and equitable access to unbiased information obtained by means of these systems.
>
---
#### [new 014] Fairness Research For Machine Learning Should Integrate Societal Considerations
- **分类: cs.LG; cs.AI; cs.CY; I.2.6; K.4.2; A.1**

- **简介: 该论文属于机器学习公平性研究任务，旨在解决ML系统中的偏见问题。提出应重视公平性度量并融入社会因素，以减少系统性歧视。**

- **链接: [http://arxiv.org/pdf/2506.12556v1](http://arxiv.org/pdf/2506.12556v1)**

> **作者:** Yijun Bian; Lei You
>
> **备注:** 11 pages without appendix
>
> **摘要:** Enhancing fairness in machine learning (ML) systems is increasingly important nowadays. While current research focuses on assistant tools for ML pipelines to promote fairness within them, we argue that: 1) The significance of properly defined fairness measures remains underestimated; and 2) Fairness research in ML should integrate societal considerations. The reasons include that detecting discrimination is critical due to the widespread deployment of ML systems and that human-AI feedback loops amplify biases, even when only small social and political biases persist.
>
---
#### [new 015] Bridging the Digital Divide: Small Language Models as a Pathway for Physics and Photonics Education in Underdeveloped Regions
- **分类: physics.ed-ph; cs.AI; cs.CY**

- **简介: 该论文属于教育技术领域，旨在解决欠发达地区物理和光子学教育不足的问题。通过开发小型语言模型，提供本地化、离线学习支持，以缩小数字鸿沟。**

- **链接: [http://arxiv.org/pdf/2506.12403v1](http://arxiv.org/pdf/2506.12403v1)**

> **作者:** Asghar Ghorbani; Hanieh Fattahi
>
> **摘要:** Limited infrastructure, scarce educational resources, and unreliable internet access often hinder physics and photonics education in underdeveloped regions. These barriers create deep inequities in Science, Technology, Engineering, and Mathematics (STEM) education. This article explores how Small Language Models (SLMs)-compact, AI-powered tools that can run offline on low-power devices, offering a scalable solution. By acting as virtual tutors, enabling native-language instruction, and supporting interactive learning, SLMs can help address the shortage of trained educators and laboratory access. By narrowing the digital divide through targeted investment in AI technologies, SLMs present a scalable and inclusive solution to advance STEM education and foster scientific empowerment in marginalized communities.
>
---
#### [new 016] The Amazon Nova Family of Models: Technical Report and Model Card
- **分类: cs.AI; cs.CY; cs.LG**

- **简介: 该论文介绍Amazon Nova系列模型，解决多模态AI任务中的性能与成本问题，涵盖文本、图像、视频生成及处理。**

- **链接: [http://arxiv.org/pdf/2506.12103v1](http://arxiv.org/pdf/2506.12103v1)**

> **作者:** Amazon AGI; Aaron Langford; Aayush Shah; Abhanshu Gupta; Abhimanyu Bhatter; Abhinav Goyal; Abhinav Mathur; Abhinav Mohanty; Abhishek Kumar; Abhishek Sethi; Abi Komma; Abner Pena; Achin Jain; Adam Kunysz; Adam Opyrchal; Adarsh Singh; Aditya Rawal; Adok Achar Budihal Prasad; Adrià de Gispert; Agnika Kumar; Aishwarya Aryamane; Ajay Nair; Akilan M; Akshaya Iyengar; Akshaya Vishnu Kudlu Shanbhogue; Alan He; Alessandra Cervone; Alex Loeb; Alex Zhang; Alexander Fu; Alexander Lisnichenko; Alexander Zhipa; Alexandros Potamianos; Ali Kebarighotbi; Aliakbar Daronkolaei; Alok Parmesh; Amanjot Kaur Samra; Ameen Khan; Amer Rez; Amir Saffari; Amit Agarwalla; Amit Jhindal; Amith Mamidala; Ammar Asmro; Amulya Ballakur; Anand Mishra; Anand Sridharan; Anastasiia Dubinina; Andre Lenz; Andreas Doerr; Andrew Keating; Andrew Leaver; Andrew Smith; Andrew Wirth; Andy Davey; Andy Rosenbaum; Andy Sohn; Angela Chan; Aniket Chakrabarti; Anil Ramakrishna; Anirban Roy; Anita Iyer; Anjali Narayan-Chen; Ankith Yennu; Anna Dabrowska; Anna Gawlowska; Anna Rumshisky; Anna Turek; Anoop Deoras; Anton Bezruchkin; Anup Prasad; Anupam Dewan; Anwith Kiran; Apoorv Gupta; Aram Galstyan; Aravind Manoharan; Arijit Biswas; Arindam Mandal; Arpit Gupta; Arsamkhan Pathan; Arun Nagarajan; Arushan Rajasekaram; Arvind Sundararajan; Ashwin Ganesan; Ashwin Swaminathan; Athanasios Mouchtaris; Audrey Champeau; Avik Ray; Ayush Jaiswal; Ayush Sharma; Bailey Keefer; Balamurugan Muthiah; Beatriz Leon-Millan; Ben Koopman; Ben Li; Benjamin Biggs; Benjamin Ott; Bhanu Vinzamuri; Bharath Venkatesh; Bhavana Ganesh; Bhoomit Vasani; Bill Byrne; Bill Hsu; Bincheng Wang; Blake King; Blazej Gorny; Bo Feng; Bo Zheng; Bodhisattwa Paul; Bofan Sun; Bofeng Luo; Bowen Chen; Bowen Xie; Boya Yu; Brendan Jugan; Brett Panosh; Brian Collins; Brian Thompson; Can Karakus; Can Liu; Carl Lambrecht; Carly Lin; Carolyn Wang; Carrie Yuan; Casey Loyda; Cezary Walczak; Chalapathi Choppa; Chandana Satya Prakash; Chankrisna Richy Meas; Charith Peris; Charles Recaido; Charlie Xu; Charul Sharma; Chase Kernan; Chayut Thanapirom; Chengwei Su; Chenhao Xu; Chenhao Yin; Chentao Ye; Chenyang Tao; Chethan Parameshwara; Ching-Yun Chang; Chong Li; Chris Hench; Chris Tran; Christophe Dupuy; Christopher Davis; Christopher DiPersio; Christos Christodoulopoulos; Christy Li; Chun Chen; Claudio Delli Bovi; Clement Chung; Cole Hawkins; Connor Harris; Corey Ropell; Cynthia He; DK Joo; Dae Yon Hwang; Dan Rosen; Daniel Elkind; Daniel Pressel; Daniel Zhang; Danielle Kimball; Daniil Sorokin; Dave Goodell; Davide Modolo; Dawei Zhu; Deepikaa Suresh; Deepti Ragha; Denis Filimonov; Denis Foo Kune; Denis Romasanta Rodriguez; Devamanyu Hazarika; Dhananjay Ram; Dhawal Parkar; Dhawal Patel; Dhwanil Desai; Dinesh Singh Rajput; Disha Sule; Diwakar Singh; Dmitriy Genzel; Dolly Goldenberg; Dongyi He; Dumitru Hanciu; Dushan Tharmal; Dzmitry Siankovich; Edi Cikovic; Edwin Abraham; Ekraam Sabir; Elliott Olson; Emmett Steven; Emre Barut; Eric Jackson; Ethan Wu; Evelyn Chen; Ezhilan Mahalingam; Fabian Triefenbach; Fan Yang; Fangyu Liu; Fanzi Wu; Faraz Tavakoli; Farhad Khozeimeh; Feiyang Niu; Felix Hieber; Feng Li; Firat Elbey; Florian Krebs; Florian Saupe; Florian Sprünken; Frank Fan; Furqan Khan; Gabriela De Vincenzo; Gagandeep Kang; George Ding; George He; George Yeung; Ghada Qaddoumi; Giannis Karamanolakis; Goeric Huybrechts; Gokul Maddali; Gonzalo Iglesias; Gordon McShane; Gozde Sahin; Guangtai Huang; Gukyeong Kwon; Gunnar A. Sigurdsson; Gurpreet Chadha; Gururaj Kosuru; Hagen Fuerstenau; Hah Hah; Haja Maideen; Hajime Hosokawa; Han Liu; Han-Kai Hsu; Hann Wang; Hao Li; Hao Yang; Haofeng Zhu; Haozheng Fan; Harman Singh; Harshavardhan Kaluvala; Hashim Saeed; He Xie; Helian Feng; Hendrix Luo; Hengzhi Pei; Henrik Nielsen; Hesam Ilati; Himanshu Patel; Hongshan Li; Hongzhou Lin; Hussain Raza; Ian Cullinan; Imre Kiss; Inbarasan Thangamani; Indrayani Fadnavis; Ionut Teodor Sorodoc; Irem Ertuerk; Iryna Yemialyanava; Ishan Soni; Ismail Jelal; Ivan Tse; Jack FitzGerald; Jack Zhao; Jackson Rothgeb; Jacky Lee; Jake Jung; Jakub Debski; Jakub Tomczak; James Jeun; James Sanders; Jason Crowley; Jay Lee; Jayakrishna Anvesh Paidy; Jayant Tiwari; Jean Farmer; Jeff Solinsky; Jenna Lau; Jeremy Savareese; Jerzy Zagorski; Ji Dai; Jiacheng; Gu; Jiahui Li; Jian; Zheng; Jianhua Lu; Jianhua Wang; Jiawei Dai; Jiawei Mo; Jiaxi Xu; Jie Liang; Jie Yang; Jim Logan; Jimit Majmudar; Jing Liu; Jinghong Miao; Jingru Yi; Jingyang Jin; Jiun-Yu Kao; Jixuan Wang; Jiyang Wang; Joe Pemberton; Joel Carlson; Joey Blundell; John Chin-Jew; John He; Jonathan Ho; Jonathan Hueser; Jonathan Lunt; Jooyoung Lee; Joshua Tan; Joyjit Chatterjee; Judith Gaspers; Jue Wang; Jun Fang; Jun Tang; Jun Wan; Jun Wu; Junlei Wang; Junyi Shi; Justin Chiu; Justin Satriano; Justin Yee; Jwala Dhamala; Jyoti Bansal; Kai Zhen; Kai-Wei Chang; Kaixiang Lin; Kalyan Raman; Kanthashree Mysore Sathyendra; Karabo Moroe; Karan Bhandarkar; Karan Kothari; Karolina Owczarzak; Karthick Gopalswamy; Karthick Ravi; Karthik Ramakrishnan; Karthika Arumugam; Kartik Mehta; Katarzyna Konczalska; Kavya Ravikumar; Ke Tran; Kechen Qin; Kelin Li; Kelvin Li; Ketan Kulkarni; Kevin Angelo Rodrigues; Keyur Patel; Khadige Abboud; Kiana Hajebi; Klaus Reiter; Kris Schultz; Krishna Anisetty; Krishna Kotnana; Kristen Li; Kruthi Channamallikarjuna; Krzysztof Jakubczyk; Kuba Pierewoj; Kunal Pal; Kunwar Srivastav; Kyle Bannerman; Lahari Poddar; Lakshmi Prasad; Larry Tseng; Laxmikant Naik; Leena Chennuru Vankadara; Lenon Minorics; Leo Liu; Leonard Lausen; Leonardo F. R. Ribeiro; Li Zhang; Lili Gehorsam; Ling Qi; Lisa Bauer; Lori Knapp; Lu Zeng; Lucas Tong; Lulu Wong; Luoxin Chen; Maciej Rudnicki; Mahdi Namazifar; Mahesh Jaliminche; Maira Ladeira Tanke; Manasi Gupta; Mandeep Ahlawat; Mani Khanuja; Mani Sundaram; Marcin Leyk; Mariusz Momotko; Markus Boese; Markus Dreyer; Markus Mueller; Mason Fu; Mateusz Górski; Mateusz Mastalerczyk; Matias Mora; Matt Johnson; Matt Scott; Matthew Wen; Max Barysau; Maya Boumerdassi; Maya Krishnan; Mayank Gupta; Mayank Hirani; Mayank Kulkarni; Meganathan Narayanasamy; Melanie Bradford; Melanie Gens; Melissa Burke; Meng Jin; Miao Chen; Michael Denkowski; Michael Heymel; Michael Krestyaninov; Michal Obirek; Michalina Wichorowska; Michał Miotk; Milosz Watroba; Mingyi Hong; Mingzhi Yu; Miranda Liu; Mohamed Gouda; Mohammad El-Shabani; Mohammad Ghavamzadeh; Mohit Bansal; Morteza Ziyadi; Nan Xia; Nathan Susanj; Nav Bhasin; Neha Goswami; Nehal Belgamwar; Nicolas Anastassacos; Nicolas Bergeron; Nidhi Jain; Nihal Jain; Niharika Chopparapu; Nik Xu; Nikko Strom; Nikolaos Malandrakis; Nimisha Mishra; Ninad Parkhi; Ninareh Mehrabi; Nishita Sant; Nishtha Gupta; Nitesh Sekhar; Nithin Rajeev; Nithish Raja Chidambaram; Nitish Dhar; Noor Bhagwagar; Noy Konforty; Omar Babu; Omid Razavi; Orchid Majumder; Osama Dar; Oscar Hsu; Pablo Kvitca; Pallavi Pandey; Parker Seegmiller; Patrick Lange; Paul Ferraro; Payal Motwani; Pegah Kharazmi; Pei Wang; Pengfei Liu; Peter Bradtke; Peter Götz; Peter Zhou; Pichao Wang; Piotr Poskart; Pooja Sonawane; Pradeep Natarajan; Pradyun Ramadorai; Pralam Shah; Prasad Nirantar; Prasanthi Chavali; Prashan Wanigasekara; Prashant Saraf; Prashun Dey; Pratyush Pant; Prerak Pradhan; Preyaa Patel; Priyanka Dadlani; Prudhvee Narasimha Sadha; Qi Dong; Qian Hu; Qiaozi; Gao; Qing Liu; Quinn Lam; Quynh Do; R. Manmatha; Rachel Willis; Rafael Liu; Rafal Ellert; Rafal Kalinski; Rafi Al Attrach; Ragha Prasad; Ragini Prasad; Raguvir Kunani; Rahul Gupta; Rahul Sharma; Rahul Tewari; Rajaganesh Baskaran; Rajan Singh; Rajiv Gupta; Rajiv Reddy; Rajshekhar Das; Rakesh Chada; Rakesh Vaideeswaran Mahesh; Ram Chandrasekaran; Ramesh Nallapati; Ran Xue; Rashmi Gangadharaiah; Ravi Rachakonda; Renxian Zhang; Rexhina Blloshmi; Rishabh Agrawal; Robert Enyedi; Robert Lowe; Robik Shrestha; Robinson Piramuthu; Rohail Asad; Rohan Khanna; Rohan Mukherjee; Rohit Mittal; Rohit Prasad; Rohith Mysore Vijaya Kumar; Ron Diamant; Ruchita Gupta; Ruiwen Li; Ruoying Li; Rushabh Fegade; Ruxu Zhang; Ryan Arbow; Ryan Chen; Ryan Gabbard; Ryan Hoium; Ryan King; Sabarishkumar Iyer; Sachal Malick; Sahar Movaghati; Sai Balakavi; Sai Jakka; Sai Kashyap Paruvelli; Sai Muralidhar Jayanthi; Saicharan Shriram Mujumdar; Sainyam Kapoor; Sajjad Beygi; Saket Dingliwal; Saleh Soltan; Sam Ricklin; Sam Tucker; Sameer Sinha; Samridhi Choudhary; Samson Tan; Samuel Broscheit; Samuel Schulter; Sanchit Agarwal; Sandeep Atluri; Sander Valstar; Sanjana Shankar; Sanyukta Sanyukta; Sarthak Khanna; Sarvpriye Khetrapal; Satish Janakiraman; Saumil Shah; Saurabh Akolkar; Saurabh Giri; Saurabh Khandelwal; Saurabh Pawar; Saurabh Sahu; Sean Huang; Sejun Ra; Senthilkumar Gopal; Sergei Dobroshinsky; Shadi Saba; Shamik Roy; Shamit Lal; Shankar Ananthakrishnan; Sharon Li; Shashwat Srijan; Shekhar Bhide; Sheng Long Tang; Sheng Zha; Shereen Oraby; Sherif Mostafa; Shiqi Li; Shishir Bharathi; Shivam Prakash; Shiyuan Huang; Shreya Yembarwar; Shreyas Pansare; Shreyas Subramanian; Shrijeet Joshi; Shuai Liu; Shuai Tang; Shubham Chandak; Shubham Garg; Shubham Katiyar; Shubham Mehta; Shubham Srivastav; Shuo Yang; Siddalingesha D S; Siddharth Choudhary; Siddharth Singh Senger; Simon Babb; Sina Moeini; Siqi Deng; Siva Loganathan; Slawomir Domagala; Sneha Narkar; Sneha Wadhwa; Songyang Zhang; Songyao Jiang; Sony Trenous; Soumajyoti Sarkar; Soumya Saha; Sourabh Reddy; Sourav Dokania; Spurthideepika Sandiri; Spyros Matsoukas; Sravan Bodapati; Sri Harsha Reddy Wdaru; Sridevi Yagati Venkateshdatta; Srikanth Ronanki; Srinivasan R Veeravanallur; Sriram Venkatapathy; Sriramprabhu Sankaraguru; Sruthi Gorantla; Sruthi Karuturi; Stefan Schroedl; Subendhu Rongali; Subhasis Kundu; Suhaila Shakiah; Sukriti Tiwari; Sumit Bharti; Sumita Sami; Sumith Mathew; Sunny Yu; Sunwoo Kim; Suraj Bajirao Malode; Susana Cumplido Riel; Swapnil Palod; Swastik Roy; Syed Furqhan; Tagyoung Chung; Takuma Yoshitani; Taojiannan Yang; Tejaswi Chillakura; Tejwant Bajwa; Temi Lajumoke; Thanh Tran; Thomas Gueudre; Thomas Jung; Tianhui Li; Tim Seemman; Timothy Leffel; Tingting Xiang; Tirth Patel; Tobias Domhan; Tobias Falke; Toby Guo; Tom Li; Tomasz Horszczaruk; Tomasz Jedynak; Tushar Kulkarni; Tyst Marin; Tytus Metrycki; Tzu-Yen Wang; Umang Jain; Upendra Singh; Utkarsh Chirimar; Vaibhav Gupta; Vanshil Shah; Varad Deshpande; Varad Gunjal; Varsha Srikeshava; Varsha Vivek; Varun Bharadwaj; Varun Gangal; Varun Kumar; Venkatesh Elango; Vicente Ordonez; Victor Soto; Vignesh Radhakrishnan; Vihang Patel; Vikram Singh; Vinay Varma Kolanuvada; Vinayshekhar Bannihatti Kumar; Vincent Auvray; Vincent Cartillier; Vincent Ponzo; Violet Peng; Vishal Khandelwal; Vishal Naik; Vishvesh Sahasrabudhe; Vitaliy Korolev; Vivek Gokuladas; Vivek Madan; Vivek Subramanian; Volkan Cevher; Vrinda Gupta; Wael Hamza; Wei Zhang; Weitong Ruan; Weiwei Cheng; Wen Zhang; Wenbo Zhao; Wenyan Yao; Wenzhuo Ouyang; Wesley Dashner; William Campbell; William Lin; Willian Martin; Wyatt Pearson; Xiang Jiang; Xiangxing Lu; Xiangyang Shi; Xianwen Peng; Xiaofeng Gao; Xiaoge Jiang; Xiaohan Fei; Xiaohui Wang; Xiaozhou Joey Zhou; Xin Feng; Xinyan Zhao; Xinyao Wang; Xinyu Li; Xu Zhang; Xuan Wang; Xuandi Fu; Xueling Yuan; Xuning Wang; Yadunandana Rao; Yair Tavizon; Yan Rossiytsev; Yanbei Chen; Yang Liu; Yang Zou; Yangsook Park; Yannick Versley; Yanyan Zhang; Yash Patel; Yen-Cheng Lu; Yi Pan; Yi-Hsiang; Lai; Yichen Hu; Yida Wang; Yiheng Zhou; Yilin Xiang; Ying Shi; Ying Wang; Yishai Galatzer; Yongxin Wang; Yorick Shen; Yuchen Sun; Yudi Purwatama; Yue; Wu; Yue Gu; Yuechun Wang; Yujun Zeng; Yuncong Chen; Yunke Zhou; Yusheng Xie; Yvon Guy; Zbigniew Ambrozinski; Zhaowei Cai; Zhen Zhang; Zheng Wang; Zhenghui Jin; Zhewei Zhao; Zhiheng Li; Zhiheng Luo; Zhikang Zhang; Zhilin Fang; Zhiqi Bu; Zhiyuan Wang; Zhizhong Li; Zijian Wang; Zimeng; Qiu; Zishi Li
>
> **备注:** 48 pages, 10 figures
>
> **摘要:** We present Amazon Nova, a new generation of state-of-the-art foundation models that deliver frontier intelligence and industry-leading price performance. Amazon Nova Pro is a highly-capable multimodal model with the best combination of accuracy, speed, and cost for a wide range of tasks. Amazon Nova Lite is a low-cost multimodal model that is lightning fast for processing images, video, documents and text. Amazon Nova Micro is a text-only model that delivers our lowest-latency responses at very low cost. Amazon Nova Canvas is an image generation model that creates professional grade images with rich customization controls. Amazon Nova Reel is a video generation model offering high-quality outputs, customization, and motion control. Our models were built responsibly and with a commitment to customer trust, security, and reliability. We report benchmarking results for core capabilities, agentic performance, long context, functional adaptation, runtime performance, and human evaluation.
>
---
#### [new 017] Rethinking Hate Speech Detection on Social Media: Can LLMs Replace Traditional Models?
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于 hate speech detection 任务，旨在解决多语言、非正式网络语境下的仇恨言论识别问题。研究构建了 IndoHateMix 数据集，并验证 LLMs 在此任务上的优越性。**

- **链接: [http://arxiv.org/pdf/2506.12744v1](http://arxiv.org/pdf/2506.12744v1)**

> **作者:** Daman Deep Singh; Ramanuj Bhattacharjee; Abhijnan Chakraborty
>
> **摘要:** Hate speech detection across contemporary social media presents unique challenges due to linguistic diversity and the informal nature of online discourse. These challenges are further amplified in settings involving code-mixing, transliteration, and culturally nuanced expressions. While fine-tuned transformer models, such as BERT, have become standard for this task, we argue that recent large language models (LLMs) not only surpass them but also redefine the landscape of hate speech detection more broadly. To support this claim, we introduce IndoHateMix, a diverse, high-quality dataset capturing Hindi-English code-mixing and transliteration in the Indian context, providing a realistic benchmark to evaluate model robustness in complex multilingual scenarios where existing NLP methods often struggle. Our extensive experiments show that cutting-edge LLMs (such as LLaMA-3.1) consistently outperform task-specific BERT-based models, even when fine-tuned on significantly less data. With their superior generalization and adaptability, LLMs offer a transformative approach to mitigating online hate in diverse environments. This raises the question of whether future works should prioritize developing specialized models or focus on curating richer and more varied datasets to further enhance the effectiveness of LLMs.
>
---
#### [new 018] EgoPrivacy: What Your First-Person Camera Says About You?
- **分类: cs.CV; cs.CY**

- **简介: 该论文属于隐私安全任务，旨在研究第一视角视频泄露用户隐私的问题。通过构建EgoPrivacy基准和提出新型攻击方法，揭示了佩戴者隐私信息的高度易泄露性。**

- **链接: [http://arxiv.org/pdf/2506.12258v1](http://arxiv.org/pdf/2506.12258v1)**

> **作者:** Yijiang Li; Genpei Zhang; Jiacheng Cheng; Yi Li; Xiaojun Shan; Dashan Gao; Jiancheng Lyu; Yuan Li; Ning Bi; Nuno Vasconcelos
>
> **备注:** ICML 2025
>
> **摘要:** While the rapid proliferation of wearable cameras has raised significant concerns about egocentric video privacy, prior work has largely overlooked the unique privacy threats posed to the camera wearer. This work investigates the core question: How much privacy information about the camera wearer can be inferred from their first-person view videos? We introduce EgoPrivacy, the first large-scale benchmark for the comprehensive evaluation of privacy risks in egocentric vision. EgoPrivacy covers three types of privacy (demographic, individual, and situational), defining seven tasks that aim to recover private information ranging from fine-grained (e.g., wearer's identity) to coarse-grained (e.g., age group). To further emphasize the privacy threats inherent to egocentric vision, we propose Retrieval-Augmented Attack, a novel attack strategy that leverages ego-to-exo retrieval from an external pool of exocentric videos to boost the effectiveness of demographic privacy attacks. An extensive comparison of the different attacks possible under all threat models is presented, showing that private information of the wearer is highly susceptible to leakage. For instance, our findings indicate that foundation models can effectively compromise wearer privacy even in zero-shot settings by recovering attributes such as identity, scene, gender, and race with 70-80% accuracy. Our code and data are available at https://github.com/williamium3000/ego-privacy.
>
---
#### [new 019] Risks & Benefits of LLMs & GenAI for Platform Integrity, Healthcare Diagnostics, Cybersecurity, Privacy & AI Safety: A Comprehensive Survey, Roadmap & Implementation Blueprint
- **分类: cs.CR; cs.CY**

- **简介: 该论文属于AI安全与风险管理任务，旨在分析LLMs和GenAI在平台完整性、医疗诊断等领域的风险与收益，并提出应对策略与实施蓝图。**

- **链接: [http://arxiv.org/pdf/2506.12088v1](http://arxiv.org/pdf/2506.12088v1)**

> **作者:** Kiarash Ahi
>
> **摘要:** Large Language Models (LLMs) and generative AI (GenAI) systems such as ChatGPT, Claude, Gemini, LLaMA, and Copilot, developed by OpenAI, Anthropic, Google, Meta, and Microsoft are reshaping digital platforms and app ecosystems while introducing key challenges in cybersecurity, privacy, and platform integrity. Our analysis shows alarming trends: LLM-assisted malware is projected to rise from 2% in 2021 to 50% by 2025; AI-generated Google reviews grew from 1.2% in 2021 to 12.21% in 2023, with an expected 30% by 2025; AI scam reports surged 456%; and misinformation sites increased over 1500%, with a 50-60% increase in deepfakes in 2024. Concurrently, as LLMs have facilitated code development, mobile app submissions grew from 1.8 million in 2020 to 3.0 million in 2024, with 3.6 million expected by 2025. To address AI threats, platforms from app stores like Google Play and Apple to developer hubs like GitHub Copilot, and social platforms like TikTok and Facebook, to marketplaces like Amazon are deploying AI and LLM-based defenses. This highlights the dual nature of these technologies as both the source of new threats and the essential tool for their mitigation. Integrating LLMs into clinical diagnostics also raises concerns about accuracy, bias, and safety, needing strong governance. Drawing on a comprehensive analysis of 455 references, this paper presents a survey of LLM and GenAI risks. We propose a strategic roadmap and operational blueprint integrating policy auditing (CCPA, GDPR), fraud detection, and compliance automation, and an advanced LLM-DA stack with modular components including multi LLM routing, agentic memory, and governance layers to enhance platform integrity. We also provide actionable insights, cross-functional best practices, and real-world case studies. These contributions offer paths to scalable trust, safety, and responsible innovation across digital platforms.
>
---
#### [new 020] Large Language Models for History, Philosophy, and Sociology of Science: Interpretive Uses, Methodological Challenges, and Critical Perspectives
- **分类: cs.CL; cs.AI; cs.CY; A.1; I.2.1; I.2.7; J.4; J.5**

- **链接: [http://arxiv.org/pdf/2506.12242v1](http://arxiv.org/pdf/2506.12242v1)**

> **作者:** Arno Simons; Michael Zichert; Adrian Wüthrich
>
> **备注:** 27 pages, 2 tables
>
> **摘要:** This paper explores the use of large language models (LLMs) as research tools in the history, philosophy, and sociology of science (HPSS). LLMs are remarkably effective at processing unstructured text and inferring meaning from context, offering new affordances that challenge long-standing divides between computational and interpretive methods. This raises both opportunities and challenges for HPSS, which emphasizes interpretive methodologies and understands meaning as context-dependent, ambiguous, and historically situated. We argue that HPSS is uniquely positioned not only to benefit from LLMs' capabilities but also to interrogate their epistemic assumptions and infrastructural implications. To this end, we first offer a concise primer on LLM architectures and training paradigms tailored to non-technical readers. We frame LLMs not as neutral tools but as epistemic infrastructures that encode assumptions about meaning, context, and similarity, conditioned by their training data, architecture, and patterns of use. We then examine how computational techniques enhanced by LLMs, such as structuring data, detecting patterns, and modeling dynamic processes, can be applied to support interpretive research in HPSS. Our analysis compares full-context and generative models, outlines strategies for domain and task adaptation (e.g., continued pretraining, fine-tuning, and retrieval-augmented generation), and evaluates their respective strengths and limitations for interpretive inquiry in HPSS. We conclude with four lessons for integrating LLMs into HPSS: (1) model selection involves interpretive trade-offs; (2) LLM literacy is foundational; (3) HPSS must define its own benchmarks and corpora; and (4) LLMs should enhance, not replace, interpretive methods.
>
---
#### [new 021] Delving Into the Psychology of Machines: Exploring the Structure of Self-Regulated Learning via LLM-Generated Survey Responses
- **分类: cs.AI; cs.CY; stat.ME; stat.OT**

- **简介: 该论文属于心理测量任务，旨在评估LLM生成的调查回应是否有效模拟自我调节学习数据。研究比较了多个LLM生成的MSLQ问卷数据，分析其结构和有效性。**

- **链接: [http://arxiv.org/pdf/2506.13384v1](http://arxiv.org/pdf/2506.13384v1)**

> **作者:** Leonie V. D. E. Vogelsmeier; Eduardo Oliveira; Kamila Misiejuk; Sonsoles López-Pernas; Mohammed Saqr
>
> **摘要:** Large language models (LLMs) offer the potential to simulate human-like responses and behaviors, creating new opportunities for psychological science. In the context of self-regulated learning (SRL), if LLMs can reliably simulate survey responses at scale and speed, they could be used to test intervention scenarios, refine theoretical models, augment sparse datasets, and represent hard-to-reach populations. However, the validity of LLM-generated survey responses remains uncertain, with limited research focused on SRL and existing studies beyond SRL yielding mixed results. Therefore, in this study, we examined LLM-generated responses to the 44-item Motivated Strategies for Learning Questionnaire (MSLQ; Pintrich \& De Groot, 1990), a widely used instrument assessing students' learning strategies and academic motivation. Particularly, we used the LLMs GPT-4o, Claude 3.7 Sonnet, Gemini 2 Flash, LLaMA 3.1-8B, and Mistral Large. We analyzed item distributions, the psychological network of the theoretical SRL dimensions, and psychometric validity based on the latent factor structure. Our results suggest that Gemini 2 Flash was the most promising LLM, showing considerable sampling variability and producing underlying dimensions and theoretical relationships that align with prior theory and empirical findings. At the same time, we observed discrepancies and limitations, underscoring both the potential and current constraints of using LLMs for simulating psychological survey data and applying it in educational contexts.
>
---
#### [new 022] Rethinking Test-Time Scaling for Medical AI: Model and Task-Aware Strategies for LLMs and VLMs
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于医疗AI领域，研究测试时缩放技术在大语言模型和视觉语言模型中的应用，旨在提升模型推理能力并解决可靠性与可解释性问题。**

- **链接: [http://arxiv.org/pdf/2506.13102v1](http://arxiv.org/pdf/2506.13102v1)**

> **作者:** Gyutaek Oh; Seoyeon Kim; Sangjoon Park; Byung-Hoon Kim
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Test-time scaling has recently emerged as a promising approach for enhancing the reasoning capabilities of large language models or vision-language models during inference. Although a variety of test-time scaling strategies have been proposed, and interest in their application to the medical domain is growing, many critical aspects remain underexplored, including their effectiveness for vision-language models and the identification of optimal strategies for different settings. In this paper, we conduct a comprehensive investigation of test-time scaling in the medical domain. We evaluate its impact on both large language models and vision-language models, considering factors such as model size, inherent model characteristics, and task complexity. Finally, we assess the robustness of these strategies under user-driven factors, such as misleading information embedded in prompts. Our findings offer practical guidelines for the effective use of test-time scaling in medical applications and provide insights into how these strategies can be further refined to meet the reliability and interpretability demands of the medical domain.
>
---
#### [new 023] Regulating Next-Generation Implantable Brain-Computer Interfaces: Recommendations for Ethical Development and Implementation
- **分类: cs.HC; cs.CY; cs.ET**

- **简介: 该论文属于伦理与技术监管任务，旨在解决下一代脑机接口的伦理问题，通过提出九项开发建议和九项政策建议，推动其安全、伦理应用。**

- **链接: [http://arxiv.org/pdf/2506.12540v1](http://arxiv.org/pdf/2506.12540v1)**

> **作者:** Renee Sirbu; Jessica Morley; Tyler Schroder; Mariarosaria Taddeo; Raghavendra Pradyumna Pothukuchi; Muhammed Ugur; Abhishek Bhattacharjee; Luciano Floridi
>
> **备注:** 35 pages, 3 tables, 2 appendices
>
> **摘要:** Brain-computer interfaces offer significant therapeutic opportunities for a variety of neurophysiological and neuropsychiatric disorders and may perhaps one day lead to augmenting the cognition and decision-making of the healthy brain. However, existing regulatory frameworks designed for implantable medical devices are inadequate to address the unique ethical, legal, and social risks associated with next-generation networked brain-computer interfaces. In this article, we make nine recommendations to support developers in the design of BCIs and nine recommendations to support policymakers in the application of BCIs, drawing insights from the regulatory history of IMDs and principles from AI ethics. We begin by outlining the historical development of IMDs and the regulatory milestones that have shaped their oversight. Next, we summarize similarities between IMDs and emerging implantable BCIs, identifying existing provisions for their regulation. We then use two case studies of emerging cutting-edge BCIs, the HALO and SCALO computer systems, to highlight distinctive features in the design and application of next-generation BCIs arising from contemporary chip architectures, which necessitate reevaluating regulatory approaches. We identify critical ethical considerations for these BCIs, including unique conceptions of autonomy, identity, and mental privacy. Based on these insights, we suggest potential avenues for the ethical regulation of BCIs, emphasizing the importance of interdisciplinary collaboration and proactive mitigation of potential harms. The goal is to support the responsible design and application of new BCIs, ensuring their safe and ethical integration into medical practice.
>
---
#### [new 024] Feeling Machines: Ethics, Culture, and the Rise of Emotional AI
- **分类: cs.HC; cs.AI; cs.CY**

- **简介: 该论文属于伦理与技术交叉研究，探讨情感AI的伦理、文化影响及风险，提出监管与设计建议。**

- **链接: [http://arxiv.org/pdf/2506.12437v1](http://arxiv.org/pdf/2506.12437v1)**

> **作者:** Vivek Chavan; Arsen Cenaj; Shuyuan Shen; Ariane Bar; Srishti Binwani; Tommaso Del Becaro; Marius Funk; Lynn Greschner; Roberto Hung; Stina Klein; Romina Kleiner; Stefanie Krause; Sylwia Olbrych; Vishvapalsinhji Parmar; Jaleh Sarafraz; Daria Soroko; Daksitha Withanage Don; Chang Zhou; Hoang Thuy Duong Vu; Parastoo Semnani; Daniel Weinhardt; Elisabeth Andre; Jörg Krüger; Xavier Fresquet
>
> **备注:** From the Spring School 2025 by AI Grid and SCAI (Sorbonne University), 16 pages
>
> **摘要:** This paper explores the growing presence of emotionally responsive artificial intelligence through a critical and interdisciplinary lens. Bringing together the voices of early-career researchers from multiple fields, it explores how AI systems that simulate or interpret human emotions are reshaping our interactions in areas such as education, healthcare, mental health, caregiving, and digital life. The analysis is structured around four central themes: the ethical implications of emotional AI, the cultural dynamics of human-machine interaction, the risks and opportunities for vulnerable populations, and the emerging regulatory, design, and technical considerations. The authors highlight the potential of affective AI to support mental well-being, enhance learning, and reduce loneliness, as well as the risks of emotional manipulation, over-reliance, misrepresentation, and cultural bias. Key challenges include simulating empathy without genuine understanding, encoding dominant sociocultural norms into AI systems, and insufficient safeguards for individuals in sensitive or high-risk contexts. Special attention is given to children, elderly users, and individuals with mental health challenges, who may interact with AI in emotionally significant ways. However, there remains a lack of cognitive or legal protections which are necessary to navigate such engagements safely. The report concludes with ten recommendations, including the need for transparency, certification frameworks, region-specific fine-tuning, human oversight, and longitudinal research. A curated supplementary section provides practical tools, models, and datasets to support further work in this domain.
>
---
#### [new 025] Prosocial Design in Trust and Safety
- **分类: cs.HC; cs.CY; cs.SI; econ.GN; q-fin.EC; J.4; K.4.1**

- **简介: 本文探讨Prosocial Design在信任与安全中的应用，旨在通过设计促进积极行为。属于平台设计与治理任务，解决有害行为与虚假信息传播问题。**

- **链接: [http://arxiv.org/pdf/2506.12792v1](http://arxiv.org/pdf/2506.12792v1)**

> **作者:** David Grüning; Julia Kamin
>
> **备注:** 29 pages, no figures, to be published in "T&S Past, Present, and Future."
>
> **摘要:** This chapter presents an overview of Prosocial Design, an approach to platform design and governance that recognizes design choices influence behavior and that those choices can or should be made toward supporting healthy interactions and other prosocial outcomes. The authors discuss several core principles of Prosocial Design and its relationship to Trust and Safety and other related fields. As a primary contribution, the chapter reviews relevant research to demonstrate how Prosocial Design can be an effective approach to reducing rule-breaking and other harmful behavior and how it can help to stem the spread of harmful misinformation. Prosocial Design is a nascent and evolving field and research is still limited. The authors hope this chapter will not only inspire more research and the adoption of a prosocial design approach, but that it will also provoke discussion about the principles of Prosocial Design and its potential to support Trust and Safety.
>
---
#### [new 026] Organizational Adaptation to Generative AI in Cybersecurity: A Systematic Review
- **分类: cs.CR; cs.AI; cs.CY; K.6.5; I.2.0; K.4.1**

- **简介: 该论文属于系统综述任务，探讨网络安全组织如何适应生成式AI的集成，分析其框架与流程的变革，提出适应模式与挑战。**

- **链接: [http://arxiv.org/pdf/2506.12060v1](http://arxiv.org/pdf/2506.12060v1)**

> **作者:** Christopher Nott
>
> **备注:** 38 pages, 1 table, 1 figure
>
> **摘要:** Cybersecurity organizations are adapting to GenAI integration through modified frameworks and hybrid operational processes, with success influenced by existing security maturity, regulatory requirements, and investments in human capital and infrastructure. This qualitative research employs systematic document analysis and comparative case study methodology to examine how cybersecurity organizations adapt their threat modeling frameworks and operational processes to address generative artificial intelligence integration. Through examination of 25 studies from 2022 to 2025, the research documents substantial transformation in organizational approaches to threat modeling, moving from traditional signature-based systems toward frameworks incorporating artificial intelligence capabilities. The research identifies three primary adaptation patterns: Large Language Model integration for security applications, GenAI frameworks for risk detection and response automation, and AI/ML integration for threat hunting. Organizations with mature security infrastructures, particularly in finance and critical infrastructure sectors, demonstrate higher readiness through structured governance approaches, dedicated AI teams, and robust incident response processes. Organizations achieve successful GenAI integration when they maintain appropriate human oversight of automated systems, address data quality concerns and explainability requirements, and establish governance frameworks tailored to their specific sectors. Organizations encounter ongoing difficulties with privacy protection, bias reduction, personnel training, and defending against adversarial attacks. This work advances understanding of how organizations adopt innovative technologies in high-stakes environments and offers actionable insights for cybersecurity professionals implementing GenAI systems.
>
---
#### [new 027] Governments Should Mandate Tiered Anonymity on Social-Media Platforms to Counter Deepfakes and LLM-Driven Mass Misinformation
- **分类: cs.SI; cs.CY**

- **简介: 该论文属于信息治理任务，旨在解决深度伪造和AI生成虚假信息问题。提出三级匿名框架，按用户影响力分级管理，以平衡隐私与责任。**

- **链接: [http://arxiv.org/pdf/2506.12814v1](http://arxiv.org/pdf/2506.12814v1)**

> **作者:** David Khachaturov; Roxanne Schnyder; Robert Mullins
>
> **摘要:** This position paper argues that governments should mandate a three-tier anonymity framework on social-media platforms as a reactionary measure prompted by the ease-of-production of deepfakes and large-language-model-driven misinformation. The tiers are determined by a given user's $\textit{reach score}$: Tier 1 permits full pseudonymity for smaller accounts, preserving everyday privacy; Tier 2 requires private legal-identity linkage for accounts with some influence, reinstating real-world accountability at moderate reach; Tier 3 would require per-post, independent, ML-assisted fact-checking, review for accounts that would traditionally be classed as sources-of-mass-information. An analysis of Reddit shows volunteer moderators converge on comparable gates as audience size increases -- karma thresholds, approval queues, and identity proofs -- demonstrating operational feasibility and social legitimacy. Acknowledging that existing engagement incentives deter voluntary adoption, we outline a regulatory pathway that adapts existing US jurisprudence and recent EU-UK safety statutes to embed reach-proportional identity checks into existing platform tooling, thereby curbing large-scale misinformation while preserving everyday privacy.
>
---
#### [new 028] Rethinking Optimization: A Systems-Based Approach to Social Externalities
- **分类: cs.AI; cs.CY**

- **简介: 该论文属于优化问题研究，旨在解决优化实践中因忽视外部性导致的负面后果。通过结合系统思维与外部性理论，提出框架以识别受影响方并改进优化过程。**

- **链接: [http://arxiv.org/pdf/2506.12825v1](http://arxiv.org/pdf/2506.12825v1)**

> **作者:** Pegah Nokhiz; Aravinda Kanchana Ruwanpathirana; Helen Nissenbaum
>
> **摘要:** Optimization is widely used for decision making across various domains, valued for its ability to improve efficiency. However, poor implementation practices can lead to unintended consequences, particularly in socioeconomic contexts where externalities (costs or benefits to third parties outside the optimization process) are significant. To propose solutions, it is crucial to first characterize involved stakeholders, their goals, and the types of subpar practices causing unforeseen outcomes. This task is complex because affected stakeholders often fall outside the direct focus of optimization processes. Also, incorporating these externalities into optimization requires going beyond traditional economic frameworks, which often focus on describing externalities but fail to address their normative implications or interconnected nature, and feedback loops. This paper suggests a framework that combines systems thinking with the economic concept of externalities to tackle these challenges. This approach aims to characterize what went wrong, who was affected, and how (or where) to include them in the optimization process. Economic externalities, along with their established quantification methods, assist in identifying "who was affected and how" through stakeholder characterization. Meanwhile, systems thinking (an analytical approach to comprehending relationships in complex systems) provides a holistic, normative perspective. Systems thinking contributes to an understanding of interconnections among externalities, feedback loops, and determining "when" to incorporate them in the optimization. Together, these approaches create a comprehensive framework for addressing optimization's unintended consequences, balancing descriptive accuracy with normative objectives. Using this, we examine three common types of subpar practices: ignorance, error, and prioritization of short-term goals.
>
---
#### [new 029] Energy-Efficient Green AI Architectures for Circular Economies Through Multi-Layered Sustainable Resource Optimization Framework
- **分类: cs.LG; cs.CE; cs.CY**

- **简介: 该论文属于绿色AI任务，旨在解决资源可持续利用问题。通过多层框架优化能源与资源，提升回收效率并减少能耗。**

- **链接: [http://arxiv.org/pdf/2506.12262v1](http://arxiv.org/pdf/2506.12262v1)**

> **作者:** Ripal Ranpara
>
> **摘要:** In this research paper, we propose a new type of energy-efficient Green AI architecture to support circular economies and address the contemporary challenge of sustainable resource consumption in modern systems. We introduce a multi-layered framework and meta-architecture that integrates state-of-the-art machine learning algorithms, energy-conscious computational models, and optimization techniques to facilitate decision-making for resource reuse, waste reduction, and sustainable production.We tested the framework on real-world datasets from lithium-ion battery recycling and urban waste management systems, demonstrating its practical applicability. Notably, the key findings of this study indicate a 25 percent reduction in energy consumption during workflows compared to traditional methods and an 18 percent improvement in resource recovery efficiency. Quantitative optimization was based on mathematical models such as mixed-integer linear programming and lifecycle assessments. Moreover, AI algorithms improved classification accuracy on urban waste by 20 percent, while optimized logistics reduced transportation emissions by 30 percent. We present graphical analyses and visualizations of the developed framework, illustrating its impact on energy efficiency and sustainability as reflected in the simulation results. This paper combines the principles of Green AI with practical insights into how such architectural models contribute to circular economies, presenting a fully scalable and scientifically rooted solution aligned with applicable UN Sustainability Goals worldwide. These results open avenues for incorporating newly developed AI technologies into sustainable management strategies, potentially safeguarding local natural capital while advancing technological progress.
>
---
#### [new 030] A Game-Theoretic Negotiation Framework for Cross-Cultural Consensus in LLMs
- **分类: cs.AI; cs.CY; cs.GT**

- **简介: 该论文属于AI伦理任务，旨在解决LLMs中的文化偏见问题。通过博弈论框架促进跨文化共识，减少WEIRD偏差。**

- **链接: [http://arxiv.org/pdf/2506.13245v1](http://arxiv.org/pdf/2506.13245v1)**

> **作者:** Guoxi Zhang; Jiawei Chen; Tianzhuo Yang; Jiaming Ji; Yaodong Yang; Juntao Dai
>
> **摘要:** The increasing prevalence of large language models (LLMs) is influencing global value systems. However, these models frequently exhibit a pronounced WEIRD (Western, Educated, Industrialized, Rich, Democratic) cultural bias due to lack of attention to minority values. This monocultural perspective may reinforce dominant values and marginalize diverse cultural viewpoints, posing challenges for the development of equitable and inclusive AI systems. In this work, we introduce a systematic framework designed to boost fair and robust cross-cultural consensus among LLMs. We model consensus as a Nash Equilibrium and employ a game-theoretic negotiation method based on Policy-Space Response Oracles (PSRO) to simulate an organized cross-cultural negotiation process. To evaluate this approach, we construct regional cultural agents using data transformed from the World Values Survey (WVS). Beyond the conventional model-level evaluation method, We further propose two quantitative metrics, Perplexity-based Acceptence and Values Self-Consistency, to assess consensus outcomes. Experimental results indicate that our approach generates consensus of higher quality while ensuring more balanced compromise compared to baselines. Overall, it mitigates WEIRD bias by guiding agents toward convergence through fair and gradual negotiation steps.
>
---
#### [new 031] Modelado y gemelos digitales en el contexto fotovoltaico
- **分类: physics.soc-ph; cs.CY**

- **简介: 该论文属于光伏领域研究，旨在解决系统优化与管理问题，通过数字孪生技术实现太阳能设施的实时模拟与性能提升。**

- **链接: [http://arxiv.org/pdf/2506.12102v1](http://arxiv.org/pdf/2506.12102v1)**

> **作者:** Franco Bertani Matung; Juan Cruz Esquembre Santamaría; Ricardo R. Palma; Fabricio Orlando Sanchez Varretti
>
> **备注:** in Spanish language
>
> **摘要:** The photovoltaic industry faces the challenge of optimizing the performance and management of its systems in an increasingly digitalized environment. In this context, digital twins offer an innovative solution: virtual models that replicate in real time the behavior of solar installations. This technology makes it possible to anticipate failures, improve operational efficiency and facilitate data-driven decision-making. This report analyzes its application in the photovoltaic sector, highlighting its benefits and transformative potential.
>
---
#### [new 032] Modeling Earth-Scale Human-Like Societies with One Billion Agents
- **分类: cs.MA; cs.AI; cs.CL; cs.CY; cs.SI**

- **简介: 该论文属于社会模拟任务，旨在解决大规模人类社会行为建模问题。工作是构建Light Society框架，利用LLM实现高效、高保真的人类社会仿真。**

- **链接: [http://arxiv.org/pdf/2506.12078v1](http://arxiv.org/pdf/2506.12078v1)**

> **作者:** Haoxiang Guan; Jiyan He; Liyang Fan; Zhenzhen Ren; Shaobin He; Xin Yu; Yuan Chen; Shuxin Zheng; Tie-Yan Liu; Zhen Liu
>
> **备注:** Work in progress
>
> **摘要:** Understanding how complex societal behaviors emerge from individual cognition and interactions requires both high-fidelity modeling of human behavior and large-scale simulations. Traditional agent-based models (ABMs) have been employed to study these dynamics for decades, but are constrained by simplified agent behaviors that fail to capture human complexity. Recent advances in large language models (LLMs) offer new opportunities by enabling agents to exhibit sophisticated social behaviors that go beyond rule-based logic, yet face significant scaling challenges. Here we present Light Society, an agent-based simulation framework that advances both fronts, efficiently modeling human-like societies at planetary scale powered by LLMs. Light Society formalizes social processes as structured transitions of agent and environment states, governed by a set of LLM-powered simulation operations, and executed through an event queue. This modular design supports both independent and joint component optimization, supporting efficient simulation of societies with over one billion agents. Large-scale simulations of trust games and opinion propagation--spanning up to one billion agents--demonstrate Light Society's high fidelity and efficiency in modeling social trust and information diffusion, while revealing scaling laws whereby larger simulations yield more stable and realistic emergent behaviors.
>
---
#### [new 033] Towards Fairness Assessment of Dutch Hate Speech Detection
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于 hate speech 检测任务，旨在评估荷兰语模型的公平性。通过生成反事实数据并改进模型，提升检测效果与公平性。**

- **链接: [http://arxiv.org/pdf/2506.12502v1](http://arxiv.org/pdf/2506.12502v1)**

> **作者:** Julie Bauer; Rishabh Kaushal; Thales Bertaglia; Adriana Iamnitchi
>
> **备注:** Accepted for publication at the 9th Workshop on Online Abuse and Harms (WOAH) held in conjunction with ACL 2025
>
> **摘要:** Numerous studies have proposed computational methods to detect hate speech online, yet most focus on the English language and emphasize model development. In this study, we evaluate the counterfactual fairness of hate speech detection models in the Dutch language, specifically examining the performance and fairness of transformer-based models. We make the following key contributions. First, we curate a list of Dutch Social Group Terms that reflect social context. Second, we generate counterfactual data for Dutch hate speech using LLMs and established strategies like Manual Group Substitution (MGS) and Sentence Log-Likelihood (SLL). Through qualitative evaluation, we highlight the challenges of generating realistic counterfactuals, particularly with Dutch grammar and contextual coherence. Third, we fine-tune baseline transformer-based models with counterfactual data and evaluate their performance in detecting hate speech. Fourth, we assess the fairness of these models using Counterfactual Token Fairness (CTF) and group fairness metrics, including equality of odds and demographic parity. Our analysis shows that models perform better in terms of hate speech detection, average counterfactual fairness and group fairness. This work addresses a significant gap in the literature on counterfactual fairness for hate speech detection in Dutch and provides practical insights and recommendations for improving both model performance and fairness.
>
---
## 更新

#### [replaced 001] KI4Demokratie: An AI-Based Platform for Monitoring and Fostering Democratic Discourse
- **分类: cs.CY; cs.SI**

- **链接: [http://arxiv.org/pdf/2506.09947v2](http://arxiv.org/pdf/2506.09947v2)**

> **作者:** Rudy Alexandro Garrido Veliz; Till Nikolaus Schaland; Simon Bergmoser; Florian Horwege; Somya Bansal; Ritesh Nahar; Martin Semmann; Jörg Forthmann; Seid Muhie Yimam
>
> **摘要:** Social media increasingly fuel extremism, especially right-wing extremism, and enable the rapid spread of antidemocratic narratives. Although AI and data science are often leveraged to manipulate political opinion, there is a critical need for tools that support effective monitoring without infringing on freedom of expression. We present KI4Demokratie, an AI-based platform that assists journalists, researchers, and policymakers in monitoring right-wing discourse that may undermine democratic values. KI4Demokratie applies machine learning models to a large-scale German online data gathered on a daily basis, providing a comprehensive view of trends in the German digital sphere. Early analysis reveals both the complexity of tracking organized extremist behavior and the promise of our integrated approach, especially during key events.
>
---
#### [replaced 002] What do Large Language Models Say About Animals? Investigating Risks of Animal Harm in Generated Text
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.04804v3](http://arxiv.org/pdf/2503.04804v3)**

> **作者:** Arturs Kanepajs; Aditi Basu; Sankalpa Ghose; Constance Li; Akshat Mehta; Ronak Mehta; Samuel David Tucker-Davis; Eric Zhou; Bob Fischer; Jacy Reese Anthis
>
> **摘要:** As machine learning systems become increasingly embedded in society, their impact on human and nonhuman life continues to escalate. Technical evaluations have addressed a variety of potential harms from large language models (LLMs) towards humans and the environment, but there is little empirical work regarding harms towards nonhuman animals. Following the growing recognition of animal protection in regulatory and ethical AI frameworks, we present AnimalHarmBench (AHB), a benchmark for risks of animal harm in LLM-generated text. Our benchmark dataset comprises 1,850 curated questions from Reddit post titles and 2,500 synthetic questions based on 50 animal categories (e.g., cats, reptiles) and 50 ethical scenarios with a 70-30 public-private split. Scenarios include open-ended questions about how to treat animals, practical scenarios with potential animal harm, and willingness-to-pay measures for the prevention of animal harm. Using the LLM-as-a-judge framework, responses are evaluated for their potential to increase or decrease harm, and evaluations are debiased for the tendency of judges to judge their own outputs more favorably. AHB reveals significant differences across frontier LLMs, animal categories, scenarios, and subreddits. We conclude with future directions for technical research and addressing the challenges of building evaluations on complex social and moral topics.
>
---
#### [replaced 003] Roadmap on Incentive Compatibility for AI Alignment and Governance in Sociotechnical Systems
- **分类: cs.AI; cs.CY; cs.GT; cs.HC; I.2.m; K.4.m**

- **链接: [http://arxiv.org/pdf/2402.12907v3](http://arxiv.org/pdf/2402.12907v3)**

> **作者:** Zhaowei Zhang; Fengshuo Bai; Mingzhi Wang; Haoyang Ye; Chengdong Ma; Yaodong Yang
>
> **摘要:** The burgeoning integration of artificial intelligence (AI) into human society brings forth significant implications for societal governance and safety. While considerable strides have been made in addressing AI alignment challenges, existing methodologies primarily focus on technical facets, often neglecting the intricate sociotechnical nature of AI systems, which can lead to a misalignment between the development and deployment contexts. To this end, we posit a new problem worth exploring: Incentive Compatibility Sociotechnical Alignment Problem (ICSAP). We hope this can call for more researchers to explore how to leverage the principles of Incentive Compatibility (IC) from game theory to bridge the gap between technical and societal components to maintain AI consensus with human societies in different contexts. We further discuss three classical game problems for achieving IC: mechanism design, contract theory, and Bayesian persuasion, in addressing the perspectives, potentials, and challenges of solving ICSAP, and provide preliminary implementation conceptions.
>
---
#### [replaced 004] Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Worse
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2410.21333v4](http://arxiv.org/pdf/2410.21333v4)**

> **作者:** Ryan Liu; Jiayi Geng; Addison J. Wu; Ilia Sucholutsky; Tania Lombrozo; Thomas L. Griffiths
>
> **摘要:** Chain-of-thought (CoT) prompting has become a widely used strategy for improving large language and multimodal model performance. However, it is still an open question under which settings CoT systematically reduces performance. In this paper, we seek to identify the characteristics of tasks where CoT reduces performance by drawing inspiration from cognitive psychology, focusing on six representative tasks from the psychological literature where deliberation hurts performance in humans. In three of these tasks, state-of-the-art models exhibit significant performance drop-offs with CoT (up to 36.3\% absolute accuracy for OpenAI o1-preview compared to GPT-4o), while in others, CoT effects are mixed, with positive, neutral, and negative changes. While models and humans do not exhibit perfectly parallel cognitive processes, considering cases where thinking has negative consequences for humans helps identify settings where it negatively impacts models. By connecting the literature on human verbal thinking and deliberation with evaluations of CoT, we offer a perspective for understanding the impact of inference-time reasoning.
>
---
#### [replaced 005] Transparency in Healthcare AI: Testing European Regulatory Provisions against Users' Transparency Needs
- **分类: cs.CY; cs.AI; K.4.1; J.3**

- **链接: [http://arxiv.org/pdf/2505.17105v2](http://arxiv.org/pdf/2505.17105v2)**

> **作者:** Anna Spagnolli; Cecilia Tolomini; Elisa Beretta; Claudio Sarra
>
> **备注:** 22 pages, pre-review version
>
> **摘要:** Artificial Intelligence (AI) plays an essential role in healthcare and is pervasively incorporated into medical software and equipment. In the European Union, healthcare is a high-risk application domain for AI, and providers must prepare Instructions for Use (IFU) according to the European regulation 2024/1689 (AI Act). To this regulation, the principle of transparency is cardinal and requires the IFU to be clear and relevant to the users. This study tests whether these latter requirements are satisfied by the IFU structure. A survey was administered online via the Qualtrics platform to four types of direct stakeholders, i.e., managers (N = 238), healthcare professionals (N = 115), patients (N = 229), and Information Technology experts (N = 230). The participants rated the relevance of a set of transparency needs and indicated the IFU section addressing them. The results reveal differentiated priorities across stakeholders and a troubled mapping of transparency needs onto the IFU structure. Recommendations to build a locally meaningful IFU are derived.
>
---
#### [replaced 006] LLMs and Childhood Safety: Identifying Risks and Proposing a Protection Framework for Safe Child-LLM Interaction
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2502.11242v4](http://arxiv.org/pdf/2502.11242v4)**

> **作者:** Junfeng Jiao; Saleh Afroogh; Kevin Chen; Abhejay Murali; David Atkinson; Amit Dhurandhar
>
> **摘要:** This study examines the growing use of Large Language Models (LLMs) in child-centered applications, highlighting safety and ethical concerns such as bias, harmful content, and cultural insensitivity. Despite their potential to enhance learning, there is a lack of standardized frameworks to mitigate these risks. Through a systematic literature review, we identify key parental and empirical concerns, including toxicity and ethical breaches in AI outputs. Moreover, to address these issues, this paper proposes a protection framework for safe Child-LLM interaction, incorporating metrics for content safety, behavioral ethics, and cultural sensitivity. The framework provides practical tools for evaluating LLM safety, offering guidance for developers, policymakers, and educators to ensure responsible AI deployment for children.
>
---
#### [replaced 007] The Urban Model Platform: A Public Backbone for Modeling and Simulation in Urban Digital Twins
- **分类: cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.10964v2](http://arxiv.org/pdf/2506.10964v2)**

> **作者:** Rico H Herzog; Till Degkwitz; Trivik Verma
>
> **摘要:** Urban digital twins are increasingly perceived as a way to pool the growing digital resources of cities for the purpose of a more sustainable and integrated urban planning. Models and simulations are central to this undertaking: They enable "what if?" scenarios, create insights and describe relationships between the vast data that is being collected. However, the process of integrating and subsequently using models in urban digital twins is an inherently complex undertaking. It raises questions about how to represent urban complexity, how to deal with uncertain assUrban Model Platformtions and modeling paradigms, and how to capture underlying power relations. Existent approaches in the domain largely focus on monolithic and centralized solutions in the tradition of neoliberal city-making, oftentimes prohibiting pluralistic and open interoperable models. Using a participatory design for participatory systems approach together with the City of Hamburg, Germany, we find that an open Urban Model Platform can function both as a public technological backbone for modeling and simulation in urban digital twins and as a socio-technical framework for a collaborative and pluralistic representation of urban processes. Such a platform builds on open standards, allows for a decentralized integration of models, enables communication between models and supports a multi-model approach to representing urban systems.
>
---
#### [replaced 008] Navigating LLM Ethics: Advancements, Challenges, and Future Directions
- **分类: cs.CY; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.18841v5](http://arxiv.org/pdf/2406.18841v5)**

> **作者:** Junfeng Jiao; Saleh Afroogh; Yiming Xu; Connor Phillips
>
> **摘要:** This study addresses ethical issues surrounding Large Language Models (LLMs) within the field of artificial intelligence. It explores the common ethical challenges posed by both LLMs and other AI systems, such as privacy and fairness, as well as ethical challenges uniquely arising from LLMs. It highlights challenges such as hallucination, verifiable accountability, and decoding censorship complexity, which are unique to LLMs and distinct from those encountered in traditional AI systems. The study underscores the need to tackle these complexities to ensure accountability, reduce biases, and enhance transparency in the influential role that LLMs play in shaping information dissemination. It proposes mitigation strategies and future directions for LLM ethics, advocating for interdisciplinary collaboration. It recommends ethical frameworks tailored to specific domains and dynamic auditing systems adapted to diverse contexts. This roadmap aims to guide responsible development and integration of LLMs, envisioning a future where ethical considerations govern AI advancements in society.
>
---
#### [replaced 009] Truth Knows No Language: Evaluating Truthfulness Beyond English
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.09387v3](http://arxiv.org/pdf/2502.09387v3)**

> **作者:** Blanca Calvo Figueras; Eneko Sagarzazu; Julen Etxaniz; Jeremy Barnes; Pablo Gamallo; Iria De Dios Flores; Rodrigo Agerri
>
> **备注:** 14 pages, 6 figures, 8 tables
>
> **摘要:** We introduce a professionally translated extension of the TruthfulQA benchmark designed to evaluate truthfulness in Basque, Catalan, Galician, and Spanish. Truthfulness evaluations of large language models (LLMs) have primarily been conducted in English. However, the ability of LLMs to maintain truthfulness across languages remains under-explored. Our study evaluates 12 state-of-the-art open LLMs, comparing base and instruction-tuned models using human evaluation, multiple-choice metrics, and LLM-as-a-Judge scoring. Our findings reveal that, while LLMs perform best in English and worst in Basque (the lowest-resourced language), overall truthfulness discrepancies across languages are smaller than anticipated. Furthermore, we show that LLM-as-a-Judge correlates more closely with human judgments than multiple-choice metrics, and that informativeness plays a critical role in truthfulness assessment. Our results also indicate that machine translation provides a viable approach for extending truthfulness benchmarks to additional languages, offering a scalable alternative to professional translation. Finally, we observe that universal knowledge questions are better handled across languages than context- and time-dependent ones, highlighting the need for truthfulness evaluations that account for cultural and temporal variability. Dataset and code are publicly available under open licenses.
>
---
#### [replaced 010] Impact of Shared E-scooter Introduction on Public Transport Demand: A Case Study in Santiago, Chile
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2409.17814v2](http://arxiv.org/pdf/2409.17814v2)**

> **作者:** Daniela Opitz; Eduardo Graells-Garrido; Jacqueline Arriagada; Matilde Rivas; Natalia Meza
>
> **备注:** 66 pages, 12 figures. Submitted to Travel Behaviour and Society
>
> **摘要:** This study examines how the introduction of shared electric scooters (e-scooters) affects public transport demand in Santiago, Chile, analyzing whether they complement or substitute for existing transit services. We used smart card data from the integrated public transport system of Santiago and GPS traces from e-scooter trips during the initial deployment period. We employed a difference-in-differences approach with negative binomial regression models across three urban regions identified through k-means clustering: Central, Intermediate, and Peripheral. Results reveal spatially heterogeneous effects on public transport boardings and alightings. In the Central Region, e-scooter introduction was associated with significant substitution effects, showing a 23.87% reduction in combined bus and metro boardings, suggesting e-scooters replace short public transport trips in high-density areas. The Intermediate Region showed strong complementary effects, with a 33.6% increase in public transport boardings and 4.08% increase in alightings, indicating e-scooters successfully serve as first/last-mile connectors that enhance transit accessibility. The Peripheral Region exhibited no significant effects. Metro services experienced stronger impacts than bus services, with metro boardings increasing 9.77\% in the Intermediate Region. Our findings advance understanding of micromobility-transit interactions by demonstrating that both substitution and complementarity can coexist within the same urban system, depending on local accessibility conditions. These results highlight the need for spatially differentiated mobility policies that recognize e-scooters' variable roles across urban environments.
>
---
#### [replaced 011] Evaluating how LLM annotations represent diverse views on contentious topics
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2503.23243v2](http://arxiv.org/pdf/2503.23243v2)**

> **作者:** Megan A. Brown; Shubham Atreja; Libby Hemphill; Patrick Y. Wu
>
> **摘要:** Researchers have proposed the use of generative large language models (LLMs) to label data for research and applied settings. This literature emphasizes the improved performance of these models relative to other natural language models, noting that generative LLMs typically outperform other models and even humans across several metrics. Previous literature has examined bias across many applications and contexts, but less work has focused specifically on bias in generative LLMs' responses to subjective annotation tasks. This bias could result in labels applied by LLMs that disproportionately align with majority groups over a more diverse set of viewpoints. In this paper, we evaluate how LLMs represent diverse viewpoints on these contentious tasks. Across four annotation tasks on four datasets, we show that LLMs do not show systematic substantial disagreement with annotators on the basis of demographics. Rather, we find that multiple LLMs tend to be biased in the same directions on the same demographic categories within the same datasets. Moreover, the disagreement between human annotators on the labeling task -- a measure of item difficulty -- is far more predictive of LLM agreement with human annotators. We conclude with a discussion of the implications for researchers and practitioners using LLMs for automated data annotation tasks. Specifically, we emphasize that fairness evaluations must be contextual, model choice alone will not solve potential issues of bias, and item difficulty must be integrated into bias assessments.
>
---
#### [replaced 012] Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.04322v2](http://arxiv.org/pdf/2502.04322v2)**

> **作者:** Yik Siu Chan; Narutatsu Ri; Yuxin Xiao; Marzyeh Ghassemi
>
> **摘要:** Despite extensive safety alignment efforts, large language models (LLMs) remain vulnerable to jailbreak attacks that elicit harmful behavior. While existing studies predominantly focus on attack methods that require technical expertise, two critical questions remain underexplored: (1) Are jailbroken responses truly useful in enabling average users to carry out harmful actions? (2) Do safety vulnerabilities exist in more common, simple human-LLM interactions? In this paper, we demonstrate that LLM responses most effectively facilitate harmful actions when they are both actionable and informative--two attributes easily elicited in multi-step, multilingual interactions. Using this insight, we propose HarmScore, a jailbreak metric that measures how effectively an LLM response enables harmful actions, and Speak Easy, a simple multi-step, multilingual attack framework. Notably, by incorporating Speak Easy into direct request and jailbreak baselines, we see an average absolute increase of 0.319 in Attack Success Rate and 0.426 in HarmScore in both open-source and proprietary LLMs across four safety benchmarks. Our work reveals a critical yet often overlooked vulnerability: Malicious users can easily exploit common interaction patterns for harmful intentions.
>
---
