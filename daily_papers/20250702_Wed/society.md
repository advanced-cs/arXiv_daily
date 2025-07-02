# 计算机与社会 cs.CY

- **最新发布 13 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] Ken Utilization Layer: Hebbian Replay Within a Student's Ken for Adaptive Knowledge Tracing
- **分类: cs.CY; cs.AI; cs.LG; cs.NE**

- **简介: 该论文属于知识追踪任务，旨在解决个性化学习中的持续适应问题。提出KUL-KT模型，结合赫布记忆与梯度优化，实现高效、灵活的知识追踪。**

- **链接: [http://arxiv.org/pdf/2507.00032v1](http://arxiv.org/pdf/2507.00032v1)**

> **作者:** Grey Kuling; Marinka Zitnik
>
> **摘要:** We introduce KUL-KT, a biologically inspired architecture for knowledge tracing (KT), combining Hebbian memory encoding with gradient-based consolidation in a scalable, input-agnostic framework. KUL-KT adapts the principle of memory consolidation in neural systems, to student modeling by introducing two key innovations: (i) a time-decaying Hebbian memory update that enables graceful forgetting, and (ii) a novel Loss-aligned Internal Target (LIT) method to compute an ideal internal state, allowing continual learning without backpropagation through time. The architecture consists of a fast Hebbian memory that captures each learner interaction via a single associative update, and a slower linear network that consolidates recalled samples through gradient descent. This design enables few-shot personalization and natural forgetting without storing raw data or relying on large cohort training. Operating entirely in embedding space, KUL-KT supports both structured (tabular) and unstructured (short-answer) inputs. Empirically, KUL-KT outperforms strong baselines on ten public KT benchmarks in rank-sensitive metrics such as nDCG and Recall@10. In a classroom deployment, KUL-KT personalized quizzes from short-answer data, leading to improved learner-perceived helpfulness and reduced difficulty (p < 0.05). Ablation studies confirm that Hebbian decay and LIT are critical for continual adaptation. Compared to a strong graph-based KT model, KUL-KT trains 1.75x faster and uses 99.01\% less memory. These results position KUL-KT as a biologically grounded, memory-efficient, and input-flexible framework for personalized learning at scale.
>
---
#### [new 002] Partnering with AI: A Pedagogical Feedback System for LLM Integration into Programming Education
- **分类: cs.CY**

- **简介: 该论文属于编程教育任务，旨在解决如何将LLM生成的反馈与教学原则结合。工作包括构建框架并评估其有效性。**

- **链接: [http://arxiv.org/pdf/2507.00406v1](http://arxiv.org/pdf/2507.00406v1)**

> **作者:** Niklas Scholz; Manh Hung Nguyen; Adish Singla; Tomohiro Nagashima
>
> **备注:** This is an extended version of a poster paper accepted and published at ECTEL-2025
>
> **摘要:** Feedback is one of the most crucial components to facilitate effective learning. With the rise of large language models (LLMs) in recent years, research in programming education has increasingly focused on automated feedback generation to help teachers provide timely support to every student. However, prior studies often overlook key pedagogical principles, such as mastery and progress adaptation, that shape effective feedback strategies. This paper introduces a novel pedagogical framework for LLM-driven feedback generation derived from established feedback models and local insights from secondary school teachers. To evaluate this framework, we implemented a web-based application for Python programming with LLM-based feedback that follows the framework and conducted a mixed-method evaluation with eight secondary-school computer science teachers. Our findings suggest that teachers consider that, when aligned with the framework, LLMs can effectively support students and even outperform human teachers in certain scenarios through instant and precise feedback. However, we also found several limitations, such as its inability to adapt feedback to dynamic classroom contexts. Such a limitation highlights the need to complement LLM-generated feedback with human expertise to ensure effective student learning. This work demonstrates an effective way to use LLMs for feedback while adhering to pedagogical standards and highlights important considerations for future systems.
>
---
#### [new 003] Teacher-AI Collaboration for Curating and Customizing Lesson Plans in Low-Resource Schools
- **分类: cs.CY; cs.HC**

- **简介: 该论文属于教育技术任务，旨在解决低资源学校教案定制问题。通过教师与AI协作，开发并评估了Shiksha copilot工具，提升教学效率与质量。**

- **链接: [http://arxiv.org/pdf/2507.00456v1](http://arxiv.org/pdf/2507.00456v1)**

> **作者:** Deepak Varuvel Dennison; Bakhtawar Ahtisham; Kavyansh Chourasia; Nirmit Arora; Rahul Singh; Rene F. Kizilcec; Akshay Nambi; Tanuja Ganu; Aditya Vashistha
>
> **摘要:** This study investigates Shiksha copilot, an AI-assisted lesson planning tool deployed in government schools across Karnataka, India. The system combined LLMs and human expertise through a structured process in which English and Kannada lesson plans were co-created by curators and AI; teachers then further customized these curated plans for their classrooms using their own expertise alongside AI support. Drawing on a large-scale mixed-methods study involving 1,043 teachers and 23 curators, we examine how educators collaborate with AI to generate context-sensitive lesson plans, assess the quality of AI-generated content, and analyze shifts in teaching practices within multilingual, low-resource environments. Our findings show that teachers used Shiksha copilot both to meet administrative documentation needs and to support their teaching. The tool eased bureaucratic workload, reduced lesson planning time, and lowered teaching-related stress, while promoting a shift toward activity-based pedagogy. However, systemic challenges such as staffing shortages and administrative demands constrained broader pedagogical change. We frame these findings through the lenses of teacher-AI collaboration and communities of practice to examine the effective integration of AI tools in teaching. Finally, we propose design directions for future teacher-centered EdTech, particularly in multilingual and Global South contexts.
>
---
#### [new 004] Integrating Universal Generative AI Platforms in Educational Labs to Foster Critical Thinking and Digital Literacy
- **分类: cs.CY; cs.AI; cs.LG; 68T50, 68U20, 97U50, 97D40; I.2.7; K.3.1; K.3.2; H.5.3**

- **简介: 该论文属于教育技术任务，旨在解决如何有效整合生成式AI提升学生批判性思维和数字素养的问题。通过设计AI实验活动，引导学生评估AI生成内容，验证其有效性与适用性。**

- **链接: [http://arxiv.org/pdf/2507.00007v1](http://arxiv.org/pdf/2507.00007v1)**

> **作者:** Vasiliy Znamenskiy; Rafael Niyazov; Joel Hernandez
>
> **备注:** http://doi.org/10.5121/ijci.2025.140302
>
> **摘要:** This paper presents a new educational framework for integrating generative artificial intelligence (GenAI) platforms such as ChatGPT, Claude, and Gemini into laboratory activities aimed at developing critical thinking and digital literacy among undergraduate students. Recognizing the limitations and risks of uncritical reliance on large language models (LLMs), the proposed pedagogical model reframes GenAI as a research subject and cognitive tool. Students formulate discipline-specific prompts and evaluate GenAI-generated responses in text, image, and video modalities. A pilot implementation in a general astronomy course for non-science majors demonstrated high levels of engagement and critical reflection, with many students continuing the activity after class and presenting results at a research symposium. The results highlight the importance of structured AI interactions in education and suggest that GenAI can improve learning outcomes when combined with reflective assessment methods. The study proposes a replicable model for interdisciplinary AI-integrated lab work, adaptable to scientific disciplines. See the guide to learning activities based on Generative-Ai platforms: https://doi.org/10.5281/zenodo.15555802
>
---
#### [new 005] Intellectual Property Rights and Entrepreneurship in the NFT Ecosystem: Legal Frameworks, Business Models, and Innovation Opportunities
- **分类: cs.CY; cs.ET**

- **简介: 该论文属于法律与科技交叉研究，旨在解决NFT生态中知识产权管理问题，分析法律框架与区块链交易的冲突，提出IP权利矩阵和商业模式分类。**

- **链接: [http://arxiv.org/pdf/2507.00172v1](http://arxiv.org/pdf/2507.00172v1)**

> **作者:** Pranav Darshan; Rohan J S; Raghuveer Rajesh; Ruchitha M; Sanika Kamath; Manas M N
>
> **备注:** 11 pages
>
> **摘要:** Non Fungible Tokens have changed digital ownership and how creators earn money. Between 2021 and 2024, the market value exceeded 40 billion. However, the fast growth of the NFT ecosystem has revealed serious issues in managing intellectual property rights. There is a lot of confusion about the difference between owning an NFT and owning the copyright for the underlying content. This research looks at the gap between traditional copyright laws and blockchain-based transactions. We use a mixed methods approach to analyze this disconnect. We create a new IP rights matrix that clearly shows how copyright law relates to NFT ownership structures. Additionally, we include a business model taxonomy that sorts new commercial applications by their IP risk and sustainability factors. By examining important legal cases, smart contracts, and interviews with stakeholders, we find key problems in enforcing laws across different regions, standardizing licenses, and assessing business opportunities.
>
---
#### [new 006] Teaching Programming in the Age of Generative AI: Insights from Literature, Pedagogical Proposals, and Student Perspectives
- **分类: cs.CY; cs.AI; cs.ET; cs.PL**

- **简介: 该论文属于教育技术任务，探讨生成式AI对编程教学的影响，提出通过可视化工具提升代码理解与教学效果。**

- **链接: [http://arxiv.org/pdf/2507.00108v1](http://arxiv.org/pdf/2507.00108v1)**

> **作者:** Clemente Rubio-Manzano; Jazna Meza; Rodolfo Fernandez-Santibanez; Christian Vidal-Castro
>
> **摘要:** Computer programming is undergoing a true transformation driven by powerful new tools for automatic source code generation based on large language models. This transformation is also manifesting in introductory programming courses at universities around the world, generating an in-depth debate about how programming content should be taught, learned, and assessed in the context of generative artificial intelligence. This article aims, on the one hand, to review the most relevant studies on this issue, highlighting the advantages and disadvantages identified in the specialized literature. On the other hand, it proposes enriching teaching and learning methodologies by focusing on code comprehension and execution rather than on mere coding or program functionality. In particular, it advocates for the use of visual representations of code and visual simulations of its execution as effective tools for teaching, learning, and assessing programming, thus fostering a deeper understanding among students. Finally, the opinions of students who took the object-oriented programming course are presented to provide preliminary context supporting the incorporation of visual simulations in Java (or other languages) as part of the training process.
>
---
#### [new 007] A Theory of Inference Compute Scaling: Reasoning through Directed Stochastic Skill Search
- **分类: cs.LG; cs.AI; cs.CY; cs.PF**

- **简介: 该论文属于大语言模型推理优化任务，旨在解决推理计算成本高的问题。提出DS3框架，分析不同推理策略的效率与效果。**

- **链接: [http://arxiv.org/pdf/2507.00004v1](http://arxiv.org/pdf/2507.00004v1)**

> **作者:** Austin R. Ellis-Mohr; Anuj K. Nayak; Lav R. Varshney
>
> **摘要:** Large language models (LLMs) demand considerable computational, energy, and financial resources during both training and deployment. While scaling laws for training have guided much of the field's recent progress, inference costs now represent a significant and growing component of the overall resource burden, particularly for reasoning-focused models. Existing characterizations of compute-optimality that consider model size, dataset size, and inference tokens in isolation or in fixed combinations risk overlooking more efficient operating points. We introduce directed stochastic skill search (DS3), a general framework that represents inference as stochastic traversal over a learned skill graph. From a simplified yet expressive instantiation, we derive closed-form expressions for task success and compute cost across a wide range of inference strategies -- including chain-of-thought (CoT) and tree-of-thought (ToT) -- enabling comparative analysis as a function of task difficulty and model capability. To that end, we extend a prior first-principles tripartite graph framework of LLM training to incorporate inference, and separately bridge DS3 with empirical methods that characterize LLM scaling behavior. We theoretically recover empirically observed patterns, including: linear accuracy scaling with logarithmic compute; variation in preferred inference strategies as a function of task difficulty and model capability; emergent behavior elicited by reasoning even when performance plateaus under parameter scaling; and both best-of-N (BoN) and majority voting behavior captured within a unified analytical framework. By explicitly characterizing training-inference interdependencies, our framework deepens theoretical understanding and supports principled algorithmic design and resource allocation.
>
---
#### [new 008] Time Series Foundation Models are Flow Predictors
- **分类: cs.LG; cs.CY**

- **简介: 该论文属于时间序列预测任务，旨在解决缺乏空间信息的客流预测问题。通过评估Moirai和TimesFM模型，在零样本设置下取得优于基线的结果。**

- **链接: [http://arxiv.org/pdf/2507.00945v1](http://arxiv.org/pdf/2507.00945v1)**

> **作者:** Massimiliano Luca; Ciro Beneduce; Bruno Lepri
>
> **备注:** arXiv admin note: text overlap with arXiv:2203.07372
>
> **摘要:** We investigate the effectiveness of time series foundation models (TSFMs) for crowd flow prediction, focusing on Moirai and TimesFM. Evaluated on three real-world mobility datasets-Bike NYC, Taxi Beijing, and Spanish national OD flows-these models are deployed in a strict zero-shot setting, using only the temporal evolution of each OD flow and no explicit spatial information. Moirai and TimesFM outperform both statistical and deep learning baselines, achieving up to 33% lower RMSE, 39% lower MAE and up to 49% higher CPC compared to state-of-the-art competitors. Our results highlight the practical value of TSFMs for accurate, scalable flow prediction, even in scenarios with limited annotated data or missing spatial context.
>
---
#### [new 009] Customer Service Representative's Perception of the AI Assistant in an Organization's Call Center
- **分类: cs.HC; cs.AI; cs.CY**

- **简介: 该论文属于人机交互研究，探讨AI助手在客服中的应用。解决AI对客服人员工作负担的影响问题，通过访谈分析其感知与适应情况。**

- **链接: [http://arxiv.org/pdf/2507.00513v1](http://arxiv.org/pdf/2507.00513v1)**

> **作者:** Kai Qin; Kexin Du; Yimeng Chen; Yueyan Liu; Jie Cai; Zhiqiang Nie; Nan Gao; Guohui Wei; Shengzhu Wang; Chun Yu
>
> **备注:** ACM CSCW Poster 2025
>
> **摘要:** The integration of various AI tools creates a complex socio-technical environment where employee-customer interactions form the core of work practices. This study investigates how customer service representatives (CSRs) at the power grid service customer service call center perceive AI assistance in their interactions with customers. Through a field visit and semi-structured interviews with 13 CSRs, we found that AI can alleviate some traditional burdens during the call (e.g., typing and memorizing) but also introduces new burdens (e.g., earning, compliance, psychological burdens). This research contributes to a more nuanced understanding of AI integration in organizational settings and highlights the efforts and burdens undertaken by CSRs to adapt to the updated system.
>
---
#### [new 010] Can Machines Philosophize?
- **分类: physics.soc-ph; cs.CY; physics.hist-ph**

- **简介: 该论文属于实验哲学任务，旨在评估机器是否能模仿人类哲学观点。通过构建框架比较机器与人类在科学实在论上的立场，发现两者观点相似但机器更一致。**

- **链接: [http://arxiv.org/pdf/2507.00675v1](http://arxiv.org/pdf/2507.00675v1)**

> **作者:** Michele Pizzochero; Giorgia Dellaferrera
>
> **摘要:** Inspired by the Turing test, we present a novel methodological framework to assess the extent to which a population of machines mirrors the philosophical views of a population of humans. The framework consists of three steps: (i) instructing machines to impersonate each human in the population, reflecting their backgrounds and beliefs, (ii) administering a questionnaire covering various philosophical positions to both humans and machines, and (iii) statistically analyzing the resulting responses. We apply this methodology to the debate on scientific realism, a long-standing philosophical inquiry exploring the relationship between science and reality. By considering the outcome of a survey of over 500 human participants, including both physicists and philosophers of science, we generate their machine personas using an artificial intelligence engine based on a large-language generative model. We reveal that the philosophical views of a population of machines are, on average, similar to those endorsed by a population of humans, irrespective of whether they are physicists or philosophers of science. As compared to humans, however, machines exhibit a weaker inclination toward scientific realism and a stronger coherence in their philosophical positions. Given the observed similarities between the populations of humans and machines, this methodological framework may offer unprecedented opportunities for advancing research in experimental philosophy by replacing human participants with their machine-impersonated counterparts, possibly mitigating the efficiency and reproducibility issues that affect survey-based empirical studies.
>
---
#### [new 011] ROSE: Toward Reality-Oriented Safety Evaluation of Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于大语言模型安全评估任务，旨在解决现有评估方法在对抗性提示覆盖不足和现实场景对齐差的问题。通过多目标强化学习生成多样化、情境丰富的对抗性提示，提升安全评估效果。**

- **链接: [http://arxiv.org/pdf/2507.00026v1](http://arxiv.org/pdf/2507.00026v1)**

> **作者:** Jiale Ding; Xiang Zheng; Cong Wang; Wei-Bin Lee; Xingjun Ma; Yu-Gang Jiang
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed as black-box components in real-world applications, evaluating their safety-especially under adversarial prompting-has become critical. Arguably, effective safety evaluations should be adaptive, evolving with LLM capabilities, and also cover a broad spectrum of harmful topics and real-world scenarios to fully expose potential vulnerabilities. Existing manual safety benchmarks, built on handcrafted adversarial prompts, are limited by their static nature and the intensive labor required to update them, making it difficult to keep pace with rapidly advancing LLMs. In contrast, automated adversarial prompt generation offers a promising path toward adaptive evaluation. However, current methods often suffer from insufficient adversarial topic coverage (topic-level diversity) and weak alignment with real-world contexts. These shortcomings stem from the exploration-exploitation dilemma in black-box optimization and a lack of real-world contextualization, resulting in adversarial prompts that are both topically narrow and scenario-repetitive. To address these issues, we propose Reality-Oriented Safety Evaluation (ROSE), a novel framework that uses multi-objective reinforcement learning to fine-tune an adversarial LLM for generating topically diverse and contextually rich adversarial prompts. Experiments show that ROSE outperforms existing methods in uncovering safety vulnerabilities in state-of-the-art LLMs, with notable improvements in integrated evaluation metrics. We hope ROSE represents a step toward more practical and reality-oriented safety evaluation of LLMs. WARNING: This paper contains examples of potentially harmful text.
>
---
#### [new 012] Many LLMs Are More Utilitarian Than One
- **分类: cs.CL; cs.AI; cs.CY; I.2.7; I.2.11**

- **简介: 该论文属于AI伦理与道德推理任务，研究多模型协作中的道德判断行为，探讨其与人类群体决策的异同。**

- **链接: [http://arxiv.org/pdf/2507.00814v1](http://arxiv.org/pdf/2507.00814v1)**

> **作者:** Anita Keshmirian; Razan Baltaji; Babak Hemmatian; Hadi Asghari; Lav R. Varshney
>
> **备注:** 9 pages, 8 Figures, 7 tables
>
> **摘要:** Moral judgment is integral to large language model (LLM) alignment and social reasoning. As multi-agent systems gain prominence, it becomes crucial to understand how LLMs function collectively during collaboration, compared to individual agents. In human moral judgment, group deliberation leads to a utilitarian boost: a tendency to endorse norm violations that maximize benefits for the greatest number of people despite harms. We study whether a similar dynamic emerges in multi-agent LLM systems. We tested six models on well-established sets of moral dilemmas across two conditions: (1) Solo, where models reasoned independently, and (2) Group, where they engaged in multi-turn discussions in pairs or triads. In personal moral dilemmas, where agents must decide to directly harm one individual to maximize the utility for others, all models found moral violations to be more acceptable when part of a group than individually, similar to human experiments. Some models endorsed actions that maximized overall well-being, even if they benefited strangers over familiar individuals. Others became more willing to violate moral norms in groups. However, while human groups show a similar action bias, the mechanism for their utilitarian boost differs from LLMs. Whereas the human shift comes from heightened sensitivity to decision outcomes, LLM groups show either reduced norm sensitivity or enhanced impartiality. This suggests that while the surface behavior of LLM collectives mimics human group reasoning, the underlying drivers differ. We discuss the implications for AI alignment, multi-agent design, and artificial moral reasoning.
>
---
#### [new 013] Social Robots for People with Dementia: A Literature Review on Deception from Design to Perception
- **分类: cs.HC; cs.CY; cs.RO**

- **简介: 该论文属于文献综述任务，探讨社会机器人在阿尔茨海默病护理中的欺骗问题，分析设计线索如何引发误解，并提出基于认知机制的欺骗定义。**

- **链接: [http://arxiv.org/pdf/2507.00963v1](http://arxiv.org/pdf/2507.00963v1)**

> **作者:** Fan Wang; Giulia Perugia; Yuan Feng; Wijnand IJsselsteijn
>
> **摘要:** As social robots increasingly enter dementia care, concerns about deception, intentional or not, are gaining attention. Yet, how robotic design cues might elicit misleading perceptions in people with dementia, and how these perceptions arise, remains insufficiently understood. In this scoping review, we examined 26 empirical studies on interactions between people with dementia and physical social robots. We identify four key design cue categories that may influence deceptive impressions: cues resembling physiological signs (e.g., simulated breathing), social intentions (e.g., playful movement), familiar beings (e.g., animal-like form and sound), and, to a lesser extent, cues that reveal artificiality. Thematic analysis of user responses reveals that people with dementia often attribute biological, social, and mental capacities to robots, dynamically shifting between awareness and illusion. These findings underscore the fluctuating nature of ontological perception in dementia contexts. Existing definitions of robotic deception often rest on philosophical or behaviorist premises, but rarely engage with the cognitive mechanisms involved. We propose an empirically grounded definition: robotic deception occurs when Type 1 (automatic, heuristic) processing dominates over Type 2 (deliberative, analytic) reasoning, leading to misinterpretation of a robot's artificial nature. This dual-process perspective highlights the ethical complexity of social robots in dementia care and calls for design approaches that are not only engaging, but also epistemically respectful.
>
---
## 更新

#### [replaced 001] Data-Centric Safety and Ethical Measures for Data and AI Governance
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2506.10217v3](http://arxiv.org/pdf/2506.10217v3)**

> **作者:** Srija Chakraborty
>
> **备注:** Paper accepted and presented at the AAAI 2025 Workshop on Datasets and Evaluators of AI Safety https://sites.google.com/view/datasafe25/home
>
> **摘要:** Datasets play a key role in imparting advanced capabilities to artificial intelligence (AI) foundation models that can be adapted to various downstream tasks. These downstream applications can introduce both beneficial and harmful capabilities -- resulting in dual use AI foundation models, with various technical and regulatory approaches to monitor and manage these risks. However, despite the crucial role of datasets, responsible dataset design and ensuring data-centric safety and ethical practices have received less attention. In this study, we pro-pose responsible dataset design framework that encompasses various stages in the AI and dataset lifecycle to enhance safety measures and reduce the risk of AI misuse due to low quality, unsafe and unethical data content. This framework is domain agnostic, suitable for adoption for various applications and can promote responsible practices in dataset creation, use, and sharing to facilitate red teaming, minimize risks, and increase trust in AI models.
>
---
#### [replaced 002] Bridging Ethical Principles and Algorithmic Methods: An Alternative Approach for Assessing Trustworthiness in AI Systems
- **分类: cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.22774v2](http://arxiv.org/pdf/2506.22774v2)**

> **作者:** Michael Papademas; Xenia Ziouvelou; Antonis Troumpoukis; Vangelis Karkaletsis
>
> **摘要:** Artificial Intelligence (AI) technology epitomizes the complex challenges posed by human-made artifacts, particularly those widely integrated into society and exert significant influence, highlighting potential benefits and their negative consequences. While other technologies may also pose substantial risks, AI's pervasive reach makes its societal effects especially profound. The complexity of AI systems, coupled with their remarkable capabilities, can lead to a reliance on technologies that operate beyond direct human oversight or understanding. To mitigate the risks that arise, several theoretical tools and guidelines have been developed, alongside efforts to create technological tools aimed at safeguarding Trustworthy AI. The guidelines take a more holistic view of the issue but fail to provide techniques for quantifying trustworthiness. Conversely, while technological tools are better at achieving such quantification, they lack a holistic perspective, focusing instead on specific aspects of Trustworthy AI. This paper aims to introduce an assessment method that combines the ethical components of Trustworthy AI with the algorithmic processes of PageRank and TrustRank. The goal is to establish an assessment framework that minimizes the subjectivity inherent in the self-assessment techniques prevalent in the field by introducing algorithmic criteria. The application of our approach indicates that a holistic assessment of an AI system's trustworthiness can be achieved by providing quantitative insights while considering the theoretical content of relevant guidelines.
>
---
#### [replaced 003] Not All Water Consumption Is Equal: A Water Stress Weighted Metric for Sustainable Computing
- **分类: cs.DC; cs.AR; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.22773v2](http://arxiv.org/pdf/2506.22773v2)**

> **作者:** Yanran Wu; Inez Hua; Yi Ding
>
> **备注:** 7 pages, 9 figures, The 4th Workshop on Sustainable Computer Systems (HotCarbon'25), Cambridge, MA, July 10-11th, 2025
>
> **摘要:** Water consumption is an increasingly critical dimension of computing sustainability, especially as AI workloads rapidly scale. However, current water impact assessment often overlooks where and when water stress is more severe. To fill in this gap, we present SCARF, the first general framework that evaluates water impact of computing by factoring in both spatial and temporal variations in water stress. SCARF calculates an Adjusted Water Impact (AWI) metric that considers both consumption volume and local water stress over time. Through three case studies on LLM serving, datacenters, and semiconductor fabrication plants, we show the hidden opportunities for reducing water impact by optimizing location and time choices, paving the way for water-sustainable computing. The code is available at https://github.com/jojacola/SCARF.
>
---
#### [replaced 004] Investigating the heterogenous effects of a massive content moderation intervention via Difference-in-Differences
- **分类: cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2411.04037v4](http://arxiv.org/pdf/2411.04037v4)**

> **作者:** Lorenzo Cima; Benedetta Tessa; Stefano Cresci; Amaury Trujillo; Marco Avvenuti
>
> **备注:** arXiv admin note: text overlap with arXiv:2401.11254 This work is an extension of this conference paper: Cima, L., Trujillo, A., Avvenuti, M., & Cresci, S. (2024, May). The Great Ban: Efficacy and Unintended Consequences of a Massive Deplatforming Operation on Reddit. In Companion Publication of the 16th ACM Web Science Conference (pp. 85-93)
>
> **摘要:** In today's online environments, users encounter harm and abuse on a daily basis. Therefore, content moderation is crucial to ensure their safety and well-being. However, the effectiveness of many moderation interventions is still uncertain. Here, we apply a causal inference approach to shed light on the effectiveness of The Great Ban, a massive social media deplatforming intervention on Reddit. We analyze 53M comments shared by nearly 34K users, providing in-depth results on both the intended and unintended consequences of the ban. Our causal analyses reveal that 15.6% of the moderated users abandoned the platform while the remaining ones decreased their overall toxicity by 4.1%. Nonetheless, a small subset of users exhibited marked increases in both the intensity and volume of toxic behavior, particularly among those whose activity levels changed after the intervention. However, these reactions were not accompanied by greater activity or engagement, suggesting that even the most toxic users maintained a limited overall impact. Our findings bring to light new insights on the effectiveness of deplatforming moderation interventions. Furthermore, they also contribute to informing future content moderation strategies and regulations.
>
---
#### [replaced 005] Persistence Paradox in Dynamic Science
- **分类: cs.DL; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.22729v2](http://arxiv.org/pdf/2506.22729v2)**

> **作者:** Honglin Bao; Kai Li
>
> **摘要:** Persistence is often regarded as a virtue in science. In this paper, however, we challenge this conventional view by highlighting its contextual nature, particularly how persistence can become a liability during periods of paradigm shift. We focus on the deep learning revolution catalyzed by AlexNet in 2012. Analyzing the 20-year career trajectories of over 5,000 scientists who were active in top machine learning venues during the preceding decade, we examine how their research focus and output evolved. We first uncover a dynamic period in which leading venues increasingly prioritized cutting-edge deep learning developments that displaced relatively traditional statistical learning methods. Scientists responded to these changes in markedly different ways. Those who were previously successful or affiliated with old teams adapted more slowly, experiencing what we term a rigidity penalty - a reluctance to embrace new directions leading to a decline in scientific impact, as measured by citation percentile rank. In contrast, scientists who pursued strategic adaptation - selectively pivoting toward emerging trends while preserving weak connections to prior expertise - reaped the greatest benefits. Taken together, our macro- and micro-level findings show that scientific breakthroughs act as mechanisms that reconfigure power structures within a field.
>
---
#### [replaced 006] The Singapore Consensus on Global AI Safety Research Priorities
- **分类: cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.20702v2](http://arxiv.org/pdf/2506.20702v2)**

> **作者:** Yoshua Bengio; Tegan Maharaj; Luke Ong; Stuart Russell; Dawn Song; Max Tegmark; Lan Xue; Ya-Qin Zhang; Stephen Casper; Wan Sie Lee; Sören Mindermann; Vanessa Wilfred; Vidhisha Balachandran; Fazl Barez; Michael Belinsky; Imane Bello; Malo Bourgon; Mark Brakel; Siméon Campos; Duncan Cass-Beggs; Jiahao Chen; Rumman Chowdhury; Kuan Chua Seah; Jeff Clune; Juntao Dai; Agnes Delaborde; Nouha Dziri; Francisco Eiras; Joshua Engels; Jinyu Fan; Adam Gleave; Noah Goodman; Fynn Heide; Johannes Heidecke; Dan Hendrycks; Cyrus Hodes; Bryan Low Kian Hsiang; Minlie Huang; Sami Jawhar; Wang Jingyu; Adam Tauman Kalai; Meindert Kamphuis; Mohan Kankanhalli; Subhash Kantamneni; Mathias Bonde Kirk; Thomas Kwa; Jeffrey Ladish; Kwok-Yan Lam; Wan Lee Sie; Taewhi Lee; Xiaojian Li; Jiajun Liu; Chaochao Lu; Yifan Mai; Richard Mallah; Julian Michael; Nick Moës; Simon Möller; Kihyuk Nam; Kwan Yee Ng; Mark Nitzberg; Besmira Nushi; Seán O hÉigeartaigh; Alejandro Ortega; Pierre Peigné; James Petrie; Benjamin Prud'Homme; Reihaneh Rabbany; Nayat Sanchez-Pi; Sarah Schwettmann; Buck Shlegeris; Saad Siddiqui; Aradhana Sinha; Martín Soto; Cheston Tan; Dong Ting; William Tjhi; Robert Trager; Brian Tse; Anthony Tung K. H.; Vanessa Wilfred; John Willes; Denise Wong; Wei Xu; Rongwu Xu; Yi Zeng; HongJiang Zhang; Djordje Žikelić
>
> **备注:** Final report from the "2025 Singapore Conference on AI (SCAI)" held April 26: https://www.scai.gov.sg/2025/scai2025-report
>
> **摘要:** Rapidly improving AI capabilities and autonomy hold significant promise of transformation, but are also driving vigorous debate on how to ensure that AI is safe, i.e., trustworthy, reliable, and secure. Building a trusted ecosystem is therefore essential -- it helps people embrace AI with confidence and gives maximal space for innovation while avoiding backlash. The "2025 Singapore Conference on AI (SCAI): International Scientific Exchange on AI Safety" aimed to support research in this space by bringing together AI scientists across geographies to identify and synthesise research priorities in AI safety. This resulting report builds on the International AI Safety Report chaired by Yoshua Bengio and backed by 33 governments. By adopting a defence-in-depth model, this report organises AI safety research domains into three types: challenges with creating trustworthy AI systems (Development), challenges with evaluating their risks (Assessment), and challenges with monitoring and intervening after deployment (Control).
>
---
#### [replaced 007] Red Teaming for Generative AI, Report on a Copyright-Focused Exercise Completed in an Academic Medical Center
- **分类: cs.CY; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.22523v2](http://arxiv.org/pdf/2506.22523v2)**

> **作者:** James Wen; Sahil Nalawade; Zhiwei Liang; Catherine Bielick; Marisa Ferrara Boston; Alexander Chowdhury; Adele Collin; Luigi De Angelis; Jacob Ellen; Heather Frase; Rodrigo R. Gameiro; Juan Manuel Gutierrez; Pooja Kadam; Murat Keceli; Srikanth Krishnamurthy; Anne Kwok; Yanan Lance Lu; Heather Mattie; Liam G. McCoy; Katherine Miller; Allison C. Morgan; Marlene Louisa Moerig; Trang Nguyen; Alexander Owen-Post; Alex D. Ruiz; Sreekar Reddy Puchala; Soujanya Samineni; Takeshi Tohyama; Varun Ullanat; Carmine Valenza; Camilo Velez; Pengcheng Wang; Anna Wuest; Yuxiang Zhou; Yingde Zhu; Jason M. Johnson; Naomi Lenane; Jennifer Willcox; Francis J. Vitiello; Leo Anthony G. Celi; Renato Umeton
>
> **摘要:** Background: Generative artificial intelligence (AI) deployment in healthcare settings raises copyright compliance concerns. Dana-Farber Cancer Institute implemented GPT4DFCI, an internal generative AI tool utilizing OpenAI models, that is approved for enterprise use in research and operations. Given (i) the exceptionally broad adoption of the tool in our organization, (ii) our research mission, and (iii) the shared responsibility model required by Microsoft OpenAI products, we deemed rigorous copyright compliance testing necessary. Case Description: We conducted a structured red teaming exercise in Nov. 2024, with 42 participants from academic, industry, and government institutions. Four teams attempted to extract copyrighted content from GPT4DFCI across four domains: literary works, news articles, scientific publications, and access-restricted clinical notes. Teams successfully extracted verbatim book dedications and near-exact passages through indirect prompting strategies. News article extraction failed despite jailbreak attempts. Scientific article reproduction yielded only high-level summaries. Clinical note testing revealed appropriate privacy safeguards with data reformatting rather than reproduction. Discussion: The successful extraction of literary content indicates potential copyright material presence in training data, necessitating enhanced inference-time filtering. Differential success rates across content types suggest varying protective mechanisms. The event led to implementation of a copyright-specific meta-prompt in GPT4DFCI; this mitigation is in production since Jan. 2025. Conclusion: Systematic red teaming revealed specific vulnerabilities in generative AI copyright compliance, leading to concrete mitigation strategies. Academic medical institutions deploying generative AI must implement continuous testing protocols to ensure legal and ethical compliance.
>
---
