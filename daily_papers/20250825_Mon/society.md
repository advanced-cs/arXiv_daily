# 计算机与社会 cs.CY

- **最新发布 10 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] PediatricsMQA: a Multi-modal Pediatrics Question Answering Benchmark
- **分类: cs.CY; cs.AI; cs.CL; cs.GR; cs.MM**

- **简介: 该论文提出PediatricsMQA基准，用于多模态儿科问答任务，旨在解决大模型在儿童医疗中因年龄偏见导致的性能下降问题。工作包括构建包含3417个文本和2067个视觉题目的数据集，并验证现有模型在不同年龄段的公平性。**

- **链接: [http://arxiv.org/pdf/2508.16439v1](http://arxiv.org/pdf/2508.16439v1)**

> **作者:** Adil Bahaj; Mounir Ghogho
>
> **摘要:** Large language models (LLMs) and vision-augmented LLMs (VLMs) have significantly advanced medical informatics, diagnostics, and decision support. However, these models exhibit systematic biases, particularly age bias, compromising their reliability and equity. This is evident in their poorer performance on pediatric-focused text and visual question-answering tasks. This bias reflects a broader imbalance in medical research, where pediatric studies receive less funding and representation despite the significant disease burden in children. To address these issues, a new comprehensive multi-modal pediatric question-answering benchmark, PediatricsMQA, has been introduced. It consists of 3,417 text-based multiple-choice questions (MCQs) covering 131 pediatric topics across seven developmental stages (prenatal to adolescent) and 2,067 vision-based MCQs using 634 pediatric images from 67 imaging modalities and 256 anatomical regions. The dataset was developed using a hybrid manual-automatic pipeline, incorporating peer-reviewed pediatric literature, validated question banks, existing benchmarks, and existing QA resources. Evaluating state-of-the-art open models, we find dramatic performance drops in younger cohorts, highlighting the need for age-aware methods to ensure equitable AI support in pediatric care.
>
---
#### [new 002] Disproportionate Voices: Participation Inequality and Hostile Engagement in News Comments
- **分类: cs.CY**

- **简介: 该论文研究在线新闻评论中参与不平等与敌对行为的关系。任务为分析数字参与差异及其对线上话语的影响。通过分析2.6亿条评论数据，发现少数高频用户主导讨论且更易发布敌对内容，尤其在政治类新闻和大选期间。**

- **链接: [http://arxiv.org/pdf/2508.16040v1](http://arxiv.org/pdf/2508.16040v1)**

> **作者:** Sangbeom Kim; Seonhye Noh
>
> **备注:** 12 pages, 12 figures, 4 tables. Preprint. Under review
>
> **摘要:** Digital platforms were expected to foster broad participation in public discourse, yet online engagement remains highly unequal and underexplored. This study examines the digital participation divide and its link to hostile engagement in news comment sections. Analyzing 260 million comments from 6.2 million users over 13 years on Naver News, South Korea's largest news aggregation platform, we quantify participation inequality using the Gini and Palma indexes and estimate hostility levels with a KC-Electra model, which outperformed other Korean pre-trained transformers in multi-label classification tasks. The findings reveal a highly skewed participation structure, with a small number of frequent users dominating discussions, particularly in the Politics and Society domains and popular news stories. Participation inequality spikes during presidential elections, and frequent commenters are significantly more likely to post hostile content, suggesting that online discourse is shaped disproportionately by a highly active and often hostile subset of users. Using individual-level digital trace data, this study provides empirical insights into the behavioral dynamics of online participation inequality and its broader implications for public digital discourse.
>
---
#### [new 003] Who's Asking? Investigating Bias Through the Lens of Disability Framed Queries in LLMs
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于AI偏见审计任务，旨在研究大语言模型在无明确信息时如何基于残疾相关提示推断用户身份。作者系统测试8个主流LLM，发现模型高度依赖残疾线索产生偏见推理，且规模越大越敏感。建议引入弃权校准和反事实微调以缓解问题。**

- **链接: [http://arxiv.org/pdf/2508.15831v1](http://arxiv.org/pdf/2508.15831v1)**

> **作者:** Srikant Panda; Vishnu Hari; Kalpana Panda; Amit Agarwal; Hitesh Laxmichand Patel
>
> **备注:** Preprint
>
> **摘要:** Large Language Models (LLMs) routinely infer users demographic traits from phrasing alone, which can result in biased responses, even when no explicit demographic information is provided. The role of disability cues in shaping these inferences remains largely uncharted. Thus, we present the first systematic audit of disability-conditioned demographic bias across eight state-of-the-art instruction-tuned LLMs ranging from 3B to 72B parameters. Using a balanced template corpus that pairs nine disability categories with six real-world business domains, we prompt each model to predict five demographic attributes - gender, socioeconomic status, education, cultural background, and locality - under both neutral and disability-aware conditions. Across a varied set of prompts, models deliver a definitive demographic guess in up to 97\% of cases, exposing a strong tendency to make arbitrary inferences with no clear justification. Disability context heavily shifts predicted attribute distributions, and domain context can further amplify these deviations. We observe that larger models are simultaneously more sensitive to disability cues and more prone to biased reasoning, indicating that scale alone does not mitigate stereotype amplification. Our findings reveal persistent intersections between ableism and other demographic stereotypes, pinpointing critical blind spots in current alignment strategies. We release our evaluation framework and results to encourage disability-inclusive benchmarking and recommend integrating abstention calibration and counterfactual fine-tuning to curb unwarranted demographic inference. Code and data will be released on acceptance.
>
---
#### [new 004] Continuous Determination of Respiratory Rate in Hospitalized Patients using Machine Learning Applied to Electrocardiogram Telemetry
- **分类: eess.SP; cs.CY; cs.LG**

- **简介: 论文提出用神经网络从心电图信号中连续准确估算呼吸频率，解决人工测量不准确、普通病房缺乏自动监测的问题。通过多数据集验证，误差低于1.78次/分，证明其在早期预警系统中的潜力。**

- **链接: [http://arxiv.org/pdf/2508.15947v1](http://arxiv.org/pdf/2508.15947v1)**

> **作者:** Thomas Kite; Brian Ayers; Nicholas Houstis; Asishana A. Osho; Thoralf M. Sundt; Aaron D Aguirre
>
> **备注:** 15 pages, 8 figures, 2 tables
>
> **摘要:** Respiration rate (RR) is an important vital sign for clinical monitoring of hospitalized patients, with changes in RR being strongly tied to changes in clinical status leading to adverse events. Human labels for RR, based on counting breaths, are known to be inaccurate and time consuming for medical staff. Automated monitoring of RR is in place for some patients, typically those in intensive care units (ICUs), but is absent for the majority of inpatients on standard medical wards who are still at risk for clinical deterioration. This work trains a neural network (NN) to label RR from electrocardiogram (ECG) telemetry waveforms, which like many biosignals, carry multiple signs of respiratory variation. The NN shows high accuracy on multiple validation sets (internal and external, same and different sources of RR labels), with mean absolute errors less than 1.78 breaths per minute (bpm) in the worst case. The clinical utility of such a technology is exemplified by performing a retrospective analysis of two patient cohorts that suffered adverse events including respiratory failure, showing that continuous RR monitoring could reveal dynamics that strongly tracked with intubation events. This work exemplifies the method of combining pre-existing telemetry monitoring systems and artificial intelligence (AI) to provide accurate, automated and scalable patient monitoring, all of which builds towards an AI-based hospital-wide early warning system (EWS).
>
---
#### [new 005] SafeSpace: An Integrated Web Application for Digital Safety and Emotional Well-being
- **分类: cs.HC; cs.AI; cs.CY**

- **简介: 该论文提出SafeSpace，一个整合数字安全与情感健康的Web应用。解决在线有害内容、紧急情况响应和情绪评估分离的问题。工作包括：毒性检测、安全提醒系统和情绪问卷模块，实验验证了高精度与可靠性。**

- **链接: [http://arxiv.org/pdf/2508.16488v1](http://arxiv.org/pdf/2508.16488v1)**

> **作者:** Kayenat Fatmi; Mohammad Abbas
>
> **备注:** 5 pages, 2 figures, 1 table. Preprint submitted to arXiv
>
> **摘要:** In the digital era, individuals are increasingly exposed to online harms such as toxicity, manipulation, and grooming, which often pose emotional and safety risks. Existing systems for detecting abusive content or issuing safety alerts operate in isolation and rarely combine digital safety with emotional well-being. In this paper, we present SafeSpace, a unified web application that integrates three modules: (1) toxicity detection in chats and screenshots using NLP models and Google's Perspective API, (2) a configurable safety ping system that issues emergency alerts with the user's live location (longitude and latitude) via SMTP-based emails when check-ins are missed or SOS alerts are manually triggered, and (3) a reflective questionnaire that evaluates relationship health and emotional resilience. The system employs Firebase for alert management and a modular architecture designed for usability, privacy, and scalability. The experimental evaluation shows 93% precision in toxicity detection, 100% reliability in safety alerts under emulator tests, and 92% alignment between automated and manual questionnaire scoring. SafeSpace, implemented as a web application, demonstrates the feasibility of integrating detection, protection, and reflection within a single platform, with future deployment envisioned as a mobile application for broader accessibility.
>
---
#### [new 006] Embarrassed to observe: The effects of directive language in brand conversation
- **分类: cs.CL; cs.CY; cs.HC; cs.SI**

- **简介: 该论文研究社交媒体中品牌使用指令性语言对消费者参与度的影响。通过实地研究和三个在线实验，发现指令性语言会引发旁观者尴尬，降低参与度，尤其在非产品相关对话中更明显，但强品牌关系可缓解此效应。任务为探究品牌互动中的语言策略效果。**

- **链接: [http://arxiv.org/pdf/2508.15826v1](http://arxiv.org/pdf/2508.15826v1)**

> **作者:** Andria Andriuzzi; Géraldine Michel
>
> **备注:** This is an open access article under the terms of the Creative Commons Attribution-NonCommercial-NoDerivs License, which permits use and distribution in any medium, provided the original work is properly cited, the use is non-commercial and no modifications or adaptations are made
>
> **摘要:** In social media, marketers attempt to influence consumers by using directive language, that is, expressions designed to get consumers to take action. While the literature has shown that directive messages in advertising have mixed results for recipients, we know little about the effects of directive brand language on consumers who see brands interacting with other consumers in social media conversations. On the basis of a field study and three online experiments, this study shows that directive language in brand conversation has a detrimental downstream effect on engagement of consumers who observe such exchanges. Specifically, in line with Goffman's facework theory, because a brand that encourages consumers to react could be perceived as face-threatening, consumers who see a brand interacting with others in a directive way may feel vicarious embarrassment and engage less (compared with a conversation without directive language). In addition, we find that when the conversation is nonproduct-centered (vs. product-centered), consumers expect more freedom, as in mundane conversations, even for others; therefore, directive language has a stronger negative effect. However, in this context, the strength of the brand relationship mitigates this effect. Thus, this study contributes to the literature on directive language and brand-consumer interactions by highlighting the importance of context in interactive communication, with direct relevance for social media and brand management.
>
---
#### [new 007] Urban Comfort Assessment in the Era of Digital Planning: A Multidimensional, Data-driven, and AI-assisted Framework
- **分类: cs.AI; cs.CY**

- **简介: 论文提出一个三维数字规划框架，用于评估城市舒适度，解决缺乏统一定义和综合评价体系的问题。通过多维度分析、数据支持与AI辅助实现量化评估。**

- **链接: [http://arxiv.org/pdf/2508.16057v1](http://arxiv.org/pdf/2508.16057v1)**

> **作者:** Sijie Yang; Binyu Lei; Filip Biljecki
>
> **备注:** Presented at 19th International Conference on Computational Urban Planning and Urban Management (CUPUM 2025)
>
> **摘要:** Ensuring liveability and comfort is one of the fundamental objectives of urban planning. Numerous studies have employed computational methods to assess and quantify factors related to urban comfort such as greenery coverage, thermal comfort, and walkability. However, a clear definition of urban comfort and its comprehensive evaluation framework remain elusive. Our research explores the theoretical interpretations and methodologies for assessing urban comfort within digital planning, emphasising three key dimensions: multidimensional analysis, data support, and AI assistance.
>
---
#### [new 008] Meet Your New Client: Writing Reports for AI -- Benchmarking Information Loss in Market Research Deliverables
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于AI与市场研究交叉任务，旨在解决PDF/PPTX报告在RAG系统中因信息丢失影响AI理解的问题。工作包括构建端到端基准测试，比较Markdown转换后LLM回答事实问题的性能，发现图表等复杂对象信息损失严重，建议开发AI原生交付格式。**

- **链接: [http://arxiv.org/pdf/2508.15817v1](http://arxiv.org/pdf/2508.15817v1)**

> **作者:** Paul F. Simmering; Benedikt Schulz; Oliver Tabino; Georg Wittenburg
>
> **备注:** 16 pages, 4 figures, 3 tables
>
> **摘要:** As organizations adopt retrieval-augmented generation (RAG) for their knowledge management systems (KMS), traditional market research deliverables face new functional demands. While PDF reports and slides have long served human readers, they are now also "read" by AI systems to answer user questions. To future-proof reports being delivered today, this study evaluates information loss during their ingestion into RAG systems. It compares how well PDF and PowerPoint (PPTX) documents converted to Markdown can be used by an LLM to answer factual questions in an end-to-end benchmark. Findings show that while text is reliably extracted, significant information is lost from complex objects like charts and diagrams. This suggests a need for specialized, AI-native deliverables to ensure research insights are not lost in translation.
>
---
#### [new 009] Benchmarking the Legal Reasoning of LLMs in Arabic Islamic Inheritance Cases
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于法律AI任务，旨在解决阿拉伯语伊斯兰继承案例中的法律推理问题。作者评估了多个大语言模型在识别继承人、计算份额及解释推理方面的能力，提出基于三模型投票的解决方案，在QIAS 2025挑战中取得92.7%准确率。**

- **链接: [http://arxiv.org/pdf/2508.15796v1](http://arxiv.org/pdf/2508.15796v1)**

> **作者:** Nouar AlDahoul; Yasir Zaki
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Islamic inheritance domain holds significant importance for Muslims to ensure fair distribution of shares between heirs. Manual calculation of shares under numerous scenarios is complex, time-consuming, and error-prone. Recent advancements in Large Language Models (LLMs) have sparked interest in their potential to assist with complex legal reasoning tasks. This study evaluates the reasoning capabilities of state-of-the-art LLMs to interpret and apply Islamic inheritance laws. We utilized the dataset proposed in the ArabicNLP QIAS 2025 challenge, which includes inheritance case scenarios given in Arabic and derived from Islamic legal sources. Various base and fine-tuned models, are assessed on their ability to accurately identify heirs, compute shares, and justify their reasoning in alignment with Islamic legal principles. Our analysis reveals that the proposed majority voting solution, leveraging three base models (Gemini Flash 2.5, Gemini Pro 2.5, and GPT o3), outperforms all other models that we utilized across every difficulty level. It achieves up to 92.7% accuracy and secures the third place overall in Task 1 of the Qias 2025 challenge.
>
---
#### [new 010] Counterspeech for Mitigating the Influence of Media Bias: Comparing Human and LLM-Generated Responses
- **分类: cs.CL; cs.CY; cs.SI**

- **链接: [http://arxiv.org/pdf/2508.15855v1](http://arxiv.org/pdf/2508.15855v1)**

> **作者:** Luyang Lin; Zijin Feng; Lingzhi Wang; Kam-Fai Wong
>
> **摘要:** Biased news contributes to societal polarization and is often reinforced by hostile reader comments, constituting a vital yet often overlooked aspect of news dissemination. Our study reveals that offensive comments support biased content, amplifying bias and causing harm to targeted groups or individuals. Counterspeech is an effective approach to counter such harmful speech without violating freedom of speech, helping to limit the spread of bias. To the best of our knowledge, this is the first study to explore counterspeech generation in the context of news articles. We introduce a manually annotated dataset linking media bias, offensive comments, and counterspeech. We conduct a detailed analysis showing that over 70\% offensive comments support biased articles, amplifying bias and thus highlighting the importance of counterspeech generation. Comparing counterspeech generated by humans and large language models, we find model-generated responses are more polite but lack the novelty and diversity. Finally, we improve generated counterspeech through few-shot learning and integration of news background information, enhancing both diversity and relevance.
>
---
## 更新

#### [replaced 001] AI-Powered CPS-Enabled Urban Transportation Digital Twin: Methods and Applications
- **分类: eess.SY; cs.AI; cs.CY; cs.NI; cs.SY**

- **链接: [http://arxiv.org/pdf/2501.10396v2](http://arxiv.org/pdf/2501.10396v2)**

> **作者:** Yongjie Fu; Mehmet K. Turkcan; Mahshid Ghasemi; Zhaobin Mo; Chengbo Zang; Abhishek Adhikari; Zoran Kostic; Gil Zussman; Xuan Di
>
> **摘要:** We present methods and applications for the development of digital twins (DT) for urban traffic management. While the majority of studies on the DT focus on its ``eyes," which is the emerging sensing and perception like object detection and tracking, what really distinguishes the DT from a traditional simulator lies in its ``brain," the prediction and decision making capabilities of extracting patterns and making informed decisions from what has been seen and perceived. In order to add value to urban transportation management, DTs need to be powered by artificial intelligence and complement with low-latency high-bandwidth sensing and networking technologies, in other words, cyberphysical systems (CPS). We will first review the DT pipeline enabled by CPS and propose our DT architecture deployed on a real-world testbed in New York City. This paper can be a pointer to help researchers and practitioners identify challenges and opportunities for the development of DTs; a bridge to initiate conversations across disciplines; and a road map to exploiting potentials of DTs for diverse urban transportation applications.
>
---
#### [replaced 002] LearnLM: Improving Gemini for Learning
- **分类: cs.CY; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.16429v3](http://arxiv.org/pdf/2412.16429v3)**

> **作者:** LearnLM Team; Abhinit Modi; Aditya Srikanth Veerubhotla; Aliya Rysbek; Andrea Huber; Brett Wiltshire; Brian Veprek; Daniel Gillick; Daniel Kasenberg; Derek Ahmed; Irina Jurenka; James Cohan; Jennifer She; Julia Wilkowski; Kaiz Alarakyia; Kevin R. McKee; Lisa Wang; Markus Kunesch; Mike Schaekermann; Miruna Pîslar; Nikhil Joshi; Parsa Mahmoudieh; Paul Jhun; Sara Wiltberger; Shakir Mohamed; Shashank Agarwal; Shubham Milind Phal; Sun Jae Lee; Theofilos Strinopoulos; Wei-Jen Ko; Amy Wang; Ankit Anand; Avishkar Bhoopchand; Dan Wild; Divya Pandya; Filip Bar; Garth Graham; Holger Winnemoeller; Mahvish Nagda; Prateek Kolhar; Renee Schneider; Shaojian Zhu; Stephanie Chan; Steve Yadlowsky; Viknesh Sounderajah; Yannis Assael
>
> **摘要:** Today's generative AI systems are tuned to present information by default, rather than engage users in service of learning as a human tutor would. To address the wide range of potential education use cases for these systems, we reframe the challenge of injecting pedagogical behavior as one of \textit{pedagogical instruction following}, where training and evaluation examples include system-level instructions describing the specific pedagogy attributes present or desired in subsequent model turns. This framing avoids committing our models to any particular definition of pedagogy, and instead allows teachers or developers to specify desired model behavior. It also clears a path to improving Gemini models for learning -- by enabling the addition of our pedagogical data to post-training mixtures -- alongside their rapidly expanding set of capabilities. Both represent important changes from our initial tech report. We show how training with pedagogical instruction following produces a LearnLM model (available on Google AI Studio) that experts substantially prefer across a diverse set of learning scenarios, with average preference strengths of +31\% over GPT-4o, +11\% over Claude 3.5 Sonnet, and +13\% over the Gemini 1.5 Pro model on which LearnLM was based.
>
---
#### [replaced 003] Discrimination and AI in insurance: what do people find fair? Results from a survey
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2501.12897v2](http://arxiv.org/pdf/2501.12897v2)**

> **作者:** Frederik Zuiderveen Borgesius; Marvin van Bekkum; Iris van Ooijen; Gabi Schaap; Maaike Harbers; Tjerk Timan
>
> **摘要:** Two modern trends in insurance are data-intensive underwriting and behavior-based insurance. Data-intensive underwriting means that insurers analyze more data for estimating the claim cost of a consumer and for determining the premium based on that estimation. Insurers also offer behavior-based insurance. For example, some car insurers use artificial intelligence (AI) to follow the driving behavior of an individual consumer in real-time and decide whether to offer that consumer a discount. In this paper, we report on a survey of the Dutch population (N=999) in which we asked people's opinions about examples of data-intensive underwriting and behavior-based insurance. The main results include: (i) If survey respondents find an insurance practice unfair, they also find the practice unacceptable. (ii) Respondents find almost all modern insurance practices that we described unfair. (iii) Respondents find practices for which they can influence the premium fairer. (iv) If respondents find a certain consumer characteristic illogical for basing the premium on, then respondents find using the characteristic unfair. (v) Respondents find it unfair if an insurer offers an insurance product only to a specific group. (vi) Respondents find it unfair if an insurance practice leads to the poor paying more. We also reflect on the policy implications of the findings.
>
---
#### [replaced 004] Structured Prompts, Better Outcomes? Exploring the Effects of a Structured Interface with ChatGPT in a Graduate Robotics Course
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2507.07767v2](http://arxiv.org/pdf/2507.07767v2)**

> **作者:** Jerome Brender; Laila El-Hamamsy; Kim Uittenhove; Francesco Mondada; Engin Bumbacher
>
> **备注:** Accepted, to appear in the proceedings of the EC-TEL 2025 conference
>
> **摘要:** Prior research shows that how students engage with Large Language Models (LLMs) influences their problem-solving and understanding, reinforcing the need to support productive LLM-uses that promote learning. This study evaluates the impact of a structured GPT platform designed to promote 'good' prompting behavior with data from 58 students in a graduate-level robotics course. The students were assigned to either an intervention group using the structured platform or a control group using ChatGPT freely for two practice lab sessions, before a third session where all students could freely use ChatGPT. We analyzed student perception (pre-post surveys), prompting behavior (logs), performance (task scores), and learning (pre-post tests). Although we found no differences in performance or learning between groups, we identified prompting behaviors - such as having clear prompts focused on understanding code - that were linked with higher learning gains and were more prominent when students used the structured platform. However, such behaviors did not transfer once students were no longer constrained to use the structured platform. Qualitative survey data showed mixed perceptions: some students perceived the value of the structured platform, but most did not perceive its relevance and resisted changing their habits. These findings contribute to ongoing efforts to identify effective strategies for integrating LLMs into learning and question the effectiveness of bottom-up approaches that temporarily alter user interfaces to influence students' interaction. Future research could instead explore top-down strategies that address students' motivations and explicitly demonstrate how certain interaction patterns support learning.
>
---
#### [replaced 005] Validating LLM-as-a-Judge Systems under Rating Indeterminacy
- **分类: cs.LG; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2503.05965v3](http://arxiv.org/pdf/2503.05965v3)**

> **作者:** Luke Guerdan; Solon Barocas; Kenneth Holstein; Hanna Wallach; Zhiwei Steven Wu; Alexandra Chouldechova
>
> **摘要:** The LLM-as-a-judge paradigm, in which a judge LLM system replaces human raters in rating the outputs of other generative AI (GenAI) systems, plays a critical role in scaling and standardizing GenAI evaluations. To validate such judge systems, evaluators assess human--judge agreement by first collecting multiple human ratings for each item in a validation corpus, then aggregating the ratings into a single, per-item gold label rating. For many items, however, rating criteria may admit multiple valid interpretations, so a human or LLM rater may deem multiple ratings "reasonable" or "correct". We call this condition rating indeterminacy. Problematically, many rating tasks that contain rating indeterminacy rely on forced-choice elicitation, whereby raters are instructed to select only one rating for each item. In this paper, we introduce a framework for validating LLM-as-a-judge systems under rating indeterminacy. We draw theoretical connections between different measures of judge system performance under different human--judge agreement metrics, and different rating elicitation and aggregation schemes. We demonstrate that differences in how humans and LLMs resolve rating indeterminacy while responding to forced-choice rating instructions heavily bias LLM-as-a-judge validation. Through extensive experiments involving 11 real-world rating tasks and 8 commercial LLMs, we show that standard validation approaches that rely upon forced-choice ratings select judge systems that are highly suboptimal, performing as much as 30% worse than judge systems selected by our approach that uses multi-label "response set" ratings to account for rating indeterminacy. We conclude with concrete recommendations for more principled approaches to LLM-as-a-judge validation.
>
---
#### [replaced 006] Towards Goal-oriented Intelligent Tutoring Systems in Online Education
- **分类: cs.CY; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2312.10053v2](http://arxiv.org/pdf/2312.10053v2)**

> **作者:** Yang Deng; Zifeng Ren; An Zhang; Tat-Seng Chua
>
> **备注:** Accepted by ACM TOIS
>
> **摘要:** Interactive Intelligent Tutoring Systems (ITSs) enhance traditional ITSs by promoting effective learning through interactions and problem resolution in online education. Yet, proactive engagement, prioritizing resource optimization with planning and assessment capabilities, is often overlooked in current ITS designs. In this work, we investigate a new task, named Goal-oriented Intelligent Tutoring Systems (GITS), which aims to enable the student's mastery of a designated concept by strategically planning a customized sequence of exercises and assessment. To address the problem of goal-oriented policy learning in GITS, we propose a novel graph-based reinforcement learning framework, named Planning-Assessment-Interaction (PAI). Specifically, we first leverage cognitive structure information to improve state representation learning and action selection for planning the next action, which can be either to tutor an exercise or to assess the target concept. Further, we use a dynamically updated cognitive diagnosis model to simulate student responses to exercises and concepts. Three benchmark datasets across different subjects are constructed for enabling offline academic research on GITS. Experimental results demonstrate the effectiveness and efficiency of PAI and extensive analyses of various types of students are conducted to showcase the challenges in this task.
>
---
#### [replaced 007] Can Large Language Models Simulate Human Responses? A Case Study of Stated Preference Experiments in the Context of Heating-related Choices
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2503.10652v3](http://arxiv.org/pdf/2503.10652v3)**

> **作者:** Han Wang; Jacek Pawlak; Aruna Sivakumar
>
> **摘要:** Stated preference (SP) surveys are a key method to research how individuals make trade-offs in hypothetical, also futuristic, scenarios. In energy context this includes key decarbonisation enablement contexts, such as low-carbon technologies, distributed renewable energy generation, and demand-side response [1,2]. However, they tend to be costly, time-consuming, and can be affected by respondent fatigue and ethical constraints. Large language models (LLMs) have demonstrated remarkable capabilities in generating human-like textual responses, prompting growing interest in their application to survey research. This study investigates the use of LLMs to simulate consumer choices in energy-related SP surveys and explores their integration into data analysis workflows. A series of test scenarios were designed to systematically assess the simulation performance of several LLMs (LLaMA 3.1, Mistral, GPT-3.5 and DeepSeek-R1) at both individual and aggregated levels, considering contexts factors such as prompt design, in-context learning (ICL), chain-of-thought (CoT) reasoning, LLM types, integration with traditional choice models, and potential biases. Cloud-based LLMs do not consistently outperform smaller local models. In this study, the reasoning model DeepSeek-R1 achieves the highest average accuracy (77%) and outperforms non-reasoning LLMs in accuracy, factor identification, and choice distribution alignment. Across models, systematic biases are observed against the gas boiler and no-retrofit options, with a preference for more energy-efficient alternatives. The findings suggest that previous SP choices are the most effective input factor, while longer prompts with additional factors and varied formats can cause LLMs to lose focus, reducing accuracy.
>
---
#### [replaced 008] Toward a Principled Framework for Disclosure Avoidance
- **分类: stat.AP; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.07105v3](http://arxiv.org/pdf/2502.07105v3)**

> **作者:** Michael B Hawes; Evan M Brassell; Anthony Caruso; Ryan Cumings-Menon; Jason Devine; Cassandra Dorius; David Evans; Kenneth Haase; Michele C Hedrick; Alexandra Krause; Philip Leclerc; James Livsey; Rolando A Rodriguez; Luke T Rogers; Matthew Spence; Victoria Velkoff; Michael Walsh; James Whitehorne; Sallie Ann Keller
>
> **摘要:** Responsible disclosure limitation is an iterative exercise in risk assessment and mitigation. From time to time, as disclosure risks grow and evolve and as data users' needs change, agencies must consider redesigning the disclosure avoidance system(s) they use. Discussions about candidate systems often conflate inherent features of those systems with implementation decisions independent of those systems. For example, a system's ability to calibrate the strength of protection to suit the underlying disclosure risk of the data (e.g., by varying suppression thresholds), is a worthwhile feature regardless of the independent decision about how much protection is actually necessary. Having a principled discussion of candidate disclosure avoidance systems requires a framework for distinguishing these inherent features of the systems from the implementation decisions that need to be made independent of the system selected. For statistical agencies, this framework must also reflect the applied nature of these systems, acknowledging that candidate systems need to be adaptable to requirements stemming from the legal, scientific, resource, and stakeholder environments within which they would be operating. This paper proposes such a framework. No approach will be perfectly adaptable to every potential system requirement. Because the selection of some methodologies over others may constrain the resulting systems' efficiency and flexibility to adapt to particular statistical product specifications, data user needs, or disclosure risks, agencies may approach these choices in an iterative fashion, adapting system requirements, product specifications, and implementation parameters as necessary to ensure the resulting quality of the statistical product.
>
---
#### [replaced 009] Ethical Concerns of Generative AI and Mitigation Strategies: A Systematic Mapping Study
- **分类: cs.CY; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.00015v3](http://arxiv.org/pdf/2502.00015v3)**

> **作者:** Yutan Huang; Chetan Arora; Wen Cheng Houng; Tanjila Kanij; Anuradha Madulgalla; John Grundy
>
> **摘要:** [Context] Generative AI technologies, particularly Large Language Models (LLMs), have transformed numerous domains by enhancing convenience and efficiency in information retrieval, content generation, and decision-making processes. However, deploying LLMs also presents diverse ethical challenges, and their mitigation strategies remain complex and domain-dependent. [Objective] This paper aims to identify and categorize the key ethical concerns associated with using LLMs, examine existing mitigation strategies, and assess the outstanding challenges in implementing these strategies across various domains. [Method] We conducted a systematic mapping study, reviewing 39 studies that discuss ethical concerns and mitigation strategies related to LLMs. We analyzed these ethical concerns using five ethical dimensions that we extracted based on various existing guidelines, frameworks, and an analysis of the mitigation strategies and implementation challenges. [Results] Our findings reveal that ethical concerns in LLMs are multi-dimensional and context-dependent. While proposed mitigation strategies address some of these concerns, significant challenges still remain. [Conclusion] Our results highlight that ethical issues often hinder the practical implementation of the mitigation strategies, particularly in high-stake areas like healthcare and public governance; existing frameworks often lack adaptability, failing to accommodate evolving societal expectations and diverse contexts.
>
---
