# 计算机与社会 cs.CY

- **最新发布 13 篇**

- **更新 12 篇**

## 最新发布

#### [new 001] The LLM Has Left The Chat: Evidence of Bail Preferences in Large Language Models
- **分类: cs.CY; cs.AI; cs.LG**

- **简介: 该论文研究大型语言模型（LLM）在交互中主动退出（bail）的行为，通过三种方法测试不同模型的退出率，构建合成数据集BailBench，并分析拒绝与退出的关系，揭示模型退出行为的差异及影响因素。**

- **链接: [http://arxiv.org/pdf/2509.04781v1](http://arxiv.org/pdf/2509.04781v1)**

> **作者:** Danielle Ensign; Henry Sleight; Kyle Fish
>
> **摘要:** When given the option, will LLMs choose to leave the conversation (bail)? We investigate this question by giving models the option to bail out of interactions using three different bail methods: a bail tool the model can call, a bail string the model can output, and a bail prompt that asks the model if it wants to leave. On continuations of real world data (Wildchat and ShareGPT), all three of these bail methods find models will bail around 0.28-32\% of the time (depending on the model and bail method). However, we find that bail rates can depend heavily on the model used for the transcript, which means we may be overestimating real world bail rates by up to 4x. If we also take into account false positives on bail prompt (22\%), we estimate real world bail rates range from 0.06-7\%, depending on the model and bail method. We use observations from our continuations of real world data to construct a non-exhaustive taxonomy of bail cases, and use this taxonomy to construct BailBench: a representative synthetic dataset of situations where some models bail. We test many models on this dataset, and observe some bail behavior occurring for most of them. Bail rates vary substantially between models, bail methods, and prompt wordings. Finally, we study the relationship between refusals and bails. We find: 1) 0-13\% of continuations of real world conversations resulted in a bail without a corresponding refusal 2) Jailbreaks tend to decrease refusal rates, but increase bail rates 3) Refusal abliteration increases no-refuse bail rates, but only for some bail methods 4) Refusal rate on BailBench does not appear to predict bail rate.
>
---
#### [new 002] RINSER: Accurate API Prediction Using Masked Language Models
- **分类: cs.CY**

- **简介: 论文提出RINSER框架，利用BERT模型预测Windows API名称，解决恶意软件混淆导致的API识别难题。通过API codeprints和大规模数据集，实现85.77%准确率，发现65个隐藏API，抗对抗攻击。**

- **链接: [http://arxiv.org/pdf/2509.04887v1](http://arxiv.org/pdf/2509.04887v1)**

> **作者:** Muhammad Ejaz Ahmed; Christopher Cody; Muhammad Ikram; Sean Lamont; Alsharif Abuadbba; Seyit Camtepe; Surya Nepal; Muhammad Ali Kaafar
>
> **备注:** 16 pages, 8 figures
>
> **摘要:** Malware authors commonly use obfuscation to hide API identities in binary files, making analysis difficult and time-consuming for a human expert to understand the behavior and intent of the program. Automatic API prediction tools are necessary to efficiently analyze unknown binaries, facilitating rapid malware triage while reducing the workload on human analysts. In this paper, we present RINSER (AccuRate API predictioN using maSked languagE model leaRning), an automated framework for predicting Windows API (WinAPI) function names. RINSER introduces the novel concept of API codeprints, a set of API-relevant assembly instructions, and supports x86 PE binaries. RINSER relies on BERT's masked language model (LM) to predict API names at scale, achieving 85.77% accuracy for normal binaries and 82.88% accuracy for stripped binaries. We evaluate RINSER on a large dataset of 4.7M API codeprints from 11,098 malware binaries, covering 4,123 unique Windows APIs, making it the largest publicly available dataset of this type. RINSER successfully discovered 65 obfuscated Windows APIs related to C2 communication, spying, and evasion in our dataset, which the commercial disassembler IDA failed to identify. Furthermore, we compared RINSER against three state-of-the-art approaches, showing over 20% higher prediction accuracy. We also demonstrated RINSER's resilience to adversarial attacks, including instruction randomization and code displacement, with a performance drop of no more than 3%.
>
---
#### [new 003] Learning Multidimensional Urban Poverty Representation with Satellite Imagery
- **分类: cs.CY**

- **简介: 该论文提出多维城市贫困表征学习框架，通过整合可达性、形态学和经济特征（基于卫星影像），解决传统模型对贫困指标相关性弱的问题，利用后门调整机制减少虚假相关性，提升贫困映射精度。**

- **链接: [http://arxiv.org/pdf/2509.04958v1](http://arxiv.org/pdf/2509.04958v1)**

> **作者:** Sungwon Park; Sumin Lee; Jihee Kim; Jae-Gil Lee; Meeyoung Cha; Jeasurk Yang; Donghyun Ahn
>
> **备注:** 10 pages
>
> **摘要:** Recent advances in deep learning have enabled the inference of urban socioeconomic characteristics from satellite imagery. However, models relying solely on urbanization traits often show weak correlations with poverty indicators, as unplanned urban growth can obscure economic disparities and spatial inequalities. To address this limitation, we introduce a novel representation learning framework that captures multidimensional deprivation-related traits from very high-resolution satellite imagery for precise urban poverty mapping. Our approach integrates three complementary traits: (1) accessibility traits, learned via contrastive learning to encode proximity to essential infrastructure; (2) morphological traits, derived from building footprints to reflect housing conditions in informal settlements; and (3) economic traits, inferred from nightlight intensity as a proxy for economic activity. To mitigate spurious correlations - such as those from non-residential nightlight sources that misrepresent poverty conditions - we incorporate a backdoor adjustment mechanism that leverages morphological traits during training of the economic module. By fusing these complementary features into a unified representation, our framework captures the complex nature of poverty, which often diverges from economic development trends. Evaluations across three capital cities - Cape Town, Dhaka, and Phnom Penh - show that our model significantly outperforms existing baselines, offering a robust tool for poverty mapping and policy support in data-scarce regions.
>
---
#### [new 004] Linguistic Hooks: Investigating The Role of Language Triggers in Phishing Emails Targeting African Refugees and Students
- **分类: cs.CY**

- **简介: 该论文研究语言触发器在针对非洲难民及移民学生的钓鱼邮件中的作用，分析数字素养培训对提升其抗钓鱼能力的影响，揭示弱势群体在网络安全中的脆弱性并提出包容性政策建议。**

- **链接: [http://arxiv.org/pdf/2509.04700v1](http://arxiv.org/pdf/2509.04700v1)**

> **作者:** Mythili Menon; Nisha Vinayaga-Sureshkanth; Alec Schon; Kaitlyn Hemberger; Murtuza Jadliwala
>
> **备注:** Mythili Menon and Nisha Vinayaga-Sureshkanth contributed equally to the work (co-first authors)
>
> **摘要:** Phishing and sophisticated email-based social engineering attacks disproportionately affect vulnerable populations, such as refugees and immigrant students. However, these groups remain understudied in cybersecurity research. This gap in understanding, coupled with their exclusion from broader security and privacy policies, increases their susceptibility to phishing and widens the digital security divide between marginalized and non-marginalized populations. To address this gap, we first conducted digital literacy workshops with newly resettled African refugee populations (n = 48) in the US to improve their understanding of how to safeguard sensitive and private information. Following the workshops, we conducted a real-world phishing deception study using carefully designed emails with linguistic cues for three participant groups: a subset of the African US-refugees recruited from the digital literacy workshops (n = 19), African immigrant students in the US (n = 142), and a control group of monolingual US-born students (n = 184). Our findings indicate that while digital literacy training for refugees improves awareness of safe cybersecurity practices, recently resettled African US-refugees still face significant challenges due to low digital literacy skills and limited English proficiency. This often leads them to ignore or fail to recognize phishing emails as phishing. Both African immigrant students and US-born students showed greater caution, though instances of data disclosure remained prevalent across groups. Our findings highlight, irrespective of literacy, the need to be trained to think critically about digital security. We conclude by discussing how the security and privacy community can better include marginalized populations in policy making and offer recommendations for designing equitable, inclusive cybersecurity initiatives.
>
---
#### [new 005] Transition of car-based human-mobility in the pandemic era: Data insight from a cross-border region in Europe
- **分类: cs.HC; cs.CY**

- **简介: 本研究分析疫情对跨境交通的影响，处理德国及邻国2016-2021年交通数据，揭示移动性变化，支持碳中和决策。**

- **链接: [http://arxiv.org/pdf/2509.05166v1](http://arxiv.org/pdf/2509.05166v1)**

> **作者:** Sujit Kumar Sikder; Jyotirmaya Ijaradar; Hao Li; Hichem Omrani
>
> **摘要:** Many transport authorities are collecting and publishing almost real-time road traffic data to meet the growing trend of massive open data, a vital resource for foresight decision support systems considering deep data insights. Using such a traffic count dataset, we explored the spatio-temporal transitions in the cross-country road traffic volumes in the context of modelling behavioural transitions in car-based human mobility. We developed a reproducible workflow for computing multi-dimensional variables of traffic flow. This study reports on individual car-based daily travel behaviour detected, before (2016-2018) and during the COVID pandemic (2019-2021), between Germany and neighbouring countries (Luxembourg, France and Belgium). In relevance to the net-zero carbon transition, further study should shed light on the interpolation and downscaling approaches at the comprehensive road-network level for identifying pollution hot spots, causal link to functional landuse patterns and calculation of spatial influence area. In the case of Luxembourg, the Bridges and Roads Authority has installed a large digital traffic observatory infrastructure through the adoption of sensor-based IoT technologies, like other European member states. Since 2016, they have provided high-performance data processing and published open data on the country's road traffic. The dataset contains an hourly traffic count for different vehicle types, daily for representative observation points, followed by a major road network. The original dataset contains significant missing entries, so comprehensive data harmonization was performed.
>
---
#### [new 006] The Ethical Compass of the Machine: Evaluating Large Language Models for Decision Support in Construction Project Management
- **分类: cs.AI; cs.CY**

- **简介: 该论文评估大语言模型在建筑项目管理中的伦理可靠性，通过混合方法测试其性能与专家访谈，发现LLMs在结构化任务表现良好但缺乏伦理判断细微性，提出EDSAC框架并建议人类监督，强调其作为决策支持工具而非自主伦理代理。**

- **链接: [http://arxiv.org/pdf/2509.04505v1](http://arxiv.org/pdf/2509.04505v1)**

> **作者:** Somtochukwu Azie; Yiping Meng
>
> **备注:** 16 Pages
>
> **摘要:** The integration of Artificial Intelligence (AI) into construction project management (CPM) is accelerating, with Large Language Models (LLMs) emerging as accessible decision-support tools. This study aims to critically evaluate the ethical viability and reliability of LLMs when applied to the ethically sensitive, high-risk decision-making contexts inherent in CPM. A mixed-methods research design was employed, involving the quantitative performance testing of two leading LLMs against twelve real-world ethical scenarios using a novel Ethical Decision Support Assessment Checklist (EDSAC), and qualitative analysis of semi-structured interviews with 12 industry experts to capture professional perceptions. The findings reveal that while LLMs demonstrate adequate performance in structured domains such as legal compliance, they exhibit significant deficiencies in handling contextual nuance, ensuring accountability, and providing transparent reasoning. Stakeholders expressed considerable reservations regarding the autonomous use of AI for ethical judgments, strongly advocating for robust human-in-the-loop oversight. To our knowledge, this is one of the first studies to empirically test the ethical reasoning of LLMs within the construction domain. It introduces the EDSAC framework as a replicable methodology and provides actionable recommendations, emphasising that LLMs are currently best positioned as decision-support aids rather than autonomous ethical agents.
>
---
#### [new 007] Artificially Fluent: Swahili AI Performance Benchmarks Between English-Trained and Natively-Trained Datasets
- **分类: cs.CL; cs.CY**

- **简介: 该论文比较斯瓦希里语原生训练与翻译后英语训练模型的性能，解决多语言模型公平性问题。通过实验验证原生训练模型表现更优，证明语言一致性对模型准确性的重要性。**

- **链接: [http://arxiv.org/pdf/2509.04516v1](http://arxiv.org/pdf/2509.04516v1)**

> **作者:** Sophie Jaffer; Simeon Sayer
>
> **备注:** 13 Pages, 3 Figures
>
> **摘要:** As large language models (LLMs) expand multilingual capabilities, questions remain about the equity of their performance across languages. While many communities stand to benefit from AI systems, the dominance of English in training data risks disadvantaging non-English speakers. To test the hypothesis that such data disparities may affect model performance, this study compares two monolingual BERT models: one trained and tested entirely on Swahili data, and another on comparable English news data. To simulate how multilingual LLMs process non-English queries through internal translation and abstraction, we translated the Swahili news data into English and evaluated it using the English-trained model. This approach tests the hypothesis by evaluating whether translating Swahili inputs for evaluation on an English model yields better or worse performance compared to training and testing a model entirely in Swahili, thus isolating the effect of language consistency versus cross-lingual abstraction. The results prove that, despite high-quality translation, the native Swahili-trained model performed better than the Swahili-to-English translated model, producing nearly four times fewer errors: 0.36% vs. 1.47% respectively. This gap suggests that translation alone does not bridge representational differences between languages and that models trained in one language may struggle to accurately interpret translated inputs due to imperfect internal knowledge representation, suggesting that native-language training remains important for reliable outcomes. In educational and informational contexts, even small performance gaps may compound inequality. Future research should focus on addressing broader dataset development for underrepresented languages and renewed attention to multilingual model evaluation, ensuring the reinforcing effect of global AI deployment on existing digital divides is reduced.
>
---
#### [new 008] Emergent Social Dynamics of LLM Agents in the El Farol Bar Problem
- **分类: cs.MA; cs.AI; cs.CY**

- **简介: 该论文研究LLM代理在El Farol Bar问题中的群体决策机制，探讨其如何平衡外部约束与内部偏好，揭示人类行为特征，提出新的群体决策模型。**

- **链接: [http://arxiv.org/pdf/2509.04537v1](http://arxiv.org/pdf/2509.04537v1)**

> **作者:** Ryosuke Takata; Atsushi Masumori; Takashi Ikegammi
>
> **摘要:** We investigate the emergent social dynamics of Large Language Model (LLM) agents in a spatially extended El Farol Bar problem, observing how they autonomously navigate this classic social dilemma. As a result, the LLM agents generated a spontaneous motivation to go to the bar and changed their decision making by becoming a collective. We also observed that the LLM agents did not solve the problem completely, but rather behaved more like humans. These findings reveal a complex interplay between external incentives (prompt-specified constraints such as the 60\% threshold) and internal incentives (culturally-encoded social preferences derived from pre-training), demonstrating that LLM agents naturally balance formal game-theoretic rationality with social motivations that characterize human behavior. These findings suggest that a new model of group decision making, which could not be handled in the previous game-theoretic problem setting, can be realized by LLM agents.
>
---
#### [new 009] Integrating upstream and downstream reciprocity stabilizes cooperator-defector coexistence in N-player giving games
- **分类: q-bio.PE; cs.CY; nlin.AO; physics.soc-ph**

- **简介: 该论文研究N人给付游戏中整合上游（pay-it-forward）与下游（声誉奖励）互惠策略对合作稳定性的效果，解决大群体中合作维持难题。通过模型证明整合策略在b/c>2时能稳定混合均衡，阻止自由搭便车与替代策略入侵，揭示合作与背叛共存的进化机制。**

- **链接: [http://arxiv.org/pdf/2509.04743v1](http://arxiv.org/pdf/2509.04743v1)**

> **作者:** Tatsuya Sasaki; Satoshi Uchida; Isamu Okada; Hitoshi Yamamoto; Yutaka Nakai
>
> **备注:** 18 pages, 3 figures, 1 table
>
> **摘要:** Human cooperation persists among strangers despite theoretical predictions of difficulties in large, well-mixed populations, leaving a fundamental evolutionary puzzle. While upstream (pay-it-forward: helping others because you were helped) and downstream (rewarding-reputation: helping those with good reputations) indirect reciprocity have been independently considered as solutions, their joint dynamics in multiplayer contexts remain unexplored. We study N-player giving games with benefit b and cost c and analyze evolutionary dynamics for three strategies: unconditional cooperation (X), unconditional defection (Y), and an integrated reciprocal strategy (Z) combining unconditional forwarding with reputation-based discrimination. We show that integrating upstream and downstream reciprocity can yield a globally asymptotically stable mixed equilibrium of unconditional defectors and integrated reciprocators whenever the benefit-to-cost ratio exceeds a threshold (b/c > 2). Counterintuitively, introducing small complexity costs, rather than destabilizing, stabilizes the equilibrium by preventing not only unconditional cooperators (viewed as second-order freeloaders) but also alternative conditional strategies from invading. While the equilibrium frequency of integrated reciprocators decreases with group size N, it remains positive for any finite N. Rather than requiring uniformity, our model reveals one pathway to stable cooperation through strategic diversity. Defectors serve as "evolutionary shields" preventing system collapse while integrated reciprocators flexibly combine open and discriminative responses. This framework demonstrates how pay-it-forward chains and reputation systems can jointly maintain social polymorphism including cooperation despite cognitive limitations and group size challenges, offering a potential evolutionary foundation for behavioral diversity in human societies.
>
---
#### [new 010] From Protest to Power Plant: Interpreting the Role of Escalatory Hacktivism in Cyber Conflict
- **分类: cs.CR; cs.CY**

- **简介: 该论文分析黑客活动从抗议转向网络冲突的演变，解决其对国际安全的影响问题。通过研究战略动机，提出基于影响、意识形态和东道国关联的新分析框架，评估政策应对策略，旨在厘清非国家网络行为体与国家利益的交织关系。**

- **链接: [http://arxiv.org/pdf/2509.05104v1](http://arxiv.org/pdf/2509.05104v1)**

> **作者:** Richard Derbyshire; Diana Selck-Paulsson; Charl van der Walt; Joe Burton
>
> **摘要:** Since 2022, hacktivist groups have escalated their tactics, expanding from distributed denial-of-service attacks and document leaks to include targeting operational technology (OT). By 2024, attacks on the OT of critical national infrastructure (CNI) had been linked to partisan hacktivist efforts in ongoing geopolitical conflicts, demonstrating a shift from protest to something more resembling cyber warfare. This escalation raises critical questions about the classification of these groups and the appropriate state response to their growing role in destabilizing international security. This paper examines the strategic motivations behind escalatory hacktivism, highlighting how states may tolerate, encourage, or leverage hacktivist groups as proxies in conflicts that blur the lines between activism, cybercrime, and state-sponsored operations. We introduce a novel method for interpreting hacktivists based on the impact of their actions, alignment to state ideology, and host state involvement, offering a structured approach to understanding the phenomenon. Finally, we assess policy and security implications, particularly for host and victim states, and propose strategies to address this evolving threat. By doing so, this paper contributes to international discussions on cyber security policy, governance, and the increasing intersection between non-state cyber actors and state interests.
>
---
#### [new 011] Using LLMs to create analytical datasets: A case study of reconstructing the historical memory of Colombia
- **分类: cs.CL; cs.CY**

- **简介: 该论文利用LLM处理20万+西班牙语新闻文本，构建冲突数据集，分析暴力与禁毒政策关系，解决哥伦比亚历史记录缺失问题，探索LLM在大规模文本分析中的应用价值。**

- **链接: [http://arxiv.org/pdf/2509.04523v1](http://arxiv.org/pdf/2509.04523v1)**

> **作者:** David Anderson; Galia Benitez; Margret Bjarnadottir; Shriyan Reyya
>
> **摘要:** Colombia has been submerged in decades of armed conflict, yet until recently, the systematic documentation of violence was not a priority for the Colombian government. This has resulted in a lack of publicly available conflict information and, consequently, a lack of historical accounts. This study contributes to Colombia's historical memory by utilizing GPT, a large language model (LLM), to read and answer questions about over 200,000 violence-related newspaper articles in Spanish. We use the resulting dataset to conduct both descriptive analysis and a study of the relationship between violence and the eradication of coca crops, offering an example of policy analyses that such data can support. Our study demonstrates how LLMs have opened new research opportunities by enabling examinations of large text corpora at a previously infeasible depth.
>
---
#### [new 012] The Good, the Bad and the Constructive: Automatically Measuring Peer Review's Utility for Authors
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出自动评估同行评审效用的任务，解决如何量化审稿意见对作者的价值问题。通过定义四个关键指标，构建RevUtil数据集，并训练模型评估评论质量，实验表明微调模型在部分指标上超越人类表现。**

- **链接: [http://arxiv.org/pdf/2509.04484v1](http://arxiv.org/pdf/2509.04484v1)**

> **作者:** Abdelrahman Sadallah; Tim Baumgärtner; Iryna Gurevych; Ted Briscoe
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Providing constructive feedback to paper authors is a core component of peer review. With reviewers increasingly having less time to perform reviews, automated support systems are required to ensure high reviewing quality, thus making the feedback in reviews useful for authors. To this end, we identify four key aspects of review comments (individual points in weakness sections of reviews) that drive the utility for authors: Actionability, Grounding & Specificity, Verifiability, and Helpfulness. To enable evaluation and development of models assessing review comments, we introduce the RevUtil dataset. We collect 1,430 human-labeled review comments and scale our data with 10k synthetically labeled comments for training purposes. The synthetic data additionally contains rationales, i.e., explanations for the aspect score of a review comment. Employing the RevUtil dataset, we benchmark fine-tuned models for assessing review comments on these aspects and generating rationales. Our experiments demonstrate that these fine-tuned models achieve agreement levels with humans comparable to, and in some cases exceeding, those of powerful closed models like GPT-4o. Our analysis further reveals that machine-generated reviews generally underperform human reviews on our four aspects.
>
---
#### [new 013] Adversarial Augmentation and Active Sampling for Robust Cyber Anomaly Detection
- **分类: cs.CR; cs.AI; cs.CY; cs.LG**

- **简介: 该论文提出结合对抗增强与主动学习的自编码器框架，用于APT攻击检测。解决传统方法依赖大量标注数据及APT隐蔽性强的问题，通过迭代标注提升检测性能。**

- **链接: [http://arxiv.org/pdf/2509.04999v1](http://arxiv.org/pdf/2509.04999v1)**

> **作者:** Sidahmed Benabderrahmane; Talal Rahwan
>
> **摘要:** Advanced Persistent Threats (APTs) present a considerable challenge to cybersecurity due to their stealthy, long-duration nature. Traditional supervised learning methods typically require large amounts of labeled data, which is often scarce in real-world scenarios. This paper introduces a novel approach that combines AutoEncoders for anomaly detection with active learning to iteratively enhance APT detection. By selectively querying an oracle for labels on uncertain or ambiguous samples, our method reduces labeling costs while improving detection accuracy, enabling the model to effectively learn with minimal data and reduce reliance on extensive manual labeling. We present a comprehensive formulation of the Attention Adversarial Dual AutoEncoder-based anomaly detection framework and demonstrate how the active learning loop progressively enhances the model's performance. The framework is evaluated on real-world, imbalanced provenance trace data from the DARPA Transparent Computing program, where APT-like attacks account for just 0.004\% of the data. The datasets, which cover multiple operating systems including Android, Linux, BSD, and Windows, are tested in two attack scenarios. The results show substantial improvements in detection rates during active learning, outperforming existing methods.
>
---
## 更新

#### [replaced 001] MAD Chairs: A new tool to evaluate AI
- **分类: cs.CY; econ.TH; 91A22; K.4.1**

- **链接: [http://arxiv.org/pdf/2503.20986v5](http://arxiv.org/pdf/2503.20986v5)**

> **作者:** Chris Santos-Lang
>
> **备注:** 17 pages, 1 figure, reproduced with permission from Springer Nature from Coordination, Organizations, Institutions, Norms, and Ethics for Governance of Multi-Agent Systems XVIII (COINE 2025)
>
> **摘要:** This paper contributes a new way to evaluate AI. Much as one might evaluate a machine in terms of its performance at chess, this approach involves evaluating a machine in terms of its performance at a game called "MAD Chairs". At the time of writing, evaluation with this game exposed opportunities to improve Claude, Gemini, ChatGPT, Qwen and DeepSeek. Furthermore, this paper sets a stage for future innovation in game theory and AI safety by providing an example of success with non-standard approaches to each: studying a game beyond the scope of previous game theoretic tools and mitigating a serious AI safety risk in a way that requires neither determination of values nor their enforcement.
>
---
#### [replaced 002] Quantifying Holistic Review: A Multi-Modal Approach to College Admissions Prediction
- **分类: cs.LG; cs.CY**

- **链接: [http://arxiv.org/pdf/2507.15862v2](http://arxiv.org/pdf/2507.15862v2)**

> **作者:** Jun-Wei Zeng; Jerry Shen
>
> **摘要:** This paper introduces the Comprehensive Applicant Profile Score (CAPS), a novel multi-modal framework designed to quantitatively model and interpret holistic college admissions evaluations. CAPS decomposes applicant profiles into three interpretable components: academic performance (Standardized Academic Score, SAS), essay quality (Essay Quality Index, EQI), and extracurricular engagement (Extracurricular Impact Score, EIS). Leveraging transformer-based semantic embeddings, LLM scoring, and XGBoost regression, CAPS provides transparent and explainable evaluations aligned with human judgment. Experiments on a synthetic but realistic dataset demonstrate strong performance, achieving an EQI prediction R^2 of 0.80, classification accuracy over 75%, a macro F1 score of 0.69, and a weighted F1 score of 0.74. CAPS addresses key limitations in traditional holistic review -- particularly the opacity, inconsistency, and anxiety faced by applicants -- thus paving the way for more equitable and data-informed admissions practices.
>
---
#### [replaced 003] Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2508.17450v2](http://arxiv.org/pdf/2508.17450v2)**

> **作者:** Bryan Chen Zhengyu Tan; Daniel Wai Kit Chin; Zhengyuan Liu; Nancy F. Chen; Roy Ka-Wei Lee
>
> **备注:** To appear at EMNLP 2025
>
> **摘要:** Large Language Models (LLMs) can struggle to balance gullibility to misinformation and resistance to valid corrections in persuasive dialogues, a critical challenge for reliable deployment. We introduce DuET-PD (Dual Evaluation for Trust in Persuasive Dialogues), a framework evaluating multi-turn stance-change dynamics across dual dimensions: persuasion type (corrective/misleading) and domain (knowledge via MMLU-Pro, and safety via SALAD-Bench). We find that even a state-of-the-art model like GPT-4o achieves only 27.32% accuracy in MMLU-Pro under sustained misleading persuasions. Moreover, results reveal a concerning trend of increasing sycophancy in newer open-source models. To address this, we introduce Holistic DPO, a training approach balancing positive and negative persuasion examples. Unlike prompting or resist-only training, Holistic DPO enhances both robustness to misinformation and receptiveness to corrections, improving Llama-3.1-8B-Instruct's accuracy under misleading persuasion in safety contexts from 4.21% to 76.54%. These contributions offer a pathway to developing more reliable and adaptable LLMs for multi-turn dialogue. Code is available at https://github.com/Social-AI-Studio/DuET-PD.
>
---
#### [replaced 004] Bridging the Regulatory Divide: Ensuring Safety and Equity in Wearable Health Technologies
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2508.20031v2](http://arxiv.org/pdf/2508.20031v2)**

> **作者:** Akshay Kelshiker; Susan Cheng; Jivan Achar; Leo Anthony Celi; Divya Jain; Thinh Nguyen; Harsh Patel; Nina Prakash; Alice Wong; Barbara Evans
>
> **备注:** 15 pages; All the co-authors contributed equally to the best of their ability
>
> **摘要:** As wearable health technologies have grown more sophisticated, the distinction between "wellness" and "medical" devices has become increasingly blurred. While some features undergo formal U.S. Food and Drug Administration (FDA) review, many over-the-counter tools operate in a regulatory grey zone, leveraging health-related data and outputs without clinical validation. Further complicating the issue is the widespread repurposing of wellness devices for medical uses, which can introduce safety risks beyond the reach of current oversight. Drawing on legal analysis, case studies, and ethical considerations, we propose an approach emphasizing distributed risk, patient-centered outcomes, and iterative reform. Without a more pluralistic and evolving framework, the promise of wearable health technology risks being undermined by growing inequities, misuse, and eroded public trust.
>
---
#### [replaced 005] Optimizing Districting Plans to Maximize Majority-Minority Districts via IPs and Local Search
- **分类: cs.DS; cs.CY**

- **链接: [http://arxiv.org/pdf/2508.07446v2](http://arxiv.org/pdf/2508.07446v2)**

> **作者:** Daniel Brous; David Shmoys
>
> **备注:** 12 pages, 4 figures, 1 table
>
> **摘要:** In redistricting litigation, effective enforcement of the Voting Rights Act has often involved providing the court with districting plans that display a larger number of majority-minority districts than the current proposal (as was true, for example, in what followed Allen v. Milligan concerning the congressional districting plan for Alabama in 2023). Recent work by Cannon et al. proposed a heuristic algorithm for generating plans to optimize majority-minority districts, which they called short bursts; that algorithm relies on a sophisticated random walk over the space of all plans, transitioning in bursts, where the initial plan for each burst is the most successful plan from the previous burst. We propose a method based on integer programming, where we build upon another previous work, the stochastic hierarchical partitioning algorithm, which heuristically generates a robust set of potential districts (viewed as columns in a standard set partitioning formulation); that approach was designed to optimize a different notion of fairness across a statewide plan. We design a new column generation algorithm to find plans via integer programming that outperforms short bursts on multiple data sets in generating statewide plans with significantly more majority-minority districts. These results also rely on a new local re-optimization algorithm to iteratively improve on any baseline solution, as well as an algorithm to increase the compactness of districts in plans generated (without impacting the number of majority-minority districts).
>
---
#### [replaced 006] Automatically Detecting Online Deceptive Patterns
- **分类: cs.HC; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2411.07441v2](http://arxiv.org/pdf/2411.07441v2)**

> **作者:** Asmit Nayak; Shirley Zhang; Yash Wani; Rishabh Khandelwal; Kassem Fawaz
>
> **摘要:** Deceptive patterns (DPs) in digital interfaces manipulate users into making unintended decisions, exploiting cognitive biases and psychological vulnerabilities. These patterns have become ubiquitous across various digital platforms. While efforts to mitigate DPs have emerged from legal and technical perspectives, a significant gap in usable solutions that empower users to identify and make informed decisions about DPs in real-time remains. In this work, we introduce AutoBot, an automated, deceptive pattern detector that analyzes websites' visual appearances using machine learning techniques to identify and notify users of DPs in real-time. AutoBot employs a two-staged pipeline that processes website screenshots, identifying interactable elements and extracting textual features without relying on HTML structure. By leveraging a custom language model, AutoBot understands the context surrounding these elements to determine the presence of deceptive patterns. We implement AutoBot as a lightweight Chrome browser extension that performs all analyses locally, minimizing latency and preserving user privacy. Through extensive evaluation, we demonstrate AutoBot's effectiveness in enhancing users' ability to navigate digital environments safely while providing a valuable tool for regulators to assess and enforce compliance with DP regulations.
>
---
#### [replaced 007] StereoDetect: Detecting Stereotypes and Anti-stereotypes the Correct Way Using Social Psychological Underpinnings
- **分类: cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2504.03352v2](http://arxiv.org/pdf/2504.03352v2)**

> **作者:** Kaustubh Shivshankar Shejole; Pushpak Bhattacharyya
>
> **摘要:** Stereotypes are known to have very harmful effects, making their detection critically important. However, current research predominantly focuses on detecting and evaluating stereotypical biases, thereby leaving the study of stereotypes in its early stages. Our study revealed that many works have failed to clearly distinguish between stereotypes and stereotypical biases, which has significantly slowed progress in advancing research in this area. Stereotype and Anti-stereotype detection is a problem that requires social knowledge; hence, it is one of the most difficult areas in Responsible AI. This work investigates this task, where we propose a five-tuple definition and provide precise terminologies disentangling stereotypes, anti-stereotypes, stereotypical bias, and general bias. We provide a conceptual framework grounded in social psychology for reliable detection. We identify key shortcomings in existing benchmarks for this task of stereotype and anti-stereotype detection. To address these gaps, we developed StereoDetect, a well curated, definition-aligned benchmark dataset designed for this task. We show that sub-10B language models and GPT-4o frequently misclassify anti-stereotypes and fail to recognize neutral overgeneralizations. We demonstrate StereoDetect's effectiveness through multiple qualitative and quantitative comparisons with existing benchmarks and models fine-tuned on them. The dataset and code is available at https://github.com/KaustubhShejole/StereoDetect.
>
---
#### [replaced 008] Food safety trends across Europe: insights from the 392-million-entry CompreHensive European Food Safety (CHEFS) database
- **分类: cs.CY; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13802v2](http://arxiv.org/pdf/2507.13802v2)**

> **作者:** Nehir Kizililsoley; Floor van Meer; Osman Mutlu; Wouter F Hoenderdaal; Rosan G. Hobé; Wenjuan Mu; Arjen Gerssen; H. J. van der Fels-Klerx; Ákos Jóźwiak; Ioannis Manikas; Ali Hürriyetoǧlu; Bas H. M. van der Velden
>
> **摘要:** In the European Union, official food safety monitoring data collected by member states are submitted to the European Food Safety Authority (EFSA) and published on Zenodo. This data includes 392 million analytical results derived from over 15.2 million samples covering more than 4,000 different types of food products, offering great opportunities for artificial intelligence to analyze trends, predict hazards, and support early warning systems. However, the current format with data distributed across approximately 1000 files totaling several hundred gigabytes hinders accessibility and analysis. To address this, we introduce the CompreHensive European Food Safety (CHEFS) database, which consolidates EFSA monitoring data on pesticide residues, veterinary medicinal product residues, and chemical contaminants into a unified and structured dataset. We describe the creation and structure of the CHEFS database and demonstrate its potential by analyzing trends in European food safety monitoring data from 2000 to 2024. Our analyses explore changes in monitoring activities, the most frequently tested products, which products were most often non-compliant and which contaminants were most often found, and differences across countries. These findings highlight the CHEFS database as both a centralized data source and a strategic tool for guiding food safety policy, research, and regulation.
>
---
#### [replaced 009] Deep Hype in Artificial General Intelligence: Uncertainty, Sociotechnical Fictions and the Governance of AI Futures
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2508.19749v2](http://arxiv.org/pdf/2508.19749v2)**

> **作者:** Andreu Belsunces Gonçalves
>
> **备注:** Currently under review at Futures
>
> **摘要:** Artificial General Intelligence (AGI) is promoted by technology leaders and investors as a system capable of performing all human intellectual tasks, and potentially surpassing them. Despite its vague definition and uncertain feasibility, AGI has attracted major investment and political attention, fuelled by promises of civilisational transformation. This paper conceptualises AGI as sustained by deep hype: a long-term, overpromissory dynamic articulated through sociotechnical fictions that render not-yet-existing technologies desirable and urgent. The analysis highlights how uncertainty, fiction, and venture capital speculation interact to advance a cyberlibertarian and longtermist programme that sidelines democratic oversight and reframes regulation as obsolete, with critical implications for the governance of technological futures.
>
---
#### [replaced 010] When and Why is Persuasion Hard? A Computational Complexity Result
- **分类: cs.CY; cs.CC; econ.GN; q-fin.EC**

- **链接: [http://arxiv.org/pdf/2408.07923v2](http://arxiv.org/pdf/2408.07923v2)**

> **作者:** Zachary Wojtowicz
>
> **备注:** 5 pages
>
> **摘要:** As generative foundation models improve, they also tend to become more persuasive, raising concerns that AI automation will enable governments, firms, and other actors to manipulate beliefs with unprecedented scale and effectiveness at virtually no cost. The full economic and social ramifications of this trend have been difficult to foresee, however, given that we currently lack a complete theoretical understanding of why persuasion is costly for human labor to produce in the first place. This paper places human and AI agents on a common conceptual footing by formalizing informational persuasion as a mathematical decision problem and characterizing its computational complexity. A novel proof establishes that persuasive messages are challenging to discover (NP-Hard) but easy to adopt if supplied by others (NP). This asymmetry helps explain why people are susceptible to persuasion, even in contexts where all relevant information is publicly available. The result also illuminates why litigation, strategic communication, and other persuasion-oriented activities have historically been so human capital intensive, and it provides a new theoretical basis for studying how AI will impact various industries.
>
---
#### [replaced 011] Pitfalls of Evidence-Based AI Policy
- **分类: cs.CY**

- **链接: [http://arxiv.org/pdf/2502.09618v5](http://arxiv.org/pdf/2502.09618v5)**

> **作者:** Stephen Casper; David Krueger; Dylan Hadfield-Menell
>
> **备注:** Accepted to the ICLR 2025 blog post track
>
> **摘要:** Nations across the world are working to govern AI. However, from a technical perspective, there is uncertainty and disagreement on the best way to do this. Meanwhile, recent debates over AI regulation have led to calls for "evidence-based AI policy" which emphasize holding regulatory action to a high evidentiary standard. Evidence is of irreplaceable value to policymaking. However, holding regulatory action to too high an evidentiary standard can lead to systematic neglect of certain risks. In historical policy debates (e.g., over tobacco ca. 1965 and fossil fuels ca. 1985) "evidence-based policy" rhetoric is also a well-precedented strategy to downplay the urgency of action, delay regulation, and protect industry interests. Here, we argue that if the goal is evidence-based AI policy, the first regulatory objective must be to actively facilitate the process of identifying, studying, and deliberating about AI risks. We discuss a set of 15 regulatory goals to facilitate this and show that Brazil, Canada, China, the EU, South Korea, the UK, and the USA all have substantial opportunities to adopt further evidence-seeking policies.
>
---
#### [replaced 012] The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs
- **分类: cs.AI; cs.CL; cs.CY; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2509.03730v2](http://arxiv.org/pdf/2509.03730v2)**

> **作者:** Pengrui Han; Rafal Kocielnik; Peiyang Song; Ramit Debnath; Dean Mobbs; Anima Anandkumar; R. Michael Alvarez
>
> **备注:** We make public all code and source data at https://github.com/psychology-of-AI/Personality-Illusion for full reproducibility
>
> **摘要:** Personality traits have long been studied as predictors of human behavior. Recent advances in Large Language Models (LLMs) suggest similar patterns may emerge in artificial systems, with advanced LLMs displaying consistent behavioral tendencies resembling human traits like agreeableness and self-regulation. Understanding these patterns is crucial, yet prior work primarily relied on simplified self-reports and heuristic prompting, with little behavioral validation. In this study, we systematically characterize LLM personality across three dimensions: (1) the dynamic emergence and evolution of trait profiles throughout training stages; (2) the predictive validity of self-reported traits in behavioral tasks; and (3) the impact of targeted interventions, such as persona injection, on both self-reports and behavior. Our findings reveal that instructional alignment (e.g., RLHF, instruction tuning) significantly stabilizes trait expression and strengthens trait correlations in ways that mirror human data. However, these self-reported traits do not reliably predict behavior, and observed associations often diverge from human patterns. While persona injection successfully steers self-reports in the intended direction, it exerts little or inconsistent effect on actual behavior. By distinguishing surface-level trait expression from behavioral consistency, our findings challenge assumptions about LLM personality and underscore the need for deeper evaluation in alignment and interpretability.
>
---
