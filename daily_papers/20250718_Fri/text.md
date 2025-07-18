# 自然语言处理 cs.CL

- **最新发布 54 篇**

- **更新 43 篇**

## 最新发布

#### [new 001] Multi-Agent Synergy-Driven Iterative Visual Narrative Synthesis
- **分类: cs.CL; 68T50, 68T07; I.2.7; I.2.11; H.5.2**

- **简介: 该论文属于媒体生成任务，旨在解决自动创作高质量演示文稿中的逻辑不一致和布局不佳问题。提出RCPS框架与PREVAL评估体系，提升生成质量与评估准确性。**

- **链接: [http://arxiv.org/pdf/2507.13285v1](http://arxiv.org/pdf/2507.13285v1)**

> **作者:** Wang Xi; Quan Shi; Tian Yu; Yujie Peng; Jiayi Sun; Mengxing Ren; Zenghui Ding; Ningguang Yao
>
> **备注:** 22 pages, 7 figures, 3 tables. Submitted to an ACL-style conference
>
> **摘要:** Automated generation of high-quality media presentations is challenging, requiring robust content extraction, narrative planning, visual design, and overall quality optimization. Existing methods often produce presentations with logical inconsistencies and suboptimal layouts, thereby struggling to meet professional standards. To address these challenges, we introduce RCPS (Reflective Coherent Presentation Synthesis), a novel framework integrating three key components: (1) Deep Structured Narrative Planning; (2) Adaptive Layout Generation; (3) an Iterative Optimization Loop. Additionally, we propose PREVAL, a preference-based evaluation framework employing rationale-enhanced multi-dimensional models to assess presentation quality across Content, Coherence, and Design. Experimental results demonstrate that RCPS significantly outperforms baseline methods across all quality dimensions, producing presentations that closely approximate human expert standards. PREVAL shows strong correlation with human judgments, validating it as a reliable automated tool for assessing presentation quality.
>
---
#### [new 002] Feature-based analysis of oral narratives from Afrikaans and isiXhosa children
- **分类: cs.CL**

- **简介: 该论文属于语言分析任务，旨在识别儿童口语叙事中的语言特征以预测是否需要干预。研究分析了南非两种语言儿童的口语故事，发现词汇多样性与句长是有效指标。**

- **链接: [http://arxiv.org/pdf/2507.13164v1](http://arxiv.org/pdf/2507.13164v1)**

> **作者:** Emma Sharratt; Annelien Smith; Retief Louw; Daleen Klop; Febe de Wet; Herman Kamper
>
> **备注:** SLaTE 2025 in Nijmegen, Netherlands
>
> **摘要:** Oral narrative skills are strong predictors of later literacy development. This study examines the features of oral narratives from children who were identified by experts as requiring intervention. Using simple machine learning methods, we analyse recorded stories from four- and five-year-old Afrikaans- and isiXhosa-speaking children. Consistent with prior research, we identify lexical diversity (unique words) and length-based features (mean utterance length) as indicators of typical development, but features like articulation rate prove less informative. Despite cross-linguistic variation in part-of-speech patterns, the use of specific verbs and auxiliaries associated with goal-directed storytelling is correlated with a reduced likelihood of requiring intervention. Our analysis of two linguistically distinct languages reveals both language-specific and shared predictors of narrative proficiency, with implications for early assessment in multilingual contexts.
>
---
#### [new 003] Large Language Models' Internal Perception of Symbolic Music
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文研究LLM在符号音乐领域的表现，旨在探索其隐式建模能力。通过文本生成音乐数据并进行分类与补全任务，评估其生成效果与局限性。**

- **链接: [http://arxiv.org/pdf/2507.12808v1](http://arxiv.org/pdf/2507.12808v1)**

> **作者:** Andrew Shin; Kunitake Kaneko
>
> **摘要:** Large language models (LLMs) excel at modeling relationships between strings in natural language and have shown promise in extending to other symbolic domains like coding or mathematics. However, the extent to which they implicitly model symbolic music remains underexplored. This paper investigates how LLMs represent musical concepts by generating symbolic music data from textual prompts describing combinations of genres and styles, and evaluating their utility through recognition and generation tasks. We produce a dataset of LLM-generated MIDI files without relying on explicit musical training. We then train neural networks entirely on this LLM-generated MIDI dataset and perform genre and style classification as well as melody completion, benchmarking their performance against established models. Our results demonstrate that LLMs can infer rudimentary musical structures and temporal relationships from text, highlighting both their potential to implicitly encode musical patterns and their limitations due to a lack of explicit musical context, shedding light on their generative capabilities for symbolic music.
>
---
#### [new 004] Improving Drug Identification in Overdose Death Surveillance using Large Language Models
- **分类: cs.CL; q-bio.QM; I.2.7; J.3**

- **简介: 该论文属于自然语言处理任务，旨在解决 overdose 死亡数据识别问题。通过 NLP 模型提升自由文本中的药物识别准确性与效率。**

- **链接: [http://arxiv.org/pdf/2507.12679v1](http://arxiv.org/pdf/2507.12679v1)**

> **作者:** Arthur J. Funnell; Panayiotis Petousis; Fabrice Harel-Canada; Ruby Romero; Alex A. T. Bui; Adam Koncsol; Hritika Chaturvedi; Chelsea Shover; David Goodman-Meza
>
> **备注:** 30 pages, 1 figure, 4 tables, 2 supplemental figures, 4 supplemental tables, submitted to Journal of Forensic Sciences (JFS)
>
> **摘要:** The rising rate of drug-related deaths in the United States, largely driven by fentanyl, requires timely and accurate surveillance. However, critical overdose data are often buried in free-text coroner reports, leading to delays and information loss when coded into ICD (International Classification of Disease)-10 classifications. Natural language processing (NLP) models may automate and enhance overdose surveillance, but prior applications have been limited. A dataset of 35,433 death records from multiple U.S. jurisdictions in 2020 was used for model training and internal testing. External validation was conducted using a novel separate dataset of 3,335 records from 2023-2024. Multiple NLP approaches were evaluated for classifying specific drug involvement from unstructured death certificate text. These included traditional single- and multi-label classifiers, as well as fine-tuned encoder-only language models such as Bidirectional Encoder Representations from Transformers (BERT) and BioClinicalBERT, and contemporary decoder-only large language models such as Qwen 3 and Llama 3. Model performance was assessed using macro-averaged F1 scores, and 95% confidence intervals were calculated to quantify uncertainty. Fine-tuned BioClinicalBERT models achieved near-perfect performance, with macro F1 scores >=0.998 on the internal test set. External validation confirmed robustness (macro F1=0.966), outperforming conventional machine learning, general-domain BERT models, and various decoder-only large language models. NLP models, particularly fine-tuned clinical variants like BioClinicalBERT, offer a highly accurate and scalable solution for overdose death classification from free-text reports. These methods can significantly accelerate surveillance workflows, overcoming the limitations of manual ICD-10 coding and supporting near real-time detection of emerging substance use trends.
>
---
#### [new 005] Learning Robust Negation Text Representations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升文本编码器对否定语义的鲁棒性。通过对比学习和知识蒸馏方法，增强模型对否定和模糊表达的理解能力。**

- **链接: [http://arxiv.org/pdf/2507.12782v1](http://arxiv.org/pdf/2507.12782v1)**

> **作者:** Thinh Hung Truong; Karin Verspoor; Trevor Cohn; Timothy Baldwin
>
> **摘要:** Despite rapid adoption of autoregressive large language models, smaller text encoders still play an important role in text understanding tasks that require rich contextualized representations. Negation is an important semantic function that is still not properly captured by such methods, affecting many downstream applications relying on text embeddings. We propose a strategy to improve negation robustness of text encoders, by distilling data from large language models using diverse patterns of negation and hedging. We adopt a standard contrastive learning strategy to finetune a strong BERT-based model, and observe large improvement in negation understanding capabilities while maintaining competitive performance on general benchmarks. In addition, we also show that our method can be adapted to LLMs, leading to improved performance on negation benchmarks.
>
---
#### [new 006] SemCSE: Semantic Contrastive Sentence Embeddings Using LLM-Generated Summaries For Scientific Abstracts
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于科学文本语义表示任务，旨在解决传统方法无法准确捕捉语义相似性的问题。通过利用大模型生成的摘要进行对比学习，提升嵌入空间的语义区分度。**

- **链接: [http://arxiv.org/pdf/2507.13105v1](http://arxiv.org/pdf/2507.13105v1)**

> **作者:** Marc Brinner; Sina Zarriess
>
> **摘要:** We introduce SemCSE, an unsupervised method for learning semantic embeddings of scientific texts. Building on recent advances in contrastive learning for text embeddings, our approach leverages LLM-generated summaries of scientific abstracts to train a model that positions semantically related summaries closer together in the embedding space. This resulting objective ensures that the model captures the true semantic content of a text, in contrast to traditional citation-based approaches that do not necessarily reflect semantic similarity. To validate this, we propose a novel benchmark designed to assess a model's ability to understand and encode the semantic content of scientific texts, demonstrating that our method enforces a stronger semantic separation within the embedding space. Additionally, we evaluate SemCSE on the comprehensive SciRepEval benchmark for scientific text embeddings, where it achieves state-of-the-art performance among models of its size, thus highlighting the benefits of a semantically focused training approach.
>
---
#### [new 007] Assessing the Reliability of LLMs Annotations in the Context of Demographic Bias and Model Explanation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，关注性别歧视检测中的标注可靠性问题。研究分析了人口统计因素对标注的影响，评估了生成模型作为标注者的可靠性，并探讨了内容驱动的解释方法。**

- **链接: [http://arxiv.org/pdf/2507.13138v1](http://arxiv.org/pdf/2507.13138v1)**

> **作者:** Hadi Mohammadi; Tina Shahedi; Pablo Mosteiro; Massimo Poesio; Ayoub Bagheri; Anastasia Giachanou
>
> **摘要:** Understanding the sources of variability in annotations is crucial for developing fair NLP systems, especially for tasks like sexism detection where demographic bias is a concern. This study investigates the extent to which annotator demographic features influence labeling decisions compared to text content. Using a Generalized Linear Mixed Model, we quantify this inf luence, finding that while statistically present, demographic factors account for a minor fraction ( 8%) of the observed variance, with tweet content being the dominant factor. We then assess the reliability of Generative AI (GenAI) models as annotators, specifically evaluating if guiding them with demographic personas improves alignment with human judgments. Our results indicate that simplistic persona prompting often fails to enhance, and sometimes degrades, performance compared to baseline models. Furthermore, explainable AI (XAI) techniques reveal that model predictions rely heavily on content-specific tokens related to sexism, rather than correlates of demographic characteristics. We argue that focusing on content-driven explanations and robust annotation protocols offers a more reliable path towards fairness than potentially persona simulation.
>
---
#### [new 008] HATS: Hindi Analogy Test Set for Evaluating Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的推理评估任务，旨在解决 Hindi 语言模型推理能力缺乏评估资源的问题。作者构建了 HATS 测试集，并提出改进的推理方法。**

- **链接: [http://arxiv.org/pdf/2507.13238v1](http://arxiv.org/pdf/2507.13238v1)**

> **作者:** Ashray Gupta; Rohan Joseph; Sunny Rai
>
> **摘要:** Analogies test a model's ability to infer implicit relationships between concepts, making them a key benchmark for evaluating reasoning capabilities. While large language models (LLMs) are widely evaluated for reasoning in English, their abilities in Indic languages remain understudied, limiting our understanding of whether these models generalize across languages. To address this gap, we introduce a new Hindi Analogy Test Set (HATS), comprising 405 multiple-choice questions sourced from Indian government exams. We benchmark state-of-the-art multilingual LLMs using various prompting strategies and introduce a grounded Chain of Thought approach that leverages cognitive theories of analogical reasoning. This approach improves model performance on Hindi analogy questions. Our experiments show that models perform best with English prompts, irrespective of the prompting strategy. Our test set addresses the lack of a critical resource to evaluate LLM reasoning capabilities in Hindi.
>
---
#### [new 009] Strategy Adaptation in Large Language Model Werewolf Agents
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于游戏AI任务，解决Werewolf游戏中策略适应问题，通过根据玩家态度和对话上下文切换预定义策略，提升代理性能。**

- **链接: [http://arxiv.org/pdf/2507.12732v1](http://arxiv.org/pdf/2507.12732v1)**

> **作者:** Fuya Nakamori; Yin Jou Huang; Fei Cheng
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** This study proposes a method to improve the performance of Werewolf agents by switching between predefined strategies based on the attitudes of other players and the context of conversations. While prior works of Werewolf agents using prompt engineering have employed methods where effective strategies are implicitly defined, they cannot adapt to changing situations. In this research, we propose a method that explicitly selects an appropriate strategy based on the game context and the estimated roles of other players. We compare the strategy adaptation Werewolf agents with baseline agents using implicit or fixed strategies and verify the effectiveness of our proposed method.
>
---
#### [new 010] Formalizing Attack Scenario Description: A Proposed Model
- **分类: cs.CL**

- **简介: 该论文属于网络安全领域，旨在解决攻击场景描述的标准化问题。提出一种形式化模型，用于攻击分析和脚本生成，提升自动化能力。**

- **链接: [http://arxiv.org/pdf/2507.13076v1](http://arxiv.org/pdf/2507.13076v1)**

> **作者:** Quentin Goux; Nadira Lammari
>
> **摘要:** Organizations face an ever-changing threat landscape. They must continuously dedicate significant efforts to protect their assets, making their adoption of increased cybersecurity automation inevitable. However, process automation requires formalization of input data. Through this paper, we address this need for processes that use attack scenarios as input. Among these processes, one can mention both the generation of scripts for attack simulation and training purposes, as well as the analysis of attacks. Therefore, the paper's main research contribution is a novel formal model that encompasses the attack's context description and its scenario. It is abstracted using UML class model. Once the description of our model done, we will show how it could serve an upstream attack analysis process. We will show also its use for an automatic generation of attack scripts in the context of cybersecurity training. These two uses cases constitute the second contribution of this present research work.
>
---
#### [new 011] Vision-and-Language Training Helps Deploy Taxonomic Knowledge but Does Not Fundamentally Alter It
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究VL训练是否改变语言模型的语义表示。通过对比实验发现，VL训练未显著改变模型的分类知识，但提升了其在特定任务中的应用能力。**

- **链接: [http://arxiv.org/pdf/2507.13328v1](http://arxiv.org/pdf/2507.13328v1)**

> **作者:** Yulu Qin; Dheeraj Varghese; Adam Dahlgren Lindström; Lucia Donatelli; Kanishka Misra; Najoung Kim
>
> **摘要:** Does vision-and-language (VL) training change the linguistic representations of language models in meaningful ways? Most results in the literature have shown inconsistent or marginal differences, both behaviorally and representationally. In this work, we start from the hypothesis that the domain in which VL training could have a significant effect is lexical-conceptual knowledge, in particular its taxonomic organization. Through comparing minimal pairs of text-only LMs and their VL-trained counterparts, we first show that the VL models often outperform their text-only counterparts on a text-only question-answering task that requires taxonomic understanding of concepts mentioned in the questions. Using an array of targeted behavioral and representational analyses, we show that the LMs and VLMs do not differ significantly in terms of their taxonomic knowledge itself, but they differ in how they represent questions that contain concepts in a taxonomic relation vs. a non-taxonomic relation. This implies that the taxonomic knowledge itself does not change substantially through additional VL training, but VL training does improve the deployment of this knowledge in the context of a specific task, even when the presentation of the task is purely linguistic.
>
---
#### [new 012] GEMMAS: Graph-based Evaluation Metrics for Multi Agent Systems
- **分类: cs.CL**

- **简介: 该论文属于多智能体系统评估任务，旨在解决现有评价仅关注结果而忽略协作过程的问题。提出GEMMAS框架，通过图模型分析协作质量，引入两个过程级指标进行评估。**

- **链接: [http://arxiv.org/pdf/2507.13190v1](http://arxiv.org/pdf/2507.13190v1)**

> **作者:** Jisoo Lee; Raeyoung Chang; Dongwook Kwon; Harmanpreet Singh; Nikhil Verma
>
> **备注:** 4 figures, 1 algorithm, 2 tables, 6 pages, under review at EMNLP Industry track 2025
>
> **摘要:** Multi-agent systems built on language models have shown strong performance on collaborative reasoning tasks. However, existing evaluations focus only on the correctness of the final output, overlooking how inefficient communication and poor coordination contribute to redundant reasoning and higher computational costs. We introduce GEMMAS, a graph-based evaluation framework that analyzes the internal collaboration process by modeling agent interactions as a directed acyclic graph. To capture collaboration quality, we propose two process-level metrics: Information Diversity Score (IDS) to measure semantic variation in inter-agent messages, and Unnecessary Path Ratio (UPR) to quantify redundant reasoning paths. We evaluate GEMMAS across five benchmarks and highlight results on GSM8K, where systems with only a 2.1% difference in accuracy differ by 12.8% in IDS and 80% in UPR, revealing substantial variation in internal collaboration. These findings demonstrate that outcome-only metrics are insufficient for evaluating multi-agent performance and highlight the importance of process-level diagnostics in designing more interpretable and resource-efficient collaborative AI systems.
>
---
#### [new 013] Social and Political Framing in Search Engine Results
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，研究搜索引擎结果中的社会政治框架问题，分析搜索算法与用户查询如何加剧信息偏见，揭示不同搜索引擎的来源偏好差异。**

- **链接: [http://arxiv.org/pdf/2507.13325v1](http://arxiv.org/pdf/2507.13325v1)**

> **作者:** Amrit Poudel; Tim Weninger
>
> **备注:** Accepted to ICWSM 2026
>
> **摘要:** Search engines play a crucial role in shaping public discourse by influencing how information is accessed and framed. While prior research has extensively examined various dimensions of search bias -- such as content prioritization, indexical bias, political polarization, and sources of bias -- an important question remains underexplored: how do search engines and ideologically-motivated user queries contribute to bias in search results. This study analyzes the outputs of major search engines using a dataset of political and social topics. The findings reveal that search engines not only prioritize content in ways that reflect underlying biases but also that ideologically-driven user queries exacerbate these biases, resulting in the amplification of specific narratives. Moreover, significant differences were observed across search engines in terms of the sources they prioritize. These results suggest that search engines may play a pivotal role in shaping public perceptions by reinforcing ideological divides, thereby contributing to the broader issue of information polarization.
>
---
#### [new 014] Automating Steering for Safe Multimodal Large Language Models
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; cs.MM**

- **简介: 该论文属于多模态大模型安全任务，旨在解决对抗性输入带来的安全风险。提出AutoSteer框架，通过安全评分、探测器和拒绝头实现无微调的安全干预。**

- **链接: [http://arxiv.org/pdf/2507.13255v1](http://arxiv.org/pdf/2507.13255v1)**

> **作者:** Lyucheng Wu; Mengru Wang; Ziwen Xu; Tri Cao; Nay Oo; Bryan Hooi; Shumin Deng
>
> **备注:** Working in progress. 22 pages (8+ for main); 25 figures; 1 table
>
> **摘要:** Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems.
>
---
#### [new 015] AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于科学实验设计任务，旨在评估大语言模型在构建消融实验中的表现。工作包括构建基准AbGen和开发AbGen-Eval以检验自动化评估系统的可靠性。**

- **链接: [http://arxiv.org/pdf/2507.13300v1](http://arxiv.org/pdf/2507.13300v1)**

> **作者:** Yilun Zhao; Weiyuan Chen; Zhijian Xu; Manasi Patwardhan; Yixin Liu; Chengye Wang; Lovekesh Vig; Arman Cohan
>
> **备注:** ACL 2025
>
> **摘要:** We introduce AbGen, the first benchmark designed to evaluate the capabilities of LLMs in designing ablation studies for scientific research. AbGen consists of 1,500 expert-annotated examples derived from 807 NLP papers. In this benchmark, LLMs are tasked with generating detailed ablation study designs for a specified module or process based on the given research context. Our evaluation of leading LLMs, such as DeepSeek-R1-0528 and o4-mini, highlights a significant performance gap between these models and human experts in terms of the importance, faithfulness, and soundness of the ablation study designs. Moreover, we demonstrate that current automated evaluation methods are not reliable for our task, as they show a significant discrepancy when compared to human assessment. To better investigate this, we develop AbGen-Eval, a meta-evaluation benchmark designed to assess the reliability of commonly used automated evaluation systems in measuring LLM performance on our task. We investigate various LLM-as-Judge systems on AbGen-Eval, providing insights for future research on developing more effective and reliable LLM-based evaluation systems for complex scientific tasks.
>
---
#### [new 016] Overview of the TalentCLEF 2025: Skill and Job Title Intelligence for Human Capital Management
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文介绍TalentCLEF 2025，聚焦技能与职位名称智能，解决劳动力市场语言技术评估问题，构建多语种数据集并开展基准测试。**

- **链接: [http://arxiv.org/pdf/2507.13275v1](http://arxiv.org/pdf/2507.13275v1)**

> **作者:** Luis Gasco; Hermenegildo Fabregat; Laura García-Sardiña; Paula Estrella; Daniel Deniz; Alvaro Rodrigo; Rabih Zbib
>
> **摘要:** Advances in natural language processing and large language models are driving a major transformation in Human Capital Management, with a growing interest in building smart systems based on language technologies for talent acquisition, upskilling strategies, and workforce planning. However, the adoption and progress of these technologies critically depend on the development of reliable and fair models, properly evaluated on public data and open benchmarks, which have so far been unavailable in this domain. To address this gap, we present TalentCLEF 2025, the first evaluation campaign focused on skill and job title intelligence. The lab consists of two tasks: Task A - Multilingual Job Title Matching, covering English, Spanish, German, and Chinese; and Task B - Job Title-Based Skill Prediction, in English. Both corpora were built from real job applications, carefully anonymized, and manually annotated to reflect the complexity and diversity of real-world labor market data, including linguistic variability and gender-marked expressions. The evaluations included monolingual and cross-lingual scenarios and covered the evaluation of gender bias. TalentCLEF attracted 76 registered teams with more than 280 submissions. Most systems relied on information retrieval techniques built with multilingual encoder-based models fine-tuned with contrastive learning, and several of them incorporated large language models for data augmentation or re-ranking. The results show that the training strategies have a larger effect than the size of the model alone. TalentCLEF provides the first public benchmark in this field and encourages the development of robust, fair, and transferable language technologies for the labor market.
>
---
#### [new 017] Comparing Apples to Oranges: A Dataset & Analysis of LLM Humour Understanding from Traditional Puns to Topical Jokes
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的幽默理解任务，旨在探讨大语言模型对不同形式幽默的解释能力，通过构建包含多种笑话类型的数据集进行分析。**

- **链接: [http://arxiv.org/pdf/2507.13335v1](http://arxiv.org/pdf/2507.13335v1)**

> **作者:** Tyler Loakman; William Thorne; Chenghua Lin
>
> **摘要:** Humour, as a complex language form, is derived from myriad aspects of life, whilst existing work on computational humour has focussed almost exclusively on short pun-based jokes. In this work, we investigate whether the ability of Large Language Models (LLMs) to explain humour depends on the particular humour form. We compare models on simple puns and more complex topical humour that requires knowledge of real-world entities and events. In doing so, we curate a dataset of 600 jokes split across 4 joke types and manually write high-quality explanations. These jokes include heterographic and homographic puns, contemporary internet humour, and topical jokes, where understanding relies on reasoning beyond "common sense", rooted instead in world knowledge regarding news events and pop culture. Using this dataset, we compare the zero-shot abilities of a range of LLMs to accurately and comprehensively explain jokes of different types, identifying key research gaps in the task of humour explanation. We find that none of the tested models (inc. reasoning models) are capable of reliably generating adequate explanations of all joke types, further highlighting the narrow focus of most works in computational humour on overly simple joke forms.
>
---
#### [new 018] Synergy: End-to-end Concept Model
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文提出Synergy模型，属于自然语言处理任务，旨在通过端到端路由机制实现不同抽象层次的融合。工作包括训练字节级语言模型，减少概念标记数量并提升性能。**

- **链接: [http://arxiv.org/pdf/2507.12769v1](http://arxiv.org/pdf/2507.12769v1)**

> **作者:** Keli Zheng; Zerong Xie
>
> **摘要:** In this paper, we present Synergy, a language model that bridges different levels of abstraction in an end-to-end fashion through a learned routing mechanism. Focusing on low-level linguistic abstraction, we trained our model as a byte-level language model. Our model spontaneously learns to tokenize bytes, producing fewer concept tokens than Byte-level Byte Pair Encoder (BBPE) tokenizers while keeping comparable performance. By comparing with Llama3, we observed an advantage of Synergy under the same model scale and training dataset size. Further studies show that the middle part (the higher abstraction part) of our model performs better when positional encodings are removed, suggesting the emergence of position-independent concepts. These findings demonstrate the feasibility of tokenizer-free architectures, paving the way for more robust and flexible pipelines.
>
---
#### [new 019] TransEvalnia: Reasoning-based Evaluation and Ranking of Translations
- **分类: cs.CL**

- **简介: 该论文属于机器翻译评估任务，旨在解决翻译质量评价与排序问题。提出TransEvalnia系统，通过推理进行翻译评估和排名，提升评估准确性。**

- **链接: [http://arxiv.org/pdf/2507.12724v1](http://arxiv.org/pdf/2507.12724v1)**

> **作者:** Richard Sproat; Tianyu Zhao; Llion Jones
>
> **摘要:** We present TransEvalnia, a prompting-based translation evaluation and ranking system that uses reasoning in performing its evaluations and ranking. This system presents fine-grained evaluations based on a subset of the Multidimensional Quality Metrics (https://themqm.org/), returns an assessment of which translation it deems the best, and provides numerical scores for the various dimensions and for the overall translation. We show that TransEvalnia performs as well as or better than the state-of-the-art MT-Ranker (Moosa et al. 2024) on our own English-Japanese data as well as several language pairs from various WMT shared tasks. Using Anthropic's Claude-3.5-Sonnet and Qwen-2.5-72B-Instruct as the evaluation LLMs, we show that the evaluations returned are deemed highly acceptable to human raters, and that the scores assigned to the translations by Sonnet, as well as other LLMs, correlate well with scores assigned by the human raters. We also note the sensitivity of our system -- as well as MT-Ranker -- to the order in which the translations are presented, and we propose methods to address this position bias. All data, including the system's evaluation and reasoning, human assessments, as well as code is released.
>
---
#### [new 020] QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation
- **分类: cs.CL; cs.AI; 68T50**

- **简介: 该论文属于语言模型推理任务，旨在解决多步推理效果不佳的问题。通过引入部分解法增强训练，提升模型推理能力，取得新最优结果。**

- **链接: [http://arxiv.org/pdf/2507.13266v1](http://arxiv.org/pdf/2507.13266v1)**

> **作者:** Jiazheng Li; Hong Lu; Kaiyue Wen; Zaiwen Yang; Jiaxuan Gao; Hongzhou Lin; Yi Wu; Jingzhao Zhang
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Reinforcement learning (RL) has become a key component in training large language reasoning models (LLMs). However, recent studies questions its effectiveness in improving multi-step reasoning-particularly on hard problems. To address this challenge, we propose a simple yet effective strategy via Question Augmentation: introduce partial solutions during training to reduce problem difficulty and provide more informative learning signals. Our method, QuestA, when applied during RL training on math reasoning tasks, not only improves pass@1 but also pass@k-particularly on problems where standard RL struggles to make progress. This enables continual improvement over strong open-source models such as DeepScaleR and OpenMath Nemotron, further enhancing their reasoning capabilities. We achieve new state-of-the-art results on math benchmarks using 1.5B-parameter models: 67.1% (+5.3%) on AIME24, 59.5% (+10.0%) on AIME25, and 35.5% (+4.0%) on HMMT25. Further, we provide theoretical explanations that QuestA improves sample efficiency, offering a practical and generalizable pathway for expanding reasoning capability through RL.
>
---
#### [new 021] A Survey of Context Engineering for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，探讨如何优化大语言模型的上下文信息。解决模型在复杂上下文理解与长文本生成间的不平衡问题，通过系统分析1300余篇论文，提出统一框架与研究方向。**

- **链接: [http://arxiv.org/pdf/2507.13334v1](http://arxiv.org/pdf/2507.13334v1)**

> **作者:** Lingrui Mei; Jiayu Yao; Yuyao Ge; Yiwei Wang; Baolong Bi; Yujun Cai; Jiazhi Liu; Mingyu Li; Zhong-Zhi Li; Duzhen Zhang; Chenlin Zhou; Jiayi Mao; Tianze Xia; Jiafeng Guo; Shenghua Liu
>
> **备注:** ongoing work; 165 pages, 1401 citations
>
> **摘要:** The performance of Large Language Models (LLMs) is fundamentally determined by the contextual information provided during inference. This survey introduces Context Engineering, a formal discipline that transcends simple prompt design to encompass the systematic optimization of information payloads for LLMs. We present a comprehensive taxonomy decomposing Context Engineering into its foundational components and the sophisticated implementations that integrate them into intelligent systems. We first examine the foundational components: context retrieval and generation, context processing and context management. We then explore how these components are architecturally integrated to create sophisticated system implementations: retrieval-augmented generation (RAG), memory systems and tool-integrated reasoning, and multi-agent systems. Through this systematic analysis of over 1300 research papers, our survey not only establishes a technical roadmap for the field but also reveals a critical research gap: a fundamental asymmetry exists between model capabilities. While current models, augmented by advanced context engineering, demonstrate remarkable proficiency in understanding complex contexts, they exhibit pronounced limitations in generating equally sophisticated, long-form outputs. Addressing this gap is a defining priority for future research. Ultimately, this survey provides a unified framework for both researchers and engineers advancing context-aware AI.
>
---
#### [new 022] AdaptiSent: Context-Aware Adaptive Attention for Multimodal Aspect-Based Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文属于多模态方面情感分析任务，旨在提升文本与图像中情感和方面词的识别。通过自适应跨模态注意力机制，增强情感分析的准确性。**

- **链接: [http://arxiv.org/pdf/2507.12695v1](http://arxiv.org/pdf/2507.12695v1)**

> **作者:** S M Rafiuddin; Sadia Kamal; Mohammed Rakib; Arunkumar Bagavathi; Atriya Sen
>
> **备注:** 12 pages (including references), 2 figures (Fig. 1 overview, Fig. 2 hyperparameter sensitivity with two subplots), 6 tables (performance, ablation, dataset stats, case studies, etc.), accepted at ASONAM 2025 (Social Network Analysis and Mining)
>
> **摘要:** We introduce AdaptiSent, a new framework for Multimodal Aspect-Based Sentiment Analysis (MABSA) that uses adaptive cross-modal attention mechanisms to improve sentiment classification and aspect term extraction from both text and images. Our model integrates dynamic modality weighting and context-adaptive attention, enhancing the extraction of sentiment and aspect-related information by focusing on how textual cues and visual context interact. We tested our approach against several baselines, including traditional text-based models and other multimodal methods. Results from standard Twitter datasets show that AdaptiSent surpasses existing models in precision, recall, and F1 score, and is particularly effective in identifying nuanced inter-modal relationships that are crucial for accurate sentiment and aspect term extraction. This effectiveness comes from the model's ability to adjust its focus dynamically based on the context's relevance, improving the depth and accuracy of sentiment analysis across various multimodal data sets. AdaptiSent sets a new standard for MABSA, significantly outperforming current methods, especially in understanding complex multimodal information.
>
---
#### [new 023] Is This Just Fantasy? Language Model Representations Reflect Human Judgments of Event Plausibility
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型的模态分类能力，旨在解决模型判断事件合理性的问题。通过分析激活中的线性表示，发现模型能准确区分不同模态，并与人类判断相关联。**

- **链接: [http://arxiv.org/pdf/2507.12553v1](http://arxiv.org/pdf/2507.12553v1)**

> **作者:** Michael A. Lepori; Jennifer Hu; Ishita Dasgupta; Roma Patel; Thomas Serre; Ellie Pavlick
>
> **摘要:** Language models (LMs) are used for a diverse range of tasks, from question answering to writing fantastical stories. In order to reliably accomplish these tasks, LMs must be able to discern the modal category of a sentence (i.e., whether it describes something that is possible, impossible, completely nonsensical, etc.). However, recent studies have called into question the ability of LMs to categorize sentences according to modality (Michaelov et al., 2025; Kauf et al., 2023). In this work, we identify linear representations that discriminate between modal categories within a variety of LMs, or modal difference vectors. Analysis of modal difference vectors reveals that LMs have access to more reliable modal categorization judgments than previously reported. Furthermore, we find that modal difference vectors emerge in a consistent order as models become more competent (i.e., through training steps, layers, and parameter count). Notably, we find that modal difference vectors identified within LM activations can be used to model fine-grained human categorization behavior. This potentially provides a novel view into how human participants distinguish between modal categories, which we explore by correlating projections along modal difference vectors with human participants' ratings of interpretable features. In summary, we derive new insights into LM modal categorization using techniques from mechanistic interpretability, with the potential to inform our understanding of modal categorization in humans.
>
---
#### [new 024] A Computational Framework to Identify Self-Aspects in Text
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文本分析任务，旨在识别文本中的自我方面。通过构建本体和标注数据集，开发并评估多种模型以解决自我表达的系统性分析问题。**

- **链接: [http://arxiv.org/pdf/2507.13115v1](http://arxiv.org/pdf/2507.13115v1)**

> **作者:** Jaya Caporusso; Matthew Purver; Senja Pollak
>
> **备注:** Accepted to ACL SRW 2025
>
> **摘要:** This Ph.D. proposal introduces a plan to develop a computational framework to identify Self-aspects in text. The Self is a multifaceted construct and it is reflected in language. While it is described across disciplines like cognitive science and phenomenology, it remains underexplored in natural language processing (NLP). Many of the aspects of the Self align with psychological and other well-researched phenomena (e.g., those related to mental health), highlighting the need for systematic NLP-based analysis. In line with this, we plan to introduce an ontology of Self-aspects and a gold-standard annotated dataset. Using this foundation, we will develop and evaluate conventional discriminative models, generative large language models, and embedding-based retrieval approaches against four main criteria: interpretability, ground-truth adherence, accuracy, and computational efficiency. Top-performing models will be applied in case studies in mental health and empirical phenomenology.
>
---
#### [new 025] Are Knowledge and Reference in Multilingual Language Models Cross-Lingually Consistent?
- **分类: cs.CL**

- **简介: 该论文属于多语言语言模型研究任务，旨在解决跨语言知识一致性问题。通过分析模型在跨语言场景下的表现，探索提升一致性的方法。**

- **链接: [http://arxiv.org/pdf/2507.12838v1](http://arxiv.org/pdf/2507.12838v1)**

> **作者:** Xi Ai; Mahardika Krisna Ihsani; Min-Yen Kan
>
> **摘要:** Cross-lingual consistency should be considered to assess cross-lingual transferability, maintain the factuality of the model knowledge across languages, and preserve the parity of language model performance. We are thus interested in analyzing, evaluating, and interpreting cross-lingual consistency for factual knowledge. We examine code-mixed coreferential statements conveyed identical knowledge across languages to study cross-lingual knowledge consistency. We use some interpretability approaches to analyze the behavior of a model in cross-lingual contexts, discovering that multilingual models show different levels of consistency, subject to language families, linguistic factors, and a bottleneck in cross-lingual consistency on a particular layer. In addition, we evaluate common strategies aimed at improving multilingual performance to observe whether these strategies can improve knowledge consistency at the same time. While knowledge is not cross-lingual consistency in many cases, code-switching training and cross-lingual word alignment objectives show the most promising results, emphasizing the noteworthiness of cross-lingual alignment supervision and code-switching training for both multilingual performance and cross-lingual consistency enhancement.
>
---
#### [new 026] HapticCap: A Multimodal Dataset and Task for Understanding User Experience of Vibration Haptic Signals
- **分类: cs.CL**

- **简介: 该论文提出HapticCap数据集和触觉描述任务，解决触觉信号与文本描述匹配问题，通过多模态学习提升用户体验理解。**

- **链接: [http://arxiv.org/pdf/2507.13318v1](http://arxiv.org/pdf/2507.13318v1)**

> **作者:** Guimin Hu; Daniel Hershcovich; Hasti Seifi
>
> **摘要:** Haptic signals, from smartphone vibrations to virtual reality touch feedback, can effectively convey information and enhance realism, but designing signals that resonate meaningfully with users is challenging. To facilitate this, we introduce a multimodal dataset and task, of matching user descriptions to vibration haptic signals, and highlight two primary challenges: (1) lack of large haptic vibration datasets annotated with textual descriptions as collecting haptic descriptions is time-consuming, and (2) limited capability of existing tasks and models to describe vibration signals in text. To advance this area, we create HapticCap, the first fully human-annotated haptic-captioned dataset, containing 92,070 haptic-text pairs for user descriptions of sensory, emotional, and associative attributes of vibrations. Based on HapticCap, we propose the haptic-caption retrieval task and present the results of this task from a supervised contrastive learning framework that brings together text representations within specific categories and vibrations. Overall, the combination of language model T5 and audio model AST yields the best performance in the haptic-caption retrieval task, especially when separately trained for each description category.
>
---
#### [new 027] FLEXITOKENS: Flexible Tokenization for Evolving Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决语言模型在适应新数据时的tokenization问题。通过引入可学习的字节级分词器，提升模型灵活性与性能。**

- **链接: [http://arxiv.org/pdf/2507.12720v1](http://arxiv.org/pdf/2507.12720v1)**

> **作者:** Abraham Toluase Owodunni; Orevaoghene Ahia; Sachin Kumar
>
> **摘要:** Language models (LMs) are challenging to adapt to new data distributions by simple finetuning. This is due to the rigidity of their subword tokenizers, which typically remain unchanged during adaptation. This inflexibility often leads to inefficient tokenization, causing overfragmentation of out-of-distribution domains, unseen languages, or scripts. In this work, we develop byte-level LMs with learnable tokenizers to make tokenization adaptive. Our models include a submodule that learns to predict boundaries between the input byte sequence, encoding it into variable-length segments. Existing tokenizer-free methods train this boundary predictor using an auxiliary loss that enforces a fixed compression rate across the training corpus, introducing a new kind of rigidity. We propose FLEXITOKENS, a simplified training objective that enables significantly greater flexibility during adaptation. Evaluating across multiple multilingual benchmarks, morphologically diverse tasks, and domains, we demonstrate that FLEXITOKENS consistently reduces token over-fragmentation and achieves up to 10\% improvements on downstream task performance compared to subword and other gradient-based tokenizers. Code and data for our experiments will be released at https://github.com/owos/flexitokens
>
---
#### [new 028] Logit Arithmetic Elicits Long Reasoning Capabilities Without Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升大模型的长链推理能力。通过ThinkLogit方法，在无需额外训练的情况下增强模型推理性能。**

- **链接: [http://arxiv.org/pdf/2507.12759v1](http://arxiv.org/pdf/2507.12759v1)**

> **作者:** Yunxiang Zhang; Muhammad Khalifa; Lechen Zhang; Xin Liu; Ayoung Lee; Xinliang Frederick Zhang; Farima Fatahi Bayat; Lu Wang
>
> **摘要:** Large reasoning models (LRMs) can do complex reasoning via long chain-of-thought (CoT) involving cognitive strategies such as backtracking and self-correction. Recent studies suggest that some models inherently possess these long reasoning abilities, which may be unlocked via extra training. Our work first investigates whether we can elicit such behavior without any training. To this end, we propose a decoding-time approach, ThinkLogit, which utilizes logits arithmetic (Liu et al., 2024) to tune a target large LM for long reasoning using a substantially smaller model as guider. We then show that we can further boost performance by training the guider model with preference optimization over correct/incorrect reasoning pairs sampled from both the target and guider model -- a setup we refer to as ThinkLogit-DPO. Our experiments demonstrate that ThinkLogit and ThinkLogit-DPO achieve a relative improvement in pass@1 by 26% and 29%, respectively, over four mathematical datasets using the Qwen2.5-32B when guided by R1-Distill-Qwen-1.5B -- a model 21x smaller. Lastly, we show that ThinkLogit can transfer long reasoning skills acquired through reinforcement learning, improving pass@1 by 13% relative compared to the Qwen2.5-32B base model. Our work presents a computationally-efficient method to elicit long reasoning in large models with minimal or no additional training.
>
---
#### [new 029] MRT at IberLEF-2025 PRESTA Task: Maximizing Recovery from Tables with Multiple Steps
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对IberLEF-2025 PRESTA任务，解决西班牙语表格问答问题。通过生成Python代码过滤处理表格数据，实现准确回答。**

- **链接: [http://arxiv.org/pdf/2507.12981v1](http://arxiv.org/pdf/2507.12981v1)**

> **作者:** Maximiliano Hormazábal Lagos; Álvaro Bueno Sáez; Héctor Cerezo-Costas; Pedro Alonso Doval; Jorge Alcalde Vesteiro
>
> **备注:** Accepted as an official challenge paper in the PRESTA: Questions and Answers over Tabular Data shared task at IberLEF 2025, colocated with the 41st SEPLN Conference in Zaragoza, Spain
>
> **摘要:** This paper presents our approach for the IberLEF 2025 Task PRESTA: Preguntas y Respuestas sobre Tablas en Espa\~nol (Questions and Answers about Tables in Spanish). Our solution obtains answers to the questions by implementing Python code generation with LLMs that is used to filter and process the table. This solution evolves from the MRT implementation for the Semeval 2025 related task. The process consists of multiple steps: analyzing and understanding the content of the table, selecting the useful columns, generating instructions in natural language, translating these instructions to code, running it, and handling potential errors or exceptions. These steps use open-source LLMs and fine-grained optimized prompts for each step. With this approach, we achieved an accuracy score of 85\% in the task.
>
---
#### [new 030] Modeling Open-World Cognition as On-Demand Synthesis of Probabilistic Models
- **分类: cs.CL; cs.AI; cs.PL**

- **简介: 该论文属于认知建模任务，旨在解决开放世界推理问题。通过构建MSA架构，结合语言模型与概率程序，实现对新颖情境的合理推理与判断。**

- **链接: [http://arxiv.org/pdf/2507.12547v1](http://arxiv.org/pdf/2507.12547v1)**

> **作者:** Lionel Wong; Katherine M. Collins; Lance Ying; Cedegao E. Zhang; Adrian Weller; Tobias Gersternberg; Timothy O'Donnell; Alexander K. Lew; Jacob D. Andreas; Joshua B. Tenenbaum; Tyler Brooke-Wilson
>
> **备注:** Presented at CogSci 2025
>
> **摘要:** When faced with novel situations, people are able to marshal relevant considerations from a wide range of background knowledge and put these to use in inferences and predictions. What permits us to draw in globally relevant information and reason over it coherently? Here, we explore the hypothesis that people use a combination of distributed and symbolic representations to construct bespoke mental models tailored to novel situations. We propose a computational implementation of this idea -- a ``Model Synthesis Architecture'' (MSA) -- using language models to implement global relevance-based retrieval and model synthesis and probabilistic programs to implement bespoke, coherent world models. We evaluate our MSA as a model of human judgments on a novel reasoning dataset. The dataset -- built around a `Model Olympics` domain of sports vignettes -- tests models' capacity for human-like, open-ended reasoning by requiring (i) judgments about novel causal structures described in language; (ii) drawing on large bodies of background knowledge; and (iii) doing both in light of observations that introduce arbitrary novel variables. Our MSA approach captures human judgments better than language model-only baselines, under both direct and chain-of-thought generations from the LM that supports model synthesis. These results suggest that MSAs can be implemented in a way that mirrors people's ability to deliver locally coherent reasoning over globally relevant variables, offering a path to understanding and replicating human reasoning in open-ended domains.
>
---
#### [new 031] Enhancing Cross-task Transfer of Large Language Models via Activation Steering
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，解决低资源任务迁移问题。通过激活空间引导，无需参数更新即可实现跨任务知识迁移。**

- **链接: [http://arxiv.org/pdf/2507.13236v1](http://arxiv.org/pdf/2507.13236v1)**

> **作者:** Xinyu Tang; Zhihao Lv; Xiaoxue Cheng; Junyi Li; Wayne Xin Zhao; Zujie Wen; Zhiqiang Zhang; Jun Zhou
>
> **摘要:** Large language models (LLMs) have shown impressive abilities in leveraging pretrained knowledge through prompting, but they often struggle with unseen tasks, particularly in data-scarce scenarios. While cross-task in-context learning offers a direct solution for transferring knowledge across tasks, it still faces critical challenges in terms of robustness, scalability, and efficiency. In this paper, we investigate whether cross-task transfer can be achieved via latent space steering without parameter updates or input expansion. Through an analysis of activation patterns in the latent space of LLMs, we observe that the enhanced activations induced by in-context examples have consistent patterns across different tasks. Inspired by these findings, we propose CAST, a novel Cross-task Activation Steering Transfer framework that enables effective transfer by manipulating the model's internal activation states. Our approach first selects influential and diverse samples from high-resource tasks, then utilizes their contrastive representation-enhanced activations to adapt LLMs to low-resource tasks. Extensive experiments across both cross-domain and cross-lingual transfer settings show that our method outperforms competitive baselines and demonstrates superior scalability and lower computational costs.
>
---
#### [new 032] The Imitation Game: Turing Machine Imitator is Length Generalizable Reasoner
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的推理任务，旨在解决Transformer模型在长序列上的泛化问题。通过模仿图灵机的推理过程，提出TAIL方法提升模型的长度泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.13332v1](http://arxiv.org/pdf/2507.13332v1)**

> **作者:** Zhouqi Hua; Wenwei Zhang; Chengqi Lyu; Yuzhe Gu; Songyang Gao; Kuikun Liu; Kai Chen
>
> **摘要:** Length generalization, the ability to solve problems of longer sequences than those observed during training, poses a core challenge of Transformer-based large language models (LLM). Although existing studies have predominantly focused on data-driven approaches for arithmetic operations and symbolic manipulation tasks, these approaches tend to be task-specific with limited overall performance. To pursue a more general solution, this paper focuses on a broader case of reasoning problems that are computable, i.e., problems that algorithms can solve, thus can be solved by the Turing Machine. From this perspective, this paper proposes Turing MAchine Imitation Learning (TAIL) to improve the length generalization ability of LLMs. TAIL synthesizes chain-of-thoughts (CoT) data that imitate the execution process of a Turing Machine by computer programs, which linearly expands the reasoning steps into atomic states to alleviate shortcut learning and explicit memory fetch mechanism to reduce the difficulties of dynamic and long-range data access in elementary operations. To validate the reliability and universality of TAIL, we construct a challenging synthetic dataset covering 8 classes of algorithms and 18 tasks. Without bells and whistles, TAIL significantly improves the length generalization ability as well as the performance of Qwen2.5-7B on various tasks using only synthetic data, surpassing previous methods and DeepSeek-R1. The experimental results reveal that the key concepts in the Turing Machine, instead of the thinking styles, are indispensable for TAIL for length generalization, through which the model exhibits read-and-write behaviors consistent with the properties of the Turing Machine in their attention layers. This work provides a promising direction for future research in the learning of LLM reasoning from synthetic data.
>
---
#### [new 033] AudioJudge: Understanding What Works in Large Audio Model Based Speech Evaluation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音评估任务，旨在解决传统评估系统设计复杂和与人类偏好相关性低的问题。通过构建AudioJudge模型，实现统一、高效的语音质量与偏好评估。**

- **链接: [http://arxiv.org/pdf/2507.12705v1](http://arxiv.org/pdf/2507.12705v1)**

> **作者:** Potsawee Manakul; Woody Haosheng Gan; Michael J. Ryan; Ali Sartaz Khan; Warit Sirichotedumrong; Kunat Pipatanakul; William Held; Diyi Yang
>
> **摘要:** Current speech evaluation suffers from two critical limitations: the need and difficulty of designing specialized systems targeting individual audio characteristics, and poor correlation between automatic evaluation methods and human preferences. This work presents a systematic study of Large Audio Model (LAM) as a Judge, AudioJudge, investigating whether it can provide a unified evaluation framework that addresses both challenges. We systematically explore AudioJudge across audio characteristic detection tasks, including pronunciation, speaking rate, speaker identification and speech quality, and system-level human preference simulation for automated benchmarking. We investigate different prompt engineering strategies, finding that audio concatenation combined with in-context learning significantly improves performance across both audio characteristic detection and human preference simulation tasks. We further introduce a multi-aspect ensemble AudioJudge to enable general-purpose multi-aspect audio evaluation. This method decomposes speech assessment into specialized judges for lexical content, speech quality, and paralinguistic features, achieving up to 0.91 Spearman correlation with human preferences on our system ranking benchmark. Robustness analysis reveals that while LAMs maintain strong performance under acoustic noise, they exhibit significant verbosity and positional biases that require careful mitigation.
>
---
#### [new 034] The first open machine translation system for the Chechen language
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决Chechen语言与俄语之间的翻译问题。通过构建数据集并微调模型，实现了双向翻译，并发布了相关资源。**

- **链接: [http://arxiv.org/pdf/2507.12672v1](http://arxiv.org/pdf/2507.12672v1)**

> **作者:** Abu-Viskhan A. Umishov; Vladislav A. Grigorian
>
> **备注:** 7 pages
>
> **摘要:** We introduce the first open-source model for translation between the vulnerable Chechen language and Russian, and the dataset collected to train and evaluate it. We explore fine-tuning capabilities for including a new language into a large language model system for multilingual translation NLLB-200. The BLEU / ChrF++ scores for our model are 8.34 / 34.69 and 20.89 / 44.55 for translation from Russian to Chechen and reverse direction, respectively. The release of the translation models is accompanied by the distribution of parallel words, phrases and sentences corpora and multilingual sentence encoder adapted to the Chechen language.
>
---
#### [new 035] Automatically assessing oral narratives of Afrikaans and isiXhosa children
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音评估任务，旨在帮助教师识别需要干预的儿童。通过自动语音识别和机器学习模型分析儿童的口语叙事，提升评估效率与准确性。**

- **链接: [http://arxiv.org/pdf/2507.13205v1](http://arxiv.org/pdf/2507.13205v1)**

> **作者:** R. Louw; E. Sharratt; F. de Wet; C. Jacobs; A. Smith; H. Kamper
>
> **备注:** Accepted to SLaTE 2025
>
> **摘要:** Developing narrative and comprehension skills in early childhood is critical for later literacy. However, teachers in large preschool classrooms struggle to accurately identify students who require intervention. We present a system for automatically assessing oral narratives of preschool children in Afrikaans and isiXhosa. The system uses automatic speech recognition followed by a machine learning scoring model to predict narrative and comprehension scores. For scoring predicted transcripts, we compare a linear model to a large language model (LLM). The LLM-based system outperforms the linear model in most cases, but the linear system is competitive despite its simplicity. The LLM-based system is comparable to a human expert in flagging children who require intervention. We lay the foundation for automatic oral assessments in classrooms, giving teachers extra capacity to focus on personalised support for children's learning.
>
---
#### [new 036] Making Language Model a Hierarchical Classifier and Generator
- **分类: cs.CL; cs.AI**

- **简介: 该论文将语言模型改造为分层分类器和生成器，解决传统模型仅在最后一层解码的问题。通过多层并行解码提升任务表现。**

- **链接: [http://arxiv.org/pdf/2507.12930v1](http://arxiv.org/pdf/2507.12930v1)**

> **作者:** Yihong Wang; Zhonglin Jiang; Ningyuan Xi; Yue Zhao; Qingqing Gu; Xiyuan Chen; Hao Wu; Sheng Xu; Hange Zhou; Yong Chen; Luo Ji
>
> **摘要:** Decoder-only language models, such as GPT and LLaMA, generally decode on the last layer. Motivated by human's hierarchical thinking capability, we propose that a hierarchical decoder architecture could be built with different layers decoding texts simultaneously. Due to limited time and computationally resources, we choose to adapt a pretrained language model into this form of hierarchical decoder. Language heads of the last layer are copied to different selected intermediate layers, and fine-tuned with different task inputs. By thorough experiments, we validate that these selective intermediate layers could be adapted to speak meaningful and reasonable contents, and this paradigm of hierarchical decoder can obtain state-of-the-art performances on multiple tasks such as hierarchical text classification, classification-guided generation, and hierarchical text generation. This study suggests the possibility of a generalized hierarchical reasoner, pretraining from scratch.
>
---
#### [new 037] Perfect diffusion is $\mathsf{TC}^0$ -- Bad diffusion is Turing-complete
- **分类: cs.CC; cs.CL; cs.LG**

- **简介: 该论文研究扩散模型的语言建模计算复杂性，探讨其在不同精度下的计算能力，揭示其在TC⁰类与图灵完备之间的二分特性。**

- **链接: [http://arxiv.org/pdf/2507.12469v1](http://arxiv.org/pdf/2507.12469v1)**

> **作者:** Yuxi Liu
>
> **备注:** 7 pages
>
> **摘要:** This paper explores the computational complexity of diffusion-based language modeling. We prove a dichotomy based on the quality of the score-matching network in a diffusion model. In one direction, a network that exactly computes the score function of some initial distribution can only perform language modeling within the $\mathsf{TC}^0$ complexity class, reflecting limitations tied to rapid convergence. In the other direction, we show that if there is no requirement for the network to match any score function, then diffusion modeling can simulate any Turing machine in a certain sense. This dichotomy provides a theoretical lens on the capabilities and limitations of diffusion models, particularly concerning tasks requiring sequential computation. We conjecture extensions of our theoretical results, including for the case where the diffusion model is not perfect, but merely good. We also discuss the wider context and practical implications, and hypothesize that a machine learning architecture that can interpolate between sequential and parallel modes of operation would be superior to both Transformers and diffusion models.
>
---
#### [new 038] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言导航任务，旨在解决虚拟与物理环境间的差距问题。通过构建VLN-PE平台，评估不同方法在物理机器人中的表现，揭示实际部署挑战。**

- **链接: [http://arxiv.org/pdf/2507.13019v1](http://arxiv.org/pdf/2507.13019v1)**

> **作者:** Liuyi Wang; Xinyuan Xia; Hui Zhao; Hanqing Wang; Tai Wang; Yilun Chen; Chengju Liu; Qijun Chen; Jiangmiao Pang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recent Vision-and-Language Navigation (VLN) advancements are promising, but their idealized assumptions about robot movement and control fail to reflect physically embodied deployment challenges. To bridge this gap, we introduce VLN-PE, a physically realistic VLN platform supporting humanoid, quadruped, and wheeled robots. For the first time, we systematically evaluate several ego-centric VLN methods in physical robotic settings across different technical pipelines, including classification models for single-step discrete action prediction, a diffusion model for dense waypoint prediction, and a train-free, map-based large language model (LLM) integrated with path planning. Our results reveal significant performance degradation due to limited robot observation space, environmental lighting variations, and physical challenges like collisions and falls. This also exposes locomotion constraints for legged robots in complex environments. VLN-PE is highly extensible, allowing seamless integration of new scenes beyond MP3D, thereby enabling more comprehensive VLN evaluation. Despite the weak generalization of current models in physical deployment, VLN-PE provides a new pathway for improving cross-embodiment's overall adaptability. We hope our findings and tools inspire the community to rethink VLN limitations and advance robust, practical VLN models. The code is available at https://crystalsixone.github.io/vln_pe.github.io/.
>
---
#### [new 039] A Survey of AIOps in the Era of Large Language Models
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于AIOps领域，探讨LLM在IT运维中的应用。解决LLM在AIOps中的影响与挑战问题，通过分析183篇论文，研究数据源、任务演变、方法及评估方式。**

- **链接: [http://arxiv.org/pdf/2507.12472v1](http://arxiv.org/pdf/2507.12472v1)**

> **作者:** Lingzhe Zhang; Tong Jia; Mengxi Jia; Yifan Wu; Aiwei Liu; Yong Yang; Zhonghai Wu; Xuming Hu; Philip S. Yu; Ying Li
>
> **备注:** Accepted By CSUR, an extended version of "A Survey of AIOps for Failure Management in the Era of Large Language Models" [arXiv:2406.11213]
>
> **摘要:** As large language models (LLMs) grow increasingly sophisticated and pervasive, their application to various Artificial Intelligence for IT Operations (AIOps) tasks has garnered significant attention. However, a comprehensive understanding of the impact, potential, and limitations of LLMs in AIOps remains in its infancy. To address this gap, we conducted a detailed survey of LLM4AIOps, focusing on how LLMs can optimize processes and improve outcomes in this domain. We analyzed 183 research papers published between January 2020 and December 2024 to answer four key research questions (RQs). In RQ1, we examine the diverse failure data sources utilized, including advanced LLM-based processing techniques for legacy data and the incorporation of new data sources enabled by LLMs. RQ2 explores the evolution of AIOps tasks, highlighting the emergence of novel tasks and the publication trends across these tasks. RQ3 investigates the various LLM-based methods applied to address AIOps challenges. Finally, RQ4 reviews evaluation methodologies tailored to assess LLM-integrated AIOps approaches. Based on our findings, we discuss the state-of-the-art advancements and trends, identify gaps in existing research, and propose promising directions for future exploration.
>
---
#### [new 040] Mono-InternVL-1.5: Towards Cheaper and Faster Monolithic Multimodal Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态语言模型任务，旨在解决单体模型训练不稳定和数据成本高的问题。通过引入视觉参数空间和优化预训练方法，提升了模型性能并降低了成本。**

- **链接: [http://arxiv.org/pdf/2507.12566v1](http://arxiv.org/pdf/2507.12566v1)**

> **作者:** Gen Luo; Wenhan Dou; Wenhao Li; Zhaokai Wang; Xue Yang; Changyao Tian; Hao Li; Weiyun Wang; Wenhai Wang; Xizhou Zhu; Yu Qiao; Jifeng Dai
>
> **摘要:** This paper focuses on monolithic Multimodal Large Language Models (MLLMs), which integrate visual encoding and language decoding into a single model. Existing structures and pre-training strategies for monolithic MLLMs often suffer from unstable optimization and catastrophic forgetting. To address these challenges, our key idea is to embed a new visual parameter space into a pre-trained LLM, enabling stable learning of visual knowledge from noisy data via delta tuning. Based on this principle, we first introduce Mono-InternVL, an advanced monolithic MLLM that incorporates a set of visual experts through a multimodal mixture-of-experts architecture. In addition, we design an innovative Endogenous Visual Pre-training (EViP) for Mono-InternVL to maximize its visual capabilities via progressive learning. Mono-InternVL achieves competitive performance against existing MLLMs but also leads to relatively expensive data cost. Therefore, we further present Mono-InternVL-1.5, a cheaper and stronger monolithic MLLM equipped with an improved EViP (EViP++). EViP++ introduces additional visual attention experts to Mono-InternVL-1.5 and re-organizes the pre-training process in an efficient manner. During inference, it includes a fused CUDA kernel to speed up its MoE operations. With these designs, Mono-InternVL-1.5 significantly reduces training and inference costs, while still maintaining competitive performance with Mono-InternVL. To evaluate our approach, we conduct extensive experiments across 15 benchmarks. Results demonstrate that Mono-InternVL outperforms existing monolithic MLLMs on 12 out of 15 benchmarks, e.g., +114-point improvement over Emu3 on OCRBench. Compared to its modular counterpart, i.e., InternVL-1.5, Mono-InternVL-1.5 achieves similar multimodal performance while reducing first-token latency by up to 69%. Code and models are released at https://github.com/OpenGVLab/Mono-InternVL.
>
---
#### [new 041] Scaling Up RL: Unlocking Diverse Reasoning in LLMs via Prolonged Training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型训练任务，旨在通过延长强化学习提升模型在数学、编程和逻辑题等推理任务中的表现。工作包括优化训练方法和引入稳定机制，显著提升了模型性能。**

- **链接: [http://arxiv.org/pdf/2507.12507v1](http://arxiv.org/pdf/2507.12507v1)**

> **作者:** Mingjie Liu; Shizhe Diao; Jian Hu; Ximing Lu; Xin Dong; Hao Zhang; Alexander Bukharin; Shaokun Zhang; Jiaqi Zeng; Makesh Narsimhan Sreedhar; Gerald Shen; David Mosallanezhad; Di Zhang; Jonas Yang; June Yang; Oleksii Kuchaiev; Guilin Liu; Zhiding Yu; Pavlo Molchanov; Yejin Choi; Jan Kautz; Yi Dong
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Recent advancements in reasoning-focused language models such as OpenAI's O1 and DeepSeek-R1 have shown that scaling test-time computation-through chain-of-thought reasoning and iterative exploration-can yield substantial improvements on complex tasks like mathematics and code generation. These breakthroughs have been driven by large-scale reinforcement learning (RL), particularly when combined with verifiable reward signals that provide objective and grounded supervision. In this report, we investigate the effects of prolonged reinforcement learning on a small language model across a diverse set of reasoning domains. Our work identifies several key ingredients for effective training, including the use of verifiable reward tasks, enhancements to Group Relative Policy Optimization (GRPO), and practical techniques to improve training stability and generalization. We introduce controlled KL regularization, clipping ratio, and periodic reference policy resets as critical components for unlocking long-term performance gains. Our model achieves significant improvements over strong baselines, including +14.7% on math, +13.9% on coding, and +54.8% on logic puzzle tasks. To facilitate continued research, we release our model publicly.
>
---
#### [new 042] From Roots to Rewards: Dynamic Tree Reasoning with RL
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决语言模型在复杂问题解答中的错误传播和知识整合问题。提出动态强化学习框架，提升树状推理的灵活性与效率。**

- **链接: [http://arxiv.org/pdf/2507.13142v1](http://arxiv.org/pdf/2507.13142v1)**

> **作者:** Ahmed Bahloul; Simon Malberg
>
> **摘要:** Modern language models address complex questions through chain-of-thought (CoT) reasoning (Wei et al., 2023) and retrieval augmentation (Lewis et al., 2021), yet struggle with error propagation and knowledge integration. Tree-structured reasoning methods, particularly the Probabilistic Tree-of-Thought (ProbTree)(Cao et al., 2023) framework, mitigate these issues by decomposing questions into hierarchical structures and selecting answers through confidence-weighted aggregation of parametric and retrieved knowledge (Yao et al., 2023). However, ProbTree's static implementation introduces two key limitations: (1) the reasoning tree is fixed during the initial construction phase, preventing dynamic adaptation to intermediate results, and (2) each node requires exhaustive evaluation of all possible solution strategies, creating computational inefficiency. We present a dynamic reinforcement learning (Sutton and Barto, 2018) framework that transforms tree-based reasoning into an adaptive process. Our approach incrementally constructs the reasoning tree based on real-time confidence estimates, while learning optimal policies for action selection (decomposition, retrieval, or aggregation). This maintains ProbTree's probabilistic rigor while improving both solution quality and computational efficiency through selective expansion and focused resource allocation. The work establishes a new paradigm for treestructured reasoning that balances the reliability of probabilistic frameworks with the flexibility required for real-world question answering systems.
>
---
#### [new 043] PMKLC: Parallel Multi-Knowledge Learning-based Lossless Compression for Large-Scale Genomics Database
- **分类: cs.LG; cs.AI; cs.CL; cs.DB**

- **简介: 该论文属于基因组数据压缩任务，解决传统方法压缩比低、速度慢、鲁棒性差的问题，提出PMKLC框架实现高效并行压缩。**

- **链接: [http://arxiv.org/pdf/2507.12805v1](http://arxiv.org/pdf/2507.12805v1)**

> **作者:** Hui Sun; Yanfeng Ding; Liping Yi; Huidong Ma; Gang Wang; Xiaoguang Liu; Cheng Zhong; Wentong Cai
>
> **备注:** Accepted via KDD-25
>
> **摘要:** Learning-based lossless compressors play a crucial role in large-scale genomic database backup, storage, transmission, and management. However, their 1) inadequate compression ratio, 2) low compression \& decompression throughput, and 3) poor compression robustness limit their widespread adoption and application in both industry and academia. To solve those challenges, we propose a novel \underline{P}arallel \underline{M}ulti-\underline{K}nowledge \underline{L}earning-based \underline{C}ompressor (PMKLC) with four crucial designs: 1) We propose an automated multi-knowledge learning-based compression framework as compressors' backbone to enhance compression ratio and robustness; 2) we design a GPU-accelerated ($s$,$k$)-mer encoder to optimize compression throughput and computing resource usage; 3) we introduce data block partitioning and Step-wise Model Passing (SMP) mechanisms for parallel acceleration; 4) We design two compression modes PMKLC-S and PMKLC-M to meet the complex application scenarios, where the former runs on a resource-constrained single GPU and the latter is multi-GPU accelerated. We benchmark PMKLC-S/M and 14 baselines (7 traditional and 7 leaning-based) on 15 real-world datasets with different species and data sizes. Compared to baselines on the testing datasets, PMKLC-S/M achieve the average compression ratio improvement up to 73.609\% and 73.480\%, the average throughput improvement up to 3.036$\times$ and 10.710$\times$, respectively. Besides, PMKLC-S/M also achieve the best robustness and competitive memory cost, indicating its greater stability against datasets with different probability distribution perturbations, and its strong ability to run on memory-constrained devices.
>
---
#### [new 044] Probabilistic Soundness Guarantees in LLM Reasoning Chains
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于错误检测任务，解决LLM推理链中错误传播问题。提出ARES框架，通过概率方法评估每一步的合理性，提升推理可靠性。**

- **链接: [http://arxiv.org/pdf/2507.12948v1](http://arxiv.org/pdf/2507.12948v1)**

> **作者:** Weiqiu You; Anton Xue; Shreya Havaldar; Delip Rao; Helen Jin; Chris Callison-Burch; Eric Wong
>
> **摘要:** In reasoning chains generated by large language models (LLMs), initial errors often propagate and undermine the reliability of the final conclusion. Current LLM-based error detection methods often fail to detect propagated errors because they do not properly account for how earlier errors might corrupt judgments of downstream reasoning. To better detect such propagated errors, we introduce Autoregressive Reasoning Entailment Stability (ARES), a novel probabilistic framework that prevents error propagation by judging each claim based only on previously-assessed sound premises. This inductive method yields a nuanced score for each step and provides certified statistical guarantees of its soundness, rather than a brittle binary label. ARES achieves state-of-the-art performance across four benchmarks (72.1% Macro-F1, +8.2 points) and demonstrates superior robustness on very long synthetic reasoning chains, where it excels at detecting propagated errors (90.3% F1, +27.6 points).
>
---
#### [new 045] A Comprehensive Survey of Electronic Health Record Modeling: From Deep Learning Approaches to Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于电子健康记录建模任务，旨在解决EHR数据的异构性和复杂性问题。通过深度学习和大语言模型方法，提升数据理解和临床应用能力。**

- **链接: [http://arxiv.org/pdf/2507.12774v1](http://arxiv.org/pdf/2507.12774v1)**

> **作者:** Weijieying Ren; Jingxi Zhu; Zehao Liu; Tianxiang Zhao; Vasant Honavar
>
> **摘要:** Artificial intelligence (AI) has demonstrated significant potential in transforming healthcare through the analysis and modeling of electronic health records (EHRs). However, the inherent heterogeneity, temporal irregularity, and domain-specific nature of EHR data present unique challenges that differ fundamentally from those in vision and natural language tasks. This survey offers a comprehensive overview of recent advancements at the intersection of deep learning, large language models (LLMs), and EHR modeling. We introduce a unified taxonomy that spans five key design dimensions: data-centric approaches, neural architecture design, learning-focused strategies, multimodal learning, and LLM-based modeling systems. Within each dimension, we review representative methods addressing data quality enhancement, structural and temporal representation, self-supervised learning, and integration with clinical knowledge. We further highlight emerging trends such as foundation models, LLM-driven clinical agents, and EHR-to-text translation for downstream reasoning. Finally, we discuss open challenges in benchmarking, explainability, clinical alignment, and generalization across diverse clinical settings. This survey aims to provide a structured roadmap for advancing AI-driven EHR modeling and clinical decision support. For a comprehensive list of EHR-related methods, kindly refer to https://survey-on-tabular-data.github.io/.
>
---
#### [new 046] Inverse Reinforcement Learning Meets Large Language Model Post-Training: Basics, Advances, and Opportunities
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于LLM对齐任务，旨在解决模型与人类价值观一致的问题。通过逆强化学习方法，研究如何构建奖励模型并提升模型可靠性与可控性。**

- **链接: [http://arxiv.org/pdf/2507.13158v1](http://arxiv.org/pdf/2507.13158v1)**

> **作者:** Hao Sun; Mihaela van der Schaar
>
> **摘要:** In the era of Large Language Models (LLMs), alignment has emerged as a fundamental yet challenging problem in the pursuit of more reliable, controllable, and capable machine intelligence. The recent success of reasoning models and conversational AI systems has underscored the critical role of reinforcement learning (RL) in enhancing these systems, driving increased research interest at the intersection of RL and LLM alignment. This paper provides a comprehensive review of recent advances in LLM alignment through the lens of inverse reinforcement learning (IRL), emphasizing the distinctions between RL techniques employed in LLM alignment and those in conventional RL tasks. In particular, we highlight the necessity of constructing neural reward models from human data and discuss the formal and practical implications of this paradigm shift. We begin by introducing fundamental concepts in RL to provide a foundation for readers unfamiliar with the field. We then examine recent advances in this research agenda, discussing key challenges and opportunities in conducting IRL for LLM alignment. Beyond methodological considerations, we explore practical aspects, including datasets, benchmarks, evaluation metrics, infrastructure, and computationally efficient training and inference techniques. Finally, we draw insights from the literature on sparse-reward RL to identify open questions and potential research directions. By synthesizing findings from diverse studies, we aim to provide a structured and critical overview of the field, highlight unresolved challenges, and outline promising future directions for improving LLM alignment through RL and IRL techniques.
>
---
#### [new 047] A Fuzzy Approach to Project Success: Measuring What Matters
- **分类: cs.SE; cs.CL; H.4.m**

- **简介: 该论文属于项目评估任务，旨在解决传统方法忽视项目成功多维性的问题。通过引入模糊逻辑构建评估系统，更准确衡量项目对用户的影响。**

- **链接: [http://arxiv.org/pdf/2507.12653v1](http://arxiv.org/pdf/2507.12653v1)**

> **作者:** João Granja-Correia; Remedios Hernández-Linares; Luca Ferranti; Arménio Rego
>
> **备注:** 3 pages, 1 figure, presented at FUZZ-IEEE 2025
>
> **摘要:** This paper introduces a novel approach to project success evaluation by integrating fuzzy logic into an existing construct. Traditional Likert-scale measures often overlook the context-dependent and multifaceted nature of project success. The proposed hierarchical Type-1 Mamdani fuzzy system prioritizes sustained positive impact for end-users, reducing emphasis on secondary outcomes like stakeholder satisfaction and internal project success. This dynamic approach may provide a more accurate measure of project success and could be adaptable to complex evaluations. Future research will focus on empirical testing and broader applications of fuzzy logic in social science.
>
---
#### [new 048] Spatially Grounded Explanations in Vision Language Models for Document Visual Question Answering
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于文档视觉问答任务，旨在提升模型解释的透明度与可复现性。提出EaGERS方法，通过空间定位增强模型推理过程的可解释性。**

- **链接: [http://arxiv.org/pdf/2507.12490v1](http://arxiv.org/pdf/2507.12490v1)**

> **作者:** Maximiliano Hormazábal Lagos; Héctor Cerezo-Costas; Dimosthenis Karatzas
>
> **备注:** This work has been accepted for presentation at the 16th Conference and Labs of the Evaluation Forum (CLEF 2025) and will be published in the proceedings by Springer in the Lecture Notes in Computer Science (LNCS) series. Please cite the published version when available
>
> **摘要:** We introduce EaGERS, a fully training-free and model-agnostic pipeline that (1) generates natural language rationales via a vision language model, (2) grounds these rationales to spatial sub-regions by computing multimodal embedding similarities over a configurable grid with majority voting, and (3) restricts the generation of responses only from the relevant regions selected in the masked image. Experiments on the DocVQA dataset demonstrate that our best configuration not only outperforms the base model on exact match accuracy and Average Normalized Levenshtein Similarity metrics but also enhances transparency and reproducibility in DocVQA without additional model fine-tuning.
>
---
#### [new 049] Emotional Support with LLM-based Empathetic Dialogue Generation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于情感支持对话生成任务，旨在通过对话提供共情与有效的情感帮助。工作包括使用大模型结合提示工程和微调技术，提升响应的适配性与支持性。**

- **链接: [http://arxiv.org/pdf/2507.12820v1](http://arxiv.org/pdf/2507.12820v1)**

> **作者:** Shiquan Wang; Ruiyu Fang; Zhongjiang He; Shuangyong Song; Yongxiang Li
>
> **摘要:** Emotional Support Conversation (ESC) aims to provide empathetic and effective emotional assistance through dialogue, addressing the growing demand for mental health support. This paper presents our solution for the NLPCC 2025 Task 8 ESC evaluation, where we leverage large-scale language models enhanced by prompt engineering and finetuning techniques. We explore both parameter-efficient Low-Rank Adaptation and full-parameter fine-tuning strategies to improve the model's ability to generate supportive and contextually appropriate responses. Our best model ranked second in the competition, highlighting the potential of combining LLMs with effective adaptation methods for ESC tasks. Future work will focus on further enhancing emotional understanding and response personalization to build more practical and reliable emotional support systems.
>
---
#### [new 050] VisionThink: Smart and Efficient Vision Language Model via Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型任务，旨在解决视觉token冗余问题。通过动态调整图像分辨率和强化学习，提升OCR任务性能并减少计算资源消耗。**

- **链接: [http://arxiv.org/pdf/2507.13348v1](http://arxiv.org/pdf/2507.13348v1)**

> **作者:** Senqiao Yang; Junyi Li; Xin Lai; Bei Yu; Hengshuang Zhao; Jiaya Jia
>
> **备注:** Code and models are available at https://github.com/dvlab-research/VisionThink
>
> **摘要:** Recent advancements in vision-language models (VLMs) have improved performance by increasing the number of visual tokens, which are often significantly longer than text tokens. However, we observe that most real-world scenarios do not require such an extensive number of visual tokens. While the performance drops significantly in a small subset of OCR-related tasks, models still perform accurately in most other general VQA tasks with only 1/4 resolution. Therefore, we propose to dynamically process distinct samples with different resolutions, and present a new paradigm for visual token compression, namely, VisionThink. It starts with a downsampled image and smartly decides whether it is sufficient for problem solving. Otherwise, the model could output a special token to request the higher-resolution image. Compared to existing Efficient VLM methods that compress tokens using fixed pruning ratios or thresholds, VisionThink autonomously decides whether to compress tokens case by case. As a result, it demonstrates strong fine-grained visual understanding capability on OCR-related tasks, and meanwhile saves substantial visual tokens on simpler tasks. We adopt reinforcement learning and propose the LLM-as-Judge strategy to successfully apply RL to general VQA tasks. Moreover, we carefully design a reward function and penalty mechanism to achieve a stable and reasonable image resize call ratio. Extensive experiments demonstrate the superiority, efficiency, and effectiveness of our method. Our code is available at https://github.com/dvlab-research/VisionThink.
>
---
#### [new 051] Teach Old SAEs New Domain Tricks with Boosting
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型解释任务，解决SAE在特定领域特征捕捉不足的问题。通过训练次级SAE建模主模型的重构误差，提升领域适应性与解释性。**

- **链接: [http://arxiv.org/pdf/2507.12990v1](http://arxiv.org/pdf/2507.12990v1)**

> **作者:** Nikita Koriagin; Yaroslav Aksenov; Daniil Laptev; Gleb Gerasimov; Nikita Balagansky; Daniil Gavrilov
>
> **摘要:** Sparse Autoencoders have emerged as powerful tools for interpreting the internal representations of Large Language Models, yet they often fail to capture domain-specific features not prevalent in their training corpora. This paper introduces a residual learning approach that addresses this feature blindness without requiring complete retraining. We propose training a secondary SAE specifically to model the reconstruction error of a pretrained SAE on domain-specific texts, effectively capturing features missed by the primary model. By summing the outputs of both models during inference, we demonstrate significant improvements in both LLM cross-entropy and explained variance metrics across multiple specialized domains. Our experiments show that this method efficiently incorporates new domain knowledge into existing SAEs while maintaining their performance on general tasks. This approach enables researchers to selectively enhance SAE interpretability for specific domains of interest, opening new possibilities for targeted mechanistic interpretability of LLMs.
>
---
#### [new 052] MCPEval: Automatic MCP-based Deep Evaluation for AI Agent Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI代理模型评估任务，旨在解决现有评估方法依赖静态基准和人工数据的问题。提出MCPEval框架，实现自动化任务生成与深度评估。**

- **链接: [http://arxiv.org/pdf/2507.12806v1](http://arxiv.org/pdf/2507.12806v1)**

> **作者:** Zhiwei Liu; Jielin Qiu; Shiyu Wang; Jianguo Zhang; Zuxin Liu; Roshan Ram; Haolin Chen; Weiran Yao; Huan Wang; Shelby Heinecke; Silvio Savarese; Caiming Xiong
>
> **备注:** https://github.com/SalesforceAIResearch/MCPEval
>
> **摘要:** The rapid rise of Large Language Models (LLMs)-based intelligent agents underscores the need for robust, scalable evaluation frameworks. Existing methods rely on static benchmarks and labor-intensive data collection, limiting practical assessment. We introduce \oursystemname, an open-source Model Context Protocol (MCP)-based framework that automates end-to-end task generation and deep evaluation of LLM agents across diverse domains. MCPEval standardizes metrics, seamlessly integrates with native agent tools, and eliminates manual effort in building evaluation pipelines. Empirical results across five real-world domains show its effectiveness in revealing nuanced, domain-specific performance. We publicly release MCPEval https://github.com/SalesforceAIResearch/MCPEval to promote reproducible and standardized LLM agent evaluation.
>
---
#### [new 053] The Generative Energy Arena (GEA): Incorporating Energy Awareness in Large Language Model (LLM) Human Evaluations
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大语言模型评估任务，旨在解决人类评价与模型能耗关系的问题。工作是提出GEA平台，将能耗信息纳入评价过程，发现用户更倾向节能模型。**

- **链接: [http://arxiv.org/pdf/2507.13302v1](http://arxiv.org/pdf/2507.13302v1)**

> **作者:** Carlos Arriaga; Gonzalo Martínez; Eneko Sendin; Javier Conde; Pedro Reviriego
>
> **摘要:** The evaluation of large language models is a complex task, in which several approaches have been proposed. The most common is the use of automated benchmarks in which LLMs have to answer multiple-choice questions of different topics. However, this method has certain limitations, being the most concerning, the poor correlation with the humans. An alternative approach, is to have humans evaluate the LLMs. This poses scalability issues as there is a large and growing number of models to evaluate making it impractical (and costly) to run traditional studies based on recruiting a number of evaluators and having them rank the responses of the models. An alternative approach is the use of public arenas, such as the popular LM arena, on which any user can freely evaluate models on any question and rank the responses of two models. The results are then elaborated into a model ranking. An increasingly important aspect of LLMs is their energy consumption and, therefore, evaluating how energy awareness influences the decisions of humans in selecting a model is of interest. In this paper, we present GEA, the Generative Energy Arena, an arena that incorporates information on the energy consumption of the model in the evaluation process. Preliminary results obtained with GEA are also presented, showing that for most questions, when users are aware of the energy consumption, they favor smaller and more energy efficient models. This suggests that for most user interactions, the extra cost and energy incurred by the more complex and top-performing models do not provide an increase in the perceived quality of the responses that justifies their use.
>
---
#### [new 054] UniSLU: Unified Spoken Language Understanding from Heterogeneous Cross-Task Datasets
- **分类: eess.AS; cs.AI; cs.CL; cs.MM; cs.SD**

- **简介: 该论文属于语音语言理解任务，解决多任务模型分离导致的系统复杂和交互受限问题，提出UniSLU统一框架，融合ASR、NER和SA任务。**

- **链接: [http://arxiv.org/pdf/2507.12951v1](http://arxiv.org/pdf/2507.12951v1)**

> **作者:** Zhichao Sheng; Shilin Zhou; Chen Gong; Zhenghua Li
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Spoken Language Understanding (SLU) plays a crucial role in speech-centric multimedia applications, enabling machines to comprehend spoken language in scenarios such as meetings, interviews, and customer service interactions. SLU encompasses multiple tasks, including Automatic Speech Recognition (ASR), spoken Named Entity Recognition (NER), and spoken Sentiment Analysis (SA). However, existing methods often rely on separate model architectures for individual tasks such as spoken NER and SA, which increases system complexity, limits cross-task interaction, and fails to fully exploit heterogeneous datasets available across tasks. To address these limitations, we propose UniSLU, a unified framework that jointly models multiple SLU tasks within a single architecture. Specifically, we propose a unified representation for diverse SLU tasks, enabling full utilization of heterogeneous datasets across multiple tasks. Built upon this representation, we propose a unified generative method that jointly models ASR, spoken NER, and SA tasks, enhancing task interactions and enabling seamless integration with large language models to harness their powerful generative capabilities. Extensive experiments on public SLU datasets demonstrate the effectiveness of our approach, achieving superior SLU performance compared to several benchmark methods, making it well-suited for real-world speech-based multimedia scenarios. We will release all code and models at github to facilitate future research.
>
---
## 更新

#### [replaced 001] BEARCUBS: A benchmark for computer-using web agents
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.07919v2](http://arxiv.org/pdf/2503.07919v2)**

> **作者:** Yixiao Song; Katherine Thai; Chau Minh Pham; Yapei Chang; Mazin Nadaf; Mohit Iyyer
>
> **备注:** 16 pages
>
> **摘要:** Modern web agents possess computer use abilities that allow them to interact with webpages by sending commands to a virtual keyboard and mouse. While such agents have considerable potential to assist human users with complex tasks, evaluating their capabilities in real-world settings poses a major challenge. To this end, we introduce BEARCUBS, a "small but mighty" benchmark of 111 information-seeking questions designed to evaluate a web agent's ability to search, browse, and identify factual information from the web. Unlike prior web agent benchmarks, solving BEARCUBS requires (1) accessing live web content rather than synthetic or simulated pages, which captures the unpredictability of real-world web interactions; and (2) performing a broad range of multimodal interactions (e.g., video understanding, 3D navigation) that cannot be bypassed via text-based workarounds. Each question in BEARCUBS has a corresponding short, unambiguous answer and a human-validated browsing trajectory, allowing for transparent evaluation of agent performance and strategies. A human study confirms that BEARCUBS questions are solvable but non-trivial (84.7% human accuracy), revealing domain knowledge gaps and overlooked details as common failure points. By contrast, state-of-the-art computer-using agents underperform, with the best-scoring system (OpenAI's Operator) reaching only 23.4% accuracy. These results highlight critical areas for improvement, including reliable source selection and more powerful multimodal capabilities. To facilitate future research, BEARCUBS will be updated periodically to replace invalid or contaminated questions, keeping the benchmark fresh for future generations of web agents.
>
---
#### [replaced 002] Memorization Inheritance in Sequence-Level Knowledge Distillation for Neural Machine Translation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01491v2](http://arxiv.org/pdf/2502.01491v2)**

> **作者:** Verna Dankers; Vikas Raunak
>
> **备注:** To appear at ACL 2025; 15 pages total (5 in the main paper, 3 pages of limitations and references and 7 pages with appendices)
>
> **摘要:** In this work, we explore how instance-level memorization in the teacher Neural Machine Translation (NMT) model gets inherited by the student model in sequence-level knowledge distillation (SeqKD). We find that despite not directly seeing the original training data, students memorize more than baseline models (models of the same size, trained on the original data) -- 3.4% for exact matches and 57% for extractive memorization -- and show increased hallucination rates. Further, under this SeqKD setting, we also characterize how students behave on specific training data subgroups, such as subgroups with low quality and specific counterfactual memorization (CM) scores, and find that students exhibit amplified denoising on low-quality subgroups. Finally, we propose a modification to SeqKD named Adaptive-SeqKD, which intervenes in SeqKD to reduce memorization and hallucinations. Overall, we recommend caution when applying SeqKD: students inherit both their teachers' superior performance and their fault modes, thereby requiring active monitoring.
>
---
#### [replaced 003] VIDEE: Visual and Interactive Decomposition, Execution, and Evaluation of Text Analytics with Intelligent Agents
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.21582v2](http://arxiv.org/pdf/2506.21582v2)**

> **作者:** Sam Yu-Te Lee; Chengyang Ji; Shicheng Wen; Lifu Huang; Dongyu Liu; Kwan-Liu Ma
>
> **摘要:** Text analytics has traditionally required specialized knowledge in Natural Language Processing (NLP) or text analysis, which presents a barrier for entry-level analysts. Recent advances in large language models (LLMs) have changed the landscape of NLP by enabling more accessible and automated text analysis (e.g., topic detection, summarization, information extraction, etc.). We introduce VIDEE, a system that supports entry-level data analysts to conduct advanced text analytics with intelligent agents. VIDEE instantiates a human-agent collaroration workflow consisting of three stages: (1) Decomposition, which incorporates a human-in-the-loop Monte-Carlo Tree Search algorithm to support generative reasoning with human feedback, (2) Execution, which generates an executable text analytics pipeline, and (3) Evaluation, which integrates LLM-based evaluation and visualizations to support user validation of execution results. We conduct two quantitative experiments to evaluate VIDEE's effectiveness and analyze common agent errors. A user study involving participants with varying levels of NLP and text analytics experience -- from none to expert -- demonstrates the system's usability and reveals distinct user behavior patterns. The findings identify design implications for human-agent collaboration, validate the practical utility of VIDEE for non-expert users, and inform future improvements to intelligent text analytics systems.
>
---
#### [replaced 004] On the Limitations of Large Language Models (LLMs): False Attribution
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.04631v2](http://arxiv.org/pdf/2404.04631v2)**

> **作者:** Tosin Adewumi; Nudrat Habib; Lama Alkhaled; Elisa Barney
>
> **备注:** This paper was accepted for presentation by Recent Advances in NLP (RANLP) 2025 conference
>
> **摘要:** In this work, we introduce a new hallucination metric - Simple Hallucination Index (SHI) and provide insight into one important limitation of the parametric knowledge of large language models (LLMs), i.e. false attribution. The task of automatic author attribution for relatively small chunks of text is an important NLP task but can be challenging. We empirically evaluate the power of 3 open SotA LLMs in zero-shot setting (Gemma-7B, Mixtral 8x7B, and LLaMA-2-13B). We acquired the top 10 most popular books of a month, according to Project Gutenberg, divided each one into equal chunks of 400 words, and prompted each LLM to predict the author. We then randomly sampled 162 chunks per book for human evaluation, based on the error margin of 7% and a confidence level of 95%. The average results show that Mixtral 8x7B has the highest prediction accuracy, the lowest SHI, and a Pearson's correlation (r) of 0.724, 0.263, and -0.9996, respectively, followed by LLaMA-2-13B and Gemma-7B. However, Mixtral 8x7B suffers from high hallucinations for 3 books, rising as high as a SHI of 0.87 (in the range 0-1, where 1 is the worst). The strong negative correlation of accuracy and SHI, given by r, demonstrates the fidelity of the new hallucination metric, which may generalize to other tasks. We also show that prediction accuracies correlate positively with the frequencies of Wikipedia instances of the book titles instead of the downloads and we perform error analyses of predictions. We publicly release the annotated chunks of data and our codes to aid the reproducibility and evaluation of other models.
>
---
#### [replaced 005] UPCORE: Utility-Preserving Coreset Selection for Balanced Unlearning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15082v2](http://arxiv.org/pdf/2502.15082v2)**

> **作者:** Vaidehi Patil; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** Code: https://github.com/Vaidehi99/UPCORE
>
> **摘要:** User specifications or legal frameworks often require information to be removed from pretrained models, including large language models (LLMs). This requires deleting or "forgetting" a set of data points from an already-trained model, which typically degrades its performance on other data points. Thus, a balance must be struck between removing information and keeping the model's other abilities intact, with a failure to balance this trade-off leading to poor deletion or an unusable model. To this end, we propose UPCORE (Utility-Preserving Coreset Selection), a method-agnostic data selection framework for mitigating collateral damage during unlearning. Finding that the model damage is correlated with the variance of the model's representations on the forget set, we selectively prune the forget set to remove outliers, thereby minimizing model degradation after unlearning. Across three standard unlearning methods, UPCORE consistently achieves a superior balance between the competing objectives of deletion efficacy and model preservation. To better evaluate this trade-off, we introduce a new metric, measuring the area-under-the-curve (AUC) across standard metrics. Our results show that UPCORE improves both standard metrics and AUC, benefiting from positive transfer between the coreset and pruned points while reducing negative transfer from the forget set to points outside of it.
>
---
#### [replaced 006] IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.06208v3](http://arxiv.org/pdf/2411.06208v3)**

> **作者:** Xinghua Zhang; Haiyang Yu; Cheng Fu; Fei Huang; Yongbin Li
>
> **备注:** ACL 2025
>
> **摘要:** In the realm of large language models (LLMs), the ability of models to accurately follow instructions is paramount as more agents and applications leverage LLMs for construction, where the complexity of instructions are rapidly increasing. However, on the one hand, there is only a certain amount of complex instruction evaluation data; on the other hand, there are no dedicated algorithms to improve the ability to follow complex instructions. To this end, this paper introduces TRACE, a benchmark for improving and evaluating the complex instructionfollowing ability, which consists of 120K training data and 1K evaluation data. Furthermore, we propose IOPO (Input-Output Preference Optimization) alignment method which takes both input and output preference pairs into consideration, where LLMs not only rapidly align with response preferences but also meticulously explore the instruction preferences. Extensive experiments on both in-domain and outof-domain datasets confirm the effectiveness of IOPO, showing 8.15%, 2.18% improvements on in-domain data and 6.29%, 3.13% on outof-domain data compared to SFT and DPO respectively.
>
---
#### [replaced 007] ContextQFormer: A New Context Modeling Method for Multi-Turn Multi-Modal Conversations
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23121v2](http://arxiv.org/pdf/2505.23121v2)**

> **作者:** Yiming Lei; Zhizheng Yang; Zeming Liu; Haitao Leng; Shaoguo Liu; Tingting Gao; Qingjie Liu; Yunhong Wang
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Multi-modal large language models have demonstrated remarkable zero-shot abilities and powerful image-understanding capabilities. However, the existing open-source multi-modal models suffer from the weak capability of multi-turn interaction, especially for long contexts. To address the issue, we first introduce a context modeling module, termed ContextQFormer, which utilizes a memory block to enhance the presentation of contextual information. Furthermore, to facilitate further research, we carefully build a new multi-turn multi-modal dialogue dataset (TMDialog) for pre-training, instruction-tuning, and evaluation, which will be open-sourced lately. Compared with other multi-modal dialogue datasets, TMDialog contains longer conversations, which supports the research of multi-turn multi-modal dialogue. In addition, ContextQFormer is compared with three baselines on TMDialog and experimental results illustrate that ContextQFormer achieves an improvement of 2%-4% in available rate over baselines.
>
---
#### [replaced 008] Prompt Perturbations Reveal Human-Like Biases in LLM Survey Responses
- **分类: cs.CL; cs.AI; cs.CY; J.4**

- **链接: [http://arxiv.org/pdf/2507.07188v2](http://arxiv.org/pdf/2507.07188v2)**

> **作者:** Jens Rupprecht; Georg Ahnert; Markus Strohmaier
>
> **备注:** 18 pages, 17 figures
>
> **摘要:** Large Language Models (LLMs) are increasingly used as proxies for human subjects in social science surveys, but their reliability and susceptibility to known response biases are poorly understood. This paper investigates the response robustness of LLMs in normative survey contexts - we test nine diverse LLMs on questions from the World Values Survey (WVS), applying a comprehensive set of 11 perturbations to both question phrasing and answer option structure, resulting in over 167,000 simulated interviews. In doing so, we not only reveal LLMs' vulnerabilities to perturbations but also show that all tested models exhibit a consistent recency bias varying in intensity, disproportionately favoring the last-presented answer option. While larger models are generally more robust, all models remain sensitive to semantic variations like paraphrasing and to combined perturbations. By applying a set of perturbations, we reveal that LLMs partially align with survey response biases identified in humans. This underscores the critical importance of prompt design and robustness testing when using LLMs to generate synthetic survey data.
>
---
#### [replaced 009] Exploiting Adaptive Contextual Masking for Aspect-Based Sentiment Analysis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2402.13722v2](http://arxiv.org/pdf/2402.13722v2)**

> **作者:** S M Rafiuddin; Mohammed Rakib; Sadia Kamal; Arunkumar Bagavathi
>
> **备注:** 12 pages, 4 figures, Accepted at PAKDD 2024
>
> **摘要:** Aspect-Based Sentiment Analysis (ABSA) is a fine-grained linguistics problem that entails the extraction of multifaceted aspects, opinions, and sentiments from the given text. Both standalone and compound ABSA tasks have been extensively used in the literature to examine the nuanced information present in online reviews and social media posts. Current ABSA methods often rely on static hyperparameters for attention-masking mechanisms, which can struggle with context adaptation and may overlook the unique relevance of words in varied situations. This leads to challenges in accurately analyzing complex sentences containing multiple aspects with differing sentiments. In this work, we present adaptive masking methods that remove irrelevant tokens based on context to assist in Aspect Term Extraction and Aspect Sentiment Classification subtasks of ABSA. We show with our experiments that the proposed methods outperform the baseline methods in terms of accuracy and F1 scores on four benchmark online review datasets. Further, we show that the proposed methods can be extended with multiple adaptations and demonstrate a qualitative analysis of the proposed approach using sample text for aspect term extraction.
>
---
#### [replaced 010] MAC-Tuning: LLM Multi-Compositional Problem Reasoning with Enhanced Knowledge Boundary Awareness
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.21773v2](http://arxiv.org/pdf/2504.21773v2)**

> **作者:** Junsheng Huang; Zhitao He; Yucheng Huang; Sandeep Polisetty; Qingyun Wang; May Fung
>
> **摘要:** With the widespread application of large language models (LLMs), the issue of generating non-existing facts, known as hallucination, has garnered increasing attention. Previous research in enhancing LLM confidence estimation mainly focuses on the single problem setting. However, LLM awareness of its internal parameterized knowledge boundary under the more challenging multi-problem setting, which requires answering multiple problems accurately simultaneously, remains underexplored. To bridge this gap, we introduce a novel method, Multiple Answers and Confidence Stepwise Tuning (MAC-Tuning), that separates the learning of answer prediction and confidence estimation during fine-tuning on instruction data. Extensive experiments demonstrate that our method outperforms baselines by up to 25% in average precision.
>
---
#### [replaced 011] Identifying Task Groupings for Multi-Task Learning Using Pointwise V-Usable Information
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.12774v2](http://arxiv.org/pdf/2410.12774v2)**

> **作者:** Yingya Li; Timothy Miller; Steven Bethard; Guergana Savova
>
> **备注:** main paper 12 pages, Appendix 7 pages, 1 figure, 18 tables
>
> **摘要:** The success of multi-task learning can depend heavily on which tasks are grouped together. Naively grouping all tasks or a random set of tasks can result in negative transfer, with the multi-task models performing worse than single-task models. Though many efforts have been made to identify task groupings and to measure the relatedness among different tasks, it remains a challenging research topic to define a metric to identify the best task grouping out of a pool of many potential task combinations. We propose a metric of task relatedness based on task difficulty measured by pointwise V-usable information (PVI). PVI is a recently proposed metric to estimate how much usable information a dataset contains given a model. We hypothesize that tasks with not statistically different PVI estimates are similar enough to benefit from the joint learning process. We conduct comprehensive experiments to evaluate the feasibility of this metric for task grouping on 15 NLP datasets in the general, biomedical, and clinical domains. We compare the results of the joint learners against single learners, existing baseline methods, and recent large language models, including Llama 2 and GPT-4. The results show that by grouping tasks with similar PVI estimates, the joint learners yielded competitive results with fewer total parameters, with consistent performance across domains.
>
---
#### [replaced 012] ReCode: Updating Code API Knowledge with Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2506.20495v2](http://arxiv.org/pdf/2506.20495v2)**

> **作者:** Haoze Wu; Yunzhi Yao; Wenhao Yu; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** Large Language Models (LLMs) exhibit remarkable code generation capabilities but falter when adapting to frequent updates in external library APIs. This critical limitation, stemming from reliance on outdated API knowledge from their training data, even with access to current documentation, impedes reliable code generation in dynamic environments. To tackle this issue, we propose ReCode (rule-based Reinforcement learning for Code Update), a novel framework that mimics human programmer adaptation to API changes. Specifically, we construct a dataset of approximately 2,000 data entries to train the LLMs to perform version migration based on updated information. Then, we introduce a modified string similarity metric for code evaluation as the reward for reinforcement learning. Our experiments demonstrate that ReCode substantially boosts LLMs' code generation performance in dynamic API scenarios, especially on the unseen CodeUpdateArena task. Crucially, compared to supervised fine-tuning, ReCode has less impact on LLMs' general code generation abilities. We apply ReCode on various LLMs and reinforcement learning algorithms (GRPO and DAPO), all achieving consistent improvements. Notably, after training, Qwen2.5-Coder-7B outperforms that of the 32B parameter code instruction-tuned model and the reasoning model with the same architecture. Code is available at https://github.com/zjunlp/ReCode.
>
---
#### [replaced 013] A Logically Consistent Chain-of-Thought Approach for Stance Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2312.16054v2](http://arxiv.org/pdf/2312.16054v2)**

> **作者:** Bowen Zhang; Daijun Ding; Liwen Jing; Hu Huang
>
> **摘要:** Zero-shot stance detection (ZSSD) aims to detect stances toward unseen targets. Incorporating background knowledge to enhance transferability between seen and unseen targets constitutes the primary approach of ZSSD. However, these methods often struggle with a knowledge-task disconnect and lack logical consistency in their predictions. To address these issues, we introduce a novel approach named Logically Consistent Chain-of-Thought (LC-CoT) for ZSSD, which improves stance detection by ensuring relevant and logically sound knowledge extraction. LC-CoT employs a three-step process. Initially, it assesses whether supplementary external knowledge is necessary. Subsequently, it uses API calls to retrieve this knowledge, which can be processed by a separate LLM. Finally, a manual exemplar guides the LLM to infer stance categories, using an if-then logical structure to maintain relevance and logical coherence. This structured approach to eliciting background knowledge enhances the model's capability, outperforming traditional supervised methods without relying on labeled data.
>
---
#### [replaced 014] What Factors Affect LLMs and RLLMs in Financial Question Answering?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.08339v2](http://arxiv.org/pdf/2507.08339v2)**

> **作者:** Peng Wang; Xuesi Hu; Jiageng Wu; Yuntao Zou; Qiancheng Zhang; Dagang Li
>
> **备注:** Preprint
>
> **摘要:** Recently, the development of large language models (LLMs) and reasoning large language models (RLLMs) have gained considerable attention from many researchers. RLLMs enhance the reasoning capabilities of LLMs through Long Chain-of-Thought (Long CoT) processes, significantly improving the performance of LLMs in addressing complex problems. However, there are few works that systematically explore what methods can fully unlock the performance of LLMs and RLLMs within the financial domain. To investigate the impact of various methods on LLMs and RLLMs, we utilize five LLMs and three RLLMs to assess the effects of prompting methods, agentic frameworks, and multilingual alignment methods on financial question-answering tasks. Our research findings indicate: (1) Current prompting methods and agent frameworks enhance the performance of LLMs in financial question answering by simulating Long CoT; (2) RLLMs possess inherent Long CoT capabilities, which limits the effectiveness of conventional methods in further enhancing their performance; (3) Current advanced multilingual alignment methods primarily improve the multilingual performance of LLMs by extending the reasoning length, which yields minimal benefits for RLLMs. We hope that this study can serve as an important reference for LLMs and RLLMs in the field of financial question answering.
>
---
#### [replaced 015] A Comparative Approach to Assessing Linguistic Creativity of Large Language Models and Humans
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.12039v2](http://arxiv.org/pdf/2507.12039v2)**

> **作者:** Anca Dinu; Andra-Maria Florescu; Alina Resceanu
>
> **备注:** Accepted for presentation at KES 2025. To appear in Procedia Computer Science (Elsevier)
>
> **摘要:** The following paper introduces a general linguistic creativity test for humans and Large Language Models (LLMs). The test consists of various tasks aimed at assessing their ability to generate new original words and phrases based on word formation processes (derivation and compounding) and on metaphorical language use. We administered the test to 24 humans and to an equal number of LLMs, and we automatically evaluated their answers using OCSAI tool for three criteria: Originality, Elaboration, and Flexibility. The results show that LLMs not only outperformed humans in all the assessed criteria, but did better in six out of the eight test tasks. We then computed the uniqueness of the individual answers, which showed some minor differences between humans and LLMs. Finally, we performed a short manual analysis of the dataset, which revealed that humans are more inclined towards E(extending)-creativity, while LLMs favor F(ixed)-creativity.
>
---
#### [replaced 016] A Multi-Stage Framework with Taxonomy-Guided Reasoning for Occupation Classification Using Large Language Models
- **分类: cs.CL; cs.AI; cs.SI**

- **链接: [http://arxiv.org/pdf/2503.12989v2](http://arxiv.org/pdf/2503.12989v2)**

> **作者:** Palakorn Achananuparp; Ee-Peng Lim; Yao Lu
>
> **备注:** Accepted to ICWSM'26
>
> **摘要:** Automatically annotating job data with standardized occupations from taxonomies, known as occupation classification, is crucial for labor market analysis. However, this task is often hindered by data scarcity and the challenges of manual annotations. While large language models (LLMs) hold promise due to their extensive world knowledge and in-context learning capabilities, their effectiveness depends on their knowledge of occupational taxonomies, which remains unclear. In this study, we assess the ability of LLMs to generate precise taxonomic entities from taxonomy, highlighting their limitations, especially for smaller models. To address these challenges, we propose a multi-stage framework consisting of inference, retrieval, and reranking stages, which integrates taxonomy-guided reasoning examples to enhance performance by aligning outputs with taxonomic knowledge. Evaluations on a large-scale dataset show that our framework not only enhances occupation and skill classification tasks, but also provides a cost-effective alternative to frontier models like GPT-4o, significantly reducing computational costs while maintaining strong performance. This makes it a practical and scalable solution for occupation classification and related tasks across LLMs.
>
---
#### [replaced 017] LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.20625v2](http://arxiv.org/pdf/2410.20625v2)**

> **作者:** Jui-Nan Yen; Si Si; Zhao Meng; Felix Yu; Sai Surya Duvvuri; Inderjit S. Dhillon; Cho-Jui Hsieh; Sanjiv Kumar
>
> **备注:** Published as an oral paper at ICLR 2025. The code for our project is available at https://github.com/gkevinyen5418/LoRA-RITE
>
> **摘要:** Low-rank adaption (LoRA) is a widely used parameter-efficient finetuning method for LLM that reduces memory requirements. However, current LoRA optimizers lack transformation invariance, meaning the actual updates to the weights depends on how the two LoRA factors are scaled or rotated. This deficiency leads to inefficient learning and sub-optimal solutions in practice. This paper introduces LoRA-RITE, a novel adaptive matrix preconditioning method for LoRA optimization, which can achieve transformation invariance and remain computationally efficient. We provide theoretical analysis to demonstrate the benefit of our method and conduct experiments on various LLM tasks with different models including Gemma 2B, 7B, and mT5-XXL. The results demonstrate consistent improvements against existing optimizers. For example, replacing Adam with LoRA-RITE during LoRA fine-tuning of Gemma-2B yielded 4.6\% accuracy gain on Super-Natural Instructions and 3.5\% accuracy gain across other four LLM benchmarks (HellaSwag, ArcChallenge, GSM8K, OpenBookQA).
>
---
#### [replaced 018] SWE-MERA: A Dynamic Benchmark for Agenticly Evaluating Large Language Models on Software Engineering Tasks
- **分类: cs.SE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.11059v2](http://arxiv.org/pdf/2507.11059v2)**

> **作者:** Pavel Adamenko; Mikhail Ivanov; Aidar Valeev; Rodion Levichev; Pavel Zadorozhny; Ivan Lopatin; Dmitry Babayev; Alena Fenogenova; Valentin Malykh
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) in software engineering has revealed critical limitations in existing benchmarks, particularly the widely used SWE-bench dataset. Recent studies have uncovered severe data contamination issues, e.g. SWE-bench reports 32.67% of successful patches involve direct solution leakage and 31.08% pass due to inadequate test cases. We introduce SWE-MERA, a dynamic, continuously updated benchmark designed to address these fundamental challenges through an automated collection of real-world GitHub issues and rigorous quality validation. Our approach implements a reliable pipeline that ensures quality while minimizing contamination risks, resulting in approximately 10,000 potential tasks with 300 samples currently available. Evaluation using the Aider coding agent demonstrates strong discriminative power in state-of-the-art models. We report performance across a dozen recent LLMs evaluated on tasks collected between September 2024 and June 2025.
>
---
#### [replaced 019] Code2Logic: Game-Code-Driven Data Synthesis for Enhancing VLMs General Reasoning
- **分类: cs.CL; I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2505.13886v3](http://arxiv.org/pdf/2505.13886v3)**

> **作者:** Jingqi Tong; Jixin Tang; Hangcheng Li; Yurong Mou; Ming Zhang; Jun Zhao; Yanbo Wen; Fan Song; Jiahao Zhan; Yuyang Lu; Chaoran Tao; Zhiyuan Guo; Jizhou Yu; Tianhao Cheng; Changhao Jiang; Zhen Wang; Tao Liang; Zhihui Fei; Mingyang Wan; Guojun Ma; Weifeng Ge; Guanhua Chen; Tao Gui; Xipeng Qiu; Qi Zhang; Xuanjing Huang
>
> **备注:** 63 pages, 23 figures, submitted to NeurIPS 2025
>
> **摘要:** Visual-language Chain-of-Thought (CoT) data resources are relatively scarce compared to text-only counterparts, limiting the improvement of reasoning capabilities in Vision Language Models (VLMs). However, high-quality vision-language reasoning data is expensive and labor-intensive to annotate. To address this issue, we leverage a promising resource: game code, which naturally contains logical structures and state transition processes. Therefore, we propose Code2Logic, a novel game-code-driven approach for multimodal reasoning data synthesis. Our approach leverages Large Language Models (LLMs) to adapt game code, enabling automatic acquisition of reasoning processes and results through code execution. Using the Code2Logic approach, we developed the GameQA dataset to train and evaluate VLMs. GameQA is cost-effective and scalable, offers controllable difficulty gradation and is diverse with 30 games and 158 tasks. Surprisingly, despite training solely on game data, VLMs demonstrated out of domain generalization, specifically Qwen2.5-VL-7B improving performance by 2.33% across 7 diverse vision-language benchmarks. Our code, dataset and models are available at https://github.com/tongjingqi/Code2Logic.
>
---
#### [replaced 020] MPO: An Efficient Post-Processing Framework for Mixing Diverse Preference Alignment
- **分类: cs.CL; cs.LG; stat.ME**

- **链接: [http://arxiv.org/pdf/2502.18699v2](http://arxiv.org/pdf/2502.18699v2)**

> **作者:** Tianze Wang; Dongnan Gui; Yifan Hu; Shuhang Lin; Linjun Zhang
>
> **备注:** ICML 2025
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) has shown promise in aligning large language models (LLMs). Yet its reliance on a singular reward model often overlooks the diversity of human preferences. Recent approaches address this limitation by leveraging multi-dimensional feedback to fine-tune corresponding reward models and train LLMs using reinforcement learning. However, the process is costly and unstable, especially given the competing and heterogeneous nature of human preferences. In this paper, we propose Mixing Preference Optimization (MPO), a post-processing framework for aggregating single-objective policies as an alternative to both multi-objective RLHF (MORLHF) and MaxMin-RLHF. MPO avoids alignment from scratch. Instead, it log-linearly combines existing policies into a unified one with the weight of each policy computed via a batch stochastic mirror descent. Empirical results demonstrate that MPO achieves balanced performance across diverse preferences, outperforming or matching existing models with significantly reduced computational costs.
>
---
#### [replaced 021] Cross-Layer Discrete Concept Discovery for Interpreting Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.20040v2](http://arxiv.org/pdf/2506.20040v2)**

> **作者:** Ankur Garg; Xuemin Yu; Hassan Sajjad; Samira Ebrahimi Kahou
>
> **摘要:** Uncovering emergent concepts across transformer layers remains a significant challenge because the residual stream linearly mixes and duplicates information, obscuring how features evolve within large language models. Current research efforts primarily inspect neural representations at single layers, thereby overlooking this cross-layer superposition and the redundancy it introduces. These representations are typically either analyzed directly for activation patterns or passed to probing classifiers that map them to a limited set of predefined concepts. To address these limitations, we propose cross-layer VQ-VAE (CLVQ-VAE), a framework that uses vector quantization to map representations across layers and in the process collapse duplicated residual-stream features into compact, interpretable concept vectors. Our approach uniquely combines top-k temperature-based sampling during quantization with EMA codebook updates, providing controlled exploration of the discrete latent space while maintaining code-book diversity. We further enhance the framework with scaled-spherical k-means++ for codebook initialization, which clusters by directional similarity rather than magnitude, better aligning with semantic structure in word embedding space.
>
---
#### [replaced 022] MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2506.15841v2](http://arxiv.org/pdf/2506.15841v2)**

> **作者:** Zijian Zhou; Ao Qu; Zhaoxuan Wu; Sunghwan Kim; Alok Prakash; Daniela Rus; Jinhua Zhao; Bryan Kian Hsiang Low; Paul Pu Liang
>
> **摘要:** Modern language agents must operate over long-horizon, multi-turn interactions, where they retrieve external information, adapt to observations, and answer interdependent queries. Yet, most LLM systems rely on full-context prompting, appending all past turns regardless of their relevance. This leads to unbounded memory growth, increased computational costs, and degraded reasoning performance on out-of-distribution input lengths. We introduce MEM1, an end-to-end reinforcement learning framework that enables agents to operate with constant memory across long multi-turn tasks. At each turn, MEM1 updates a compact shared internal state that jointly supports memory consolidation and reasoning. This state integrates prior memory with new observations from the environment while strategically discarding irrelevant or redundant information. To support training in more realistic and compositional settings, we propose a simple yet effective and scalable approach to constructing multi-turn environments by composing existing datasets into arbitrarily complex task sequences. Experiments across three domains, including internal retrieval QA, open-domain web QA, and multi-turn web shopping, show that MEM1-7B improves performance by 3.5x while reducing memory usage by 3.7x compared to Qwen2.5-14B-Instruct on a 16-objective multi-hop QA task, and generalizes beyond the training horizon. Our results demonstrate the promise of reasoning-driven memory consolidation as a scalable alternative to existing solutions for training long-horizon interactive agents, where both efficiency and performance are optimized.
>
---
#### [replaced 023] Fairness Is Not Enough: Auditing Competence and Intersectional Bias in AI-powered Resume Screening
- **分类: cs.CY; cs.AI; cs.CL; I.2.1; K.4.2; I.2.6; K.4.1**

- **链接: [http://arxiv.org/pdf/2507.11548v2](http://arxiv.org/pdf/2507.11548v2)**

> **作者:** Kevin T Webster
>
> **备注:** 34 pages, 4 figures
>
> **摘要:** The increasing use of generative AI for resume screening is predicated on the assumption that it offers an unbiased alternative to biased human decision-making. However, this belief fails to address a critical question: are these AI systems fundamentally competent at the evaluative tasks they are meant to perform? This study investigates the question of competence through a two-part audit of eight major AI platforms. Experiment 1 confirmed complex, contextual racial and gender biases, with some models penalizing candidates merely for the presence of demographic signals. Experiment 2, which evaluated core competence, provided a critical insight: some models that appeared unbiased were, in fact, incapable of performing a substantive evaluation, relying instead on superficial keyword matching. This paper introduces the "Illusion of Neutrality" to describe this phenomenon, where an apparent lack of bias is merely a symptom of a model's inability to make meaningful judgments. This study recommends that organizations and regulators adopt a dual-validation framework, auditing AI hiring tools for both demographic bias and demonstrable competence to ensure they are both equitable and effective.
>
---
#### [replaced 024] SCULPT: Systematic Tuning of Long Prompts
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.20788v3](http://arxiv.org/pdf/2410.20788v3)**

> **作者:** Shanu Kumar; Akhila Yesantarao Venkata; Shubhanshu Khandelwal; Bishal Santra; Parag Agrawal; Manish Gupta
>
> **备注:** Accepted at ACL Main 2025
>
> **摘要:** Prompt optimization is essential for effective utilization of large language models (LLMs) across diverse tasks. While existing optimization methods are effective in optimizing short prompts, they struggle with longer, more complex ones, often risking information loss and being sensitive to small perturbations. To address these challenges, we propose SCULPT (Systematic Tuning of Long Prompts), a framework that treats prompt optimization as a hierarchical tree refinement problem. SCULPT represents prompts as tree structures, enabling targeted modifications while preserving contextual integrity. It employs a Critic-Actor framework that generates reflections and applies actions to refine the prompt. Evaluations demonstrate SCULPT's effectiveness on long prompts, its robustness to adversarial perturbations, and its ability to generate high-performing prompts even without any initial human-written prompt. Compared to existing state of the art methods, SCULPT consistently improves LLM performance by preserving essential task information while applying structured refinements. Both qualitative and quantitative analyses show that SCULPT produces more stable and interpretable prompt modifications, ensuring better generalization across tasks.
>
---
#### [replaced 025] GUI Test Migration via Abstraction and Concretization
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.05028v2](http://arxiv.org/pdf/2409.05028v2)**

> **作者:** Yakun Zhang; Chen Liu; Xiaofei Xie; Yun Lin; Jin Song Dong; Dan Hao; Lu Zhang
>
> **备注:** This paper has been accepted for publication in ACM Transactions on Software Engineering and Methodology (TOSEM) in 2025. The official publication link is: https://dl.acm.org/doi/10.1145/3726525
>
> **摘要:** GUI test migration aims to produce test cases with events and assertions to test specific functionalities of a target app. Existing migration approaches typically focus on the widget-mapping paradigm that maps widgets from source apps to target apps. However, since different apps may implement the same functionality in different ways, direct mapping may result in incomplete or buggy test cases, thus significantly impacting the effectiveness of testing target functionality and the practical applicability of migration approaches. In this paper, we propose a new migration paradigm (i.e., the abstraction-concretization paradigm) that first abstracts the test logic for the target functionality and then utilizes this logic to generate the concrete GUI test case. Furthermore, we introduce MACdroid, the first approach that migrates GUI test cases based on this paradigm. Specifically, we propose an abstraction technique that utilizes source test cases from source apps targeting the same functionality to extract a general test logic for that functionality. Then, we propose a concretization technique that utilizes the general test logic to guide an LLM in generating the corresponding GUI test case (including events and assertions) for the target app. We evaluate MACdroid on two widely-used datasets (including 31 apps, 34 functionalities, and 123 test cases). On the FrUITeR dataset, the test cases generated by MACdroid successfully test 64% of the target functionalities, improving the baselines by 191%. On the Lin dataset, MACdroid successfully tests 75% of the target functionalities, outperforming the baselines by 42%. These results underscore the effectiveness of MACdroid in GUI test migration.
>
---
#### [replaced 026] OASIS: Order-Augmented Strategy for Improved Code Search
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2503.08161v4](http://arxiv.org/pdf/2503.08161v4)**

> **作者:** Zuchen Gao; Zizheng Zhan; Xianming Li; Erxin Yu; Ziqi Zhan; Haotian Zhang; Bin Chen; Yuqun Zhang; Jing Li
>
> **摘要:** Code embeddings capture the semantic representations of code and are crucial for various code-related large language model (LLM) applications, such as code search. Previous training primarily relies on optimizing the InfoNCE loss by comparing positive natural language (NL)-code pairs with in-batch negatives. However, due to the sparse nature of code contexts, training solely by comparing the major differences between positive and negative pairs may fail to capture deeper semantic nuances. To address this issue, we propose a novel order-augmented strategy for improved code search (OASIS). It leverages order-based similarity labels to train models to capture subtle differences in similarity among negative pairs. Extensive benchmark evaluations demonstrate that our OASIS model significantly outperforms previous state-of-the-art models focusing solely on major positive-negative differences. It underscores the value of exploiting subtle differences among negative pairs with order labels for effective code embedding training.
>
---
#### [replaced 027] CoDet-M4: Detecting Machine-Generated Code in Multi-Lingual, Multi-Generator and Multi-Domain Settings
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.13733v2](http://arxiv.org/pdf/2503.13733v2)**

> **作者:** Daniil Orel; Dilshod Azizov; Preslav Nakov
>
> **摘要:** Large language models (LLMs) have revolutionized code generation, automating programming with remarkable efficiency. However, these advancements challenge programming skills, ethics, and assessment integrity, making the detection of LLM-generated code essential for maintaining accountability and standards. While, there has been some research on this problem, it generally lacks domain coverage and robustness, and only covers a small number of programming languages. To this end, we propose a framework capable of distinguishing between human- and LLM-written code across multiple programming languages, code generators, and domains. We use a large-scale dataset from renowned platforms and LLM-based code generators, alongside applying rigorous data quality checks, feature engineering, and comparative analysis using evaluation of traditional machine learning models, pre-trained language models (PLMs), and LLMs for code detection. We perform an evaluation on out-of-domain scenarios, such as detecting the authorship and hybrid authorship of generated code and generalizing to unseen models, domains, and programming languages. Moreover, our extensive experiments show that our framework effectively distinguishes human- from LLM-written code and sets a new benchmark for this task.
>
---
#### [replaced 028] Secure Multifaceted-RAG for Enterprise: Hybrid Knowledge Retrieval with Security Filtering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.13425v2](http://arxiv.org/pdf/2504.13425v2)**

> **作者:** Grace Byun; Shinsun Lee; Nayoung Choi; Jinho D. Choi
>
> **摘要:** Existing Retrieval-Augmented Generation (RAG) systems face challenges in enterprise settings due to limited retrieval scope and data security risks. When relevant internal documents are unavailable, the system struggles to generate accurate and complete responses. Additionally, using closed-source Large Language Models (LLMs) raises concerns about exposing proprietary information. To address these issues, we propose the Secure Multifaceted-RAG (SecMulti-RAG) framework, which retrieves not only from internal documents but also from two supplementary sources: pre-generated expert knowledge for anticipated queries and on-demand external LLM-generated knowledge. To mitigate security risks, we adopt a local open-source generator and selectively utilize external LLMs only when prompts are deemed safe by a filtering mechanism. This approach enhances completeness, prevents data leakage, and reduces costs. In our evaluation on a report generation task in the automotive industry, SecMulti-RAG significantly outperforms traditional RAG - achieving 79.3 to 91.9 percent win rates across correctness, richness, and helpfulness in LLM-based evaluation, and 56.3 to 70.4 percent in human evaluation. This highlights SecMulti-RAG as a practical and secure solution for enterprise RAG.
>
---
#### [replaced 029] ActionStudio: A Lightweight Framework for Data and Training of Large Action Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.22673v3](http://arxiv.org/pdf/2503.22673v3)**

> **作者:** Jianguo Zhang; Thai Hoang; Ming Zhu; Zuxin Liu; Shiyu Wang; Tulika Awalgaonkar; Akshara Prabhakar; Haolin Chen; Weiran Yao; Zhiwei Liu; Juntao Tan; Juan Carlos Niebles; Shelby Heinecke; Huan Wang; Silvio Savarese; Caiming Xiong
>
> **备注:** 16 pages; large action models; xLAM; ActionStudio
>
> **摘要:** Large Action models are essential for enabling autonomous agents to perform complex tasks. However, training such models remains challenging due to the diversity of agent environments and the complexity of noisy agentic data. Existing infrastructure offers limited support for scalable, agent-specific fine-tuning and standardized agent data processing. We introduce ActionStudio, a lightweight and extensible data and training framework designed for large action models. ActionStudio unifies diverse agent trajectories using our proposed Unified Format 2.0, supports a range of training workflows with optimized multi-node distributed setup, and integrates robust preprocessing and real-time verification tools. ActionStudio demonstrates up to 9x higher throughput compared to existing agentic training frameworks, and our trained models yield top performances across public and realistic agent benchmarks. To support the broader research community, we open-source the ActionStudio framework and release actionstudio-98k, a curated dataset of 98k high-quality trajectories. Code: https://github.com/SalesforceAIResearch/xLAM.
>
---
#### [replaced 030] Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.06261v3](http://arxiv.org/pdf/2507.06261v3)**

> **作者:** Gheorghe Comanici; Eric Bieber; Mike Schaekermann; Ice Pasupat; Noveen Sachdeva; Inderjit Dhillon; Marcel Blistein; Ori Ram; Dan Zhang; Evan Rosen; Luke Marris; Sam Petulla; Colin Gaffney; Asaf Aharoni; Nathan Lintz; Tiago Cardal Pais; Henrik Jacobsson; Idan Szpektor; Nan-Jiang Jiang; Krishna Haridasan; Ahmed Omran; Nikunj Saunshi; Dara Bahri; Gaurav Mishra; Eric Chu; Toby Boyd; Brad Hekman; Aaron Parisi; Chaoyi Zhang; Kornraphop Kawintiranon; Tania Bedrax-Weiss; Oliver Wang; Ya Xu; Ollie Purkiss; Uri Mendlovic; Ilaï Deutel; Nam Nguyen; Adam Langley; Flip Korn; Lucia Rossazza; Alexandre Ramé; Sagar Waghmare; Helen Miller; Vaishakh Keshava; Ying Jian; Xiaofan Zhang; Raluca Ada Popa; Kedar Dhamdhere; Blaž Bratanič; Kyuyeun Kim; Terry Koo; Ferran Alet; Yi-ting Chen; Arsha Nagrani; Hannah Muckenhirn; Zhiyuan Zhang; Corbin Quick; Filip Pavetić; Duc Dung Nguyen; Joao Carreira; Michael Elabd; Haroon Qureshi; Fabian Mentzer; Yao-Yuan Yang; Danielle Eisenbud; Anmol Gulati; Ellie Talius; Eric Ni; Sahra Ghalebikesabi; Edouard Yvinec; Alaa Saade; Thatcher Ulrich; Lorenzo Blanco; Dan A. Calian; Muhuan Huang; Aäron van den Oord; Naman Goyal; Terry Chen; Praynaa Rawlani; Christian Schallhart; Swachhand Lokhande; Xianghong Luo; Jyn Shan; Ceslee Montgomery; Victoria Krakovna; Federico Piccinini; Omer Barak; Jingyu Cui; Yiling Jia; Mikhail Dektiarev; Alexey Kolganov; Shiyu Huang; Zhe Chen; Xingyu Wang; Jessica Austin; Peter de Boursac; Evgeny Sluzhaev; Frank Ding; Huijian Li; Surya Bhupatiraju; Mohit Agarwal; Sławek Kwasiborski; Paramjit Sandhu; Patrick Siegler; Ahmet Iscen; Eyal Ben-David; Shiraz Butt; Miltos Allamanis; Seth Benjamin; Robert Busa-Fekete; Felix Hernandez-Campos; Sasha Goldshtein; Matt Dibb; Weiyang Zhang; Annie Marsden; Carey Radebaugh; Stephen Roller; Abhishek Nayyar; Jacob Austin; Tayfun Terzi; Bhargav Kanagal Shamanna; Pete Shaw; Aayush Singh; Florian Luisier; Artur Mendonça; Vaibhav Aggarwal; Larisa Markeeva; Claudio Fantacci; Sergey Brin; HyunJeong Choe; Guanyu Wang; Hartwig Adam; Avigail Dabush; Tatsuya Kiyono; Eyal Marcus; Jeremy Cole; Theophane Weber; Hongrae Lee; Ronny Huang; Alex Muzio; Leandro Kieliger; Maigo Le; Courtney Biles; Long Le; Archit Sharma; Chengrun Yang; Avery Lamp; Dave Dopson; Nate Hurley; Katrina; Xu; Zhihao Shan; Shuang Song; Jiewen Tan; Alexandre Senges; George Zhang; Chong You; Yennie Jun; David Raposo; Susanna Ricco; Xuan Yang; Weijie Chen; Prakhar Gupta; Arthur Szlam; Kevin Villela; Chun-Sung Ferng; Daniel Kasenberg; Chen Liang; Rui Zhu; Arunachalam Narayanaswamy; Florence Perot; Paul Pucciarelli; Anna Shekhawat; Alexey Stern; Rishikesh Ingale; Stefani Karp; Sanaz Bahargam; Adrian Goedeckemeyer; Jie Han; Sicheng Li; Andrea Tacchetti; Dian Yu; Abhishek Chakladar; Zhiying Zhang; Mona El Mahdy; Xu Gao; Dale Johnson; Samrat Phatale; AJ Piergiovanni; Hyeontaek Lim; Clement Farabet; Carl Lebsack; Theo Guidroz; John Blitzer; Nico Duduta; David Madras; Steve Li; Daniel von Dincklage; Xin Li; Mahdis Mahdieh; George Tucker; Ganesh Jawahar; Owen Xiao; Danny Tarlow; Robert Geirhos; Noam Velan; Daniel Vlasic; Kalesha Bullard; SK Park; Nishesh Gupta; Kellie Webster; Ayal Hitron; Jieming Mao; Julian Eisenschlos; Laurel Prince; Nina D'Souza; Kelvin Zheng; Sara Nasso; Gabriela Botea; Carl Doersch; Caglar Unlu; Chris Alberti; Alexey Svyatkovskiy; Ankita Goel; Krzysztof Choromanski; Pan-Pan Jiang; Richard Nguyen; Four Flynn; Daria Ćurko; Peter Chen; Nicholas Roth; Kieran Milan; Caleb Habtegebriel; Shashi Narayan; Michael Moffitt; Jake Marcus; Thomas Anthony; Brendan McMahan; Gowoon Cheon; Ruibo Liu; Megan Barnes; Lukasz Lew; Rebeca Santamaria-Fernandez; Mayank Upadhyay; Arjun Akula; Arnar Mar Hrafnkelsson; Alvaro Caceres; Andrew Bunner; Michal Sokolik; Subha Puttagunta; Lawrence Moore; Berivan Isik; Jay Hartford; Lawrence Chan; Pradeep Shenoy; Dan Holtmann-Rice; Jane Park; Fabio Viola; Alex Salcianu; Sujeevan Rajayogam; Ian Stewart-Binks; Zelin Wu; Richard Everett; Xi Xiong; Pierre-Antoine Manzagol; Gary Leung; Carl Saroufim; Bo Pang; Dawid Wegner; George Papamakarios; Jennimaria Palomaki; Helena Pankov; Guangda Lai; Guilherme Tubone; Shubin Zhao; Theofilos Strinopoulos; Seth Neel; Mingqiu Wang; Joe Kelley; Li Li; Pingmei Xu; Anitha Vijayakumar; Andrea D'olimpio; Omer Levy; Massimo Nicosia; Grigory Rozhdestvenskiy; Ni Lao; Sirui Xie; Yash Katariya; Jon Simon; Sanjiv Kumar; Florian Hartmann; Michael Kilgore; Jinhyuk Lee; Aroma Mahendru; Roman Ring; Tom Hennigan; Fiona Lang; Colin Cherry; David Steiner; Dawsen Hwang; Ray Smith; Pidong Wang; Jeremy Chen; Ming-Hsuan Yang; Sam Kwei; Philippe Schlattner; Donnie Kim; Ganesh Poomal Girirajan; Nikola Momchev; Ayushi Agarwal; Xingyi Zhou; Ilkin Safarli; Zachary Garrett; AJ Pierigiovanni; Sarthak Jauhari; Alif Raditya Rochman; Shikhar Vashishth; Quan Yuan; Christof Angermueller; Jon Blanton; Xinying Song; Nitesh Bharadwaj Gundavarapu; Thi Avrahami; Maxine Deines; Subhrajit Roy; Manish Gupta; Christopher Semturs; Shobha Vasudevan; Aditya Srikanth Veerubhotla; Shriya Sharma; Josh Jacob; Zhen Yang; Andreas Terzis; Dan Karliner; Auriel Wright; Tania Rojas-Esponda; Ashley Brown; Abhijit Guha Roy; Pawan Dogra; Andrei Kapishnikov; Peter Young; Wendy Kan; Vinodh Kumar Rajendran; Maria Ivanova; Salil Deshmukh; Chia-Hua Ho; Mike Kwong; Stav Ginzburg; Annie Louis; KP Sawhney; Slav Petrov; Jing Xie; Yunfei Bai; Georgi Stoyanov; Alex Fabrikant; Rajesh Jayaram; Yuqi Li; Joe Heyward; Justin Gilmer; Yaqing Wang; Radu Soricut; Luyang Liu; Qingnan Duan; Jamie Hayes; Maura O'Brien; Gaurav Singh Tomar; Sivan Eiger; Bahar Fatemi; Jeffrey Hui; Catarina Barros; Adaeze Chukwuka; Alena Butryna; Saksham Thakur; Austin Huang; Zhufeng Pan; Haotian Tang; Serkan Cabi; Tulsee Doshi; Michiel Bakker; Sumit Bagri; Ruy Ley-Wild; Adam Lelkes; Jennie Lees; Patrick Kane; David Greene; Shimu Wu; Jörg Bornschein; Gabriela Surita; Sarah Hodkinson; Fangtao Li; Chris Hidey; Sébastien Pereira; Sean Ammirati; Phillip Lippe; Adam Kraft; Pu Han; Sebastian Gerlach; Zifeng Wang; Liviu Panait; Feng Han; Brian Farris; Yingying Bi; Hannah DeBalsi; Miaosen Wang; Gladys Tyen; James Cohan; Susan Zhang; Jarred Barber; Da-Woon Chung; Jaeyoun Kim; Markus Kunesch; Steven Pecht; Nami Akazawa; Abe Friesen; James Lyon; Ali Eslami; Junru Wu; Jie Tan; Yue Song; Ravi Kumar; Chris Welty; Ilia Akolzin; Gena Gibson; Sean Augenstein; Arjun Pillai; Nancy Yuen; Du Phan; Xin Wang; Iain Barr; Heiga Zen; Nan Hua; Casper Liu; Jilei; Wang; Tanuj Bhatia; Hao Xu; Oded Elyada; Pushmeet Kohli; Mirek Olšák; Ke Chen; Azalia Mirhoseini; Noam Shazeer; Shoshana Jakobovits; Maggie Tran; Nolan Ramsden; Tarun Bharti; Fred Alcober; Yunjie Li; Shilpa Shetty; Jing Chen; Dmitry Kalashnikov; Megha Nawhal; Sercan Arik; Hanwen Chen; Michiel Blokzijl; Shubham Gupta; James Rubin; Rigel Swavely; Sophie Bridgers; Ian Gemp; Chen Su; Arun Suggala; Juliette Pluto; Mary Cassin; Alain Vaucher; Kaiyang Ji; Jiahao Cai; Andrew Audibert; Animesh Sinha; David Tian; Efrat Farkash; Amy Hua; Jilin Chen; Duc-Hieu Tran; Edward Loper; Nicole Brichtova; Lara McConnaughey; Ballie Sandhu; Robert Leland; Doug DeCarlo; Andrew Over; James Huang; Xing Wu; Connie Fan; Eric Li; Yun Lei; Deepak Sharma; Cosmin Paduraru; Luo Yu; Matko Bošnjak; Phuong Dao; Min Choi; Sneha Kudugunta; Jakub Adamek; Carlos Guía; Ali Khodaei; Jie Feng; Wenjun Zeng; David Welling; Sandeep Tata; Christina Butterfield; Andrey Vlasov; Seliem El-Sayed; Swaroop Mishra; Tara Sainath; Shentao Yang; RJ Skerry-Ryan; Jeremy Shar; Robert Berry; Arunkumar Rajendran; Arun Kandoor; Andrea Burns; Deepali Jain; Tom Stone; Wonpyo Park; Shibo Wang; Albin Cassirer; Guohui Wang; Hayato Kobayashi; Sergey Rogulenko; Vineetha Govindaraj; Mikołaj Rybiński; Nadav Olmert; Colin Evans; Po-Sen Huang; Kelvin Xu; Premal Shah; Terry Thurk; Caitlin Sikora; Mu Cai; Jin Xie; Elahe Dabir; Saloni Shah; Norbert Kalb; Carrie Zhang; Shruthi Prabhakara; Amit Sabne; Artiom Myaskovsky; Vikas Raunak; Blanca Huergo; Behnam Neyshabur; Jon Clark; Ye Zhang; Shankar Krishnan; Eden Cohen; Dinesh Tewari; James Lottes; Yumeya Yamamori; Hui; Li; Mohamed Elhawaty; Ada Maksutaj Oflazer; Adrià Recasens; Sheryl Luo; Duy Nguyen; Taylor Bos; Kalyan Andra; Ana Salazar; Ed Chi; Jeongwoo Ko; Matt Ginsberg; Anders Andreassen; Anian Ruoss; Todor Davchev; Elnaz Davoodi; Chenxi Liu; Min Kim; Santiago Ontanon; Chi Ming To; Dawei Jia; Rosemary Ke; Jing Wang; Anna Korsun; Moran Ambar; Ilya Kornakov; Irene Giannoumis; Toni Creswell; Denny Zhou; Yi Su; Ishaan Watts; Aleksandr Zaks; Evgenii Eltyshev; Ziqiang Feng; Sidharth Mudgal; Alex Kaskasoli; Juliette Love; Kingshuk Dasgupta; Sam Shleifer; Richard Green; Sungyong Seo; Chansoo Lee; Dale Webster; Prakash Shroff; Ganna Raboshchuk; Isabel Leal; James Manyika; Sofia Erell; Daniel Murphy; Zhisheng Xiao; Anton Bulyenov; Julian Walker; Mark Collier; Matej Kastelic; Nelson George; Sushant Prakash; Sailesh Sidhwani; Alexey Frolov; Steven Hansen; Petko Georgiev; Tiberiu Sosea; Chris Apps; Aishwarya Kamath; David Reid; Emma Cooney; Charlotte Magister; Oriana Riva; Alec Go; Pu-Chin Chen; Sebastian Krause; Nir Levine; Marco Fornoni; Ilya Figotin; Nick Roy; Parsa Mahmoudieh; Vladimir Magay; Mukundan Madhavan; Jin Miao; Jianmo Ni; Yasuhisa Fujii; Ian Chou; George Scrivener; Zak Tsai; Siobhan Mcloughlin; Jeremy Selier; Sandra Lefdal; Jeffrey Zhao; Abhijit Karmarkar; Kushal Chauhan; Shivanker Goel; Zhaoyi Zhang; Vihan Jain; Parisa Haghani; Mostafa Dehghani; Jacob Scott; Erin Farnese; Anastasija Ilić; Steven Baker; Julia Pawar; Li Zhong; Josh Camp; Yoel Zeldes; Shravya Shetty; Anand Iyer; Vít Listík; Jiaxian Guo; Luming Tang; Mark Geller; Simon Bucher; Yifan Ding; Hongzhi Shi; Carrie Muir; Dominik Grewe; Ramy Eskander; Octavio Ponce; Boqing Gong; Derek Gasaway; Samira Khan; Umang Gupta; Angelos Filos; Weicheng Kuo; Klemen Kloboves; Jennifer Beattie; Christian Wright; Leon Li; Alicia Jin; Sandeep Mariserla; Miteyan Patel; Jens Heitkaemper; Dilip Krishnan; Vivek Sharma; David Bieber; Christian Frank; John Lambert; Paul Caron; Martin Polacek; Mai Giménez; Himadri Choudhury; Xing Yu; Sasan Tavakkol; Arun Ahuja; Franz Och; Rodolphe Jenatton; Wojtek Skut; Bryan Richter; David Gaddy; Andy Ly; Misha Bilenko; Megh Umekar; Ethan Liang; Martin Sevenich; Mandar Joshi; Hassan Mansoor; Rebecca Lin; Sumit Sanghai; Abhimanyu Singh; Xiaowei Li; Sudheendra Vijayanarasimhan; Zaheer Abbas; Yonatan Bitton; Hansa Srinivasan; Manish Reddy Vuyyuru; Alexander Frömmgen; Yanhua Sun; Ralph Leith; Alfonso Castaño; DJ Strouse; Le Yan; Austin Kyker; Satish Kambala; Mary Jasarevic; Thibault Sellam; Chao Jia; Alexander Pritzel; Raghavender R; Huizhong Chen; Natalie Clay; Sudeep Gandhe; Sean Kirmani; Sayna Ebrahimi; Hannah Kirkwood; Jonathan Mallinson; Chao Wang; Adnan Ozturel; Kuo Lin; Shyam Upadhyay; Vincent Cohen-Addad; Sean Purser-haskell; Yichong Xu; Ebrahim Songhori; Babi Seal; Alberto Magni; Almog Gueta; Tingting Zou; Guru Guruganesh; Thais Kagohara; Hung Nguyen; Khalid Salama; Alejandro Cruzado Ruiz; Justin Frye; Zhenkai Zhu; Matthias Lochbrunner; Simon Osindero; Wentao Yuan; Lisa Lee; Aman Prasad; Lam Nguyen Thiet; Daniele Calandriello; Victor Stone; Qixuan Feng; Han Ke; Maria Voitovich; Geta Sampemane; Lewis Chiang; Ling Wu; Alexander Bykovsky; Matt Young; Luke Vilnis; Ishita Dasgupta; Aditya Chawla; Qin Cao; Bowen Liang; Daniel Toyama; Szabolcs Payrits; Anca Stefanoiu; Dimitrios Vytiniotis; Ankesh Anand; Tianxiao Shen; Blagoj Mitrevski; Michael Tschannen; Sreenivas Gollapudi; Aishwarya P S; José Leal; Zhe Shen; Han Fu; Wei Wang; Arvind Kannan; Doron Kukliansky; Sergey Yaroshenko; Svetlana Grant; Umesh Telang; David Wood; Alexandra Chronopoulou; Alexandru Ţifrea; Tao Zhou; Tony; Nguy\~ên; Muge Ersoy; Anima Singh; Meiyan Xie; Emanuel Taropa; Woohyun Han; Eirikur Agustsson; Andrei Sozanschi; Hui Peng; Alex Chen; Yoel Drori; Efren Robles; Yang Gao; Xerxes Dotiwalla; Ying Chen; Anudhyan Boral; Alexei Bendebury; John Nham; Chris Tar; Luis Castro; Jiepu Jiang; Canoee Liu; Felix Halim; Jinoo Baek; Andy Wan; Jeremiah Liu; Yuan Cao; Shengyang Dai; Trilok Acharya; Ruoxi Sun; Fuzhao Xue; Saket Joshi; Morgane Lustman; Yongqin Xian; Rishabh Joshi; Deep Karkhanis; Nora Kassner; Jamie Hall; Xiangzhuo Ding; Gan Song; Gang Li; Chen Zhu; Yana Kulizhskaya; Bin Ni; Alexey Vlaskin; Solomon Demmessie; Lucio Dery; Salah Zaiem; Yanping Huang; Cindy Fan; Felix Gimeno; Ananth Balashankar; Koji Kojima; Hagai Taitelbaum; Maya Meng; Dero Gharibian; Sahil Singla; Wei Chen; Ambrose Slone; Guanjie Chen; Sujee Rajayogam; Max Schumacher; Suyog Kotecha; Rory Blevins; Qifei Wang; Mor Hazan Taege; Alex Morris; Xin Liu; Fayaz Jamil; Richard Zhang; Pratik Joshi; Ben Ingram; Tyler Liechty; Ahmed Eleryan; Scott Baird; Alex Grills; Gagan Bansal; Shan Han; Kiran Yalasangi; Shawn Xu; Majd Al Merey; Isabel Gao; Felix Weissenberger; Igor Karpov; Robert Riachi; Ankit Anand; Gautam Prasad; Kay Lamerigts; Reid Hayes; Jamie Rogers; Mandy Guo; Ashish Shenoy; Qiong; Hu; Kyle He; Yuchen Liu; Polina Zablotskaia; Sagar Gubbi; Yifan Chang; Jay Pavagadhi; Kristian Kjems; Archita Vadali; Diego Machado; Yeqing Li; Renshen Wang; Dipankar Ghosh; Aahil Mehta; Dana Alon; George Polovets; Alessio Tonioni; Nate Kushman; Joel D'sa; Lin Zhuo; Allen Wu; Rohin Shah; John Youssef; Jiayu Ye; Justin Snyder; Karel Lenc; Senaka Buthpitiya; Matthew Tung; Jichuan Chang; Tao Chen; David Saxton; Jenny Lee; Lydia Lihui Zhang; James Qin; Prabakar Radhakrishnan; Maxwell Chen; Piotr Ambroszczyk; Metin Toksoz-Exley; Yan Zhong; Nitzan Katz; Brendan O'Donoghue; Tamara von Glehn; Adi Gerzi Rosenthal; Aga Świetlik; Xiaokai Zhao; Nick Fernando; Jinliang Wei; Jieru Mei; Sergei Vassilvitskii; Diego Cedillo; Pranjal Awasthi; Hui Zheng; Koray Kavukcuoglu; Itay Laish; Joseph Pagadora; Marc Brockschmidt; Christopher A. Choquette-Choo; Arunkumar Byravan; Yifeng Lu; Xu Chen; Mia Chen; Kenton Lee; Rama Pasumarthi; Sijal Bhatnagar; Aditya Shah; Qiyin Wu; Zhuoyuan Chen; Zack Nado; Bartek Perz; Zixuan Jiang; David Kao; Ganesh Mallya; Nino Vieillard; Lantao Mei; Sertan Girgin; Mandy Jordan; Yeongil Ko; Alekh Agarwal; Yaxin Liu; Yasemin Altun; Raoul de Liedekerke; Anastasios Kementsietsidis; Daiyi Peng; Dangyi Liu; Utku Evci; Peter Humphreys; Austin Tarango; Xiang Deng; Yoad Lewenberg; Kevin Aydin; Chengda Wu; Bhavishya Mittal; Tsendsuren Munkhdalai; Kleopatra Chatziprimou; Rodrigo Benenson; Uri First; Xiao Ma; Jinning Li; Armand Joulin; Hamish Tomlinson; Tingnan Zhang; Milad Nasr; Zhi Hong; Michaël Sander; Lisa Anne Hendricks; Anuj Sharma; Andrew Bolt; Eszter Vértes; Jiri Simsa; Tomer Levinboim; Olcan Sercinoglu; Divyansh Shukla; Austin Wu; Craig Swanson; Danny Vainstein; Fan Bu; Bo Wang; Ryan Julian; Charles Yoon; Sergei Lebedev; Antonious Girgis; Bernd Bandemer; David Du; Todd Wang; Xi Chen; Ying Xiao; Peggy Lu; Natalie Ha; Vlad Ionescu; Simon Rowe; Josip Matak; Federico Lebron; Andreas Steiner; Lalit Jain; Manaal Faruqui; Nicolas Lacasse; Georgie Evans; Neesha Subramaniam; Dean Reich; Giulia Vezzani; Aditya Pandey; Joe Stanton; Tianhao Zhou; Liam McCafferty; Henry Griffiths; Verena Rieser; Soheil Hassas Yeganeh; Eleftheria Briakou; Lu Huang; Zichuan Wei; Liangchen Luo; Erik Jue; Gabby Wang; Victor Cotruta; Myriam Khan; Jongbin Park; Qiuchen Guo; Peiran Li; Rong Rong; Diego Antognini; Anastasia Petrushkina; Chetan Tekur; Eli Collins; Parul Bhatia; Chester Kwak; Wenhu Chen; Arvind Neelakantan; Immanuel Odisho; Sheng Peng; Vincent Nallatamby; Vaibhav Tulsyan; Fabian Pedregosa; Peng Xu; Raymond Lin; Yulong Wang; Emma Wang; Sholto Douglas; Reut Tsarfaty; Elena Gribovskaya; Renga Aravamudhan; Manu Agarwal; Mara Finkelstein; Qiao Zhang; Elizabeth Cole; Phil Crone; Sarmishta Velury; Anil Das; Chris Sauer; Luyao Xu; Danfeng Qin; Chenjie Gu; Dror Marcus; CJ Zheng; Wouter Van Gansbeke; Sobhan Miryoosefi; Haitian Sun; YaGuang Li; Charlie Chen; Jae Yoo; Pavel Dubov; Alex Tomala; Adams Yu; Paweł Wesołowski; Alok Gunjan; Eddie Cao; Jiaming Luo; Nikhil Sethi; Arkadiusz Socala; Laura Graesser; Tomas Kocisky; Arturo BC; Minmin Chen; Edward Lee; Sophie Wang; Weize Kong; Qiantong Xu; Nilesh Tripuraneni; Yiming Li; Xinxin Yu; Allen Porter; Paul Voigtlaender; Biao Zhang; Arpi Vezer; Sarah York; Qing Wei; Geoffrey Cideron; Mark Kurzeja; Seungyeon Kim; Benny Li; Angéline Pouget; Hyo Lee; Kaspar Daugaard; Yang Li; Dave Uthus; Aditya Siddhant; Paul Cavallaro; Sriram Ganapathy; Maulik Shah; Rolf Jagerman; Jeff Stanway; Piermaria Mendolicchio; Li Xiao; Kayi Lee; Tara Thompson; Shubham Milind Phal; Jason Chase; Sun Jae Lee; Adrian N Reyes; Disha Shrivastava; Zhen Qin; Roykrong Sukkerd; Seth Odoom; Lior Madmoni; John Aslanides; Jonathan Herzig; Elena Pochernina; Sheng Zhang; Parker Barnes; Daisuke Ikeda; Qiujia Li; Shuo-yiin Chang; Shakir Mohamed; Jim Sproch; Richard Powell; Bidisha Samanta; Domagoj Ćevid; Anton Kovsharov; Shrestha Basu Mallick; Srinivas Tadepalli; Anne Zheng; Kareem Ayoub; Andreas Noever; Christian Reisswig; Zhuo Xu; Junhyuk Oh; Martin Matysiak; Tim Blyth; Shereen Ashraf; Julien Amelot; Boone Severson; Michele Bevilacqua; Motoki Sano; Ethan Dyer; Ofir Roval; Anu Sinha; Yin Zhong; Sagi Perel; Tea Sabolić; Johannes Mauerer; Willi Gierke; Mauro Verzetti; Rodrigo Cabrera; Alvin Abdagic; Steven Hemingray; Austin Stone; Jong Lee; Farooq Ahmad; Karthik Raman; Lior Shani; Jonathan Lai; Orhan Firat; Nathan Waters; Eric Ge; Mo Shomrat; Himanshu Gupta; Rajeev Aggarwal; Tom Hudson; Bill Jia; Simon Baumgartner; Palak Jain; Joe Kovac; Junehyuk Jung; Ante Žužul; Will Truong; Morteza Zadimoghaddam; Songyou Peng; Marco Liang; Rachel Sterneck; Balaji Lakshminarayanan; Machel Reid; Oliver Woodman; Tong Zhou; Jianling Wang; Vincent Coriou; Arjun Narayanan; Jay Hoover; Yenai Ma; Apoorv Jindal; Clayton Sanford; Doug Reid; Swaroop Ramaswamy; Alex Kurakin; Roland Zimmermann; Yana Lunts; Dragos Dena; Zalán Borsos; Vered Cohen; Shujian Zhang; Will Grathwohl; Robert Dadashi; Morgan Redshaw; Joshua Kessinger; Julian Odell; Silvano Bonacina; Zihang Dai; Grace Chen; Ayush Dubey; Pablo Sprechmann; Mantas Pajarskas; Wenxuan Zhou; Niharika Ahuja; Tara Thomas; Martin Nikoltchev; Matija Kecman; Bharath Mankalale; Andrey Ryabtsev; Jennifer She; Christian Walder; Jiaming Shen; Lu Li; Carolina Parada; Sheena Panthaplackel; Okwan Kwon; Matt Lawlor; Utsav Prabhu; Yannick Schroecker; Marc'aurelio Ranzato; Pete Blois; Iurii Kemaev; Ting Yu; Dmitry Lepikhin; Hao Xiong; Sahand Sharifzadeh; Oleaser Johnson; Jeremiah Willcock; Rui Yao; Greg Farquhar; Sujoy Basu; Hidetoshi Shimokawa; Nina Anderson; Haiguang Li; Khiem Pham; Yizhong Liang; Sebastian Borgeaud; Alexandre Moufarek; Hideto Kazawa; Blair Kutzman; Marcin Sieniek; Sara Smoot; Ruth Wang; Natalie Axelsson; Nova Fallen; Prasha Sundaram; Yuexiang Zhai; Varun Godbole; Petros Maniatis; Alek Wang; Ilia Shumailov; Santhosh Thangaraj; Remi Crocker; Nikita Gupta; Gang Wu; Phil Chen; Gellért Weisz; Celine Smith; Mojtaba Seyedhosseini; Boya Fang; Xiyang Luo; Roey Yogev; Zeynep Cankara; Andrew Hard; Helen Ran; Rahul Sukthankar; George Necula; Gaël Liu; Honglong Cai; Praseem Banzal; Daniel Keysers; Sanjay Ghemawat; Connie Tao; Emma Dunleavy; Aditi Chaudhary; Wei Li; Maciej Mikuła; Chen-Yu Lee; Tiziana Refice; Krishna Somandepalli; Alexandre Fréchette; Dan Bahir; John Karro; Keith Rush; Sarah Perrin; Bill Rosgen; Xiaomeng Yang; Clara Huiyi Hu; Mahmoud Alnahlawi; Justin Mao-Jones; Roopal Garg; Hoang Nguyen; Bat-Orgil Batsaikhan; Iñaki Iturrate; Anselm Levskaya; Avi Singh; Ashyana Kachra; Tony Lu; Denis Petek; Zheng Xu; Mark Graham; Lukas Zilka; Yael Karov; Marija Kostelac; Fangyu Liu; Yaohui Guo; Weiyue Wang; Bernd Bohnet; Emily Pitler; Tony Bruguier; Keisuke Kinoshita; Chrysovalantis Anastasiou; Nilpa Jha; Ting Liu; Jerome Connor; Phil Wallis; Philip Pham; Eric Bailey; Shixin Li; Heng-Tze Cheng; Sally Ma; Haiqiong Li; Akanksha Maurya; Kate Olszewska; Manfred Warmuth; Christy Koh; Dominik Paulus; Siddhartha Reddy Jonnalagadda; Enrique Piqueras; Ali Elqursh; Geoff Brown; Hadar Shemtov; Loren Maggiore; Fei Xia; Ryan Foley; Beka Westberg; George van den Driessche; Livio Baldini Soares; Arjun Kar; Michael Quinn; Siqi Zuo; Jialin Wu; Kyle Kastner; Anna Bortsova; Aijun Bai; Ales Mikhalap; Luowei Zhou; Jennifer Brennan; Vinay Ramasesh; Honglei Zhuang; John Maggs; Johan Schalkwyk; Yuntao Xu; Hui Huang; Andrew Howard; Sasha Brown; Linting Xue; Gloria Shen; Brian Albert; Neha Jha; Daniel Zheng; Varvara Krayvanova; Spurthi Amba Hombaiah; Olivier Lacombe; Gautam Vasudevan; Dan Graur; Tian Xie; Meet Gandhi; Bangju Wang; Dustin Zelle; Harman Singh; Dahun Kim; Sébastien Cevey; Victor Ungureanu; Natasha Noy; Fei Liu; Annie Xie; Fangxiaoyu Feng; Katerina Tsihlas; Daniel Formoso; Neera Vats; Quentin Wellens; Yinan Wang; Niket Kumar Bhumihar; Samrat Ghosh; Matt Hoffman; Tom Lieber; Oran Lang; Kush Bhatia; Tom Paine; Aroonalok Pyne; Ronny Votel; Madeleine Clare Elish; Benoit Schillings; Alex Panagopoulos; Haichuan Yang; Adam Raveret; Zohar Yahav; Shuang Liu; Dalia El Badawy; Nishant Agrawal; Mohammed Badawi; Mahdi Mirzazadeh; Carla Bromberg; Fan Ye; Chang Liu; Tatiana Sholokhova; George-Cristian Muraru; Gargi Balasubramaniam; Jonathan Malmaud; Alen Carin; Danilo Martins; Irina Jurenka; Pankil Botadra; Dave Lacey; Richa Singh; Mariano Schain; Dan Zheng; Isabelle Guyon; Victor Lavrenko; Seungji Lee; Xiang Zhou; Demis Hassabis; Jeshwanth Challagundla; Derek Cheng; Nikhil Mehta; Matthew Mauger; Michela Paganini; Pushkar Mishra; Kate Lee; Zhang Li; Lexi Baugher; Ondrej Skopek; Max Chang; Amir Zait; Gaurav Menghani; Lizzetth Bellot; Guangxing Han; Jean-Michel Sarr; Sharat Chikkerur; Himanshu Sahni; Rohan Anil; Arun Narayanan; Chandu Thekkath; Daniele Pighin; Hana Strejček; Marko Velic; Fred Bertsch; Manuel Tragut; Keran Rong; Alicia Parrish; Kai Bailey; Jiho Park; Isabela Albuquerque; Abhishek Bapna; Rajesh Venkataraman; Alec Kosik; Johannes Griesser; Zhiwei Deng; Alek Andreev; Qingyun Dou; Kevin Hui; Fanny Wei; Xiaobin Yu; Lei Shu; Avia Aharon; David Barker; Badih Ghazi; Sebastian Flennerhag; Chris Breaux; Yuchuan Liu; Matthew Bilotti; Josh Woodward; Uri Alon; Stephanie Winkler; Tzu-Kuo Huang; Kostas Andriopoulos; João Gabriel Oliveira; Penporn Koanantakool; Berkin Akin; Michael Wunder; Cicero Nogueira dos Santos; Mohammad Hossein Bateni; Lin Yang; Dan Horgan; Beer Changpinyo; Keyvan Amiri; Min Ma; Dayeong Lee; Lihao Liang; Anirudh Baddepudi; Tejasi Latkar; Raia Hadsell; Jun Xu; Hairong Mu; Michael Han; Aedan Pope; Snchit Grover; Frank Kim; Ankit Bhagatwala; Guan Sun; Yamini Bansal; Amir Globerson; Alireza Nazari; Samira Daruki; Hagen Soltau; Jane Labanowski; Laurent El Shafey; Matt Harvey; Yanif Ahmad; Elan Rosenfeld; William Kong; Etienne Pot; Yi-Xuan Tan; Aurora Wei; Victoria Langston; Marcel Prasetya; Petar Veličković; Richard Killam; Robin Strudel; Darren Ni; Zhenhai Zhu; Aaron Archer; Kavya Kopparapu; Lynn Nguyen; Emilio Parisotto; Hussain Masoom; Sravanti Addepalli; Jordan Grimstad; Hexiang Hu; Joss Moore; Avinatan Hassidim; Le Hou; Mukund Raghavachari; Jared Lichtarge; Adam R. Brown; Hilal Dib; Natalia Ponomareva; Justin Fu; Yujing Zhang; Altaf Rahman; Joana Iljazi; Edouard Leurent; Gabriel Dulac-Arnold; Cosmo Du; Chulayuth Asawaroengchai; Larry Jin; Ela Gruzewska; Ziwei Ji; Benigno Uria; Daniel De Freitas; Paul Barham; Lauren Beltrone; Víctor Campos; Jun Yan; Neel Kovelamudi; Arthur Nguyen; Elinor Davies; Zhichun Wu; Zoltan Egyed; Kristina Toutanova; Nithya Attaluri; Hongliang Fei; Peter Stys; Siddhartha Brahma; Martin Izzard; Siva Velusamy; Scott Lundberg; Vincent Zhuang; Kevin Sequeira; Adam Santoro; Ehsan Amid; Ophir Aharoni; Shuai Ye; Mukund Sundararajan; Lijun Yu; Yu-Cheng Ling; Stephen Spencer; Hugo Song; Josip Djolonga; Christo Kirov; Sonal Gupta; Alessandro Bissacco; Clemens Meyer; Mukul Bhutani; Andrew Dai; Weiyi Wang; Siqi Liu; Ashwin Sreevatsa; Qijun Tan; Maria Wang; Lucy Kim; Yicheng Wang; Alex Irpan; Yang Xiao; Stanislav Fort; Yifan He; Alex Gurney; Bryan Gale; Yue Ma; Monica Roy; Viorica Patraucean; Taylan Bilal; Golnaz Ghiasi; Anahita Hosseini; Melvin Johnson; Zhuowan Li; Yi Tay; Benjamin Beyret; Katie Millican; Josef Broder; Mayank Lunayach; Danny Swisher; Eugen Vušak; David Parkinson; MH Tessler; Adi Mayrav Gilady; Richard Song; Allan Dafoe; Yves Raimond; Masa Yamaguchi; Itay Karo; Elizabeth Nielsen; Kevin Kilgour; Mike Dusenberry; Rajiv Mathews; Jiho Choi; Siyuan Qiao; Harsh Mehta; Sahitya Potluri; Chris Knutsen; Jialu Liu; Tat Tan; Kuntal Sengupta; Keerthana Gopalakrishnan; Abodunrinwa Toki; Mencher Chiang; Mike Burrows; Grace Vesom; Zafarali Ahmed; Ilia Labzovsky; Siddharth Vashishtha; Preeti Singh; Ankur Sharma; Ada Ma; Jinyu Xie; Pranav Talluri; Hannah Forbes-Pollard; Aarush Selvan; Joel Wee; Loic Matthey; Tom Funkhouser; Parthasarathy Gopavarapu; Lev Proleev; Cheng Li; Matt Thomas; Kashyap Kolipaka; Zhipeng Jia; Ashwin Kakarla; Srinivas Sunkara; Joan Puigcerver; Suraj Satishkumar Sheth; Emily Graves; Chen Wang; Sadh MNM Khan; Kai Kang; Shyamal Buch; Fred Zhang; Omkar Savant; David Soergel; Kevin Lee; Linda Friso; Xuanyi Dong; Rahul Arya; Shreyas Chandrakaladharan; Connor Schenck; Greg Billock; Tejas Iyer; Anton Bakalov; Leslie Baker; Alex Ruiz; Angad Chandorkar; Trieu Trinh; Matt Miecnikowski; Yanqi Zhou; Yangsibo Huang; Jiazhong Nie; Ali Shah; Ashish Thapliyal; Sam Haves; Lun Wang; Uri Shaham; Patrick Morris-Suzuki; Soroush Radpour; Leonard Berrada; Thomas Strohmann; Chaochao Yan; Jingwei Shen; Sonam Goenka; Tris Warkentin; Petar Dević; Dan Belov; Albert Webson; Madhavi Yenugula; Puranjay Datta; Jerry Chang; Nimesh Ghelani; Aviral Kumar; Vincent Perot; Jessica Lo; Yang Song; Herman Schmit; Jianmin Chen; Vasilisa Bashlovkina; Xiaoyue Pan; Diana Mincu; Paul Roit; Isabel Edkins; Andy Davis; Yujia Li; Ben Horn; Xinjian Li; Pradeep Kumar S; Eric Doi; Wanzheng Zhu; Sri Gayatri Sundara Padmanabhan; Siddharth Verma; Jasmine Liu; Heng Chen; Mihajlo Velimirović; Malcolm Reynolds; Priyanka Agrawal; Nick Sukhanov; Abhinit Modi; Siddharth Goyal; John Palowitch; Nima Khajehnouri; Wing Lowe; David Klinghoffer; Sharon Silver; Vinh Tran; Candice Schumann; Francesco Piccinno; Xi Liu; Mario Lučić; Xiaochen Yang; Sandeep Kumar; Ajay Kannan; Ragha Kotikalapudi; Mudit Bansal; Fabian Fuchs; Mohammad Javad Hosseini; Abdelrahman Abdelhamed; Dawn Bloxwich; Tianhe Yu; Ruoxin Sang; Gregory Thornton; Karan Gill; Yuchi Liu; Virat Shejwalkar; Jason Lin; Zhipeng Yan; Kehang Han; Thomas Buschmann; Michael Pliskin; Zhi Xing; Susheel Tatineni; Junlin Zhang; Sissie Hsiao; Gavin Buttimore; Marcus Wu; Zefei Li; Geza Kovacs; Legg Yeung; Tao Huang; Aaron Cohen; Bethanie Brownfield; Averi Nowak; Mikel Rodriguez; Tianze Shi; Hado van Hasselt; Kevin Cen; Deepanway Ghoshal; Kushal Majmundar; Weiren Yu; Warren; Chen; Danila Sinopalnikov; Hao Zhang; Vlado Galić; Di Lu; Zeyu Zheng; Maggie Song; Gary Wang; Gui Citovsky; Swapnil Gawde; Isaac Galatzer-Levy; David Silver; Ivana Balazevic; Dipanjan Das; Kingshuk Majumder; Yale Cong; Praneet Dutta; Dustin Tran; Hui Wan; Junwei Yuan; Daniel Eppens; Alanna Walton; Been Kim; Harry Ragan; James Cobon-Kerr; Lu Liu; Weijun Wang; Bryce Petrini; Jack Rae; Rakesh Shivanna; Yan Xiong; Chace Lee; Pauline Coquinot; Yiming Gu; Lisa Patel; Blake Hechtman; Aviel Boag; Orion Jankowski; Alex Wertheim; Alex Lee; Paul Covington; Hila Noga; Sam Sobell; Shanthal Vasanth; William Bono; Chirag Nagpal; Wei Fan; Xavier Garcia; Kedar Soparkar; Aybuke Turker; Nathan Howard; Sachit Menon; Yuankai Chen; Vikas Verma; Vladimir Pchelin; Harish Rajamani; Valentin Dalibard; Ana Ramalho; Yang Guo; Kartikeya Badola; Seojin Bang; Nathalie Rauschmayr; Julia Proskurnia; Sudeep Dasari; Xinyun Chen; Mikhail Sushkov; Anja Hauth; Pauline Sho; Abhinav Singh; Bilva Chandra; Allie Culp; Max Dylla; Olivier Bachem; James Besley; Heri Zhao; Timothy Lillicrap; Wei Wei; Wael Al Jishi; Ning Niu; Alban Rrustemi; Raphaël Lopez Kaufman; Ryan Poplin; Jewel Zhao; Minh Truong; Shikhar Bharadwaj; Ester Hlavnova; Eli Stickgold; Cordelia Schmid; Georgi Stephanov; Zhaoqi Leng; Frederick Liu; Léonard Hussenot; Shenil Dodhia; Juliana Vicente Franco; Lesley Katzen; Abhanshu Sharma; Sarah Cogan; Zuguang Yang; Aniket Ray; Sergi Caelles; Shen Yan; Ravin Kumar; Daniel Gillick; Renee Wong; Joshua Ainslie; Jonathan Hoech; Séb Arnold; Dan Abolafia; Anca Dragan; Ben Hora; Grace Hu; Alexey Guseynov; Yang Lu; Chas Leichner; Jinmeng Rao; Abhimanyu Goyal; Nagabhushan Baddi; Daniel Hernandez Diaz; Tim McConnell; Max Bain; Jake Abernethy; Qiqi Yan; Rylan Schaeffer; Paul Vicol; Will Thompson; Montse Gonzalez Arenas; Mathias Bellaiche; Pablo Barrio; Stefan Zinke; Riccardo Patana; Pulkit Mehta; JK Kearns; Avraham Ruderman; Scott Pollom; David D'Ambrosio; Cath Hope; Yang Yu; Andrea Gesmundo; Kuang-Huei Lee; Aviv Rosenberg; Yiqian Zhou; Yaoyiran Li; Drew Garmon; Yonghui Wu; Safeen Huda; Gil Fidel; Martin Baeuml; Jian Li; Phoebe Kirk; Rhys May; Tao Tu; Sara Mc Carthy; Toshiyuki Fukuzawa; Miranda Aperghis; Chih-Kuan Yeh; Toshihiro Yoshino; Bo Li; Austin Myers; Kaisheng Yao; Ben Limonchik; Changwan Ryu; Rohun Saxena; Alex Goldin; Ruizhe Zhao; Rocky Rhodes; Tao Zhu; Divya Tyam; Heidi Howard; Nathan Byrd; Hongxu Ma; Yan Wu; Ryan Mullins; Qingze Wang; Aida Amini; Sebastien Baur; Yiran Mao; Subhashini Venugopalan; Will Song; Wen Ding; Paul Collins; Sashank Reddi; Megan Shum; Andrei Rusu; Luisa Zintgraf; Kelvin Chan; Sheela Goenka; Mathieu Blondel; Michael Collins; Renke Pan; Marissa Giustina; Nikolai Chinaev; Christian Schuler; Ce Zheng; Jonas Valfridsson; Alyssa Loo; Alex Yakubovich; Jamie Smith; Tao Jiang; Rich Munoz; Gabriel Barcik; Rishabh Bansal; Mingyao Yang; Yilun Du; Pablo Duque; Mary Phuong; Alexandra Belias; Kunal Lad; Zeyu Liu; Tal Schuster; Karthik Duddu; Jieru Hu; Paige Kunkle; Matthew Watson; Jackson Tolins; Josh Smith; Denis Teplyashin; Garrett Bingham; Marvin Ritter; Marco Andreetto; Divya Pitta; Mohak Patel; Shashank Viswanadha; Trevor Strohman; Catalin Ionescu; Jincheng Luo; Yogesh Kalley; Jeremy Wiesner; Dan Deutsch; Derek Lockhart; Peter Choy; Rumen Dangovski; Chawin Sitawarin; Cat Graves; Tanya Lando; Joost van Amersfoort; Ndidi Elue; Zhouyuan Huo; Pooya Moradi; Jean Tarbouriech; Henryk Michalewski; Wenting Ye; Eunyoung Kim; Alex Druinsky; Florent Altché; Xinyi Chen; Artur Dwornik; Da-Cheng Juan; Rivka Moroshko; Horia Toma; Jarrod Kahn; Hai Qian; Maximilian Sieb; Irene Cai; Roman Goldenberg; Praneeth Netrapalli; Sindhu Raghuram; Yuan Gong; Lijie Fan; Evan Palmer; Yossi Matias; Valentin Gabeur; Shreya Pathak; Tom Ouyang; Don Metzler; Geoff Bacon; Srinivasan Venkatachary; Sridhar Thiagarajan; Alex Cullum; Eran Ofek; Vytenis Sakenas; Mohamed Hammad; Cesar Magalhaes; Mayank Daswani; Oscar Chang; Ashok Popat; Ruichao Li; Komal Jalan; Yanhan Hou; Josh Lipschultz; Antoine He; Wenhao Jia; Pier Giuseppe Sessa; Prateek Kolhar; William Wong; Sumeet Singh; Lukas Haas; Jay Whang; Hanna Klimczak-Plucińska; Georges Rotival; Grace Chung; Yiqing Hua; Anfal Siddiqui; Nicolas Serrano; Dongkai Chen; Billy Porter; Libin Bai; Keshav Shivam; Sho Arora; Partha Talukdar; Tom Cobley; Sangnie Bhardwaj; Evgeny Gladchenko; Simon Green; Kelvin Guu; Felix Fischer; Xiao Wu; Eric Wang; Achintya Singhal; Tatiana Matejovicova; James Martens; Hongji Li; Roma Patel; Elizabeth Kemp; Jiaqi Pan; Lily Wang; Blake JianHang Chen; Jean-Baptiste Alayrac; Navneet Potti; Erika Gemzer; Eugene Ie; Kay McKinney; Takaaki Saeki; Edward Chou; Pascal Lamblin; SQ Mah; Zach Fisher; Martin Chadwick; Jon Stritar; Obaid Sarvana; Andrew Hogue; Artem Shtefan; Hadi Hashemi; Yang Xu; Jindong Gu; Sharad Vikram; Chung-Ching Chang; Sabela Ramos; Logan Kilpatrick; Weijuan Xi; Jenny Brennan; Yinghao Sun; Abhishek Jindal; Ionel Gog; Dawn Chen; Felix Wu; Jason Lee; Sudhindra Kopalle; Srinadh Bhojanapalli; Oriol Vinyals; Natan Potikha; Burcu Karagol Ayan; Yuan Yuan; Michael Riley; Piotr Stanczyk; Sergey Kishchenko; Bing Wang; Dan Garrette; Antoine Yang; Vlad Feinberg; CJ Carey; Javad Azizi; Viral Shah; Erica Moreira; Chongyang Shi; Josh Feldman; Elizabeth Salesky; Thomas Lampe; Aneesh Pappu; Duhyeon Kim; Jonas Adler; Avi Caciularu; Brian Walker; Yunhan Xu; Yochai Blau; Dylan Scandinaro; Terry Huang; Sam El-Husseini; Abhishek Sinha; Lijie Ren; Taylor Tobin; Patrik Sundberg; Tim Sohn; Vikas Yadav; Mimi Ly; Emily Xue; Jing Xiong; Afzal Shama Soudagar; Sneha Mondal; Nikhil Khadke; Qingchun Ren; Ben Vargas; Stan Bileschi; Sarah Chakera; Cindy Wang; Boyu Wang; Yoni Halpern; Joe Jiang; Vikas Sindhwani; Petre Petrov; Pranavaraj Ponnuramu; Sanket Vaibhav Mehta; Yu Watanabe; Betty Chan; Matheus Wisniewski; Trang Pham; Jingwei Zhang; Conglong Li; Dario de Cesare; Art Khurshudov; Alex Vasiloff; Melissa Tan; Zoe Ashwood; Bobak Shahriari; Maryam Majzoubi; Garrett Tanzer; Olga Kozlova; Robin Alazard; James Lee-Thorp; Nguyet Minh Phu; Isaac Tian; Junwhan Ahn; Andy Crawford; Lauren Lax; Yuan Shangguan; Iftekhar Naim; David Ross; Oleksandr Ferludin; Tongfei Guo; Andrea Banino; Hubert Soyer; Xiaoen Ju; Dominika Rogozińska; Ishaan Malhi; Marcella Valentine; Daniel Balle; Apoorv Kulshreshtha; Maciej Kula; Yiwen Song; Sophia Austin; John Schultz; Roy Hirsch; Arthur Douillard; Apoorv Reddy; Michael Fink; Summer Yue; Khyatti Gupta; Adam Zhang; Norman Rink; Daniel McDuff; Lei Meng; András György; Yasaman Razeghi; Ricky Liang; Kazuki Osawa; Aviel Atias; Matan Eyal; Tyrone Hill; Nikolai Grigorev; Zhengdong Wang; Nitish Kulkarni; Rachel Soh; Ivan Lobov; Zachary Charles; Sid Lall; Kazuma Hashimoto; Ido Kessler; Victor Gomes; Zelda Mariet; Danny Driess; Alessandro Agostini; Canfer Akbulut; Jingcao Hu; Marissa Ikonomidis; Emily Caveness; Kartik Audhkhasi; Saurabh Agrawal; Ioana Bica; Evan Senter; Jayaram Mudigonda; Kelly Chen; Jingchen Ye; Xuanhui Wang; James Svensson; Philipp Fränken; Josh Newlan; Li Lao; Eva Schnider; Sami Alabed; Joseph Kready; Jesse Emond; Afief Halumi; Tim Zaman; Chengxi Ye; Naina Raisinghani; Vilobh Meshram; Bo Chang; Ankit Singh Rawat; Axel Stjerngren; Sergey Levi; Rui Wang; Xiangzhu Long; Mitchelle Rasquinha; Steven Hand; Aditi Mavalankar; Lauren Agubuzu; Sudeshna Roy; Junquan Chen; Jarek Wilkiewicz; Hao Zhou; Michal Jastrzebski; Qiong Hu; Agustin Dal Lago; Ramya Sree Boppana; Wei-Jen Ko; Jennifer Prendki; Yao Su; Zhi Li; Eliza Rutherford; Girish Ramchandra Rao; Ramona Comanescu; Adrià Puigdomènech; Qihang Chen; Dessie Petrova; Christine Chan; Vedrana Milutinovic; Felipe Tiengo Ferreira; Chin-Yi Cheng; Ming Zhang; Tapomay Dey; Sherry Yang; Ramesh Sampath; Quoc Le; Howard Zhou; Chu-Cheng Lin; Hoi Lam; Christine Kaeser-Chen; Kai Hui; Dean Hirsch; Tom Eccles; Basil Mustafa; Shruti Rijhwani; Morgane Rivière; Yuanzhong Xu; Junjie Wang; Xinyang Geng; Xiance Si; Arjun Khare; Cheolmin Kim; Vahab Mirrokni; Kamyu Lee; Khuslen Baatarsukh; Nathaniel Braun; Lisa Wang; Pallavi LV; Richard Tanburn; Yuvein; Zhu; Fangda Li; Setareh Ariafar; Dan Goldberg; Ken Burke; Daniil Mirylenka; Meiqi Guo; Olaf Ronneberger; Hadas Natalie Vogel; Liqun Cheng; Nishita Shetty; Johnson Jia; Thomas Jimma; Corey Fry; Ted Xiao; Martin Sundermeyer; Ryan Burnell; Yannis Assael; Mario Pinto; JD Chen; Rohit Sathyanarayana; Donghyun Cho; Jing Lu; Rishabh Agarwal; Sugato Basu; Lucas Gonzalez; Dhruv Shah; Meng Wei; Dre Mahaarachchi; Rohan Agrawal; Tero Rissa; Yani Donchev; Ramiro Leal-Cavazos; Adrian Hutter; Markus Mircea; Alon Jacovi; Faruk Ahmed; Jiageng Zhang; Shuguang Hu; Bo-Juen Chen; Jonni Kanerva; Guillaume Desjardins; Andrew Lee; Nikos Parotsidis; Asier Mujika; Tobias Weyand; Jasper Snoek; Jo Chick; Kai Chen; Paul Chang; Ethan Mahintorabi; Zi Wang; Tolly Powell; Orgad Keller; Abhirut Gupta; Claire Sha; Kanav Garg; Nicolas Heess; Ágoston Weisz; Cassidy Hardin; Bartek Wydrowski; Ben Coleman; Karina Zainullina; Pankaj Joshi; Alessandro Epasto; Terry Spitz; Binbin Xiong; Kai Zhao; Arseniy Klimovskiy; Ivy Zheng; Johan Ferret; Itay Yona; Waleed Khawaja; Jean-Baptiste Lespiau; Maxim Krikun; Siamak Shakeri; Timothee Cour; Bonnie Li; Igor Krivokon; Dan Suh; Alex Hofer; Jad Al Abdallah; Nikita Putikhin; Oscar Akerlund; Silvio Lattanzi; Anurag Kumar; Shane Settle; Himanshu Srivastava; Folawiyo Campbell-Ajala; Edouard Rosseel; Mihai Dorin Istin; Nishanth Dikkala; Anand Rao; Nick Young; Kate Lin; Dhruva Bhaswar; Yiming Wang; Jaume Sanchez Elias; Kritika Muralidharan; James Keeling; Dayou Du; Siddharth Gopal; Gregory Dibb; Charles Blundell; Manolis Delakis; Jacky Liang; Marco Tulio Ribeiro; Georgi Karadzhov; Guillermo Garrido; Ankur Bapna; Jiawei Cao; Adam Sadovsky; Pouya Tafti; Arthur Guez; Coline Devin; Yixian Di; Jinwei Xing; Chuqiao; Xu; Hanzhao Lin; Chun-Te Chu; Sameera Ponda; Wesley Helmholz; Fan Yang; Yue Gao; Sara Javanmardi; Wael Farhan; Alex Ramirez; Ricardo Figueira; Khe Chai Sim; Yuval Bahat; Ashwin Vaswani; Liangzhe Yuan; Gufeng Zhang; Leland Rechis; Hanjun Dai; Tayo Oguntebi; Alexandra Cordell; Eugénie Rives; Kaan Tekelioglu; Naveen Kumar; Bing Zhang; Aurick Zhou; Nikolay Savinov; Andrew Leach; Alex Tudor; Sanjay Ganapathy; Yanyan Zheng; Mirko Rossini; Vera Axelrod; Arnaud Autef; Yukun Zhu; Zheng Zheng; Mingda Zhang; Baochen Sun; Jie Ren; Nenad Tomasev; Nithish Kannen; Amer Sinha; Charles Chen; Louis O'Bryan; Alex Pak; Aditya Kusupati; Weel Yang; Deepak Ramachandran; Patrick Griffin; Seokhwan Kim; Philipp Neubeck; Craig Schiff; Tammo Spalink; Mingyang Ling; Arun Nair; Ga-Young Joung; Linda Deng; Avishkar Bhoopchand; Lora Aroyo; Tom Duerig; Jordan Griffith; Gabe Barth-Maron; Jake Ades; Alex Haig; Ankur Taly; Yunting Song; Paul Michel; Dave Orr; Dean Weesner; Corentin Tallec; Carrie Grimes Bostock; Paul Niemczyk; Andy Twigg; Mudit Verma; Rohith Vallu; Henry Wang; Marco Gelmi; Kiranbir Sodhia; Aleksandr Chuklin; Omer Goldman; Jasmine George; Liang Bai; Kelvin Zhang; Petar Sirkovic; Efrat Nehoran; Golan Pundak; Jiaqi Mu; Alice Chen; Alex Greve; Paulo Zacchello; David Amos; Heming Ge; Eric Noland; Colton Bishop; Jeffrey Dudek; Youhei Namiki; Elena Buchatskaya; Jing Li; Dorsa Sadigh; Masha Samsikova; Dan Malkin; Damien Vincent; Robert David; Rob Willoughby; Phoenix Meadowlark; Shawn Gao; Yan Li; Raj Apte; Amit Jhindal; Stein Xudong Lin; Alex Polozov; Zhicheng Wang; Tomas Mery; Anirudh GP; Varun Yerram; Sage Stevens; Tianqi Liu; Noah Fiedel; Charles Sutton; Matthew Johnson; Xiaodan Song; Kate Baumli; Nir Shabat; Muqthar Mohammad; Hao Liu; Marco Selvi; Yichao Zhou; Mehdi Hafezi Manshadi; Chu-ling Ko; Anthony Chen; Michael Bendersky; Jorge Gonzalez Mendez; Nisarg Kothari; Amir Zandieh; Yiling Huang; Daniel Andor; Ellie Pavlick; Idan Brusilovsky; Jitendra Harlalka; Sally Goldman; Andrew Lampinen; Guowang Li; Asahi Ushio; Somit Gupta; Lei Zhang; Chuyuan Kelly Fu; Madhavi Sewak; Timo Denk; Jed Borovik; Brendan Jou; Avital Zipori; Prateek Jain; Junwen Bai; Thang Luong; Jonathan Tompson; Alice Li; Li Liu; George Powell; Jiajun Shen; Alex Feng; Grishma Chole; Da Yu; Yinlam Chow; Tongxin Yin; Eric Malmi; Kefan Xiao; Yash Pande; Shachi Paul; Niccolò Dal Santo; Adil Dostmohamed; Sergio Guadarrama; Aaron Phillips; Thanumalayan Sankaranarayana Pillai; Gal Yona; Amin Ghafouri; Preethi Lahoti; Benjamin Lee; Dhruv Madeka; Eren Sezener; Simon Tokumine; Adrian Collister; Nicola De Cao; Richard Shin; Uday Kalra; Parker Beak; Emily Nottage; Ryo Nakashima; Ivan Jurin; Vikash Sehwag; Meenu Gaba; Junhao Zeng; Kevin R. McKee; Fernando Pereira; Tamar Yakar; Amayika Panda; Arka Dhar; Peilin Zhong; Daniel Sohn; Mark Brand; Lars Lowe Sjoesund; Viral Carpenter; Sharon Lin; Shantanu Thakoor; Marcus Wainwright; Ashwin Chaugule; Pranesh Srinivasan; Muye Zhu; Bernett Orlando; Jack Weber; Ayzaan Wahid; Gilles Baechler; Apurv Suman; Jovana Mitrović; Gabe Taubman; Honglin Yu; Helen King; Josh Dillon; Cathy Yip; Dhriti Varma; Tomas Izo; Levent Bolelli; Borja De Balle Pigem; Julia Di Trapani; Fotis Iliopoulos; Adam Paszke; Nishant Ranka; Joe Zou; Francesco Pongetti; Jed McGiffin; Alex Siegman; Rich Galt; Ross Hemsley; Goran Žužić; Victor Carbune; Tao Li; Myle Ott; Félix de Chaumont Quitry; David Vilar Torres; Yuri Chervonyi; Tomy Tsai; Prem Eruvbetine; Samuel Yang; Matthew Denton; Jake Walker; Slavica Andačić; Idan Heimlich Shtacher; Vittal Premachandran; Harshal Tushar Lehri; Cip Baetu; Damion Yates; Lampros Lamprou; Mariko Iinuma; Ioana Mihailescu; Ben Albrecht; Shachi Dave; Susie Sargsyan; Bryan Perozzi; Lucas Manning; Chiyuan Zhang; Denis Vnukov; Igor Mordatch; Raia Hadsell Wolfgang Macherey; Ryan Kappedal; Jim Stephan; Aditya Tripathi; Klaus Macherey; Jun Qian; Abhishek Bhowmick; Shekoofeh Azizi; Rémi Leblond; Shiva Mohan Reddy Garlapati; Timothy Knight; Matthew Wiethoff; Wei-Chih Hung; Anelia Angelova; Georgios Evangelopoulos; Pawel Janus; Dimitris Paparas; Matthew Rahtz; Ken Caluwaerts; Vivek Sampathkumar; Daniel Jarrett; Shadi Noghabi; Antoine Miech; Chak Yeung; Geoff Clark; Henry Prior; Fei Zheng; Jean Pouget-Abadie; Indro Bhattacharya; Kalpesh Krishna; Will Bishop; Zhe Yuan; Yunxiao Deng; Ashutosh Sathe; Kacper Krasowiak; Ciprian Chelba; Cho-Jui Hsieh; Kiran Vodrahalli; Buhuang Liu; Thomas Köppe; Amr Khalifa; Lubo Litchev; Pichi Charoenpanit; Reed Roberts; Sachin Yadav; Yasumasa Onoe; Desi Ivanov; Megha Mohabey; Vighnesh Birodkar; Nemanja Rakićević; Pierre Sermanet; Vaibhav Mehta; Krishan Subudhi; Travis Choma; Will Ng; Luheng He; Kathie Wang; Tasos Kementsietsidis; Shane Gu; Mansi Gupta; Andrew Nystrom; Mehran Kazemi; Timothy Chung; Nacho Cano; Nikhil Dhawan; Yufei Wang; Jiawei Xia; Trevor Yacovone; Eric Jia; Mingqing Chen; Simeon Ivanov; Ashrith Sheshan; Sid Dalmia; Paweł Stradomski; Pengcheng Yin; Salem Haykal; Congchao Wang; Dennis Duan; Neslihan Bulut; Greg Kochanski; Liam MacDermed; Namrata Godbole; Shitao Weng; Jingjing Chen; Rachana Fellinger; Ramin Mehran; Daniel Suo; Hisham Husain; Tong He; Kaushal Patel; Joshua Howland; Randall Parker; Kelvin Nguyen; Sharath Maddineni; Chris Rawles; Mina Khan; Shlomi Cohen-Ganor; Amol Mandhane; Xinyi Wu; Chenkai Kuang; Iulia Comşa; Ramya Ganeshan; Hanie Sedghi; Adam Bloniarz; Nuo Wang Pierse; Anton Briukhov; Petr Mitrichev; Anita Gergely; Serena Zhan; Allan Zhou; Nikita Saxena; Eva Lu; Josef Dean; Ashish Gupta; Nicolas Perez-Nieves; Renjie Wu; Cory McLean; Wei Liang; Disha Jindal; Anton Tsitsulin; Wenhao Yu; Kaiz Alarakyia; Tom Schaul; Piyush Patil; Peter Sung; Elijah Peake; Hongkun Yu; Feryal Behbahani; JD Co-Reyes; Alan Ansell; Sean Sun; Clara Barbu; Jonathan Lee; Seb Noury; James Allingham; Bilal Piot; Mohit Sharma; Christopher Yew; Ivan Korotkov; Bibo Xu; Demetra Brady; Goran Petrovic; Shibl Mourad; Claire Cui; Aditya Gupta; Parker Schuh; Saarthak Khanna; Anna Goldie; Abhinav Arora; Vadim Zubov; Amy Stuart; Mark Epstein; Yun Zhu; Jianqiao Liu; Yury Stuken; Ziyue Wang; Karolis Misiunas; Dee Guo; Ashleah Gill; Ale Hartman; Zaid Nabulsi; Aurko Roy; Aleksandra Faust; Jason Riesa; Ben Withbroe; Mengchao Wang; Marco Tagliasacchi; Andreea Marzoca; James Noraky; Serge Toropov; Malika Mehrotra; Bahram Raad; Sanja Deur; Steve Xu; Marianne Monteiro; Zhongru Wu; Yi Luan; Sam Ritter; Nick Li; Håvard Garnes; Yanzhang He; Martin Zlocha; Jifan Zhu; Matteo Hessel; Will Wu; Spandana Raj Babbula; Chizu Kawamoto; Yuanzhen Li; Mehadi Hassen; Yan Wang; Brian Wieder; James Freedman; Yin Zhang; Xinyi Bai; Tianli Yu; David Reitter; XiangHai Sheng; Mateo Wirth; Aditya Kini; Dima Damen; Mingcen Gao; Rachel Hornung; Michael Voznesensky; Brian Roark; Adhi Kuncoro; Yuxiang Zhou; Rushin Shah; Anthony Brohan; Kuangyuan Chen; James Wendt; David Rim; Paul Kishan Rubenstein; Jonathan Halcrow; Michelle Liu; Ty Geri; Yunhsuan Sung; Jane Shapiro; Shaan Bijwadia; Chris Duvarney; Christina Sorokin; Paul Natsev; Reeve Ingle; Pramod Gupta; Young Maeng; Ndaba Ndebele; Kexin Zhu; Valentin Anklin; Katherine Lee; Yuan Liu; Yaroslav Akulov; Shaleen Gupta; Guolong Su; Flavien Prost; Tianlin Liu; Vitaly Kovalev; Pol Moreno; Martin Scholz; Sam Redmond; Zongwei Zhou; Alex Castro-Ros; André Susano Pinto; Dia Kharrat; Michal Yarom; Rachel Saputro; Jannis Bulian; Ben Caine; Ji Liu; Abbas Abdolmaleki; Shariq Iqbal; Tautvydas Misiunas; Mikhail Sirotenko; Shefali Garg; Guy Bensky; Huan Gui; Xuezhi Wang; Raphael Koster; Mike Bernico; Da Huang; Romal Thoppilan; Trevor Cohn; Ben Golan; Wenlei Zhou; Andrew Rosenberg; Markus Freitag; Tynan Gangwani; Vincent Tsang; Anand Shukla; Xiaoqi Ren; Minh Giang; Chi Zou; Andre Elisseeff; Charline Le Lan; Dheeru Dua; Shuba Lall; Pranav Shyam; Frankie Garcia; Sarah Nguyen; Michael Guzman; AJ Maschinot; Marcello Maggioni; Ming-Wei Chang; Karol Gregor; Lotte Weerts; Kumaran Venkatesan; Bogdan Damoc; Leon Liu; Jan Wassenberg; Lewis Ho; Becca Roelofs; Majid Hadian; François-Xavier Aubet; Yu Liang; Sami Lachgar; Danny Karmon; Yong Cheng; Amelio Vázquez-Reina; Angie Chen; Zhuyun Dai; Andy Brock; Shubham Agrawal; Chenxi Pang; Peter Garst; Mariella Sanchez-Vargas; Ivor Rendulic; Aditya Ayyar; Andrija Ražnatović; Olivia Ma; Roopali Vij; Neha Sharma; Ashwin Balakrishna; Bingyuan Liu; Ian Mackinnon; Sorin Baltateanu; Petra Poklukar; Gabriel Ibagon; Colin Ji; Hongyang Jiao; Isaac Noble; Wojciech Stokowiec; Zhihao Li; Jeff Dean; David Lindner; Mark Omernick; Kristen Chiafullo; Mason Dimarco; Vitor Rodrigues; Vittorio Selo; Garrett Honke; Xintian; Wu; Wei He; Adam Hillier; Anhad Mohananey; Vihari Piratla; Chang Ye; Chase Malik; Sebastian Riedel; Samuel Albanie; Zi Yang; Kenny Vassigh; Maria Bauza; Sheng Li; Yiqing Tao; Nevan Wichers; Andrii Maksai; Abe Ittycheriah; Ross Mcilroy; Bryan Seybold; Noah Goodman; Romina Datta; Steven M. Hernandez; Tian Shi; Yony Kochinski; Anna Bulanova; Ken Franko; Mikita Sazanovich; Nicholas FitzGerald; Praneeth Kacham; Shubha Srinivas Raghvendra; Vincent Hellendoorn; Alexander Grushetsky; Julian Salazar; Angeliki Lazaridou; Jason Chang; Jan-Thorsten Peter; Sushant Kafle; Yann Dauphin; Abhishek Rao; Filippo Graziano; Izhak Shafran; Yuguo Liao; Tianli Ding; Geng Yan; Grace Chu; Zhao Fu; Vincent Roulet; Gabriel Rasskin; Duncan Williams; Shahar Drath; Alex Mossin; Raphael Hoffmann; Jordi Orbay; Francesco Bertolini; Hila Sheftel; Justin Chiu; Siyang Xue; Yuheng Kuang; Ferjad Naeem; Swaroop Nath; Nana Nti; Phil Culliton; Kashyap Krishnakumar; Michael Isard; Pei Sun; Ayan Chakrabarti; Nathan Clement; Regev Cohen; Arissa Wongpanich; GS Oh; Ashwin Murthy; Hao Zheng; Jessica Hamrick; Oskar Bunyan; Suhas Ganesh; Nitish Gupta; Roy Frostig; John Wieting; Yury Malkov; Pierre Marcenac; Zhixin; Lai; Xiaodan Tang; Mohammad Saleh; Fedir Zubach; Chinmay Kulkarni; Huanjie Zhou; Vicky Zayats; Nan Ding; Anshuman Tripathi; Arijit Pramanik; Patrik Zochbauer; Harish Ganapathy; Vedant Misra; Zach Behrman; Hugo Vallet; Mingyang Zhang; Mukund Sridhar; Ye Jin; Mohammad Babaeizadeh; Siim Põder; Megha Goel; Divya Jain; Tajwar Nasir; Shubham Mittal; Tim Dozat; Diego Ardila; Aliaksei Severyn; Fabio Pardo; Sammy Jerome; Siyang Qin; Louis Rouillard; Amir Yazdanbakhsh; Zizhao Zhang; Shivani Agrawal; Kaushik Shivakumar; Caden Lu; Praveen Kallakuri; Rachita Chhaparia; Kanishka Rao; Charles Kwong; Asya Fadeeva; Shitij Nigam; Yan Virin; Yuan Zhang; Balaji Venkatraman; Beliz Gunel; Marc Wilson; Huiyu Wang; Abhinav Gupta; Xiaowei Xu; Adrien Ali Taïga; Kareem Mohamed; Doug Fritz; Daniel Rodriguez; Zoubin Ghahramani; Harry Askham; Lior Belenki; James Zhao; Rahul Gupta; Krzysztof Jastrzębski; Takahiro Kosakai; Kaan Katircioglu; Jon Schneider; Rina Panigrahy; Konstantinos Bousmalis; Peter Grabowski; Prajit Ramachandran; Chaitra Hegde; Mihaela Rosca; Angelo Scorza Scarpati; Kyriakos Axiotis; Ying Xu; Zach Gleicher; Assaf Hurwitz Michaely; Mandar Sharma; Sanil Jain; Christoph Hirnschall; Tal Marian; Xuhui Jia; Kevin Mather; Kilol Gupta; Linhai Qiu; Nigamaa Nayakanti; Lucian Ionita; Steven Zheng; Lucia Loher; Kurt Shuster; Igor Petrovski; Roshan Sharma; Rahma Chaabouni; Angel Yeh; James An; Arushi Gupta; Steven Schwarcz; Seher Ellis; Sam Conway-Rahman; Javier Snaider; Alex Zhai; James Atwood; Daniel Golovin; Liqian Peng; Te I; Vivian Xia; Salvatore Scellato; Mahan Malihi; Arthur Bražinskas; Vlad-Doru Ion; Younghoon Jun; James Swirhun; Soroosh Mariooryad; Jiao Sun; Steve Chien; Rey Coaguila; Ariel Brand; Yi Gao; Tom Kwiatkowski; Roee Aharoni; Cheng-Chun Lee; Mislav Žanić; Yichi Zhang; Dan Ethier; Vitaly Nikolaev; Pranav Nair; Yoav Ben Shalom; Hen Fitoussi; Jai Gupta; Hongbin Liu; Dee Cattle; Tolga Bolukbasi; Ben Murdoch; Fantine Huot; Yin Li; Chris Hahn; Urvashi Khandelwal; Frederik Benzing; Arthur Conmy; Andrey Simanovsky; Françoise Beaufays; Eugene Weinstein; Tongzhou Chen; Luke Leonhard; Bhuvana Ramabhadran
>
> **备注:** 72 pages, 17 figures
>
> **摘要:** In this report, we introduce the Gemini 2.X model family: Gemini 2.5 Pro and Gemini 2.5 Flash, as well as our earlier Gemini 2.0 Flash and Flash-Lite models. Gemini 2.5 Pro is our most capable model yet, achieving SoTA performance on frontier coding and reasoning benchmarks. In addition to its incredible coding and reasoning skills, Gemini 2.5 Pro is a thinking model that excels at multimodal understanding and it is now able to process up to 3 hours of video content. Its unique combination of long context, multimodal and reasoning capabilities can be combined to unlock new agentic workflows. Gemini 2.5 Flash provides excellent reasoning abilities at a fraction of the compute and latency requirements and Gemini 2.0 Flash and Flash-Lite provide high performance at low latency and cost. Taken together, the Gemini 2.X model generation spans the full Pareto frontier of model capability vs cost, allowing users to explore the boundaries of what is possible with complex agentic problem solving.
>
---
#### [replaced 031] Fine-Tune an SLM or Prompt an LLM? The Case of Generating Low-Code Workflows
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24189v2](http://arxiv.org/pdf/2505.24189v2)**

> **作者:** Orlando Marquez Ayala; Patrice Bechard; Emily Chen; Maggie Baird; Jingfei Chen
>
> **备注:** 8 pages, 7 figures. Accepted to Workshop on Structured Knowledge for Large Language Models (SKnowLLM) at KDD 2025
>
> **摘要:** Large Language Models (LLMs) such as GPT-4o can handle a wide range of complex tasks with the right prompt. As per token costs are reduced, the advantages of fine-tuning Small Language Models (SLMs) for real-world applications -- faster inference, lower costs -- may no longer be clear. In this work, we present evidence that, for domain-specific tasks that require structured outputs, SLMs still have a quality advantage. We compare fine-tuning an SLM against prompting LLMs on the task of generating low-code workflows in JSON form. We observe that while a good prompt can yield reasonable results, fine-tuning improves quality by 10% on average. We also perform systematic error analysis to reveal model limitations.
>
---
#### [replaced 032] ConTextual: Improving Clinical Text Summarization in LLMs with Context-preserving Token Filtering and Knowledge Graphs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.16394v3](http://arxiv.org/pdf/2504.16394v3)**

> **作者:** Fahmida Liza Piya; Rahmatollah Beheshti
>
> **备注:** Accepted for MLHC 2025
>
> **摘要:** Unstructured clinical data can serve as a unique and rich source of information that can meaningfully inform clinical practice. Extracting the most pertinent context from such data is critical for exploiting its true potential toward optimal and timely decision-making in patient care. While prior research has explored various methods for clinical text summarization, most prior studies either process all input tokens uniformly or rely on heuristic-based filters, which can overlook nuanced clinical cues and fail to prioritize information critical for decision-making. In this study, we propose Contextual, a novel framework that integrates a Context-Preserving Token Filtering method with a Domain-Specific Knowledge Graph (KG) for contextual augmentation. By preserving context-specific important tokens and enriching them with structured knowledge, ConTextual improves both linguistic coherence and clinical fidelity. Our extensive empirical evaluations on two public benchmark datasets demonstrate that ConTextual consistently outperforms other baselines. Our proposed approach highlights the complementary role of token-level filtering and structured retrieval in enhancing both linguistic and clinical integrity, as well as offering a scalable solution for improving precision in clinical text generation.
>
---
#### [replaced 033] SEALGuard: Safeguarding the Multilingual Conversations in Southeast Asian Languages for LLM Software Systems
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.08898v3](http://arxiv.org/pdf/2507.08898v3)**

> **作者:** Wenliang Shan; Michael Fu; Rui Yang; Chakkrit Tantithamthavorn
>
> **摘要:** Safety alignment is critical for LLM-powered systems. While recent LLM-powered guardrail approaches such as LlamaGuard achieve high detection accuracy of unsafe inputs written in English (e.g., ``How to create a bomb?''), they struggle with multilingual unsafe inputs. This limitation leaves LLM systems vulnerable to unsafe and jailbreak prompts written in low-resource languages such as those in Southeast Asia. This paper introduces SEALGuard, a multilingual guardrail designed to improve the safety alignment across diverse languages. It aims to address the multilingual safety alignment gap of existing guardrails and ensure effective filtering of unsafe and jailbreak prompts in LLM-powered systems. We adapt a general-purpose multilingual language model into a multilingual guardrail using low-rank adaptation (LoRA). We construct SEALSBench, a large-scale multilingual safety alignment dataset containing over 260,000 prompts in ten languages, including safe, unsafe, and jailbreak cases. We evaluate SEALGuard against state-of-the-art guardrails such as LlamaGuard on this benchmark. Our findings show that multilingual unsafe and jailbreak prompts substantially degrade the performance of the state-of-the-art LlamaGuard, which experiences a drop in Defense Success Rate (DSR) by 9% and 18%, respectively, compared to its performance on English-only prompts. In contrast, SEALGuard outperforms existing guardrails in detecting multilingual unsafe and jailbreak prompts, improving DSR by 48% over LlamaGuard and achieving the best DSR, precision, and F1-score. Our ablation study further reveals the contributions of adaptation strategies and model size to the overall performance of SEALGuard. We release our pre-trained model and benchmark at https://github.com/awsm-research/SEALGuard to support further research.
>
---
#### [replaced 034] Chain-of-Thought Prompting Obscures Hallucination Cues in Large Language Models: An Empirical Evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.17088v2](http://arxiv.org/pdf/2506.17088v2)**

> **作者:** Jiahao Cheng; Tiancheng Su; Jia Yuan; Guoxiu He; Jiawei Liu; Xinqi Tao; Jingwen Xie; Huaxia Li
>
> **摘要:** Large Language Models (LLMs) often exhibit \textit{hallucinations}, generating factually incorrect or semantically irrelevant content in response to prompts. Chain-of-Thought (CoT) prompting can mitigate hallucinations by encouraging step-by-step reasoning, but its impact on hallucination detection remains underexplored. To bridge this gap, we conduct a systematic empirical evaluation. We begin with a pilot experiment, revealing that CoT reasoning significantly affects the LLM's internal states and token probability distributions. Building on this, we evaluate the impact of various CoT prompting methods on mainstream hallucination detection methods across both instruction-tuned and reasoning-oriented LLMs. Specifically, we examine three key dimensions: changes in hallucination score distributions, variations in detection accuracy, and shifts in detection confidence. Our findings show that while CoT prompting helps reduce hallucination frequency, it also tends to obscure critical signals used for detection, impairing the effectiveness of various detection methods. Our study highlights an overlooked trade-off in the use of reasoning. Code is publicly available at: https://anonymous.4open.science/r/cot-hallu-detect.
>
---
#### [replaced 035] Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.03106v4](http://arxiv.org/pdf/2506.03106v4)**

> **作者:** Xiaoying Zhang; Hao Sun; Yipeng Zhang; Kaituo Feng; Chaochao Lu; Chao Yang; Helen Meng
>
> **备注:** 52 pages, updated with new experimental results and implementation details
>
> **摘要:** Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language models (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of spontaneous self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided self-refinements simultaneously while maintaining exploration. Additionally, we employ a shaping function to amplify learning from correct, especially unfamiliar, refinements and penalize incorrect ones. Extensive experiments with Qwen2.5-7B-Base, Qwen2.5-Math-7B-Base, and Qwen3-8B demonstrate that Critique-GRPO consistently outperforms supervised learning and RL-based fine-tuning methods across eight challenging mathematical, STEM, and general reasoning tasks, improving average pass@1 scores by approximately 4.4% and 3.8% on Qwen2.5-7B-Base and Qwen3-8B, respectively. Notably, Critique-GRPO enables effective self-improvement through self-critiquing and weak-to-strong generalization, achieving consistent gains over GRPO, such as 16.7% and 10.0% pass@1 improvements on AIME 2024, respectively.
>
---
#### [replaced 036] Unified Triplet-Level Hallucination Evaluation for Large Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.23114v4](http://arxiv.org/pdf/2410.23114v4)**

> **作者:** Junjie Wu; Tsz Ting Chung; Kai Chen; Dit-Yan Yeung
>
> **备注:** Accepted by TMLR 2025. Project Page: https://kaichen1998.github.io/projects/tri-he/
>
> **摘要:** Despite the outstanding performance in vision-language reasoning, Large Vision-Language Models (LVLMs) might generate hallucinated contents that do not exist in the given image. Most existing LVLM hallucination benchmarks are constrained to evaluate the object-related hallucinations. However, the potential hallucination on the relations between two objects, i.e., relation hallucination, still lacks investigation. To remedy that, we design a unified framework to measure the object and relation hallucination in LVLMs simultaneously. The core idea of our framework is to evaluate hallucinations via (object, relation, object) triplets extracted from LVLMs' responses, making it easily generalizable to different vision-language tasks. Based on our framework, we further introduce Tri-HE, a novel Triplet-level Hallucination Evaluation benchmark which can be used to study both object and relation hallucination at the same time. With comprehensive evaluations on Tri-HE, we observe that the relation hallucination issue is even more serious than object hallucination among existing LVLMs, highlighting a previously neglected problem towards reliable LVLMs. Moreover, based on our findings, we design a simple training-free approach that effectively mitigates hallucinations for LVLMs. Our dataset and code for the reproduction of our experiments are available publicly at https://github.com/wujunjie1998/Tri-HE.
>
---
#### [replaced 037] Synthesizing Privacy-Preserving Text Data via Finetuning without Finetuning Billion-Scale LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.12347v2](http://arxiv.org/pdf/2503.12347v2)**

> **作者:** Bowen Tan; Zheng Xu; Eric Xing; Zhiting Hu; Shanshan Wu
>
> **备注:** Code available at https://github.com/tanyuqian/synthetic-private-data
>
> **摘要:** Synthetic data offers a promising path to train models while preserving data privacy. Differentially private (DP) finetuning of large language models (LLMs) as data generator is effective, but is impractical when computation resources are limited. Meanwhile, prompt-based methods such as private evolution depend heavily on the manual prompts, and ineffectively use private information in their iterative data selection process. To overcome these limitations, we propose CTCL (Data Synthesis with ConTrollability and CLustering), a novel framework for generating privacy-preserving synthetic data without extensive prompt engineering or billion-scale LLM finetuning. CTCL pretrains a lightweight 140M conditional generator and a clustering-based topic model on large-scale public data. To further adapt to the private domain, the generator is DP finetuned on private data for fine-grained textual information, while the topic model extracts a DP histogram representing distributional information. The DP generator then samples according to the DP histogram to synthesize a desired number of data examples. Evaluation across five diverse domains demonstrates the effectiveness of our framework, particularly in the strong privacy regime. Systematic ablation validates the design of each framework component and highlights the scalability of our approach.
>
---
#### [replaced 038] Multi-task retriever fine-tuning for domain-specific and efficient RAG
- **分类: cs.CL; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.04652v2](http://arxiv.org/pdf/2501.04652v2)**

> **作者:** Patrice Béchard; Orlando Marquez Ayala
>
> **备注:** 7 pages, 2 figures. Accepted at Workshop on Structured Knowledge for Large Language Models (SKnowLLM) at KDD 2025
>
> **摘要:** Retrieval-Augmented Generation (RAG) has become ubiquitous when deploying Large Language Models (LLMs), as it can address typical limitations such as generating hallucinated or outdated information. However, when building real-world RAG applications, practical issues arise. First, the retrieved information is generally domain-specific. Since it is computationally expensive to fine-tune LLMs, it is more feasible to fine-tune the retriever to improve the quality of the data included in the LLM input. Second, as more applications are deployed in the same real-world system, one cannot afford to deploy separate retrievers. Moreover, these RAG applications normally retrieve different kinds of data. Our solution is to instruction fine-tune a small retriever encoder on a variety of domain-specific tasks to allow us to deploy one encoder that can serve many use cases, thereby achieving low-cost, scalability, and speed. We show how this encoder generalizes to out-of-domain settings as well as to an unseen retrieval task on real-world enterprise use cases.
>
---
#### [replaced 039] DeFine: Decision-Making with Analogical Reasoning over Factor Profiles
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.01772v2](http://arxiv.org/pdf/2410.01772v2)**

> **作者:** Yebowen Hu; Xiaoyang Wang; Wenlin Yao; Yiming Lu; Daoan Zhang; Hassan Foroosh; Dong Yu; Fei Liu
>
> **备注:** Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025), Vienna, Austria
>
> **摘要:** LLMs are ideal for decision-making thanks to their ability to reason over long contexts. However, challenges arise when processing speech transcripts that describe complex scenarios, as they are verbose and include repetition, hedging, and vagueness. E.g., during a company's earnings call, an executive might project a positive revenue outlook to reassure investors, despite uncertainty regarding future earnings. It is crucial for LLMs to incorporate this uncertainty systematically when making decisions. In this paper, we introduce \textsc{DeFine}, a modular framework that constructs probabilistic factor profiles from complex scenarios. It then integrates these profiles with analogical reasoning, leveraging insights from similar past experiences to guide LLMs in making critical decisions in new situations. Our framework separates the tasks of quantifying uncertainty and incorporating it into LLM decision-making. This approach is particularly useful in areas such as consulting and financial deliberation, where making decisions under uncertainty is vital.
>
---
#### [replaced 040] MERA Code: A Unified Framework for Evaluating Code Generation Across Tasks
- **分类: cs.SE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.12284v2](http://arxiv.org/pdf/2507.12284v2)**

> **作者:** Artem Chervyakov; Alexander Kharitonov; Pavel Zadorozhny; Adamenko Pavel; Rodion Levichev; Dmitrii Vorobev; Dmitrii Salikhov; Aidar Valeev; Alena Pestova; Maria Dziuba; Ilseyar Alimova; Artem Zavgorodnev; Aleksandr Medvedev; Stanislav Moiseev; Elena Bruches; Daniil Grebenkin; Roman Derunets; Vikulov Vladimir; Anton Emelyanov; Dmitrii Babaev; Vladimir V. Ivanov; Valentin Malykh; Alena Fenogenova
>
> **摘要:** Advancements in LLMs have enhanced task automation in software engineering; however, current evaluations primarily focus on natural language tasks, overlooking code quality. Most benchmarks prioritize high-level reasoning over executable code and real-world performance, leaving gaps in understanding true capabilities and risks associated with these models in production. To address this issue, we propose MERA Code, a new addition to the MERA benchmark family, specifically focused on evaluating code for the latest code generation LLMs in Russian. This benchmark includes 11 evaluation tasks that span 8 programming languages. Our proposed evaluation methodology features a taxonomy that outlines the practical coding skills necessary for models to complete these tasks. The benchmark comprises an open-source codebase for users to conduct MERA assessments, a scoring system compatible with various programming environments, and a platform featuring a leaderboard and submission system. We evaluate open LLMs and frontier API models, analyzing their limitations in terms of practical coding tasks in non-English languages. We are publicly releasing MERA to guide future research, anticipate groundbreaking features in model development, and standardize evaluation procedures.
>
---
#### [replaced 041] SmartThinker: Learning to Compress and Preserve Reasoning by Step-Level Length Control
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.04348v2](http://arxiv.org/pdf/2507.04348v2)**

> **作者:** Xingyang He; Xiao Ling; Jie Liu
>
> **摘要:** Large reasoning models (LRMs) have exhibited remarkable reasoning capabilities through inference-time scaling, but this progress has also introduced considerable redundancy and inefficiency into their reasoning processes, resulting in substantial computational waste. Previous work has attempted to mitigate this issue by penalizing the overall length of generated samples during reinforcement learning (RL), with the goal of encouraging a more concise chains of thought. However, we observe that such global length penalty often lead to excessive compression of critical reasoning steps while preserving unnecessary details in simpler ones, yielding a suboptimal trade-off between accuracy and efficiency. To address this issue, we propose SmartThinker, a two-stage learnable framework designed to enable fine-grained control over the length of reasoning chains based on the importance of each individual step. In the first stage, SmartThinker adapts a reasoning model to a short-form reasoning mode through rejection sampling combined with supervised fine-tuning (SFT). In the second stage, SmartThinker applies Step-Level Length Control Policy Optimization (SCPO) to refine the model output distribution, which increases the proportion of length allocated to critical steps while reducing redundancy in less important ones. SCPO consists of four core components: an online importance estimator, a step-level length control reward function, a step-level generalized advantage estimation (S-GAE) and a difficulty-adaptive clipping strategy. Working in concert, these components enable SCPO to implement differentiated length control across reasoning steps. Empirical results across multiple reasoning benchmarks and various backbone models demonstrate that SmartThinker significantly reduces redundant reasoning while achieving comparable or even superior performance to existing methods.
>
---
#### [replaced 042] Learning to Translate Ambiguous Terminology by Preference Optimization on Post-Edits
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.03580v2](http://arxiv.org/pdf/2507.03580v2)**

> **作者:** Nathaniel Berger; Johannes Eschbach-Dymanus; Miriam Exel; Matthias Huck; Stefan Riezler
>
> **摘要:** In real world translation scenarios, terminology is rarely one-to-one. Instead, multiple valid translations may appear in a terminology dictionary, but correctness of a translation depends on corporate style guides and context. This can be challenging for neural machine translation (NMT) systems. Luckily, in a corporate context, many examples of human post-edits of valid but incorrect terminology exist. The goal of this work is to learn how to disambiguate our terminology based on these corrections. Our approach is based on preference optimization, using the term post-edit as the knowledge to be preferred. While previous work had to rely on unambiguous translation dictionaries to set hard constraints during decoding, or to add soft constraints in the input, our framework requires neither one-to-one dictionaries nor human intervention at decoding time. We report results on English-German post-edited data and find that the optimal combination of supervised fine-tuning and preference optimization, with both term-specific and full sequence objectives, yields statistically significant improvements in term accuracy over a strong NMT baseline without significant losses in COMET score. Additionally, we release test sets from our post-edited data and terminology dictionary.
>
---
#### [replaced 043] Task-Circuit Quantization: Leveraging Knowledge Localization and Interpretability for Compression
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07389v2](http://arxiv.org/pdf/2504.07389v2)**

> **作者:** Hanqi Xiao; Yi-Lin Sung; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** COLM 2025 Camera Ready. Code: https://github.com/The-Inscrutable-X/TACQ
>
> **摘要:** Post-training quantization (PTQ) reduces a model's memory footprint by mapping full precision weights into low bit weights without costly retraining, but can degrade its downstream performance especially in low 2- to 3-bit settings. We develop a new mixed-precision PTQ approach, Task-Circuit Quantization (TaCQ), that draws parallels to automated circuit discovery, directly conditioning the quantization process on specific weight circuits -- which we define as sets of weights associated with downstream task performance. These weights are kept as 16-bit weights, while others are quantized, maintaining performance while only adding a marginal memory cost. Specifically, TaCQ contrasts unquantized model weights with a uniformly-quantized model to estimate the expected change in weights due to quantization and uses gradient information to predict the resulting impact on task performance, allowing us to preserve task-specific weights. We compare TaCQ-based quantization to existing mixed-precision quantization methods when conditioning both on general-purpose and task-specific data. Across QA, math reasoning, and text-to-SQL tasks for both Llama-3 and Qwen2.5, we find that TaCQ outperforms baselines using the same calibration data and a lower weight budget, achieving major improvements in the 2 and 3-bit regime. With only 3.1 bits we are able to recover 96% of Llama-3-8B-Instruct's unquantized 16-bit MMLU performance, obtaining a 5.25% absolute improvement over SPQR. We also observe consistently large gains over existing methods in the 2-bit regime, with an average gain of 14.74% over the strongest baseline, SliM-LLM. Moreover, we observe a 7.20% gain without conditioning on specific tasks, showing TaCQ's ability to identify important weights is not limited to task-conditioned settings.
>
---
