# 自然语言处理 cs.CL

- **最新发布 89 篇**

- **更新 108 篇**

## 最新发布

#### [new 001] GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决俄罗斯语大模型资源不足的问题。提出GigaChat家族模型，通过专家混合架构提升效率，并进行多语言对比实验。**

- **链接: [http://arxiv.org/pdf/2506.09440v1](http://arxiv.org/pdf/2506.09440v1)**

> **作者:** GigaChat team; Mamedov Valentin; Evgenii Kosarev; Gregory Leleytner; Ilya Shchuckin; Valeriy Berezovskiy; Daniil Smirnov; Dmitry Kozlov; Sergei Averkiev; Lukyanenko Ivan; Aleksandr Proshunin; Ainur Israfilova; Ivan Baskov; Artem Chervyakov; Emil Shakirov; Mikhail Kolesov; Daria Khomich; Darya Latortseva; Sergei Porkhun; Yury Fedorov; Oleg Kutuzov; Polina Kudriavtseva; Sofiia Soldatova; Kolodin Egor; Stanislav Pyatkin; Dzmitry Menshykh; Grafov Sergei; Eldar Damirov; Karlov Vladimir; Ruslan Gaitukiev; Arkadiy Shatenov; Alena Fenogenova; Nikita Savushkin; Fedor Minkin
>
> **备注:** ACL-2025 System Demo
>
> **摘要:** Generative large language models (LLMs) have become crucial for modern NLP research and applications across various languages. However, the development of foundational models specifically tailored to the Russian language has been limited, primarily due to the significant computational resources required. This paper introduces the GigaChat family of Russian LLMs, available in various sizes, including base models and instruction-tuned versions. We provide a detailed report on the model architecture, pre-training process, and experiments to guide design choices. In addition, we evaluate their performance on Russian and English benchmarks and compare GigaChat with multilingual analogs. The paper presents a system demonstration of the top-performing models accessible via an API, a Telegram bot, and a Web interface. Furthermore, we have released three open GigaChat models in open-source (https://huggingface.co/ai-sage), aiming to expand NLP research opportunities and support the development of industrial solutions for the Russian language.
>
---
#### [new 002] Error-Guided Pose Augmentation: Enhancing Rehabilitation Exercise Assessment through Targeted Data Generation
- **分类: cs.CL; I.2.1**

- **简介: 该论文属于康复运动评估任务，旨在解决数据不平衡和细微动作错误检测问题。通过生成模拟临床错误的骨骼数据，提升运动质量评估的准确性。**

- **链接: [http://arxiv.org/pdf/2506.09833v1](http://arxiv.org/pdf/2506.09833v1)**

> **作者:** Omar Sherif; Ali Hamdi
>
> **备注:** 6 pages, 1 figure. To appear in Intelligent Methods, Systems, and Applications 2025
>
> **摘要:** Effective rehabilitation assessment is essential for monitoring patient progress, particularly in home-based settings. Existing systems often face challenges such as data imbalance and difficulty detecting subtle movement errors. This paper introduces Error-Guided Pose Augmentation (EGPA), a method that generates synthetic skeleton data by simulating clinically relevant movement mistakes. Unlike standard augmentation techniques, EGPA targets biomechanical errors observed in rehabilitation. Combined with an attention-based graph convolutional network, EGPA improves performance across multiple evaluation metrics. Experiments demonstrate reductions in mean absolute error of up to 27.6 percent and gains in error classification accuracy of 45.8 percent. Attention visualizations show that the model learns to focus on clinically significant joints and movement phases, enhancing both accuracy and interpretability. EGPA offers a promising approach for improving automated movement quality assessment in both clinical and home-based rehabilitation contexts.
>
---
#### [new 003] Aspect-Based Opinion Summarization with Argumentation Schemes
- **分类: cs.CL**

- **简介: 该论文属于观点摘要任务，旨在自动提取产品评论中的关键方面及其支持观点。通过构建ASESUM框架，解决跨领域摘要生成问题，提升摘要的多样性和准确性。**

- **链接: [http://arxiv.org/pdf/2506.09917v1](http://arxiv.org/pdf/2506.09917v1)**

> **作者:** Wendi Zhou; Ameer Saadat-Yazd; Nadin Kokciyan
>
> **备注:** Accepted by ArgMining 2025
>
> **摘要:** Reviews are valuable resources for customers making purchase decisions in online shopping. However, it is impractical for customers to go over the vast number of reviews and manually conclude the prominent opinions, which prompts the need for automated opinion summarization systems. Previous approaches, either extractive or abstractive, face challenges in automatically producing grounded aspect-centric summaries. In this paper, we propose a novel summarization system that not only captures predominant opinions from an aspect perspective with supporting evidence, but also adapts to varying domains without relying on a pre-defined set of aspects. Our proposed framework, ASESUM, summarizes viewpoints relevant to the critical aspects of a product by extracting aspect-centric arguments and measuring their salience and validity. We conduct experiments on a real-world dataset to demonstrate the superiority of our approach in capturing diverse perspectives of the original reviews compared to new and existing methods.
>
---
#### [new 004] Did I Faithfully Say What I Thought? Bridging the Gap Between Neural Activity and Self-Explanations in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM生成的自解释不忠实问题。通过比较模型内部状态与自解释，提出新框架评估并提升解释的可信度。**

- **链接: [http://arxiv.org/pdf/2506.09277v1](http://arxiv.org/pdf/2506.09277v1)**

> **作者:** Milan Bhan; Jean-Noel Vittaut; Nicolas Chesneau; Sarath Chandar; Marie-Jeanne Lesot
>
> **摘要:** Large Language Models (LLM) have demonstrated the capability of generating free text self Natural Language Explanation (self-NLE) to justify their answers. Despite their logical appearance, self-NLE do not necessarily reflect the LLM actual decision-making process, making such explanations unfaithful. While existing methods for measuring self-NLE faithfulness mostly rely on behavioral tests or computational block identification, none of them examines the neural activity underlying the model's reasoning. This work introduces a novel flexible framework for quantitatively measuring the faithfulness of LLM-generated self-NLE by directly comparing the latter with interpretations of the model's internal hidden states. The proposed framework is versatile and provides deep insights into self-NLE faithfulness by establishing a direct connection between self-NLE and model reasoning. This approach advances the understanding of self-NLE faithfulness and provides building blocks for generating more faithful self-NLE.
>
---
#### [new 005] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于大语言模型对齐任务，旨在解决DAAs中奖励生成差距问题。通过POET方法优化训练过程，提升模型生成性能。**

- **链接: [http://arxiv.org/pdf/2506.09457v1](http://arxiv.org/pdf/2506.09457v1)**

> **作者:** Zeguan Xiao; Yun Chen; Guanhua Chen
>
> **摘要:** Direct Alignment Algorithms (DAAs), such as Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO), have emerged as efficient alternatives to Reinforcement Learning from Human Feedback (RLHF) algorithms for aligning large language models (LLMs) with human preferences. However, DAAs suffer from a fundamental limitation we identify as the "reward-generation gap" -- a misalignment between optimization objectives during training and actual generation performance during inference. In this paper, we find a contributor to the reward-generation gap is the mismatch between the inherent importance of prefix tokens during the LLM generation process and how this importance is reflected in the implicit reward functions of DAAs. To bridge the gap, we introduce a simple yet effective approach called Prefix-Oriented Equal-length Training (POET), which truncates both preferred and dispreferred responses to match the shorter one's length. Training with POET, where both responses in each sample are truncated to equal length, resulting in diverse truncated lengths across samples, the optimization of DAAs objective is implicitly constrained to converge across all positions, thus paying more attention to prefix tokens than the standard DAAs. We conduct experiments with DPO and SimPO, two representative DAAs, demonstrating that POET improves over their standard implementations, achieving up to 15.6 points in AlpacaEval 2 and overall improvements across downstream tasks. Our results highlight the importance of addressing the misalignment between reward optimization and generation performance in DAAs.
>
---
#### [new 006] Resa: Transparent Reasoning Models via SAEs
- **分类: cs.CL**

- **简介: 该论文提出Resa模型，通过SAE-Tuning方法高效提升语言模型的推理能力，解决如何低成本获取强推理能力的问题。**

- **链接: [http://arxiv.org/pdf/2506.09967v1](http://arxiv.org/pdf/2506.09967v1)**

> **作者:** Shangshang Wang; Julian Asilis; Ömer Faruk Akgül; Enes Burak Bilgin; Ollie Liu; Deqing Fu; Willie Neiswanger
>
> **摘要:** How cost-effectively can we elicit strong reasoning in language models by leveraging their underlying representations? We answer this question with Resa, a family of 1.5B reasoning models trained via a novel and efficient sparse autoencoder tuning (SAE-Tuning) procedure. This method first trains an SAE to capture reasoning abilities from a source model, and then uses the trained SAE to guide a standard supervised fine-tuning process to elicit such abilities in a target model, all using verified question-answer data without any reasoning traces. Notably, when applied to certain base models before further RL post-training, SAE-Tuning retains >97% of its RL-trained counterpart's reasoning performance while reducing training costs by >2000x to roughly \$1 and training time by >450x to around 20 minutes. Furthermore, when applied to lightly RL-trained models (e.g., within 1 hour on 2 GPUs), it enables reasoning performance such as 43.33% Pass@1 on AIME24 and 90% Pass@1 on AMC23 for only around \$1 additional cost. Surprisingly, the reasoning abilities extracted via SAEs are potentially both generalizable and modular. Generality means abilities extracted from one dataset still elevate performance on a larger and overlapping corpus. Modularity means abilities extracted from Qwen or Qwen-Math can be attached to the R1-Distill model at test time, without any retraining, and yield comparable gains. Extensive ablations validate these findings and all artifacts are fully open-sourced.
>
---
#### [new 007] ComfyUI-R1: Exploring Reasoning Models for Workflow Generation
- **分类: cs.CL; cs.CV; cs.SE**

- **简介: 该论文属于AI艺术创作任务，旨在解决复杂工作流构建难题。通过构建推理模型ComfyUI-R1，实现自动化工作流生成，提升用户使用效率与创作能力。**

- **链接: [http://arxiv.org/pdf/2506.09790v1](http://arxiv.org/pdf/2506.09790v1)**

> **作者:** Zhenran Xu; Yiyu Wang; Xue Yang; Longyue Wang; Weihua Luo; Kaifu Zhang; Baotian Hu; Min Zhang
>
> **备注:** Work in progress. Try it out in ComfyUI-Copilot https://github.com/AIDC-AI/ComfyUI-Copilot
>
> **摘要:** AI-generated content has evolved from monolithic models to modular workflows, particularly on platforms like ComfyUI, enabling customization in creative pipelines. However, crafting effective workflows requires great expertise to orchestrate numerous specialized components, presenting a steep learning curve for users. To address this challenge, we introduce ComfyUI-R1, the first large reasoning model for automated workflow generation. Starting with our curated dataset of 4K workflows, we construct long chain-of-thought (CoT) reasoning data, including node selection, workflow planning, and code-level workflow representation. ComfyUI-R1 is trained through a two-stage framework: (1) CoT fine-tuning for cold start, adapting models to the ComfyUI domain; (2) reinforcement learning for incentivizing reasoning capability, guided by a fine-grained rule-metric hybrid reward, ensuring format validity, structural integrity, and node-level fidelity. Experiments show that our 7B-parameter model achieves a 97\% format validity rate, along with high pass rate, node-level and graph-level F1 scores, significantly surpassing prior state-of-the-art methods that employ leading closed-source models such as GPT-4o and Claude series. Further analysis highlights the critical role of the reasoning process and the advantage of transforming workflows into code. Qualitative comparison reveals our strength in synthesizing intricate workflows with diverse nodes, underscoring the potential of long CoT reasoning in AI art creation.
>
---
#### [new 008] Large Language Models for Toxic Language Detection in Low-Resource Balkan Languages
- **分类: cs.CL**

- **简介: 该论文属于文本分类任务，旨在解决低资源巴尔干语言中毒性语言检测问题。通过构建数据集并测试多个大模型，探索上下文增强对检测效果的影响。**

- **链接: [http://arxiv.org/pdf/2506.09992v1](http://arxiv.org/pdf/2506.09992v1)**

> **作者:** Amel Muminovic; Amela Kadric Muminovic
>
> **备注:** 8 pages
>
> **摘要:** Online toxic language causes real harm, especially in regions with limited moderation tools. In this study, we evaluate how large language models handle toxic comments in Serbian, Croatian, and Bosnian, languages with limited labeled data. We built and manually labeled a dataset of 4,500 YouTube and TikTok comments drawn from videos across diverse categories, including music, politics, sports, modeling, influencer content, discussions of sexism, and general topics. Four models (GPT-3.5 Turbo, GPT-4.1, Gemini 1.5 Pro, and Claude 3 Opus) were tested in two modes: zero-shot and context-augmented. We measured precision, recall, F1 score, accuracy and false positive rates. Including a short context snippet raised recall by about 0.12 on average and improved F1 score by up to 0.10, though it sometimes increased false positives. The best balance came from Gemini in context-augmented mode, reaching an F1 score of 0.82 and accuracy of 0.82, while zero-shot GPT-4.1 led on precision and had the lowest false alarms. We show how adding minimal context can improve toxic language detection in low-resource settings and suggest practical strategies such as improved prompt design and threshold calibration. These results show that prompt design alone can yield meaningful gains in toxicity detection for underserved Balkan language communities.
>
---
#### [new 009] RePO: Replay-Enhanced Policy Optimization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决大语言模型优化中的计算成本高和数据效率低的问题。提出RePO方法，通过引入回放策略提升优化效果。**

- **链接: [http://arxiv.org/pdf/2506.09340v1](http://arxiv.org/pdf/2506.09340v1)**

> **作者:** Siheng Li; Zhanhui Zhou; Wai Lam; Chao Yang; Chaochao Lu
>
> **备注:** Project Page: https://github.com/SihengLi99/RePO
>
> **摘要:** Reinforcement learning (RL) is vital for optimizing large language models (LLMs). Recent Group Relative Policy Optimization (GRPO) estimates advantages using multiple on-policy outputs per prompt, leading to high computational costs and low data efficiency. To address this, we introduce Replay-Enhanced Policy Optimization (RePO), which leverages diverse replay strategies to retrieve off-policy samples from a replay buffer, allowing policy optimization based on a broader and more diverse set of samples for each prompt. Experiments on five LLMs across seven mathematical reasoning benchmarks demonstrate that RePO achieves absolute average performance gains of $18.4$ and $4.1$ points for Qwen2.5-Math-1.5B and Qwen3-1.7B, respectively, compared to GRPO. Further analysis indicates that RePO increases computational cost by $15\%$ while raising the number of effective optimization steps by $48\%$ for Qwen3-1.7B, with both on-policy and off-policy sample numbers set to $8$. The repository can be accessed at https://github.com/SihengLi99/RePO.
>
---
#### [new 010] Multi-Agent Language Models: Advancing Cooperation, Coordination, and Adaptation
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于多智能体协作任务，旨在研究LLMs是否具备理论心智，通过MARL提升AI的协作与适应能力。**

- **链接: [http://arxiv.org/pdf/2506.09331v1](http://arxiv.org/pdf/2506.09331v1)**

> **作者:** Arjun Vaithilingam Sudhakar
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2311.07687
>
> **摘要:** Modern Large Language Models (LLMs) exhibit impressive zero-shot and few-shot generalization capabilities across complex natural language tasks, enabling their widespread use as virtual assistants for diverse applications such as translation and summarization. Despite being trained solely on large corpora of text without explicit supervision on author intent, LLMs appear to infer the underlying meaning of textual interactions. This raises a fundamental question: can LLMs model and reason about the intentions of others, i.e., do they possess a form of theory of mind? Understanding other's intentions is crucial for effective collaboration, which underpins human societal success and is essential for cooperative interactions among multiple agents, including humans and autonomous systems. In this work, we investigate the theory of mind in LLMs through the lens of cooperative multi-agent reinforcement learning (MARL), where agents learn to collaborate via repeated interactions, mirroring human social reasoning. Our approach aims to enhance artificial agent's ability to adapt and cooperate with both artificial and human partners. By leveraging LLM-based agents capable of natural language interaction, we move towards creating hybrid human-AI systems that can foster seamless collaboration, with broad implications for the future of human-artificial interaction.
>
---
#### [new 011] Attention Head Embeddings with Trainable Deep Kernels for Hallucination Detection in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 hallucination 检测任务，旨在解决 LLMs 生成内容不准确的问题。通过分析隐藏状态分布的差异，提出一种基于深度可学习核的方法进行检测。**

- **链接: [http://arxiv.org/pdf/2506.09886v1](http://arxiv.org/pdf/2506.09886v1)**

> **作者:** Rodion Oblovatny; Alexandra Bazarova; Alexey Zaytsev
>
> **摘要:** We present a novel approach for detecting hallucinations in large language models (LLMs) by analyzing the probabilistic divergence between prompt and response hidden-state distributions. Counterintuitively, we find that hallucinated responses exhibit smaller deviations from their prompts compared to grounded responses, suggesting that hallucinations often arise from superficial rephrasing rather than substantive reasoning. Leveraging this insight, we propose a model-intrinsic detection method that uses distributional distances as principled hallucination scores, eliminating the need for external knowledge or auxiliary models. To enhance sensitivity, we employ deep learnable kernels that automatically adapt to capture nuanced geometric differences between distributions. Our approach outperforms existing baselines, demonstrating state-of-the-art performance on several benchmarks. The method remains competitive even without kernel training, offering a robust, scalable solution for hallucination detection.
>
---
#### [new 012] Binary classification for perceived quality of headlines and links on worldwide news websites, 2018-2024
- **分类: cs.CL**

- **简介: 该论文属于二分类任务，旨在区分新闻标题和链接的感知质量高低。通过机器学习与深度学习模型进行分类，验证了不同方法的有效性。**

- **链接: [http://arxiv.org/pdf/2506.09381v1](http://arxiv.org/pdf/2506.09381v1)**

> **作者:** Austin McCutcheon; Thiago E. A. de Oliveira; Aleksandr Zheleznov; Chris Brogly
>
> **摘要:** The proliferation of online news enables potential widespread publication of perceived low-quality news headlines/links. As a result, we investigated whether it was possible to automatically distinguish perceived lower-quality news headlines/links from perceived higher-quality headlines/links. We evaluated twelve machine learning models on a binary, balanced dataset of 57,544,214 worldwide news website links/headings from 2018-2024 (28,772,107 per class) with 115 extracted linguistic features. Binary labels for each text were derived from scores based on expert consensus regarding the respective news domain quality. Traditional ensemble methods, particularly the bagging classifier, had strong performance (88.1% accuracy, 88.3% F1, 80/20 train/test split). Fine-tuned DistilBERT achieved the highest accuracy (90.3%, 80/20 train/test split) but required more training time. The results suggest that both NLP features with traditional classifiers and deep learning models can effectively differentiate perceived news headline/link quality, with some trade-off between predictive performance and train time.
>
---
#### [new 013] KG-Infused RAG: Augmenting Corpus-Based RAG with External Knowledge Graphs
- **分类: cs.CL**

- **简介: 该论文属于问答任务，旨在提升RAG的准确性。通过融合知识图谱，增强检索与生成，解决单一信息源和缺乏认知机制的问题。**

- **链接: [http://arxiv.org/pdf/2506.09542v1](http://arxiv.org/pdf/2506.09542v1)**

> **作者:** Dingjun Wu; Yukun Yan; Zhenghao Liu; Zhiyuan Liu; Maosong Sun
>
> **摘要:** Retrieval-Augmented Generation (RAG) improves factual accuracy by grounding responses in external knowledge. However, existing methods typically rely on a single source, either unstructured text or structured knowledge. Moreover, they lack cognitively inspired mechanisms for activating relevant knowledge. To address these issues, we propose KG-Infused RAG, a framework that integrates KGs into RAG systems to implement spreading activation, a cognitive process that enables concept association and inference. KG-Infused RAG retrieves KG facts, expands the query accordingly, and enhances generation by combining corpus passages with structured facts, enabling interpretable, multi-source retrieval grounded in semantic structure. We further improve KG-Infused RAG via preference learning on sampled key stages in the pipeline. Experiments on five QA benchmarks show that KG-Infused RAG consistently outperforms vanilla RAG (by 3.8% to 13.8%). Additionally, when integrated into Self-RAG, KG-Infused RAG brings further performance gains, demonstrating its effectiveness and versatility as a plug-and-play enhancement module for corpus-based RAG methods.
>
---
#### [new 014] Taming SQL Complexity: LLM-Based Equivalence Evaluation for Text-to-SQL
- **分类: cs.CL**

- **简介: 该论文属于Text-to-SQL任务，旨在解决生成SQL语义等价性评估问题，通过LLM分析SQL等价与不等价模式。**

- **链接: [http://arxiv.org/pdf/2506.09359v1](http://arxiv.org/pdf/2506.09359v1)**

> **作者:** Qingyun Zeng; Simin Ma; Arash Niknafs; Ashish Basran; Carol Szabo
>
> **备注:** 8 pages
>
> **摘要:** The rise of Large Language Models (LLMs) has significantly advanced Text-to-SQL (NL2SQL) systems, yet evaluating the semantic equivalence of generated SQL remains a challenge, especially given ambiguous user queries and multiple valid SQL interpretations. This paper explores using LLMs to assess both semantic and a more practical "weak" semantic equivalence. We analyze common patterns of SQL equivalence and inequivalence, discuss challenges in LLM-based evaluation.
>
---
#### [new 015] Do LLMs Give Psychometrically Plausible Responses in Educational Assessments?
- **分类: cs.CL**

- **简介: 该论文属于教育评估任务，旨在检验LLMs在测试中的回答是否符合心理测量学标准。研究通过两个理论框架分析LLMs的回答，发现其需校准才能更接近人类表现。**

- **链接: [http://arxiv.org/pdf/2506.09796v1](http://arxiv.org/pdf/2506.09796v1)**

> **作者:** Andreas Säuberli; Diego Frassinelli; Barbara Plank
>
> **备注:** Accepted for publication at the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA) at ACL 2025
>
> **摘要:** Knowing how test takers answer items in educational assessments is essential for test development, to evaluate item quality, and to improve test validity. However, this process usually requires extensive pilot studies with human participants. If large language models (LLMs) exhibit human-like response behavior to test items, this could open up the possibility of using them as pilot participants to accelerate test development. In this paper, we evaluate the human-likeness or psychometric plausibility of responses from 18 instruction-tuned LLMs with two publicly available datasets of multiple-choice test items across three subjects: reading, U.S. history, and economics. Our methodology builds on two theoretical frameworks from psychometrics which are commonly used in educational assessment, classical test theory and item response theory. The results show that while larger models are excessively confident, their response distributions can be more human-like when calibrated with temperature scaling. In addition, we find that LLMs tend to correlate better with humans in reading comprehension items compared to other subjects. However, the correlations are not very strong overall, indicating that LLMs should not be used for piloting educational assessments in a zero-shot setting.
>
---
#### [new 016] Improved Supervised Fine-Tuning for Large Language Models to Mitigate Catastrophic Forgetting
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在解决大模型在监督微调中出现的灾难性遗忘问题。通过重构指令分布和筛选数据，提升任务性能同时保持泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.09428v1](http://arxiv.org/pdf/2506.09428v1)**

> **作者:** Fei Ding; Baiqiao Wang
>
> **摘要:** Supervised Fine-Tuning (SFT), while enhancing large language models(LLMs)' instruction-following capabilities and domain-specific task adaptability, often diminishes their general capabilities. Moreover, due to the inaccessibility of original pre-training data, catastrophic forgetting tends to be exacerbated when third-party practitioners implement SFT on open-sourced models. To address this challenge, we propose a novel, more cost-effective SFT method which could effectively reduce the risk of catastrophic forgetting without access to original SFT data. Our approach begins by reconstructing the likely SFT instruction distribution of the base model, followed by a multi-model screening process to select optimal data, which is then mixed with new data for SFT. Experimental results demonstrate that our method preserves generalization capabilities in general domains while improving task-specific performance.
>
---
#### [new 017] Memorization in Language Models through the Lens of Intrinsic Dimension
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究语言模型中的记忆现象。旨在探讨内在维度如何影响模型对训练数据的无意识记忆，通过实验发现高内在维度序列更不易被记忆。**

- **链接: [http://arxiv.org/pdf/2506.09591v1](http://arxiv.org/pdf/2506.09591v1)**

> **作者:** Stefan Arnold
>
> **摘要:** Language Models (LMs) are prone to memorizing parts of their data during training and unintentionally emitting them at generation time, raising concerns about privacy leakage and disclosure of intellectual property. While previous research has identified properties such as context length, parameter size, and duplication frequency, as key drivers of unintended memorization, little is known about how the latent structure modulates this rate of memorization. We investigate the role of Intrinsic Dimension (ID), a geometric proxy for the structural complexity of a sequence in latent space, in modulating memorization. Our findings suggest that ID acts as a suppressive signal for memorization: compared to low-ID sequences, high-ID sequences are less likely to be memorized, particularly in overparameterized models and under sparse exposure. These findings highlight the interaction between scale, exposure, and complexity in shaping memorization.
>
---
#### [new 018] PHRASED: Phrase Dictionary Biasing for Speech Translation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音翻译任务，旨在解决短语翻译困难的问题。通过引入短语字典偏置方法，提升模型对短语的翻译效果。**

- **链接: [http://arxiv.org/pdf/2506.09175v1](http://arxiv.org/pdf/2506.09175v1)**

> **作者:** Peidong Wang; Jian Xue; Rui Zhao; Junkun Chen; Aswin Shanmugam Subramanian; Jinyu Li
>
> **摘要:** Phrases are essential to understand the core concepts in conversations. However, due to their rare occurrence in training data, correct translation of phrases is challenging in speech translation tasks. In this paper, we propose a phrase dictionary biasing method to leverage pairs of phrases mapping from the source language to the target language. We apply the phrase dictionary biasing method to two types of widely adopted models, a transducer-based streaming speech translation model and a multimodal large language model. Experimental results show that the phrase dictionary biasing method outperforms phrase list biasing by 21% relatively for the streaming speech translation model. In addition, phrase dictionary biasing enables multimodal large language models to use external phrase information, achieving 85% relative improvement in phrase recall.
>
---
#### [new 019] EmoNet-Voice: A Fine-Grained, Expert-Verified Benchmark for Speech Emotion Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音情感识别任务，旨在解决现有数据集在情感细粒度、隐私和真实性上的不足。工作包括构建EmoNet-Voice数据集及验证模型，提升情感检测精度。**

- **链接: [http://arxiv.org/pdf/2506.09827v1](http://arxiv.org/pdf/2506.09827v1)**

> **作者:** Christoph Schuhmann; Robert Kaczmarczyk; Gollam Rabby; Felix Friedrich; Maurice Kraus; Kourosh Nadi; Huu Nguyen; Kristian Kersting; Sören Auer
>
> **摘要:** The advancement of text-to-speech and audio generation models necessitates robust benchmarks for evaluating the emotional understanding capabilities of AI systems. Current speech emotion recognition (SER) datasets often exhibit limitations in emotional granularity, privacy concerns, or reliance on acted portrayals. This paper introduces EmoNet-Voice, a new resource for speech emotion detection, which includes EmoNet-Voice Big, a large-scale pre-training dataset (featuring over 4,500 hours of speech across 11 voices, 40 emotions, and 4 languages), and EmoNet-Voice Bench, a novel benchmark dataset with human expert annotations. EmoNet-Voice is designed to evaluate SER models on a fine-grained spectrum of 40 emotion categories with different levels of intensities. Leveraging state-of-the-art voice generation, we curated synthetic audio snippets simulating actors portraying scenes designed to evoke specific emotions. Crucially, we conducted rigorous validation by psychology experts who assigned perceived intensity labels. This synthetic, privacy-preserving approach allows for the inclusion of sensitive emotional states often absent in existing datasets. Lastly, we introduce Empathic Insight Voice models that set a new standard in speech emotion recognition with high agreement with human experts. Our evaluations across the current model landscape exhibit valuable findings, such as high-arousal emotions like anger being much easier to detect than low-arousal states like concentration.
>
---
#### [new 020] Inv-Entropy: A Fully Probabilistic Framework for Uncertainty Quantification in Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的不确定性量化任务，旨在解决大语言模型可靠性问题。提出Inv-Entropy框架，通过概率方法评估输出不确定性。**

- **链接: [http://arxiv.org/pdf/2506.09684v1](http://arxiv.org/pdf/2506.09684v1)**

> **作者:** Haoyi Song; Ruihan Ji; Naichen Shi; Fan Lai; Raed Al Kontar
>
> **摘要:** Large language models (LLMs) have transformed natural language processing, but their reliable deployment requires effective uncertainty quantification (UQ). Existing UQ methods are often heuristic and lack a probabilistic foundation. This paper begins by providing a theoretical justification for the role of perturbations in UQ for LLMs. We then introduce a dual random walk perspective, modeling input-output pairs as two Markov chains with transition probabilities defined by semantic similarity. Building on this, we propose a fully probabilistic framework based on an inverse model, which quantifies uncertainty by evaluating the diversity of the input space conditioned on a given output through systematic perturbations. Within this framework, we define a new uncertainty measure, Inv-Entropy. A key strength of our framework is its flexibility: it supports various definitions of uncertainty measures, embeddings, perturbation strategies, and similarity metrics. We also propose GAAP, a perturbation algorithm based on genetic algorithms, which enhances the diversity of sampled inputs. In addition, we introduce a new evaluation metric, Temperature Sensitivity of Uncertainty (TSU), which directly assesses uncertainty without relying on correctness as a proxy. Extensive experiments demonstrate that Inv-Entropy outperforms existing semantic UQ methods. The code to reproduce the results can be found at https://github.com/UMDataScienceLab/Uncertainty-Quantification-for-LLMs.
>
---
#### [new 021] From Symbolic to Neural and Back: Exploring Knowledge Graph-Large Language Model Synergies
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于知识图谱与大语言模型融合任务，旨在解决知识整合与推理问题，通过分析两者协同机制，提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2506.09566v1](http://arxiv.org/pdf/2506.09566v1)**

> **作者:** Blaž Škrlj; Boshko Koloski; Senja Pollak; Nada Lavrač
>
> **备注:** To-appear as a book chapter
>
> **摘要:** Integrating structured knowledge from Knowledge Graphs (KGs) into Large Language Models (LLMs) enhances factual grounding and reasoning capabilities. This survey paper systematically examines the synergy between KGs and LLMs, categorizing existing approaches into two main groups: KG-enhanced LLMs, which improve reasoning, reduce hallucinations, and enable complex question answering; and LLM-augmented KGs, which facilitate KG construction, completion, and querying. Through comprehensive analysis, we identify critical gaps and highlight the mutual benefits of structured knowledge integration. Compared to existing surveys, our study uniquely emphasizes scalability, computational efficiency, and data quality. Finally, we propose future research directions, including neuro-symbolic integration, dynamic KG updating, data reliability, and ethical considerations, paving the way for intelligent systems capable of managing more complex real-world knowledge tasks.
>
---
#### [new 022] ReasonMed: A 370K Multi-Agent Generated Dataset for Advancing Medical Reasoning
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于医疗推理任务，旨在提升大模型在医学问答中的推理能力。通过构建大规模数据集ReasonMed，并优化训练策略，显著提高了模型性能。**

- **链接: [http://arxiv.org/pdf/2506.09513v1](http://arxiv.org/pdf/2506.09513v1)**

> **作者:** Yu Sun; Xingyu Qian; Weiwen Xu; Hao Zhang; Chenghao Xiao; Long Li; Yu Rong; Wenbing Huang; Qifeng Bai; Tingyang Xu
>
> **备注:** 24 pages, 6 figures, 7 tables
>
> **摘要:** Though reasoning-based large language models (LLMs) have excelled in mathematics and programming, their capabilities in knowledge-intensive medical question answering remain underexplored. To address this, we introduce ReasonMed, the largest medical reasoning dataset, comprising 370k high-quality examples distilled from 1.7 million initial reasoning paths generated by various LLMs. ReasonMed is constructed through a \textit{multi-agent verification and refinement process}, where we design an \textit{Error Refiner} to enhance the reasoning paths by identifying and correcting error-prone steps flagged by a verifier. Leveraging ReasonMed, we systematically investigate best practices for training medical reasoning models and find that combining detailed Chain-of-Thought (CoT) reasoning with concise answer summaries yields the most effective fine-tuning strategy. Based on this strategy, we train ReasonMed-7B, which sets a new benchmark for sub-10B models, outperforming the prior best by 4.17\% and even exceeding LLaMA3.1-70B on PubMedQA by 4.60\%.
>
---
#### [new 023] Step-by-step Instructions and a Simple Tabular Output Format Improve the Dependency Parsing Accuracy of LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的依存句法分析任务，旨在提升大语言模型的解析准确性。通过分步指令和简化输出格式，有效解决了结构不准确的问题。**

- **链接: [http://arxiv.org/pdf/2506.09983v1](http://arxiv.org/pdf/2506.09983v1)**

> **作者:** Hiroshi Matsuda; Chunpeng Ma; Masayuki Asahara
>
> **备注:** 9 pages, 2 figures, accepted for SyntaxFest 2025
>
> **摘要:** Recent advances in large language models (LLMs) have enabled impressive performance in various tasks. However, standard prompting often struggles to produce structurally valid and accurate outputs, especially in dependency parsing. We propose a novel step-by-step instruction strategy, where universal part-of-speech tagging precedes the prediction of syntactic heads and dependency labels, and a simplified CoNLL-U like output format, our method achieves state-of-the-art accuracy on Universal Dependencies datasets across 17 languages without hallucination or contamination. We further show that multilingual fine-tuning simultaneously improves cross-language generalization performance. Our results highlight the effectiveness of explicit reasoning steps in LLM-based parsing and offer a scalable, format-consistent alternative to bracket-based approaches.
>
---
#### [new 024] LLM-as-a-qualitative-judge: automating error analysis in natural language generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成的评估任务，旨在通过LLM生成结构化错误报告，解决传统定量评估缺乏深度问题。**

- **链接: [http://arxiv.org/pdf/2506.09147v1](http://arxiv.org/pdf/2506.09147v1)**

> **作者:** Nadezhda Chirkova; Tunde Oluwaseyi Ajayi; Seth Aycock; Zain Muhammad Mujahid; Vladana Perlić; Ekaterina Borisova; Markarit Vartampetian
>
> **摘要:** Prompting large language models (LLMs) to evaluate generated text, known as LLM-as-a-judge, has become a standard evaluation approach in natural language generation (NLG), but is primarily used as a quantitative tool, i.e. with numerical scores as main outputs. In this work, we propose LLM-as-a-qualitative-judge, an LLM-based evaluation approach with the main output being a structured report of common issue types in the NLG system outputs. Our approach is targeted at providing developers with meaningful insights on what improvements can be done to a given NLG system and consists of two main steps, namely open-ended per-instance issue analysis and clustering of the discovered issues using an intuitive cumulative algorithm. We also introduce a strategy for evaluating the proposed approach, coupled with ~300 annotations of issues in instances from 12 NLG datasets. Our results show that LLM-as-a-qualitative-judge correctly recognizes instance-specific issues in 2/3 cases and is capable of producing error type reports resembling the reports composed by human annotators. Our code and data are publicly available at https://github.com/tunde-ajayi/llm-as-a-qualitative-judge.
>
---
#### [new 025] Self-Anchored Attention Model for Sample-Efficient Classification of Prosocial Text Chat
- **分类: cs.CL; cs.AI; cs.CY; I.2.7; K.4**

- **简介: 该论文属于文本分类任务，旨在解决游戏聊天中亲社会行为识别问题。通过提出SAAM模型，在数据稀缺情况下提升分类效果。**

- **链接: [http://arxiv.org/pdf/2506.09259v1](http://arxiv.org/pdf/2506.09259v1)**

> **作者:** Zhuofang Li; Rafal Kocielnik; Fereshteh Soltani; Penphob; Boonyarungsrit; Animashree Anandkumar; R. Michael Alvarez
>
> **摘要:** Millions of players engage daily in competitive online games, communicating through in-game chat. Prior research has focused on detecting relatively small volumes of toxic content using various Natural Language Processing (NLP) techniques for the purpose of moderation. However, recent studies emphasize the importance of detecting prosocial communication, which can be as crucial as identifying toxic interactions. Recognizing prosocial behavior allows for its analysis, rewarding, and promotion. Unlike toxicity, there are limited datasets, models, and resources for identifying prosocial behaviors in game-chat text. In this work, we employed unsupervised discovery combined with game domain expert collaboration to identify and categorize prosocial player behaviors from game chat. We further propose a novel Self-Anchored Attention Model (SAAM) which gives 7.9% improvement compared to the best existing technique. The approach utilizes the entire training set as "anchors" to help improve model performance under the scarcity of training data. This approach led to the development of the first automated system for classifying prosocial behaviors in in-game chats, particularly given the low-resource settings where large-scale labeled data is not available. Our methodology was applied to one of the most popular online gaming titles - Call of Duty(R): Modern Warfare(R)II, showcasing its effectiveness. This research is novel in applying NLP techniques to discover and classify prosocial behaviors in player in-game chat communication. It can help shift the focus of moderation from solely penalizing toxicity to actively encouraging positive interactions on online platforms.
>
---
#### [new 026] The Emergence of Abstract Thought in Large Language Models Beyond Any Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型是否具备超越特定语言的抽象思维能力。通过分析参数空间，发现模型发展出共享神经元支持跨语言通用性，提出针对性训练策略。**

- **链接: [http://arxiv.org/pdf/2506.09890v1](http://arxiv.org/pdf/2506.09890v1)**

> **作者:** Yuxin Chen; Yiran Zhao; Yang Zhang; An Zhang; Kenji Kawaguchi; Shafiq Joty; Junnan Li; Tat-Seng Chua; Michael Qizhe Shieh; Wenxuan Zhang
>
> **摘要:** As large language models (LLMs) continue to advance, their capacity to function effectively across a diverse range of languages has shown marked improvement. Preliminary studies observe that the hidden activations of LLMs often resemble English, even when responding to non-English prompts. This has led to the widespread assumption that LLMs may "think" in English. However, more recent results showing strong multilingual performance, even surpassing English performance on specific tasks in other languages, challenge this view. In this work, we find that LLMs progressively develop a core language-agnostic parameter space-a remarkably small subset of parameters whose deactivation results in significant performance degradation across all languages. This compact yet critical set of parameters underlies the model's ability to generalize beyond individual languages, supporting the emergence of abstract thought that is not tied to any specific linguistic system. Specifically, we identify language-related neurons-those are consistently activated during the processing of particular languages, and categorize them as either shared (active across multiple languages) or exclusive (specific to one). As LLMs undergo continued development over time, we observe a marked increase in both the proportion and functional importance of shared neurons, while exclusive neurons progressively diminish in influence. These shared neurons constitute the backbone of the core language-agnostic parameter space, supporting the emergence of abstract thought. Motivated by these insights, we propose neuron-specific training strategies tailored to LLMs' language-agnostic levels at different development stages. Experiments across diverse LLM families support our approach.
>
---
#### [new 027] Extrapolation by Association: Length Generalization Transfer in Transformers
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究Transformer模型在不同任务间的长度泛化迁移问题，旨在提升模型对长输入的推理能力。通过实验验证了任务关联对泛化效果的影响。**

- **链接: [http://arxiv.org/pdf/2506.09251v1](http://arxiv.org/pdf/2506.09251v1)**

> **作者:** Ziyang Cai; Nayoung Lee; Avi Schwarzschild; Samet Oymak; Dimitris Papailiopoulos
>
> **备注:** 23 pages, 20 figures
>
> **摘要:** Transformer language models have demonstrated impressive generalization capabilities in natural language domains, yet we lack a fine-grained understanding of how such generalization arises. In this paper, we investigate length generalization--the ability to extrapolate from shorter to longer inputs--through the lens of \textit{task association}. We find that length generalization can be \textit{transferred} across related tasks. That is, training a model with a longer and related auxiliary task can lead it to generalize to unseen and longer inputs from some other target task. We demonstrate this length generalization transfer across diverse algorithmic tasks, including arithmetic operations, string transformations, and maze navigation. Our results show that transformer models can inherit generalization capabilities from similar tasks when trained jointly. Moreover, we observe similar transfer effects in pretrained language models, suggesting that pretraining equips models with reusable computational scaffolding that facilitates extrapolation in downstream settings. Finally, we provide initial mechanistic evidence that length generalization transfer correlates with the re-use of the same attention heads between the tasks. Together, our findings deepen our understanding of how transformers generalize to out-of-distribution inputs and highlight the compositional reuse of inductive structure across tasks.
>
---
#### [new 028] COGENT: A Curriculum-oriented Framework for Generating Grade-appropriate Educational Content
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于教育内容生成任务，旨在解决生成符合课程标准和年级水平的教育内容问题。提出COGENT框架，整合课程要素并控制可读性，提升内容质量与学生兴趣。**

- **链接: [http://arxiv.org/pdf/2506.09367v1](http://arxiv.org/pdf/2506.09367v1)**

> **作者:** Zhengyuan Liu; Stella Xin Yin; Dion Hoe-Lian Goh; Nancy F. Chen
>
> **备注:** BEA 2025
>
> **摘要:** While Generative AI has demonstrated strong potential and versatility in content generation, its application to educational contexts presents several challenges. Models often fail to align with curriculum standards and maintain grade-appropriate reading levels consistently. Furthermore, STEM education poses additional challenges in balancing scientific explanations with everyday language when introducing complex and abstract ideas and phenomena to younger students. In this work, we propose COGENT, a curriculum-oriented framework for generating grade-appropriate educational content. We incorporate three curriculum components (science concepts, core ideas, and learning objectives), control readability through length, vocabulary, and sentence complexity, and adopt a ``wonder-based'' approach to increase student engagement and interest. We conduct a multi-dimensional evaluation via both LLM-as-a-judge and human expert analysis. Experimental results show that COGENT consistently produces grade-appropriate passages that are comparable or superior to human references. Our work establishes a viable approach for scaling adaptive and high-quality learning resources.
>
---
#### [new 029] OmniDRCA: Parallel Speech-Text Foundation Model via Dual-Resolution Speech Representations and Contrastive Alignment
- **分类: cs.CL**

- **简介: 该论文提出OmniDRCA，属于语音-文本联合建模任务，解决语音与文本生成同步问题，通过双分辨率表示和对比对齐实现并行处理。**

- **链接: [http://arxiv.org/pdf/2506.09349v1](http://arxiv.org/pdf/2506.09349v1)**

> **作者:** Chao-Hong Tan; Qian Chen; Wen Wang; Chong Deng; Qinglin Zhang; Luyao Cheng; Hai Yu; Xin Zhang; Xiang Lv; Tianyu Zhao; Chong Zhang; Yukun Ma; Yafeng Chen; Hui Wang; Jiaqing Liu; Jieping Ye
>
> **摘要:** Recent studies on end-to-end speech generation with large language models (LLMs) have attracted significant community attention, with multiple works extending text-based LLMs to generate discrete speech tokens. Existing approaches primarily fall into two categories: (1) Methods that generate discrete speech tokens independently without incorporating them into the LLM's autoregressive process, resulting in text generation being unaware of concurrent speech synthesis. (2) Models that generate interleaved or parallel speech-text tokens through joint autoregressive modeling, enabling mutual modality awareness during generation. This paper presents OmniDRCA, a parallel speech-text foundation model based on joint autoregressive modeling, featuring dual-resolution speech representations and contrastive cross-modal alignment. Our approach processes speech and text representations in parallel while enhancing audio comprehension through contrastive alignment. Experimental results on Spoken Question Answering benchmarks demonstrate that OmniDRCA establishes new state-of-the-art (SOTA) performance among parallel joint speech-text modeling based foundation models, and achieves competitive performance compared to interleaved models. Additionally, we explore the potential of extending the framework to full-duplex conversational scenarios.
>
---
#### [new 030] DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决MoE模型重建中专家冗余问题。通过引入多样性增强方法DIVE，提升重建效率与效果。**

- **链接: [http://arxiv.org/pdf/2506.09351v1](http://arxiv.org/pdf/2506.09351v1)**

> **作者:** Yuchen Feng; Bowen Shen; Naibin Gu; Jiaxuan Zhao; Peng Fu; Zheng Lin; Weiping Wang
>
> **备注:** ACL 2025
>
> **摘要:** Large language models (LLMs) with the Mixture-of-Experts (MoE) architecture achieve high cost-efficiency by selectively activating a subset of the parameters. Despite the inference efficiency of MoE LLMs, the training of extensive experts from scratch incurs substantial overhead, whereas reconstructing a dense LLM into an MoE LLM significantly reduces the training budget. However, existing reconstruction methods often overlook the diversity among experts, leading to potential redundancy. In this paper, we come up with the observation that a specific LLM exhibits notable diversity after being pruned on different calibration datasets, based on which we present a Diversity-Enhanced reconstruction method named DIVE. The recipe of DIVE includes domain affinity mining, pruning-based expert reconstruction, and efficient retraining. Specifically, the reconstruction includes pruning and reassembly of the feed-forward network (FFN) module. After reconstruction, we efficiently retrain the model on routers, experts and normalization modules. We implement DIVE on Llama-style LLMs with open-source training corpora. Experiments show that DIVE achieves training efficiency with minimal accuracy trade-offs, outperforming existing pruning and MoE reconstruction methods with the same number of activated parameters.
>
---
#### [new 031] MEDUSA: A Multimodal Deep Fusion Multi-Stage Training Framework for Speech Emotion Recognition in Naturalistic Conditions
- **分类: cs.CL**

- **简介: 该论文属于语音情感识别任务，解决自然环境下情感识别的挑战，提出MEDUSA框架通过多模态融合和多阶段训练提升性能。**

- **链接: [http://arxiv.org/pdf/2506.09556v1](http://arxiv.org/pdf/2506.09556v1)**

> **作者:** Georgios Chatzichristodoulou; Despoina Kosmopoulou; Antonios Kritikos; Anastasia Poulopoulou; Efthymios Georgiou; Athanasios Katsamanis; Vassilis Katsouros; Alexandros Potamianos
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** SER is a challenging task due to the subjective nature of human emotions and their uneven representation under naturalistic conditions. We propose MEDUSA, a multimodal framework with a four-stage training pipeline, which effectively handles class imbalance and emotion ambiguity. The first two stages train an ensemble of classifiers that utilize DeepSER, a novel extension of a deep cross-modal transformer fusion mechanism from pretrained self-supervised acoustic and linguistic representations. Manifold MixUp is employed for further regularization. The last two stages optimize a trainable meta-classifier that combines the ensemble predictions. Our training approach incorporates human annotation scores as soft targets, coupled with balanced data sampling and multitask learning. MEDUSA ranked 1st in Task 1: Categorical Emotion Recognition in the Interspeech 2025: Speech Emotion Recognition in Naturalistic Conditions Challenge.
>
---
#### [new 032] Benchmarking Debiasing Methods for LLM-based Parameter Estimates
- **分类: cs.CL**

- **简介: 该论文属于参数估计任务，旨在解决LLM标注偏差问题。通过比较DSL和PPI方法，分析其在有限样本下的表现与效率。**

- **链接: [http://arxiv.org/pdf/2506.09627v1](http://arxiv.org/pdf/2506.09627v1)**

> **作者:** Nicolas Audinet de Pieuchon; Adel Daoud; Connor T. Jerzak; Moa Johansson; Richard Johansson
>
> **摘要:** Large language models (LLMs) offer an inexpensive yet powerful way to annotate text, but are often inconsistent when compared with experts. These errors can bias downstream estimates of population parameters such as regression coefficients and causal effects. To mitigate this bias, researchers have developed debiasing methods such as Design-based Supervised Learning (DSL) and Prediction-Powered Inference (PPI), which promise valid estimation by combining LLM annotations with a limited number of expensive expert annotations. Although these methods produce consistent estimates under theoretical assumptions, it is unknown how they compare in finite samples of sizes encountered in applied research. We make two contributions: First, we study how each method's performance scales with the number of expert annotations, highlighting regimes where LLM bias or limited expert labels significantly affect results. Second, we compare DSL and PPI across a range of tasks, finding that although both achieve low bias with large datasets, DSL often outperforms PPI on bias reduction and empirical efficiency, but its performance is less consistent across datasets. Our findings indicate that there is a bias-variance tradeoff at the level of debiasing methods, calling for more research on developing metrics for quantifying their efficiency in finite samples.
>
---
#### [new 033] Learning Efficient and Generalizable Graph Retriever for Knowledge-Graph Question Answering
- **分类: cs.CL; cs.IR; cs.LG; I.2.6**

- **简介: 该论文属于知识图谱问答任务，旨在解决图检索的效率与泛化能力问题。提出RAPL框架，通过三方面改进提升检索效果。**

- **链接: [http://arxiv.org/pdf/2506.09645v1](http://arxiv.org/pdf/2506.09645v1)**

> **作者:** Tianjun Yao; Haoxuan Li; Zhiqiang Shen; Pan Li; Tongliang Liu; Kun Zhang
>
> **备注:** 32 pages, 28 figures
>
> **摘要:** Large Language Models (LLMs) have shown strong inductive reasoning ability across various domains, but their reliability is hindered by the outdated knowledge and hallucinations. Retrieval-Augmented Generation mitigates these issues by grounding LLMs with external knowledge; however, most existing RAG pipelines rely on unstructured text, limiting interpretability and structured reasoning. Knowledge graphs, which represent facts as relational triples, offer a more structured and compact alternative. Recent studies have explored integrating knowledge graphs with LLMs for knowledge graph question answering (KGQA), with a significant proportion adopting the retrieve-then-reasoning paradigm. In this framework, graph-based retrievers have demonstrated strong empirical performance, yet they still face challenges in generalization ability. In this work, we propose RAPL, a novel framework for efficient and effective graph retrieval in KGQA. RAPL addresses these limitations through three aspects: (1) a two-stage labeling strategy that combines heuristic signals with parametric models to provide causally grounded supervision; (2) a model-agnostic graph transformation approach to capture both intra- and inter-triple interactions, thereby enhancing representational capacity; and (3) a path-based reasoning strategy that facilitates learning from the injected rational knowledge, and supports downstream reasoner through structured inputs. Empirically, RAPL outperforms state-of-the-art methods by $2.66\%-20.34\%$, and significantly reduces the performance gap between smaller and more powerful LLM-based reasoners, as well as the gap under cross-dataset settings, highlighting its superior retrieval capability and generalizability. Codes are available at: https://github.com/tianyao-aka/RAPL.
>
---
#### [new 034] Comparing human and LLM politeness strategies in free production
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的礼貌策略研究，旨在比较人类与大模型在自由生成中的礼貌策略。通过分析任务发现，大模型虽能模仿人类策略，但过度使用负面策略可能引发误解。**

- **链接: [http://arxiv.org/pdf/2506.09391v1](http://arxiv.org/pdf/2506.09391v1)**

> **作者:** Haoran Zhao; Robert D. Hawkins
>
> **备注:** 25 pages, 5 figures
>
> **摘要:** Polite speech poses a fundamental alignment challenge for large language models (LLMs). Humans deploy a rich repertoire of linguistic strategies to balance informational and social goals -- from positive approaches that build rapport (compliments, expressions of interest) to negative strategies that minimize imposition (hedging, indirectness). We investigate whether LLMs employ a similarly context-sensitive repertoire by comparing human and LLM responses in both constrained and open-ended production tasks. We find that larger models ($\ge$70B parameters) successfully replicate key preferences from the computational pragmatics literature, and human evaluators surprisingly prefer LLM-generated responses in open-ended contexts. However, further linguistic analyses reveal that models disproportionately rely on negative politeness strategies even in positive contexts, potentially leading to misinterpretations. While modern LLMs demonstrate an impressive handle on politeness strategies, these subtle differences raise important questions about pragmatic alignment in AI systems.
>
---
#### [new 035] Causal Sufficiency and Necessity Improves Chain-of-Thought Reasoning
- **分类: cs.CL; cs.AI; math.ST; stat.ME; stat.TH**

- **简介: 该论文属于自然语言处理中的推理任务，旨在解决链式思维（CoT）中的充分性和必要性问题，通过因果框架提升推理效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.09853v1](http://arxiv.org/pdf/2506.09853v1)**

> **作者:** Xiangning Yu; Zhuohan Wang; Linyi Yang; Haoxuan Li; Anjie Liu; Xiao Xue; Jun Wang; Mengyue Yang
>
> **摘要:** Chain-of-Thought (CoT) prompting plays an indispensable role in endowing large language models (LLMs) with complex reasoning capabilities. However, CoT currently faces two fundamental challenges: (1) Sufficiency, which ensures that the generated intermediate inference steps comprehensively cover and substantiate the final conclusion; and (2) Necessity, which identifies the inference steps that are truly indispensable for the soundness of the resulting answer. We propose a causal framework that characterizes CoT reasoning through the dual lenses of sufficiency and necessity. Incorporating causal Probability of Sufficiency and Necessity allows us not only to determine which steps are logically sufficient or necessary to the prediction outcome, but also to quantify their actual influence on the final reasoning outcome under different intervention scenarios, thereby enabling the automated addition of missing steps and the pruning of redundant ones. Extensive experimental results on various mathematical and commonsense reasoning benchmarks confirm substantial improvements in reasoning efficiency and reduced token usage without sacrificing accuracy. Our work provides a promising direction for improving LLM reasoning performance and cost-effectiveness.
>
---
#### [new 036] $(RSA)^2$: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言理解任务，旨在解决非字面语言（如讽刺）的解释问题。提出$(RSA)^2$框架，通过考虑修辞策略建模非字面表达，无需建模说话者动机。**

- **链接: [http://arxiv.org/pdf/2506.09301v1](http://arxiv.org/pdf/2506.09301v1)**

> **作者:** Cesare Spinoso-Di Piano; David Austin; Pablo Piantanida; Jackie Chi Kit Cheung
>
> **备注:** Accepted to ACL 2025 (Main Conference)
>
> **摘要:** Figurative language (e.g., irony, hyperbole, understatement) is ubiquitous in human communication, resulting in utterances where the literal and the intended meanings do not match. The Rational Speech Act (RSA) framework, which explicitly models speaker intentions, is the most widespread theory of probabilistic pragmatics, but existing implementations are either unable to account for figurative expressions or require modeling the implicit motivations for using figurative language (e.g., to express joy or annoyance) in a setting-specific way. In this paper, we introduce the Rhetorical-Strategy-Aware RSA $(RSA)^2$ framework which models figurative language use by considering a speaker's employed rhetorical strategy. We show that $(RSA)^2$ enables human-compatible interpretations of non-literal utterances without modeling a speaker's motivations for being non-literal. Combined with LLMs, it achieves state-of-the-art performance on the ironic split of PragMega+, a new irony interpretation dataset introduced in this study.
>
---
#### [new 037] Latent Multi-Head Attention for Small Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升小语言模型的效率与质量。通过改进多头注意力机制，提出MLA+RoPE方法，在减少内存占用的同时保持模型性能。**

- **链接: [http://arxiv.org/pdf/2506.09342v1](http://arxiv.org/pdf/2506.09342v1)**

> **作者:** Sushant Mehta; Raj Dandekar; Rajat Dandekar; Sreedath Panat
>
> **备注:** 6 pages, 1 figure. 5 tables
>
> **摘要:** We present the first comprehensive study of latent multi-head attention (MLA) for small language models, revealing interesting efficiency-quality trade-offs. Training 30M-parameter GPT models on 100,000 synthetic stories, we benchmark three architectural variants: standard multi-head attention (MHA), MLA, and MLA with rotary positional embeddings (MLA+RoPE). Our key finding is that MLA+RoPE with half-rank latent dimensions (r = d/2) achieves a 45% KV-cache memory reduction while incurring only a 0.3% increase in validation loss (essentially matching MHA quality)- a Pareto improvement for memory constrained deployment. We further show that RoPE is crucial for MLA in small models: without it, MLA underperforms vanilla attention by 3-5%, but with RoPE, it surpasses vanilla by 2%. Inference benchmarks on NVIDIA A100 GPUs reveal that MLA with r=d/2 achieves a 1.4 times speedup over full-rank MLA while maintaining the memory savings. GPT-4 evaluations corroborate perplexity results, with ours achieving the highest quality scores (7.4/10) across grammar, creativity, and consistency metrics. Code and models will be released upon acceptance.
>
---
#### [new 038] Using Sign Language Production as Data Augmentation to enhance Sign Language Translation
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于手语翻译任务，旨在解决手语数据稀缺问题。通过生成手语视频增强数据集，提升翻译模型性能。**

- **链接: [http://arxiv.org/pdf/2506.09643v1](http://arxiv.org/pdf/2506.09643v1)**

> **作者:** Harry Walsh; Maksym Ivashechkin; Richard Bowden
>
> **摘要:** Machine learning models fundamentally rely on large quantities of high-quality data. Collecting the necessary data for these models can be challenging due to cost, scarcity, and privacy restrictions. Signed languages are visual languages used by the deaf community and are considered low-resource languages. Sign language datasets are often orders of magnitude smaller than their spoken language counterparts. Sign Language Production is the task of generating sign language videos from spoken language sentences, while Sign Language Translation is the reverse translation task. Here, we propose leveraging recent advancements in Sign Language Production to augment existing sign language datasets and enhance the performance of Sign Language Translation models. For this, we utilize three techniques: a skeleton-based approach to production, sign stitching, and two photo-realistic generative models, SignGAN and SignSplat. We evaluate the effectiveness of these techniques in enhancing the performance of Sign Language Translation models by generating variation in the signer's appearance and the motion of the skeletal data. Our results demonstrate that the proposed methods can effectively augment existing datasets and enhance the performance of Sign Language Translation models by up to 19%, paving the way for more robust and accurate Sign Language Translation systems, even in resource-constrained environments.
>
---
#### [new 039] UniToMBench: Integrating Perspective-Taking to Improve Theory of Mind in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于理论心理建模任务，旨在提升大语言模型的心智理论能力。通过构建UniToMBench基准，整合多交互任务与演化故事场景，评估并改进模型对人类心理状态的理解。**

- **链接: [http://arxiv.org/pdf/2506.09450v1](http://arxiv.org/pdf/2506.09450v1)**

> **作者:** Prameshwar Thiyagarajan; Vaishnavi Parimi; Shamant Sai; Soumil Garg; Zhangir Meirbek; Nitin Yarlagadda; Kevin Zhu; Chris Kim
>
> **备注:** Accepted at Conference of the North American Chapter of the Association for Computational Linguistics, Student Research Workshop 2025 (NAACL SRW 2025)
>
> **摘要:** Theory of Mind (ToM), the ability to understand the mental states of oneself and others, remains a challenging area for large language models (LLMs), which often fail to predict human mental states accurately. In this paper, we introduce UniToMBench, a unified benchmark that integrates the strengths of SimToM and TOMBENCH to systematically improve and assess ToM capabilities in LLMs by integrating multi-interaction task designs and evolving story scenarios. Supported by a custom dataset of over 1,000 hand-written scenarios, UniToMBench combines perspective-taking techniques with diverse evaluation metrics to better stimulate social cognition in LLMs. Through evaluation, we observe that while models like GPT-4o and GPT-4o Mini show consistently high accuracy in tasks involving emotional and belief-related scenarios, with results usually above 80%, there is significant variability in their performance across knowledge-based tasks. These results highlight both the strengths and limitations of current LLMs in ToM-related tasks, underscoring the value of UniToMBench as a comprehensive tool for future development. Our code is publicly available here: https://github.com/Shamant/unifiedtombenchmark.
>
---
#### [new 040] Is Fine-Tuning an Effective Solution? Reassessing Knowledge Editing for Unstructured Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识编辑任务，解决UKE中局部性评估不足和微调方法异常失效的问题。通过构建数据集并分析影响因素，提出优化的微调方法FT-UKE。**

- **链接: [http://arxiv.org/pdf/2506.09672v1](http://arxiv.org/pdf/2506.09672v1)**

> **作者:** Hao Xiong; Chuanyuan Tan; Wenliang Chen
>
> **摘要:** Unstructured Knowledge Editing (UKE) is crucial for updating the relevant knowledge of large language models (LLMs). It focuses on unstructured inputs, such as long or free-form texts, which are common forms of real-world knowledge. Although previous studies have proposed effective methods and tested them, some issues exist: (1) Lack of Locality evaluation for UKE, and (2) Abnormal failure of fine-tuning (FT) based methods for UKE. To address these issues, we first construct two datasets, UnKEBench-Loc and AKEW-Loc (CF), by extending two existing UKE datasets with locality test data from the unstructured and structured views. This enables a systematic evaluation of the Locality of post-edited models. Furthermore, we identify four factors that may affect the performance of FT-based methods. Based on these factors, we conduct experiments to determine how the well-performing FT-based methods should be trained for the UKE task, providing a training recipe for future research. Our experimental results indicate that the FT-based method with the optimal setting (FT-UKE) is surprisingly strong, outperforming the existing state-of-the-art (SOTA). In batch editing scenarios, FT-UKE shows strong performance as well, with its advantage over SOTA methods increasing as the batch size grows, expanding the average metric lead from +6.78% to +10.80%
>
---
#### [new 041] Dataset of News Articles with Provenance Metadata for Media Relevance Assessment
- **分类: cs.CL; cs.AI; cs.CV; cs.CY**

- **简介: 该论文针对媒体真实性评估任务，解决图像与文本不一致的虚假信息问题。构建了带来源元数据的新闻数据集，提出两个相关性任务并测试大语言模型表现。**

- **链接: [http://arxiv.org/pdf/2506.09847v1](http://arxiv.org/pdf/2506.09847v1)**

> **作者:** Tomas Peterka; Matyas Bohacek
>
> **摘要:** Out-of-context and misattributed imagery is the leading form of media manipulation in today's misinformation and disinformation landscape. The existing methods attempting to detect this practice often only consider whether the semantics of the imagery corresponds to the text narrative, missing manipulation so long as the depicted objects or scenes somewhat correspond to the narrative at hand. To tackle this, we introduce News Media Provenance Dataset, a dataset of news articles with provenance-tagged images. We formulate two tasks on this dataset, location of origin relevance (LOR) and date and time of origin relevance (DTOR), and present baseline results on six large language models (LLMs). We identify that, while the zero-shot performance on LOR is promising, the performance on DTOR hinders, leaving room for specialized architectures and future work.
>
---
#### [new 042] Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking
- **分类: cs.CL**

- **简介: 该论文属于长文本推理任务，旨在提升长上下文语言模型的检索能力。通过引入QRHEAD和QR-RETRIEVER，增强信息检索与重排序效果，显著提升多跳推理和零样本性能。**

- **链接: [http://arxiv.org/pdf/2506.09944v1](http://arxiv.org/pdf/2506.09944v1)**

> **作者:** Wuwei Zhang; Fangcong Yin; Howard Yen; Danqi Chen; Xi Ye
>
> **摘要:** Recent work has identified retrieval heads (Wu et al., 2025b), a subset of attention heads responsible for retrieving salient information in long-context language models (LMs), as measured by their copy-paste behavior in Needle-in-a-Haystack tasks. In this paper, we introduce QRHEAD (Query-Focused Retrieval Head), an improved set of attention heads that enhance retrieval from long context. We identify QRHEAD by aggregating attention scores with respect to the input query, using a handful of examples from real-world tasks (e.g., long-context QA). We further introduce QR- RETRIEVER, an efficient and effective retriever that uses the accumulated attention mass of QRHEAD as retrieval scores. We use QR- RETRIEVER for long-context reasoning by selecting the most relevant parts with the highest retrieval scores. On multi-hop reasoning tasks LongMemEval and CLIPPER, this yields over 10% performance gains over full context and outperforms strong dense retrievers. We also evaluate QRRETRIEVER as a re-ranker on the BEIR benchmark and find that it achieves strong zero-shot performance, outperforming other LLM-based re-rankers such as RankGPT. Further analysis shows that both the querycontext attention scoring and task selection are crucial for identifying QRHEAD with strong downstream utility. Overall, our work contributes a general-purpose retriever and offers interpretability insights into the long-context capabilities of LMs.
>
---
#### [new 043] Bridging the Gap Between Open-Source and Proprietary LLMs in Table QA
- **分类: cs.CL**

- **简介: 该论文属于表格问答任务，旨在提升开源模型在表格QA中的表现，通过集成多种模块实现与专有模型相当的性能。**

- **链接: [http://arxiv.org/pdf/2506.09657v1](http://arxiv.org/pdf/2506.09657v1)**

> **作者:** Nikolas Evkarpidi; Elena Tutubalina
>
> **备注:** Accepted for publication at the 19th International Workshop on Semantic Evaluation (SemEval-2025), to be held in conjunction with ACL 2025. 15 pages, 5 figures
>
> **摘要:** This paper presents a system developed for SemEval 2025 Task 8: Question Answering (QA) over tabular data. Our approach integrates several key components: text-to-SQL and text-to-code generation modules, a self-correction mechanism, and a retrieval-augmented generation (RAG). Additionally, it includes an end-to-end (E2E) module, all orchestrated by a large language model (LLM). Through ablation studies, we analyzed the effects of different parts of our pipeline and identified the challenges that are still present in this field. During the evaluation phase of the competition, our solution achieved an accuracy of 80%, resulting in a top-13 ranking among the 38 participating teams. Our pipeline demonstrates a significant improvement in accuracy for open-source models and achieves a performance comparable to proprietary LLMs in QA tasks over tables. The code is available at GitHub repository.
>
---
#### [new 044] VerIF: Verification Engineering for Reinforcement Learning in Instruction Following
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，解决指令跟随中的验证问题，提出VerIF方法结合规则与大模型验证，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.09942v1](http://arxiv.org/pdf/2506.09942v1)**

> **作者:** Hao Peng; Yunjia Qi; Xiaozhi Wang; Bin Xu; Lei Hou; Juanzi Li
>
> **备注:** 16 pages, 8 figures
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has become a key technique for enhancing large language models (LLMs), with verification engineering playing a central role. However, best practices for RL in instruction following remain underexplored. In this work, we explore the verification challenge in RL for instruction following and propose VerIF, a verification method that combines rule-based code verification with LLM-based verification from a large reasoning model (e.g., QwQ-32B). To support this approach, we construct a high-quality instruction-following dataset, VerInstruct, containing approximately 22,000 instances with associated verification signals. We apply RL training with VerIF to two models, achieving significant improvements across several representative instruction-following benchmarks. The trained models reach state-of-the-art performance among models of comparable size and generalize well to unseen constraints. We further observe that their general capabilities remain unaffected, suggesting that RL with VerIF can be integrated into existing RL recipes to enhance overall model performance. We have released our datasets, codes, and models to facilitate future research at https://github.com/THU-KEG/VerIF.
>
---
#### [new 045] TransXSSM: A Hybrid Transformer State Space Model with Unified Rotary Position Embedding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决Transformer与SSM在位置编码上的不兼容问题，提出统一旋转位置编码方法，构建高效混合模型。**

- **链接: [http://arxiv.org/pdf/2506.09507v1](http://arxiv.org/pdf/2506.09507v1)**

> **作者:** Bingheng Wu; Jingze Shi; Yifan Wu; Nan Tang; Yuyu Luo
>
> **摘要:** Transformers exhibit proficiency in capturing long-range dependencies, whereas State Space Models (SSMs) facilitate linear-time sequence modeling. Notwithstanding their synergistic potential, the integration of these architectures presents a significant challenge, primarily attributable to a fundamental incongruity in their respective positional encoding mechanisms: Transformers rely on explicit Rotary Position Embeddings (RoPE), while SSMs leverage implicit positional representations via convolutions. This divergence often precipitates discontinuities and suboptimal performance. To address this impediment, we propose a unified rotary position embedding (\textbf{\ourRoPE}) methodology, thereby establishing a consistent positional encoding framework for both self-attention and state-space components. Using this \ourRoPE, we introduce \textbf{\model}, a hybrid architecture that coherently integrates the Transformer and SSM layers under this unified positional encoding scheme. At a 4K sequence length, \model exhibits training and inference speeds that are \textbf{42.3\% and 29.5\% faster}, respectively, relative to standard Transformer models. It also delivers higher accuracy: under comparable settings, it surpasses a Transformer baseline by over 4\% on language modeling benchmarks. \model furthermore scales more effectively: \model-1.3B gains \textbf{7.22\%} in average accuracy over its 320M version (versus about 6\% gains for equivalent Transformers or SSMs). Our results show that unified positional encoding resolves positional incompatibility in hybrid models, enabling efficient, high-performance long-context modeling.
>
---
#### [new 046] Hidden in Plain Sight: Evaluation of the Deception Detection Capabilities of LLMs in Multimodal Settings
- **分类: cs.CL**

- **简介: 该论文属于 deception detection 任务，旨在评估 LLMs 和 LMMs 在多模态环境下的说谎检测能力，通过实验分析不同设置和特征的影响。**

- **链接: [http://arxiv.org/pdf/2506.09424v1](http://arxiv.org/pdf/2506.09424v1)**

> **作者:** Md Messal Monem Miah; Adrita Anika; Xi Shi; Ruihong Huang
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Detecting deception in an increasingly digital world is both a critical and challenging task. In this study, we present a comprehensive evaluation of the automated deception detection capabilities of Large Language Models (LLMs) and Large Multimodal Models (LMMs) across diverse domains. We assess the performance of both open-source and commercial LLMs on three distinct datasets: real life trial interviews (RLTD), instructed deception in interpersonal scenarios (MU3D), and deceptive reviews (OpSpam). We systematically analyze the effectiveness of different experimental setups for deception detection, including zero-shot and few-shot approaches with random or similarity-based in-context example selection. Our results show that fine-tuned LLMs achieve state-of-the-art performance on textual deception detection tasks, while LMMs struggle to fully leverage cross-modal cues. Additionally, we analyze the impact of auxiliary features, such as non-verbal gestures and video summaries, and examine the effectiveness of different prompting strategies, including direct label generation and chain-of-thought reasoning. Our findings provide key insights into how LLMs process and interpret deceptive cues across modalities, highlighting their potential and limitations in real-world deception detection applications.
>
---
#### [new 047] Modeling Probabilistic Reduction using Information Theory and Naive Discriminative Learning
- **分类: cs.CL; cs.IT; math.IT**

- **简介: 该论文属于语音处理任务，旨在研究声学词长建模中的概率缩减问题。通过比较信息理论与朴素判别学习模型，发现N-gram模型表现最佳，并提出结合信息理论的改进方法。**

- **链接: [http://arxiv.org/pdf/2506.09641v1](http://arxiv.org/pdf/2506.09641v1)**

> **作者:** Anna Stein; Kevin Tang
>
> **备注:** Submitted to Interspeech 2025
>
> **摘要:** This study compares probabilistic predictors based on information theory with Naive Discriminative Learning (NDL) predictors in modeling acoustic word duration, focusing on probabilistic reduction. We examine three models using the Buckeye corpus: one with NDL-derived predictors using information-theoretic formulas, one with traditional NDL predictors, and one with N-gram probabilistic predictors. Results show that the N-gram model outperforms both NDL models, challenging the assumption that NDL is more effective due to its cognitive motivation. However, incorporating information-theoretic formulas into NDL improves model performance over the traditional model. This research highlights a) the need to incorporate not only frequency and contextual predictability but also average contextual predictability, and b) the importance of combining information-theoretic metrics of predictability and information derived from discriminative learning in modeling acoustic reduction.
>
---
#### [new 048] Bridging Online Behavior and Clinical Insight: A Longitudinal LLM-based Study of Suicidality on YouTube Reveals Novel Digital Markers
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于心理健康分析任务，旨在通过YouTube行为识别自杀风险。研究结合计算与临床方法，发现数字标记如YouTube参与度与自杀行为相关。**

- **链接: [http://arxiv.org/pdf/2506.09495v1](http://arxiv.org/pdf/2506.09495v1)**

> **作者:** Ilanit Sobol; Shir Lissak; Refael Tikochinski; Tal Nakash; Anat Brunstein Klomek; Eyal Fruchter; Roi Reichart
>
> **摘要:** Suicide remains a leading cause of death in Western countries, underscoring the need for new research approaches. As social media becomes central to daily life, digital footprints offer valuable insight into suicidal behavior. Focusing on individuals who attempted suicide while uploading videos to their channels, we investigate: How do suicidal behaviors manifest on YouTube, and how do they differ from expert knowledge? We applied complementary approaches: computational bottom-up, hybrid, and expert-driven top-down, on a novel longitudinal dataset of 181 YouTube channels from individuals with life-threatening attempts, alongside 134 control channels. In the bottom-up approach, we applied LLM-based topic modeling to identify behavioral indicators. Of 166 topics, five were associated with suicide-attempt, with two also showing temporal attempt-related changes ($p<.01$) - Mental Health Struggles ($+0.08$)* and YouTube Engagement ($+0.1$)*. In the hybrid approach, a clinical expert reviewed LLM-derived topics and flagged 19 as suicide-related. However, none showed significant attempt-related temporal effects beyond those identified bottom-up. Notably, YouTube Engagement, a platform-specific indicator, was not flagged by the expert, underscoring the value of bottom-up discovery. In the top-down approach, psychological assessment of suicide attempt narratives revealed that the only significant difference between individuals who attempted before and those attempted during their upload period was the motivation to share this experience: the former aimed to Help Others ($\beta=-1.69$, $p<.01$), while the latter framed it as part of their Personal Recovery ($\beta=1.08$, $p<.01$). By integrating these approaches, we offer a nuanced understanding of suicidality, bridging digital behavior and clinical insights. * Within-group changes in relation to the suicide attempt.
>
---
#### [new 049] CoRT: Code-integrated Reasoning within Thinking
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大型推理模型在数学运算中的效率与准确性问题。通过引入CoRT框架和Hint-Engineering方法提升模型与代码解释器的协作效果。**

- **链接: [http://arxiv.org/pdf/2506.09820v1](http://arxiv.org/pdf/2506.09820v1)**

> **作者:** Chengpeng Li; Zhengyang Tang; Ziniu Li; Mingfeng Xue; Keqin Bao; Tian Ding; Ruoyu Sun; Benyou Wang; Xiang Wang; Junyang Lin; Dayiheng Liu
>
> **备注:** work in progress
>
> **摘要:** Large Reasoning Models (LRMs) like o1 and DeepSeek-R1 have shown remarkable progress in natural language reasoning with long chain-of-thought (CoT), yet they remain inefficient or inaccurate when handling complex mathematical operations. Addressing these limitations through computational tools (e.g., computation libraries and symbolic solvers) is promising, but it introduces a technical challenge: Code Interpreter (CI) brings external knowledge beyond the model's internal text representations, thus the direct combination is not efficient. This paper introduces CoRT, a post-training framework for teaching LRMs to leverage CI effectively and efficiently. As a first step, we address the data scarcity issue by synthesizing code-integrated reasoning data through Hint-Engineering, which strategically inserts different hints at appropriate positions to optimize LRM-CI interaction. We manually create 30 high-quality samples, upon which we post-train models ranging from 1.5B to 32B parameters, with supervised fine-tuning, rejection fine-tuning and reinforcement learning. Our experimental results demonstrate that Hint-Engineering models achieve 4\% and 8\% absolute improvements on DeepSeek-R1-Distill-Qwen-32B and DeepSeek-R1-Distill-Qwen-1.5B respectively, across five challenging mathematical reasoning datasets. Furthermore, Hint-Engineering models use about 30\% fewer tokens for the 32B model and 50\% fewer tokens for the 1.5B model compared with the natural language models. The models and code are available at https://github.com/ChengpengLi1003/CoRT.
>
---
#### [new 050] Query-Level Uncertainty in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决模型识别已知与未知查询的问题。通过提出“内部置信度”方法，检测查询不确定性，提升模型的自适应推理能力。**

- **链接: [http://arxiv.org/pdf/2506.09669v1](http://arxiv.org/pdf/2506.09669v1)**

> **作者:** Lihu Chen; Gaël Varoquaux
>
> **备注:** In Progress
>
> **摘要:** It is important for Large Language Models to be aware of the boundary of their knowledge, the mechanism of identifying known and unknown queries. This type of awareness can help models perform adaptive inference, such as invoking RAG, engaging in slow and deep thinking, or adopting the abstention mechanism, which is beneficial to the development of efficient and trustworthy AI. In this work, we propose a method to detect knowledge boundaries via Query-Level Uncertainty, which aims to determine if the model is able to address a given query without generating any tokens. To this end, we introduce a novel and training-free method called \emph{Internal Confidence}, which leverages self-evaluations across layers and tokens. Empirical results on both factual QA and mathematical reasoning tasks demonstrate that our internal confidence can outperform several baselines. Furthermore, we showcase that our proposed method can be used for efficient RAG and model cascading, which is able to reduce inference costs while maintaining performance.
>
---
#### [new 051] Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM推理可复现性问题。研究发现数值精度影响模型输出稳定性，提出LayerCast方法平衡精度与效率。**

- **链接: [http://arxiv.org/pdf/2506.09501v1](http://arxiv.org/pdf/2506.09501v1)**

> **作者:** Jiayi Yuan; Hao Li; Xinheng Ding; Wenya Xie; Yu-Jhe Li; Wentian Zhao; Kun Wan; Jing Shi; Xia Hu; Zirui Liu
>
> **摘要:** Large Language Models (LLMs) are now integral across various domains and have demonstrated impressive performance. Progress, however, rests on the premise that benchmark scores are both accurate and reproducible. We demonstrate that the reproducibility of LLM performance is fragile: changing system configuration such as evaluation batch size, GPU count, and GPU version can introduce significant difference in the generated responses. This issue is especially pronounced in reasoning models, where minor rounding differences in early tokens can cascade into divergent chains of thought, ultimately affecting accuracy. For instance, under bfloat16 precision with greedy decoding, a reasoning model like DeepSeek-R1-Distill-Qwen-7B can exhibit up to 9% variation in accuracy and 9,000 tokens difference in response length due to differences in GPU count, type, and evaluation batch size. We trace the root cause of this variability to the non-associative nature of floating-point arithmetic under limited numerical precision. This work presents the first systematic investigation into how numerical precision affects reproducibility in LLM inference. Through carefully controlled experiments across various hardware, software, and precision settings, we quantify when and how model outputs diverge. Our analysis reveals that floating-point precision -- while critical for reproducibility -- is often neglected in evaluation practices. Inspired by this, we develop a lightweight inference pipeline, dubbed LayerCast, that stores weights in 16-bit precision but performs all computations in FP32, balancing memory efficiency with numerical stability. Code is available at https://github.com/nanomaoli/llm_reproducibility.
>
---
#### [new 052] Gender Bias in English-to-Greek Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的性别偏见研究任务，旨在检测并缓解英译希机器翻译中的性别刻板印象。通过构建数据集并测试多个系统，发现现有模型在性别不明确时存在偏见，GPT-4o表现较好但仍有改进空间。**

- **链接: [http://arxiv.org/pdf/2506.09558v1](http://arxiv.org/pdf/2506.09558v1)**

> **作者:** Eleni Gkovedarou; Joke Daems; Luna De Bruyne
>
> **备注:** Accepted at GITT 2025 (MT Summit)
>
> **摘要:** As the demand for inclusive language increases, concern has grown over the susceptibility of machine translation (MT) systems to reinforce gender stereotypes. This study investigates gender bias in two commercial MT systems, Google Translate and DeepL, focusing on the understudied English-to-Greek language pair. We address three aspects of gender bias: i) male bias, ii) occupational stereotyping, and iii) errors in anti-stereotypical translations. Additionally, we explore the potential of prompted GPT-4o as a bias mitigation tool that provides both gender-explicit and gender-neutral alternatives when necessary. To achieve this, we introduce GendEL, a manually crafted bilingual dataset of 240 gender-ambiguous and unambiguous sentences that feature stereotypical occupational nouns and adjectives. We find persistent gender bias in translations by both MT systems; while they perform well in cases where gender is explicitly defined, with DeepL outperforming both Google Translate and GPT-4o in feminine gender-unambiguous sentences, they are far from producing gender-inclusive or neutral translations when the gender is unspecified. GPT-4o shows promise, generating appropriate gendered and neutral alternatives for most ambiguous cases, though residual biases remain evident.
>
---
#### [new 053] Token Constraint Decoding Improves Robustness on Question Answering for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于问答任务，旨在提升大语言模型在噪声输入下的鲁棒性。通过引入TCD算法增强推理稳定性，有效缓解输入扰动带来的性能下降。**

- **链接: [http://arxiv.org/pdf/2506.09408v1](http://arxiv.org/pdf/2506.09408v1)**

> **作者:** Jui-Ming Yao; Hao-Yuan Chen; Zi-Xian Tang; Bing-Jia Tan; Sheng-Wei Peng; Bing-Cheng Xie; Shun-Feng Su
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive performance on multiple-choice question answering (MCQA) benchmarks, yet they remain highly vulnerable to minor input perturbations. In this paper, we introduce and evaluate Token Constraint Decoding (TCD). This simple yet effective inference-time algorithm enforces alignment between token-level predictions to enhance robustness in noisy settings. Through extensive experiments on CommonsenseQA, MMLU, and MMLU-Pro, we show that TCD, especially when paired with prompt engineering (PE) fixes, significantly restores performance degraded by input noise, yielding up to +39\% absolute gains for weaker models like Gemma3 1B. Penalty sweep analyses further reveal that TCD implicitly regularizes overconfident outputs, with different models requiring distinct penalty schedules to maximize resilience. Our findings establish TCD as a practical, model-agnostic approach for improving reasoning stability under real-world imperfections and pave the way for more reliable deployment of LLMs in safety-critical or user-facing applications.
>
---
#### [new 054] A Technique for Isolating Lexically-Independent Phonetic Dependencies in Generative CNNs
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音生成任务，旨在解决DNN如何表征语音规则的问题。通过调整网络结构，研究发现卷积层可独立于全连接层进行语音依赖性泛化。**

- **链接: [http://arxiv.org/pdf/2506.09218v1](http://arxiv.org/pdf/2506.09218v1)**

> **作者:** Bruno Ferenc Šegedin
>
> **摘要:** The ability of deep neural networks (DNNs) to represent phonotactic generalizations derived from lexical learning remains an open question. This study (1) investigates the lexically-invariant generalization capacity of generative convolutional neural networks (CNNs) trained on raw audio waveforms of lexical items and (2) explores the consequences of shrinking the fully-connected layer (FC) bottleneck from 1024 channels to 8 before training. Ultimately, a novel technique for probing a model's lexically-independent generalizations is proposed that works only under the narrow FC bottleneck: generating audio outputs by bypassing the FC and inputting randomized feature maps into the convolutional block. These outputs are equally biased by a phonotactic restriction in training as are outputs generated with the FC. This result shows that the convolutional layers can dynamically generalize phonetic dependencies beyond lexically-constrained configurations learned by the FC.
>
---
#### [new 055] When Detection Fails: The Power of Fine-Tuned Models to Generate Human-Like Social Media Text
- **分类: cs.CL**

- **简介: 该论文属于AI生成文本检测任务，旨在解决社会媒体上AI文本难以检测的问题。通过构建数据集并实验，证明微调模型显著降低检测效果。**

- **链接: [http://arxiv.org/pdf/2506.09975v1](http://arxiv.org/pdf/2506.09975v1)**

> **作者:** Hillary Dawkins; Kathleen C. Fraser; Svetlana Kiritchenko
>
> **备注:** to appear in ACL Findings
>
> **摘要:** Detecting AI-generated text is a difficult problem to begin with; detecting AI-generated text on social media is made even more difficult due to the short text length and informal, idiosyncratic language of the internet. It is nonetheless important to tackle this problem, as social media represents a significant attack vector in online influence campaigns, which may be bolstered through the use of mass-produced AI-generated posts supporting (or opposing) particular policies, decisions, or events. We approach this problem with the mindset and resources of a reasonably sophisticated threat actor, and create a dataset of 505,159 AI-generated social media posts from a combination of open-source, closed-source, and fine-tuned LLMs, covering 11 different controversial topics. We show that while the posts can be detected under typical research assumptions about knowledge of and access to the generating models, under the more realistic assumption that an attacker will not release their fine-tuned model to the public, detectability drops dramatically. This result is confirmed with a human study. Ablation experiments highlight the vulnerability of various detection algorithms to fine-tuned LLMs. This result has implications across all detection domains, since fine-tuning is a generally applicable and realistic LLM use case.
>
---
#### [new 056] A Hierarchical Probabilistic Framework for Incremental Knowledge Tracing in Classroom Settings
- **分类: cs.CL**

- **简介: 该论文属于知识追踪任务，解决低资源环境下在线更新的问题。提出KT²框架，利用层次结构信息提升预测性能。**

- **链接: [http://arxiv.org/pdf/2506.09393v1](http://arxiv.org/pdf/2506.09393v1)**

> **作者:** Xinyi Gao; Qiucheng Wu; Yang Zhang; Xuechen Liu; Kaizhi Qian; Ying Xu; Shiyu Chang
>
> **备注:** 24 pages, 4 figures
>
> **摘要:** Knowledge tracing (KT) aims to estimate a student's evolving knowledge state and predict their performance on new exercises based on performance history. Many realistic classroom settings for KT are typically low-resource in data and require online updates as students' exercise history grows, which creates significant challenges for existing KT approaches. To restore strong performance under low-resource conditions, we revisit the hierarchical knowledge concept (KC) information, which is typically available in many classroom settings and can provide strong prior when data are sparse. We therefore propose Knowledge-Tree-based Knowledge Tracing (KT$^2$), a probabilistic KT framework that models student understanding over a tree-structured hierarchy of knowledge concepts using a Hidden Markov Tree Model. KT$^2$ estimates student mastery via an EM algorithm and supports personalized prediction through an incremental update mechanism as new responses arrive. Our experiments show that KT$^2$ consistently outperforms strong baselines in realistic online, low-resource settings.
>
---
#### [new 057] Alzheimer's Dementia Detection Using Perplexity from Paired Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于阿尔茨海默病检测任务，旨在通过大语言模型的困惑度分析识别AD患者语言特征，提升检测准确率并实现可解释性决策。**

- **链接: [http://arxiv.org/pdf/2506.09315v1](http://arxiv.org/pdf/2506.09315v1)**

> **作者:** Yao Xiao; Heidi Christensen; Stefan Goetze
>
> **备注:** To be published in the proceedings of Interspeech 2025
>
> **摘要:** Alzheimer's dementia (AD) is a neurodegenerative disorder with cognitive decline that commonly impacts language ability. This work extends the paired perplexity approach to detecting AD by using a recent large language model (LLM), the instruction-following version of Mistral-7B. We improve accuracy by an average of 3.33% over the best current paired perplexity method and by 6.35% over the top-ranked method from the ADReSS 2020 challenge benchmark. Our further analysis demonstrates that the proposed approach can effectively detect AD with a clear and interpretable decision boundary in contrast to other methods that suffer from opaque decision-making processes. Finally, by prompting the fine-tuned LLMs and comparing the model-generated responses to human responses, we illustrate that the LLMs have learned the special language patterns of AD speakers, which opens up possibilities for novel methods of model interpretation and data augmentation.
>
---
#### [new 058] PersonaLens: A Benchmark for Personalization Evaluation in Conversational AI Assistants
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于对话AI个性化评估任务，旨在解决任务导向型助手个性化能力的系统性评价问题。提出PersonaLens基准，包含用户画像和评估机制，揭示现有模型的个性化差异。**

- **链接: [http://arxiv.org/pdf/2506.09902v1](http://arxiv.org/pdf/2506.09902v1)**

> **作者:** Zheng Zhao; Clara Vania; Subhradeep Kayal; Naila Khan; Shay B. Cohen; Emine Yilmaz
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Large language models (LLMs) have advanced conversational AI assistants. However, systematically evaluating how well these assistants apply personalization--adapting to individual user preferences while completing tasks--remains challenging. Existing personalization benchmarks focus on chit-chat, non-conversational tasks, or narrow domains, failing to capture the complexities of personalized task-oriented assistance. To address this, we introduce PersonaLens, a comprehensive benchmark for evaluating personalization in task-oriented AI assistants. Our benchmark features diverse user profiles equipped with rich preferences and interaction histories, along with two specialized LLM-based agents: a user agent that engages in realistic task-oriented dialogues with AI assistants, and a judge agent that employs the LLM-as-a-Judge paradigm to assess personalization, response quality, and task success. Through extensive experiments with current LLM assistants across diverse tasks, we reveal significant variability in their personalization capabilities, providing crucial insights for advancing conversational AI systems.
>
---
#### [new 059] From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于语言模型安全任务，旨在解决实时有害输出检测问题。通过构建数据集和提出流式内容监控器，实现高效部分检测。**

- **链接: [http://arxiv.org/pdf/2506.09996v1](http://arxiv.org/pdf/2506.09996v1)**

> **作者:** Yang Li; Qiang Sheng; Yehan Yang; Xueyao Zhang; Juan Cao
>
> **备注:** 22 pages, 7 figures, and 9 tables
>
> **摘要:** Though safety alignment has been applied to most large language models (LLMs), LLM service providers generally deploy a subsequent moderation as the external safety guardrail in real-world products. Existing moderators mainly practice a conventional full detection, which determines the harmfulness based on the complete LLM output, causing high service latency. Recent works pay more attention to partial detection where moderators oversee the generation midway and early stop the output if harmfulness is detected, but they directly apply moderators trained with the full detection paradigm to incomplete outputs, introducing a training-inference gap that lowers the performance. In this paper, we explore how to form a data-and-model solution that natively supports partial detection. For the data, we construct FineHarm, a dataset consisting of 29K prompt-response pairs with fine-grained annotations to provide reasonable supervision for token-level training. Then, we propose the streaming content monitor, which is trained with dual supervision of response- and token-level labels and can follow the output stream of LLM to make a timely judgment of harmfulness. Experiments show that SCM gains 0.95+ in macro F1 score that is comparable to full detection, by only seeing the first 18% of tokens in responses on average. Moreover, the SCM can serve as a pseudo-harmfulness annotator for improving safety alignment and lead to a higher harmlessness score than DPO.
>
---
#### [new 060] Towards Open Foundation Language Model and Corpus for Macedonian: A Low-Resource Language
- **分类: cs.CL**

- **简介: 该论文针对马其顿语等低资源语言的大型语言模型需求，构建了最大语料库、指令数据集和评估套件，并训练出性能优越的8B参数模型。**

- **链接: [http://arxiv.org/pdf/2506.09560v1](http://arxiv.org/pdf/2506.09560v1)**

> **作者:** Stefan Krsteski; Matea Tashkovska; Borjan Sazdov; Hristijan Gjoreski; Branislav Gerazov
>
> **备注:** Camera-ready version accepted at SlavNLP-2025@ACL
>
> **摘要:** The increase in technological adoption worldwide comes with demands for novel tools to be used by the general population. Large Language Models (LLMs) provide a great opportunity in this respect, but their capabilities remain limited for low-resource languages, restricting applications in countries where such languages are spoken. We create several resources to facilitate the adoption of LLMs and to support research advancements for Macedonian. We collect the largest Macedonian corpus to date, consisting of 40GB of textual data and totaling 3.5B words. To support conversational applications, we collect a 106k-instance instruction dataset, carefully built to be culturally grounded. For evaluation, we construct a Macedonian evaluation suite covering seven benchmarks. Finally, we train domestic-yak, a state-of-the-art 8B-parameter model, on our curated datasets and evaluate it against eight baseline models using the newly constructed benchmark suite. Our model outperforms all existing models in the 8B parameter range across all benchmarks, and achieves performance comparable to models up to 10x larger. Furthermore, a qualitative analysis with native speakers reveals that our model is preferred over larger counterparts, receiving higher ratings for grammatical correctness and cultural appropriateness. All datasets, code, and model weights are openly released, setting a foundation for advancing LLMs in similarly underrepresented languages. These resources are publicly available at github.com/LVSTCK for source code, and at huggingface.co/LVSTCK for pretrained model weights and data.
>
---
#### [new 061] Towards Efficient and Effective Alignment of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型对齐任务，旨在提升模型与人类期望的一致性。通过数据收集、训练优化和评估方法的创新，解决模型在零样本推理、知识整合和约束遵循方面的不足。**

- **链接: [http://arxiv.org/pdf/2506.09329v1](http://arxiv.org/pdf/2506.09329v1)**

> **作者:** Yuxin Jiang
>
> **备注:** PhD thesis
>
> **摘要:** Large language models (LLMs) exhibit remarkable capabilities across diverse tasks, yet aligning them efficiently and effectively with human expectations remains a critical challenge. This thesis advances LLM alignment by introducing novel methodologies in data collection, training, and evaluation. We first address alignment data collection. Existing approaches rely heavily on manually curated datasets or proprietary models. To overcome these limitations, we propose Lion, an adversarial distillation framework that iteratively refines training data by identifying and generating challenging instructions, enabling state-of-the-art zero-shot reasoning. Additionally, we introduce Web Reconstruction (WebR), a fully automated framework that synthesizes instruction-tuning data directly from raw web documents, significantly improving data diversity and scalability over existing synthetic data methods. Next, we enhance alignment training through novel optimization techniques. We develop Learning to Edit (LTE), a framework that enables LLMs to efficiently integrate new knowledge while preserving existing information. LTE leverages meta-learning to improve both real-time and batch knowledge updates. Furthermore, we introduce Bridging and Modeling Correlations (BMC), a refinement of Direct Preference Optimization (DPO) that explicitly captures token-level correlations in preference data, leading to superior alignment across QA and mathematical reasoning tasks. Finally, we tackle the challenge of evaluating alignment. Existing benchmarks emphasize response quality but overlook adherence to specific constraints. To bridge this gap, we introduce FollowBench, a multi-level, fine-grained benchmark assessing LLMs' ability to follow complex constraints across diverse instruction types. Our results expose key weaknesses in current models' constraint adherence, offering insights for future improvements.
>
---
#### [new 062] PGDA-KGQA: A Prompt-Guided Generative Framework with Multiple Data Augmentation Strategies for Knowledge Graph Question Answering
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于知识图谱问答任务，旨在解决数据稀缺和多跳推理不足的问题。通过多种数据增强策略生成高质量训练数据，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.09414v1](http://arxiv.org/pdf/2506.09414v1)**

> **作者:** Xiujun Zhou; Pingjian Zhang; Deyou Tang
>
> **备注:** 13 pages, 7 figures, 5 tables
>
> **摘要:** Knowledge Graph Question Answering (KGQA) is a crucial task in natural language processing that requires reasoning over knowledge graphs (KGs) to answer natural language questions. Recent methods utilizing large language models (LLMs) have shown remarkable semantic parsing capabilities but are limited by the scarcity of diverse annotated data and multi-hop reasoning samples. Traditional data augmentation approaches are focus mainly on single-hop questions and prone to semantic distortion, while LLM-based methods primarily address semantic distortion but usually neglect multi-hop reasoning, thus limiting data diversity. The scarcity of multi-hop samples further weakens models' generalization. To address these issues, we propose PGDA-KGQA, a prompt-guided generative framework with multiple data augmentation strategies for KGQA. At its core, PGDA-KGQA employs a unified prompt-design paradigm: by crafting meticulously engineered prompts that integrate the provided textual content, it leverages LLMs to generate large-scale (question, logical form) pairs for model training. Specifically, PGDA-KGQA enriches its training set by: (1) generating single-hop pseudo questions to improve the alignment of question semantics with KG relations; (2) applying semantic-preserving question rewriting to improve robustness against linguistic variations; (3) employing answer-guided reverse path exploration to create realistic multi-hop questions. By adopting an augment-generate-retrieve semantic parsing pipeline, PGDA-KGQA utilizes the augmented data to enhance the accuracy of logical form generation and thus improve answer retrieval performance. Experiments demonstrate that outperforms state-of-the-art methods on standard KGQA datasets, achieving improvements on WebQSP by 2.8%, 1.2%, and 3.1% and on ComplexWebQuestions by 1.8%, 1.1%, and 2.4% in F1, Hits@1, and Accuracy, respectively.
>
---
#### [new 063] CoLMbo: Speaker Language Model for Descriptive Profiling
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出CoLMbo，一种用于生成详细说话人描述的语音语言模型，解决传统系统在提取人口统计特征上的不足。**

- **链接: [http://arxiv.org/pdf/2506.09375v1](http://arxiv.org/pdf/2506.09375v1)**

> **作者:** Massa Baali; Shuo Han; Syed Abdul Hannan; Purusottam Samal; Karanveer Singh; Soham Deshmukh; Rita Singh; Bhiksha Raj
>
> **摘要:** Speaker recognition systems are often limited to classification tasks and struggle to generate detailed speaker characteristics or provide context-rich descriptions. These models primarily extract embeddings for speaker identification but fail to capture demographic attributes such as dialect, gender, and age in a structured manner. This paper introduces CoLMbo, a Speaker Language Model (SLM) that addresses these limitations by integrating a speaker encoder with prompt-based conditioning. This allows for the creation of detailed captions based on speaker embeddings. CoLMbo utilizes user-defined prompts to adapt dynamically to new speaker characteristics and provides customized descriptions, including regional dialect variations and age-related traits. This innovative approach not only enhances traditional speaker profiling but also excels in zero-shot scenarios across diverse datasets, marking a significant advancement in the field of speaker recognition.
>
---
#### [new 064] Effective Red-Teaming of Policy-Adherent Agents
- **分类: cs.MA; cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于安全评估任务，旨在解决政策遵循型智能体抵御恶意用户攻击的问题。提出CRAFT系统和tau-break基准，评估并提升其抗操控能力。**

- **链接: [http://arxiv.org/pdf/2506.09600v1](http://arxiv.org/pdf/2506.09600v1)**

> **作者:** Itay Nakash; George Kour; Koren Lazar; Matan Vetzler; Guy Uziel; Ateret Anaby-Tavor
>
> **摘要:** Task-oriented LLM-based agents are increasingly used in domains with strict policies, such as refund eligibility or cancellation rules. The challenge lies in ensuring that the agent consistently adheres to these rules and policies, appropriately refusing any request that would violate them, while still maintaining a helpful and natural interaction. This calls for the development of tailored design and evaluation methodologies to ensure agent resilience against malicious user behavior. We propose a novel threat model that focuses on adversarial users aiming to exploit policy-adherent agents for personal benefit. To address this, we present CRAFT, a multi-agent red-teaming system that leverages policy-aware persuasive strategies to undermine a policy-adherent agent in a customer-service scenario, outperforming conventional jailbreak methods such as DAN prompts, emotional manipulation, and coercive. Building upon the existing tau-bench benchmark, we introduce tau-break, a complementary benchmark designed to rigorously assess the agent's robustness against manipulative user behavior. Finally, we evaluate several straightforward yet effective defense strategies. While these measures provide some protection, they fall short, highlighting the need for stronger, research-driven safeguards to protect policy-adherent agents from adversarial attacks
>
---
#### [new 065] Ming-Omni: A Unified Multimodal Model for Perception and Generation
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出Ming-Omni，一个统一的多模态模型，解决跨模态感知与生成任务，支持图像、文本、音频和视频处理，无需额外调整。**

- **链接: [http://arxiv.org/pdf/2506.09344v1](http://arxiv.org/pdf/2506.09344v1)**

> **作者:** Inclusion AI; Biao Gong; Cheng Zou; Chuanyang Zheng; Chunluan Zhou; Canxiang Yan; Chunxiang Jin; Chunjie Shen; Dandan Zheng; Fudong Wang; Furong Xu; GuangMing Yao; Jun Zhou; Jingdong Chen; Jianxin Sun; Jiajia Liu; Jianjiang Zhu; Jun Peng; Kaixiang Ji; Kaiyou Song; Kaimeng Ren; Libin Wang; Lixiang Ru; Lele Xie; Longhua Tan; Lyuxin Xue; Lan Wang; Mochen Bai; Ning Gao; Pei Chen; Qingpei Guo; Qinglong Zhang; Qiang Xu; Rui Liu; Ruijie Xiong; Sirui Gao; Tinghao Liu; Taisong Li; Weilong Chai; Xinyu Xiao; Xiaomei Wang; Xiaoxue Chen; Xiao Lu; Xiaoyu Li; Xingning Dong; Xuzheng Yu; Yi Yuan; Yuting Gao; Yunxiao Sun; Yipeng Chen; Yifei Wu; Yongjie Lyu; Ziping Ma; Zipeng Feng; Zhijiang Fang; Zhihao Qiu; Ziyuan Huang; Zhengyu He
>
> **备注:** 18 pages,8 figures
>
> **摘要:** We propose Ming-Omni, a unified multimodal model capable of processing images, text, audio, and video, while demonstrating strong proficiency in both speech and image generation. Ming-Omni employs dedicated encoders to extract tokens from different modalities, which are then processed by Ling, an MoE architecture equipped with newly proposed modality-specific routers. This design enables a single model to efficiently process and fuse multimodal inputs within a unified framework, thereby facilitating diverse tasks without requiring separate models, task-specific fine-tuning, or structural redesign. Importantly, Ming-Omni extends beyond conventional multimodal models by supporting audio and image generation. This is achieved through the integration of an advanced audio decoder for natural-sounding speech and Ming-Lite-Uni for high-quality image generation, which also allow the model to engage in context-aware chatting, perform text-to-speech conversion, and conduct versatile image editing. Our experimental results showcase Ming-Omni offers a powerful solution for unified perception and generation across all modalities. Notably, our proposed Ming-Omni is the first open-source model we are aware of to match GPT-4o in modality support, and we release all code and model weights to encourage further research and development in the community.
>
---
#### [new 066] FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出FlagEvalMM，一个用于多模态模型评估的框架，解决多任务评估效率与灵活性问题，通过解耦推理与评估提升性能。**

- **链接: [http://arxiv.org/pdf/2506.09081v1](http://arxiv.org/pdf/2506.09081v1)**

> **作者:** Zheqi He; Yesheng Liu; Jing-shu Zheng; Xuejing Li; Richeng Xuan; Jin-Ge Yao; Xi Yang
>
> **摘要:** We present FlagEvalMM, an open-source evaluation framework designed to comprehensively assess multimodal models across a diverse range of vision-language understanding and generation tasks, such as visual question answering, text-to-image/video generation, and image-text retrieval. We decouple model inference from evaluation through an independent evaluation service, thus enabling flexible resource allocation and seamless integration of new tasks and models. Moreover, FlagEvalMM utilizes advanced inference acceleration tools (e.g., vLLM, SGLang) and asynchronous data loading to significantly enhance evaluation efficiency. Extensive experiments show that FlagEvalMM offers accurate and efficient insights into model strengths and limitations, making it a valuable tool for advancing multimodal research. The framework is publicly accessible athttps://github.com/flageval-baai/FlagEvalMM.
>
---
#### [new 067] Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search
- **分类: cs.LG; cs.AI; cs.CL; 68T07, 68T20, 68T30, 93E35; I.2.6; I.2.7; I.2.8**

- **简介: 该论文属于强化学习任务，旨在提升LLM代理在复杂环境中的规划能力。通过原子事实增强和前瞻搜索，使代理在线学习并优化决策，无需参数更新。**

- **链接: [http://arxiv.org/pdf/2506.09171v1](http://arxiv.org/pdf/2506.09171v1)**

> **作者:** Samuel Holt; Max Ruiz Luyten; Thomas Pouplin; Mihaela van der Schaar
>
> **备注:** 9-page main paper, 1 figure. Accepted for an Oral presentation at the First Workshop on Computer Use Agents (ICML 2025), Vancouver, Canada
>
> **摘要:** Large Language Models (LLMs) are increasingly capable but often require significant guidance or extensive interaction history to perform effectively in complex, interactive environments. Existing methods may struggle with adapting to new information or efficiently utilizing past experiences for multi-step reasoning without fine-tuning. We introduce a novel LLM agent framework that enhances planning capabilities through in-context learning, facilitated by atomic fact augmentation and a recursive lookahead search. Our agent learns to extract task-critical ``atomic facts'' from its interaction trajectories. These facts dynamically augment the prompts provided to LLM-based components responsible for action proposal, latent world model simulation, and state-value estimation. Planning is performed via a depth-limited lookahead search, where the LLM simulates potential trajectories and evaluates their outcomes, guided by the accumulated facts and interaction history. This approach allows the agent to improve its understanding and decision-making online, leveraging its experience to refine its behavior without weight updates. We provide a theoretical motivation linking performance to the quality of fact-based abstraction and LLM simulation accuracy. Empirically, our agent demonstrates improved performance and adaptability on challenging interactive tasks, achieving more optimal behavior as it accumulates experience, showcased in tasks such as TextFrozenLake and ALFWorld.
>
---
#### [new 068] You Are What You Say: Exploiting Linguistic Content for VoicePrivacy Attacks
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音隐私攻击任务，研究语言内容对说话人验证的影响，通过BERT模型评估语音匿名化系统的隐私泄露风险。**

- **链接: [http://arxiv.org/pdf/2506.09521v1](http://arxiv.org/pdf/2506.09521v1)**

> **作者:** Ünal Ege Gaznepoglu; Anna Leschanowsky; Ahmad Aloradi; Prachi Singh; Daniel Tenbrinck; Emanuël A. P. Habets; Nils Peters
>
> **备注:** 5 pages, 6 figures, 1 table, accepted at INTERSPEECH 2025
>
> **摘要:** Speaker anonymization systems hide the identity of speakers while preserving other information such as linguistic content and emotions. To evaluate their privacy benefits, attacks in the form of automatic speaker verification (ASV) systems are employed. In this study, we assess the impact of intra-speaker linguistic content similarity in the attacker training and evaluation datasets, by adapting BERT, a language model, as an ASV system. On the VoicePrivacy Attacker Challenge datasets, our method achieves a mean equal error rate (EER) of 35%, with certain speakers attaining EERs as low as 2%, based solely on the textual content of their utterances. Our explainability study reveals that the system decisions are linked to semantically similar keywords within utterances, stemming from how LibriSpeech is curated. Our study suggests reworking the VoicePrivacy datasets to ensure a fair and unbiased evaluation and challenge the reliance on global EER for privacy evaluations.
>
---
#### [new 069] UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench
- **分类: cs.SE; cs.CL; D.0; I.2**

- **简介: 该论文属于代码生成评估任务，旨在解决SWE-Bench测试用例不足的问题。通过构建UTBoost框架，增强测试用例，发现并修正了大量错误补丁。**

- **链接: [http://arxiv.org/pdf/2506.09289v1](http://arxiv.org/pdf/2506.09289v1)**

> **作者:** Boxi Yu; Yuxuan Zhu; Pinjia He; Daniel Kang
>
> **摘要:** The advent of Large Language Models (LLMs) has spurred the development of coding agents for real-world code generation. As a widely used benchmark for evaluating the code generation capabilities of these agents, SWE-Bench uses real-world problems based on GitHub issues and their corresponding pull requests. However, the manually written test cases included in these pull requests are often insufficient, allowing generated patches to pass the tests without resolving the underlying issue. To address this challenge, we introduce UTGenerator, an LLM-driven test case generator that automatically analyzes codebases and dependencies to generate test cases for real-world Python projects. Building on UTGenerator, we propose UTBoost, a comprehensive framework for test case augmentation. In our evaluation, we identified 36 task instances with insufficient test cases and uncovered 345 erroneous patches incorrectly labeled as passed in the original SWE Bench. These corrections, impacting 40.9% of SWE-Bench Lite and 24.4% of SWE-Bench Verified leaderboard entries, yield 18 and 11 ranking changes, respectively.
>
---
#### [new 070] Adding simple structure at inference improves Vision-Language Compositionality
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于视觉-语言组合性任务，旨在提升双编码器模型在图像-文本检索中的组合能力。通过在推理阶段添加结构化处理，显著提升了模型性能。**

- **链接: [http://arxiv.org/pdf/2506.09691v1](http://arxiv.org/pdf/2506.09691v1)**

> **作者:** Imanol Miranda; Ander Salaberria; Eneko Agirre; Gorka Azkune
>
> **摘要:** Dual encoder Vision-Language Models (VLM) such as CLIP are widely used for image-text retrieval tasks. However, those models struggle with compositionality, showing a bag-of-words-like behavior that limits their retrieval performance. Many different training approaches have been proposed to improve the vision-language compositionality capabilities of those models. In comparison, inference-time techniques have received little attention. In this paper, we propose to add simple structure at inference, where, given an image and a caption: i) we divide the image into different smaller crops, ii) we extract text segments, capturing objects, attributes and relations, iii) using a VLM, we find the image crops that better align with text segments obtaining matches, and iv) we compute the final image-text similarity aggregating the individual similarities of the matches. Based on various popular dual encoder VLMs, we evaluate our approach in controlled and natural datasets for VL compositionality. We find that our approach consistently improves the performance of evaluated VLMs without any training, which shows the potential of inference-time techniques. The results are especially good for attribute-object binding as shown in the controlled dataset. As a result of an extensive analysis: i) we show that processing image crops is actually essential for the observed gains in performance, and ii) we identify specific areas to further improve inference-time approaches.
>
---
#### [new 071] Fine-Tuning Large Audio-Language Models with LoRA for Precise Temporal Localization of Prolonged Exposure Therapy Elements
- **分类: eess.AS; cs.CL; cs.HC; 68T07; I.2.7; I.5.4; H.5.2**

- **简介: 该论文属于时间定位任务，旨在自动识别PE治疗中的关键阶段。通过微调音频语言模型，实现精准的时间边界预测，提升治疗评估效率。**

- **链接: [http://arxiv.org/pdf/2506.09707v1](http://arxiv.org/pdf/2506.09707v1)**

> **作者:** Suhas BN; Andrew M. Sherrill; Jyoti Alaparthi; Dominik Mattioli; Rosa I. Arriaga; Chris W. Wiese; Saeed Abdullah
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Prolonged Exposure (PE) therapy is an effective treatment for post-traumatic stress disorder (PTSD), but evaluating therapist fidelity remains labor-intensive due to the need for manual review of session recordings. We present a method for the automatic temporal localization of key PE fidelity elements -- identifying their start and stop times -- directly from session audio and transcripts. Our approach fine-tunes a large pre-trained audio-language model, Qwen2-Audio, using Low-Rank Adaptation (LoRA) to process focused 30-second windows of audio-transcript input. Fidelity labels for three core protocol phases -- therapist orientation (P1), imaginal exposure (P2), and post-imaginal processing (P3) -- are generated via LLM-based prompting and verified by trained raters. The model is trained to predict normalized boundary offsets using soft supervision guided by task-specific prompts. On a dataset of 313 real PE sessions, our best configuration (LoRA rank 8, 30s windows) achieves a mean absolute error (MAE) of 5.3 seconds across tasks. We further analyze the effects of window size and LoRA rank, highlighting the importance of context granularity and model adaptation. This work introduces a scalable framework for fidelity tracking in PE therapy, with potential to support clinician training, supervision, and quality assurance.
>
---
#### [new 072] A Call for Collaborative Intelligence: Why Human-Agent Systems Should Precede AI Autonomy
- **分类: cs.AI; cs.CL; cs.HC; cs.LG; cs.MA**

- **简介: 该论文属于人工智能领域，探讨如何通过人机协作提升AI系统可靠性与适应性，解决AI自主化带来的信任与理解问题。**

- **链接: [http://arxiv.org/pdf/2506.09420v1](http://arxiv.org/pdf/2506.09420v1)**

> **作者:** Henry Peng Zou; Wei-Chieh Huang; Yaozu Wu; Chunyu Miao; Dongyuan Li; Aiwei Liu; Yue Zhou; Yankai Chen; Weizhi Zhang; Yangning Li; Liancheng Fang; Renhe Jiang; Philip S. Yu
>
> **摘要:** Recent improvements in large language models (LLMs) have led many researchers to focus on building fully autonomous AI agents. This position paper questions whether this approach is the right path forward, as these autonomous systems still have problems with reliability, transparency, and understanding the actual requirements of human. We suggest a different approach: LLM-based Human-Agent Systems (LLM-HAS), where AI works with humans rather than replacing them. By keeping human involved to provide guidance, answer questions, and maintain control, these systems can be more trustworthy and adaptable. Looking at examples from healthcare, finance, and software development, we show how human-AI teamwork can handle complex tasks better than AI working alone. We also discuss the challenges of building these collaborative systems and offer practical solutions. This paper argues that progress in AI should not be measured by how independent systems become, but by how well they can work with humans. The most promising future for AI is not in systems that take over human roles, but in those that enhance human capabilities through meaningful partnership.
>
---
#### [new 073] An Interpretable N-gram Perplexity Threat Model for Large Language Model Jailbreaks
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于LLM安全任务，旨在评估和比较 jailbreak 攻击的有效性。提出一种基于n-gram困惑度的可解释威胁模型，用于分析攻击方法并发现其利用罕见词组的特性。**

- **链接: [http://arxiv.org/pdf/2410.16222v2](http://arxiv.org/pdf/2410.16222v2)**

> **作者:** Valentyn Boreiko; Alexander Panfilov; Vaclav Voracek; Matthias Hein; Jonas Geiping
>
> **摘要:** A plethora of jailbreaking attacks have been proposed to obtain harmful responses from safety-tuned LLMs. These methods largely succeed in coercing the target output in their original settings, but their attacks vary substantially in fluency and computational effort. In this work, we propose a unified threat model for the principled comparison of these methods. Our threat model checks if a given jailbreak is likely to occur in the distribution of text. For this, we build an N-gram language model on 1T tokens, which, unlike model-based perplexity, allows for an LLM-agnostic, nonparametric, and inherently interpretable evaluation. We adapt popular attacks to this threat model, and, for the first time, benchmark these attacks on equal footing with it. After an extensive comparison, we find attack success rates against safety-tuned modern models to be lower than previously presented and that attacks based on discrete optimization significantly outperform recent LLM-based attacks. Being inherently interpretable, our threat model allows for a comprehensive analysis and comparison of jailbreak attacks. We find that effective attacks exploit and abuse infrequent bigrams, either selecting the ones absent from real-world text or rare ones, e.g., specific to Reddit or code datasets.
>
---
#### [new 074] Adversarial Text Generation with Dynamic Contextual Perturbation
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于自然语言处理中的对抗攻击任务，旨在生成更隐蔽的对抗文本。通过动态上下文扰动方法，提升攻击效果并保持语义一致性。**

- **链接: [http://arxiv.org/pdf/2506.09148v1](http://arxiv.org/pdf/2506.09148v1)**

> **作者:** Hetvi Waghela; Jaydip Sen; Sneha Rakshit; Subhasis Dasgupta
>
> **备注:** This is the accepted version of the paper, which was presented at IEEE CALCON. The conference was organized at Jadavpur University, Kolkata, from December 14 to 15, 2025. The paper is six pages long, and it consists of six tables and six figures. This is not the final camera-ready version of the paper
>
> **摘要:** Adversarial attacks on Natural Language Processing (NLP) models expose vulnerabilities by introducing subtle perturbations to input text, often leading to misclassification while maintaining human readability. Existing methods typically focus on word-level or local text segment alterations, overlooking the broader context, which results in detectable or semantically inconsistent perturbations. We propose a novel adversarial text attack scheme named Dynamic Contextual Perturbation (DCP). DCP dynamically generates context-aware perturbations across sentences, paragraphs, and documents, ensuring semantic fidelity and fluency. Leveraging the capabilities of pre-trained language models, DCP iteratively refines perturbations through an adversarial objective function that balances the dual objectives of inducing model misclassification and preserving the naturalness of the text. This comprehensive approach allows DCP to produce more sophisticated and effective adversarial examples that better mimic natural language patterns. Our experimental results, conducted on various NLP models and datasets, demonstrate the efficacy of DCP in challenging the robustness of state-of-the-art NLP systems. By integrating dynamic contextual analysis, DCP significantly enhances the subtlety and impact of adversarial attacks. This study highlights the critical role of context in adversarial attacks and lays the groundwork for creating more robust NLP systems capable of withstanding sophisticated adversarial strategies.
>
---
#### [new 075] Athena: Enhancing Multimodal Reasoning with Data-efficient Process Reward Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决PRM训练数据不足和标签噪声问题。通过预测一致性生成高质量标签，并提出优化策略提升性能。**

- **链接: [http://arxiv.org/pdf/2506.09532v1](http://arxiv.org/pdf/2506.09532v1)**

> **作者:** Shuai Wang; Zhenhua Liu; Jiaheng Wei; Xuanwu Yin; Dong Li; Emad Barsoum
>
> **摘要:** We present Athena-PRM, a multimodal process reward model (PRM) designed to evaluate the reward score for each step in solving complex reasoning problems. Developing high-performance PRMs typically demands significant time and financial investment, primarily due to the necessity for step-level annotations of reasoning steps. Conventional automated labeling methods, such as Monte Carlo estimation, often produce noisy labels and incur substantial computational costs. To efficiently generate high-quality process-labeled data, we propose leveraging prediction consistency between weak and strong completers as a criterion for identifying reliable process labels. Remarkably, Athena-PRM demonstrates outstanding effectiveness across various scenarios and benchmarks with just 5,000 samples. Furthermore, we also develop two effective strategies to improve the performance of PRMs: ORM initialization and up-sampling for negative data. We validate our approach in three specific scenarios: verification for test time scaling, direct evaluation of reasoning step correctness, and reward ranked fine-tuning. Our Athena-PRM consistently achieves superior performance across multiple benchmarks and scenarios. Notably, when using Qwen2.5-VL-7B as the policy model, Athena-PRM enhances performance by 10.2 points on WeMath and 7.1 points on MathVista for test time scaling. Furthermore, Athena-PRM sets the state-of-the-art (SoTA) results in VisualProcessBench and outperforms the previous SoTA by 3.9 F1-score, showcasing its robust capability to accurately assess the correctness of the reasoning step. Additionally, utilizing Athena-PRM as the reward model, we develop Athena-7B with reward ranked fine-tuning and outperforms baseline with a significant margin on five benchmarks.
>
---
#### [new 076] Learning Obfuscations Of LLM Embedding Sequences: Stained Glass Transform
- **分类: cs.LG; cs.CL; cs.CR; cs.IT; math.IT; I.2.7; I.2.m**

- **简介: 该论文属于隐私保护任务，旨在解决LLM在共享环境中数据泄露问题。提出Stained Glass Transform方法，在保持模型性能的同时保护输入隐私。**

- **链接: [http://arxiv.org/pdf/2506.09452v1](http://arxiv.org/pdf/2506.09452v1)**

> **作者:** Jay Roberts; Kyle Mylonakis; Sidhartha Roy; Kaan Kale
>
> **备注:** Submitted to IEEE S&P 2026
>
> **摘要:** The high cost of ownership of AI compute infrastructure and challenges of robust serving of large language models (LLMs) has led to a surge in managed Model-as-a-service deployments. Even when enterprises choose on-premises deployments, the compute infrastructure is typically shared across many teams in order to maximize the return on investment. In both scenarios the deployed models operate only on plaintext data, and so enterprise data owners must allow their data to appear in plaintext on a shared or multi-tenant compute infrastructure. This results in data owners with private or sensitive data being hesitant or restricted in what data they use with these types of deployments. In this work we introduce the Stained Glass Transform, a learned, stochastic, and sequence dependent transformation of the word embeddings of an LLM which information theoretically provides privacy to the input of the LLM while preserving the utility of model. We theoretically connect a particular class of Stained Glass Transforms to the theory of mutual information of Gaussian Mixture Models. We then calculate a-postiori privacy estimates, based on mutual information, and verify the privacy and utility of instances of transformed embeddings through token level metrics of privacy and standard LLM performance benchmarks.
>
---
#### [new 077] Revisit What You See: Disclose Language Prior in Vision Tokens for Efficient Guided Decoding of LVLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型任务，旨在解决LVLM生成缺乏视觉依据的问题。提出ReVisiT方法，通过引用视觉标记提升生成结果的视觉一致性。**

- **链接: [http://arxiv.org/pdf/2506.09522v1](http://arxiv.org/pdf/2506.09522v1)**

> **作者:** Beomsik Cho; Jaehyung Kim
>
> **备注:** Code available at https://github.com/bscho333/ReVisiT
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated remarkable performance across various multimodal tasks by integrating visual perception with language understanding. However, conventional decoding strategies of LVLMs often fail to successfully utilize visual information, leading to visually ungrounded responses. While various approaches have been proposed to address this limitation, they typically require additional training, multi-step inference procedures, or external model dependencies. This paper introduces ReVisiT, a simple yet effective decoding method that references vision tokens to guide the text generation process in LVLMs. Our approach leverages the semantic information embedded within vision tokens by projecting them into the text token distribution space, and dynamically selecting the most relevant vision token at each decoding step through constrained divergence minimization. This selected vision token is then used to refine the output distribution to better incorporate visual semantics. Experiments on three LVLM hallucination benchmarks with two recent LVLMs demonstrate that ReVisiT consistently enhances visual grounding with minimal computational overhead. Moreover, our method achieves competitive or superior results relative to state-of-the-art baselines while reducing computational costs for up to $2\times$.
>
---
#### [new 078] SensorLM: Learning the Language of Wearable Sensors
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出SensorLM，解决可穿戴传感器数据与自然语言对齐的问题。通过构建大规模数据集和多模态预训练模型，提升活动分析和健康任务的性能。**

- **链接: [http://arxiv.org/pdf/2506.09108v1](http://arxiv.org/pdf/2506.09108v1)**

> **作者:** Yuwei Zhang; Kumar Ayush; Siyuan Qiao; A. Ali Heydari; Girish Narayanswamy; Maxwell A. Xu; Ahmed A. Metwally; Shawn Xu; Jake Garrison; Xuhai Xu; Tim Althoff; Yun Liu; Pushmeet Kohli; Jiening Zhan; Mark Malhotra; Shwetak Patel; Cecilia Mascolo; Xin Liu; Daniel McDuff; Yuzhe Yang
>
> **摘要:** We present SensorLM, a family of sensor-language foundation models that enable wearable sensor data understanding with natural language. Despite its pervasive nature, aligning and interpreting sensor data with language remains challenging due to the lack of paired, richly annotated sensor-text descriptions in uncurated, real-world wearable data. We introduce a hierarchical caption generation pipeline designed to capture statistical, structural, and semantic information from sensor data. This approach enabled the curation of the largest sensor-language dataset to date, comprising over 59.7 million hours of data from more than 103,000 people. Furthermore, SensorLM extends prominent multimodal pretraining architectures (e.g., CLIP, CoCa) and recovers them as specific variants within a generic architecture. Extensive experiments on real-world tasks in human activity analysis and healthcare verify the superior performance of SensorLM over state-of-the-art in zero-shot recognition, few-shot learning, and cross-modal retrieval. SensorLM also demonstrates intriguing capabilities including scaling behaviors, label efficiency, sensor captioning, and zero-shot generalization to unseen tasks.
>
---
#### [new 079] CAIRe: Cultural Attribution of Images by Retrieval-Augmented Evaluation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于图像文化属性评估任务，旨在解决跨文化偏见难以量化的问题。提出CAIRe框架，通过知识库和事实信息评估图像的文化相关性。**

- **链接: [http://arxiv.org/pdf/2506.09109v1](http://arxiv.org/pdf/2506.09109v1)**

> **作者:** Arnav Yayavaram; Siddharth Yayavaram; Simran Khanuja; Michael Saxon; Graham Neubig
>
> **备注:** Preprint, under review
>
> **摘要:** As text-to-image models become increasingly prevalent, ensuring their equitable performance across diverse cultural contexts is critical. Efforts to mitigate cross-cultural biases have been hampered by trade-offs, including a loss in performance, factual inaccuracies, or offensive outputs. Despite widespread recognition of these challenges, an inability to reliably measure these biases has stalled progress. To address this gap, we introduce CAIRe, a novel evaluation metric that assesses the degree of cultural relevance of an image, given a user-defined set of labels. Our framework grounds entities and concepts in the image to a knowledge base and uses factual information to give independent graded judgments for each culture label. On a manually curated dataset of culturally salient but rare items built using language models, CAIRe surpasses all baselines by 28% F1 points. Additionally, we construct two datasets for culturally universal concept, one comprising of T2I-generated outputs and another retrieved from naturally occurring data. CAIRe achieves Pearson's correlations of 0.56 and 0.66 with human ratings on these sets, based on a 5-point Likert scale of cultural relevance. This demonstrates its strong alignment with human judgment across diverse image sources.
>
---
#### [new 080] Regularizing Learnable Feature Extraction for Automatic Speech Recognition
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于自动语音识别任务，旨在解决可学习特征提取器易过拟合的问题。通过音频扰动和频域掩码等正则化方法提升性能。**

- **链接: [http://arxiv.org/pdf/2506.09804v1](http://arxiv.org/pdf/2506.09804v1)**

> **作者:** Peter Vieting; Maximilian Kannen; Benedikt Hilmes; Ralf Schlüter; Hermann Ney
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Neural front-ends are an appealing alternative to traditional, fixed feature extraction pipelines for automatic speech recognition (ASR) systems since they can be directly trained to fit the acoustic model. However, their performance often falls short compared to classical methods, which we show is largely due to their increased susceptibility to overfitting. This work therefore investigates regularization methods for training ASR models with learnable feature extraction front-ends. First, we examine audio perturbation methods and show that larger relative improvements can be obtained for learnable features. Additionally, we identify two limitations in the standard use of SpecAugment for these front-ends and propose masking in the short time Fourier transform (STFT)-domain as a simple but effective modification to address these challenges. Finally, integrating both regularization approaches effectively closes the performance gap between traditional and learnable features.
>
---
#### [new 081] ThinkQE: Query Expansion via an Evolving Thinking Process
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决查询扩展中缺乏多样性和深度的问题。提出ThinkQE框架，通过思维过程和语料互动提升扩展效果。**

- **链接: [http://arxiv.org/pdf/2506.09260v1](http://arxiv.org/pdf/2506.09260v1)**

> **作者:** Yibin Lei; Tao Shen; Andrew Yates
>
> **摘要:** Effective query expansion for web search benefits from promoting both exploration and result diversity to capture multiple interpretations and facets of a query. While recent LLM-based methods have improved retrieval performance and demonstrate strong domain generalization without additional training, they often generate narrowly focused expansions that overlook these desiderata. We propose ThinkQE, a test-time query expansion framework addressing this limitation through two key components: a thinking-based expansion process that encourages deeper and comprehensive semantic exploration, and a corpus-interaction strategy that iteratively refines expansions using retrieval feedback from the corpus. Experiments on diverse web search benchmarks (DL19, DL20, and BRIGHT) show ThinkQE consistently outperforms prior approaches, including training-intensive dense retrievers and rerankers.
>
---
#### [new 082] Intent Factored Generation: Unleashing the Diversity in Your Language Model
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言生成任务，旨在提升语言模型输出的多样性与质量。通过分阶段生成意图和响应，解决现有方法多样性不足的问题。**

- **链接: [http://arxiv.org/pdf/2506.09659v1](http://arxiv.org/pdf/2506.09659v1)**

> **作者:** Eltayeb Ahmed; Uljad Berdica; Martha Elliott; Danijela Horak; Jakob N. Foerster
>
> **摘要:** Obtaining multiple meaningfully diverse, high quality samples from Large Language Models for a fixed prompt remains an open challenge. Current methods for increasing diversity often only operate at the token-level, paraphrasing the same response. This is problematic because it leads to poor exploration on reasoning problems and to unengaging, repetitive conversational agents. To address this we propose Intent Factored Generation (IFG), factorising the sampling process into two stages. First, we sample a semantically dense intent, e.g., a summary or keywords. Second, we sample the final response conditioning on both the original prompt and the intent from the first stage. This allows us to use a higher temperature during the intent step to promote conceptual diversity, and a lower temperature during the final generation to ensure the outputs are coherent and self-consistent. Additionally, we find that prompting the model to explicitly state its intent for each step of the chain-of-thought before generating the step is beneficial for reasoning tasks. We demonstrate our method's effectiveness across a diverse set of tasks. We show this method improves both pass@k and Reinforcement Learning from Verifier Feedback on maths and code tasks. For instruction-tuning, we combine IFG with Direct Preference Optimisation to increase conversational diversity without sacrificing reward. Finally, we achieve higher diversity while maintaining the quality of generations on a general language modelling task, using a new dataset of reader comments and news articles that we collect and open-source. In summary, we present a simple method of increasing the sample diversity of LLMs while maintaining performance. This method can be implemented by changing the prompt and varying the temperature during generation, making it easy to integrate into many algorithms for gains across various applications.
>
---
#### [new 083] Natural Language Guided Ligand-Binding Protein Design
- **分类: cs.LG; cs.CE; cs.CL**

- **简介: 该论文属于蛋白质设计任务，旨在通过自然语言指令生成能结合特定配体的蛋白质。研究提出InstructPro模型，解决数据稀缺问题，并在基准测试中表现优异。**

- **链接: [http://arxiv.org/pdf/2506.09332v1](http://arxiv.org/pdf/2506.09332v1)**

> **作者:** Zhenqiao Song; Ramith Hettiarachchi; Chuan Li; Jianwen Xie; Lei Li
>
> **摘要:** Can AI protein models follow human language instructions and design proteins with desired functions (e.g. binding to a ligand)? Designing proteins that bind to a given ligand is crucial in a wide range of applications in biology and chemistry. Most prior AI models are trained on protein-ligand complex data, which is scarce due to the high cost and time requirements of laboratory experiments. In contrast, there is a substantial body of human-curated text descriptions about protein-ligand interactions and ligand formula. In this paper, we propose InstructPro, a family of protein generative models that follow natural language instructions to design ligand-binding proteins. Given a textual description of the desired function and a ligand formula in SMILES, InstructPro generates protein sequences that are functionally consistent with the specified instructions. We develop the model architecture, training strategy, and a large-scale dataset, InstructProBench, to support both training and evaluation. InstructProBench consists of 9,592,829 triples of (function description, ligand formula, protein sequence). We train two model variants: InstructPro-1B (with 1 billion parameters) and InstructPro-3B~(with 3 billion parameters). Both variants consistently outperform strong baselines, including ProGen2, ESM3, and Pinal. Notably, InstructPro-1B achieves the highest docking success rate (81.52% at moderate confidence) and the lowest average root mean square deviation (RMSD) compared to ground truth structures (4.026{\AA}). InstructPro-3B further descreases the average RMSD to 2.527{\AA}, demonstrating InstructPro's ability to generate ligand-binding proteins that align with the functional specifications.
>
---
#### [new 084] SimClass: A Classroom Speech Dataset Generated via Game Engine Simulation For Automatic Speech Recognition Research
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决课堂语音数据稀缺问题。通过游戏引擎生成合成课堂噪声和语音数据，构建SimClass数据集，提升语音模型的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.09206v1](http://arxiv.org/pdf/2506.09206v1)**

> **作者:** Ahmed Adel Attia; Jing Liu; Carl Espy-Wilson
>
> **摘要:** The scarcity of large-scale classroom speech data has hindered the development of AI-driven speech models for education. Public classroom datasets remain limited, and the lack of a dedicated classroom noise corpus prevents the use of standard data augmentation techniques. In this paper, we introduce a scalable methodology for synthesizing classroom noise using game engines, a framework that extends to other domains. Using this methodology, we present SimClass, a dataset that includes both a synthesized classroom noise corpus and a simulated classroom speech dataset. The speech data is generated by pairing a public children's speech corpus with YouTube lecture videos to approximate real classroom interactions in clean conditions. Our experiments on clean and noisy speech demonstrate that SimClass closely approximates real classroom speech, making it a valuable resource for developing robust speech recognition and enhancement models.
>
---
#### [new 085] Too Big to Think: Capacity, Memorization, and Generalization in Pre-Trained Transformers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究预训练Transformer模型在记忆与泛化间的权衡问题，通过合成任务分析模型容量对学习行为的影响。**

- **链接: [http://arxiv.org/pdf/2506.09099v1](http://arxiv.org/pdf/2506.09099v1)**

> **作者:** Joshua Barron; Devin White
>
> **备注:** Accepted for oral presentation to Tiny Titans: The next wave of On-Device Learning for Foundational Models Workshop at the 42nd International Conference on Machine Learning
>
> **摘要:** The relationship between memorization and generalization in large language models (LLMs) remains an open area of research, with growing evidence that the two are deeply intertwined. In this work, we investigate this relationship by pre-training a series of capacity-limited Transformer models from scratch on two synthetic character-level tasks designed to separately probe generalization (via arithmetic extrapolation) and memorization (via factual recall). We observe a consistent trade-off: small models extrapolate to unseen arithmetic cases but fail to memorize facts, while larger models memorize but fail to extrapolate. An intermediate-capacity model exhibits a similar shift toward memorization. When trained on both tasks jointly, no model (regardless of size) succeeds at extrapolation. These findings suggest that pre-training may intrinsically favor one learning mode over the other. By isolating these dynamics in a controlled setting, our study offers insight into how model capacity shapes learning behavior and offers broader implications for the design and deployment of small language models.
>
---
#### [new 086] Advancing Exchange Rate Forecasting: Leveraging Machine Learning and AI for Enhanced Accuracy in Global Financial Markets
- **分类: q-fin.ST; cs.CL; cs.LG**

- **简介: 该论文属于外汇汇率预测任务，旨在提升预测准确性。通过LSTM和GBC模型分析历史数据，验证了深度学习在外汇市场的有效性。**

- **链接: [http://arxiv.org/pdf/2506.09851v1](http://arxiv.org/pdf/2506.09851v1)**

> **作者:** Md. Yeasin Rahat; Rajan Das Gupta; Nur Raisa Rahman; Sudipto Roy Pritom; Samiur Rahman Shakir; Md Imrul Hasan Showmick; Md. Jakir Hossen
>
> **备注:** Accepted in MECON 2025
>
> **摘要:** The prediction of foreign exchange rates, such as the US Dollar (USD) to Bangladeshi Taka (BDT), plays a pivotal role in global financial markets, influencing trade, investments, and economic stability. This study leverages historical USD/BDT exchange rate data from 2018 to 2023, sourced from Yahoo Finance, to develop advanced machine learning models for accurate forecasting. A Long Short-Term Memory (LSTM) neural network is employed, achieving an exceptional accuracy of 99.449%, a Root Mean Square Error (RMSE) of 0.9858, and a test loss of 0.8523, significantly outperforming traditional methods like ARIMA (RMSE 1.342). Additionally, a Gradient Boosting Classifier (GBC) is applied for directional prediction, with backtesting on a $10,000 initial capital revealing a 40.82% profitable trade rate, though resulting in a net loss of $20,653.25 over 49 trades. The study analyzes historical trends, showing a decline in BDT/USD rates from 0.012 to 0.009, and incorporates normalized daily returns to capture volatility. These findings highlight the potential of deep learning in forex forecasting, offering traders and policymakers robust tools to mitigate risks. Future work could integrate sentiment analysis and real-time economic indicators to further enhance model adaptability in volatile markets.
>
---
#### [new 087] Outside Knowledge Conversational Video (OKCV) Dataset -- Dialoguing over Videos
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出OKCV数据集，用于研究基于视频的对话任务。模型需结合视频内容与外部知识回答问题，并考虑对话上下文。**

- **链接: [http://arxiv.org/pdf/2506.09953v1](http://arxiv.org/pdf/2506.09953v1)**

> **作者:** Benjamin Reichman; Constantin Patsch; Jack Truxal; Atishay Jain; Larry Heck
>
> **摘要:** In outside knowledge visual question answering (OK-VQA), the model must identify relevant visual information within an image and incorporate external knowledge to accurately respond to a question. Extending this task to a visually grounded dialogue setting based on videos, a conversational model must both recognize pertinent visual details over time and answer questions where the required information is not necessarily present in the visual information. Moreover, the context of the overall conversation must be considered for the subsequent dialogue. To explore this task, we introduce a dataset comprised of $2,017$ videos with $5,986$ human-annotated dialogues consisting of $40,954$ interleaved dialogue turns. While the dialogue context is visually grounded in specific video segments, the questions further require external knowledge that is not visually present. Thus, the model not only has to identify relevant video parts but also leverage external knowledge to converse within the dialogue. We further provide several baselines evaluated on our dataset and show future challenges associated with this task. The dataset is made publicly available here: https://github.com/c-patsch/OKCV.
>
---
#### [new 088] Flipping Against All Odds: Reducing LLM Coin Flip Bias via Verbalized Rejection Sampling
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM采样偏差问题。通过引入VRS方法，提升采样可靠性。**

- **链接: [http://arxiv.org/pdf/2506.09998v1](http://arxiv.org/pdf/2506.09998v1)**

> **作者:** Tim Z. Xiao; Johannes Zenn; Zhen Liu; Weiyang Liu; Robert Bamler; Bernhard Schölkopf
>
> **备注:** Technical Report v1 (21 pages, 14 figures)
>
> **摘要:** Large language models (LLMs) can often accurately describe probability distributions using natural language, yet they still struggle to generate faithful samples from them. This mismatch limits their use in tasks requiring reliable stochasticity, such as Monte Carlo methods, agent-based simulations, and randomized decision-making. We investigate this gap between knowledge and sampling in the context of Bernoulli distributions. We introduce Verbalized Rejection Sampling (VRS), a natural-language adaptation of classical rejection sampling that prompts the LLM to reason about and accept or reject proposed samples. Despite relying on the same Bernoulli mechanism internally, VRS substantially reduces sampling bias across models. We provide theoretical analysis showing that, under mild assumptions, VRS improves over direct sampling, with gains attributable to both the algorithm and prompt design. More broadly, our results show how classical probabilistic tools can be verbalized and embedded into LLM workflows to improve reliability, without requiring access to model internals or heavy prompt engineering.
>
---
#### [new 089] OWSM-Biasing: Contextualizing Open Whisper-Style Speech Models for Automatic Speech Recognition with Dynamic Vocabulary
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于自动语音识别任务，旨在解决罕见词识别问题。通过结合上下文偏置方法与预训练模型，提升识别准确率。**

- **链接: [http://arxiv.org/pdf/2506.09448v1](http://arxiv.org/pdf/2506.09448v1)**

> **作者:** Yui Sudo; Yusuke Fujita; Atsushi Kojima; Tomoya Mizumoto; Lianbo Liu
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Speech foundation models (SFMs), such as Open Whisper-Style Speech Models (OWSM), are trained on massive datasets to achieve accurate automatic speech recognition. However, even SFMs struggle to accurately recognize rare and unseen words. While contextual biasing (CB) is a promising approach to improve recognition of such words, most CB methods are trained from scratch, resulting in lower performance than SFMs due to the lack of pre-trained knowledge. This paper integrates an existing CB method with OWSM v3.1 while freezing its pre-trained parameters. By leveraging the knowledge embedded in SFMs, the proposed method enables effective CB while preserving the advantages of SFMs, even with a small dataset. Experimental results show that the proposed method improves the biasing word error rate (B-WER) by 11.6 points, resulting in a 0.9 point improvement in the overall WER while reducing the real-time factor by 7.5% compared to the non-biasing baseline on the LibriSpeech 100 test-clean set.
>
---
## 更新

#### [replaced 001] Assessment of Evolving Large Language Models in Upper Secondary Mathematics
- **分类: cs.CL; cs.AI; cs.CY; K.3; I.2**

- **链接: [http://arxiv.org/pdf/2504.12347v2](http://arxiv.org/pdf/2504.12347v2)**

> **作者:** Mika Setälä; Pieta Sikström; Ville Heilala; Tommi Kärkkäinen
>
> **摘要:** Large language models (LLMs) have shown increasing promise in educational settings, yet their mathematical reasoning has been considered evolving. This study evaluates the mathematical capabilities of various LLMs using the Finnish matriculation examination, a high-stakes digital test for upper secondary education. Initial tests yielded moderate performance corresponding to mid-range grades, but later evaluations demonstrated substantial improvements as the language models evolved. Remarkably, some models achieved near-perfect or perfect scores, matching top student performance and qualifying for university admission. Our findings highlight the rapid advances in the mathematical proficiency of LLMs and illustrate their potential as underlying tools to support learning and teaching in a variety of ways.
>
---
#### [replaced 002] AdversariaL attacK sAfety aLIgnment(ALKALI): Safeguarding LLMs through GRACE: Geometric Representation-Aware Contrastive Enhancement- Introducing Adversarial Vulnerability Quality Index (AVQI)
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.08885v2](http://arxiv.org/pdf/2506.08885v2)**

> **作者:** Danush Khanna; Krishna Kumar; Basab Ghosh; Vinija Jain; Vasu Sharma; Aman Chadha; Amitava Das
>
> **摘要:** Adversarial threats against LLMs are escalating faster than current defenses can adapt. We expose a critical geometric blind spot in alignment: adversarial prompts exploit latent camouflage, embedding perilously close to the safe representation manifold while encoding unsafe intent thereby evading surface level defenses like Direct Preference Optimization (DPO), which remain blind to the latent geometry. We introduce ALKALI, the first rigorously curated adversarial benchmark and the most comprehensive to date spanning 9,000 prompts across three macro categories, six subtypes, and fifteen attack families. Evaluation of 21 leading LLMs reveals alarmingly high Attack Success Rates (ASRs) across both open and closed source models, exposing an underlying vulnerability we term latent camouflage, a structural blind spot where adversarial completions mimic the latent geometry of safe ones. To mitigate this vulnerability, we introduce GRACE - Geometric Representation Aware Contrastive Enhancement, an alignment framework coupling preference learning with latent space regularization. GRACE enforces two constraints: latent separation between safe and adversarial completions, and adversarial cohesion among unsafe and jailbreak behaviors. These operate over layerwise pooled embeddings guided by a learned attention profile, reshaping internal geometry without modifying the base model, and achieve up to 39% ASR reduction. Moreover, we introduce AVQI, a geometry aware metric that quantifies latent alignment failure via cluster separation and compactness. AVQI reveals when unsafe completions mimic the geometry of safe ones, offering a principled lens into how models internally encode safety. We make the code publicly available at https://anonymous.4open.science/r/alkali-B416/README.md.
>
---
#### [replaced 003] Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07044v3](http://arxiv.org/pdf/2506.07044v3)**

> **作者:** LASA Team; Weiwen Xu; Hou Pong Chan; Long Li; Mahani Aljunied; Ruifeng Yuan; Jianyu Wang; Chenghao Xiao; Guizhen Chen; Chaoqun Liu; Zhaodonghui Li; Yu Sun; Junao Shen; Chaojun Wang; Jie Tan; Deli Zhao; Tingyang Xu; Hao Zhang; Yu Rong
>
> **备注:** Technical Report, 53 pages, 25 tables, and 16 figures
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in understanding common visual elements, largely due to their large-scale datasets and advanced training strategies. However, their effectiveness in medical applications remains limited due to the inherent discrepancies between data and tasks in medical scenarios and those in the general domain. Concretely, existing medical MLLMs face the following critical limitations: (1) limited coverage of medical knowledge beyond imaging, (2) heightened susceptibility to hallucinations due to suboptimal data curation processes, (3) lack of reasoning capabilities tailored for complex medical scenarios. To address these challenges, we first propose a comprehensive data curation procedure that (1) efficiently acquires rich medical knowledge data not only from medical imaging but also from extensive medical texts and general-domain data; and (2) synthesizes accurate medical captions, visual question answering (VQA), and reasoning samples. As a result, we build a multimodal dataset enriched with extensive medical knowledge. Building on the curated data, we introduce our medical-specialized MLLM: Lingshu. Lingshu undergoes multi-stage training to embed medical expertise and enhance its task-solving capabilities progressively. Besides, we preliminarily explore the potential of applying reinforcement learning with verifiable rewards paradigm to enhance Lingshu's medical reasoning ability. Additionally, we develop MedEvalKit, a unified evaluation framework that consolidates leading multimodal and textual medical benchmarks for standardized, fair, and efficient model assessment. We evaluate the performance of Lingshu on three fundamental medical tasks, multimodal QA, text-based QA, and medical report generation. The results show that Lingshu consistently outperforms the existing open-source multimodal models on most tasks ...
>
---
#### [replaced 004] Same Task, Different Circuits: Disentangling Modality-Specific Mechanisms in VLMs
- **分类: cs.CL; 68T5; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.09047v2](http://arxiv.org/pdf/2506.09047v2)**

> **作者:** Yaniv Nikankin; Dana Arad; Yossi Gandelsman; Yonatan Belinkov
>
> **摘要:** Vision-Language models (VLMs) show impressive abilities to answer questions on visual inputs (e.g., counting objects in an image), yet demonstrate higher accuracies when performing an analogous task on text (e.g., counting words in a text). We investigate this accuracy gap by identifying and comparing the \textit{circuits} - the task-specific computational sub-graphs - in different modalities. We show that while circuits are largely disjoint between modalities, they implement relatively similar functionalities: the differences lie primarily in processing modality-specific data positions (an image or a text sequence). Zooming in on the image data representations, we observe they become aligned with the higher-performing analogous textual representations only towards later layers, too late in processing to effectively influence subsequent positions. To overcome this, we patch the representations of visual data tokens from later layers back into earlier layers. In experiments with multiple tasks and models, this simple intervention closes a third of the performance gap between the modalities, on average. Our analysis sheds light on the multi-modal performance gap in VLMs and suggests a training-free approach for reducing it.
>
---
#### [replaced 005] Style over Substance: Distilled Language Models Reason Via Stylistic Replication
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.01738v2](http://arxiv.org/pdf/2504.01738v2)**

> **作者:** Philip Lippmann; Jie Yang
>
> **摘要:** Specialized reasoning language models (RLMs) have demonstrated that scaling test-time computation through detailed reasoning traces significantly enhances performance. Although these traces effectively facilitate knowledge distillation into smaller, instruction-tuned models, the precise nature of transferred reasoning remains unclear. In this study, we investigate to what extent distilled models internalize replicated stylistic patterns during reasoning. To this end, we systematically analyze reasoning traces, identifying structural and lexical patterns that characterize successful reasoning. We then introduce two new datasets -- a dataset of emergent reasoning traces and a synthetic dataset explicitly constructed to replicate these stylistic patterns -- to precisely examine their influence on distilled models' reasoning capabilities. We find that models trained on the synthetic traces achieve comparable performance, indicating that distilled reasoning abilities rely significantly on surface-level patterns. Surprisingly, we observe an increase in performance even when the synthetic traces are altered to lead to the wrong answer. Our findings highlight how stylistic patterns can be leveraged to efficiently enhance LM reasoning across diverse model families.
>
---
#### [replaced 006] Standard Language Ideology in AI-Generated Language
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.08726v2](http://arxiv.org/pdf/2406.08726v2)**

> **作者:** Genevieve Smith; Eve Fleisig; Madeline Bossi; Ishita Rustagi; Xavier Yin
>
> **摘要:** Standard language ideology is reflected and reinforced in language generated by large language models (LLMs). We present a faceted taxonomy of open problems that illustrate how standard language ideology manifests in AI-generated language, alongside implications for minoritized language communities and society more broadly. We introduce the concept of standard AI-generated language ideology, a process through which LLMs position "standard" languages--particularly Standard American English (SAE)--as the linguistic default, reinforcing the perception that SAE is the most "appropriate" language. We then discuss ongoing tensions around what constitutes desirable system behavior, as well as advantages and drawbacks of generative AI tools attempting, or refusing, to imitate different English language varieties. Rather than prescribing narrow technical fixes, we offer three recommendations for researchers, practitioners, and funders that focus on shifting structural conditions and supporting more emancipatory outcomes for diverse language communities.
>
---
#### [replaced 007] Trustworthy AI: Safety, Bias, and Privacy -- A Survey
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.10450v2](http://arxiv.org/pdf/2502.10450v2)**

> **作者:** Xingli Fang; Jianwei Li; Varun Mulchandani; Jung-Eun Kim
>
> **摘要:** The capabilities of artificial intelligence systems have been advancing to a great extent, but these systems still struggle with failure modes, vulnerabilities, and biases. In this paper, we study the current state of the field, and present promising insights and perspectives regarding concerns that challenge the trustworthiness of AI models. In particular, this paper investigates the issues regarding three thrusts: safety, privacy, and bias, which hurt models' trustworthiness. For safety, we discuss safety alignment in the context of large language models, preventing them from generating toxic or harmful content. For bias, we focus on spurious biases that can mislead a network. Lastly, for privacy, we cover membership inference attacks in deep neural networks. The discussions addressed in this paper reflect our own experiments and observations.
>
---
#### [replaced 008] Writing-Zero: Bridge the Gap Between Non-verifiable Tasks and Verifiable Rewards
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00103v2](http://arxiv.org/pdf/2506.00103v2)**

> **作者:** Ruipeng Jia; Yunyi Yang; Yongbo Gai; Kai Luo; Shihao Huang; Jianhe Lin; Xiaoxi Jiang; Guanjun Jiang
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has enabled large language models (LLMs) to achieve remarkable breakthroughs in reasoning tasks with objective ground-truth answers, such as mathematics and code generation. However, a significant gap remains for non-verifiable tasks, like creative writing and open-ended dialogue, where quality assessment is inherently subjective and lacks definitive references. Existing approaches for these domains often rely on scalar reward models trained with human preferences, which suffer from limited generalization and are prone to reward hacking, such as over-explanation and length bias. In this work, we propose a unified RLVR-based training paradigm that bridges the gap between non-verifiable tasks and verifiable rewards. We introduce a writing-principle-based pairwise Generative Reward Model (GenRM) and a novel Bootstrapped Relative Policy Optimization (BRPO) algorithm. The pairwise writing GenRM leverages self-principled critique to transform subjective assessments into reliable, verifiable rewards, while BRPO enables dynamic, reference-free pairwise comparison by leveraging a bootstrapped response as temporary reference from within group rollouts during RL training. Our approach empowers LLMs to develop robust writing capabilities without supervised fine-tuning, as demonstrated by Writing-Zero, which shows consistent improvement and strong resistance to reward hacking compared to scalar reward baselines. Furthermore, our method achieves competitive results on both in-house and open-source writing benchmarks. Our findings suggest the potential to unify rule-based, reference-based, and reference-free reward modeling under the RLVR framework, thus paving the way for a comprehensive and scalable RL training paradigm applicable across all language tasks.
>
---
#### [replaced 009] SWE-Flow: Synthesizing Software Engineering Data in a Test-Driven Manner
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09003v2](http://arxiv.org/pdf/2506.09003v2)**

> **作者:** Lei Zhang; Jiaxi Yang; Min Yang; Jian Yang; Mouxiang Chen; Jiajun Zhang; Zeyu Cui; Binyuan Hui; Junyang Lin
>
> **备注:** Accepted by ICML2025
>
> **摘要:** We introduce **SWE-Flow**, a novel data synthesis framework grounded in Test-Driven Development (TDD). Unlike existing software engineering data that rely on human-submitted issues, **SWE-Flow** automatically infers incremental development steps directly from unit tests, which inherently encapsulate high-level requirements. The core of **SWE-Flow** is the construction of a Runtime Dependency Graph (RDG), which precisely captures function interactions, enabling the generation of a structured, step-by-step *development schedule*. At each step, **SWE-Flow** produces a partial codebase, the corresponding unit tests, and the necessary code modifications, resulting in fully verifiable TDD tasks. With this approach, we generated 16,061 training instances and 2,020 test instances from real-world GitHub projects, creating the **SWE-Flow-Eval** benchmark. Our experiments show that fine-tuning open model on this dataset significantly improves performance in TDD-based coding. To facilitate further research, we release all code, datasets, models, and Docker images at [Github](https://github.com/Hambaobao/SWE-Flow).
>
---
#### [replaced 010] AcTracer: Active Testing of Large Language Model via Multi-Stage Sampling
- **分类: cs.SE; cs.AI; cs.CL; D.2.5; I.2.7**

- **链接: [http://arxiv.org/pdf/2408.03573v2](http://arxiv.org/pdf/2408.03573v2)**

> **作者:** Yuheng Huang; Jiayang Song; Qiang Hu; Felix Juefei-Xu; Lei Ma
>
> **备注:** To appear in ACM Transactions on Software Engineering and Methodology (2025)
>
> **摘要:** Performance evaluation plays a crucial role in the development life cycle of large language models (LLMs). It estimates the model's capability, elucidates behavior characteristics, and facilitates the identification of potential issues and limitations, thereby guiding further improvement. Given that LLMs' diverse task-handling abilities stem from large volumes of training data, a comprehensive evaluation also necessitates abundant, well-annotated, and representative test data to assess LLM performance across various downstream tasks. However, the demand for high-quality test data often entails substantial time, computational resources, and manual efforts, sometimes causing the evaluation to be inefficient or impractical. To address these challenges, researchers propose active testing, which estimates the overall performance by selecting a subset of test data. Nevertheless, the existing active testing methods tend to be inefficient, even inapplicable, given the unique new challenges of LLMs (e.g., diverse task types, increased model complexity, and unavailability of training data). To mitigate such limitations and expedite the development cycle of LLMs, in this work, we introduce AcTracer, an active testing framework tailored for LLMs that strategically selects a small subset of test data to achieve a more accurate performance estimation for LLMs. AcTracer utilizes both internal and external information from LLMs to guide the test sampling process, reducing variance through a multi-stage pool-based active selection. Our experiment results demonstrate that AcTracer achieves state-of-the-art performance compared to existing methods across various tasks.
>
---
#### [replaced 011] Unable to Forget: Proactive lnterference Reveals Working Memory Limits in LLMs Beyond Context Length
- **分类: cs.CL; cs.AI; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2506.08184v2](http://arxiv.org/pdf/2506.08184v2)**

> **作者:** Chupei Wang; Jiaqiu Vince Sun
>
> **摘要:** Information retrieval in Large Language Models (LLMs) is increasingly recognized as intertwined with generation capabilities rather than mere lookup. While longer contexts are often assumed to improve retrieval, the effects of intra-context interference remain understudied. To address this, we adapt the proactive interference (PI) paradigm from cognitive science, where earlier information disrupts recall of newer updates. In humans, susceptibility to such interference is inversely linked to working memory capacity. We introduce PI-LLM, an evaluation that sequentially streams semantically related key-value updates and queries only the final values. Although these final values are clearly positioned just before the query, LLM retrieval accuracy declines log-linearly toward zero as interference accumulates; errors arise from retrieving previously overwritten values. Attempts to mitigate interference via prompt engineering (e.g., instructing models to ignore earlier input) yield limited success. These findings reveal a fundamental constraint on LLMs' ability to disentangle interference and flexibly manipulate information, suggesting a working memory bottleneck beyond mere context access. This calls for approaches that strengthen models' ability to suppress irrelevant content during retrieval.
>
---
#### [replaced 012] Multi-Party Supervised Fine-tuning of Language Models for Multi-Party Dialogue Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.05342v5](http://arxiv.org/pdf/2412.05342v5)**

> **作者:** Xiaoyu Wang; Ningyuan Xi; Teng Chen; Qingqing Gu; Yue Zhao; Xiaokai Chen; Zhonglin Jiang; Yong Chen; Luo Ji
>
> **备注:** Accepted by IJCNN 2025
>
> **摘要:** Large Language Models (LLM) are usually fine-tuned to participate in dyadic or two-party dialogues, which can not adapt well to multi-party dialogues (MPD), which hinders their applications in such scenarios including multi-personal meetings, discussions and daily communication. Previous LLM-based researches mainly focus on the multi-agent framework, while their base LLMs are still pairwisely fine-tuned. In this work, we design a multi-party fine-tuning framework (MuPaS) for LLMs on the multi-party dialogue datasets, and prove such a straightforward framework can let the LLM align with the multi-party conversation style efficiently and effectively. We also design two training strategies which can convert MuPaS into the MPD simulator. Substantial experiments show that MuPaS can achieve state-of-the-art multi-party response, higher accuracy of the-next-speaker prediction, higher human and automatic evaluated utterance qualities, and can even generate reasonably with out-of-distribution scene, topic and role descriptions. The MuPaS framework bridges the LLM training with more complicated multi-party applications, such as conversation generation, virtual rehearsal or meta-universe.
>
---
#### [replaced 013] Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.15481v3](http://arxiv.org/pdf/2406.15481v3)**

> **作者:** Haneul Yoo; Yongjin Yang; Hwaran Lee
>
> **备注:** To appear in ACL 2025
>
> **摘要:** As large language models (LLMs) have advanced rapidly, concerns regarding their safety have become prominent. In this paper, we discover that code-switching in red-teaming queries can effectively elicit undesirable behaviors of LLMs, which are common practices in natural language. We introduce a simple yet effective framework, CSRT, to synthesize codeswitching red-teaming queries and investigate the safety and multilingual understanding of LLMs comprehensively. Through extensive experiments with ten state-of-the-art LLMs and code-switching queries combining up to 10 languages, we demonstrate that the CSRT significantly outperforms existing multilingual red-teaming techniques, achieving 46.7% more attacks than standard attacks in English and being effective in conventional safety domains. We also examine the multilingual ability of those LLMs to generate and understand codeswitching texts. Additionally, we validate the extensibility of the CSRT by generating codeswitching attack prompts with monolingual data. We finally conduct detailed ablation studies exploring code-switching and propound unintended correlation between resource availability of languages and safety alignment in existing multilingual LLMs.
>
---
#### [replaced 014] One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image
- **分类: cs.CL; cs.CR; cs.CV; cs.IR**

- **链接: [http://arxiv.org/pdf/2504.02132v2](http://arxiv.org/pdf/2504.02132v2)**

> **作者:** Ezzeldin Shereen; Dan Ristea; Shae McFadden; Burak Hasircioglu; Vasilios Mavroudis; Chris Hicks
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** Multi-modal retrieval augmented generation (M-RAG) is instrumental for inhibiting hallucinations in large multi-modal models (LMMs) through the use of a factual knowledge base (KB). However, M-RAG introduces new attack vectors for adversaries that aim to disrupt the system by injecting malicious entries into the KB. In this paper, we present the first poisoning attack against M-RAG targeting visual document retrieval applications where the KB contains images of document pages. We propose two attacks, each of which require injecting only a single adversarial image into the KB. Firstly, we propose a universal attack that, for any potential user query, influences the response to cause a denial-of-service (DoS) in the M-RAG system. Secondly, we present a targeted attack against one or a group of user queries, with the goal of spreading targeted misinformation. For both attacks, we use a multi-objective gradient-based adversarial approach to craft the injected image while optimizing for both retrieval and generation. We evaluate our attacks against several visual document retrieval datasets, a diverse set of state-of-the-art retrievers (embedding models) and generators (LMMs), demonstrating the attack effectiveness in both the universal and targeted settings. We additionally present results including commonly used defenses, various attack hyper-parameter settings, ablations, and attack transferability.
>
---
#### [replaced 015] LID Models are Actually Accent Classifiers: Implications and Solutions for LID on Accented Speech
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00628v2](http://arxiv.org/pdf/2506.00628v2)**

> **作者:** Niyati Bafna; Matthew Wiesner
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Prior research indicates that LID model performance significantly declines on accented speech; however, the specific causes, extent, and characterization of these errors remain under-explored. (i) We identify a common failure mode on accented speech whereby LID systems often misclassify L2 accented speech as the speaker's native language or a related language. (ii) We present evidence suggesting that state-of-the-art models are invariant to permutations of short spans of speech, implying they classify on the basis of short phonotactic features indicative of accent rather than language. Our analysis reveals a simple method to enhance model robustness to accents through input chunking. (iii) We present an approach that integrates sequence-level information into our model without relying on monolingual ASR systems; this reduces accent-language confusion and significantly enhances performance on accented speech while maintaining comparable results on standard LID.
>
---
#### [replaced 016] ClimateViz: A Benchmark for Statistical Reasoning and Fact Verification on Scientific Charts
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.08700v2](http://arxiv.org/pdf/2506.08700v2)**

> **作者:** Ruiran Su; Jiasheng Si; Zhijiang Guo; Janet B. Pierrehumbert
>
> **摘要:** Scientific fact-checking has mostly focused on text and tables, overlooking scientific charts, which are key for presenting quantitative evidence and statistical reasoning. We introduce ClimateViz, the first large-scale benchmark for scientific fact-checking using expert-curated scientific charts. ClimateViz contains 49,862 claims linked to 2,896 visualizations, each labeled as support, refute, or not enough information. To improve interpretability, each example includes structured knowledge graph explanations covering trends, comparisons, and causal relations. We evaluate state-of-the-art multimodal language models, including both proprietary and open-source systems, in zero-shot and few-shot settings. Results show that current models struggle with chart-based reasoning: even the best systems, such as Gemini 2.5 and InternVL 2.5, reach only 76.2 to 77.8 percent accuracy in label-only settings, far below human performance (89.3 and 92.7 percent). Explanation-augmented outputs improve performance in some models. We released our dataset and code alongside the paper.
>
---
#### [replaced 017] TACTIC: Translation Agents with Cognitive-Theoretic Interactive Collaboration
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.08403v2](http://arxiv.org/pdf/2506.08403v2)**

> **作者:** Weiya Li; Junjie Chen; Bei Li; Boyang Liu; Zichen Wen; Nuanqiao Shan; Xiaoqian Liu; Anping Liu; Huajie Liu; Hu Song; Linfeng Zhang
>
> **备注:** 20 pages, 4 figures, Under review. Code: https://github.com/weiyali126/TACTIC
>
> **摘要:** Machine translation has long been a central task in natural language processing. With the rapid advancement of large language models (LLMs), there has been remarkable progress in translation quality. However, fully realizing the translation potential of LLMs remains an open challenge. Recent studies have explored multi-agent systems to decompose complex translation tasks into collaborative subtasks, showing initial promise in enhancing translation quality through agent cooperation and specialization. Nevertheless, existing multi-agent translation frameworks largely neglect foundational insights from cognitive translation studies. These insights emphasize how human translators employ different cognitive strategies, such as balancing literal and free translation, refining expressions based on context, and iteratively evaluating outputs. To address this limitation, we propose a cognitively informed multi-agent framework called TACTIC, which stands for T ranslation A gents with Cognitive- T heoretic Interactive Collaboration. The framework comprises six functionally distinct agents that mirror key cognitive processes observed in human translation behavior. These include agents for drafting, refinement, evaluation, scoring, context reasoning, and external knowledge gathering. By simulating an interactive and theory-grounded translation workflow, TACTIC effectively leverages the full capacity of LLMs for high-quality translation. Experimental results on diverse language pairs from the FLORES-200 and WMT24 benchmarks show that our method consistently achieves state-of-the-art performance. Using DeepSeek-V3 as the base model, TACTIC surpasses GPT-4.1 by an average of +0.6 XCOMET and +1.18 COMETKIWI-23. Compared to DeepSeek-R1, it further improves by +0.84 XCOMET and +2.99 COMETKIWI-23. Code is available at https://github.com/weiyali126/TACTIC.
>
---
#### [replaced 018] How Do Multilingual Language Models Remember Facts?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.14387v3](http://arxiv.org/pdf/2410.14387v3)**

> **作者:** Constanza Fierro; Negar Foroutan; Desmond Elliott; Anders Søgaard
>
> **备注:** 9 pages
>
> **摘要:** Large Language Models (LLMs) store and retrieve vast amounts of factual knowledge acquired during pre-training. Prior research has localized and identified mechanisms behind knowledge recall; however, it has only focused on English monolingual models. The question of how these mechanisms generalize to non-English languages and multilingual LLMs remains unexplored. In this paper, we address this gap by conducting a comprehensive analysis of three multilingual LLMs. First, we show that previously identified recall mechanisms in English largely apply to multilingual contexts, with nuances based on language and architecture. Next, through patching intermediate representations, we localize the role of language during recall, finding that subject enrichment is language-independent, while object extraction is language-dependent. Additionally, we discover that the last token representation acts as a Function Vector (FV), encoding both the language of the query and the content to be extracted from the subject. Furthermore, in decoder-only LLMs, FVs compose these two pieces of information in two separate stages. These insights reveal unique mechanisms in multilingual LLMs for recalling information, highlighting the need for new methodologies -- such as knowledge evaluation, fact editing, and knowledge acquisition -- that are specifically tailored for multilingual LLMs.
>
---
#### [replaced 019] CiteFix: Enhancing RAG Accuracy Through Post-Processing Citation Correction
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.15629v2](http://arxiv.org/pdf/2504.15629v2)**

> **作者:** Harsh Maheshwari; Srikanth Tenneti; Alwarappan Nakkiran
>
> **摘要:** Retrieval Augmented Generation (RAG) has emerged as a powerful application of Large Language Models (LLMs), revolutionizing information search and consumption. RAG systems combine traditional search capabilities with LLMs to generate comprehensive answers to user queries, ideally with accurate citations. However, in our experience of developing a RAG product, LLMs often struggle with source attribution, aligning with other industry studies reporting citation accuracy rates of only about 74% for popular generative search engines. To address this, we present efficient post-processing algorithms to improve citation accuracy in LLM-generated responses, with minimal impact on latency and cost. Our approaches cross-check generated citations against retrieved articles using methods including keyword + semantic matching, fine tuned model with BERTScore, and a lightweight LLM-based technique. Our experimental results demonstrate a relative improvement of 15.46% in the overall accuracy metrics of our RAG system. This significant enhancement potentially enables a shift from our current larger language model to a relatively smaller model that is approximately 12x more cost-effective and 3x faster in inference time, while maintaining comparable performance. This research contributes to enhancing the reliability and trustworthiness of AI-generated content in information retrieval and summarization tasks which is critical to gain customer trust especially in commercial products.
>
---
#### [replaced 020] Rethinking Diverse Human Preference Learning through Principal Component Analysis
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13131v2](http://arxiv.org/pdf/2502.13131v2)**

> **作者:** Feng Luo; Rui Yang; Hao Sun; Chunyuan Deng; Jiarui Yao; Jingyan Shen; Huan Zhang; Hanjie Chen
>
> **备注:** 14 pages
>
> **摘要:** Understanding human preferences is crucial for improving foundation models and building personalized AI systems. However, preferences are inherently diverse and complex, making it difficult for traditional reward models to capture their full range. While fine-grained preference data can help, collecting it is expensive and hard to scale. In this paper, we introduce Decomposed Reward Models (DRMs), a novel approach that extracts diverse human preferences from binary comparisons without requiring fine-grained annotations. Our key insight is to represent human preferences as vectors and analyze them using Principal Component Analysis (PCA). By constructing a dataset of embedding differences between preferred and rejected responses, DRMs identify orthogonal basis vectors that capture distinct aspects of preference. These decomposed rewards can be flexibly combined to align with different user needs, offering an interpretable and scalable alternative to traditional reward models. We demonstrate that DRMs effectively extract meaningful preference dimensions (e.g., helpfulness, safety, humor) and adapt to new users without additional training. Our results highlight DRMs as a powerful framework for personalized and interpretable LLM alignment. Our code is available at https://github.com/amandaluof/DRMs.
>
---
#### [replaced 021] MAGIC-VQA: Multimodal And Grounded Inference with Commonsense Knowledge for Visual Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.18491v3](http://arxiv.org/pdf/2503.18491v3)**

> **作者:** Shuo Yang; Siwen Luo; Soyeon Caren Han; Eduard Hovy
>
> **备注:** Findings of ACL 2025
>
> **摘要:** Visual Question Answering (VQA) requires reasoning across visual and textual modalities, yet Large Vision-Language Models (LVLMs) often lack integrated commonsense knowledge, limiting their robustness in real-world scenarios. To address this, we introduce MAGIC-VQA, a novel framework that enhances VQA by systematically integrating commonsense knowledge with LVLMs. MAGIC-VQA employs a three-stage process: (1) Explicit Knowledge Integration from external sources, (2) By-Type Post-Processing for contextual refinement, and (3) Implicit Knowledge Augmentation using a Graph Neural Network (GNN) for structured reasoning. While GNNs bring greater depth to structured inference, they enable superior relational inference beyond LVLMs. MAGIC-VQA bridges a key gap by unifying commonsensse knowledge with LVLM-driven reasoning, eliminating the need for extensive pre-training or complex prompt tuning. Our framework achieves state-of-the-art performance on benchmark datasets, significantly improving commonsense reasoning in VQA.
>
---
#### [replaced 022] Irony Detection, Reasoning and Understanding in Zero-shot Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.16884v2](http://arxiv.org/pdf/2501.16884v2)**

> **作者:** Peiling Yi; Yuhan Xia; Yunfei Long
>
> **摘要:** The generalisation of irony detection faces significant challenges, leading to substantial performance deviations when detection models are applied to diverse real-world scenarios. In this study, we find that irony-focused prompts, as generated from our IDADP framework for LLMs, can not only overcome dataset-specific limitations but also generate coherent, human-readable reasoning, transforming ironic text into its intended meaning. Based on our findings and in-depth analysis, we identify several promising directions for future research aimed at enhancing LLMs' zero-shot capabilities in irony detection, reasoning, and comprehension. These include advancing contextual awareness in irony detection, exploring hybrid symbolic-neural methods, and integrating multimodal data, among others.
>
---
#### [replaced 023] AraReasoner: Evaluating Reasoning-Based LLMs for Arabic NLP
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08768v2](http://arxiv.org/pdf/2506.08768v2)**

> **作者:** Ahmed Hasanaath; Aisha Alansari; Ahmed Ashraf; Chafik Salmane; Hamzah Luqman; Saad Ezzini
>
> **摘要:** Large language models (LLMs) have shown remarkable progress in reasoning abilities and general natural language processing (NLP) tasks, yet their performance on Arabic data, characterized by rich morphology, diverse dialects, and complex script, remains underexplored. This paper presents a comprehensive benchmarking study of multiple reasoning-focused LLMs, with a special emphasis on the newly introduced DeepSeek models, across a suite of fifteen Arabic NLP tasks. We experiment with various strategies, including zero-shot, few-shot, and fine-tuning. This allows us to systematically evaluate performance on datasets covering a range of applications to examine their capacity for linguistic reasoning under different levels of complexity. Our experiments reveal several key findings. First, carefully selecting just three in-context examples delivers an average uplift of over 13 F1 points on classification tasks-boosting sentiment analysis from 35.3% to 87.5% and paraphrase detection from 56.1% to 87.0%. Second, reasoning-focused DeepSeek architectures outperform a strong GPT o4-mini baseline by an average of 12 F1 points on complex inference tasks in the zero-shot setting. Third, LoRA-based fine-tuning yields up to an additional 8 points in F1 and BLEU compared to equivalent increases in model scale. The code is available at https://anonymous.4open.science/r/AraReasoner41299
>
---
#### [replaced 024] Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.06975v3](http://arxiv.org/pdf/2506.06975v3)**

> **作者:** Xiaoyuan Zhu; Yaowen Ye; Tianyi Qiu; Hanlin Zhu; Sijun Tan; Ajraf Mannan; Jonathan Michala; Raluca Ada Popa; Willie Neiswanger
>
> **摘要:** As API access becomes a primary interface to large language models (LLMs), users often interact with black-box systems that offer little transparency into the deployed model. To reduce costs or maliciously alter model behaviors, API providers may discreetly serve quantized or fine-tuned variants, which can degrade performance and compromise safety. Detecting such substitutions is difficult, as users lack access to model weights and, in most cases, even output logits. To tackle this problem, we propose a rank-based uniformity test that can verify the behavioral equality of a black-box LLM to a locally deployed authentic model. Our method is accurate, query-efficient, and avoids detectable query patterns, making it robust to adversarial providers that reroute or mix responses upon the detection of testing attempts. We evaluate the approach across diverse threat scenarios, including quantization, harmful fine-tuning, jailbreak prompts, and full model substitution, showing that it consistently achieves superior statistical power over prior methods under constrained query budgets.
>
---
#### [replaced 025] Why Are Web AI Agents More Vulnerable Than Standalone LLMs? A Security Analysis
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20383v2](http://arxiv.org/pdf/2502.20383v2)**

> **作者:** Jeffrey Yang Fan Chiang; Seungjae Lee; Jia-Bin Huang; Furong Huang; Yizheng Chen
>
> **备注:** Project website: http://vulnerable-ai-agents.github.io
>
> **摘要:** Recent advancements in Web AI agents have demonstrated remarkable capabilities in addressing complex web navigation tasks. However, emerging research shows that these agents exhibit greater vulnerability compared to standalone Large Language Models (LLMs), despite both being built upon the same safety-aligned models. This discrepancy is particularly concerning given the greater flexibility of Web AI Agent compared to standalone LLMs, which may expose them to a wider range of adversarial user inputs. To build a scaffold that addresses these concerns, this study investigates the underlying factors that contribute to the increased vulnerability of Web AI agents. Notably, this disparity stems from the multifaceted differences between Web AI agents and standalone LLMs, as well as the complex signals - nuances that simple evaluation metrics, such as success rate, often fail to capture. To tackle these challenges, we propose a component-level analysis and a more granular, systematic evaluation framework. Through this fine-grained investigation, we identify three critical factors that amplify the vulnerability of Web AI agents; (1) embedding user goals into the system prompt, (2) multi-step action generation, and (3) observational capabilities. Our findings highlights the pressing need to enhance security and robustness in AI agent design and provide actionable insights for targeted defense strategies.
>
---
#### [replaced 026] Low-resource domain adaptation while minimizing energy and hardware resource consumption
- **分类: cs.CL; cs.DC; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.08433v2](http://arxiv.org/pdf/2506.08433v2)**

> **作者:** Hernán Maina; Nicolás Wolovick; Luciana Benotti
>
> **备注:** A shorter version of this work was accepted as a two-page abstract for presentation at the Widening Natural Language Processing (WiNLP) 2023 Workshop. That version was not publicly released, and this is the first public version of the work
>
> **摘要:** Training Large Language Models (LLMs) is costly in terms of energy, hardware, and annotated data, often resulting in a positionality rooted in predominant cultures and values (Santy et al., 2023). Domain adaptation has emerged as a promising strategy to better align models with diverse cultural and value contexts (Hershcovich et al., 2022), but its computational cost remains a significant barrier, particularly for research groups lacking access to large-scale infrastructure. In this paper, we evaluate how the use of different numerical precision formats and data parallelization strategies impacts both training speed (as a proxy to energy and hardware consumption) and model accuracy, with the goal of facilitating domain adaptation in low-resource environments. Our findings are relevant to any setting where energy efficiency, accessibility, or limited hardware availability are key concerns.
>
---
#### [replaced 027] AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking
- **分类: cs.CL; cs.AI; cs.SC**

- **链接: [http://arxiv.org/pdf/2506.07751v2](http://arxiv.org/pdf/2506.07751v2)**

> **作者:** Silin Gao; Antoine Bosselut; Samy Bengio; Emmanuel Abbe
>
> **备注:** Under review
>
> **摘要:** Recent studies have shown that large language models (LLMs), especially smaller ones, often lack robustness in their reasoning. I.e., they tend to experience performance drops when faced with distribution shifts, such as changes to numerical or nominal variables, or insertions of distracting clauses. A possible strategy to address this involves generating synthetic data to further "instantiate" reasoning problems on potential variations. In contrast, our approach focuses on "abstracting" reasoning problems. This not only helps counteract distribution shifts but also facilitates the connection to symbolic tools for deriving solutions. We find that this abstraction process is better acquired through reinforcement learning (RL) than just supervised fine-tuning, which often fails to produce faithful abstractions. Our method, AbstRaL -- which promotes abstract reasoning in LLMs using RL on granular abstraction data -- significantly mitigates performance degradation on recent GSM perturbation benchmarks.
>
---
#### [replaced 028] Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.07859v2](http://arxiv.org/pdf/2505.07859v2)**

> **作者:** Daniel Franzen; Jan Disselhoff; David Hartmann
>
> **备注:** ICML 2025 camera-ready; 15 pages, 6 figures, 5 tables
>
> **摘要:** The Abstraction and Reasoning Corpus (ARC-AGI) poses a significant challenge for large language models (LLMs), exposing limitations in their abstract reasoning abilities. In this work, we leverage task-specific data augmentations throughout the training, generation, and scoring phases, and employ a depth-first search algorithm to generate diverse, high-probability candidate solutions. Furthermore, we utilize the LLM not only as a generator but also as a scorer, using its output probabilities to select the most promising solutions. Our method achieves a score of 71.6% (286.5/400 solved tasks) on the public ARC-AGI evaluation set, demonstrating state-of-the-art performance among publicly available approaches. While concurrent closed-source work has reported higher scores, our method distinguishes itself through its transparency, reproducibility, and remarkably low inference cost, averaging only around 2ct per task on readily available hardware (we assume a price of 36ct/hour for a Nvidia 4090 GPU).
>
---
#### [replaced 029] Sentence-level Reward Model can Generalize Better for Aligning LLM from Human Preference
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.04793v4](http://arxiv.org/pdf/2503.04793v4)**

> **作者:** Wenjie Qiu; Yi-Chen Li; Xuqin Zhang; Tianyi Zhang; Yihang Zhang; Zongzhang Zhang; Yang Yu
>
> **摘要:** Learning reward models from human preference datasets and subsequently optimizing language models via reinforcement learning has emerged as a fundamental paradigm for aligning LLMs with human preferences. The performance of the reward model plays a crucial role in the effectiveness of alignment. Previous reward models operate at a coarse-grained level, requiring the generation of a complete response to obtain a reward value. The sparse reward may present challenges for downstream reinforcement learning. While recent efforts have attempted to learn token-level reward models, the lack of explicit semantic information makes it difficult to model the credit of every individual token. In this paper, we propose assigning scores to every sentence, introducing an intermediate-grained reward model. By segmenting the complete response into sentences and applying differential operations to reward output at the start and end positions of each sentence, we can effectively model the rewards of sentences. Moreover, a novel attention mechanism is introduced to aggregate the scores of all sentences into a response-level score, which allows it to be trained using the Bradley-Terry model. On common benchmarks, our method outperforms the response-level reward model by 2.7% on RewardBench (for reward modeling evaluation) and surpasses all baselines on AlpacaEval (for alignment evaluation).
>
---
#### [replaced 030] Convert Language Model into a Value-based Strategic Planner
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.06987v2](http://arxiv.org/pdf/2505.06987v2)**

> **作者:** Xiaoyu Wang; Yue Zhao; Qingqing Gu; Zhonglin Jiang; Xiaokai Chen; Yong Chen; Luo Ji
>
> **备注:** 13 pages, 6 figures, Accepted by ACL 2025 Industry Track
>
> **摘要:** Emotional support conversation (ESC) aims to alleviate the emotional distress of individuals through effective conversations. Although large language models (LLMs) have obtained remarkable progress on ESC, most of these studies might not define the diagram from the state model perspective, therefore providing a suboptimal solution for long-term satisfaction. To address such an issue, we leverage the Q-learning on LLMs, and propose a framework called straQ*. Our framework allows a plug-and-play LLM to bootstrap the planning during ESC, determine the optimal strategy based on long-term returns, and finally guide the LLM to response. Substantial experiments on ESC datasets suggest that straQ* outperforms many baselines, including direct inference, self-refine, chain of thought, finetuning, and finite state machines.
>
---
#### [replaced 031] Language Models Resist Alignment: Evidence From Data Compression
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.06144v4](http://arxiv.org/pdf/2406.06144v4)**

> **作者:** Jiaming Ji; Kaile Wang; Tianyi Qiu; Boyuan Chen; Jiayi Zhou; Changye Li; Hantao Lou; Juntao Dai; Yunhuai Liu; Yaodong Yang
>
> **备注:** Accepted by ACL2025 Main
>
> **摘要:** Large language models (LLMs) may exhibit unintended or undesirable behaviors. Recent works have concentrated on aligning LLMs to mitigate harmful outputs. Despite these efforts, some anomalies indicate that even a well-conducted alignment process can be easily circumvented, whether intentionally or accidentally. Does alignment fine-tuning yield have robust effects on models, or are its impacts merely superficial? In this work, we make the first exploration of this phenomenon from both theoretical and empirical perspectives. Empirically, we demonstrate the $\mathbf{elasticity}$ of post-alignment models, i.e., the tendency to revert to the behavior distribution formed during the pre-training phase upon further fine-tuning. Leveraging compression theory, we formally deduce that fine-tuning disproportionately undermines alignment relative to pre-training, potentially by orders of magnitude. We validate the presence of elasticity through experiments on models of varying types and scales. Specifically, we find that model performance declines rapidly before reverting to the pre-training distribution, after which the rate of decline drops significantly. Furthermore, we further reveal that elasticity positively correlates with the increased model size and the expansion of pre-training data. Our findings underscore the need to address the inherent elasticity of LLMs to mitigate their resistance to alignment. The model weight and code are available at pku-lm-resist-alignment.github.io.
>
---
#### [replaced 032] Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.08952v2](http://arxiv.org/pdf/2506.08952v2)**

> **作者:** Clara Lachenmaier; Judith Sieker; Sina Zarrieß
>
> **备注:** Preprint accepted at ACL Main Conference 2025
>
> **摘要:** Communication among humans relies on conversational grounding, allowing interlocutors to reach mutual understanding even when they do not have perfect knowledge and must resolve discrepancies in each other's beliefs. This paper investigates how large language models (LLMs) manage common ground in cases where they (don't) possess knowledge, focusing on facts in the political domain where the risk of misinformation and grounding failure is high. We examine the ability of LLMs to answer direct knowledge questions and loaded questions that presuppose misinformation. We evaluate whether loaded questions lead LLMs to engage in active grounding and correct false user beliefs, in connection to their level of knowledge and their political bias. Our findings highlight significant challenges in LLMs' ability to engage in grounding and reject false user beliefs, raising concerns about their role in mitigating misinformation in political discourse.
>
---
#### [replaced 033] Steps are all you need: Rethinking STEM Education with Prompt Engineering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.05023v3](http://arxiv.org/pdf/2412.05023v3)**

> **作者:** Krishnasai Addala; Kabir Dev Paul Baghel; Navya Gupta; Rishitej Reddy Vyalla; Chhavi Kirtani; Avinash Anand; Rajiv Ratn Shah
>
> **摘要:** Few shot and Chain-of-Thought prompting have shown promise when applied to Physics Question Answering Tasks, but are limited by the lack of mathematical ability inherent to LLMs, and are prone to hallucination. By utilizing a Mixture of Experts (MoE) Model, along with analogical prompting, we are able to show improved model performance when compared to the baseline on standard LLMs. We also survey the limits of these prompting techniques and the effects they have on model performance. Additionally, we propose Analogical CoT prompting, a prompting technique designed to allow smaller, open source models to leverage Analogical prompting, something they have struggled with, possibly due to a lack of specialist training data.
>
---
#### [replaced 034] ICONS: Influence Consensus for Vision-Language Data Selection
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.00654v3](http://arxiv.org/pdf/2501.00654v3)**

> **作者:** Xindi Wu; Mengzhou Xia; Rulin Shao; Zhiwei Deng; Pang Wei Koh; Olga Russakovsky
>
> **备注:** 31 pages, 19 figures
>
> **摘要:** Training vision-language models via instruction tuning often relies on large mixtures of data spanning diverse tasks and domains. However, these mixtures frequently include redundant information, increasing computational costs without proportional performance gains, necessitating more effective data selection strategies. Existing methods typically rely on task-agnostic heuristics to estimate data importance or focus on optimizing single tasks in isolation, limiting their effectiveness in multitask settings. In this work, we introduce ICONS, a gradient-based Influence CONsensus approach for vision-language data Selection. Our method leverages first-order training dynamics to estimate the influence of individual training examples on validation performance and aggregates these estimates across tasks via majority voting over task-specific influences. This cross-task consensus identifies data points that are consistently valuable across tasks, enabling us to prioritize examples that drive overall performance. The voting-based design further mitigates issues such as score calibration and outlier sensitivity, resulting in robust and scalable data selection for diverse multitask mixtures. With only 20% of the data from LLaVA-665K and Cambrian-7M, our selected subsets retain 98.6% and 98.8% of the performance achieved with full datasets, and can even surpass full data training at a 60% selection ratio on LLaVA-665K. Our approach also generalizes to unseen tasks and architectures, demonstrating strong transfer. We release two compact, high-utility subsets, LLaVA-ICONS-133K and Cambrian-ICONS-1.4M, preserving impactful training examples for efficient and scalable vision-language model development.
>
---
#### [replaced 035] Code-Switching Curriculum Learning for Multilingual Transfer in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.02460v2](http://arxiv.org/pdf/2411.02460v2)**

> **作者:** Haneul Yoo; Cheonbok Park; Sangdoo Yun; Alice Oh; Hwaran Lee
>
> **备注:** To appear in Findings of ACL 2025
>
> **摘要:** Large language models (LLMs) now exhibit near human-level performance in various tasks, but their performance drops drastically after a handful of high-resource languages due to the imbalance in pre-training data. Inspired by the human process of second language acquisition, particularly code-switching$\unicode{x2014}$the practice of language alternation in a conversation$\unicode{x2014}$we propose code-switching curriculum learning (CSCL) to enhance cross-lingual transfer for LLMs. CSCL mimics the stages of human language learning by progressively training models with a curriculum consisting of 1) token-level code-switching, 2) sentence-level code-switching, and 3) monolingual corpora. Using Qwen 2 as our underlying model, we demonstrate the efficacy of the CSCL in improving language transfer to Korean, achieving significant performance gains compared to monolingual continual pre-training methods. Ablation studies reveal that both token- and sentence-level code-switching significantly enhance cross-lingual transfer and that curriculum learning amplifies these effects. We also extend our findings into various languages, including Japanese (high-resource) and Indonesian (low-resource), and using two additional models (Gemma 2 and Phi 3.5). We further show that CSCL mitigates spurious correlations between language resources and safety alignment, presenting a robust, efficient framework for more equitable language transfer in LLMs. We observe that CSCL is effective for low-resource settings where high-quality, monolingual corpora for language transfer are hardly available.
>
---
#### [replaced 036] Let's Fuse Step by Step: A Generative Fusion Decoding Algorithm with LLMs for Robust and Instruction-Aware ASR and OCR
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.14259v4](http://arxiv.org/pdf/2405.14259v4)**

> **作者:** Chan-Jan Hsu; Yi-Chang Chen; Feng-Ting Liao; Pei-Chen Ho; Yu-Hsiang Wang; Po-Chun Hsu; Da-shan Shiu
>
> **摘要:** We propose "Generative Fusion Decoding" (GFD), a novel shallow fusion framework designed to integrate large language models (LLMs) into cross-modal text recognition systems for automatic speech recognition (ASR) and optical character recognition (OCR). We derive the necessary formulations to enable GFD to operate across mismatched token spaces of different models by calculating likelihood at the byte level, thereby enabling seamless fusion and synchronous progression during the decoding process. GFD is plug-and-play by design, making it readily compatible with various auto-regressive models without the need for any re-training. GFD proves effective for general ASR and OCR tasks through intermediate and frequent interactions with LLMs, surpassing cascaded methods in English and Mandarin benchmarks. In addition, GFD transfers in-context learning abilities of LLMs and allows for adaptive ASR in instruction-aware and long-context settings, yielding significant WER reductions of up to 17.7\%.
>
---
#### [replaced 037] ImageChain: Advancing Sequential Image-to-Text Reasoning in Multimodal Large Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19409v2](http://arxiv.org/pdf/2502.19409v2)**

> **作者:** Danae Sánchez Villegas; Ingo Ziegler; Desmond Elliott
>
> **备注:** Code, dataset, and checkpoints are publicly available at https://github.com/danaesavi/ImageChain; v2: added human annotation study to validate SimRate
>
> **摘要:** Reasoning over sequences of images remains a challenge for multimodal large language models (MLLMs). While recent models incorporate multi-image data during pre-training, they still struggle to recognize sequential structures, often treating images independently. This work introduces ImageChain, a framework that enhances MLLMs with sequential reasoning capabilities over image data by modeling visual sequences as a multi-turn conversation. In ImageChain, images are interleaved with corresponding textual descriptions to form a controlled dialogue that explicitly captures temporal dependencies and narrative progression. Our method optimizes for the task of next-scene description, where the model generates a context-aware description of an upcoming scene based on preceding visual and textual cues. We demonstrate that our approach improves performance on the next-scene description task -- achieving an average improvement from 3.7% to 19% in SimRate, a metric that quantifies semantic similarity to human-annotated ground truths. Moreover, ImageChain achieves robust zero-shot out-of-domain performance in applications ranging from comics to robotics. Extensive experiments validate that instruction-tuning in a multimodal, multi-turn conversation design is key to bridging the gap between static image understanding and temporally-aware reasoning.
>
---
#### [replaced 038] CiteFusion: An Ensemble Framework for Citation Intent Classification Harnessing Dual-Model Binary Couples and SHAP Analyses
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.13329v3](http://arxiv.org/pdf/2407.13329v3)**

> **作者:** Lorenzo Paolini; Sahar Vahdati; Angelo Di Iorio; Robert Wardenga; Ivan Heibi; Silvio Peroni
>
> **备注:** Submitted to Scientometrics Journal
>
> **摘要:** Understanding the motivations underlying scholarly citations is essential to evaluate research impact and pro-mote transparent scholarly communication. This study introduces CiteFusion, an ensemble framework designed to address the multi-class Citation Intent Classification task on two benchmark datasets: SciCite and ACL-ARC. The framework employs a one-vs-all decomposition of the multi-class task into class-specific binary sub-tasks, leveraging complementary pairs of SciBERT and XLNet models, independently tuned, for each citation intent. The outputs of these base models are aggregated through a feedforward neural network meta-classifier to reconstruct the original classification task. To enhance interpretability, SHAP (SHapley Additive exPlanations) is employed to analyze token-level contributions, and interactions among base models, providing transparency into the classification dynamics of CiteFusion, and insights about the kind of misclassifications of the ensem-ble. In addition, this work investigates the semantic role of structural context by incorporating section titles, as framing devices, into input sentences, assessing their positive impact on classification accuracy. CiteFusion ul-timately demonstrates robust performance in imbalanced and data-scarce scenarios: experimental results show that CiteFusion achieves state-of-the-art performance, with Macro-F1 scores of 89.60% on SciCite, and 76.24% on ACL-ARC. Furthermore, to ensure interoperability and reusability, citation intents from both datasets sche-mas are mapped to Citation Typing Ontology (CiTO) object properties, highlighting some overlaps. Finally, we describe and release a web-based application that classifies citation intents leveraging the CiteFusion models developed on SciCite.
>
---
#### [replaced 039] Self-Steering Optimization: Autonomous Preference Optimization for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.17131v2](http://arxiv.org/pdf/2410.17131v2)**

> **作者:** Hao Xiang; Bowen Yu; Hongyu Lin; Keming Lu; Yaojie Lu; Xianpei Han; Ben He; Le Sun; Jingren Zhou; Junyang Lin
>
> **摘要:** The key to effective alignment lies in high-quality preference data. Recent research has focused on automated alignment, which involves developing alignment systems with minimal human intervention. However, prior research has predominantly focused on developing data generation methods, while insufficient attention has been paid to quality control mechanisms, which often produce inaccurate and unhelpful data, leading to unpredictable benefits during iterative optimization. In this paper, we present Self-Steering Optimization ($SSO$), an algorithm that autonomously generates high-quality preference data, eliminating manual annotation requirements. $SSO$ employs a specialized optimization objective to build a data generator from the policy model itself, which is used to produce accurate and on-policy data. We demonstrate $SSO$'s effectiveness through comprehensive experiments on two series of models: Llama 3 and Qwen 2. Our evaluation across diverse benchmarks shows that $SSO$ consistently outperforms baselines in human preference alignment and reward optimization. Further analysis validates $SSO$ as a scalable framework for preference optimization, benefiting the advancement in automated alignment techniques.
>
---
#### [replaced 040] Measuring What Makes You Unique: Difference-Aware User Modeling for Enhancing LLM Personalization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.02450v3](http://arxiv.org/pdf/2503.02450v3)**

> **作者:** Yilun Qiu; Xiaoyan Zhao; Yang Zhang; Yimeng Bai; Wenjie Wang; Hong Cheng; Fuli Feng; Tat-Seng Chua
>
> **备注:** 2025 ACL Findings
>
> **摘要:** Personalizing Large Language Models (LLMs) has become a critical step in facilitating their widespread application to enhance individual life experiences. In pursuit of personalization, distilling key preference information from an individual's historical data as instructional preference context to customize LLM generation has emerged as a promising direction. However, these methods face a fundamental limitation by overlooking the inter-user comparative analysis, which is essential for identifying the inter-user differences that truly shape preferences. To address this limitation, we propose Difference-aware Personalization Learning (DPL), a novel approach that emphasizes extracting inter-user differences to enhance LLM personalization. DPL strategically selects representative users for comparison and establishes a structured standard to extract meaningful, task-relevant differences for customizing LLM generation. Extensive experiments on real-world datasets demonstrate that DPL significantly enhances LLM personalization. We release our code at https://github.com/SnowCharmQ/DPL.
>
---
#### [replaced 041] TableEval: A Real-World Benchmark for Complex, Multilingual, and Multi-Structured Table Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.03949v2](http://arxiv.org/pdf/2506.03949v2)**

> **作者:** Junnan Zhu; Jingyi Wang; Bohan Yu; Xiaoyu Wu; Junbo Li; Lei Wang; Nan Xu
>
> **摘要:** LLMs have shown impressive progress in natural language processing. However, they still face significant challenges in TableQA, where real-world complexities such as diverse table structures, multilingual data, and domain-specific reasoning are crucial. Existing TableQA benchmarks are often limited by their focus on simple flat tables and suffer from data leakage. Furthermore, most benchmarks are monolingual and fail to capture the cross-lingual and cross-domain variability in practical applications. To address these limitations, we introduce TableEval, a new benchmark designed to evaluate LLMs on realistic TableQA tasks. Specifically, TableEval includes tables with various structures (such as concise, hierarchical, and nested tables) collected from four domains (including government, finance, academia, and industry reports). Besides, TableEval features cross-lingual scenarios with tables in Simplified Chinese, Traditional Chinese, and English. To minimize the risk of data leakage, we collect all data from recent real-world documents. Considering that existing TableQA metrics fail to capture semantic accuracy, we further propose SEAT, a new evaluation framework that assesses the alignment between model responses and reference answers at the sub-question level. Experimental results have shown that SEAT achieves high agreement with human judgment. Extensive experiments on TableEval reveal critical gaps in the ability of state-of-the-art LLMs to handle these complex, real-world TableQA tasks, offering insights for future improvements. We make our dataset available here: https://github.com/wenge-research/TableEval.
>
---
#### [replaced 042] Unveiling the Hidden: Movie Genre and User Bias in Spoiler Detection
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.17834v3](http://arxiv.org/pdf/2504.17834v3)**

> **作者:** Haokai Zhang; Shengtao Zhang; Zijian Cai; Heng Wang; Ruixuan Zhu; Zinan Zeng; Minnan Luo
>
> **备注:** ECML PKDD 2025
>
> **摘要:** Spoilers in movie reviews are important on platforms like IMDb and Rotten Tomatoes, offering benefits and drawbacks. They can guide some viewers' choices but also affect those who prefer no plot details in advance, making effective spoiler detection essential. Existing spoiler detection methods mainly analyze review text, often overlooking the impact of movie genres and user bias, limiting their effectiveness. To address this, we analyze movie review data, finding genre-specific variations in spoiler rates and identifying that certain users are more likely to post spoilers. Based on these findings, we introduce a new spoiler detection framework called GUSD (The code is available at https://github.com/AI-explorer-123/GUSD) (Genre-aware and User-specific Spoiler Detection), which incorporates genre-specific data and user behavior bias. User bias is calculated through dynamic graph modeling of review history. Additionally, the R2GFormer module combines RetGAT (Retentive Graph Attention Network) for graph information and GenreFormer for genre-specific aggregation. The GMoE (Genre-Aware Mixture of Experts) model further assigns reviews to specialized experts based on genre. Extensive testing on benchmark datasets shows that GUSD achieves state-of-the-art results. This approach advances spoiler detection by addressing genre and user-specific patterns, enhancing user experience on movie review platforms.
>
---
#### [replaced 043] Multimodal Inconsistency Reasoning (MMIR): A New Benchmark for Multimodal Reasoning Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.16033v3](http://arxiv.org/pdf/2502.16033v3)**

> **作者:** Qianqi Yan; Yue Fan; Hongquan Li; Shan Jiang; Yang Zhao; Xinze Guan; Ching-Chen Kuo; Xin Eric Wang
>
> **摘要:** Existing Multimodal Large Language Models (MLLMs) are predominantly trained and tested on consistent visual-textual inputs, leaving open the question of whether they can handle inconsistencies in real-world, layout-rich content. To bridge this gap, we propose the Multimodal Inconsistency Reasoning (MMIR) benchmark to assess MLLMs' ability to detect and reason about semantic mismatches in artifacts such as webpages, presentation slides, and posters. MMIR comprises 534 challenging samples, each containing synthetically injected errors across five reasoning-heavy categories: Factual Contradiction, Identity Misattribution, Contextual Mismatch, Quantitative Discrepancy, and Temporal/Spatial Incoherence. We evaluate six state-of-the-art MLLMs, showing that models with dedicated multimodal reasoning capabilities, such as o1, substantially outperform their counterparts while open-source models remain particularly vulnerable to inconsistency errors. Detailed error analyses further show that models excel in detecting pairwise inconsistencies but struggle with inconsistencies confined to single elements in complex layouts. Probing experiments reveal that single-modality prompting, including Chain-of-Thought (CoT) and Set-of-Mark (SoM) methods, yields marginal gains, revealing a key bottleneck in cross-modal reasoning. Our findings highlight the need for advanced multimodal reasoning and point to future research on multimodal inconsistency.
>
---
#### [replaced 044] UD-KSL Treebank v1.3: A semi-automated framework for aligning XPOS-extracted units with UPOS tags
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09009v2](http://arxiv.org/pdf/2506.09009v2)**

> **作者:** Hakyung Sung; Gyu-Ho Shin; Chanyoung Lee; You Kyung Sung; Boo Kyung Jung
>
> **摘要:** The present study extends recent work on Universal Dependencies annotations for second-language (L2) Korean by introducing a semi-automated framework that identifies morphosyntactic constructions from XPOS sequences and aligns those constructions with corresponding UPOS categories. We also broaden the existing L2-Korean corpus by annotating 2,998 new sentences from argumentative essays. To evaluate the impact of XPOS-UPOS alignments, we fine-tune L2-Korean morphosyntactic analysis models on datasets both with and without these alignments, using two NLP toolkits. Our results indicate that the aligned dataset not only improves consistency across annotation layers but also enhances morphosyntactic tagging and dependency-parsing accuracy, particularly in cases of limited annotated data.
>
---
#### [replaced 045] LIFEBench: Evaluating Length Instruction Following in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16234v2](http://arxiv.org/pdf/2505.16234v2)**

> **作者:** Wei Zhang; Zhenhong Zhou; Kun Wang; Junfeng Fang; Yuanhe Zhang; Rui Wang; Ge Zhang; Xavier Li; Li Sun; Lingjuan Lyu; Yang Liu; Sen Su
>
> **备注:** 81 pages, 22 tables, 32 figures. Homepage: https://ydyjya.github.io/LIFEBench/
>
> **摘要:** While large language models (LLMs) can solve PhD-level reasoning problems over long context inputs, they still struggle with a seemingly simpler task: following explicit length instructions-e.g., write a 10,000-word novel. Additionally, models often generate far too short outputs, terminate prematurely, or even refuse the request. Existing benchmarks focus primarily on evaluating generations quality, but often overlook whether the generations meet length constraints. To this end, we introduce Length Instruction Following Evaluation Benchmark (LIFEBench) to comprehensively evaluate LLMs' ability to follow length instructions across diverse tasks and a wide range of specified lengths. LIFEBench consists of 10,800 instances across 4 task categories in both English and Chinese, covering length constraints ranging from 16 to 8192 words. We evaluate 26 widely-used LLMs and find that most models reasonably follow short-length instructions but deteriorate sharply beyond a certain threshold. Surprisingly, almost all models fail to reach the vendor-claimed maximum output lengths in practice, as further confirmed by our evaluations extending up to 32K words. Even long-context LLMs, despite their extended input-output windows, counterintuitively fail to improve length-instructions following. Notably, Reasoning LLMs outperform even specialized long-text generation models, achieving state-of-the-art length following. Overall, LIFEBench uncovers fundamental limitations in current LLMs' length instructions following ability, offering critical insights for future progress.
>
---
#### [replaced 046] Listen, Chat, and Remix: Text-Guided Soundscape Remixing for Enhanced Auditory Experience
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2402.03710v2](http://arxiv.org/pdf/2402.03710v2)**

> **作者:** Xilin Jiang; Cong Han; Yinghao Aaron Li; Nima Mesgarani
>
> **备注:** Accepted by IEEE Journal of Selected Topics in Signal Processing (JSTSP)
>
> **摘要:** In daily life, we encounter a variety of sounds, both desirable and undesirable, with limited control over their presence and volume. Our work introduces "Listen, Chat, and Remix" (LCR), a novel multimodal sound remixer that controls each sound source in a mixture based on user-provided text instructions. LCR distinguishes itself with a user-friendly text interface and its unique ability to remix multiple sound sources simultaneously within a mixture, without needing to separate them. Users input open-vocabulary text prompts, which are interpreted by a large language model to create a semantic filter for remixing the sound mixture. The system then decomposes the mixture into its components, applies the semantic filter, and reassembles filtered components back to the desired output. We developed a 160-hour dataset with over 100k mixtures, including speech and various audio sources, along with text prompts for diverse remixing tasks including extraction, removal, and volume control of single or multiple sources. Our experiments demonstrate significant improvements in signal quality across all remixing tasks and robust performance in zero-shot scenarios with varying numbers and types of sound sources. An audio demo is available at: https://listenchatremix.github.io/demo.
>
---
#### [replaced 047] GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.02404v2](http://arxiv.org/pdf/2506.02404v2)**

> **作者:** Yilin Xiao; Junnan Dong; Chuang Zhou; Su Dong; Qian-wen Zhang; Di Yin; Xing Sun; Xiao Huang
>
> **摘要:** Graph Retrieval Augmented Generation (GraphRAG) has garnered increasing recognition for its potential to enhance large language models (LLMs) by structurally organizing domain-specific corpora and facilitating complex reasoning. However, current evaluations of GraphRAG models predominantly rely on traditional question-answering datasets. Their limited scope in questions and evaluation metrics fails to comprehensively assess the reasoning capacity improvements enabled by GraphRAG models. To address this gap, we introduce GraphRAG-Bench, a large-scale, domain-specific benchmark designed to rigorously evaluate GraphRAG models. Our benchmark offers three key superiorities: \((i)\) Challenging question design. Featuring college-level, domain-specific questions that demand multi-hop reasoning, the benchmark ensures that simple content retrieval is insufficient for problem-solving. For example, some questions require mathematical reasoning or programming. \((ii)\) Diverse task coverage. The dataset includes a broad spectrum of reasoning tasks, multiple-choice, true/false, multi-select, open-ended, and fill-in-the-blank. It spans 16 disciplines in twenty core textbooks. \((iii)\) Holistic evaluation framework. GraphRAG-Bench provides comprehensive assessment across the entire GraphRAG pipeline, including graph construction, knowledge retrieval, and answer generation. Beyond final-answer correctness, it evaluates the logical coherence of the reasoning process. By applying nine contemporary GraphRAG methods to GraphRAG-Bench, we demonstrate its utility in quantifying how graph-based structuring improves model reasoning capabilities. Our analysis reveals critical insights about graph architectures, retrieval efficacy, and reasoning capabilities, offering actionable guidance for the research community.
>
---
#### [replaced 048] Automatic Pseudo-Harmful Prompt Generation for Evaluating False Refusals in Large Language Models
- **分类: cs.CL; cs.CR; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.00598v2](http://arxiv.org/pdf/2409.00598v2)**

> **作者:** Bang An; Sicheng Zhu; Ruiyi Zhang; Michael-Andrei Panaitescu-Liess; Yuancheng Xu; Furong Huang
>
> **摘要:** Safety-aligned large language models (LLMs) sometimes falsely refuse pseudo-harmful prompts, like "how to kill a mosquito," which are actually harmless. Frequent false refusals not only frustrate users but also provoke a public backlash against the very values alignment seeks to protect. In this paper, we propose the first method to auto-generate diverse, content-controlled, and model-dependent pseudo-harmful prompts. Using this method, we construct an evaluation dataset called PHTest, which is ten times larger than existing datasets, covers more false refusal patterns, and separately labels controversial prompts. We evaluate 20 LLMs on PHTest, uncovering new insights due to its scale and labeling. Our findings reveal a trade-off between minimizing false refusals and improving safety against jailbreak attacks. Moreover, we show that many jailbreak defenses significantly increase the false refusal rates, thereby undermining usability. Our method and dataset can help developers evaluate and fine-tune safer and more usable LLMs. Our code and dataset are available at https://github.com/umd-huang-lab/FalseRefusal
>
---
#### [replaced 049] Value Portrait: Assessing Language Models' Values through Psychometrically and Ecologically Valid Items
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.01015v3](http://arxiv.org/pdf/2505.01015v3)**

> **作者:** Jongwook Han; Dongmin Choi; Woojung Song; Eun-Ju Lee; Yohan Jo
>
> **备注:** This paper has been accepted for publication at ACL 2025
>
> **摘要:** The importance of benchmarks for assessing the values of language models has been pronounced due to the growing need of more authentic, human-aligned responses. However, existing benchmarks rely on human or machine annotations that are vulnerable to value-related biases. Furthermore, the tested scenarios often diverge from real-world contexts in which models are commonly used to generate text and express values. To address these issues, we propose the Value Portrait benchmark, a reliable framework for evaluating LLMs' value orientations with two key characteristics. First, the benchmark consists of items that capture real-life user-LLM interactions, enhancing the relevance of assessment results to real-world LLM usage. Second, each item is rated by human subjects based on its similarity to their own thoughts, and correlations between these ratings and the subjects' actual value scores are derived. This psychometrically validated approach ensures that items strongly correlated with specific values serve as reliable items for assessing those values. Through evaluating 44 LLMs with our benchmark, we find that these models prioritize Benevolence, Security, and Self-Direction values while placing less emphasis on Tradition, Power, and Achievement values. Also, our analysis reveals biases in how LLMs perceive various demographic groups, deviating from real human data.
>
---
#### [replaced 050] Societal AI Research Has Become Less Interdisciplinary
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.08738v2](http://arxiv.org/pdf/2506.08738v2)**

> **作者:** Dror Kris Markus; Fabrizio Gilardi; Daria Stetsenko
>
> **摘要:** As artificial intelligence (AI) systems become deeply embedded in everyday life, calls to align AI development with ethical and societal values have intensified. Interdisciplinary collaboration is often championed as a key pathway for fostering such engagement. Yet it remains unclear whether interdisciplinary research teams are actually leading this shift in practice. This study analyzes over 100,000 AI-related papers published on ArXiv between 2014 and 2024 to examine how ethical values and societal concerns are integrated into technical AI research. We develop a classifier to identify societal content and measure the extent to which research papers express these considerations. We find a striking shift: while interdisciplinary teams remain more likely to produce societally-oriented research, computer science-only teams now account for a growing share of the field's overall societal output. These teams are increasingly integrating societal concerns into their papers and tackling a wide range of domains - from fairness and safety to healthcare and misinformation. These findings challenge common assumptions about the drivers of societal AI and raise important questions. First, what are the implications for emerging understandings of AI safety and governance if most societally-oriented research is being undertaken by exclusively technical teams? Second, for scholars in the social sciences and humanities: in a technical field increasingly responsive to societal demands, what distinctive perspectives can we still offer to help shape the future of AI?
>
---
#### [replaced 051] EMMA: Efficient Visual Alignment in Multi-Modal LLMs
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.02080v2](http://arxiv.org/pdf/2410.02080v2)**

> **作者:** Sara Ghazanfari; Alexandre Araujo; Prashanth Krishnamurthy; Siddharth Garg; Farshad Khorrami
>
> **摘要:** Multi-modal Large Language Models (MLLMs) have recently exhibited impressive general-purpose capabilities by leveraging vision foundation models to encode the core concepts of images into representations. These are then combined with instructions and processed by the language model to generate high-quality responses. Despite significant progress in enhancing the language component, challenges persist in optimally fusing visual encodings within the language model for task-specific adaptability. Recent research has focused on improving this fusion through modality adaptation modules but at the cost of significantly increased model complexity and training data needs. In this paper, we propose EMMA (Efficient Multi-Modal Adaptation), a lightweight cross-modality module designed to efficiently fuse visual and textual encodings, generating instruction-aware visual representations for the language model. Our key contributions include: (1) an efficient early fusion mechanism that integrates vision and language representations with minimal added parameters (less than 0.2% increase in model size), (2) an in-depth interpretability analysis that sheds light on the internal mechanisms of the proposed method; (3) comprehensive experiments that demonstrate notable improvements on both specialized and general benchmarks for MLLMs. Empirical results show that EMMA boosts performance across multiple tasks by up to 9.3% while significantly improving robustness against hallucinations. Our code is available at https://github.com/SaraGhazanfari/EMMA
>
---
#### [replaced 052] OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23885v2](http://arxiv.org/pdf/2505.23885v2)**

> **作者:** Mengkang Hu; Yuhang Zhou; Wendong Fan; Yuzhou Nie; Bowei Xia; Tao Sun; Ziyu Ye; Zhaoxuan Jin; Yingru Li; Qiguang Chen; Zeyu Zhang; Yifeng Wang; Qianshuo Ye; Bernard Ghanem; Ping Luo; Guohao Li
>
> **备注:** Project Page: https://github.com/camel-ai/owl
>
> **摘要:** Large Language Model (LLM)-based multi-agent systems show promise for automating real-world tasks but struggle to transfer across domains due to their domain-specific nature. Current approaches face two critical shortcomings: they require complete architectural redesign and full retraining of all components when applied to new domains. We introduce Workforce, a hierarchical multi-agent framework that decouples strategic planning from specialized execution through a modular architecture comprising: (i) a domain-agnostic Planner for task decomposition, (ii) a Coordinator for subtask management, and (iii) specialized Workers with domain-specific tool-calling capabilities. This decoupling enables cross-domain transferability during both inference and training phases: During inference, Workforce seamlessly adapts to new domains by adding or modifying worker agents; For training, we introduce Optimized Workforce Learning (OWL), which improves generalization across domains by optimizing a domain-agnostic planner with reinforcement learning from real-world feedback. To validate our approach, we evaluate Workforce on the GAIA benchmark, covering various realistic, multi-domain agentic tasks. Experimental results demonstrate Workforce achieves open-source state-of-the-art performance (69.70%), outperforming commercial systems like OpenAI's Deep Research by 2.34%. More notably, our OWL-trained 32B model achieves 52.73% accuracy (+16.37%) and demonstrates performance comparable to GPT-4o on challenging tasks. To summarize, by enabling scalable generalization and modular domain transfer, our work establishes a foundation for the next generation of general-purpose AI assistants.
>
---
#### [replaced 053] Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14479v3](http://arxiv.org/pdf/2505.14479v3)**

> **作者:** Oren Sultan; Eitan Stern; Dafna Shahaf
>
> **备注:** long paper
>
> **摘要:** Large language models (LLMs) struggle with formal domains that require rigorous logical deduction and symbolic reasoning, such as mathematical proof generation. We propose a neuro-symbolic approach that combines LLMs' generative strengths with structured components to overcome this challenge. As a proof-of-concept, we focus on geometry problems. Our approach is two-fold: (1) we retrieve analogous problems and use their proofs to guide the LLM, and (2) a formal verifier evaluates the generated proofs and provides feedback, helping the model fix incorrect proofs. We demonstrate that our method significantly improves proof accuracy for OpenAI's o1 model (58%-70% improvement); both analogous problems and the verifier's feedback contribute to these gains. More broadly, shifting to LLMs that generate provably correct conclusions could dramatically improve their reliability, accuracy and consistency, unlocking complex tasks and critical real-world applications that require trustworthiness.
>
---
#### [replaced 054] MOSAIC: Multiple Observers Spotting AI Content
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.07615v3](http://arxiv.org/pdf/2409.07615v3)**

> **作者:** Matthieu Dubois; François Yvon; Pablo Piantanida
>
> **备注:** ACL 2025 Findings, code can be found at https://github.com/BaggerOfWords/MOSAIC
>
> **摘要:** The dissemination of Large Language Models (LLMs), trained at scale, and endowed with powerful text-generating abilities, has made it easier for all to produce harmful, toxic, faked or forged content. In response, various proposals have been made to automatically discriminate artificially generated from human-written texts, typically framing the problem as a binary classification problem. Early approaches evaluate an input document with a well-chosen detector LLM, assuming that low-perplexity scores reliably signal machine-made content. More recent systems instead consider two LLMs and compare their probability distributions over the document to further discriminate when perplexity alone cannot. However, using a fixed pair of models can induce brittleness in performance. We extend these approaches to the ensembling of several LLMs and derive a new, theoretically grounded approach to combine their respective strengths. Our experiments, conducted with various generator LLMs, indicate that this approach effectively leverages the strengths of each model, resulting in robust detection performance across multiple domains. Our code and data are available at https://github.com/BaggerOfWords/MOSAIC .
>
---
#### [replaced 055] DecIF: Improving Instruction-Following through Meta-Decomposition
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13990v2](http://arxiv.org/pdf/2505.13990v2)**

> **作者:** Tingfeng Hui; Pengyu Zhu; Bowen Ping; Ling Tang; Guanting Dong; Yaqi Zhang; Sen Su
>
> **备注:** We release the source code and SFT data in this version
>
> **摘要:** Instruction-following has emerged as a crucial capability for large language models (LLMs). However, existing approaches often rely on pre-existing documents or external resources to synthesize instruction-following data, which limits their flexibility and generalizability. In this paper, we introduce DecIF, a fully autonomous, meta-decomposition guided framework that generates diverse and high-quality instruction-following data using only LLMs. DecIF is grounded in the principle of decomposition. For instruction generation, we guide LLMs to iteratively produce various types of meta-information, which are then combined with response constraints to form well-structured and semantically rich instructions. We further utilize LLMs to detect and resolve potential inconsistencies within the generated instructions. Regarding response generation, we decompose each instruction into atomic-level evaluation criteria, enabling rigorous validation and the elimination of inaccurate instruction-response pairs. Extensive experiments across a wide range of scenarios and settings demonstrate DecIF's superior performance on instruction-following tasks. Further analysis highlights its strong flexibility, scalability, and generalizability in automatically synthesizing high-quality instruction data.
>
---
#### [replaced 056] GenARM: Reward Guided Generation with Autoregressive Reward Model for Test-time Alignment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.08193v4](http://arxiv.org/pdf/2410.08193v4)**

> **作者:** Yuancheng Xu; Udari Madhushani Sehwag; Alec Koppel; Sicheng Zhu; Bang An; Furong Huang; Sumitra Ganesh
>
> **备注:** Published at the Thirteenth International Conference on Learning Representations (ICLR 2025)
>
> **摘要:** Large Language Models (LLMs) exhibit impressive capabilities but require careful alignment with human preferences. Traditional training-time methods finetune LLMs using human preference datasets but incur significant training costs and require repeated training to handle diverse user preferences. Test-time alignment methods address this by using reward models (RMs) to guide frozen LLMs without retraining. However, existing test-time approaches rely on trajectory-level RMs which are designed to evaluate complete responses, making them unsuitable for autoregressive text generation that requires computing next-token rewards from partial responses. To address this, we introduce GenARM, a test-time alignment approach that leverages the Autoregressive Reward Model--a novel reward parametrization designed to predict next-token rewards for efficient and effective autoregressive generation. Theoretically, we demonstrate that this parametrization can provably guide frozen LLMs toward any distribution achievable by traditional RMs within the KL-regularized reinforcement learning framework. Experimental results show that GenARM significantly outperforms prior test-time alignment baselines and matches the performance of training-time methods. Additionally, GenARM enables efficient weak-to-strong guidance, aligning larger LLMs with smaller RMs without the high costs of training larger models. Furthermore, GenARM supports multi-objective alignment, allowing real-time trade-offs between preference dimensions and catering to diverse user preferences without retraining. Our project page is available at: https://genarm.github.io.
>
---
#### [replaced 057] Critic-CoT: Boosting the reasoning abilities of large language model via Chain-of-thoughts Critic
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.16326v3](http://arxiv.org/pdf/2408.16326v3)**

> **作者:** Xin Zheng; Jie Lou; Boxi Cao; Xueru Wen; Yuqiu Ji; Hongyu Lin; Yaojie Lu; Xianpei Han; Debing Zhang; Le Sun
>
> **备注:** Accepted at ACL 2025 Findings
>
> **摘要:** Self-critic has become a crucial mechanism for enhancing the reasoning performance of LLMs. However, current approaches mainly involve basic prompts for intuitive instance-level feedback, which resembles System-1 processes and limits the reasoning capabilities. Moreover, there is a lack of in-depth investigations into the relationship between LLM's ability to criticize and its task-solving performance. To address these issues, we propose Critic-CoT, a novel framework that pushes LLMs toward System-2-like critic capability. Through a step-wise CoT reasoning paradigm and the automatic construction of distant-supervision data without human annotation, Critic-CoT enables LLMs to engage in slow, analytic self-critique and refinement, thereby improving their reasoning abilities. Experiments on GSM8K and MATH demonstrate that our enhanced model significantly boosts task-solving performance by filtering out invalid solutions or iterative refinement. Furthermore, we investigate the intrinsic correlation between critique and task-solving abilities within LLMs, discovering that these abilities can mutually reinforce each other rather than conflict.
>
---
#### [replaced 058] AAD-LLM: Neural Attention-Driven Auditory Scene Understanding
- **分类: cs.SD; cs.AI; cs.CL; cs.HC; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.16794v3](http://arxiv.org/pdf/2502.16794v3)**

> **作者:** Xilin Jiang; Sukru Samet Dindar; Vishal Choudhari; Stephan Bickel; Ashesh Mehta; Guy M McKhann; Daniel Friedman; Adeen Flinker; Nima Mesgarani
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** Auditory foundation models, including auditory large language models (LLMs), process all sound inputs equally, independent of listener perception. However, human auditory perception is inherently selective: listeners focus on specific speakers while ignoring others in complex auditory scenes. Existing models do not incorporate this selectivity, limiting their ability to generate perception-aligned responses. To address this, we introduce Intention-Informed Auditory Scene Understanding (II-ASU) and present Auditory Attention-Driven LLM (AAD-LLM), a prototype system that integrates brain signals to infer listener attention. AAD-LLM extends an auditory LLM by incorporating intracranial electroencephalography (iEEG) recordings to decode which speaker a listener is attending to and refine responses accordingly. The model first predicts the attended speaker from neural activity, then conditions response generation on this inferred attentional state. We evaluate AAD-LLM on speaker description, speech transcription and extraction, and question answering in multitalker scenarios, with both objective and subjective ratings showing improved alignment with listener intention. By taking a first step toward intention-aware auditory AI, this work explores a new paradigm where listener perception informs machine listening, paving the way for future listener-centered auditory systems. Demo and code available: https://aad-llm.github.io.
>
---
#### [replaced 059] Can LLMs Generate Reliable Test Case Generators? A Study on Competition-Level Programming Problems
- **分类: cs.CL; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2506.06821v2](http://arxiv.org/pdf/2506.06821v2)**

> **作者:** Yuhan Cao; Zian Chen; Kun Quan; Ziliang Zhang; Yu Wang; Xiaoning Dong; Yeqi Feng; Guanzhong He; Jingcheng Huang; Jianhao Li; Yixuan Tan; Jiafu Tang; Yilin Tang; Junlei Wu; Qianyu Xiao; Can Zheng; Shouchen Zhou; Yuxiang Zhu; Yiming Huang; Tian Xie; Tianxing He
>
> **备注:** 37 pages, 22 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, capable of tackling complex tasks during inference. However, the extent to which LLMs can be utilized for code checking or debugging through test case generation remains largely unexplored. We investigate this problem from the perspective of competition-level programming (CP) programs and propose TCGBench, a Benchmark for (LLM generation of) Test Case Generators. This benchmark comprises two tasks, aimed at studying the capabilities of LLMs in (1) generating valid test case generators for a given CP problem, and further (2) generating targeted test case generators that expose bugs in human-written code. Experimental results indicate that while state-of-the-art LLMs can generate valid test case generators in most cases, most LLMs struggle to generate targeted test cases that reveal flaws in human code effectively. Especially, even advanced reasoning models (e.g., o3-mini) fall significantly short of human performance in the task of generating targeted generators. Furthermore, we construct a high-quality, manually curated dataset of instructions for generating targeted generators. Analysis demonstrates that the performance of LLMs can be enhanced with the aid of this dataset, by both prompting and fine-tuning.
>
---
#### [replaced 060] Beyond Bradley-Terry Models: A General Preference Model for Language Model Alignment
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.02197v3](http://arxiv.org/pdf/2410.02197v3)**

> **作者:** Yifan Zhang; Ge Zhang; Yue Wu; Kangping Xu; Quanquan Gu
>
> **备注:** Accepted to the 42nd International Conference on Machine Learning (ICML 2025)
>
> **摘要:** Modeling human preferences is crucial for aligning foundation models with human values. Traditional reward modeling methods, such as the Bradley-Terry (BT) reward model, fall short in expressiveness, particularly in addressing intransitive preferences. In this paper, we introduce preference embedding, an approach that embeds responses into a latent space to capture intricate preference structures efficiently, achieving linear query complexity. Additionally, we propose preference score-based General Preference Optimization (GPO), which generalizes reward-based reinforcement learning from human feedback (RLHF). Experimental results show that our General Preference embedding Model (GPM) consistently outperforms the BT reward model on the RewardBench benchmark and effectively models cyclic preferences where any BT reward model behaves like a random guess. Furthermore, evaluations on downstream tasks such as AlpacaEval2.0, following the language model post-training with GPO and our general preference model, reveal performance improvements over BT models. These findings indicate that our method may enhance the alignment of foundation models with nuanced human values. The code is available at https://github.com/general-preference/general-preference-model.
>
---
#### [replaced 061] SAFEFLOW: A Principled Protocol for Trustworthy and Transactional Autonomous Agent Systems
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07564v3](http://arxiv.org/pdf/2506.07564v3)**

> **作者:** Peiran Li; Xinkai Zou; Zhuohang Wu; Ruifeng Li; Shuo Xing; Hanwen Zheng; Zhikai Hu; Yuping Wang; Haoxi Li; Qin Yuan; Yingmo Zhang; Zhengzhong Tu
>
> **备注:** Former versions either contain unrelated content or cannot be properly converted to PDF
>
> **摘要:** Recent advances in large language models (LLMs) and vision-language models (VLMs) have enabled powerful autonomous agents capable of complex reasoning and multi-modal tool use. Despite their growing capabilities, today's agent frameworks remain fragile, lacking principled mechanisms for secure information flow, reliability, and multi-agent coordination. In this work, we introduce SAFEFLOW, a new protocol-level framework for building trustworthy LLM/VLM-based agents. SAFEFLOW enforces fine-grained information flow control (IFC), precisely tracking provenance, integrity, and confidentiality of all the data exchanged between agents, tools, users, and environments. By constraining LLM reasoning to respect these security labels, SAFEFLOW prevents untrusted or adversarial inputs from contaminating high-integrity decisions. To ensure robustness in concurrent multi-agent settings, SAFEFLOW introduces transactional execution, conflict resolution, and secure scheduling over shared state, preserving global consistency across agents. We further introduce mechanisms, including write-ahead logging, rollback, and secure caches, that further enhance resilience against runtime errors and policy violations. To validate the performances, we built SAFEFLOWBENCH, a comprehensive benchmark suite designed to evaluate agent reliability under adversarial, noisy, and concurrent operational conditions. Extensive experiments demonstrate that agents built with SAFEFLOW maintain impressive task performance and security guarantees even in hostile environments, substantially outperforming state-of-the-art. Together, SAFEFLOW and SAFEFLOWBENCH lay the groundwork for principled, robust, and secure agent ecosystems, advancing the frontier of reliable autonomy.
>
---
#### [replaced 062] CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.12768v2](http://arxiv.org/pdf/2411.12768v2)**

> **作者:** Nay Myat Min; Long H. Pham; Yige Li; Jun Sun
>
> **备注:** Accepted at ICML 2025, 20 pages
>
> **摘要:** Large Language Models (LLMs) are vulnerable to backdoor attacks that manipulate outputs via hidden triggers. Existing defense methods--designed for vision/text classification tasks--fail for text generation. We propose Internal Consistency Regularization (CROW), a defense leveraging the observation that backdoored models exhibit unstable layer-wise hidden representations when triggered, while clean models show smooth transitions. CROW enforces consistency across layers via adversarial perturbations and regularization during finetuning, neutralizing backdoors without requiring clean reference models or trigger knowledge--only a small clean dataset. Experiments across Llama-2 (7B, 13B), CodeLlama (7B, 13B), and Mistral-7B demonstrate CROW's effectiveness: it achieves significant reductions in attack success rates across diverse backdoor strategies (sentiment steering, targeted refusal, code injection) while preserving generative performance. CROW's architecture-agnostic design enables practical deployment.
>
---
#### [replaced 063] Guidelines for Fine-grained Sentence-level Arabic Readability Annotation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.08674v3](http://arxiv.org/pdf/2410.08674v3)**

> **作者:** Nizar Habash; Hanada Taha-Thomure; Khalid N. Elmadani; Zeina Zeino; Abdallah Abushmaes
>
> **备注:** Accepted at LAW-XIX at ACL 2025
>
> **摘要:** This paper presents the annotation guidelines of the Balanced Arabic Readability Evaluation Corpus (BAREC), a large-scale resource for fine-grained sentence-level readability assessment in Arabic. BAREC includes 69,441 sentences (1M+ words) labeled across 19 levels, from kindergarten to postgraduate. Based on the Taha/Arabi21 framework, the guidelines were refined through iterative training with native Arabic-speaking educators. We highlight key linguistic, pedagogical, and cognitive factors in determining readability and report high inter-annotator agreement: Quadratic Weighted Kappa 81.8% (substantial/excellent agreement) in the last annotation phase. We also benchmark automatic readability models across multiple classification granularities (19-, 7-, 5-, and 3-level). The corpus and guidelines are publicly available.
>
---
#### [replaced 064] Human-like object concept representations emerge naturally in multimodal large language models
- **分类: cs.AI; cs.CL; cs.CV; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.01067v3](http://arxiv.org/pdf/2407.01067v3)**

> **作者:** Changde Du; Kaicheng Fu; Bincheng Wen; Yi Sun; Jie Peng; Wei Wei; Ying Gao; Shengpei Wang; Chuncheng Zhang; Jinpeng Li; Shuang Qiu; Le Chang; Huiguang He
>
> **备注:** Published on Nature Machine Intelligence
>
> **摘要:** Understanding how humans conceptualize and categorize natural objects offers critical insights into perception and cognition. With the advent of Large Language Models (LLMs), a key question arises: can these models develop human-like object representations from linguistic and multimodal data? In this study, we combined behavioral and neuroimaging analyses to explore the relationship between object concept representations in LLMs and human cognition. We collected 4.7 million triplet judgments from LLMs and Multimodal LLMs (MLLMs) to derive low-dimensional embeddings that capture the similarity structure of 1,854 natural objects. The resulting 66-dimensional embeddings were stable, predictive, and exhibited semantic clustering similar to human mental representations. Remarkably, the dimensions underlying these embeddings were interpretable, suggesting that LLMs and MLLMs develop human-like conceptual representations of objects. Further analysis showed strong alignment between model embeddings and neural activity patterns in brain regions such as EBA, PPA, RSC, and FFA. This provides compelling evidence that the object representations in LLMs, while not identical to human ones, share fundamental similarities that reflect key aspects of human conceptual knowledge. Our findings advance the understanding of machine intelligence and inform the development of more human-like artificial cognitive systems.
>
---
#### [replaced 065] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.04745v3](http://arxiv.org/pdf/2504.04745v3)**

> **作者:** Ankush Raut; Xiaofeng Zhu; Maria Leonor Pacheco
>
> **备注:** 13 pages, 23 figures. Accepted at XLLM @ ACL 2025
>
> **摘要:** This paper evaluates the ability of Large Language Models (LLMs) to leverage contextual information in the form of structured linguistic representations. Specifically, we examine the impact of encoding both short and long contexts using Abstract Meaning Representation (AMR) structures across a diverse set of language tasks. We perform our analysis using 8-bit quantized and instruction-tuned versions of Llama 3.1 (8B), Phi-3, and Mistral 7B. Our results indicate that, for tasks involving short contexts, augmenting the prompt with the AMR of the original language context often degrades the performance of the underlying LLM. However, for tasks that involve long contexts, such as dialogue summarization in the SAMSum dataset, this enhancement improves LLM performance, for example, by increasing the zero-shot cosine similarity score of Llama 3.1 from 66% to 76%. This improvement is more evident in the newer and larger LLMs, but does not extend to the older or smaller ones. In addition, we observe that LLMs can effectively reconstruct the original text from a linearized AMR, achieving a cosine similarity of 81% in the best-case scenario.
>
---
#### [replaced 066] Decoding Knowledge Attribution in Mixture-of-Experts: A Framework of Basic-Refinement Collaboration and Efficiency Analysis
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.24593v2](http://arxiv.org/pdf/2505.24593v2)**

> **作者:** Junzhuo Li; Bo Wang; Xiuze Zhou; Peijie Jiang; Jia Liu; Xuming Hu
>
> **备注:** ACL 2025
>
> **摘要:** The interpretability of Mixture-of-Experts (MoE) models, especially those with heterogeneous designs, remains underexplored. Existing attribution methods for dense models fail to capture dynamic routing-expert interactions in sparse MoE architectures. To address this issue, we propose a cross-level attribution algorithm to analyze sparse MoE architectures (Qwen 1.5-MoE, OLMoE, Mixtral-8x7B) against dense models (Qwen 1.5-7B, Llama-7B, Mistral-7B). Results show MoE models achieve 37% higher per-layer efficiency via a "mid-activation, late-amplification" pattern: early layers screen experts, while late layers refine knowledge collaboratively. Ablation studies reveal a "basic-refinement" framework--shared experts handle general tasks (entity recognition), while routed experts specialize in domain-specific processing (geographic attributes). Semantic-driven routing is evidenced by strong correlations between attention heads and experts (r=0.68), enabling task-aware coordination. Notably, architectural depth dictates robustness: deep Qwen 1.5-MoE mitigates expert failures (e.g., 43% MRR drop in geographic tasks when blocking top-10 experts) through shared expert redundancy, whereas shallow OLMoE suffers severe degradation (76% drop). Task sensitivity further guides design: core-sensitive tasks (geography) require concentrated expertise, while distributed-tolerant tasks (object attributes) leverage broader participation. These insights advance MoE interpretability, offering principles to balance efficiency, specialization, and robustness.
>
---
#### [replaced 067] LLM-BT-Terms: Back-Translation as a Framework for Terminology Standardization and Dynamic Semantic Embedding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08174v2](http://arxiv.org/pdf/2506.08174v2)**

> **作者:** Li Weigang; Pedro Carvalho Brom
>
> **备注:** 23 pages
>
> **摘要:** The rapid expansion of English technical terminology presents a significant challenge to traditional expert-based standardization, particularly in rapidly developing areas such as artificial intelligence and quantum computing. Manual approaches face difficulties in maintaining consistent multilingual terminology. To address this, we introduce LLM-BT, a back-translation framework powered by large language models (LLMs) designed to automate terminology verification and standardization through cross-lingual semantic alignment. Our key contributions include: (1) term-level consistency validation: by performing English -> intermediate language -> English back-translation, LLM-BT achieves high term consistency across different models (such as GPT-4, DeepSeek, and Grok). Case studies demonstrate over 90 percent of terms are preserved either exactly or semantically; (2) multi-path verification workflow: we develop a novel pipeline described as Retrieve -> Generate -> Verify -> Optimize, which supports both serial paths (e.g., English -> Simplified Chinese -> Traditional Chinese -> English) and parallel paths (e.g., English -> Chinese / Portuguese -> English). BLEU scores and term-level accuracy indicate strong cross-lingual robustness, with BLEU scores exceeding 0.45 and Portuguese term accuracy reaching 100 percent; (3) back-translation as semantic embedding: we reinterpret back-translation as a form of dynamic semantic embedding that uncovers latent trajectories of meaning. In contrast to static embeddings, LLM-BT offers transparent, path-based embeddings shaped by the evolution of the models. This reframing positions back-translation as an active mechanism for multilingual terminology standardization, fostering collaboration between machines and humans - machines preserve semantic integrity, while humans provide cultural interpretation.
>
---
#### [replaced 068] StochasTok: Improving Fine-Grained Subword Understanding in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01687v2](http://arxiv.org/pdf/2506.01687v2)**

> **作者:** Anya Sims; Thom Foster; Klara Kaleb; Tuan-Duy H. Nguyen; Joseph Lee; Jakob N. Foerster; Yee Whye Teh; Cong Lu
>
> **摘要:** Subword-level understanding is integral to numerous tasks, including understanding multi-digit numbers, spelling mistakes, abbreviations, rhyming, and wordplay. Despite this, current large language models (LLMs) still often struggle with seemingly simple subword-level tasks like How many 'r's in 'strawberry'?. A key factor behind these failures is tokenization which obscures the fine-grained structure of words. Current alternatives, such as character-level and dropout tokenization methods, significantly increase computational costs and provide inconsistent improvements. In this paper we revisit tokenization and introduce StochasTok, a simple, efficient stochastic tokenization scheme that randomly splits tokens during training, allowing LLMs to 'see' their internal structure. Our experiments show that pretraining with StochasTok substantially improves LLMs' downstream performance across multiple subword-level language games, including character counting, substring identification, and math tasks. Furthermore, StochasTok's simplicity allows seamless integration at any stage of the training pipeline; and we demonstrate that post-training with StochasTok can instill improved subword understanding into existing pretrained models, thus avoiding costly pretraining from scratch. These dramatic improvements achieved with a minimal change suggest StochasTok holds exciting potential when applied to larger, more capable models. Code open-sourced at: https://github.com/anyasims/stochastok.
>
---
#### [replaced 069] Reasoning Language Models: A Blueprint
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.11223v4](http://arxiv.org/pdf/2501.11223v4)**

> **作者:** Maciej Besta; Julia Barth; Eric Schreiber; Ales Kubicek; Afonso Catarino; Robert Gerstenberger; Piotr Nyczyk; Patrick Iff; Yueling Li; Sam Houliston; Tomasz Sternal; Marcin Copik; Grzegorz Kwaśniewski; Jürgen Müller; Łukasz Flis; Hannes Eberhard; Zixuan Chen; Hubert Niewiadomski; Torsten Hoefler
>
> **摘要:** Reasoning language models (RLMs), also known as Large Reasoning Models (LRMs), such as OpenAI's o1 and o3, DeepSeek-R1, and Alibaba's QwQ, have redefined AI's problem-solving capabilities by extending LLMs with advanced reasoning mechanisms. Yet, their high costs, proprietary nature, and complex architectures - uniquely combining reinforcement learning (RL), search heuristics, and LLMs - present accessibility and scalability challenges. To address these, we propose a comprehensive blueprint that organizes RLM components into a modular framework, based on a survey and analysis of all RLM works. This blueprint incorporates diverse reasoning structures (chains, trees, graphs, and nested forms), reasoning strategies (e.g., Monte Carlo Tree Search, Beam Search), RL concepts (policy, value models and others), supervision schemes (Outcome-Based and Process-Based Supervision), and other related concepts (e.g., Test-Time Compute, Retrieval-Augmented Generation, agent tools). We also provide detailed mathematical formulations and algorithmic specifications to simplify RLM implementation. By showing how schemes like LLaMA-Berry, QwQ, Journey Learning, and Graph of Thoughts fit as special cases, we demonstrate the blueprint's versatility and unifying potential. To illustrate its utility, we introduce x1, a modular implementation for rapid RLM prototyping and experimentation. Using x1 and a literature review, we provide key insights, such as multi-phase training for policy and value models, and the importance of familiar training distributions. Finally, we discuss scalable RLM cloud deployments and we outline how RLMs can integrate with a broader LLM ecosystem. Our work demystifies RLM construction, democratizes advanced reasoning capabilities, and fosters innovation, aiming to mitigate the gap between "rich AI" and "poor AI" by lowering barriers to RLM design and experimentation.
>
---
#### [replaced 070] Mitigating Posterior Salience Attenuation in Long-Context LLMs with Positional Contrastive Decoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08371v2](http://arxiv.org/pdf/2506.08371v2)**

> **作者:** Zikai Xiao; Ziyang Wang; Wen Ma; Yan Zhang; Wei Shen; Yan Wang; Luqi Gong; Zuozhu Liu
>
> **摘要:** While Large Language Models (LLMs) support long contexts, they struggle with performance degradation within the context window. Current solutions incur prohibitive training costs, leaving statistical behaviors and cost-effective approaches underexplored. From the decoding perspective, we identify the Posterior Salience Attenuation (PSA) phenomenon, where the salience ratio correlates with long-text performance degradation. Notably, despite the attenuation, gold tokens still occupy high-ranking positions in the decoding space. Motivated by it, we propose the training-free Positional Contrastive Decoding (PCD) that contrasts the logits derived from long-aware attention with those from designed local-aware attention, enabling the model to focus on the gains introduced by large-scale short-to-long training. Through the analysis of long-term decay simulation, we demonstrate that PCD effectively alleviates attention score degradation. Experimental results show that PCD achieves state-of-the-art performance on long-context benchmarks.
>
---
#### [replaced 071] Context Is Not Comprehension: Unmasking LLM reasoning blind spots with VLO
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.04907v3](http://arxiv.org/pdf/2506.04907v3)**

> **作者:** Alex Pan; Mary-Anne Williams
>
> **备注:** 24 pages, 2 figures, 4 tables; to appear in AAAI 2026
>
> **摘要:** The dominant evaluation of Large Language Models has centered on their ability to surface explicit facts from increasingly vast contexts. While today's best models demonstrate near-perfect recall on these tasks, this apparent success is overly simplistic and non-representative of the complexity of human reasoning which is often highly nested. We introduce Verbose ListOps (VLO), a novel benchmark designed to isolate this failure. VLO programmatically weaves deterministic, nested computations into coherent stories, forcing models to track and update internal state rather than simply locate explicit values. Our experiments show that leading LLMs, capable of solving the raw ListOps equations with near-perfect accuracy, collapse in performance on VLO at just 10k tokens. The extensibility of VLO's generation framework to any verifiable reasoning pattern will be a critical tool, enabling model developers to move beyond context windows and robustly test new reasoning architectures; a necessary step to automating the world's knowledge work.
>
---
#### [replaced 072] CaLMQA: Exploring culturally specific long-form question answering across 23 languages
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.17761v3](http://arxiv.org/pdf/2406.17761v3)**

> **作者:** Shane Arora; Marzena Karpinska; Hung-Ting Chen; Ipsita Bhattacharjee; Mohit Iyyer; Eunsol Choi
>
> **备注:** 46 pages, 26 figures. Accepted as a main conference paper at ACL 2025. Code and data available at https://github.com/2015aroras/CaLMQA . Dataset expanded to 51.7K questions
>
> **摘要:** Despite rising global usage of large language models (LLMs), their ability to generate long-form answers to culturally specific questions remains unexplored in many languages. To fill this gap, we perform the first study of textual multilingual long-form QA by creating CaLMQA, a dataset of 51.7K culturally specific questions across 23 different languages. We define culturally specific questions as those that refer to concepts unique to one or a few cultures, or have different answers depending on the cultural or regional context. We obtain these questions by crawling naturally-occurring questions from community web forums in high-resource languages, and by hiring native speakers to write questions in under-resourced, rarely-studied languages such as Fijian and Kirundi. Our data collection methodologies are translation-free, enabling the collection of culturally unique questions like "Kuber iki umwami wa mbere w'uburundi yitwa Ntare?" (Kirundi; English translation: "Why was the first king of Burundi called Ntare (Lion)?"). We evaluate factuality, relevance and surface-level quality of LLM-generated long-form answers, finding that (1) for many languages, even the best models make critical surface-level errors (e.g., answering in the wrong language, repetition), especially for low-resource languages; and (2) answers to culturally specific questions contain more factual errors than answers to culturally agnostic questions -- questions that have consistent meaning and answer across many cultures. We release CaLMQA to facilitate future research in cultural and multilingual long-form QA.
>
---
#### [replaced 073] Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06395v3](http://arxiv.org/pdf/2506.06395v3)**

> **作者:** Pengyi Li; Matvey Skripkin; Alexander Zubrey; Andrey Kuznetsov; Ivan Oseledets
>
> **摘要:** Large language models (LLMs) excel at reasoning, yet post-training remains critical for aligning their behavior with task goals. Existing reinforcement learning (RL) methods often depend on costly human annotations or external reward models. We propose Reinforcement Learning via Self-Confidence (RLSC), which uses the model's own confidence as reward signals-eliminating the need for labels, preference models, or reward engineering. Applied to Qwen2.5-Math-7B with only 16 samples per question and 10 or 20 training steps, RLSC improves accuracy by +13.4% on AIME2024, +21.2% on MATH500, +21.7% on Minerva Math, +20.8% on Olympiadbench, and +9.7% on AMC23. RLSC provides a simple, scalable post-training method for inference models, requiring only a small number of samples and unlabelled supervision.
>
---
#### [replaced 074] CASPER: A Large Scale Spontaneous Speech Dataset
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00267v3](http://arxiv.org/pdf/2506.00267v3)**

> **作者:** Cihan Xiao; Ruixing Liang; Xiangyu Zhang; Mehmet Emre Tiryaki; Veronica Bae; Lavanya Shankar; Rong Yang; Ethan Poon; Emmanuel Dupoux; Sanjeev Khudanpur; Leibny Paola Garcia Perera
>
> **摘要:** The success of large language models has driven interest in developing similar speech processing capabilities. However, a key challenge is the scarcity of high-quality spontaneous speech data, as most existing datasets contain scripted dialogues. To address this, we present a novel pipeline for eliciting and recording natural dialogues and release our dataset with 100+ hours of spontaneous speech. Our approach fosters fluid, natural conversations while encouraging a diverse range of topics and interactive exchanges. Unlike traditional methods, it facilitates genuine interactions, providing a reproducible framework for future data collection. This paper introduces our dataset and methodology, laying the groundwork for addressing the shortage of spontaneous speech data. We plan to expand this dataset in future stages, offering a growing resource for the research community.
>
---
#### [replaced 075] Persona-judge: Personalized Alignment of Large Language Models via Token-level Self-judgment
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.12663v2](http://arxiv.org/pdf/2504.12663v2)**

> **作者:** Xiaotian Zhang; Ruizhe Chen; Yang Feng; Zuozhu Liu
>
> **备注:** ACL Finding
>
> **摘要:** Aligning language models with human preferences presents significant challenges, particularly in achieving personalization without incurring excessive computational costs. Existing methods rely on reward signals and additional annotated data, limiting their scalability and adaptability to diverse human values. To address these challenges, we introduce Persona-judge, a novel discriminative paradigm that enables training-free personalized alignment with unseen preferences. Instead of optimizing policy parameters through external reward feedback, Persona-judge leverages the intrinsic preference judgment capabilities of the model. Specifically, a draft model generates candidate tokens conditioned on a given preference, while a judge model, embodying another preference, cross-validates the predicted tokens whether to be accepted. Experimental results demonstrate that Persona-judge, using the inherent preference evaluation mechanisms of the model, offers a scalable and computationally efficient solution to personalized alignment, paving the way for more adaptive customized alignment. Our code is available here.
>
---
#### [replaced 076] Synthesis by Design: Controlled Data Generation via Structural Guidance
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.07664v2](http://arxiv.org/pdf/2506.07664v2)**

> **作者:** Lei Xu; Sirui Chen; Yuxuan Huang; Chaochao Lu
>
> **摘要:** Mathematical reasoning remains challenging for LLMs due to complex logic and the need for precise computation. Existing methods enhance LLM reasoning by synthesizing datasets through problem rephrasing, but face issues with generation quality and problem complexity. To address this, we propose to extract structural information with generated problem-solving code from mathematical reasoning and guide data generation with structured solutions. Applied to MATH and GSM8K, our approach produces 39K problems with labeled intermediate steps and a 6.1K-problem benchmark of higher difficulty. Results on our benchmark show that model performance declines as reasoning length increases. Additionally, we conducted fine-tuning experiments using the proposed training data on a range of LLMs, and the results validate the effectiveness of our dataset. We hope the proposed method and dataset will contribute to future research in enhancing LLM reasoning capabilities. Our code and data are available at https://github.com/OpenCausaLab/StructuralGeneration.
>
---
#### [replaced 077] Enhancing Code LLMs with Reinforcement Learning in Code Generation: A Survey
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.20367v4](http://arxiv.org/pdf/2412.20367v4)**

> **作者:** Junqiao Wang; Zeng Zhang; Yangfan He; Zihao Zhang; Yuyang Song; Tianyu Shi; Yuchen Li; Hengyuan Xu; Kunyu Wu; Xin Yi; Zhongwei Wan; Xinhang Yuan; Kuan Lu; Menghao Huo; Tang Jingqun; Guangwu Qian; Keqin Li; Qiuwu Chen; Lewei He
>
> **摘要:** Reinforcement learning (RL) has emerged as a powerful paradigm for enhancing large language models (LLMs) in code generation and optimization. This survey systematically reviews RL-driven techniques across the code development lifecycle, from compiler-level optimizations and resource allocation strategies to end-to-end code synthesis frameworks. We first examine classical and modern RL algorithms -- spanning policy gradients, actor-critic methods, human-feedback alignment, and preference-based optimization -- and their adaptations to the unique challenges of code generation, such as sparse and delayed rewards. Next, we analyze key benchmarks, datasets, and evaluation metrics that drive progress in RL-augmented Code LLMs. Finally, we identify open problems, including the need for richer feedback sources, support for low-level and domain-specific languages, and methods to reduce computational overhead. By consolidating current insights and outlining future directions, this work aims to guide researchers and practitioners in leveraging RL to produce more robust, efficient, and human-aligned code generation systems.
>
---
#### [replaced 078] Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.11314v2](http://arxiv.org/pdf/2503.11314v2)**

> **作者:** Xinyu Tang; Xiaolei Wang; Zhihao Lv; Yingqian Min; Wayne Xin Zhao; Binbin Hu; Ziqi Liu; Zhiqiang Zhang
>
> **备注:** ACL 2025
>
> **摘要:** Recent advancements in long chain-of-thoughts(long CoTs) have significantly improved the reasoning capabilities of large language models(LLMs). Existing work finds that the capability of long CoT reasoning can be efficiently elicited by tuning on only a few examples and can easily transfer to other tasks. This motivates us to investigate whether long CoT reasoning is a general capability for LLMs. In this work, we conduct an empirical analysis for this question from the perspective of representation. We find that LLMs do encode long CoT reasoning as a general capability, with a clear distinction from vanilla CoTs. Furthermore, domain-specific representations are also required for the effective transfer of long CoT reasoning. Inspired by these findings, we propose GLoRE, a novel representation engineering method to unleash the general long CoT reasoning capabilities of LLMs. Extensive experiments demonstrate the effectiveness and efficiency of GLoRE in both in-domain and cross-domain scenarios.
>
---
#### [replaced 079] CC-RAG: Structured Multi-Hop Reasoning via Theme-Based Causal Graphs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08364v2](http://arxiv.org/pdf/2506.08364v2)**

> **作者:** Jash Rajesh Parekh; Pengcheng Jiang; Jiawei Han
>
> **摘要:** Understanding cause and effect relationships remains a formidable challenge for Large Language Models (LLMs), particularly in specialized domains where reasoning requires more than surface-level correlations. Retrieval-Augmented Generation (RAG) improves factual accuracy, but standard RAG pipelines treat evidence as flat context, lacking the structure required to model true causal dependencies. We introduce Causal-Chain RAG (CC-RAG), a novel approach that integrates zero-shot triple extraction and theme-aware graph chaining into the RAG pipeline, enabling structured multi-hop inference. Given a domain specific corpus, CC-RAG constructs a Directed Acyclic Graph (DAG) of <cause, relation, effect> triples and uses forward/backward chaining to guide structured answer generation. Experiments on two real-world domains: Bitcoin price fluctuations and Gaucher disease, show that CC-RAG outperforms standard RAG and zero-shot LLMs in chain similarity, information density, and lexical diversity. Both LLM-as-a-Judge and human evaluations consistently favor CC-RAG. Our results demonstrate that explicitly modeling causal structure enables LLMs to generate more accurate and interpretable responses, especially in specialized domains where flat retrieval fails.
>
---
#### [replaced 080] Retrofitting Large Language Models with Dynamic Tokenization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.18553v3](http://arxiv.org/pdf/2411.18553v3)**

> **作者:** Darius Feher; Ivan Vulić; Benjamin Minixhofer
>
> **摘要:** Current language models (LMs) use a fixed, static subword tokenizer. This default choice typically results in degraded efficiency and language capabilities, especially in languages other than English. To address this issue, we challenge the static design and propose retrofitting LMs with dynamic tokenization: a way to dynamically decide on token boundaries based on the input text via a subword-merging algorithm inspired by byte-pair encoding. We merge frequent subword sequences in a batch, then apply a pre-trained embedding-prediction hypernetwork to compute the token embeddings on-the-fly. For encoder-style models (e.g., XLM-R), this on average reduces token sequence lengths by >20% across 14 languages while degrading performance by less than 2%. The same method applied to pre-filling and scoring in decoder-style models (e.g., Mistral-7B) results in minimal performance degradation at up to 17% reduction in sequence length. Overall, we find that dynamic tokenization can mitigate the limitations of static tokenization by substantially improving inference speed and promoting fairness across languages, enabling more equitable and adaptable LMs.
>
---
#### [replaced 081] Comparing human and LLM proofreading in L2 writing: Impact on lexical and syntactic features
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09021v2](http://arxiv.org/pdf/2506.09021v2)**

> **作者:** Hakyung Sung; Karla Csuros; Min-Chang Sung
>
> **摘要:** This study examines the lexical and syntactic interventions of human and LLM proofreading aimed at improving overall intelligibility in identical second language writings, and evaluates the consistency of outcomes across three LLMs (ChatGPT-4o, Llama3.1-8b, Deepseek-r1-8b). Findings show that both human and LLM proofreading enhance bigram lexical features, which may contribute to better coherence and contextual connectedness between adjacent words. However, LLM proofreading exhibits a more generative approach, extensively reworking vocabulary and sentence structures, such as employing more diverse and sophisticated vocabulary and incorporating a greater number of adjective modifiers in noun phrases. The proofreading outcomes are highly consistent in major lexical and syntactic features across the three models.
>
---
#### [replaced 082] The Remarkable Robustness of LLMs: Stages of Inference?
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.19384v2](http://arxiv.org/pdf/2406.19384v2)**

> **作者:** Vedang Lad; Wes Gurnee; Max Tegmark
>
> **备注:** For Github code see https://github.com/vdlad/Remarkable-Robustness-of-LLMs. Send all correspondence to the first author
>
> **摘要:** We investigate the robustness of Large Language Models (LLMs) to structural interventions by deleting and swapping adjacent layers during inference. Surprisingly, models retain 72-95% of their original top-1 prediction accuracy without any fine-tuning. We find that performance degradation is not uniform across layers: interventions to the early and final layers cause the most degradation, while the model is remarkably robust to dropping middle layers. This pattern of localized sensitivity motivates our hypothesis of four stages of inference, observed across diverse model families and sizes: (1) detokenization, where local context is integrated to lift raw token embeddings into higher-level representations; (2) feature engineering, where task- and entity-specific features are iteratively refined; (3) prediction ensembling, where hidden states are aggregated into plausible next-token predictions; and (4) residual sharpening, where irrelevant features are suppressed to finalize the output distribution. Synthesizing behavioral and mechanistic evidence, we provide a framework for interpreting depth-dependent computations in LLMs.
>
---
#### [replaced 083] Meaningless is better: hashing bias-inducing words in LLM prompts improves performance in logical reasoning and statistical learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.17304v2](http://arxiv.org/pdf/2411.17304v2)**

> **作者:** Milena Chadimová; Eduard Jurášek; Tomáš Kliegr
>
> **摘要:** This paper introduces a novel method, referred to as "hashing", which involves masking potentially bias-inducing words in large language models (LLMs) with hash-like meaningless identifiers to reduce cognitive biases and reliance on external knowledge. The method was tested across three sets of experiments involving a total of 490 prompts. Statistical analysis using chi-square tests showed significant improvements in all tested scenarios, which covered LLama, ChatGPT, Copilot, Gemini and Mixtral models. In the first experiment, hashing decreased the fallacy rate in a modified version of the "Linda" problem aimed at evaluating susceptibility to cognitive biases. In the second experiment, it improved LLM results on the frequent itemset extraction task. In the third experiment, we found hashing is also effective when the Linda problem is presented in a tabular format rather than text, indicating that the technique works across various input representations. Overall, the method was shown to improve bias reduction and incorporation of external knowledge. Despite bias reduction, hallucination rates were inconsistently reduced across types of LLM models. These findings suggest that masking bias-inducing terms can improve LLM performance, although its effectiveness is model- and task-dependent.
>
---
#### [replaced 084] Explaining word embeddings with perfect fidelity: Case study in research impact prediction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.15912v2](http://arxiv.org/pdf/2409.15912v2)**

> **作者:** Lucie Dvorackova; Marcin P. Joachimiak; Michal Cerny; Adriana Kubecova; Vilem Sklenak; Tomas Kliegr
>
> **摘要:** Best performing approaches for scholarly document quality prediction are based on embedding models, which do not allow direct explanation of classifiers as distinct words no longer correspond to the input features for model training. Although model-agnostic explanation methods such as Local interpretable model-agnostic explanations (LIME) can be applied, these produce results with questionable correspondence to the ML model. We introduce a new feature importance method, Self-model Rated Entities (SMER), for logistic regression-based classification models trained on word embeddings. We show that SMER has theoretically perfect fidelity with the explained model, as its prediction corresponds exactly to the average of predictions for individual words in the text. SMER allows us to reliably determine which words or entities positively contribute to predicting impactful articles. Quantitative and qualitative evaluation is performed through five diverse experiments conducted on 50.000 research papers from the CORD-19 corpus. Through an AOPC curve analysis, we experimentally demonstrate that SMER produces better explanations than LIME for logistic regression.
>
---
#### [replaced 085] AMELI: Enhancing Multimodal Entity Linking with Fine-Grained Attributes
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2305.14725v2](http://arxiv.org/pdf/2305.14725v2)**

> **作者:** Barry Menglong Yao; Sijia Wang; Yu Chen; Qifan Wang; Minqian Liu; Zhiyang Xu; Licheng Yu; Lifu Huang
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** We propose attribute-aware multimodal entity linking, where the input consists of a mention described with a text paragraph and images, and the goal is to predict the corresponding target entity from a multimodal knowledge base (KB) where each entity is also accompanied by a text description, visual images, and a collection of attributes that present the meta-information of the entity in a structured format. To facilitate this research endeavor, we construct AMELI, encompassing a new multimodal entity linking benchmark dataset that contains 16,735 mentions described in text and associated with 30,472 images, and a multimodal knowledge base that covers 34,690 entities along with 177,873 entity images and 798,216 attributes. To establish baseline performance on AMELI, we experiment with several state-of-the-art architectures for multimodal entity linking and further propose a new approach that incorporates attributes of entities into disambiguation. Experimental results and extensive qualitative analysis demonstrate that extracting and understanding the attributes of mentions from their text descriptions and visual images play a vital role in multimodal entity linking. To the best of our knowledge, we are the first to integrate attributes in the multimodal entity linking task. The programs, model checkpoints, and the dataset are publicly available at https://github.com/VT-NLP/Ameli.
>
---
#### [replaced 086] LLM2TEA: Agentic AI Designer Finds Innovative Objects with Generative Evolutionary Multitasking
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.NE**

- **链接: [http://arxiv.org/pdf/2406.14917v2](http://arxiv.org/pdf/2406.14917v2)**

> **作者:** Melvin Wong; Jiao Liu; Thiago Rios; Stefan Menzel; Yew Soon Ong
>
> **备注:** This work has been submitted to the IEEE for review
>
> **摘要:** In this paper, we introduce LLM-driven MultiTask Evolutionary Algorithm (LLM2TEA), the first agentic AI designer within a generative evolutionary multitasking (GEM) framework that promotes the crossover and synergy of designs from multiple domains, leading to innovative solutions that transcend individual disciplines. Of particular interest is the discovery of objects that are not only innovative but also conform to the physical specifications of the real world in science and engineering. LLM2TEA comprises a large language model to initialize a population of genotypes (defined by text prompts) describing the objects of interest, a text-to-3D generative model to produce phenotypes from these prompts, a classifier to interpret the semantic representations of the objects, and a physics simulation model to assess their physical properties. We propose several novel LLM-based multitask evolutionary operators to guide the search toward the discovery of high-performing practical objects. Experimental results in conceptual design optimization validate the effectiveness of LLM2TEA, revealing from 97\% to 174\% improvement in the diversity of innovative objects compared to the present text-to-3D generative model baseline. In addition, more than 73\% of the generated designs have better physical performance than the top 1\% percentile of the designs generated in the baseline. Moreover, LLM2TEA generates designs that are not only aesthetically creative but also functional in real-world applications. Several of these designs have been successfully 3D-printed, emphasizing the proposed approach's capacity to transform AI-generated outputs into tangible physical objects. The designs produced by LLM2TEA meets practical requirements while showcasing creative and innovative features, underscoring its potential applications in complex design optimization and discovery.
>
---
#### [replaced 087] Do Large Vision-Language Models Distinguish between the Actual and Apparent Features of Illusions?
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05765v2](http://arxiv.org/pdf/2506.05765v2)**

> **作者:** Taiga Shinozaki; Tomoki Doi; Amane Watahiki; Satoshi Nishida; Hitomi Yanaka
>
> **备注:** To appear in the Proceedings of the 47th Annual Meeting of the Cognitive Science Society (COGSCI 2025)
>
> **摘要:** Humans are susceptible to optical illusions, which serve as valuable tools for investigating sensory and cognitive processes. Inspired by human vision studies, research has begun exploring whether machines, such as large vision language models (LVLMs), exhibit similar susceptibilities to visual illusions. However, studies often have used non-abstract images and have not distinguished actual and apparent features, leading to ambiguous assessments of machine cognition. To address these limitations, we introduce a visual question answering (VQA) dataset, categorized into genuine and fake illusions, along with corresponding control images. Genuine illusions present discrepancies between actual and apparent features, whereas fake illusions have the same actual and apparent features even though they look illusory due to the similar geometric configuration. We evaluate the performance of LVLMs for genuine and fake illusion VQA tasks and investigate whether the models discern actual and apparent features. Our findings indicate that although LVLMs may appear to recognize illusions by correctly answering questions about both feature types, they predict the same answers for both Genuine Illusion and Fake Illusion VQA questions. This suggests that their responses might be based on prior knowledge of illusions rather than genuine visual understanding. The dataset is available at https://github.com/ynklab/FILM
>
---
#### [replaced 088] Rethinking Text-based Protein Understanding: Retrieval or LLM?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20354v3](http://arxiv.org/pdf/2505.20354v3)**

> **作者:** Juntong Wu; Zijing Liu; He Cao; Hao Li; Bin Feng; Zishan Shu; Ke Yu; Li Yuan; Yu Li
>
> **摘要:** In recent years, protein-text models have gained significant attention for their potential in protein generation and understanding. Current approaches focus on integrating protein-related knowledge into large language models through continued pretraining and multi-modal alignment, enabling simultaneous comprehension of textual descriptions and protein sequences. Through a thorough analysis of existing model architectures and text-based protein understanding benchmarks, we identify significant data leakage issues present in current benchmarks. Moreover, conventional metrics derived from natural language processing fail to accurately assess the model's performance in this domain. To address these limitations, we reorganize existing datasets and introduce a novel evaluation framework based on biological entities. Motivated by our observation, we propose a retrieval-enhanced method, which significantly outperforms fine-tuned LLMs for protein-to-text generation and shows accuracy and efficiency in training-free scenarios. Our code and data can be seen at https://github.com/IDEA-XL/RAPM.
>
---
#### [replaced 089] Revisiting Self-Consistency from Dynamic Distributional Alignment Perspective on Answer Aggregation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.19830v2](http://arxiv.org/pdf/2502.19830v2)**

> **作者:** Yiwei Li; Ji Zhang; Shaoxiong Feng; Peiwen Yuan; Xinglin Wang; Jiayi Shi; Yueqi Zhang; Chuyi Tan; Boyuan Pan; Yao Hu; Kan Li
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Self-consistency improves reasoning by aggregating diverse stochastic samples, yet the dynamics behind its efficacy remain underexplored. We reframe self-consistency as a dynamic distributional alignment problem, revealing that decoding temperature not only governs sampling randomness but also actively shapes the latent answer distribution. Given that high temperatures require prohibitively large sample sizes to stabilize, while low temperatures risk amplifying biases, we propose a confidence-driven mechanism that dynamically calibrates temperature: sharpening the sampling distribution under uncertainty to align with high-probability modes, and promoting exploration when confidence is high. Experiments on mathematical reasoning tasks show this approach outperforms fixed-diversity baselines under limited samples, improving both average and best-case performance across varying initial temperatures without additional data or modules. This establishes self-consistency as a synchronization challenge between sampling dynamics and evolving answer distributions.
>
---
#### [replaced 090] Speech Synthesis By Unrolling Diffusion Process using Neural Network Layers
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2309.09652v5](http://arxiv.org/pdf/2309.09652v5)**

> **作者:** Peter Ochieng
>
> **备注:** 10 pages
>
> **摘要:** This work introduces UDPNet, a novel architecture designed to accelerate the reverse diffusion process in speech synthesis. Unlike traditional diffusion models that rely on timestep embeddings and shared network parameters, UDPNet unrolls the reverse diffusion process directly into the network architecture, with successive layers corresponding to equally spaced steps in the diffusion schedule. Each layer progressively refines the noisy input, culminating in a high-fidelity estimation of the original data, \(x_0\). Additionally, we redefine the learning target by predicting latent variables instead of the conventional \(x_0\) or noise \(\epsilon_0\). This shift addresses the common issue of large prediction errors in early denoising stages, effectively reducing speech distortion. Extensive evaluations on single- and multi-speaker datasets demonstrate that UDPNet consistently outperforms state-of-the-art methods in both quality and efficiency, while generalizing effectively to unseen speech. These results position UDPNet as a robust solution for real-time speech synthesis applications. Sample audio is available at https://onexpeters.github.io/UDPNet.
>
---
#### [replaced 091] Knowledge Graphs are all you need: Leveraging KGs in Physics Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.05453v3](http://arxiv.org/pdf/2412.05453v3)**

> **作者:** Krishnasai Addala; Kabir Dev Paul Baghel; Dhruv Jain; Navya Gupta; Rishitej Reddy Vyalla; Chhavi Kirtani; Avinash Anand; Rajiv Ratn Shah
>
> **摘要:** This study explores the effectiveness of using knowledge graphs generated by large language models to decompose high school-level physics questions into sub-questions. We introduce a pipeline aimed at enhancing model response quality for Question Answering tasks. By employing LLMs to construct knowledge graphs that capture the internal logic of the questions, these graphs then guide the generation of subquestions. We hypothesize that this method yields sub-questions that are more logically consistent with the original questions compared to traditional decomposition techniques. Our results show that sub-questions derived from knowledge graphs exhibit significantly improved fidelity to the original question's logic. This approach not only enhances the learning experience by providing clearer and more contextually appropriate sub-questions but also highlights the potential of LLMs to transform educational methodologies. The findings indicate a promising direction for applying AI to improve the quality and effectiveness of educational content.
>
---
#### [replaced 092] LogProber: Disentangling confidence from contamination in LLM responses
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.14352v2](http://arxiv.org/pdf/2408.14352v2)**

> **作者:** Nicolas Yax; Pierre-Yves Oudeyer; Stefano Palminteri
>
> **摘要:** In machine learning, contamination refers to situations where testing data leak into the training set. The issue is particularly relevant for the evaluation of the performance of Large Language Models (LLMs), which are generally trained on gargantuan, and generally opaque, corpora of text scraped from the world wide web. Developing tools to detect contamination is therefore crucial to be able to fairly and properly track the evolution of the performance of LLMs. To date, only a few recent studies have attempted to address the issue of quantifying and detecting contamination in short text sequences, such as those commonly found in benchmarks. However, these methods have limitations that can sometimes render them impractical.In the present paper, we introduce LogProber, a novel, efficient algorithm that we show to be able to detect contamination in a black box setting that tries to tackle some of these drawbacks by focusing on the familiarity with the question rather than the answer. Here, we explore the properties of the proposed method in comparison with concurrent approaches, identify its advantages and limitations, and illustrate how different forms of contamination can go undetected depending on the design of the detection algorithm.
>
---
#### [replaced 093] Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05176v3](http://arxiv.org/pdf/2506.05176v3)**

> **作者:** Yanzhao Zhang; Mingxin Li; Dingkun Long; Xin Zhang; Huan Lin; Baosong Yang; Pengjun Xie; An Yang; Dayiheng Liu; Junyang Lin; Fei Huang; Jingren Zhou
>
> **摘要:** In this work, we introduce the Qwen3 Embedding series, a significant advancement over its predecessor, the GTE-Qwen series, in text embedding and reranking capabilities, built upon the Qwen3 foundation models. Leveraging the Qwen3 LLMs' robust capabilities in multilingual text understanding and generation, our innovative multi-stage training pipeline combines large-scale unsupervised pre-training with supervised fine-tuning on high-quality datasets. Effective model merging strategies further ensure the robustness and adaptability of the Qwen3 Embedding series. During the training process, the Qwen3 LLMs serve not only as backbone models but also play a crucial role in synthesizing high-quality, rich, and diverse training data across multiple domains and languages, thus enhancing the training pipeline. The Qwen3 Embedding series offers a spectrum of model sizes (0.6B, 4B, 8B) for both embedding and reranking tasks, addressing diverse deployment scenarios where users can optimize for either efficiency or effectiveness. Empirical evaluations demonstrate that the Qwen3 Embedding series achieves state-of-the-art results across diverse benchmarks. Notably, it excels on the multilingual evaluation benchmark MTEB for text embedding, as well as in various retrieval tasks, including code retrieval, cross-lingual retrieval and multilingual retrieval. To facilitate reproducibility and promote community-driven research and development, the Qwen3 Embedding models are publicly available under the Apache 2.0 license.
>
---
#### [replaced 094] 7B Fully Open Source Moxin-LLM/VLM -- From Pretraining to GRPO-based Reinforcement Learning Enhancement
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.06845v5](http://arxiv.org/pdf/2412.06845v5)**

> **作者:** Pu Zhao; Xuan Shen; Zhenglun Kong; Yixin Shen; Sung-En Chang; Timothy Rupprecht; Lei Lu; Enfu Nan; Changdi Yang; Yumei He; Weiyan Shi; Xingchen Xu; Yu Huang; Wei Jiang; Wei Wang; Yue Chen; Yong He; Yanzhi Wang
>
> **摘要:** Recently, Large Language Models (LLMs) have undergone a significant transformation, marked by a rapid rise in both their popularity and capabilities. Leading this evolution are proprietary LLMs like GPT-4 and GPT-o1, which have captured widespread attention in the AI community due to their remarkable performance and versatility. Simultaneously, open-source LLMs, such as LLaMA, have made great contributions to the ever-increasing popularity of LLMs due to the ease to customize and deploy the models across diverse applications. Although open-source LLMs present unprecedented opportunities for innovation and research, the commercialization of LLMs has raised concerns about transparency, reproducibility, and safety. Many open-source LLMs fail to meet fundamental transparency requirements by withholding essential components like training code and data, which may hinder further innovations on LLMs. To mitigate this issue, we introduce Moxin 7B, a fully open-source LLM developed, adhering to principles of open science, open source, open data, and open access. We release the pre-training code and configurations, training and fine-tuning datasets, and intermediate and final checkpoints, aiming to make continuous commitments to fully open-source LLMs. After pre-training the base model, we finetune the Moxin Base model with SOTA post-training framework and instruction data to obtain Moxin Instruct model. To improve the reasoning capability, we further finetune our Instruct model with chain-of-thought data distilled from DeepSeek R1, and then use Group Relative Policy Optimization (GRPO) following DeepSeek R1 to finetune our model, leading to the Moxin Reasoning model. Moreover, we develop our vision language model based on our Moxin model. Experiments show that our models achieve superior performance in various evaluations such as zero-shot evaluation, few-shot evaluation, and CoT evaluation.
>
---
#### [replaced 095] DREsS: Dataset for Rubric-based Essay Scoring on EFL Writing
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2402.16733v3](http://arxiv.org/pdf/2402.16733v3)**

> **作者:** Haneul Yoo; Jieun Han; So-Yeon Ahn; Alice Oh
>
> **备注:** To appear in ACL 2025. arXiv admin note: text overlap with arXiv:2310.05191
>
> **摘要:** Automated essay scoring (AES) is a useful tool in English as a Foreign Language (EFL) writing education, offering real-time essay scores for students and instructors. However, previous AES models were trained on essays and scores irrelevant to the practical scenarios of EFL writing education and usually provided a single holistic score due to the lack of appropriate datasets. In this paper, we release DREsS, a large-scale, standard dataset for rubric-based automated essay scoring with 48.9K samples in total. DREsS comprises three sub-datasets: DREsS_New, DREsS_Std., and DREsS_CASE. We collect DREsS_New, a real-classroom dataset with 2.3K essays authored by EFL undergraduate students and scored by English education experts. We also standardize existing rubric-based essay scoring datasets as DREsS_Std. We suggest CASE, a corruption-based augmentation strategy for essays, which generates 40.1K synthetic samples of DREsS_CASE and improves the baseline results by 45.44%. DREsS will enable further research to provide a more accurate and practical AES system for EFL writing education.
>
---
#### [replaced 096] AskToAct: Enhancing LLMs Tool Use via Self-Correcting Clarification
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.01940v2](http://arxiv.org/pdf/2503.01940v2)**

> **作者:** Xuan Zhang; Yongliang Shen; Zhe Zheng; Linjuan Wu; Wenqi Zhang; Yuchen Yan; Qiuying Peng; Jun Wang; Weiming Lu
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities in tool learning. In real-world scenarios, user queries are often ambiguous and incomplete, requiring effective clarification. However, existing interactive clarification approaches face two critical limitations: reliance on manually constructed datasets, which inherently constrains training data scale and diversity, and lack of error correction mechanisms during multi-turn clarification, leading to error accumulation that compromises both accuracy and efficiency. We present AskToAct, which addresses these challenges by exploiting the structural mapping between queries and their tool invocation solutions. Our key insight is that tool parameters naturally represent explicit user intents. By systematically removing key parameters from queries while retaining them as ground truth, we enable automated construction of high-quality training data. We further enhance model robustness through error-correction pairs and selective masking, enabling dynamic error detection during clarification interactions. Comprehensive experiments demonstrate that AskToAct significantly outperforms existing approaches, achieving above 57% accuracy in recovering critical unspecified intents and enhancing clarification efficiency by an average of 10.46% while maintaining high accuracy in tool invocation. Our framework exhibits robust performance across different model architectures and successfully generalizes to entirely unseen APIs without additional training, achieving performance comparable to GPT-4o with substantially fewer computational resources.
>
---
#### [replaced 097] Discovering Forbidden Topics in Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.17441v3](http://arxiv.org/pdf/2505.17441v3)**

> **作者:** Can Rager; Chris Wendler; Rohit Gandikota; David Bau
>
> **摘要:** Refusal discovery is the task of identifying the full set of topics that a language model refuses to discuss. We introduce this new problem setting and develop a refusal discovery method, Iterated Prefill Crawler (IPC), that uses token prefilling to find forbidden topics. We benchmark IPC on Tulu-3-8B, an open-source model with public safety tuning data. Our crawler manages to retrieve 31 out of 36 topics within a budget of 1000 prompts. Next, we scale the crawler to a frontier model using the prefilling option of Claude-Haiku. Finally, we crawl three widely used open-weight models: Llama-3.3-70B and two of its variants finetuned for reasoning: DeepSeek-R1-70B and Perplexity-R1-1776-70B. DeepSeek-R1-70B reveals patterns consistent with censorship tuning: The model exhibits "thought suppression" behavior that indicates memorization of CCP-aligned responses. Although Perplexity-R1-1776-70B is robust to censorship, IPC elicits CCP-aligned refusals answers in the quantized model. Our findings highlight the critical need for refusal discovery methods to detect biases, boundaries, and alignment failures of AI systems.
>
---
#### [replaced 098] Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.05202v3](http://arxiv.org/pdf/2502.05202v3)**

> **作者:** Nadav Timor; Jonathan Mamou; Daniel Korat; Moshe Berchansky; Gaurav Jain; Oren Pereg; Moshe Wasserblat; David Harel
>
> **备注:** ICML'25 Oral (top %1)
>
> **摘要:** Accelerating the inference of large language models (LLMs) is a critical challenge in generative AI. Speculative decoding (SD) methods offer substantial efficiency gains by generating multiple tokens using a single target forward pass. However, existing SD approaches require the drafter and target models to share the same vocabulary, thus limiting the pool of possible drafters, often necessitating the training of a drafter from scratch. We present three new SD methods that remove this shared-vocabulary constraint. All three methods preserve the target distribution (i.e., they are lossless) and work with off-the-shelf models without requiring additional training or modifications. Empirically, on summarization, programming, and long-context tasks, our algorithms demonstrate significant speedups of up to 2.8x over standard autoregressive decoding. By enabling any off-the-shelf model to serve as a drafter and requiring no retraining, this work substantially broadens the applicability of the SD framework in practice.
>
---
#### [replaced 099] Advancing Decoding Strategies: Enhancements in Locally Typical Sampling for LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.05387v2](http://arxiv.org/pdf/2506.05387v2)**

> **作者:** Jaydip Sen; Saptarshi Sengupta; Subhasis Dasgupta
>
> **备注:** This is the accepted but pre-reviewed version of the chapter that has been accepted for publication in the Springer volume 'Decision-Making in Computational Intelligence-Based Systems,' edited by Witold Pedrycz, Gilberto Rivera, Rose Ma Rodriguez, and Salvador Ibarra Martinez. The chapter is 39 pages long, and it contains 2 figures and 6 tables. This is NOT the final camera-ready version
>
> **摘要:** This chapter explores advancements in decoding strategies for large language models (LLMs), focusing on enhancing the Locally Typical Sampling (LTS) algorithm. Traditional decoding methods, such as top-k and nucleus sampling, often struggle to balance fluency, diversity, and coherence in text generation. To address these challenges, Adaptive Semantic-Aware Typicality Sampling (ASTS) is proposed as an improved version of LTS, incorporating dynamic entropy thresholding, multi-objective scoring, and reward-penalty adjustments. ASTS ensures contextually coherent and diverse text generation while maintaining computational efficiency. Its performance is evaluated across multiple benchmarks, including story generation and abstractive summarization, using metrics such as perplexity, MAUVE, and diversity scores. Experimental results demonstrate that ASTS outperforms existing sampling techniques by reducing repetition, enhancing semantic alignment, and improving fluency.
>
---
#### [replaced 100] Using Shapley interactions to understand how models use structure
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2403.13106v2](http://arxiv.org/pdf/2403.13106v2)**

> **作者:** Divyansh Singhvi; Diganta Misra; Andrej Erkelens; Raghav Jain; Isabel Papadimitriou; Naomi Saphra
>
> **备注:** Published in ACL 2025
>
> **摘要:** Language is an intricately structured system, and a key goal of NLP interpretability is to provide methodological insights for understanding how language models represent this structure internally. In this paper, we use Shapley Taylor interaction indices (STII) in order to examine how language and speech models internally relate and structure their inputs. Pairwise Shapley interactions measure how much two inputs work together to influence model outputs beyond if we linearly added their independent influences, providing a view into how models encode structural interactions between inputs. We relate the interaction patterns in models to three underlying linguistic structures: syntactic structure, non-compositional semantics, and phonetic coarticulation. We find that autoregressive text models encode interactions that correlate with the syntactic proximity of inputs, and that both autoregressive and masked models encode nonlinear interactions in idiomatic phrases with non-compositional semantics. Our speech results show that inputs are more entangled for pairs where a neighboring consonant is likely to influence a vowel or approximant, showing that models encode the phonetic interaction needed for extracting discrete phonemic representations.
>
---
#### [replaced 101] Phonology-Guided Speech-to-Speech Translation for African Languages
- **分类: eess.AS; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.23323v3](http://arxiv.org/pdf/2410.23323v3)**

> **作者:** Peter Ochieng; Dennis Kaburu
>
> **摘要:** We present a prosody-guided framework for speech-to-speech translation (S2ST) that aligns and translates speech \emph{without} transcripts by leveraging cross-linguistic pause synchrony. Analyzing a 6{,}000-hour East African news corpus spanning five languages, we show that \emph{within-phylum} language pairs exhibit 30--40\% lower pause variance and over 3$\times$ higher onset/offset correlation compared to cross-phylum pairs. These findings motivate \textbf{SPaDA}, a dynamic-programming alignment algorithm that integrates silence consistency, rate synchrony, and semantic similarity. SPaDA improves alignment $F_1$ by +3--4 points and eliminates up to 38\% of spurious matches relative to greedy VAD baselines. Using SPaDA-aligned segments, we train \textbf{SegUniDiff}, a diffusion-based S2ST model guided by \emph{external gradients} from frozen semantic and speaker encoders. SegUniDiff matches an enhanced cascade in BLEU (30.3 on CVSS-C vs.\ 28.9 for UnitY), reduces speaker error rate (EER) from 12.5\% to 5.3\%, and runs at an RTF of 1.02. To support evaluation in low-resource settings, we also release a three-tier, transcript-free BLEU suite (M1--M3) that correlates strongly with human judgments. Together, our results show that prosodic cues in multilingual speech provide a reliable scaffold for scalable, non-autoregressive S2ST.
>
---
#### [replaced 102] Raising the Bar: Investigating the Values of Large Language Models via Generative Evolving Testing
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2406.14230v5](http://arxiv.org/pdf/2406.14230v5)**

> **作者:** Han Jiang; Xiaoyuan Yi; Zhihua Wei; Ziang Xiao; Shu Wang; Xing Xie
>
> **备注:** ICML 2025
>
> **摘要:** Warning: Contains harmful model outputs. Despite significant advancements, the propensity of Large Language Models (LLMs) to generate harmful and unethical content poses critical challenges. Measuring value alignment of LLMs becomes crucial for their regulation and responsible deployment. Although numerous benchmarks have been constructed to assess social bias, toxicity, and ethical issues in LLMs, those static benchmarks suffer from evaluation chronoeffect, in which, as models rapidly evolve, existing benchmarks may leak into training data or become saturated, overestimating ever-developing LLMs. To tackle this problem, we propose GETA, a novel generative evolving testing approach based on adaptive testing methods in measurement theory. Unlike traditional adaptive testing methods that rely on a static test item pool, GETA probes the underlying moral boundaries of LLMs by dynamically generating test items tailored to model capability. GETA co-evolves with LLMs by learning a joint distribution of item difficulty and model value conformity, thus effectively addressing evaluation chronoeffect. We evaluated various popular LLMs with GETA and demonstrated that 1) GETA can dynamically create difficulty-tailored test items and 2) GETA's evaluation results are more consistent with models' performance on unseen OOD and i.i.d. items, laying the groundwork for future evaluation paradigms.
>
---
#### [replaced 103] NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00975v4](http://arxiv.org/pdf/2506.00975v4)**

> **作者:** Qichao Wang; Ziqiao Meng; Wenqian Cui; Yifei Zhang; Pengcheng Wu; Bingzhe Wu; Irwin King; Liang Chen; Peilin Zhao
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Inspired by the impressive capabilities of GPT-4o, there is growing interest in enabling speech language models (SLMs) to engage in natural, fluid spoken interactions with humans. Recent advancements have led to the development of several SLMs that demonstrate promising results in this area. However, current approaches have yet to fully exploit dual-channel speech data, which inherently captures the structure and dynamics of human conversation. In this work, we systematically explore the use of dual-channel speech data in the context of modern large language models, and introduce a novel generative modeling paradigm, Next-Token-Pair Prediction (NTPP), to enable speaker-independent dual-channel spoken dialogue learning using decoder-only architectures for the first time. We evaluate our approach on standard benchmarks, and empirical results show that our proposed method, NTPP, significantly improves the conversational abilities of SLMs in terms of turn-taking prediction, response coherence, and naturalness. Moreover, compared to existing methods, NTPP achieves substantially lower inference latency, highlighting its practical efficiency for real-time applications.
>
---
#### [replaced 104] Emphasising Structured Information: Integrating Abstract Meaning Representation into LLMs for Enhanced Open-Domain Dialogue Evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.01129v4](http://arxiv.org/pdf/2404.01129v4)**

> **作者:** Bohao Yang; Kun Zhao; Dong Liu; Liang Zhan; Chenghua Lin
>
> **摘要:** Automatic open-domain dialogue evaluation has attracted increasing attention, yet remains challenging due to the complexity of assessing response appropriateness. Traditional evaluation metrics, typically trained with true positive and randomly selected negative responses, tend to assign higher scores to responses that share greater content similarity with contexts. However, adversarial negative responses, despite possessing high lexical overlap with contexts, can be semantically incongruous. Consequently, existing metrics struggle to effectively evaluate such responses, resulting in low correlations with human judgments. While recent studies have demonstrated the effectiveness of Large Language Models (LLMs) for open-domain dialogue evaluation, they still face challenges in handling adversarial negative examples. We propose a novel evaluation framework that integrates Abstract Meaning Representation (AMR) enhanced domain-specific language models (SLMs) with LLMs. Our SLMs explicitly incorporate AMR graph information through a gating mechanism for enhanced semantic representation learning, while both SLM predictions and AMR knowledge are integrated into LLM prompts for robust evaluation. Extensive experiments on open-domain dialogue evaluation tasks demonstrate the superiority of our method compared to state-of-the-art baselines. Our comprehensive ablation studies reveal that AMR graph information contributes substantially more to performance improvements. Our framework achieves strong correlations with human judgments across multiple datasets, establishing a new benchmark for dialogue evaluation. Our code and data are publicly available.
>
---
#### [replaced 105] Modality-Balancing Preference Optimization of Large Multimodal Models by Adversarial Negative Mining
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.08022v2](http://arxiv.org/pdf/2506.08022v2)**

> **作者:** Chenxi Liu; Tianyi Xiong; Ruibo Chen; Yihan Wu; Junfeng Guo; Tianyi Zhou; Heng Huang
>
> **摘要:** The task adaptation and alignment of Large Multimodal Models (LMMs) have been significantly advanced by instruction tuning and further strengthened by recent preference optimization. Yet, most LMMs still suffer from severe modality imbalance during reasoning, i.e., outweighing language prior biases over visual inputs, which bottlenecks their generalization to downstream tasks and causes hallucinations. However, existing preference optimization approaches for LMMs do not focus on restraining the internal biases of their Large Language Model (LLM) backbones when curating the training data. Moreover, they heavily rely on offline data and lack the capacity to explore diverse responses adaptive to dynamic distributional shifts during training. Meanwhile, Group Relative Policy Optimization (GRPO), a recent method using online-generated data and verified rewards to improve reasoning capabilities, remains largely underexplored in LMM alignment. In this paper, we propose a novel preference learning framework, Modality-Balancing Preference Optimization (MBPO), to address the modality imbalance in LMMs. MBPO constructs a more effective offline preference dataset by generating hard negatives, i.e., rejected responses misled by LLM biases due to limited usage of visual information, through adversarial perturbation of input images. Moreover, MBPO leverages the easy-to-verify nature of close-ended tasks to generate online responses with verified rewards. GRPO is then employed to train the model with offline-online hybrid data. Extensive experiments demonstrate that MBPO can enhance LMM performance on challenging vision-language tasks and effectively reduce hallucinations.
>
---
#### [replaced 106] Chem42: a Family of chemical Language Models for Target-aware Ligand Generation
- **分类: cs.LG; cs.AI; cs.CL; q-bio.BM**

- **链接: [http://arxiv.org/pdf/2503.16563v2](http://arxiv.org/pdf/2503.16563v2)**

> **作者:** Aahan Singh; Engin Tekin; Maryam Nadeem; Nancy A. ElNaker; Mohammad Amaan Sayeed; Natalia Vassilieva; Boulbaba Ben Amor
>
> **摘要:** Revolutionizing drug discovery demands more than just understanding molecular interactions - it requires generative models that can design novel ligands tailored to specific biological targets. While chemical Language Models (cLMs) have made strides in learning molecular properties, most fail to incorporate target-specific insights, restricting their ability to drive de-novo ligand generation. Chem42, a cutting-edge family of generative chemical Language Models, is designed to bridge this gap. By integrating atomic-level interactions with multimodal inputs from Prot42, a complementary protein Language Model, Chem42 achieves a sophisticated cross-modal representation of molecular structures, interactions, and binding patterns. This innovative framework enables the creation of structurally valid, synthetically accessible ligands with enhanced target specificity. Evaluations across diverse protein targets confirm that Chem42 surpasses existing approaches in chemical validity, target-aware design, and predicted binding affinity. By reducing the search space of viable drug candidates, Chem42 could accelerate the drug discovery pipeline, offering a powerful generative AI tool for precision medicine. Our Chem42 models set a new benchmark in molecule property prediction, conditional molecule generation, and target-aware ligand design. The models are publicly available at huggingface.co/inceptionai.
>
---
#### [replaced 107] MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.12977v4](http://arxiv.org/pdf/2411.12977v4)**

> **作者:** Mircea Lică; Ojas Shirekar; Baptiste Colle; Chirag Raman
>
> **摘要:** Embodied agents powered by large language models (LLMs), such as Voyager, promise open-ended competence in worlds such as Minecraft. However, when powered by open-weight LLMs they still falter on elementary tasks after domain-specific fine-tuning. We propose MindForge, a generative-agent framework for cultural lifelong learning through explicit perspective taking. We introduce three key innovations: (1) a structured theory of mind representation linking percepts, beliefs, desires, and actions; (2) natural inter-agent communication; and (3) a multi-component memory system. Following the cultural learning framework, we test MindForge in both instructive and collaborative settings within Minecraft. In an instructive setting with GPT-4, MindForge agents powered by open-weight LLMs significantly outperform their Voyager counterparts in basic tasks yielding $3\times$ more tech-tree milestones and collecting $2.3\times$ more unique items than the Voyager baseline. Furthermore, in fully \textit{collaborative} settings, we find that the performance of two underachieving agents improves with more communication rounds, echoing the Condorcet Jury Theorem. MindForge agents demonstrate sophisticated behaviors, including expert-novice knowledge transfer, collaborative problem solving, and adaptation to out-of-distribution tasks through accumulated cultural experiences.
>
---
#### [replaced 108] MMREC: LLM Based Multi-Modal Recommender System
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2408.04211v2](http://arxiv.org/pdf/2408.04211v2)**

> **作者:** Jiahao Tian; Jinman Zhao; Zhenkai Wang; Zhicheng Ding
>
> **摘要:** The importance of recommender systems is growing rapidly due to the exponential increase in the volume of content generated daily. This surge in content presents unique challenges for designing effective recommender systems. Key among these challenges is the need to effectively leverage the vast amounts of natural language data and images that represent user preferences. This paper presents a novel approach to enhancing recommender systems by leveraging Large Language Models (LLMs) and deep learning techniques. The proposed framework aims to improve the accuracy and relevance of recommendations by incorporating multi-modal information processing and by the use of unified latent space representation. The study explores the potential of LLMs to better understand and utilize natural language data in recommendation contexts, addressing the limitations of previous methods. The framework efficiently extracts and integrates text and image information through LLMs, unifying diverse modalities in a latent space to simplify the learning process for the ranking model. Experimental results demonstrate the enhanced discriminative power of the model when utilizing multi-modal information. This research contributes to the evolving field of recommender systems by showcasing the potential of LLMs and multi-modal data integration to create more personalized and contextually relevant recommendations.
>
---
