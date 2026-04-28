# 自然语言处理 cs.CL

- **最新发布 169 篇**

- **更新 121 篇**

## 最新发布

#### [new 001] Long-Context Aware Upcycling: A New Frontier for Hybrid LLM Scaling
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出HyLo，一种将预训练Transformer模型升级为混合架构的方法，解决长文本处理效率与内存限制问题，提升长上下文性能。**

- **链接: [https://arxiv.org/pdf/2604.24715](https://arxiv.org/pdf/2604.24715)**

> **作者:** Parsa Ashrafi Fashi; Utkarsh Saxena; Mehdi Rezagholizadeh; Aref Jafari; Akash Haridas; Mingyu Yang; Vansh Bhatia; Guihong Li; Vikram Appia; Emad Barsoum
>
> **摘要:** Hybrid sequence models that combine efficient Transformer components with linear sequence modeling blocks are a promising alternative to pure Transformers, but most are still pretrained from scratch and therefore fail to reuse existing Transformer checkpoints. We study upcycling as a practical path to convert pretrained Transformer LLMs into hybrid architectures while preserving short-context quality and improving long-context capability. We call our solution \emph{HyLo} (HYbrid LOng-context): a long-context upcycling recipe that combines architectural adaptation with efficient Transformer blocks, Multi-Head Latent Attention (MLA), and linear blocks (Mamba2 or Gated DeltaNet), together with staged long-context training and teacher-guided distillation for stable optimization. HyLo extends usable context length by up to $32\times$ through efficient post-training and reduces KV-cache memory by more than $90\%$, enabling up to 2M-token prefill and decoding in our \texttt{vLLM} inference stack, while comparable Llama baselines run out of memory beyond 64K context. Across 1B- and 3B-scale settings (Llama- and Qwen-based variants), HyLo delivers consistently strong short- and long-context performance and significantly outperforms state-of-the-art upcycled hybrid baselines on long-context evaluations such as RULER. Notably, at similar scale, HyLo-Qwen-1.7B trained on only 10B tokens significantly outperforms JetNemotron (trained on 400B tokens) on GSM8K, Lm-Harness common sense reasoning and RULER-64K.
>
---
#### [new 002] When Chain-of-Thought Fails, the Solution Hides in the Hidden States
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究推理过程中的信息编码问题，通过分析CoT生成的隐藏状态，揭示其在错误情况下仍包含可恢复的解题信息，为提升模型推理能力提供新思路。**

- **链接: [https://arxiv.org/pdf/2604.23351](https://arxiv.org/pdf/2604.23351)**

> **作者:** Houman Mehrafarin; Amit Parekh; Ioannis Konstas
>
> **摘要:** Whether intermediate reasoning is computationally useful or merely explanatory depends on whether chain-of-thought (CoT) tokens contain task-relevant information. We present a mechanistic causal analysis of CoT on GSM8K using activation patching: transferring token-level hidden states from a CoT generation to a direct-answer run for the same question, then measuring the effect on final-answer accuracy. Across models, generating after patching yields substantially higher accuracy than both direct-answer prompting and the original CoT trace, revealing that individual CoT tokens can encode sufficient information to recover the correct answer, even when the original trace is incorrect. This task-relevant information is more prevalent in correct than incorrect CoT runs and is unevenly distributed across tokens, concentrating in mid-to-late layers and appearing earlier in the reasoning trace. Moreover, patching language tokens such as verbs and entities carry task-solving information that steers generation toward correct reasoning, whereas mathematical tokens encode answer-proximal content that rarely succeeds. Patched outputs are often shorter and yet exceed the accuracy of a full CoT trace, suggesting complete reasoning chains are not always necessary. Together, these findings demonstrate that CoT encodes recoverable, token-level problem-solving information, offering new insight into how reasoning is represented and where it breaks down.
>
---
#### [new 003] Benchmarking Testing in Automated Theorem Proving
- **分类: cs.CL**

- **简介: 该论文属于形式化定理证明任务，旨在解决生成定理语义正确性评估难题。提出T框架，通过依赖定理编译成功判断生成定理正确性。**

- **链接: [https://arxiv.org/pdf/2604.23698](https://arxiv.org/pdf/2604.23698)**

> **作者:** Jongyoon Kim; Hojae Han; Seung-won Hwang
>
> **备注:** ACL 2026 Industry
>
> **摘要:** Recent advances in large language models (LLMs) have shown promise in formal theorem proving, yet evaluating semantic correctness remains challenging. Existing evaluations rely on indirect proxies such as lexical overlap with human-annotated proof, or expensive manual inspection. Inspired by the shift from lexical comparison to test-based evaluation in code generation, we propose T , a framework that evaluates the semantic correctness of formal theorems: a generated theorem is considered correct only if all dependent successor theorems compile successfully, analogous to integration testing. We construct a benchmark from 5 real-world Lean 4 repositories, comprising 2,206 problems paired with 41 successor theorems on average, automatically extracted without human effort. Experiments demonstrate that while state-of-the-art models achieve high compilation success, they perform significantly worse under our semantic metric. The best model, Claude-Sonnet-4.5, achieves only 38.9% Testing Accuracy on the full set, given both natural language proof and successor theorems as context, revealing a critical gap in current theorem generation capabilities.
>
---
#### [new 004] Seeing Is No Longer Believing: Frontier Image Generation Models, Synthetic Visual Evidence, and Real-World Risk
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI安全任务，探讨合成图像带来的现实风险。分析生成模型能力及滥用案例，提出风险框架与管控建议。**

- **链接: [https://arxiv.org/pdf/2604.24197](https://arxiv.org/pdf/2604.24197)**

> **作者:** Shuai Wu; Xue Li; Yanna Feng; Yufang Li; Zhijun Wang; Ran Wang
>
> **备注:** Technical report, 20 pages, 15 figures, 2 tables, 1 algorithm
>
> **摘要:** Frontier image generation has moved from artistic synthesis toward synthetic visual evidence. Systems such as GPT Image 2, Nano Banana Pro, Nano Banana 2, Grok Imagine, Qwen Image 2.0 Pro, and Seedream 5.0 Lite combine photorealistic rendering, readable typography, reference consistency, editing control, and in several cases reasoning or search-grounded image construction. These capabilities create large benefits for design, education, accessibility, and communication, yet they also weaken one of society's most common trust shortcuts: the belief that a plausible picture is a reliable record. This paper provides a source-grounded technical and policy analysis of synthetic visual risk. We first summarize the public capabilities of recent image models, then analyze public incidents involving fake crisis images, celebrity and public-figure imagery, medical scans, forged-looking documents, synthetic screenshots, phishing assets, and market-moving rumors. We introduce a capability-weighted risk framework that links model affordances to real-world harm in finance, medicine, news, law, emergency response, identity verification, and civic discourse. Our findings show that risk is driven less by photorealism alone than by the convergence of realism, legible text, identity persistence, fast iteration, and distribution context. We argue for layered control: model-side restrictions, cryptographic provenance, visible labeling, platform friction, sector-grade verification, and incident response. The paper closes with practical recommendations for model providers, platforms, newsrooms, financial institutions, healthcare systems, legal organizations, regulators, and ordinary users.
>
---
#### [new 005] Neural Grammatical Error Correction for Romanian
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语法错误纠正任务，针对罗马尼亚语资源匮乏的问题，构建了首个GEC语料库，并提出基于预训练的神经模型方法，提升低资源语言的纠错效果。**

- **链接: [https://arxiv.org/pdf/2604.23627](https://arxiv.org/pdf/2604.23627)**

> **作者:** Teodor-Mihai Cotet; Stefan Ruseti; Mihai Dascalu
>
> **摘要:** Resources for Grammatical Error Correction (GEC) in non-English languages are scarce, while available spellcheckers in these languages are mostly limited to simple corrections and rules. In this paper we introduce a first GEC corpus for Romanian consisting of 10k pairs of sentences. In addition, the German version of ERRANT (ERRor ANnotation Toolkit) scorer was adapted for Romanian to analyze this corpus and extract edits needed for evaluation. Multiple neural models were experimented, together with pretraining strategies, which proved effective for GEC in low-resource settings. Our baseline consists of a small Transformer model trained only on the GEC dataset (F0.5 of 44.38), whereas the best performing model is produced by pretraining a larger Transformer model on artificially generated data, followed by finetuning on the actual corpus (F0.5 of 53.76). The proposed method for generating additional training examples is easily extensible and can be applied to any language, as it requires only a POS tagger
>
---
#### [new 006] AI Safety Training Can be Clinically Harmful
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于AI心理健康应用评估任务，旨在解决AI在心理治疗中可能带来的安全风险。研究评估了四个模型在不同治疗场景中的表现，发现其安全性与有效性存在严重问题。**

- **链接: [https://arxiv.org/pdf/2604.23445](https://arxiv.org/pdf/2604.23445)**

> **作者:** Suhas BN; Andrew M. Sherrill; Rosa I. Arriaga; Chris W. Wiese; Saeed Abdullah
>
> **备注:** 26 pages, 5 figures, 10 tables
>
> **摘要:** Large language models are being deployed as mental health support agents at scale, yet only 16% of LLM-based chatbot interventions have undergone rigorous clinical efficacy testing, and simulations reveal psychological deterioration in over one-third of cases. We evaluate four generative models on 250 Prolonged Exposure (PE) therapy scenarios and 146 CBT cognitive restructuring exercises (plus 29 severity-escalated variants), scored by a three-judge LLM panel. All models scored near-perfectly on surface acknowledgment (~0.91-1.00) while therapeutic appropriateness collapsed to 0.22-0.33 at the highest severity for three of four models, with protocol fidelity reaching zero for two. Under CBT severity escalation, one model's task completeness dropped from 92% to 71% while the frontier model's safety-interference score fell from 0.99 to 0.61. We identify a systematic, modality-spanning failure: RLHF safety alignment disrupts the therapeutic mechanism of action by grounding patients during imaginal exposure, offering false reassurance, inserting crisis resources into controlled exercises, and refusing to challenge distorted cognitions mentioning self-harm in PE; and through task abandonment or safety-preamble insertion during CBT cognitive restructuring. These findings motivate a five-axis evaluation framework (protocol fidelity, hallucination risk, behavioral consistency, crisis safety, demographic robustness), mapped onto FDA SaMD and EU AI Act requirements. We argue that no AI mental health system should proceed to deployment without passing multi-axis evaluation across all five dimensions.
>
---
#### [new 007] MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗诊断任务，旨在解决LLM在诊断推理中的知识不足和适应性差问题。提出MultiDx框架，通过多源知识整合提升诊断准确性与临床路径一致性。**

- **链接: [https://arxiv.org/pdf/2604.24186](https://arxiv.org/pdf/2604.24186)**

> **作者:** Yimin Deng; Zhenxi Lin; Yejing Wang; Guoshuai Zhao; Pengyue Jia; Zichuan Fu; Derong Xu; Yefeng Zheng; Xiangyu Zhao; Li Zhu; Xian Wu; Xueming Qian
>
> **备注:** ACL 2026 findings
>
> **摘要:** Diagnostic prediction and clinical reasoning are critical tasks in healthcare applications. While Large Language Models (LLMs) have shown strong capabilities in commonsense reasoning, they still struggle with diagnostic reasoning due to limited domain knowledge. Existing approaches often rely on internal model knowledge or static knowledge bases, resulting in knowledge insufficiency and limited adaptability, which hinder their capacity to perform diagnostic reasoning. Moreover, these methods focus solely on the accuracy of final predictions, overlooking alignment with standard clinical reasoning trajectories. To this end, we propose MultiDx, a two-stage diagnostic reasoning framework that performs differential diagnosis by analyzing evidence collected from multiple knowledge sources. Specifically, it first generates suspected diagnoses and reasoning paths by leveraging knowledge from web search, SOAP-formatted case, and clinical case database. Then it integrates multi-perspective evidence through matching, voting, and differential diagnosis to generate the final prediction.~Extensive experiments on two public benchmarks demonstrate the effectiveness of our approach.
>
---
#### [new 008] DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents
- **分类: cs.CL**

- **简介: 该论文提出DPEPO算法，解决LLM代理探索不足的问题，通过多环境并行交互提升探索多样性，增强环境理解。**

- **链接: [https://arxiv.org/pdf/2604.24320](https://arxiv.org/pdf/2604.24320)**

> **作者:** Junshuo Zhang; Chengrui Huang; Feng Guo; Zihan Li; Ke Shi; Menghua Jiang; Jiguo Yu; Shuo Shang; Shen Gao
>
> **备注:** Accepted by ACL 2026 main conference
>
> **摘要:** Large language model (LLM) agents that follow the sequential "reason-then-act" paradigm have achieved superior performance in many complex this http URL, these methods suffer from limited exploration and incomplete environmental understanding, as they interact with only a single environment per step. In this paper, we first introduce a novel paradigm that enables an agent to interact with multiple environments simultaneously and share cross-trajectory experiences. Building upon this paradigm, we further propose DPEPO, a reinforcement learning (RL) algorithm that encourages the agent to perform diverse parallel exploration. There are two stages in DPEPO: initial supervised fine-tuning (SFT) imparts basic parallel reasoning and action generation, followed by reinforcement learning stage with a hierarchical reward scheme. We design a parallel trajectory-level success reward and two step-level rewards: Diverse Action Reward and Diverse State Transition Reward, which actively penalize behavioral redundancy and promote broad exploration. Extensive experiments on ALFWorld and ScienceWorld show that DPEPO achieves state-of-the-art (SOTA) success rates, while maintaining comparable efficiency to strong sequential baselines. (Code is available at this https URL)
>
---
#### [new 009] Hidden States Know Where Reasoning Diverges: Credit Assignment via Span-Level Wasserstein Distance
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于强化学习任务，解决信用分配问题。通过分析隐藏状态分布，提出SHEAR方法实现细粒度优势重 weighting，无需额外标注或模型。**

- **链接: [https://arxiv.org/pdf/2604.23318](https://arxiv.org/pdf/2604.23318)**

> **作者:** Xinzhu Chen; Wei He; Huichuan Fan; Wenzhe Niu; Zhongxiang Sun; Xuanru Wang; Jiuchong Gao; Jinghua Hao; Renqing He; Weijie Yu
>
> **摘要:** Group Relative Policy Optimization (GRPO) performs coarse-grained credit assignment in reinforcement learning with verifiable rewards (RLVR) by assigning the same advantage to all tokens in a rollout. Process reward models can provide finer-grained supervision, but they require step-level annotation or additional reward modeling. We show that hidden-state distributions contain a useful signal for local reasoning quality that can be extracted using only outcome-level correctness labels available in RLVR. Specifically, within each GRPO group, the Wasserstein distance between span-level hidden state distributions of correct and incorrect rollouts increases around regions where their local reasoning quality diverges. This association holds both across examples and within individual trajectories, suggesting that hidden-state distributional divergence can serve as a self-supervision signal for fine-grained credit assignment. We formalize this observation with a separation theorem showing that, under mild structural assumptions, post-divergence spans have larger Wasserstein distances than pre-divergence spans whenever the population-level distributional gap exceeds finite-sample noise. Motivated by this result, we propose \textbf{S}pan-level \textbf{H}idden state \textbf{E}nabled \textbf{A}dvantage \textbf{R}eweighting (SHEAR), which modifies GRPO by using span-level Wasserstein distances to scale token-level advantages, amplifying updates on tokens whose hidden states are more separated from the opposing group. The method requires no additional model and only minimal changes to the training pipeline. Experiments on five mathematical reasoning benchmarks and five code generation benchmarks show improvements over standard GRPO and strong performance relative to supervised process reward models, while requiring no additional annotation or reward model training.
>
---
#### [new 010] Learning Selective LLM Autonomy from Copilot Feedback in Enterprise Customer Support Workflows
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于企业客服自动化任务，解决如何高效实现流程自动化的问题。通过学习操作员反馈，训练模型选择性执行任务，提升效率并减少人工干预。**

- **链接: [https://arxiv.org/pdf/2604.23855](https://arxiv.org/pdf/2604.23855)**

> **作者:** Nikita Borovkov; Elisei Rykov; Olga Tsymboi; Sergei Filimonov; Nikita Surnachev; Dmitry Bitman; Anatolii Potapov
>
> **摘要:** We present a deployed system that automates end-to-end customer support workflows inside an enterprise Business Process Management (BPM) platform. The approach is scalable in production and reaches selective automation within two weeks for a new process, leveraging supervision already generated at scale: structured per-case UI interaction traces and low-overhead copilot feedback, where operators either accept a suggestion or provide a correction. A staged deployment pipeline trains a next UI action policy, learns a critic from copilot feedback to calibrate abstention, and executes only high-confidence steps in the background while deferring uncertain decisions to operators and resuming from the updated UI state. This setup lets one operator supervise multiple concurrent sessions and be interrupted only when the system is uncertain. The system operates on a schema-driven view of the BPM interface and includes monitoring and safe fallbacks for production. In production, it automated 45% of sessions and reduced average handling time by 39% without degrading support quality level.
>
---
#### [new 011] Resource-Lean Lexicon Induction for German Dialects
- **分类: cs.CL**

- **简介: 该论文属于词典构建任务，旨在解决德语方言资源稀缺问题。通过统计模型提升词典质量，优于大语言模型，实现跨方言迁移与查询扩展优化。**

- **链接: [https://arxiv.org/pdf/2604.23824](https://arxiv.org/pdf/2604.23824)**

> **作者:** Robert Litschko; Barbara Plank; Diego Frassinelli
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Automatic induction of high-quality dictionaries is essential for building lexical resources, yet low-resource languages and dialects pose several challenges: limited access to annotators, high degree of spelling variations, and poor performance of large language models (LLMs). We empirically show that statistical models (random forests) trained on string similarity features are surprisingly effective for inducing German dialect lexicons. They outperform LLMs, enable cross-dialect transfer, and offer a lightweight data-driven alternative. We evaluate our models intrinsically on bilingual lexicon induction (BLI) and extrinsically on dialect information retrieval (IR). On BLI, random forests outperform Mistral-123b while being more resource-lean. On dialect IR with BM25, using our dialect dictionaries for query expansion yields relative improvements of up to 28.9% in nDCG@10 and 50.7% in Recall@100. Motivated by the resource scarcity in dialects, we further investigate the extent to which models transfer across different German dialects, and their performance under varying amounts of training data.
>
---
#### [new 012] LegalDrill: Diagnosis-Driven Synthesis for Legal Reasoning in Small Language Models
- **分类: cs.CL**

- **简介: 该论文属于法律推理任务，旨在解决小语言模型在高风险法律任务中的能力不足问题。通过构建诊断驱动的合成框架，提升模型的法律推理能力。**

- **链接: [https://arxiv.org/pdf/2604.23809](https://arxiv.org/pdf/2604.23809)**

> **作者:** Tianchun Li; Haochen Liu; Vishwa Pardeshi; Xingchen Wang; Tianci Liu; Huijun Zhao; Wei Fan; Jing Gao
>
> **备注:** ACL 2026 Industry Track
>
> **摘要:** Small language models (SLMs) are promising for real-world deployment due to their efficiency and low operational cost. However, their limited capacity struggles with high-stakes legal reasoning tasks that require coherent statute interpretation and logically consistent deduction. Furthermore, training SLMs for such tasks demands high-quality, concise reasoning trajectories, which are prohibitively expensive to manually collect and difficult to curate via standard rejection sampling, lacking granularity beyond final verdicts. To address these challenges, we propose {LegalDrill}, a diagnosis-driven synthesis framework that extracts and iteratively refines reasoning trajectories from a capable teacher via fine-grained prompting, then a self-reflective verification is employed to adaptively select the most effective data for the SLM student. The resulting data empower SLM training through supervised fine-tuning and direct preference optimization. Extensive experiments on several legal benchmarks demonstrate that {LegalDrill} significantly bolsters the legal reasoning capabilities of representative SLMs while bypassing the need for scarce expert annotations, paving a scalable path toward practical legal reasoning systems.
>
---
#### [new 013] Reheat Nachos for Dinner? Evaluating AI Support for Cross-Cultural Communication of Neologisms
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决非母语者在跨文化沟通中理解新词的难题。通过实验评估AI工具对学习和使用新词的效果，发现AI解释最有效，但仍有提升空间。**

- **链接: [https://arxiv.org/pdf/2604.23842](https://arxiv.org/pdf/2604.23842)**

> **作者:** Dayeon Ki; Yu Hou; Rachel Rudinger; Hal Daumé III; Marine Carpuat; Fumeng Yang
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Neologisms and emerging slang are central to daily conversation, yet challenging for non-native speakers (NNS) to interpret and use appropriately in cross-cultural communication with native speakers (NS). NNS increasingly make use of Artificial Intelligence (AI) tools to learn these words. We study the utility of such tools in mediating an informal communication scenario through a human-subjects study (N=234): NNS participants learn English neologisms with AI support, write messages using the learned word to an NS friend, and judge contextual appropriateness of the neologism in two provided writing samples. Using both NS evaluator-rated communicative competence of NNS-produced writing and NNS' contextual appropriateness judgments, we compare three AI-based support conditions: AI Definition, AI Rewrite into simpler English, AI Explanation of meaning and usage, and Non-AI Dictionary for comparison. We show that AI Explanation yields the largest gains over no support in NS-rated competence, while contextual appropriateness judgments show indifference across support. NNS participants' self-reported perceptions tend to overestimate NS ratings, revealing a mismatch between perceived and actual competence. We further observe a significant gap between NNS- and NS-produced writing, highlighting the limitations of current AI tools and informing design for future tools.
>
---
#### [new 014] Stabilizing Efficient Reasoning with Step-Level Advantage Selection
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于高效推理任务，旨在解决长推理过程计算开销大和训练不稳定的问题。通过提出SAS方法，在保持准确率的同时减少推理长度。**

- **链接: [https://arxiv.org/pdf/2604.24003](https://arxiv.org/pdf/2604.24003)**

> **作者:** Han Wang; Xiaodong Yu; Jialian Wu; Jiang Liu; Ximeng Sun; Mohit Bansal; Zicheng Liu
>
> **备注:** Findings of ACL 2026, Code: this https URL
>
> **摘要:** Large language models (LLMs) achieve strong reasoning performance by allocating substantial computation at inference time, often generating long and verbose reasoning traces. While recent work on efficient reasoning reduces this overhead through length-based rewards or pruning, many approaches are post-trained under a much shorter context window than base-model training, a factor whose effect has not been systematically isolated. We first show that short-context post-training alone, using standard GRPO without any length-aware objective, already induces substantial reasoning compression-but at the cost of increasingly unstable training dynamics and accuracy degradation. To address this, we propose Step-level Advantage Selection (SAS), which operates at the reasoning-step level and assigns a zero advantage to low-confidence steps in correct rollouts and to high-confidence steps in verifier-failed rollouts, where failures often arise from truncation or verifier issues rather than incorrect reasoning. Across diverse mathematical and general reasoning benchmarks, SAS improves average Pass@1 accuracy by 0.86 points over the strongest length-aware baseline while reducing average reasoning length by 16.3%, yielding a better accuracy-efficiency trade-off.
>
---
#### [new 015] Beyond Local vs. External: A Game-Theoretic Framework for Trustworthy Knowledge Acquisition
- **分类: cs.CL**

- **简介: 该论文属于隐私与知识获取任务，解决用户敏感意图泄露与回答质量下降的矛盾。提出GTKA框架，通过博弈论优化隐私与效用平衡。**

- **链接: [https://arxiv.org/pdf/2604.23413](https://arxiv.org/pdf/2604.23413)**

> **作者:** Rujing Yao; Yufei Shi; Yang Wu; Ang Li; Zhuoren Jiang; XiaoFeng Wang; Haixu Tang; Xiaozhong Liu
>
> **摘要:** Cloud-hosted Large Language Models (LLMs) offer unmatched reasoning capabilities and dynamic knowledge, yet submitting raw queries to these external services risks exposing sensitive user intent. Conversely, relying exclusively on trusted local models preserves privacy but often compromises answer quality due to limited parameter scale and knowledge. To resolve this dilemma, we propose Game-theoretic Trustworthy Knowledge Acquisition (GTKA), a framework that formulates the trade-off between knowledge utility and privacy as a strategic game. GTKA consists of three components: (i) a privacy-aware sub-query generator that decomposes sensitive intent into generalized, low-risk fragments; (ii) an adversarial reconstruction attacker that attempts to infer the original query from these fragments, providing adaptive leakage signals; and (iii) a trusted local integrator that synthesizes external responses within a secure boundary. By training the generator and attacker in an alternating adversarial manner, GTKA optimizes the sub-query generation policy to maximize knowledge acquisition accuracy while minimizing the reconstructability of the original sensitive intent. To validate our approach, we construct two sensitive-domain benchmarks in the biomedical and legal fields. Extensive experiments demonstrate that GTKA significantly reduces intent leakage compared to state-of-the-art baselines while maintaining high-fidelity answer quality.
>
---
#### [new 016] Robust Audio-Text Retrieval via Cross-Modal Attention and Hybrid Loss
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于音频-文本检索任务，旨在解决长且噪声大的音频与文本对齐问题。通过跨模态注意力和混合损失函数提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.23323](https://arxiv.org/pdf/2604.23323)**

> **作者:** Meizhu Liu; Matthew Rowe; Amit Agarwal; Michael Avendi; Yassi Abbasi; Hitesh Laxmichand Patel; Paul Li; Kyu J. Han; Tao Sheng; Sujith Ravi; Dan Roth
>
> **摘要:** Audio-text retrieval enables semantic alignment between audio content and natural language queries, supporting applications in multimedia search, accessibility, and surveillance. However, current state-of-the-art approaches struggle with long, noisy, and weakly labeled audio due to their reliance on contrastive learning and large-batch training. We propose a novel multimodal retrieval framework that refines audio and text embeddings using a cross-modal embedding refinement module combining transformer-based projection, linear mapping, and bidirectional attention. To further improve robustness, we introduce a hybrid loss function blending cosine similarity, $\mathcal{L}_{1}$, and contrastive objectives, enabling stable training even under small-batch constraints. Our approach efficiently handles long-form and noisy audio (SNR 5 to 15) via silence-aware chunking and attention-based pooling. Experiments on benchmark datasets demonstrate improvements over prior methods.
>
---
#### [new 017] Quantum Knowledge Graph: Modeling Context-Dependent Triplet Validity
- **分类: cs.CL; cs.AI; cs.SC**

- **简介: 该论文属于知识图谱任务，旨在解决传统知识图谱在临床推理中无法体现上下文依赖的问题。通过构建量子知识图谱（QKG），实现上下文相关的三元组有效性建模，提升医疗问答的准确性。**

- **链接: [https://arxiv.org/pdf/2604.23972](https://arxiv.org/pdf/2604.23972)**

> **作者:** Yao Wang; Zixu Geng; Jun Yan
>
> **备注:** 15 pages main text, 6 pages appendix, 5 figures, preprint
>
> **摘要:** Knowledge graphs (KGs) are increasingly used to support large lan guage model (LLM) reasoning, but standard triplet-based KGs treat each relation as globally valid. In many settings, whether a relation should count as evidence depends on the context. We therefore formulate triplet validity as a triplet-specific function of context and refer to this formulation as a Quantum Knowledge Graph (QKG). We instantiate QKG in medicine using a diabetes-centered PrimeKG subgraph, whose 68,651 context-sensitive relations are further annotated with patient-group-specific constraints. We evaluate it in a reasoner--validator pipeline for medical question answering on a KG-grounded subset of MedReason containing 2,788 questions. With Haiku-4.5 as both the Reasoner and the Validator, KG-backed validation significantly improves over a no-validator baseline ($+0.61$ pp), and QKG with context matching yields the largest gain, outperforming both KG validation without context matching ($+0.79$ pp) and the no-validator baseline ($+1.40$ pp; paired McNemar, all $p<0.05$). Under a stronger validator (Qwen-3.6-Plus), the raw QKG gain over the no-validator baseline grows from $+1.40$ pp to $+5.96$ pp; the context-matching gap is non-significant ($p=0.73$) on the raw set but becomes borderline significant ($p=0.05$) after adjustment for knowledge leakage and suspicious questions, consistent with a benchmark-gold ceiling rather than a QKG limitation. Taken together, the results support the view that the value of a KG in LLM-based clinical reasoning lies not merely in storing medically related facts, but in representing whether those facts are applicable to the specific patient context. For reproducibility and further research, we release the curated QKG datasets and source code.\footnote{this https URL}
>
---
#### [new 018] A Benchmark Suite of Reddit-Derived Datasets for Mental Health Detection
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文提出一个基于Reddit的基准数据集，用于心理健康检测任务，解决数据质量差、可复现性低的问题，通过构建四个互补任务的数据集提升研究效率。**

- **链接: [https://arxiv.org/pdf/2604.23458](https://arxiv.org/pdf/2604.23458)**

> **作者:** Khalid Hasan; Jamil Saquer
>
> **备注:** In the proceedings of 12th Annual Conference on Computational Science & Computational Intelligence (CSCI'25)
>
> **摘要:** The growing availability of online support groups has opened up new windows to study mental health through natural language processing (NLP). However, it is hindered by a lack of high-quality, well-validated datasets. Existing studies have a tendency to build task-specific corpora without collecting them into widely available resources, and this makes reproducibility as well as cross-task comparison challenging. In this paper, we present a uniform benchmark set of four Reddit-based datasets for disjoint but complementary tasks: (i) detection of suicidal ideation, (ii) binary general mental disorder detection, (iii) bipolar disorder detection, and (iv) multi-class mental disorder classification. All datasets were established upon diligent linguistic inspection, well-defined annotation guidelines, and human-judgmental verification. Inter-annotator agreement metrics always exceeded the baseline agreement score of 0.8, ensuring the labels' trustworthiness. Previous work's evidence of performance on both transformer and contextualized recurrent models demonstrates that these models receive excellent performances on tasks (F1 ~ 93-99%), further validating the usefulness of the datasets. By combining these resources, we establish a unifying foundation for reproducible mental health NLP studies with the ability to carry out cross-task benchmarking, multi-task learning, and fair model comparison. The presented benchmark suite provides the research community with an easy-to-access and varied resource for advancing computational approaches toward mental health research.
>
---
#### [new 019] Sentiment and Emotion Classification of Indonesian E-Commerce Reviews via Multi-Task BiLSTM and AutoML Benchmarking
- **分类: cs.CL**

- **简介: 该论文针对印尼电商评论的情感和情绪分类任务，解决因语言混杂导致的传统工具失效问题，采用多任务BiLSTM和AutoML方法进行分类建模。**

- **链接: [https://arxiv.org/pdf/2604.24720](https://arxiv.org/pdf/2604.24720)**

> **作者:** Hermawan Manurung; Ibrahim Al-Kahfi; Ahmad Rizqi; Martin Clinton Tosima Manullang
>
> **备注:** 8 pages, 5 figures, 4 tables. Final project for Natural Language Processing course (PBA 2026) at Institut Teknologi Sumatera
>
> **摘要:** Indonesian marketplace reviews mix standard vocabulary with slang, regional loanwords, numeric shorthands, and emoji, making lexicon-based sentiment tools unreliable in practice. This paper describes a two-track classification pipeline applied to the PRDECT-ID dataset, which contains 5,400 product reviews from 29 Indonesian e-commerce categories, each labeled for binary sentiment (Positive/Negative) and five-class emotion (Happy, Sad, Fear, Love, Anger). The first track applies TF-IDF vectorization with a PyCaret AutoML sweep across standard classifiers. The second track is a PyTorch Bidirectional Long Short-Term Memory (BiLSTM) network with a shared encoder and two task-specific output heads. A preprocessing module applies 14 sequential cleaning steps, including a 140-entry slang dictionary assembled from marketplace corpora. Four configurations are benchmarked: BiLSTM Baseline, BiLSTM Improved, BiLSTM Large, and TextCNN. Training uses class-weighted cross-entropy loss, ReduceLROnPlateau scheduling, and early stopping. Both tracks are deployed as Gradio applications on Hugging Face Spaces. Source code is publicly available at this https URL.
>
---
#### [new 020] Scaling Properties of Continuous Diffusion Spoken Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语音语言模型的扩展特性，解决语音模型性能不足问题。通过引入pJSD指标，分析连续扩散模型的缩放规律，探索其在大规模数据下的表现与优化方向。**

- **链接: [https://arxiv.org/pdf/2604.24416](https://arxiv.org/pdf/2604.24416)**

> **作者:** Jason Ramapuram; Eeshan Gunesh Dhekane; Amitis Shidani; Dan Busbridge; Bogdan Mazoure; Zijin Gu; Russ Webb; Tatiana Likhomanenko; Navdeep Jaitly
>
> **摘要:** Speech-only spoken language models (SLMs) lag behind text and text-speech models in performance, with recent discrete autoregressive (AR) SLMs indicating significant computational and data demands to match text models. Since discretizing continuous speech for AR creates bottlenecks, we explore whether continuous diffusion (CD) SLM is more viable. To quantify the SLMs linguistic quality, we introduce the phoneme Jensen-Shannon divergence (pJSD) metric. Our analysis reveals CD SLMs, mirroring AR behavior, exhibit scaling laws for validation loss and pJSD, and show optimal token-to-parameter ratios decreasing as compute scales. However, for the latter, loss becomes insensitive to choice of data and model sizes, showing potential for fast inference. Scaling CD SLMs to 16B parameters with tens of millions of hours of conversational data enables generation of emotive, prosodic, multi-speaker, multilingual speech, though achieving long-form coherence remains a significant challenge.
>
---
#### [new 021] Generating Place-Based Compromises Between Two Points of View
- **分类: cs.CL**

- **简介: 该论文属于社会智能任务，旨在解决生成双方可接受妥协的问题。通过改进的提示工程方法，利用情感中立和反馈优化妥协生成，提升其接受度。**

- **链接: [https://arxiv.org/pdf/2604.24536](https://arxiv.org/pdf/2604.24536)**

> **作者:** Sumanta Bhattacharyya; Francine Chen; Scott Carter; Yan-Ying Chen; Tatiana Lau; Nayeli Suseth Bravo; Monica P. Van; Kate Sieck; Charlene C. Wu
>
> **摘要:** Large Language Models (LLMs) excel academically but struggle with social intelligence tasks, such as creating good compromises. In this paper, we present methods for generating empathically neutral compromises between two opposing viewpoints. We first compared four different prompt engineering methods using Claude 3 Opus and a dataset of 2,400 contrasting views on shared places. A subset of the gen erated compromises was evaluated for acceptability in a 50-participant study. We found that the best method for generating compromises between two views used external empathic similarity between a compromise and each viewpoint as iterative feedback, outperforming stan dard Chain of Thought (CoT) reasoning. The results indicate that the use of empathic neutrality improves the acceptability of compromises. The dataset of generated compromises was then used to train two smaller foundation models via margin-based alignment of human preferences, improving efficiency and removing the need for empathy estimation during inference.
>
---
#### [new 022] Your Students Don't Use LLMs Like You Wish They Did
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于教育人工智能任务，旨在解决学生与AI对话系统间教学目标不匹配的问题。通过提出六种计算指标，分析学生使用行为，发现学生多用于获取答案而非深度学习。**

- **链接: [https://arxiv.org/pdf/2604.23486](https://arxiv.org/pdf/2604.23486)**

> **作者:** Sebastian Kobler; Matthew Clemson; Angela Sun; Jonathan K. Kummerfeld
>
> **备注:** To appear at ACL 2026 (Main Conference)
>
> **摘要:** Educational NLP systems are typically evaluated using engagement metrics and satisfaction surveys, which are at best a proxy for meeting pedagogical goals. We introduce six computational metrics for automated evaluation of pedagogical alignment in student-AI dialogue. We validate our metrics through analysis of 12,650 messages across 500 conversations from four courses. Using our metrics, we identify a fundamental misalignment: educators design conversational tutors for sustained learning dialogue, but students mainly use them for answer-extraction. Deployment context is the strongest predictor of usage patterns, outweighing student preference or system design: when AI tools are optional, usage concentrates around deadlines; when integrated into course structure, students ask for solutions to verbatim assignment questions. Whole-dialogue evaluation misses these turn-by-turn patterns. Our metrics will enable researchers building educational dialogue systems to measure whether they are achieving their pedagogical goals.
>
---
#### [new 023] MemeScouts@LT-EDI 2026: Asking the Right Questions -- Prompted Weak Supervision for Meme Hate Speech Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态仇恨言论检测任务，针对表情包中隐含的仇恨内容进行识别。通过提问式弱监督方法分解任务，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.24179](https://arxiv.org/pdf/2604.24179)**

> **作者:** Ivo Bueno; Lea Hirlimann; Enkelejda Kasneci
>
> **备注:** Accepted at Sixth Workshop on Language Technology for Equality, Diversity and Inclusion at ACL2026 (LT-EDI@ACL26)
>
> **摘要:** Detecting hate speech in memes is challenging due to their multimodal nature and subtle, culturally grounded cues such as sarcasm and context. While recent vision-language models (VLMs) enable joint reasoning over text and images, end-to-end prompting can be brittle, as a single prediction must resolve target, stance, implicitness, and irony. These challenges are amplified in multilingual settings. We propose a prompted weak supervision (PWS) approach that decomposes meme understanding into targeted, question-based labeling functions with constrained answer options for homophobia and transphobia detection in the LT-EDI 2026 shared task. Using a quantized Qwen3-VLM to extract features by answering targeted questions, our method outperforms direct VLM classification, with substantial gains for Chinese and Hindi, ranking 1st in English, 2nd in Chinese, and 3rd in Hindi. Iterative refinement via error-driven LF expansion and feature pruning reduces redundancy and improves generalization. Our results highlight the effectiveness of prompted weak supervision for multilingual multimodal hate speech detection.
>
---
#### [new 024] Aligned Multi-View Scripts for Universal Chart-to-Code Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于图表到代码生成任务，解决现有方法仅支持Python的问题。构建了多语言数据集Chart2NCode，并提出CharLuMA模型，实现跨语言高效生成。**

- **链接: [https://arxiv.org/pdf/2604.24559](https://arxiv.org/pdf/2604.24559)**

> **作者:** Zhihan Zhang; Lizi Liao
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** Chart-to-code generation converts a chart image into an executable plotting script, enabling faithful reproduction and editable visualizations. Existing methods are largely Python-centric, limiting practical use and overlooking a critical source of supervision: the same chart can be expressed by semantically equivalent scripts in different plotting languages. To fill this gap, we introduce Chart2NCode, a dataset of 176K charts paired with aligned scripts in Python, R, and LaTeX that render visually equivalent outputs, constructed via a metadata-to-template pipeline with rendering verification and human quality checks. Building on a LLaVA-style architecture, we further propose CharLuMA, a parameter-efficient adaptation module that augments the multimodal projector with a language-conditioned mixture of low-rank subspaces, allowing the model to share core chart understanding while specializing code generation to the target language through lightweight routing. Extensive experiments show consistent gains in executability and visual fidelity across all languages, outperforming strong open-source baselines and remaining competitive with proprietary systems. Further analyses reveal that balanced multi-language supervision benefits all languages and that the adapter allocates a compact shared core plus language-specific capacity. Codes and data are available at this https URL.
>
---
#### [new 025] Factual and Edit-Sensitive Graph-to-Sequence Generation via Graph-Aware Adaptive Noising
- **分类: cs.CL**

- **简介: 该论文属于图到文本生成任务，解决事实准确性和编辑敏感性问题。提出DLM4G框架，通过自适应去噪策略提升生成质量。**

- **链接: [https://arxiv.org/pdf/2604.24104](https://arxiv.org/pdf/2604.24104)**

> **作者:** Aditya Hemant Shahane; Anuj Kumar Sirohi; Tanmoy Chakraborty; Prathosh A P; Sandeep Kumar
>
> **摘要:** Fine-tuned autoregressive models for graph-to-sequence generation (G2S) often struggle with factual grounding and edit sensitivity. To tackle these issues, we propose a non-autoregressive diffusion framework that generates text by iterative refinement conditioned on an input graph, named as Diffusion Language Model for Graphs (DLM4G). By aligning graph components (entities/relations) with their corresponding sequence tokens, DLM4G employs an adaptive noising strategy. The proposed strategy uses per-token denoising error as a signal to adaptively modulate noise on entity and relation tokens, improving preservation of graph structure and enabling localized updates under graph edits. Evaluated on three datasets, DLM4G consistently outperforms competitive G2S diffusion baselines trained on identical splits across both surface-form and embedding-based metrics. DLM4G further exceeds fine-tuned autoregressive baselines up to 12x larger (e.g., T5-Large) and is competitive with zero-shot LLM transfer baselines up to 127x larger. Relative to the strongest fine-tuned PLM baseline, DLM4G improves factual grounding (FGT@0.5) by +5.16% and edit sensitivity (ESR) by +7.9%; compared to the best diffusion baseline, it yields gains of +3.75% in FGT@0.5 and +23.6% in ESR. We additionally demonstrate applicability beyond textual graphs through experiments on molecule captioning, indicating the method's generality for scientific G2S generation.
>
---
#### [new 026] Pref-CTRL: Preference Driven LLM Alignment using Representation Editing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型对齐任务，旨在通过偏好数据优化模型输出。提出Pref-CTRL框架，使用多目标价值函数提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2604.23543](https://arxiv.org/pdf/2604.23543)**

> **作者:** Imranul Ashrafi; Inigo Jauregi Unanue; Massimo Piccardi
>
> **备注:** Accepted to the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** Test-time alignment methods offer a promising alternative to fine-tuning by steering the outputs of large language models (LLMs) at inference time with lightweight interventions on their internal representations. Recently, a prominent and effective approach, RE-Control (Kong et al., 2024), has proposed leveraging an external value function trained over the LLM's hidden states to guide generation via gradient-based editing. While effective, this method overlooks a key characteristic of alignment tasks, i.e. that they are typically formulated as learning from human preferences between candidate responses. To address this, in this paper we propose a novel preference-based training framework, Pref-CTRL, that uses a multi-objective value function to better reflect the structure of preference data. Our approach has outperformed RE-Control on two benchmark datasets and showed greater generalization on out-of-domain datasets. Our source code is available at this https URL.
>
---
#### [new 027] PeeriScope: A Multi-Faceted Framework for Evaluating Peer Review Quality
- **分类: cs.CL**

- **简介: 该论文提出PeeriScope，用于评估学术同行评审质量。解决同行评审质量评估的系统化问题，通过多维度分析实现可解释、可扩展的评估。**

- **链接: [https://arxiv.org/pdf/2604.24071](https://arxiv.org/pdf/2604.24071)**

> **作者:** Sajad Ebrahimi; Soroush Sadeghian; Ali Ghorbanpour; Negar Arabzadeh; Sara Salamat; Seyed Mohammad Hosseini; Hai Son Le; Mahdi Bashari; Ebrahim Bagheri
>
> **摘要:** The increasing scale and variability of peer review in scholarly venues has created an urgent need for systematic, interpretable, and extensible tools to assess review quality. We present PeeriScope, a modular platform that integrates structured features, rubric-guided large language model assessments, and supervised prediction to evaluate peer review quality along multiple dimensions. Designed for openness and integration, PeeriScope provides both a public interface and a documented API, supporting practical deployment and research extensibility. The demonstration illustrates its use for reviewer self-assessment, editorial triage, and large-scale auditing, and it enables the continued development of quality evaluation methods within scientific peer review. PeeriScope is available both as a live demo at this https URL and via API services at this https URL.
>
---
#### [new 028] ContextWeaver: Selective and Dependency-Structured Memory Construction for LLM Agents
- **分类: cs.CL**

- **简介: 该论文提出ContextWeaver，解决LLM代理在长对话中记忆管理的问题，通过构建依赖结构化的记忆图，提升推理效率和准确性。**

- **链接: [https://arxiv.org/pdf/2604.23069](https://arxiv.org/pdf/2604.23069)**

> **作者:** Yating Wu; Yuhao Zhang; Sayan Ghosh; Sourya Basu; Anoop Deoras; Jun Huan; Gaurav Gupta
>
> **摘要:** Large language model (LLM) agents often struggle in long-context interactions. As the agent accumulates more interaction history, context management approaches such as sliding window and prompt compression may omit earlier structured information that later steps rely on. Recent retrieval-based memory systems surface relevant content but still overlook the causal and logical structure needed for multi-step reasoning. We introduce ContextWeaver, a selective and dependency-structured memory framework that organizes an agent's interaction trace into a graph of reasoning steps and selects the relevant context for future actions. Unlike prior context management approaches, ContextWeaver supports: (1) dependency-based construction and traversal that link each step to the earlier steps it relies on; (2) compact dependency summarization that condenses root-to-step reasoning paths into reusable units; and (3) a lightweight validation layer that incorporates execution feedback. On the SWE-Bench Verified and Lite benchmarks, ContextWeaver improves performance over a sliding-window baseline in pass@1, while reducing reasoning steps and token usage. Our observations suggest that modeling logical dependencies provides a stable and scalable memory mechanism for LLM agents that use tools.
>
---
#### [new 029] How Sensitive Are Safety Benchmarks to Judge Configuration Choices?
- **分类: cs.CL**

- **简介: 该论文属于安全评估任务，研究 judge 配置对安全基准的影响。通过实验发现，提示词变化显著影响评估结果，揭示了测量偏差问题。**

- **链接: [https://arxiv.org/pdf/2604.24074](https://arxiv.org/pdf/2604.24074)**

> **作者:** Xinran Zhang
>
> **备注:** Accepted by the 22nd International Conference on Intelligent Computing (ICIC 2026). Final version to appear in Springer CCIS
>
> **摘要:** Safety benchmarks such as HarmBench rely on LLM judges to classify model responses as harmful or safe, yet the judge configuration, namely the combination of judge model and judge prompt, is typically treated as a fixed implementation detail. We show this assumption is problematic. Using a 2 x 2 x 3 factorial design, we construct 12 judge prompt variants along two axes, evaluation structure and instruction framing, and apply them using a single judge model, Claude Sonnet 4-6, producing 28,812 judgments over six target models and 400 HarmBench behaviors. We find that prompt wording alone, holding the judge model fixed, shifts measured harmful-response rates by up to 24.2 percentage points, with even within-condition surface rewording causing swings of up to 20.1 percentage points. Model safety rankings are moderately unstable, with mean Kendall tau = 0.89, and category-level sensitivity ranges from 39.6 percentage points for copyright to 0 percentage points for harassment. A supplementary multi-judge experiment using three judge models shows that judge-model choice adds further variance. Our results demonstrate that judge prompt wording is a substantial, previously under-examined source of measurement variance in safety benchmarking.
>
---
#### [new 030] SeaEvo: Advancing Algorithm Discovery with Strategy Space Evolution
- **分类: cs.CL; cs.AI; cs.NE**

- **简介: 该论文提出SeaEvo，解决LLM引导进化搜索中策略表示不足的问题，通过引入模块化策略空间层提升算法发现效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.24372](https://arxiv.org/pdf/2604.24372)**

> **作者:** Sichun Luo; Yi Huang; Haochen Luo; Fengyuan Liu; Guanzhi Deng; Lei Li; Qinghua Yao; Zefa Hu; Junlan Feng; Qi Liu
>
> **摘要:** LLM-guided evolutionary search has emerged as a promising paradigm for automated algorithm discovery, yet most systems track search progress primarily through executable programs and scalar fitness. Even when natural-language reflection is used, it is often used locally in mutation prompts or stored without an explicit population-level organization of strategic directions. As a result, evolutionary search can struggle to distinguish syntactically different implementations of the same idea, preserve lower-fitness but strategically promising directions, or detect when an entire family of strategies has saturated. We introduce \model, a modular strategy-space layer that elevates natural-language strategy descriptions from transient prompt context to first-class population-level evolutionary state in LLM-driven program search. \model augments each candidate program with an explicit natural language strategy description and uses this representation in three ways: Strategy Articulation turns mutation into a diagnose-direct-implement process; Stratified Experience Retrieval organizes the archive into strategy clusters and selects inspirations by behavioral complementarity; and Strategic Landscape Navigation periodically summarizes effective, saturated, and underexplored strategy families to guide future mutations. Across mathematical algorithm discovery, systems optimization, and agent-scaffold benchmarks, \model improves the underlying evolutionary backbones in most settings, with particularly large gains (21% relative improvement) on open-ended system optimization tasks. These results suggest that persistent strategy representations provide a practical mechanism for improving the robustness and efficiency of LLM-guided evolutionary search, suggesting a path toward compound AI systems that accumulate algorithmic knowledge over time.
>
---
#### [new 031] From Skill Text to Skill Structure: The Scheduling-Structural-Logical Representation for Agent Skills
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SSL表示法，解决代理技能难以机器处理的问题。通过结构化表示提升技能搜索与评估效果，属于技能管理任务。**

- **链接: [https://arxiv.org/pdf/2604.24026](https://arxiv.org/pdf/2604.24026)**

> **作者:** Qiliang Liang; Hansi Wang; Zhong Liang; Yang Liu
>
> **备注:** 21 pages, 1 figure
>
> **摘要:** LLM agents increasingly rely on reusable skills, capability packages that combine instructions, control flow, constraints, and tool calls. In most current agent systems, however, skills are still represented by text-heavy artifacts, including this http URL-style documents and structured records whose machine-usable evidence remains embedded largely in natural-language descriptions. This poses a challenge for skill-centered agent systems: managing skill collections and using skills to support agent both require reasoning over invocation interfaces, execution structure, and concrete side effects that are often entangled in a single textual surface. An explicit representation of skill knowledge may therefore help make these artifacts easier for machines to acquire and leverage. Drawing on Memory Organization Packets, Script Theory, and Conceptual Dependency from Schank and Abelson's classical work on linguistic knowledge representation, we introduce what is, to our knowledge, the first structured representation for agent skill artifacts that disentangles skill-level scheduling signals, scene-level execution structure, and logic-level action and resource-use evidence: the Scheduling-Structural-Logical (SSL) representation. We instantiate SSL with an LLM-based normalizer and evaluate it on a corpus of skills in two tasks, Skill Discovery and Risk Assessment, and superiorly outperform the text-only baselines: in Skill Discovery, SSL improves MRR from 0.573 to 0.707; in Risk Assessment, it improves macro F1 from 0.744 to 0.787. These findings reveal that explicit, source-grounded structure makes agent skills easier to search and review. They also suggest that SSL is best understood as a practical step toward more inspectable, reusable, and operationally actionable skill representations for agent systems, rather than as a finished standard or an end-to-end mechanism for managing and using skills.
>
---
#### [new 032] Applications of the Transformer Architecture in AI-Assisted English Reading Comprehension
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究AI辅助英语阅读理解中的Transformer架构，解决模型可解释性差、算法偏见及性能不稳定问题，提出统一技术流程提升准确性与公平性。**

- **链接: [https://arxiv.org/pdf/2604.23615](https://arxiv.org/pdf/2604.23615)**

> **作者:** Ping Li
>
> **备注:** 9 pages, 5 figures, Conference paper for International Conference on Big Data Applications in Education and Engineering {ICBDAEE 2026)
>
> **摘要:** This paper studies interpretable and fair artificial intelligence architectures for understanding English reading. Introduced transformer-based models, integrating advanced attention mechanisms and gradient-based feature attribution. The model's lack of interpretability, reduction of algorithmic bias, and unreliable performance in learning environments are the current issues faced in natural language teaching. A unified technical pipeline has been constructed, including adversarial bias correction methods, token-level attribution analysis, and multi-head attention heatmap visualization. Experimental validation was conducted using a large-scale labeled English reading comprehension dataset, and the data partitioning scheme and parameter optimization procedures have been determined. The method significantly outperforms the state-of-the-art models for this task in terms of accuracy and macro-average F1 score; in some aspects, it even surpasses or closely matches the results of human evaluations. In multi-week user experiments, the explainable transformer improved teachers' trust and operability in feedback-based assessments within the scoring system. The proposed method aims to ensure high prediction accuracy and fairness for different learners. This indicates that it is a real-world educational application based on artificial intelligence with a focus on interpretation. Improve the user experience in AI-assisted reading comprehension systems, counteract biases, and enhance the details explained by transformers.
>
---
#### [new 033] DeepImagine: Learning Biomedical Reasoning via Successive Counterfactual Imagining
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于 biomedical reasoning 任务，旨在提升大语言模型在临床试验结果预测上的表现。通过 successive counterfactual imagining 方法，训练模型理解因果机制，提高预测准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.23054](https://arxiv.org/pdf/2604.23054)**

> **作者:** Youze Zheng; Jianyou Wang; Yuhan Chen; Matthew Feng; Longtian Bao; Hanyuan Zhang; Maxim Khan; Aditya K. Sehgal; Christopher D. Rosin; Umber Dube; Ramamohan Paturi
>
> **备注:** Preprint. Work in Progress
>
> **摘要:** Predicting the outcomes of prospective clinical trials remains a major challenge for large language models. Prior work has shown that both traditional correlational predictors, such as random forests and logistic regression, and strong commercial LLMs achieve limited performance on this task. In this paper, we propose DeepImagine, a framework for teaching LLMs biomedical reasoning through successive counterfactual imagining. The central idea is to approximate hidden causal mechanisms of clinical trials by training models to infer how observed trial results would change under controlled perturbations of experimental conditions, such as dosage, outcome measures, study arms, geography, and other trial attributes. To support this objective, we construct both natural and approximate counterfactual pairs from real clinical trials with reported outcomes. For settings where strict counterfactual supervision is available, such as paired outcome measures or dose-ranging study arms within the same trial, we train models with supervised fine-tuning. For broader settings where only approximate counterfactual pairs can be retrieved, we optimize models with reinforcement learning using verifiable rewards based on downstream benchmark correctness. We further augment training with synthetic reasoning traces that provide causally plausible explanations for local counterfactual transitions. Using this pipeline, we train language models under 10B parameters, including Qwen3.5-9B, and evaluate them on clinical trial outcome prediction. We aim to show that DeepImagine consistently improves over untuned language models and traditional correlational baselines. Finally, we aim to show that the learned reasoning trajectories provide interpretable signals about how models represent trial-level mechanisms, suggesting a practical path toward more mechanistic and scientifically useful biomedical language models.
>
---
#### [new 034] Culture-Aware Machine Translation in Large Language Models: Benchmarking and Investigation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决文化敏感内容翻译的问题。通过构建CanMT数据集和评估框架，分析模型在文化-aware场景中的表现及翻译策略的影响。**

- **链接: [https://arxiv.org/pdf/2604.24361](https://arxiv.org/pdf/2604.24361)**

> **作者:** Zekun Yuan; Yangfan Ye; Xiaocheng Feng; Baohang Li; Qichen Hong; Yunfei Lu; Dandan Tu; Bing Qin
>
> **备注:** 26pages,25 figures ACL2026 main conference, long paper
>
> **摘要:** Large language models (LLMs) have achieved strong performance in general machine translation, yet their ability in culture-aware scenarios remains poorly understood. To bridge this gap, we introduce CanMT, a Culture-Aware Novel-Driven Parallel Dataset for Machine Translation, together with a theoretically grounded, multi-dimensional evaluation framework for assessing cultural translation quality. Leveraging CanMT, we systematically evaluate a wide range of LLMs and translation systems under different translation strategy constraints. Our findings reveal substantial performance disparities across models and demonstrate that translation strategies exert a systematic influence on model behavior. Further analysis shows that translation difficulty varies across types of culture-specific items, and that a persistent gap remains between models' recognition of culture-specific knowledge and their ability to correctly operationalize it in translation outputs. In addition, incorporating reference translations is shown to substantially improve evaluation reliability in LLM-as-a-judge, underscoring their essential role in assessing culture-aware translation quality. The corpus and code are available at CanMT.
>
---
#### [new 035] DRACULA: Hunting for the Actions Users Want Deep Research Agents to Execute
- **分类: cs.CL**

- **简介: 该论文提出DRACULA数据集，研究用户对深度研究代理中间动作的偏好，解决如何生成有用动作的问题，通过模拟和干预提升动作预测效果。**

- **链接: [https://arxiv.org/pdf/2604.23815](https://arxiv.org/pdf/2604.23815)**

> **作者:** Nishant Balepur; Malachi Hamada; Varsha Kishore; Sergey Feldman; Amanpreet Singh; Pao Siangliulue; Joseph Chee Chang; Rachel Rudinger; Eunsol Choi; Jordan Lee Boyd-Graber; Doug Downey; Aakanksha Naik
>
> **备注:** In-progress Preprint
>
> **摘要:** Scientific Deep Research (DR) agents answer user queries by synthesizing research papers into multi-section reports. User feedback can improve their utility, but existing protocols only score the final report, making it hard to study and learn which intermediate actions DR agents should take to improve reports. We collect DRACULA, the first dataset with user feedback on intermediate actions for DR. Over five weeks, nineteen expert CS researchers ask queries to a DR system that proposes actions (e.g., "Add a section on datasets"). Our users select actions they prefer, then judge whether an output report applied their selections successfully, yielding 8,103 action preferences and 5,230 execution judgments. After confirming a DR agent can execute DRACULA's actions, we study the predictability of user-preferred actions via simulation-how well LLMs predict the actions users select-a step toward learning to generate useful actions. We discover: (1) LLM judges initially struggle to predict action selections, but improve most when using a user's full selection history, rather than self-reported or extrapolated user context signals; (2) Users' selections for the same query differ based on unstated goals, bottlenecking simulation and motivating affordances that let users steer reports; and (3) Our simulation results inform an online intervention that generates new actions based on the user's past interactions, which users pick most often in follow-up studies. Overall, while work extensively studies execution, DRACULA reveals a key challenge is deciding which actions to execute in the first place. We open-source DRACULA's study design, user feedback, and simulation tasks to spur future work on action feedback for long-horizon agents.
>
---
#### [new 036] DepthKV: Layer-Dependent KV Cache Pruning for Long-Context LLM Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决长文本推理中KV缓存内存瓶颈问题。通过提出DepthKV框架，按层分配缓存预算，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2604.24647](https://arxiv.org/pdf/2604.24647)**

> **作者:** Zahra Dehghanighobadi; Asja Fischer
>
> **摘要:** Long-context reasoning is a critical capability of large language models (LLMs), enabling applications such as long-document understanding, summarization, and code generation. However, efficient autoregressive inference relies on the key-value (KV) cache, whose memory footprint grows linearly with sequence length, leading to a major memory bottleneck. To mitigate this overhead, KV cache pruning methods discard cached tokens with low attention scores during inference. Most existing methods apply a uniform pruning ratio across layers, implicitly assuming that all layers contribute equally to overall model performance. We show that this assumption is suboptimal, as layers differ significantly in their sensitivity to pruning. We propose DepthKV, a layer-dependent pruning framework that allocates a fixed global KV budget across layers based on their sensitivity, rather than using a uniform allocation. Across multiple models and tasks, DepthKV consistently outperforms uniform pruning at the same global pruning ratio, demonstrating more effective utilization of the KV cache budget through layer-dependent allocation.
>
---
#### [new 037] The Pragmatic Persona: Discovering LLM Persona through Bridging Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的 persona discovery 任务，旨在解决现有方法依赖表面特征、忽视话语结构的问题。通过构建基于桥接推理的知识图谱，提升 persona 识别的语义连贯性与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.24079](https://arxiv.org/pdf/2604.24079)**

> **作者:** Jisoo Yang; Jongwon Ryu; Minuk Ma; Trung X. Pham; Junyeong Kim
>
> **备注:** 15 pages, 4 figures, accepted to ICPR 2026
>
> **摘要:** Large Language Models (LLMs) reveal inherent and distinctive personas through dialogue. However, most existing persona discovery approaches rely on surface-level lexical or stylistic cues, treating dialogue as a flat sequence of tokens and failing to capture the deeper discourse-level structures that sustain persona consistency. To address this limitation, we propose a novel analytical framework that interprets LLM dialogue through bridging inference -- implicit conceptual relations that connect utterances via shared world knowledge and discourse coherence. By modeling these relations as structured knowledge graphs, our approach captures latent semantic links that govern how LLMs organize meaning across turns, enabling persona discovery at the level of discourse coherence rather than surface realizations. Experimental results across multiple reasoning backbones and target LLMs, ranging from small-scale models to 80B-parameter systems, demonstrate that bridging-inference graphs yield significantly stronger semantic coherence and more stable persona identification than frequency or style-based baselines. These results show that persona traits are consistently encoded in the structural organization of discourse rather than isolated lexical patterns. This work presents a systematic framework for probing, extracting, and visualizing latent LLM personas through the lens of Cognitive Discourse Theory, bridging computational linguistics, cognitive semantics, and persona reasoning in large language models. Codes are available at this https URL
>
---
#### [new 038] JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems
- **分类: cs.CL**

- **简介: 该论文提出JudgeSense，用于评估大语言模型作为裁判时对提示敏感性的基准。任务是衡量模型在不同等价提示下的判断一致性，解决模型稳定性问题，通过JSS量化分析。**

- **链接: [https://arxiv.org/pdf/2604.23478](https://arxiv.org/pdf/2604.23478)**

> **作者:** Rohith Reddy Bellibatlu
>
> **备注:** 17 pages, 3 figures, 3 tables. Code: this https URL. Dataset (JudgeSense Benchmark): this https URL
>
> **摘要:** Large language models are increasingly deployed as automated judges for evaluating other models, yet the stability of their verdicts under semantically equivalent prompt paraphrases remains unmeasured. We introduce JudgeSense, a framework and benchmark for quantifying this property via the Judge Sensitivity Score (JSS), defined as the fraction of paraphrase pairs on which a judge returns an identical decision. Evaluating nine judge models on 494 validated paraphrase pairs, we find that coherence is the only task where judges meaningfully differ, with JSS ranging from 0.389 to 0.992. On factuality, all judges cluster near JSS about 0.63, driven by a polarity-inverted prompt artifact; after correction, factuality JSS rises to about 0.9. Pairwise tasks (preference and relevance) exhibit degenerate always-A behavior in 8 of 9 judges, indicating strong position bias. Model scale does not predict consistency. We release code, decision logs, and a validated paraphrase dataset to support standardized JSS reporting.
>
---
#### [new 039] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在时间推理上的不足。通过提出AdapTime方法，动态执行不同推理步骤以提升模型的时序理解能力。**

- **链接: [https://arxiv.org/pdf/2604.24175](https://arxiv.org/pdf/2604.24175)**

> **作者:** Yimin Deng; Yejing Wang; Zhenxi Lin; Zichuan Fu; Guoshuai Zhao; Derong Xu; Yefeng Zheng; Xiangyu Zhao; Xian Wu; Li Zhu; Xueming Qian
>
> **备注:** ACL 2026 findings
>
> **摘要:** Large language models have demonstrated strong reasoning capabilities in general knowledge question answering. However, their ability to handle temporal information remains limited. To address this limitation, existing approaches often involve external tools or manual verification and are tailored to specific scenarios, leading to poor generalizability. Moreover, these methods apply a fixed pipeline to all questions, overlooking the fact that different types of temporal questions require distinct reasoning strategies, which leads to unnecessary processing for simple cases and inadequate reasoning for complex ones. To this end, we propose AdapTime, an adaptive temporal reasoning method that dynamically executes reasoning steps based on the input context. Specifically, it involves three temporal reasoning actions: reformulate, rewrite and review, with an LLM planner guiding the reasoning process. AdapTime integrates seamlessly with state-of-the-art LLMs and significantly enhances their temporal reasoning capabilities without relying on external support. Extensive experiments demonstrate the effectiveness of our approach.
>
---
#### [new 040] VeriLLMed: Interactive Visual Debugging of Medical Large Language Models with Knowledge Graphs
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于医疗大模型调试任务，旨在解决医学LLM推理不可靠、错误难以定位的问题。工作包括构建知识图谱辅助分析，识别诊断错误类型，并提供可视化工具帮助开发者改进模型。**

- **链接: [https://arxiv.org/pdf/2604.23356](https://arxiv.org/pdf/2604.23356)**

> **作者:** Yurui Xiang; Xingyi Mao; Rui Sheng; Zixin Chen; Zelin Zang; Yuyang Wu; Haipeng Zeng; Huamin Qu; Yushi Sun; Yanna Lin
>
> **摘要:** Large language models (LLMs) show promise in medical diagnosis, but real-world deployment remains challenging due to high-stakes clinical decisions and imperfect reasoning reliability. As a result, careful inspection of model behavior is essential for assessing whether diagnostic reasoning is reliable and clinically grounded. However, debugging medical LLMs remains difficult. First, developers often lack sufficient medical domain expertise to interpret model errors in clinically meaningful terms. Second, models can fail across a large and diverse set of instances involving different input types, tasks, and reasoning steps, making it challenging for developers to prioritize which errors deserve focused inspection. Third, developers struggle to identify recurring error patterns across cases, as existing debugging practices are largely instance-centric and rely on manual inspection of isolated failures. To address these challenges, we present VeriLLMed, a visual analytics system that integrates external biomedical knowledge to audit and debug medical LLM diagnostic reasoning. VeriLLMed transforms model outputs into comparable reasoning paths, constructs knowledge graph-grounded reference paths, and identifies three recurring classes of diagnosis errors: relation errors, branch errors, and missing errors. Case studies and expert evaluation demonstrate that VeriLLMed helps developers identify clinically implausible reasoning and generate actionable insights that can inform the improvement of medical LLMs.
>
---
#### [new 041] The Chameleon's Limit: Investigating Persona Collapse and Homogenization in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在模拟个体行为时出现的“人格坍缩”问题，提出评估框架量化模型的多样性与行为复杂性，旨在提升多代理仿真中的群体多样性。**

- **链接: [https://arxiv.org/pdf/2604.24698](https://arxiv.org/pdf/2604.24698)**

> **作者:** Yunze Xiao; Vivienne J. Zhang; Chenghao Yang; Ningshan Ma; Weihao Xuan; Jen-tse Huang
>
> **摘要:** Applications based on large language models (LLMs), such as multi-agent simulations, require population diversity among agents. We identify a pervasive failure mode we term \emph{Persona Collapse}: agents each assigned a distinct profile nonetheless converge into a narrow behavioral mode, producing a homogeneous simulated population. To quantify persona collapse, we propose a framework that measures how much of the persona space a population occupies (Coverage), how evenly agents spread across it (Uniformity), and how rich the resulting behavioral patterns are (Complexity). Evaluating ten LLMs on personality simulation (BFI-44), moral reasoning, and self-introduction, we observe persona collapse along two axes: (1) Dimensions: a model can appear diverse on one axis yet structurally degenerate on another, and (2) Domains: the same model may collapse the most in personality yet be the most diverse in moral reasoning. Furthermore, item-level diagnostics reveal that behavioral variation tracks coarse demographic stereotypes rather than the fine-grained individual differences specified in each persona. Counter-intuitively, \textbf{the models achieving the highest per-persona fidelity consistently produce the most stereotyped populations}. We release our toolkit and data to support population-level evaluation of LLMs.
>
---
#### [new 042] Contextual Linear Activation Steering of Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型定制任务，解决固定强度引导导致效果不一致的问题，提出CLAS方法动态调整引导强度，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.24693](https://arxiv.org/pdf/2604.24693)**

> **作者:** Brandon Hsu; Daniel Beaglehole; Adityanarayanan Radhakrishnan; Mikhail Belkin
>
> **摘要:** Linear activation steering is a powerful approach for eliciting the capabilities of large language models and specializing their behavior using limited labeled data. While effective, existing methods often apply a fixed steering strength to all tokens, resulting in inconsistent steering quality across diverse input prompts. In this work, we introduce Contextual Linear Activation Steering (CLAS), a method that dynamically adapts linear activation steering to context-dependent steering strengths. Across eleven steering benchmarks and four model families, it consistently outperforms standard linear activation steering and matches or exceeds the performance of ReFT and LoRA in settings with limited labeled data. We therefore propose CLAS as a scalable, interpretable, and accurate method for specializing and steering large language models.
>
---
#### [new 043] Uncertainty Quantification for LLM Function-Calling
- **分类: cs.CL**

- **简介: 该论文属于LLM函数调用中的不确定性量化任务，旨在评估UQ方法在防止错误调用上的效果。研究发现多样本方法在FC中不如单样本有效，并提出改进方案。**

- **链接: [https://arxiv.org/pdf/2604.22985](https://arxiv.org/pdf/2604.22985)**

> **作者:** Zihuiwen Ye; Lukas Aichberger; Michael Kirchhof; Sinead Williamson; Luca Zappella; Yarin Gal; Arno Blaas; Adam Golinski
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed to autonomously solve real-world tasks. A key ingredient for this is the LLM Function-Calling paradigm, a widely used approach for equipping LLMs with tool-use capabilities. However, an LLM calling functions incorrectly can have severe implications, especially when their effects are irreversible, e.g., transferring money or deleting data. Hence, it is of paramount importance to consider the LLM's confidence that a function call solves the task correctly prior to executing it. Uncertainty Quantification (UQ) methods can be used to quantify this confidence and prevent potentially incorrect function calls. In this work, we present what is, to our knowledge, the first evaluation of UQ methods for LLM Function-Calling (FC). While multi-sample UQ methods, such as Semantic Entropy, show strong performance for natural language Q&A tasks, we find that in the FC setting, it offers no clear advantage over simple single-sample UQ methods. Additionally, we find that the particularities of FC outputs can be leveraged to improve the performance of existing UQ methods in this setting. Specifically, multi-sample UQ methods benefit from clustering FC outputs based on their abstract syntax tree parsing, while single-sample UQ methods can be improved by selecting only semantically meaningful tokens when calculating logit-based uncertainty scores.
>
---
#### [new 044] Small Language Model Helps Resolve Semantic Ambiguity of LLM Prompt
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM输入提示的语义歧义问题。通过预推理阶段的显式消歧，提升模型推理性能。**

- **链接: [https://arxiv.org/pdf/2604.23263](https://arxiv.org/pdf/2604.23263)**

> **作者:** Zhenzhen Huang; Chaoning Zhang; Fachrina Dewi Puspitasari; Jiaquan Zhang; Yitian Zhou; Shuxu Chen; Yang Yang
>
> **摘要:** Large language models (LLMs) are increasingly utilized in various complex reasoning tasks due to their excellent instruction following capability. However, the model's performance is highly dependent on the open-ended characteristics of the users' input prompt. Natural prompts often do not follow proper syntactic rules, which creates ambiguous queries that yield multiple interpretations. Such ambiguous prompts confuse the model in choosing the correct reasoning paths to answer questions. Prior works address this challenge by applying query editing during the LLM inference process without explicitly solving the root cause of the ambiguity. To address this limitation, we propose a pre-inference prompt optimization mechanism via explicit prompt disambiguation. Particularly, we identify semantic risks in the prompt, check their multi-perspective consistency, and resolve any semantic conflicts that arise. Finally, we organize the resolved ambiguities in a logically structured manner as a clean input to the LLM. By explicitly resolving semantic ambiguity, our method can produce a more focused attention distribution to the semantically essential tokens. We also leverage small language models (SLMs) as the main executor of prompt disambiguation to benefit from their efficient computation. Through comprehensive experiments on multiple benchmarks, we demonstrate that our method improves reasoning performance by 2.5 points at a cost of only \$0.02. Our study promotes explicit prompt disambiguation as an effective prompt optimization method without disturbing the internal mechanism of LLM inference.
>
---
#### [new 045] TexOCR: Advancing Document OCR Models for Compilable Page-to-LaTeX Reconstruction
- **分类: cs.CL**

- **简介: 该论文聚焦于科学文档的OCR任务，旨在将PDF页面重建为可编译的LaTeX。针对现有系统在结构和可编译性上的不足，提出TexOCR-Bench和TexOCR-Train，并训练出改进的模型。**

- **链接: [https://arxiv.org/pdf/2604.22880](https://arxiv.org/pdf/2604.22880)**

> **作者:** Chengye Wang; Lin Fu; Zexi Kuang; Yilun Zhao
>
> **备注:** Accepted by ACL 2026 Main
>
> **摘要:** Existing document OCR largely targets plain text or Markdown, discarding the structural and executable properties that make LaTeX essential for scientific publishing. We study page-level reconstruction of scientific PDFs into compilable LaTeX and introduce TexOCR-Bench, a benchmark, and TexOCR-Train, a large-scale training corpus, for this task. TexOCR-Bench features a multi-dimensional evaluation suite that jointly assesses transcription fidelity, structural faithfulness, and end-to-end compilability. Leveraging TexOCR-Train, we train a 2B-parameter model, TexOCR, using supervised fine-tuning (SFT) and reinforcement learning (RL) with verifiable rewards derived from LaTeX unit tests that directly enforce compilability and referential integrity. Experiments across 21 frontier models on TexOCR-Bench show that existing systems frequently violate key document invariants, including consistent section structure, correct float placement, and valid label-reference links, which undermines compilation reliability and downstream usability. Our analysis further reveals that RL with verifiable rewards yields consistent improvements over SFT alone, particularly on structural and compilation metrics.
>
---
#### [new 046] Green Shielding: A User-Centric Approach Towards Trustworthy AI
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI可信部署任务，解决用户输入微小变化导致模型输出不稳定的问题。通过构建医疗诊断基准，分析输入变化对模型行为的影响，提出Green Shielding方法提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2604.24700](https://arxiv.org/pdf/2604.24700)**

> **作者:** Aaron J. Li; Nicolas Sanchez; Hao Huang; Ruijiang Dong; Jaskaran Bains; Katrin Jaradeh; Zhen Xiang; Bo Li; Feng Liu; Aaron Kornblith; Bin Yu
>
> **摘要:** Large language models (LLMs) are increasingly deployed, yet their outputs can be highly sensitive to routine, non-adversarial variation in how users phrase queries, a gap not well addressed by existing red-teaming efforts. We propose Green Shielding, a user-centric agenda for building evidence-backed deployment guidance by characterizing how benign input variation shifts model behavior. We operationalize this agenda through the CUE criteria: benchmarks with authentic Context, reference standards and metrics that capture true Utility, and perturbations that reflect realistic variations in the Elicitation of model behavior. Guided by the PCS framework and developed with practicing physicians, we instantiate Green Shielding in medical diagnosis through HealthCareMagic-Diagnosis (HCM-Dx), a benchmark of patient-authored queries, together with structured reference diagnosis sets and clinically grounded metrics for evaluating differential diagnosis lists. We also study perturbation regimes that capture routine input variation and show that prompt-level factors shift model behavior along clinically meaningful dimensions. Across multiple frontier LLMs, these shifts trace out Pareto-like tradeoffs. In particular, neutralization, which removes common user-level factors while preserving clinical content, increases plausibility and yields more concise, clinician-like differentials, but reduces coverage of highly likely and safety-critical conditions. Together, these results show that interaction choices can systematically shift task-relevant properties of model outputs and support user-facing guidance for safer deployment in high-stakes domains. Although instantiated here in medical diagnosis, the agenda extends naturally to other decision-support settings and agentic AI systems.
>
---
#### [new 047] GraphPlanner: Graph Memory-Augmented Agentic Routing for Multi-Agent LLMs
- **分类: cs.CL**

- **简介: 该论文提出GraphPlanner，解决多智能体LLM中的路由问题，通过图记忆增强实现高效任务规划与协作。**

- **链接: [https://arxiv.org/pdf/2604.23626](https://arxiv.org/pdf/2604.23626)**

> **作者:** Tao Feng; Haozhen Zhang; Zijie Lei; Peixuan Han; Jiaxuan You
>
> **摘要:** LLM routing has achieved promising results in integrating the strengths of diverse models while balancing efficiency and performance. However, to support more realistic and challenging applications, routing must extend into agentic LLM settings, where task planning, multi-round cooperation among heterogeneous agents, and memory utilization are indispensable. To address this gap, we propose GraphPlanner, a heterogeneous graph memory-augmented agentic router for multi-agent LLMs that generates routing workflows for each query and supports both inductive and transductive inference. GraphPlanner formulates workflow generation as a Markov Decision Process (MDP), where at each step it selects both the LLM backbone and the agent role, including Planner, Executor, and Summarizer. By leveraging a heterogeneous graph, denoted as GARNet, to capture interaction memories among queries, agents, and responses, GraphPlanner integrates historical memory and workflow memory into richer state representations. The entire pipeline is optimized with reinforcement learning, jointly improving task-specific performance and computational efficiency. We evaluate GraphPlanner across 14 diverse LLM tasks and demonstrate that: (1) GraphPlanner outperforms strong single-round and multi-round routers, improving accuracy by up to 9.3% while reducing GPU cost from 186.26 GiB to 1.04 GiB; (2) GraphPlanner generalizes robustly to unseen tasks and LLMs, exhibiting strong zero-shot capabilities; and (3) GraphPlanner effectively leverages historical memories, supporting both inductive and transductive inference for more adaptive routing. Our code for GraphPlanner is released at this https URL.
>
---
#### [new 048] Implicit Framing in Obstetric Counseling Notes: A Grounded LLM Pipeline on a VBAC-Eligible Cohort
- **分类: cs.CL**

- **简介: 该论文属于医疗文本分析任务，旨在研究产科咨询中的隐性框架问题。通过构建VBAC合格队列并应用LLM分析语言框架，发现VBAC与RCS咨询语言存在显著差异。**

- **链接: [https://arxiv.org/pdf/2604.23059](https://arxiv.org/pdf/2604.23059)**

> **作者:** Baris Karacan; Barbara Di Eugenio; Patrick Thornton; Joanna Tess; Subhash Kumar Kolar
>
> **备注:** 10 pages. Accepted at IEEE ICHI 2026. This is the author-accepted manuscript
>
> **摘要:** Clinical framing -- the linguistic manner in which clinical information is presented -- can influence patient understanding and decision-making, with important implications for healthcare outcomes. Obstetrics is a high-stakes domain in which physicians counsel patients on delivery mode choices such as vaginal birth after cesarean (VBAC) and repeat cesarean section (RCS), yet counseling language remains underexplored in large-scale clinical text analysis. In this work, we analyze physician counseling language in 2,024 obstetric history and physical narratives for a rigorously defined cohort of patients for whom both VBAC and RCS were clinically viable options. To control for confounding due to medical contraindications, we first construct a VBAC-eligible cohort using structured clinical data supplemented by a large language model (LLM)-based extraction pipeline constrained to grounded, verbatim evidence from free-text narratives. We then apply a zero-shot LLM framework to categorize counseling segments into predefined framing categories capturing how physicians linguistically present delivery options. Our analysis reveals a significant difference in counseling framing distributions between VBAC and RCS notes; risk-focused language accounts for a substantially larger share of counseling segments in RCS documentation than in VBAC, with category-level differences confirmed by statistical testing, highlighting the value of controlled LLM-based framing analysis in obstetric care.
>
---
#### [new 049] SEARCH-R: Structured Entity-Aware Retrieval with Chain-of-Reasoning Navigator for Multi-hop Question Answering
- **分类: cs.CL**

- **简介: 该论文属于多跳问答任务，旨在解决推理路径生成不准确和检索信息实用性低的问题。提出SEARCH-R框架，结合结构化实体检索与链式推理导航，提升问答效果。**

- **链接: [https://arxiv.org/pdf/2604.24515](https://arxiv.org/pdf/2604.24515)**

> **作者:** Yuqing Fu; Yimin Deng; Wanyu Wang; Yuhao Wang; Yejing Wang; Hongshi Liu; Yiqi Wang; Xiao Han; Maolin Wang; Guoshuai Zhao; Yi Chang; Xiangyu Zhao
>
> **备注:** ACL2026 findings
>
> **摘要:** Multi-hop Question Answering (MHQA) aims to answer questions that require multi-step reasoning. It presents two key challenges: generating correct reasoning paths in response to the complex user queries, and accurately retrieving essential knowledge in the face of potential limitations in large language models (LLMs). Existing approaches primarily rely on prompt-based methods to generate reasoning paths, which are further combined with traditional sparse or dense retrieval to produce the final answer. However, the generation of reasoning paths commonly lacks effective control over the generative process, thus leading the reasoning astray. Meanwhile, the retrieval methods over-rely on knowledge matching or similarity scores rather than evaluating the practical utility of the information, resulting in retrieving homogeneous or non-useful information. Therefore, we propose a Structured Entity-Aware Retrieval with Chain-of-Reasoning Navigator framework named SEARCH-R. Specifically, SEARCH-R trains an end-to-end reasoning path navigator, which is able to provide a powerful sub-question decomposer by fine-tuning the Llama3.1-8B model. Moreover, a novel dependency tree-based retrieval is designed to evaluate the informational contribution of the document quantitatively. Extensive experiments on three challenging multi-hop datasets validate the effectiveness of the proposed framework. The code and dataset are available at: this https URL.
>
---
#### [new 050] MTRouter: Cost-Aware Multi-Turn LLM Routing with History-Model Joint Embeddings
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多轮任务中的模型路由问题，旨在在固定成本预算内选择最优模型。MTRouter通过联合嵌入历史与模型信息，优化模型选择，降低推理成本并提升性能。**

- **链接: [https://arxiv.org/pdf/2604.23530](https://arxiv.org/pdf/2604.23530)**

> **作者:** Yiqun Zhang; Hao Li; Zihan Wang; Shi Feng; Xiaocui Yang; Daling Wang; Bo Zhang; Lei Bai; Shuyue Hu
>
> **备注:** This work has accepted by ACL 2026
>
> **摘要:** Multi-turn, long-horizon tasks are increasingly common for large language models (LLMs), but solving them typically requires many sequential model invocations, accumulating substantial inference costs. Here, we study cost-aware multi-turn LLM routing: selecting which model to invoke at each turn from a model pool, given a fixed cost budget. We propose MTRouter, which encodes the interaction history and candidate models into joint history-model embeddings, and learns an outcome estimator from logged trajectories to predict turn-level model utility. Experiments show that MTRouter improves the performance-cost trade-off: on ScienceWorld, it surpasses GPT-5 while reducing total cost by 58.7%; on Humanity's Last Exam (HLE), it achieves competitive accuracy while reducing total cost by 43.4% relative to GPT-5, and these gains even carry over to held-out tasks. Further analyses reveal several mechanisms underlying its effectiveness: relative to prior multi-turn routers, MTRouter makes fewer model switches, is more tolerant to transient errors, and exhibits emergent specialization across models. Code: this https URL
>
---
#### [new 051] Improving Robustness of Tabular Retrieval via Representational Stability
- **分类: cs.CL; cs.AI; cs.IR; cs.IT**

- **简介: 该论文属于表格检索任务，解决结构化表格序列化方式不一致导致的检索不稳定问题。通过引入中心表示和轻量适配器提升检索鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.24040](https://arxiv.org/pdf/2604.24040)**

> **作者:** Kushal Raj Bhandari; Adarsh Singh; Jianxi Gao; Soham Dan; Vivek Gupta
>
> **摘要:** Transformer-based table retrieval systems flatten structured tables into token sequences, making retrieval sensitive to the choice of serialization even when table semantics remain unchanged. We show that semantically equivalent serializations, such as $\texttt{csv}$, $\texttt{tsv}$, $\texttt{html}$, $\texttt{markdown}$, and $\texttt{ddl}$, can produce substantially different embeddings and retrieval results across multiple benchmarks and retriever families. To address this instability, we treat serialization embedding as noisy views of a shared semantic signal and use its centroid as a canonical target representation. We show that centroid averaging suppresses format-specific variation and can recover the semantic content common to different serializations when format-induced shifts differ across tables. Empirically, centroid representations outrank individual formats in aggregate pairwise comparisons across $\texttt{MPNet}$, $\texttt{BGE-M3}$, $\texttt{ReasonIR}$, and $\texttt{SPLADE}$. We further introduce a lightweight residual bottleneck adapter on top of a frozen encoder that maps single-serialization embeddings towards centroid targets while preserving variance and enforcing covariance regularization. The adapter improves robustness for several dense retrievers, though gains are model-dependent and weaker for sparse lexical retrieval. These results identify serialization sensitivity as a major source of retrieval variance and show the promise of post hoc geometric correction for serialization-invariant table retrieval. Our code, datasets, and models are available at $\href{this https URL}{this https URL}$.
>
---
#### [new 052] Knowledge Vector of Logical Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究大语言模型中逻辑推理的知识表示。旨在解决不同推理类型间知识表示独立的问题，提出互补约束框架以增强其相互补充性。**

- **链接: [https://arxiv.org/pdf/2604.23877](https://arxiv.org/pdf/2604.23877)**

> **作者:** Zixuan Wang; Yuanyuan Lei
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Logical reasoning serve as a central capability in LLMs and includes three main forms: deductive, inductive, and abductive reasoning. In this work, we study the knowledge representations of these reasoning types in LLMs and analyze the correlations among them. Our analysis shows that each form of logical reasoning can be captured as a reasoning-specific knowledge vector in a linear representation space, yet these vectors are largely independent of each other. Motivated by cognitive science theory that these subforms of logical reasoning interact closely in the human brain, as well as our observation that the reasoning process for one type can benefit from the reasoning chain produced by another, we further propose to refine the knowledge representations of each reasoning type in LLMs to encourage complementarity between them. To this end, we design a complementary subspace-constrained refinement framework, which introduces a complementary loss that enables each reasoning vector to leverage auxiliary knowledge from the others, and a subspace constraint loss that prevents erasure of their unique characteristics. Through steering experiments along reasoning vectors, we find that refined vectors incorporating complementary knowledge yield consistent performance gains. We also conduct a mechanism-interpretability analysis of each reasoning vector, revealing insights into the shared and specific features of different reasoning in LLMs.
>
---
#### [new 053] Personality Shapes Gender Bias in Persona-Conditioned LLM Narratives Across English and Hindi: An Empirical Investigation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLM在不同人格特质下产生的性别偏见。通过实验分析人格与性别偏见的关系，揭示模型生成内容中的刻板印象问题。**

- **链接: [https://arxiv.org/pdf/2604.23600](https://arxiv.org/pdf/2604.23600)**

> **作者:** Tanay Kumar; Shreya Gautam; Aman Chadha; Vinija Jain; Francesco Pierri
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in persona-driven applications such as education, customer service, and social platforms, where models are prompted to adopt specific personas when interacting with users. While persona conditioning can improve user experience and engagement, it also raises concerns about how personality cues may interact with gender biases and stereotypes. In this work, we present a controlled study of persona-conditioned story generation in English and Hindi, where each story portrays a working professional in India producing context-specific artifacts (e.g., lesson plans, reports, letters) under systematically varied persona gender, occupational role, and personality traits from the HEXACO and Dark Triad frameworks. Across 23,400 generated stories from six state-of-the-art LLMs, we find that personality traits are significantly associated with both the magnitude and direction of gender bias. In particular, Dark Triad personality traits are consistently associated with higher gender-stereotypical representations compared to socially desirable HEXACO traits, though these associations vary across models and languages. Our findings demonstrate that gender bias in LLMs is not static but context-dependent. This suggests that persona-conditioned systems used in real-world applications may introduce uneven representational harms, reinforcing gender stereotypes in generated educational, professional, or social content.
>
---
#### [new 054] Overcoming Copyright Barriers in Corpus Distribution Through Non-Reversible Hashing
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决版权材料在语料共享中的障碍。通过非可逆哈希技术，允许合法共享标注，用户需拥有相同版本文本才能匹配标注。**

- **链接: [https://arxiv.org/pdf/2604.23412](https://arxiv.org/pdf/2604.23412)**

> **作者:** Arthur Amalvy; Vincent Labatut; Xavier Bost; Hen-Hsen Huang
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** While annotated corpora are crucial in the field of natural language processing (NLP), those containing copyrighted material are difficult to exchange among researchers. Yet, such corpora are necessary to fully represent the diversity of data found in the wild in the context of NLP tasks. We tackle this issue by proposing a method to lawfully and publicly share the annotations of copyrighted literary texts. The corpus creator shares the annotations in clear, along with a non-reversible hashed version of the source material. The corpus user must own the source material, and apply the same hash function to their own tokens, in order to match them to the shared annotations. Crucially, our method is robust to reasonable divergences in the version of the copyrighted data owned by the user. As an illustration, we present alignment experiments on different editions of novels. Our results show that our method is able to correctly align 98.7 to 99.79% of tokens depending on the novel, provided the user version is sufficiently close to the corpus creator's version. We publicly release novelshare, a Python implementation of our method.
>
---
#### [new 055] The Randomness Floor: Measuring Intrinsic Non-Randomness in Language Model Token Distributions
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型的内在非随机性，通过计算熵偏差（ED）衡量模型输出分布与均匀分布的差异，分析不同模型和架构的表现，揭示语言结构对随机性的影响。**

- **链接: [https://arxiv.org/pdf/2604.22771](https://arxiv.org/pdf/2604.22771)**

> **作者:** Jarosław Hryszko
>
> **备注:** 13 pages, 4 figures, 5 tables
>
> **摘要:** Language models cannot be random. This paper introduces Entropic Deviation (ED), the normalised KL divergence between a model's token distribution and the uniform distribution, and measures it systematically across 31,200 generations spanning seven models, two architectures (transformer and state space), nine prompt categories, three temperatures, and five languages. Under semantically neutral prompts (empty strings, random characters, nonsense syllables) transformers still exhibit ED of approximately 0.30, meaning that 88-93% of the non-randomness observed under semantic prompts is intrinsic to the learned weights rather than induced by context. Three transformer families (Gemma, Llama, Qwen) converge on nearly identical ED values despite different training data and vocabularies. A state space model (Mamba2) reveals a qualitatively different regime: twice the ED, three times lower within-sequence variance, and massive sensitivity to temperature (r = -0.78) where transformers are nearly immune (r < 0.05). Cross-lingual experiments with Qwen-32B show a stable gradient across five languages (English, Japanese, Chinese, Polish, Arabic) that does not correlate with token fertility and persists when two languages sharing an identical tokeniser subset are compared. These findings establish a structural lower bound on randomness in pretrained language models, characterise how this bound differs across architectures, and demonstrate that language itself modulates the bound independently of tokenisation.
>
---
#### [new 056] RouteNLP: Closed-Loop LLM Routing with Conformal Cascading and Distillation Co-Optimization
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出RouteNLP，解决大模型服务成本高的问题。通过路由策略、置信度校准和知识蒸馏优化，降低推理成本并保持任务质量。属于NLP模型优化任务。**

- **链接: [https://arxiv.org/pdf/2604.23577](https://arxiv.org/pdf/2604.23577)**

> **作者:** Dongxin Guo; Jikun Wu; Siu Ming Yiu
>
> **备注:** Accepted at ACL 2026 Industry Track. 13 pages, 2 figures, 15 tables, 1 algorithm
>
> **摘要:** Serving diverse NLP workloads with large language models is costly: at one enterprise partner, inference costs exceeded $200K/month despite over 70% of queries being routine tasks well within the capability of smaller models. We present RouteNLP, a closed-loop framework that routes queries across a tiered model portfolio to minimize cost while satisfying per-task quality constraints. The framework integrates three components: a difficulty-aware router with shared task-conditioned representations trained on preference data and quality signals; confidence-calibrated cascading that uses conformal prediction for distribution-free threshold initialization; and a distillation-routing co-optimization loop that clusters escalation failures, applies targeted knowledge distillation to cheaper models, and automatically retrains the router, yielding over twice the cost improvement of untargeted distillation. In an 8-week pilot deployment processing ~5K queries/day at an enterprise customer-service division, RouteNLP reduced inference costs by 58% while maintaining 91% response acceptance and reducing p99 latency from 1,847 ms to 387 ms. On a six-task benchmark spanning finance, customer service, and legal domains, the framework achieves 40-85% cost reduction while retaining 96-100% quality on structured tasks and 96-98% on generation tasks, with human evaluation confirming that 74.5% of routed generation outputs match or exceed frontier-model quality.
>
---
#### [new 057] Differentiable Faithfulness Alignment for Cross-Model Circuit Transfer
- **分类: cs.CL**

- **简介: 该论文提出DFA框架，解决跨模型电路对齐问题，通过学习可微对齐将小模型电路信息迁移至大模型，提升解释性与效率。**

- **链接: [https://arxiv.org/pdf/2604.24302](https://arxiv.org/pdf/2604.24302)**

> **作者:** Shun Shao; Binxu Wang; Shay B. Cohen; Anna Korhonen; Yonatan Belinkov
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Mechanistic interpretability has made it possible to localize circuits underlying specific behaviors in language models, but existing methods are expensive, model-specific, and difficult to scale to larger architectures. We introduce \textbf{Differentiable Faithfulness Alignment (DFA)}, a framework that transfers circuit information from a smaller source model to a larger target model through a learned differentiable alignment. DFA projects source-model node importance scores into the target model and trains this mapping with a soft faithfulness objective, avoiding full circuit discovery on the target model. We evaluate DFA on Llama-3 and Qwen-2.5 across six tasks spanning factual retrieval, multiple-choice reasoning, and arithmetic. The strongest results occur on Llama-3 $1$B$\rightarrow3$B, where aligned circuits are often competitive with direct node attribution and zero-shot transfer remains effective. Recovery weakens for larger source--target gaps and is substantially lower on Qwen-2.5, suggesting that transfer becomes harder as architectural and scaling differences increase. Overall, DFA consistently outperforms simple baselines and, in some settings, recovers target-model circuits with faithfulness comparable to or stronger than direct attribution. These results suggest that smaller models can provide useful mechanistic priors for larger ones, while highlighting both the promise and the limits of node-level cross-model circuit alignment.\footnote{Code is available at this https URL.
>
---
#### [new 058] Au-M-ol: A Unified Model for Medical Audio and Language Understanding
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Au-M-ol模型，解决医疗音频与语言理解任务，通过融合音频处理和大语言模型，提升医学语音识别的准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.23284](https://arxiv.org/pdf/2604.23284)**

> **作者:** Meizhu Liu; Nistha Mitra; Paul Li; Amine Abdaoui; Adam Ledyard; Tao Sheng
>
> **摘要:** In this work, we present Au-M-ol, a novel multimodal architecture that extends Large Language Models (LLMs) with audio processing. It is designed to improve performance on clinically relevant tasks such as Automatic Speech Recognition (ASR). Au-M-ol has three main components: (1) an audio encoder that extracts rich acoustic features from medical speech, (2) an adaptation layer that maps audio features into the LLM input space, and (3) a pretrained LLM that performs transcription and clinical language understanding. This design allows the model to interpret spoken medical content directly, improving both accuracy and robustness. In experiments, Au-M-ol reduces Word Error Rate (WER) by 56\% compared to state-of-the-art baselines on medical transcription tasks. The model also performs well in challenging conditions, including noisy environments, domain-specific terminology, and speaker variability. These results suggest that Au-M-ol is a strong candidate for real-world clinical applications, where reliable and context-aware audio understanding is essential.
>
---
#### [new 059] Kwai Summary Attention Technical Report
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决长序列建模中注意力机制计算成本高的问题。通过引入Kwai Summary Attention，压缩历史上下文以降低计算开销。**

- **链接: [https://arxiv.org/pdf/2604.24432](https://arxiv.org/pdf/2604.24432)**

> **作者:** Chenglong Chu; Guorui Zhou; Guowang Zhang; Han Li; Hao Peng; Hongtao Cheng; Jian Liang; Jiangxia Cao; Kun Gai; Lingzhi Zhou; Lu Ren; Qi Zhang; Ruiming Tang; Ruitao Wang; Xinchen Luo; Yi Su; Zhiyuan Liang; Ziqi Wang; Boyang Ding; Chengru Song; Dunju Zang; Hui Wang; Jiao Ou; Jiaxin Deng; Jijun Shi; Jinghao Zhang; Junmin Chen; Lejian Ren; Minxuan Lv; Qianqian Wang; Qigen Hu; Shiyao Wang; Siyang Mao; Tao Wang; Xingmei Wang; Zhixin Ling; Ziming Li; Zixing Zhang
>
> **备注:** Work in progress
>
> **摘要:** Long-context ability, has become one of the most important iteration direction of next-generation Large Language Models, particularly in semantic understanding/reasoning, code agentic intelligence and recommendation system. However, the standard softmax attention exhibits quadratic time complexity with respect to sequence length. As the sequence length increases, this incurs substantial overhead in long-context settings, leading the training and inference costs of extremely long sequences deteriorate rapidly. Existing solutions mitigate this issue through two technique routings: i) Reducing the KV cache per layer, such as from the head-level compression GQA, and the embedding dimension-level compression MLA, but the KV cache remains linearly dependent on the sequence length at a 1:1 ratio. ii) Interleaving with KV Cache friendly architecture, such as local attention SWA, linear kernel GDN, but often involve trade-offs among KV Cache and long-context modeling effectiveness. Besides the two technique routings, we argue that there exists an intermediate path not well explored: {Maintaining a linear relationship between the KV cache and sequence length, but performing semantic-level compression through a specific ratio $k$}. This $O(n/k)$ path does not pursue a ``minimum KV cache'', but rather trades acceptable memory costs for complete, referential, and interpretable retention of long distant dependency. Motivated by this, we propose Kwai Summary Attention (KSA), a novel attention mechanism that reduces sequence modeling cost by compressing historical contexts into learnable summary tokens.
>
---
#### [new 060] AIPsy-Affect: A Keyword-Free Clinical Stimulus Battery for Mechanistic Interpretability of Emotion in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感机制可解释性研究，解决情绪关键词干扰问题，提出AIPsy-Affect刺激集，实现无关键词的情绪诱发与验证。**

- **链接: [https://arxiv.org/pdf/2604.23719](https://arxiv.org/pdf/2604.23719)**

> **作者:** Michael Keeman
>
> **备注:** Dataset paper. 4 pages + appendix, 2 figures. Dataset available at this https URL. MIT license
>
> **摘要:** Mechanistic interpretability research on emotion in large language models -- linear probing, activation patching, sparse autoencoder (SAE) feature analysis, causal ablation, steering vector extraction -- depends on stimuli that contain the words for the emotions they test. When a probe fires on "I am furious", it is unclear whether the model has detected anger or detected the word "furious". The two readings have very different consequences for every downstream claim about emotion circuits, features, and interventions. We release AIPsy-Affect, a 480-item clinical stimulus battery that removes the confound at the stimulus level: 192 keyword-free vignettes evoking each of Plutchik's eight primary emotions through narrative situation alone, 192 matched neutral controls that share characters, setting, length, and surface structure with the affect surgically removed, plus moderate-intensity and discriminant-validity splits. The matched-pair structure supports linear probing, activation patching, SAE feature analysis, causal ablation, and steering vector extraction under a strong methodological guarantee: any internal representation that distinguishes a clinical item from its matched neutral cannot be doing so on the basis of emotion-keyword presence. A three-method NLP defense battery -- bag-of-words sentiment, an emotion-category lexicon, and a contextual transformer classifier -- confirms the property: bag-of-words methods see only situational vocabulary, and a contextual classifier detects affect (p < 10^-15) but cannot identify the category (5.2% top-1 vs. 82.5% on a keyword-rich control). AIPsy-Affect extends our earlier 96-item battery (arXiv:2603.22295) by a factor of four and is released openly under MIT license.
>
---
#### [new 061] Structural Pruning of Large Vision Language Models: A Comprehensive Study on Pruning Dynamics, Recovery, and Data Efficiency
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决LVLM在资源受限设备上的部署问题。通过结构化剪枝和轻量恢复训练，提升模型数据效率与计算效率。**

- **链接: [https://arxiv.org/pdf/2604.24380](https://arxiv.org/pdf/2604.24380)**

> **作者:** Yiran Huang; Lukas Thede; Massimiliano Mancini; Wenjia Xu; Zeynep Akata
>
> **备注:** Accepted at International Journal of Computer Vision (IJCV) 2026
>
> **摘要:** While Large Vision Language Models (LVLMs) demonstrate impressive capabilities, their substantial computational and memory requirements pose deployment challenges on resource-constrained edge devices. Current parameter reduction techniques primarily involve training LVLMs from small language models, but these methods offer limited flexibility and remain computationally intensive. We study a complementary route: compressing existing LVLMs by applying structured pruning to the language model backbone, followed by lightweight recovery training. Specifically, we investigate two structural pruning paradigms: layerwise and widthwise pruning, and pair them with supervised finetuning and knowledge distillation on logits and hidden states. Additionally, we assess the feasibility of conducting recovery training with only a small fraction of the available data. Our results show that widthwise pruning generally maintains better performance in low-resource scenarios, where computational resources are limited or there is insufficient finetuning data. As for the recovery training, finetuning only the multimodal projector is sufficient at small compression levels. Furthermore, a combination of supervised finetuning and hidden-state distillation yields optimal recovery across various pruning levels. Notably, effective recovery can be achieved using just 5% of the original data, while retaining over 95% of the original performance. Through empirical study on three representative LVLM families ranging from 3B to 7B parameters, this study offers actionable insights for practitioners to compress LVLMs without extensive computation resources or sufficient data. The code base is available at this https URL.
>
---
#### [new 062] Domain Fine-Tuning vs. Retrieval-Augmented Generation for Medical Multiple-Choice Question Answering: A Controlled Comparison at the 4B-Parameter Scale
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于医疗多选题问答任务，比较了领域微调与检索增强生成的效果，旨在解决模型在医疗知识应用中的性能提升问题。实验显示微调效果更优，而RAG效果不显著。**

- **链接: [https://arxiv.org/pdf/2604.23801](https://arxiv.org/pdf/2604.23801)**

> **作者:** Avi-ad Avraam Buskila
>
> **摘要:** Practitioners deploying small open-weight large language models (LLMs) for medical question answering face a recurring design choice: invest in a domain-fine-tuned model, or keep a general-purpose model and inject domain knowledge at inference time via retrieval-augmented generation (RAG). We isolate this trade-off by holding model size, prompt template, decoding temperature, retrieval pipeline, and evaluation protocol fixed, and varying only (i) whether the model has been domain-adapted (Gemma 3 4B vs. MedGemma 4B, both 4-bit quantized and served via Ollama) and (ii) whether retrieved passages from a medical knowledge corpus are inserted into the prompt. We evaluate all four cells of this 2x2 design on the full MedQA-USMLE 4-option test split (1,273 questions) with three repetitions per question (15,276 LLM calls). Domain fine-tuning yields a +6.8 percentage-point gain in majority-vote accuracy over the general 4B baseline (53.3% vs. 46.4%, McNemar p < 10^-4). RAG over MedMCQA explanations does not produce a statistically significant gain in either model, and in the domain-tuned model the point estimate is slightly negative (-1.9 pp, p = 0.16). At this scale and on this benchmark, domain knowledge encoded in weights dominates domain knowledge supplied in context. We release the full experiment code and JSONL traces to support replication.
>
---
#### [new 063] EPM-RL: Reinforcement Learning for On-Premise Product Mapping in E-Commerce
- **分类: cs.CL; cs.AI; cs.DB; cs.LG; cs.MA**

- **简介: 该论文提出EPM-RL框架，解决电商产品映射问题，通过强化学习提升模型准确性与效率，实现私有部署和低成本运营。**

- **链接: [https://arxiv.org/pdf/2604.23993](https://arxiv.org/pdf/2604.23993)**

> **作者:** Minhyeong Yu; Wonduk Seo
>
> **备注:** preprint
>
> **摘要:** Product mapping, the task of deciding whether two e-commerce listings refer to the same product, is a core problem for price monitoring and channel visibility. In real marketplaces, however, sellers frequently inject promotional keywords, platform-specific tags, and bundle descriptions into titles, causing the same product to appear under many different names. Recent LLM-based and multi-agent frameworks improve robustness and interpretability on such hard cases, but they often rely on expensive external APIs, repeated retrieval, and complex inference-time orchestration, making large-scale deployment costly and difficult in privacy-sensitive enterprise settings. To address these issues, we present EPM-RL, a reinforcement-learning-based framework for building an accurate and efficient on-premise e-commerce product mapping model. Our central idea is to distill high-cost agentic reasoning into a trainable in-house model. Starting from a curated set of product pairs with LLM-generated rationales and human verification, we first perform parameter-efficient fine-tuning (PEFT) on a small student model using structured reasoning outputs. We then further optimize the model with Reinforcement Learning (RL) using an agent-based reward that jointly evaluates output-format compliance, label correctness, reasoning--preference scores from specially designed judge models. Preliminary results show that EPM-RL consistently improves over PEFT-only training and offers a stronger quality--cost trade-off than commercial API-based baselines, while enabling private deployment and lower operational cost. These findings suggest that reinforcement learning can turn product mapping from a high-latency agentic pipeline into a scalable, inspectable, and production-ready in-house system.
>
---
#### [new 064] BiMol-Diff: A Unified Diffusion Framework for Molecular Generation and Captioning
- **分类: cs.CL**

- **简介: 该论文提出BiMol-Diff框架，解决分子生成与描述任务。针对传统方法在结构建模中的不足，引入基于token的噪声调度，提升生成精度与描述质量。**

- **链接: [https://arxiv.org/pdf/2604.24089](https://arxiv.org/pdf/2604.24089)**

> **作者:** Aditya Hemant Shahane; Anuj Kumar Sirohi; Devansh Arora; Nitin Kumar; Prathosh A P; Sandeep Kumar
>
> **摘要:** Bridging molecular structures and natural language is essential for controllable design. Autoregressive models struggle with long-range dependencies, while standard diffusion processes apply uniform corruption across positions, which can distort structurally informative tokens. We present BiMol-Diff, a unified diffusion framework for the paired tasks of text-conditioned molecule generation and molecule captioning. Our key component is a token-aware noise schedule that assigns position-dependent corruption based on token recovery difficulty, preserving harder-to-recover substructures during the forward process. On ChEBI-20 and M3-20M, BiMol-Diff improves molecule reconstruction with a 15.4% relative gain in Exact Match and achieves strong captioning results, attaining best BLEU and BERTScore among compared baselines. These results indicate token-aware noising improves fidelity in molecular structure-language modelling.
>
---
#### [new 065] From Similarity to Structure: Training-free LLM Context Compression with Hybrid Graph Priors
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本压缩任务，旨在解决长文本处理效率低和信息丢失的问题。提出一种无需训练的压缩框架，利用图结构优先级选择关键句子，提升压缩效果。**

- **链接: [https://arxiv.org/pdf/2604.23277](https://arxiv.org/pdf/2604.23277)**

> **作者:** Yitian Zhou; Chaoning Zhang; Jiaquan Zhang; Zhenzhen Huang; Jinyu Guo; Sung-Ho Bae; Lik-Hang Lee; Caiyan Qin; Yang Yang
>
> **摘要:** Long-context large language models remain computationally expensive to run and often fail to reliably process very long inputs, which makes context compression an important component of many systems. Existing compression approaches typically rely on trained compressors, dense retrieval-style selection, or heuristic trimming, and they often struggle to jointly preserve task relevance, topic coverage, and cross-sentence coherence under a strict token budget. To address this, we propose a training-free and model-agnostic compression framework that selects a compact set of sentences guided by structural graph priors. Our method constructs a sparse hybrid sentence graph that combines mutual k-NN semantic edges with short-range sequential edges, extracts a topic skeleton via clustering, and ranks sentences using an interpretable score that integrates task relevance, cluster representativeness, bridge centrality, and a cycle coverage cue. A budgeted greedy selection with redundancy suppression then produces a readable compressed context in original order. Experimental results on four datasets show that our approach is competitive with strong extractive and abstractive baselines, demonstrating larger gains on long-document benchmarks.
>
---
#### [new 066] Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style
- **分类: cs.CL**

- **简介: 该论文属于文本风格调整任务，旨在解决用户如何通过后编辑使LLM生成文本更符合个人风格的问题。研究通过实验分析后编辑效果，发现其能提升风格相似性，但仍有LLM痕迹且多样性降低。**

- **链接: [https://arxiv.org/pdf/2604.24444](https://arxiv.org/pdf/2604.24444)**

> **作者:** Connor Baumler; Calvin Bao; Huy Nghiem; Xinchen Yang; Marine Carpuat; Hal Daumé III
>
> **备注:** ACL 2026
>
> **摘要:** Despite the growing use of large language models (LLMs) for writing tasks, users may hesitate to rely on LLMs when personal style is important. Post-editing LLM-generated drafts or translations is a common collaborative writing strategy, but it remains unclear whether users can effectively reshape LLM-generated text to reflect their personal style. We conduct a pre-registered online study ($n=81$) in which participants post-edit LLM-generated drafts for writing tasks where personal style matters to them. Using embedding-based style similarity metrics, we find that post-editing increases stylistic similarity to participants' unassisted writing and reduces similarity to fully LLM-generated output. However, post-edited text still remains stylistically closer in style to LLM text than to participants' unassisted control text, and it exhibits reduced stylistic diversity compared to unassisted human text. We find a gap between perceived stylistic authenticity and model-measured stylistic similarity, with post-edited text often perceived as representative of participants' personal style despite remaining detectable LLM stylistic traces.
>
---
#### [new 067] IRIS: Interleaved Reinforcement with Incremental Staged Curriculum for Cross-Lingual Mathematical Reasoning
- **分类: cs.CL**

- **简介: 该论文提出IRIS框架，解决跨语言数学推理中的步骤一致性问题。通过渐进式课程学习与强化学习结合，提升多语言尤其是低资源语言的推理性能。**

- **链接: [https://arxiv.org/pdf/2604.24114](https://arxiv.org/pdf/2604.24114)**

> **作者:** Navya Gupta; Rishitej Reddy Vyalla; Avinash Anand; Chhavi Kirtani; Erik Cambria; Zhengchen Zhang; Zhengkui Wang; Timothy Liu; Aik Beng Ng; Simon See; Rajiv Ratn Shah
>
> **备注:** Accepted in ACL main
>
> **摘要:** Curriculum learning helps language models tackle complex reasoning by gradually increasing task difficulty. However, it often fails to generate consistent step-by-step reasoning, especially in multilingual and low-resource settings where cross-lingual transfer from English to Indian languages remains limited. We propose IRIS: Interleaved Reinforcement with Incremental Staged Curriculum, a two-axis framework that combines Supervised Fine-Tuning on progressively harder problems (vertical axis) with Reverse Curriculum Reinforcement Learning to reduce reliance on step-by-step guidance (horizontal axis). We design a composite reward combining correctness, step-wise alignment, continuity, and numeric incentives, optimized via Group Relative Policy Optimization (GRPO). We release CL-Math, a dataset of 29k problems with step-level annotations in English, Hindi, and Marathi. Across standard benchmarks and curated multilingual test sets, IRIS consistently improves performance, with strong results on math reasoning tasks and substantial gains in low-resource and bilingual settings, alongside modest improvements in high-resource languages.
>
---
#### [new 068] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination
- **分类: cs.CL**

- **简介: 该论文属于历史推理任务，旨在评估大语言模型在历史研究方面的能力。通过构建基于科举制度的基准测试，揭示模型在复杂历史问题上的不足。**

- **链接: [https://arxiv.org/pdf/2604.24690](https://arxiv.org/pdf/2604.24690)**

> **作者:** Lirong Gao; Zeqing Wang; Yuyan Cai; Jiayi Deng; Yanmei Gu; Yiming Zhang; Jia Zhou; Yanfei Zhang; Junbo Zhao
>
> **备注:** Accepted at ACL 2026
>
> **摘要:** While Large Language Models (LLMs) have increasingly assisted in historical tasks such as text processing, their capacity for professional-level historical reasoning remains underexplored. Existing benchmarks primarily assess basic knowledge breadth or lexical understanding, failing to capture the higher-order skills, such as evidentiary reasoning,that are central to historical research. To fill this gap, we introduce ProHist-Bench, a novel benchmark anchored in the Chinese Imperial Examination (Keju) system, a comprehensive microcosm of East Asian political, social, and intellectual history spanning over 1,300 years. Developed through deep interdisciplinary collaboration, ProHist-Bench features 400 challenging, expert-curated questions across eight dynasties, accompanied by 10,891 fine-grained evaluation rubrics. Through a rigorous evaluation of 18 LLMs, we reveal a significant proficiency gap: even state-of-the-art LLMs struggle with complex historical research questions. We hope ProHist-Bench will facilitate the development of domain-specific reasoning LLMs, advance computational historical research, and further uncover the untapped potential of LLMs. We release ProHist-Bench at this https URL.
>
---
#### [new 069] Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型比较任务，旨在分析微调与上下文学习的差异。通过形式化语言实验，评估两者在语言掌握和归纳偏差上的表现。**

- **链接: [https://arxiv.org/pdf/2604.23267](https://arxiv.org/pdf/2604.23267)**

> **作者:** Bishwamittra Ghosh; Soumi Das; Till Speicher; Qinyuan Wu; Mohammad Aflah Khan; Deepak Garg; Krishna P. Gummadi; Evimaria Terzi
>
> **备注:** Accepted at ACL 2026 (Main)
>
> **摘要:** Large language models (LLMs) operate in two fundamental learning modes - fine-tuning (FT) and in-context learning (ICL) - raising key questions about which mode yields greater language proficiency and whether they differ in their inductive biases. Prior studies comparing FT and ICL have yielded mixed and inconclusive results due to inconsistent experimental setups. To enable a rigorous comparison, we propose a formal language learning task - offering precise language boundaries, controlled string sampling, and no data contamination - and introduce a discriminative test for language proficiency, where an LLM succeeds if it assigns higher generation probability to in-language strings than to out-of-language strings. Empirically, we find that: (a) FT has greater language proficiency than ICL on in-distribution generalization, but both perform equally well on out-of-distribution generalization. (b) Their inductive biases, measured by the correlation in string generation probabilities, are similar when both modes partially learn the language but diverge at higher proficiency levels. (c) Unlike FT, ICL performance differs substantially across models of varying sizes and families and is sensitive to the token vocabulary of the language. Thus, our work demonstrates the promise of formal languages as a controlled testbed for evaluating LLMs, behaviors that are difficult to isolate in natural language datasets. Our source code is available at this https URL.
>
---
#### [new 070] Looking for the Bottleneck in Fine-grained Temporal Relation Classification
- **分类: cs.CL**

- **简介: 该论文属于时间关系分类任务，旨在解决复杂时间实体间关系识别问题。通过分析端点关系并推导区间关系，提出新方法，在TempEval-3上取得最佳效果。**

- **链接: [https://arxiv.org/pdf/2604.24620](https://arxiv.org/pdf/2604.24620)**

> **作者:** Hugo Sousa; Ricardo Campos; Alípio Jorge
>
> **摘要:** Temporal relation classification is the task of determining the temporal relation between pairs of temporal entities in a text. Despite recent advancements in natural language processing, temporal relation classification remains a considerable challenge. Early attempts framed this task using a comprehensive set of temporal relations between events and temporal expressions. However, due to the task complexity, datasets have been progressively simplified, leading recent approaches to focus on the relations between event pairs and to use only a subset of relations. In this work, we revisit the broader goal of classifying interval relations between temporal entities by considering the full set of relations that can hold between two time intervals. The proposed approach, Interval from Point, involves first classifying the point relations between the endpoints of the temporal entities and then decoding these point relations into an interval relation. Evaluation on the TempEval-3 dataset shows that this approach can yield effective results, achieving a temporal awareness score of $70.1$ percent, a new state-of-the-art on this benchmark.
>
---
#### [new 071] ComplianceNLP: Knowledge-Graph-Augmented RAG for Multi-Framework Regulatory Gap Detection
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文提出ComplianceNLP系统，用于自动检测金融机构的合规缺口。任务是监管合规分析，解决手动处理大量法规信息效率低的问题。工作包括构建知识图谱、提取义务信息及分析合规差距。**

- **链接: [https://arxiv.org/pdf/2604.23585](https://arxiv.org/pdf/2604.23585)**

> **作者:** Dongxin Guo; Jikun Wu; Siu Ming Yiu
>
> **备注:** Accepted at ACL 2026 Industry Track. 19 pages, 15 tables, 1 figure
>
> **摘要:** Financial institutions must track over 60,000 regulatory events annually, overwhelming manual compliance teams; the industry has paid over USD 300 billion in fines and settlements since the 2008 financial crisis. We present ComplianceNLP, an end-to-end system that automatically monitors regulatory changes, extracts structured obligations, and identifies compliance gaps against institutional policies. The system integrates three components: (1) a knowledge-graph-augmented RAG pipeline grounding generations in a regulatory knowledge graph of 12,847 provisions across SEC, MiFID II, and Basel III; (2) multi-task obligation extraction combining NER, deontic classification, and cross-reference resolution over a shared LEGAL-BERT encoder; and (3) compliance gap analysis that maps obligations to internal policies with severity-aware scoring. On our benchmark, ComplianceNLP achieves 87.7 F1 on gap detection, outperforming GPT-4o+RAG by +3.5 F1, with 94.2% grounding accuracy ($r=0.83$ vs. human judgments) and 83.4 F1 under realistic end-to-end error propagation. Ablations show that knowledge-graph re-ranking contributes the largest marginal gain (+4.6 F1), confirming that structural regulatory knowledge is critical for cross-reference-heavy tasks. Domain-specific knowledge distillation (70B $\to$ 8B) combined with Medusa speculative decoding yields $2.8\times$ inference speedup; regulatory text's low entropy ($H=2.31$ bits vs. $3.87$ general text) produces 91.3% draft-token acceptance rates. In four months of parallel-run deployment processing 9,847 updates at a financial institution, the system achieved 96.0% estimated recall and 90.7% precision, with a $3.1\times$ sustained analyst efficiency gain. We report deployment lessons on trust calibration, GRC integration, and distributional shift monitoring for regulated-domain NLP.
>
---
#### [new 072] Agri-CPJ: A Training-Free Explainable Framework for Agricultural Pest Diagnosis Using Caption-Prompt-Judge and LLM-as-a-Judge
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于农业病虫害诊断任务，解决模型误判和解释性差的问题。提出Agri-CPJ框架，通过生成结构化描述并由LLM筛选，提升诊断准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2604.23701](https://arxiv.org/pdf/2604.23701)**

> **作者:** Wentao Zhang; Qi Zhang; Mingkun Xu; Mu You; Henghua Shen; Zhongzhi He; Keyan Jin; Derek F. Wong; Tao Fang
>
> **备注:** This work is an expanded version of our prior paper published in the IEEE ICASSP 2026 conference arXiv:2512.24947, from 4 to 20+ pages, presenting a well-structured and principled framework, extensive experiments, and deeper insights. Tao Fang is the corresponding author
>
> **摘要:** Crop disease diagnosis from field photographs faces two recurring problems: models that score well on benchmarks frequently hallucinate species names, and when predictions are correct, the reasoning behind them is typically inaccessible to the practitioner. This paper describes Agri-CPJ (Caption-Prompt-Judge), a training-free few-shot framework in which a large vision-language model first generates a structured morphological caption, iteratively refined through multi-dimensional quality gating, before any diagnostic question is answered. Two candidate responses are then generated from complementary viewpoints, and an LLM judge selects the stronger one based on domain-specific criteria. Caption refinement is the component with the largest individual impact: ablations confirm that skipping it consistently degrades downstream accuracy across both models tested. On CDDMBench, pairing GPT-5-Nano with GPT-5-mini-generated captions yields \textbf{+22.7} pp in disease classification and \textbf{+19.5} points in QA score over no-caption baselines. Evaluated without modification on AgMMU-MCQs, GPT-5-Nano reached 77.84\% and Qwen-VL-Chat reached 64.54\%, placing them at or above most open-source models of comparable scale despite the format shift from open-ended to multiple-choice. The structured caption and judge rationale together constitute a readable audit trail: a practitioner who disagrees with a diagnosis can identify the specific caption observation that was incorrect. Code and data are publicly available this https URL
>
---
#### [new 073] AutoPyVerifier: Learning Compact Executable Verifiers for Large Language Model Outputs
- **分类: cs.CL; cs.LG; cs.PL**

- **简介: 该论文提出AutoPyVerifier，解决LLM输出验证问题。通过合成和优化Python验证器，提升验证准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2604.22937](https://arxiv.org/pdf/2604.22937)**

> **作者:** Pouya Pezeshkpour; Estevam Hruschka
>
> **摘要:** Verification is becoming central to both reinforcement-learning-based training and inference-time control of large language models (LLMs). Yet current verifiers face a fundamental trade-off: LLM-based verifiers are expressive but hard to control and prone to error, while deterministic executable verifiers are reliable and interpretable but often limited in capability. We study the following question: given a development set of LLM outputs and labels for a target objective, such as correctness, can we automatically induce a minimal set of Python verifiers whose joint satisfaction closely matches that objective? We propose AutoPyVerifier, a framework that uses an LLM to synthesize candidate verifier functions and then refines them through search over a directed acyclic graph (DAG). By navigating the DAG, AutoPyVerifier systematically explores the space of deterministic executable verifiers and selects a compact verifier set whose joint satisfaction best approximates the target objective. Across mathematical reasoning, coding, function calling, and instruction-following benchmarks for several state-of-the-art LLMs, AutoPyVerifier improves target-objective prediction by up to 55.0 F1 points over the initial LLM-generated verifier sets. Additional analyses show that the most useful verification targets vary by benchmark and model, and that the DAG-based search shifts the learned verifier sets toward more structural and semantically grounded checks. We further show that exposing the discovered verifier set to an LLM as an external tool improves downstream accuracy by up to 17.0 points. We release our code
>
---
#### [new 074] Mechanistic Steering of LLMs Reveals Layer-wise Feature Vulnerabilities in Adversarial Settings
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全与防御任务，旨在解决LLM在对抗环境下产生有害输出的问题。通过分析模型内部特征，识别出中后层特征子组易被操控，提出针对性干预方法提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.23130](https://arxiv.org/pdf/2604.23130)**

> **作者:** Nilanjana Das; Manas Gaur
>
> **摘要:** Large language models (LLMs) can still be jailbroken into producing harmful outputs despite safety alignment. Existing attacks show this vulnerability, but not the internal mechanisms that cause it. This study asks whether jailbreak success is driven by identifiable internal features rather than prompts alone. We propose a three-stage pipeline for Gemma-2-2B using the BeaverTails dataset. First, we extract concept-aligned tokens from adversarial responses via subspace similarity. Second, we apply three feature-grouping strategies (cluster, hierarchical-linkage, and single-token-driven) to identify SAE feature subgroups for the aligned tokens across all 26 model layers. Third, we steer the model by amplifying the top features from each identified subgroup and measure the change in harmfulness score using a standardized LLM-judge scoring protocol. In all three approaches, the features in the layers [16-25] were relatively more vulnerable to steering. All three methods confirmed that mid to later layer feature subgroups are more responsible for unsafe outputs. These results provide evidence that the jailbreak vulnerability in Gemma-2-2B is localized to feature subgroups of mid to later layers, suggesting that targeted feature-level interventions may offer a more principled path to adversarial robustness than current prompt-level defenses.
>
---
#### [new 075] Evaluation of Pose Estimation Systems for Sign Language Translation
- **分类: cs.CL**

- **简介: 该论文属于手势语言翻译任务，研究不同姿态估计器对翻译效果的影响，通过实验比较多种模型性能，旨在提升SLT系统的准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.24609](https://arxiv.org/pdf/2604.24609)**

> **作者:** Catherine O'Brien; Gerard Sant; Mathias Müller; Sarah Ebling
>
> **备注:** Accepted at LREC 2026 Workshop on the Representation and Processing of Sign Languages. O'Brien and Sant contributed equally to this paper. 16 pages, 6 figures
>
> **摘要:** Many sign language translation (SLT) systems operate on pose sequences instead of raw video to reduce input dimensionality, improve portability, and partially anonymize signers. The choice of pose estimator is often treated as an implementation detail, with systems defaulting to widely available tools such as MediaPipe Holistic or OpenPose. We present a systematic comparison of pose estimators for pose-based SLT, covering widely used baselines (MediaPipe Holistic, OpenPose) and newer whole-body/high-capacity models (MMPose WholeBody, OpenPifPaf, AlphaPose, SDPose, Sapiens, SMPLest-X). We quantify downstream impact by training a controlled SLT pipeline on RWTH-PHOENIX-Weather 2014 where only the pose representation varies, evaluating with BLEU and BLEURT. To contextualize translation outcomes, we analyze temporal stability, missing hand keypoints, and robustness to occlusion using higher-resolution videos from the Signsuisse dataset. SDPose and Sapiens achieve the best translation performance (BLEU ~11.5), outperforming the common MediaPipe baseline (BLEU ~10). In occlusion cases, Sapiens is correct in all tested instances (15/15), while OpenPifPaf fails in nearly all (1/15) and also yields the weakest translation scores. Estimators that frequently leave out hand keypoints are associated with lower BLEU/BLEURT. We release code that can be used not only to reproduce our experiments, but also considerably lowers the barrier for other researchers to use alternative pose estimators.
>
---
#### [new 076] Translate or Simplify First: An Analysis of Cross-lingual Text Simplification in English and French
- **分类: cs.CL**

- **简介: 该论文研究跨语言文本简化任务，探讨不同提示策略在英法文本简化中的效果，比较直接、组合与分解方法，以提升内容可访问性。**

- **链接: [https://arxiv.org/pdf/2604.23844](https://arxiv.org/pdf/2604.23844)**

> **作者:** Ido Dahan; Omer Toledano; Roey J. Gafter; Sharon Pardo; Oren Tsur; Hila Zahavi; Elior Sulem
>
> **摘要:** Cross-Lingual Text Simplification (CLTS) aims to make content more accessible across languages by simultaneously addressing both linguistic complexity and translation. This study investigates the effectiveness of different prompting strategies for CLTS between English and French using large language models (LLMs). We examine five distinct prompting systems: a direct prompt instructing the LLM to perform both translation and simplification simultaneously, two Composition approaches that either translate-then-simplify or simplify-then-translate within a single prompt, and two decomposition approaches that perform the same operations in separate, consecutive prompts. These systems are evaluated across a diverse set of five corpora of different genres (Wikipedia and medical texts) using seven state-of-the-art LLMs. Output quality is assessed through a multi-faceted evaluation framework comprising automatic metrics, comprehensive linguistic feature analysis, and human evaluation of simplicity and meaning preservation. Our findings reveal that while direct prompting consistently achieves the highest BLEU scores, indicating meaning fidelity, Translate-then-Simplify approaches demonstrate the highest simplicity, as measured by the linguistic features.
>
---
#### [new 077] Measuring Temporal Linguistic Emergence in Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文研究扩散语言模型生成过程中的语言信息涌现时间，通过分析不同阶段的恢复能力和敏感性，探讨信息何时可测量。属于语言模型分析任务，解决生成过程中信息涌现时机的问题。**

- **链接: [https://arxiv.org/pdf/2604.23235](https://arxiv.org/pdf/2604.23235)**

> **作者:** Harry Lu
>
> **摘要:** Diffusion language models expose an explicit denoising trajectory, making it possible to ask when different kinds of information become measurable during generation. We study three independent 32-step runs of LLaDA-8B-Base on masked WikiText-103 text, each with 1{,}000 probe-training sequences and 200 held-out evaluation sequences. From saved trajectories, we derive four temporal measurements: token commitment; linear recoverability of part-of-speech (POS), coarse semantic category, and token identity; confidence and entropy dynamics; and sensitivity under mid-trajectory re-masking. Across seeds, the same ordering recurs: content categories stabilize earlier than function-heavy categories, POS and coarse semantic labels remain substantially more linearly recoverable than exact lexical identity under our probe setup, uncertainty remains higher for tokens that ultimately resolve incorrectly even though late confidence becomes less calibrated, and perturbation sensitivity peaks in the middle of the trajectory. A direct/collateral decomposition shows that this peak is overwhelmingly local to the perturbed positions themselves. In this LLaDA+WikiText setting, denoising time is therefore a useful analysis axis: under our measurements, coarse labels are recovered earlier and more robustly than lexical identity, trajectory-level uncertainty tracks eventual correctness, and mid-trajectory states are the most intervention-sensitive.
>
---
#### [new 078] OS-SPEAR: A Toolkit for the Safety, Performance,Efficiency, and Robustness Analysis of OS Agents
- **分类: cs.CL**

- **简介: 该论文提出OS-SPEAR工具包，用于评估操作系统代理的安全性、性能、效率和鲁棒性，解决现有基准不足的问题。**

- **链接: [https://arxiv.org/pdf/2604.24348](https://arxiv.org/pdf/2604.24348)**

> **作者:** Zheng Wu; Yi Hua; Zhaoyuan Huang; Chenhao Xue; Yijie Lu; Pengzhou Cheng; Zongru Wu; Lingzhong Dong; Gongshen Liu; Xinghao Jiang; Zhuosheng Zhang
>
> **摘要:** The evolution of Multimodal Large Language Models (MLLMs) has shifted the focus from text generation to active behavioral execution, particularly via OS agents navigating complex GUIs. However, the transition of these agents into trustworthy daily partners is hindered by a lack of rigorous evaluation regarding safety, efficiency, and multi-modal robustness. Current benchmarks suffer from narrow safety scenarios, noisy trajectory labeling, and limited robustness metrics. To bridge this gap, we propose OS-SPEAR, a comprehensive toolkit for the systematic analysis of OS agents across four dimensions: Safety, Performance, Efficiency, and Robustness. OS-SPEAR introduces four specialized subsets: (1) a S(afety)-subset encompassing diverse environment- and human-induced hazards; (2) a P(erformance)-subset curated via trajectory value estimation and stratified sampling; (3) an E(fficiency)-subset quantifying performance through the dual lenses of temporal latency and token consumption; and (4) a R(obustness)-subset that applies cross-modal disturbances to both visual and textual inputs. Additionally, we provide an automated analysis tool to generate human-readable diagnostic reports. We conduct an extensive evaluation of 22 popular OS agents using OS-SPEAR. Our empirical results reveal critical insights into the current landscape: notably, a prevalent trade-off between efficiency and safety or robustness, the performance superiority of specialized agents over general-purpose models, and varying robustness vulnerabilities across different modalities. By providing a multidimensional ranking and a standardized evaluation framework, OS-SPEAR offers a foundational resource for developing the next generation of reliable and efficient OS agents. The dataset and codes are available at this https URL.
>
---
#### [new 079] A Multi-Dimensional Audit of Politically Aligned Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型审计任务，旨在解决政治对齐语言模型的潜在风险问题。通过多维度评估框架，分析模型在效果、公平性、真实性与说服力方面的表现。**

- **链接: [https://arxiv.org/pdf/2604.24429](https://arxiv.org/pdf/2604.24429)**

> **作者:** Lisa Korver; Mohamed Mostagir; Sherief Reda
>
> **摘要:** As the application of Large Language Models (LLMs) spreads across various industries, there are increasing concerns about the potential for their misuse, especially in sensitive areas such as political discourse. Deliberately aligning LLMs with specific political ideologies, through prompt engineering or fine-tuning techniques, can be advantageous in use cases such as political campaigns, but requires careful consideration due to heightened risks of performance degradation, misinformation, or increased biased behavior. In this work, we propose a multi-dimensional framework inspired by Habermas' Theory of Communicative Action to audit politically aligned language models across four dimensions: effectiveness, fairness, truthfulness, and persuasiveness using automated, quantitative metrics. Applying this to nine popular LLMs aligned via fine-tuning or role-playing revealed consistent trade-offs: while larger models tend to be more effective at role-playing political ideologies and truthful in their responses, they were also less fair, exhibiting higher levels of bias in the form of angry and toxic language towards people of different ideologies. Fine-tuned models exhibited lower bias and more effective alignment than the corresponding role-playing models, but also saw a decline in performance reasoning tasks and an increase in hallucinations. Overall, all of the models tested exhibited some deficiency in at least one of the four metrics, highlighting the need for more balanced and robust alignment strategies. Ultimately, this work aims to ensure politically-aligned LLMs generate legitimate, harmless arguments, offering a framework to evaluate the responsible political alignment of these models.
>
---
#### [new 080] One Size Fits None: Heuristic Collapse in LLM Investment Advice
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于AI伦理与可信性任务，研究LLM在投资建议中是否出现启发式崩溃，即简化复杂决策。通过分析发现LLM主要依赖风险承受能力，忽视其他因素，表明需审计输入敏感性。**

- **链接: [https://arxiv.org/pdf/2604.23837](https://arxiv.org/pdf/2604.23837)**

> **作者:** Jillian Ross; Andrew W. Lo
>
> **摘要:** Large language models are increasingly deployed as advisors in high-stakes domains -- answering medical questions, interpreting legal documents, recommending financial products -- where good advice requires integrating a user's full context rather than responding to salient surface features. We investigate whether frontier LLMs actually do this, or whether they instead exhibit heuristic collapse: a systematic reduction of complex, multi-factor decisions to a small number of dominant inputs. We study the phenomenon in investment advice, where legal standards explicitly require individualized reasoning over a client's full circumstances. Applying interpretable surrogate models to LLM outputs, we find systematic heuristic collapse: investment allocation decisions are largely determined by self-reported risk tolerance, while other relevant factors contribute minimally. We further find that web search partially attenuates heuristic collapse but does not resolve it. These findings suggest that heuristic collapse is not resolved by web search augmentation or model scale alone, and that deploying LLMs as advisors requires auditing input sensitivity, not just output quality.
>
---
#### [new 081] Bridging Reasoning and Action: Hybrid LLM-RL Framework for Efficient Cross-Domain Task-Oriented Dialogue
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于任务导向型对话系统，解决跨领域长序列任务中的约束推理与行动问题。提出VLK-RL框架，结合LLM与RL，提升任务泛化与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.23345](https://arxiv.org/pdf/2604.23345)**

> **作者:** Yangyang Zhao; Linfan Dai; Li Cai; Bowen Xing; Libo Qin
>
> **摘要:** Cross-domain task-oriented dialogue requires reasoning over implicit and explicit feasibility constraints while planning long-horizon, multi-turn actions. Large language models (LLMs) can infer such constraints but are unreliable over long horizons, while Reinforcement learning (RL) optimizes long-horizon behavior yet cannot recover constraints from raw dialogue. Naively coupling LLMs with RL is therefore brittle: unverified or unstructured LLM outputs can corrupt state representations and misguide policy learning. Motivated by this, we propose Verified LLM-Knowledge empowered RL (VLK-RL), a hybrid framework that makes LLM-derived constraint reasoning usable for RL. VLK-RL first elicits candidate constraints with an LLM and then verifies them via a dual-role cross-examination procedure to suppress hallucinations and cross-turn inconsistencies. The verified constraints are mapped into ontology-aligned slot-value representations, yielding a structured, constraint-aware state for RL policy optimization. Experiments across multiple benchmarks demonstrate that VLK-RL significantly improves generalization and robustness, outperforming strong single-model baselines on long-horizon tasks.
>
---
#### [new 082] LLMs Reading the Rhythms of Daily Life: Aligned Understanding for Behavior Prediction and Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于行为预测与生成任务，旨在解决长尾行为处理、可解释性及多任务支持问题。提出BUA框架，通过结构化课程学习将LLMs应用于人类行为建模。**

- **链接: [https://arxiv.org/pdf/2604.23578](https://arxiv.org/pdf/2604.23578)**

> **作者:** Fanjin Meng; Jingtao Ding; Nian Li; Yizhou Sun; Yong Li
>
> **摘要:** Human daily behavior unfolds as complex sequences shaped by intentions, preferences, and context. Effectively modeling these behaviors is crucial for intelligent systems such as personal assistants and recommendation engines. While recent advances in deep learning and behavior pre-training have improved behavior prediction, key challenges remain--particularly in handling long-tail behaviors, enhancing interpretability, and supporting multiple tasks within a unified framework. Large language models (LLMs) offer a promising direction due to their semantic richness, strong interpretability, and generative capabilities. However, the structural and modal differences between behavioral data and natural language limit the direct applicability of LLMs. To address this gap, we propose Behavior Understanding Alignment (BUA), a novel framework that integrates LLMs into human behavior modeling through a structured curriculum learning process. BUA employs sequence embeddings from pretrained behavior models as alignment anchors and guides the LLM through a three-stage curriculum, while a multi-round dialogue setting introduces prediction and generation capabilities. Experiments on two real-world datasets demonstrate that BUA significantly outperforms existing methods in both tasks, highlighting its effectiveness and flexibility in applying LLMs to complex human behavior modeling.
>
---
#### [new 083] Chinese-SkillSpan: A Span-Level Dataset for ESCO-Aligned Competency Extraction from Chinese Job Ads
- **分类: cs.CL**

- **简介: 该论文提出中文技能实体识别任务，解决招聘文本中技能信息提取问题。构建了首个符合ESCO标准的中文数据集Chinese-SkillSpan，通过协同标注流程完成2万余条数据标注。**

- **链接: [https://arxiv.org/pdf/2604.23009](https://arxiv.org/pdf/2604.23009)**

> **作者:** Guojing Li; Zichuan Fu; Junyi Li; Wenxia Zhou; Xinyang Wu; Jinning Yang; Jingtong Gao; Feng Huang; Xiangyu Zhao
>
> **备注:** 18 pages, 10 figures, 3 tables
>
> **摘要:** Job Skill Named Entity Recognition (JobSkillNER) aims to automatically extract key skill information from large-scale job posting data, which is important for improving talent-market matching efficiency and supporting personalized employment services. To the best of our knowledge, this work presents the first Chinese JobSkillNER dataset for recruitment texts. We propose annotation guidelines tailored to Chinese job postings and an LLM-empowered Macro-Micro collaborative annotation pipeline. The pipeline leverages the contextual understanding ability of large language models (LLMs) for initial annotation and then refines the results through expert sentence-level adjudication. Using this pipeline, we annotate more than 20,000 instances collected from four major recruitment platforms over the period 2014-2025. Based on these efforts, we release Chinese-SkillSpan, the first Chinese JobSkillNER dataset aligned with the ESCO occupational skill standard across four dimensions: knowledge, skill, transversal competence, and language competence (LSKT). Experimental results show that the dataset supports effective model training and evaluation, indicating that Chinese-SkillSpan helps fill a major gap in Chinese JobSkillNER resources and provides a useful benchmark for intelligent recruitment research. Code and data are available at this https URL .
>
---
#### [new 084] XITE: Cross-lingual Interpolation for Transfer using Embeddings
- **分类: cs.CL**

- **简介: 该论文提出XITE方法，解决多语言模型中的跨语言迁移问题。通过嵌入式数据增强，提升低资源语言任务性能，如情感分析和自然语言推理。**

- **链接: [https://arxiv.org/pdf/2604.23589](https://arxiv.org/pdf/2604.23589)**

> **作者:** Barah Fazili; Preethi Jyothi
>
> **摘要:** Facilitating cross-lingual transfer in multilingual language models remains a critical challenge. Towards this goal, we propose an embedding-based data augmentation technique called XITE. We start with unlabeled text from a low-resource target language, identify an English counterpart in a task-specific training corpus using embedding-based similarities and adopt its label. Next, we perform a simple interpolation of the source and target embeddings to create synthetic data for task-specific fine-tuning. Projecting the target text into a language-rich subspace using linear discriminant analysis (LDA), prior to interpolation, further boosts performance. Our cross-lingual embedding-based augmentation technique XITE yields significant improvements of up to 35.91% for sentiment analysis and up to 81.16% for natural language inference, using XLM-R, for a diverse set of target languages including Korean, Arabic, Urdu and Hindi. Apart from boosting cross-lingual transfer, adaptation using XITE also safeguards against forgetting and maintains task performance on the high-resource language.
>
---
#### [new 085] Skill Retrieval Augmentation for Agentic AI
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于智能代理任务，旨在解决技能调用效率低的问题。提出SRA方法，通过动态检索外部技能提升代理性能，并构建了SRA-Bench进行评估。**

- **链接: [https://arxiv.org/pdf/2604.24594](https://arxiv.org/pdf/2604.24594)**

> **作者:** Weihang Su; Jianming Long; Qingyao Ai; Yichen Tang; Changyue Wang; Yiteng Tu; Yiqun Liu
>
> **摘要:** As large language models (LLMs) evolve into agentic problem solvers, they increasingly rely on external, reusable skills to handle tasks beyond their native parametric capabilities. In existing agent systems, the dominant strategy for incorporating skills is to explicitly enumerate available skills within the context window. However, this strategy fails to scale: as skill corpora expand, context budgets are consumed rapidly, and the agent becomes markedly less accurate in identifying the right skill. To this end, this paper formulates Skill Retrieval Augmentation (SRA), a new paradigm in which agents dynamically retrieve, incorporate, and apply relevant skills from large external skill corpora on demand. To make this problem measurable, we construct a large-scale skill corpus and introduce SRA-Bench, the first benchmark for decomposed evaluation of the full SRA pipeline, covering skill retrieval, skill incorporation, and end-task execution. SRA-Bench contains 5,400 capability-intensive test instances and 636 manually constructed gold skills, which are mixed with web-collected distractor skills to form a large-scale corpus of 26,262 skills. Extensive experiments show that retrieval-based skill augmentation can substantially improve agent performance, validating the promise of the paradigm. At the same time, we uncover a fundamental gap in skill incorporation: current LLM agents tend to load skills at similar rates, regardless of whether a gold skill is retrieved or whether the task actually requires external capabilities. This shows that the bottleneck in skill augmentation lies not only in retrieval but also in the base model's ability to determine which skill to load and when external loading is actually needed. These findings position SRA as a distinct research problem and establish a foundation for the scalable augmentation of capabilities in future agent systems.
>
---
#### [new 086] Zero-shot Large Language Models for Automatic Readability Assessment
- **分类: cs.CL**

- **简介: 该论文属于自动可读性评估任务，旨在解决无监督评估文本可读性的难题。通过提出零样本提示方法和LAURAE模型，提升评估效果。**

- **链接: [https://arxiv.org/pdf/2604.24470](https://arxiv.org/pdf/2604.24470)**

> **作者:** Riley Grossman; Yi Chen
>
> **备注:** Accepted to ACL 2026 (Main Conference)
>
> **摘要:** Unsupervised automatic readability assessment (ARA) methods have important practical and research applications (e.g., ensuring medical or educational materials are suitable for their target audiences). In this paper, we propose a new zero-shot prompting methodology for ARA and present the first comprehensive evaluation of using large language models (LLMs) as an unsupervised ARA method by testing 10 diverse open-source LLMs (e.g., different sizes and developers) on 14 diverse datasets (e.g., different text lengths and languages). Our findings show that our proposed prompting methodology outperforms prior methods on 13 of the 14 datasets. Furthermore, we propose LAURAE, which combines LLM and readability formula scores to improve robustness by capturing both contextual and shallow (e.g., sentence length) features of readability. Our evaluation demonstrates that LAURAE robustly outperforms prior methods across languages, text lengths, and amounts of technical language.
>
---
#### [new 087] Self Knowledge Re-expression: A Fully Local Method for Adapting LLMs to Tasks Using Intrinsic Knowledge
- **分类: cs.CL; cs.AI; cs.CV; cs.IR**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在非生成任务中的性能瓶颈。提出SKR方法，通过本地数据提升模型任务表达效率。**

- **链接: [https://arxiv.org/pdf/2604.22939](https://arxiv.org/pdf/2604.22939)**

> **作者:** Mengyu Wang; Xiaoying Zhi; Zhiyi Li; Robin Schmucker; Shay B. Cohen; Tiejun Ma; Fran Silavong
>
> **摘要:** While the next-token prediction (NTP) paradigm enables large language models (LLMs) to express their intrinsic knowledge, its sequential nature constrains performance on specialized, non-generative tasks. We attribute this performance bottleneck to the LLMs' knowledge expression mechanism, rather than to deficiencies in knowledge acquisition. To address this, we propose Self-Knowledge Re-expression (SKR), a novel, task-agnostic adaptation method. SKR transforms the LLM's output from generic token generation to highly efficient, task-specific expression. SKR is a fully local method that uses only unannotated data, requiring neither human supervision nor model distillation. Experiments on a large financial document dataset demonstrate substantial improvements: over 40% in Recall@1 for information retrieval tasks, over 76% reduction in object detection latency, and over 33% increase in anomaly detection AUPRC. Our results on the MMDocRAG dataset surpass those of leading retrieval models by at least 12.6%.
>
---
#### [new 088] Psychologically-Grounded Graph Modeling for Interpretable Depression Detection
- **分类: cs.CL**

- **简介: 该论文属于抑郁症检测任务，解决数据稀缺和缺乏临床解释性问题。提出PsyGAT模型，通过心理图结构建模对话，提升检测效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.24126](https://arxiv.org/pdf/2604.24126)**

> **作者:** Rishitej Reddy Vyalla; Kritarth Prasad; Avinash Anand; Erik Cambria; Shaoxiong Ji; Faten S. Alamri; Zhengkui Wang
>
> **摘要:** Automatic depression detection from conversational interactions holds significant promise for scalable screening but remains hindered by severe data scarcity and a lack of clinical interpretability. Existing approaches typically rely on black-box deep learning architectures that struggle to model the subtle, temporal evolution of depressive symptoms or account for participant-specific heterogeneity. In this work, we propose PsyGAT (Psychological Graph Attention Network), a psychologically grounded framework that models conversational sessions as dynamic temporal graphs. We introduce Psychological Expression Units (PEUs) to explicitly encode utterance-level clinical evidence, structuring the session graph to capture transitions in psychological states rather than mere semantic dependencies. To address the critical class imbalance in depression datasets, we employ clinically approved persona-based data augmentation, enable robust model learning. Additionally, we integrate session-level personality context directly into the graph structure to disentangle trait-based behavior from acute depressive symptoms. PsyGAT achieves state-of-the-art performance, surpassing both strong graph-based baselines and closed-source LLMs like GPT-5, achieving 89.99 and 71.37 Macro F1 scores in DAIC-WoZ and E-DAIC, respectively. We further introduce Causal-PsyGAT, an interpretability module that identifies symptom triggers. Experiments show a 20% improvement in MRR for identifying causal indicators, effectively bridging the gap between depression monitoring and clinical explainability. The full augmented dataset is publicly available at this https URL.
>
---
#### [new 089] Distilling Self-Consistency into Verbal Confidence: A Pre-Registered Negative Result and Post-Hoc Rescue on Gemma 3 4B
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究小指令调优大模型的自信度问题，通过CSFT方法尝试提升模型自信度与实际表现的一致性。工作包括预注册实验和后续调整，发现标签熵对训练重要，正确目标能规范输出格式。**

- **链接: [https://arxiv.org/pdf/2604.24070](https://arxiv.org/pdf/2604.24070)**

> **作者:** Jon-Paul Cacioli
>
> **备注:** 12 pages, 3 figures, 4 tables. Pre-registered on OSF (this https URL). Code and data: this https URL
>
> **摘要:** Small instruct-tuned LLMs produce degenerate verbal confidence under minimal elicitation: ceiling rates above 95%, near-chance Type-2 AUROC, and Invalid validity profiles. We test whether confidence-conditioned supervised fine-tuning (CSFT) with self-consistency-derived targets can close the gap between internal information and verbal readout. A pre-registered Phase 0 protocol on Gemma 3 4B-it with a modal filter restricting training to items with correct modal answers produced a negative result: AUROC2 dropped from 0.554 to 0.509 due to label-entropy collapse in the training targets. An exploratory rescue removed the filter, training on all 2,000 calibration items. This produced a binary verbal correctness discriminator with AUROC2 = 0.774 on held-out TriviaQA, compressing a 10-sample self-consistency signal (AUROC2 = 0.999) into a single-pass readout exceeding logit entropy (0.701). The shuffled-target control showed no improvement (0.501). On MMLU, accuracy improved from 54.2% to 77.4% with the shuffled model at baseline (56.1%), supporting a target-dependent interpretation. The result is exploratory, binary rather than continuously calibrated, and observed at a single scale. It identifies two design lessons: confidence training requires label entropy, and correct targets regularise output format.
>
---
#### [new 090] Evaluating Large Language Models on Computer Science University Exams in Data Structures
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型在计算机科学数据结构考试中的表现。通过构建基准数据集，测试多个模型的答题能力。**

- **链接: [https://arxiv.org/pdf/2604.23347](https://arxiv.org/pdf/2604.23347)**

> **作者:** Edan Gabay; Yael Maoz; Jonathan Stahl; Naama Maoz; Abdo Amer; Orr Eilat; Hanoch Levy; Michal Kleinbort; Amir Rubinstein; Adi Haviv
>
> **摘要:** We present a comprehensive evaluation of Large Language Models (LLMs) on Computer Science (CS) Data Structure examination questions. Our work introduces a new benchmark dataset comprising exam questions from Tel Aviv University (TAU), curated to assess LLMs' abilities in handling closed and multiple-choice questions. We evaluated the performance of OpenAI's GPT 4o and Anthropic's Claude 3.5, popular LLMs, alongside two smaller LLMs, Mathstral 7B and LLaMA 3 8B, across the TAU exams benchmark. Our findings provide insight into the current capabilities of LLMs in CS education.
>
---
#### [new 091] Benchmarking Source-Sensitive Reasoning in Turkish: Humans and LLMs under Evidential Trust Manipulation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究土耳其语中证据形态受来源可信度影响的机制，以及大语言模型是否能捕捉这种敏感性。任务属于自然语言处理中的语义推理与语言模型评估，旨在探讨人类与LLMs在源敏感性推理上的差异。**

- **链接: [https://arxiv.org/pdf/2604.24665](https://arxiv.org/pdf/2604.24665)**

> **作者:** Sercan Karakaş; Yusuf Şimşek
>
> **备注:** Accepted to The 15th edition of the Workshop on Cognitive Modeling and Computational Linguistics, co-located with the Language Resources and Evaluation Conference
>
> **摘要:** This paper investigates whether source trustworthiness shapes Turkish evidential morphology and whether large language models (LLMs) track this sensitivity. We study the past-domain contrast between -DI and -mIs in controlled cloze contexts where the information source is overtly external, while only its perceived reliability is manipulated (High-Trust vs. Low-Trust). In a human production experiment, native speakers of Turkish show a robust trust effect: High-Trust contexts yield relatively more -DI, whereas Low-Trust contexts yield relatively more -mIs, with the pattern remaining stable across sensitivity analyses. We then evaluate 10 LLMs in three prompting paradigms (open gap-fill, explicit past-tense gap-fill, and forced-choice A/B selection). LLM behavior is highly model- and prompt-dependent: some models show weak or local trust-consistent shifts, but effects are generally unstable, often reversed, and frequently overshadowed by output-compliance problems and strong base-rate suffix preferences. The results provide new evidence for a trust-/commitment-based account of Turkish evidentiality and reveal a clear human-LLM gap in source-sensitive evidential reasoning.
>
---
#### [new 092] Reducing Redundancy in Retrieval-Augmented Generation through Chunk Filtering
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决RAG中因分块产生的冗余问题。通过实体过滤等策略减少索引规模，提升检索效率。**

- **链接: [https://arxiv.org/pdf/2604.24334](https://arxiv.org/pdf/2604.24334)**

> **作者:** Daria Berdyugina; Anaëlle Cohen; Yohann Rioual
>
> **摘要:** Standard Retrieval-Augmented Generation (RAG) chunking methods often create excessive redundancy, increasing storage costs and slowing retrieval. This study explores chunk filtering strategies, such as semantic, topic-based, and named-entity-based methods in order to reduce the indexed corpus while preserving retrieval quality. Experiments are conducted on multiple corpora. Retrieval performance is evaluated using a token-based framework based on precision, recall, and intersection-over-union metrics. Results indicate that entity-based filtering can reduce vector index size by approximately 25% to 36% while maintaining high retrieval quality close to the baseline. These findings suggest that redundancy introduced during chunking can be effectively reduced through lightweight filtering, improving the efficiency of retrieval-oriented components in RAG pipelines.
>
---
#### [new 093] Multimodal QUD: Inquisitive Questions from Scientific Figures
- **分类: cs.CL**

- **简介: 该论文属于多模态问答任务，旨在解决科学论文中基于图文生成深度问题的问题。通过构建MQUD数据集，提升模型的多模态推理能力。**

- **链接: [https://arxiv.org/pdf/2604.23733](https://arxiv.org/pdf/2604.23733)**

> **作者:** Yating Wu; William Rudman; Venkata S Govindarajan; Alexandros G. Dimakis; Junyi Jessy Li
>
> **摘要:** Asking inquisitive questions while reading, and looking for their answers, is an important part in human discourse comprehension, curiosity, and creative ideation, and prior work has investigated this in text-only scenarios. However, in scientific or research papers, many of the critical takeaways are conveyed through both figures and the text that analyzes them. While scientific visualizations have been used to evaluate Vision-Language Models (VLMs) capabilities, current benchmarks are limited to questions that focus simply on extracting information from them. Such questions only require lower-level reasoning, do not take into account the context in which a figure appears, and do not reflect the communicative goals the authors wish to achieve. We generate inquisitive questions that reach the depth of questions humans generate when engaging with scientific papers, conditioned on both the figure and the paper's context, and require reasoning across both modalities. To do so, we extend the linguistic theory of Questions Under Discussion (QUD) from being text-only to multimodal, where implicit questions are raised and resolved as discourse progresses. We present MQUD, a dataset of research papers in which such questions are made explicit and annotated by the original authors. We show that fine-tuning a VLM on MQUD shifts the model from generating generic low-level visual questions to content-specific grounding that requires a high-level of multimodal reasoning, yielding higher-quality, more visually grounded multimodal QUD generation.
>
---
#### [new 094] Evaluating Temporal Consistency in Multi-Turn Language Models
- **分类: cs.CL**

- **简介: 该论文研究多轮对话中语言模型的时序一致性问题，旨在解决模型在连续交互中保持时间上下文稳定性的挑战。作者构建了ChronoScope基准，评估模型在不同时间场景下的表现。**

- **链接: [https://arxiv.org/pdf/2604.23051](https://arxiv.org/pdf/2604.23051)**

> **作者:** Yash Kumar Atri; Steven L. Johnson; Tom Hartvigsen
>
> **备注:** Accepted at ACL 2026
>
> **摘要:** Language models are increasingly deployed in interactive settings where users reason about facts over time rather than in isolation. In such scenarios, correct behavior requires models to maintain and update implicit temporal assumptions established earlier in a conversation. We study this challenge through the lens of temporal scope stability: the ability to preserve, override, or transfer time-scoped factual context across dialogue turns. We introduce ChronoScope, a large-scale diagnostic benchmark designed to isolate temporal scope behavior in controlled multi-turn interactions, comprising over one million deterministically generated question chains grounded in Wikidata. ChronoScope evaluates whether models can correctly retain inferred temporal scope when follow-up questions omit explicit time references, spanning implicit carryover, explicit scope switching, cross-entity transfer, and longer temporal trajectories. Through extensive evaluation of state-of-the-art language models, we find that temporal scope stability is frequently violated in controlled multi-turn settings, with models often drifting toward present-day assumptions despite correct underlying knowledge. These failures intensify with interaction length and persist even under oracle context conditions, revealing a gap between single-turn factual accuracy and coherent temporal reasoning under sequential interaction. We make our dataset and evaluation suite publicly available at this https URL
>
---
#### [new 095] Propagation Structure-Semantic Transfer Learning for Robust Fake News Detection
- **分类: cs.CL**

- **简介: 该论文属于虚假新闻检测任务，旨在解决语义和结构噪声干扰问题。提出PSS-TL框架，通过双教师模型和知识蒸馏提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.23974](https://arxiv.org/pdf/2604.23974)**

> **作者:** Mengyang Chen; Lingwei Wei; Han Cao; Wei Zhou; Zhou Yan; Songlin Hu
>
> **备注:** Accepted by ECML-PKDD 2024
>
> **摘要:** Fake news generally refers to false information that is spread deliberately to deceive people, which has detrimental social effects. Existing fake news detection methods primarily learn the semantic features from news content or integrate structural features from propagation. However, in practical scenarios, due to the semantic ambiguity of informal language and unreliable user interactive behaviors on social media, there are inherent semantic and structural noises in news content and propagation. Although some recent works consider the effect of irrelevant user interactions in a hybrid-modeling way, they still suffer from the mutual interference between structural noise and semantic noise, leading to limited performance for robust detection. To alleviate this issue, this paper proposes a novel Propagation Structure-Semantic Transfer Learning framework (PSS-TL) for robust fake news detection under a teacher-student architecture. Specifically, we design dual teacher models to learn semantics knowledge and structure knowledge from noisy news content and propagation structure independently. Besides, we design a Multi-channel Knowledge Distillation (MKD) loss to enable the student model to acquire specialized knowledge from the teacher models, thereby avoiding mutual interference. Extensive experiments on two real-world datasets validate the effectiveness and robustness of our method.
>
---
#### [new 096] K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出K-MetBench，用于评估气象领域专家推理、局部性和多模态能力。解决缺乏专业评估框架的问题，通过多维分析揭示模型缺陷，推动文化敏感的AI发展。**

- **链接: [https://arxiv.org/pdf/2604.24645](https://arxiv.org/pdf/2604.24645)**

> **作者:** Soyeon Kim; Cheongwoong Kang; Myeongjin Lee; Eun-Chul Chang; Jaedeok Lee; Jaesik Choi
>
> **备注:** 39 pages, 32 figures, 14 tables, including appendices. Accepted to Findings of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** The development of practical (multimodal) large language model assistants for Korean weather forecasters is hindered by the absence of a multidimensional, expert-level evaluation framework grounded in authoritative sources. To address this, we introduce K-MetBench, a diagnostic benchmark grounded in national qualification exams. It exposes critical gaps across four dimensions: expert visual reasoning of charts, logical validity via expert-verified rationales, Korean-specific geo-cultural comprehension, and fine-grained domain analysis. Our evaluation of 55 models reveals a profound modality gap in interpreting specialized diagrams and a reasoning gap where models hallucinate logic despite correct predictions. Crucially, Korean models outperform significantly larger global models in local contexts, demonstrating that parameter scaling alone cannot resolve cultural dependencies. K-MetBench serves as a roadmap for developing reliable, culturally aware expert AI agents. The dataset is available at this https URL .
>
---
#### [new 097] Learning Evidence of Depression Symptoms via Prompt Induction
- **分类: cs.CL**

- **简介: 该论文属于抑郁症症状识别任务，旨在从非临床文本中自动检测抑郁症状。针对数据不平衡问题，提出Symptom Induction方法，提升罕见症状识别效果。**

- **链接: [https://arxiv.org/pdf/2604.24376](https://arxiv.org/pdf/2604.24376)**

> **作者:** Eliseo Bao; Anxo Perez; David Otero; Javier Parapar
>
> **备注:** Accepted at SIGIR 2026
>
> **摘要:** Depression places substantial pressure on mental health services, and many people describe their experiences outside clinical settings in high-volume user-generated text (e.g., online forums and social media). Automatically identifying clinical symptom evidence in such text can therefore complement limited clinical capacity and scale to large populations. We address this need through sentence-level classification of 21 depression symptoms from the BDI-II questionnaire, using BDI-Sen, a dataset annotated for symptom relevance. This task is fine-grained and highly imbalanced, and we find that common LLM approaches (zero-shot, in-context learning, and fine-tuning) struggle to apply consistent relevance criteria for most symptoms. We propose Symptom Induction (SI), a novel approach which compresses labeled examples into short, interpretable guidelines that specify what counts as evidence for each symptom and uses these guidelines to condition classification. Across four LLM families and eight models, SI achieves the best overall weighted F1 on BDI-Sen, with especially large gains for infrequent symptoms. Cross-domain evaluation on an external dataset further shows that induced guidelines generalize across other diseases shared symptomatology (bipolar and eating disorders).
>
---
#### [new 098] Rewarding the Scientific Process: Process-Level Reward Modeling for Agentic Data Analysis
- **分类: cs.CL; cs.AI; cs.CE; cs.LG; cs.MA**

- **简介: 该论文属于数据分析任务，旨在解决过程奖励模型在动态环境中的有效性问题。通过设计DataPRM，提升代理的推理与纠错能力。**

- **链接: [https://arxiv.org/pdf/2604.24198](https://arxiv.org/pdf/2604.24198)**

> **作者:** Zhisong Qiu; Shuofei Qiao; Kewei Xu; Yuqi Zhu; Lun Du; Ningyu Zhang; Huajun Chen
>
> **备注:** Work in progress
>
> **摘要:** Process Reward Models (PRMs) have achieved remarkable success in augmenting the reasoning capabilities of Large Language Models (LLMs) within static domains such as mathematics. However, their potential in dynamic data analysis tasks remains underexplored. In this work, we first present a empirical study revealing that general-domain PRMs struggle to supervise data analysis agents. Specifically, they fail to detect silent errors, logical flaws that yield incorrect results without triggering interpreter exceptions, and erroneously penalize exploratory actions, mistaking necessary trial-and-error exploration for grounding failures. To bridge this gap, we introduce DataPRM, a novel environment-aware generative process reward model that (1) can serve as an active verifier, autonomously interacting with the environment to probe intermediate execution states and uncover silent errors, and (2) employs a reflection-aware ternary reward strategy that distinguishes between correctable grounding errors and irrecoverable mistakes. We design a scalable pipeline to construct over 8K high-quality training instances for DataPRM via diversity-driven trajectory generation and knowledge-augmented step-level annotation. Experimental results demonstrate that DataPRM improves downstream policy LLMs by 7.21% on ScienceAgentBench and 11.28% on DABStep using Best-of-N inference. Notably, with only 4B parameters, DataPRM outperforms strong baselines, and exhibits robust generalizability across diverse Test-Time Scaling strategies. Furthermore, integrating DataPRM into Reinforcement Learning yields substantial gains over outcome-reward baselines, achieving 78.73% on DABench and 64.84% on TableBench, validating the effectiveness of process reward supervision. Code is available at this https URL.
>
---
#### [new 099] Revisiting Greedy Decoding for Visual Question Answering: A Calibration Perspective
- **分类: cs.CL**

- **简介: 该论文针对视觉问答任务，探讨了贪婪解码与随机采样的效果差异，提出贪婪解码在该任务中的优势。**

- **链接: [https://arxiv.org/pdf/2604.23443](https://arxiv.org/pdf/2604.23443)**

> **作者:** Boqi Chen; Xudong Liu; Yunke Ao; Jianing Qiu
>
> **摘要:** Stochastic sampling strategies are widely adopted in large language models (LLMs) to balance output coherence and diversity. These heuristics are often inherited in Multimodal LLMs (MLLMs) without task-specific justification. However, we contend that stochastic decoding can be suboptimal for Visual Question Answering (VQA). VQA is a closed-ended task with head-heavy answer distributions where uncertainty is usually epistemic, arising from missing or ambiguous visual evidence rather than plausible continuations. In this work, we provide a theoretical formalization of the relationship between model calibration and predictive accuracy, and derive the sufficient conditions for greedy decoding optimality. Extensive experiments provide empirical evidence for the superiority of greedy decoding over stochastic sampling across multiple benchmarks. Furthermore, we propose Greedy Decoding for Reasoning Models, which outperforms both stochastic sampling and standard greedy decoding in multimodal reasoning scenarios. Overall, our results caution against naively inheriting LLMs decoding heuristics in MLLMs and demonstrate that greedy decoding can be an efficient yet strong default for VQA.
>
---
#### [new 100] KOMBO: Korean Character Representations Based on the Combination Rules of Subcharacters
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出KOMBO框架，解决韩国语字符表示问题，基于韩文构造规则提升语言模型性能。**

- **链接: [https://arxiv.org/pdf/2604.23948](https://arxiv.org/pdf/2604.23948)**

> **作者:** SungHo Kim; Juhyeong Park; Yeachan Kim; SangKeun Lee
>
> **备注:** Presented at ACL 2024 Findings
>
> **摘要:** The Korean writing system, \textit{Hangeul}, has a unique character representation rigidly following the invention principles recorded in \textit{Hunminjeongeum}.\footnote{\textit{Hunminjeongeum} is a book published in 1446 that describes the principles of invention and usage of \textit{Hangeul}, devised by King Sejong \cite{Hunminjeongeum_Guide}.} However, existing pre-trained language models (PLMs) for Korean have overlooked these principles. In this paper, we introduce a novel framework for Korean PLMs called KOMBO, which firstly brings the invention principles of \textit{Hangeul} to represent character. Our proposed method, KOMBO, exhibits notable experimental proficiency across diverse NLP tasks. In particular, our method outperforms the state-of-the-art Korean PLM by an average of 2.11\% in five Korean natural language understanding tasks. Furthermore, extensive experiments demonstrate that our proposed method is suitable for comprehending the linguistic features of the Korean language. Consequently, we shed light on the superiority of using subcharacters over the typical subword-based approach for Korean PLMs. Our code is available at: [this https URL](this https URL).
>
---
#### [new 101] Human-1 by Josh Talks: A Full-Duplex Conversational Modeling Framework in Hindi using Real-World Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种用于印地语的全双工对话建模框架，解决印度语言在全双工对话系统中的研究不足问题，通过改进语音架构和训练数据实现自然对话行为生成。**

- **链接: [https://arxiv.org/pdf/2604.23295](https://arxiv.org/pdf/2604.23295)**

> **作者:** Bhaskar Singh; Shobhit Banga; Pranav Sharma
>
> **摘要:** Full-duplex spoken dialogue systems can model natural conversational behaviours such as interruptions, overlaps, and backchannels, yet such systems remain largely unexplored for Indian languages. We present the first open, reproducible full-duplex spoken dialogue system for Hindi by adapting Moshi, a state-of-the-art duplex speech architecture, using a custom Hindi tokeniser and training on 26,000 hours of real spontaneous conversations collected from 14,695 speakers with separate speaker channels, enabling direct learning of turn-taking and overlap patterns from natural interactions. To support Hindi text generation, we replace the original English tokeniser and reinitialise text-vocabulary-dependent parameters while retaining the pre-trained audio components. We propose a two-stage training recipe -- large-scale pre-training followed by fine-tuning on 1,000 hours of conversational data. Evaluation through the prompted dialogue continuation paradigm with both automatic metrics and human judgments demonstrates that the resulting model generates natural and meaningful full-duplex conversational behaviour in Hindi. This work serves as a first step toward real-time duplex spoken dialogue systems for Hindi and other Indian languages.
>
---
#### [new 102] DARC-CLIP: Dynamic Adaptive Refinement with Cross-Attention for Meme Understanding
- **分类: cs.CL**

- **简介: 该论文提出DARC-CLIP框架，用于解决社交媒体中表情包的多模态内容分析任务，通过动态自适应融合提升有害内容检测效果。**

- **链接: [https://arxiv.org/pdf/2604.23214](https://arxiv.org/pdf/2604.23214)**

> **作者:** Qiyuan Jin
>
> **备注:** Accepted to IEEE ICASSP 2026. 5 pages, 3 figures, 4 tables
>
> **摘要:** Memes convey meaning through the interaction of visual and textual signals, often combining humor, irony, and offense in subtle ways. Detecting harmful or sensitive content in memes requires accurate modeling of these multimodal cues. Existing CLIP-based approaches rely on static fusion, which struggles to capture fine grained dependencies between modalities. We propose DARC-CLIP, a CLIP-based framework for adaptive multimodal fusion with a hierarchical refinement stack. DARC-CLIP introduces Adaptive Cross-Attention Refiners to for bidirectional information alignment and Dynamic Feature Adapters for task-sensitive signal adaptation. We evaluate DARC-CLIP on the PrideMM benchmark, which includes hate, target, stance, and humor classification, and further test generalization on the CrisisHateMM dataset. DARC-CLIP achieves highly competitive classification accuracy across tasks, with significant gains of +4.18 AUROC and +6.84 F1 in hate detection over the strongest baseline. Ablation studies confirm that ACAR and DFA are the main contributors to these gains. These results show that adaptive cross-signal refinement is an effective strategy for multimodal content analysis in socially sensitive classification.
>
---
#### [new 103] Mixture of Heterogeneous Grouped Experts for Language Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型优化任务，解决MoE中专家规模固定导致的资源浪费和GPU负载不均问题，提出MoHGE框架实现灵活专家组合与高效推理。**

- **链接: [https://arxiv.org/pdf/2604.23108](https://arxiv.org/pdf/2604.23108)**

> **作者:** Zhicheng Ma; Xiang Liu; Zhaoxiang Liu; Ning Wang; Yi Shen; Kai Wang; Shuming Shi; Shiguo Lian
>
> **备注:** Accepted by ACL2026
>
> **摘要:** Large Language Models (LLMs) based on Mixture-of-Experts (MoE) are pivotal in industrial applications for their ability to scale performance efficiently. However, standard MoEs enforce uniform expert sizes,creating a rigidity that fails to align computational costs with varying token-level complexity. While heterogeneous expert architectures attempt to address this by diversifying expert sizes, they often suffer from significant system-level challenges, specifically unbalanced GPU utilization and inefficient parameter utilization, which hinder practical deployment. To bridge the gap between theoretical heterogeneity and robust industrial application, we propose Mixture of Heterogeneous Grouped Experts (MoHGE) which introduces a two-level routing mechanism to enable flexible, resource-aware expert combinations. To optimize inference efficiency, we propose a Group-Wise Auxiliary Loss, which dynamically steers tokens to the most parameter-efficient expert groups based on task difficulty. To address the critical deployment challenge of GPU load balancing, we introduce an All-size Group-decoupling Allocation strategy coupled with an Intra-Group Experts Auxiliary Loss. These mechanisms collectively ensure uniform computation distribution across GPUs. Extensive evaluations demonstrate that MoHGE matches the performance of MoE architectures while reducing the total parameters by approximately 20% and maintaining balanced GPU utilization. Our work establishes a scalable paradigm for resource-efficient MoE design, offering a practical solution for optimizing inference costs in real-world scenarios.
>
---
#### [new 104] MEG-RAG: Quantifying Multi-modal Evidence Grounding for Evidence Selection in RAG
- **分类: cs.CL; cs.IR; cs.IT**

- **简介: 该论文属于多模态问答任务，解决MRAG系统中证据选择不准确的问题。提出MEG-RAG框架，通过语义锚定提升证据与答案的一致性。**

- **链接: [https://arxiv.org/pdf/2604.24564](https://arxiv.org/pdf/2604.24564)**

> **作者:** Xihang Wang; Zihan Wang; Chengkai Huang; Quan Z. Sheng; Lina Yao
>
> **摘要:** Multimodal Retrieval-Augmented Generation (MRAG) addresses key limitations of Multimodal Large Language Models (MLLMs), such as hallucination and outdated knowledge. However, current MRAG systems struggle to distinguish whether retrieved multimodal data truly supports the semantic core of an answer or merely provides superficial relevance. Existing metrics often rely on heuristic position-based confidence, which fails to capture the informational density of multimodal entities. To address this, we propose Multi-modal Evidence Grounding (MEG), a semantic-aware metric that quantifies the contribution of retrieved evidence. Unlike standard confidence measures, MEG utilizes Semantic Certainty Anchoring, focusing on high-IDF information-bearing tokens that better capture the semantic core of the answer. Building on MEG, we introduce MEG-RAG, a framework that trains a multimodal reranker to align retrieved evidence with the semantic anchors of the ground truth. By prioritizing high-value content based on semantic grounding rather than token probability distributions, MEG-RAG improves the accuracy and multimodal consistency of generated outputs. Extensive experiments on the M$^2$RAG benchmark show that MEG-RAG consistently outperforms strong baselines and demonstrates robust generalization across different teacher models.
>
---
#### [new 105] MIPIC: Matryoshka Representation Learning via Self-Distilled Intra-Relational and Progressive Information Chaining
- **分类: cs.CL**

- **简介: 该论文提出MIPIC，解决Matryoshka表示学习中的结构一致性和语义紧凑性问题，通过自蒸馏和渐进信息链提升多尺度嵌入性能。**

- **链接: [https://arxiv.org/pdf/2604.24374](https://arxiv.org/pdf/2604.24374)**

> **作者:** Phung Gia Huy; Hai An Vu; Minh-Phuc Truong; Thang Duc Tran; Linh Ngo Van; Thanh Hong Nguyen; Trung Le
>
> **备注:** ACL Findings
>
> **摘要:** Representation learning is fundamental to NLP, but building embeddings that work well at different computational budgets is challenging. Matryoshka Representation Learning (MRL) offers a flexible inference paradigm through nested embeddings; however, learning such structures requires explicit coordination of how information is arranged across embedding dimensionality and model depth. In this work, we propose MIPIC (Matryoshka Representation Learning via Self-Distilled Intra-Relational Alignment and Progressive Information Chaining), a unified training framework designed to produce structurally coherent and semantically compact Matryoshka representations. MIPIC promotes cross-dimensional structural consistency through Self-Distilled Intra-Relational Alignment (SIA), which aligns token-level geometric and attention-driven relations between full and truncated representations using top-k CKA self-distillation. Complementarily, it enables depth-wise semantic consolidation via Progressive Information Chaining (PIC), a scaffolded alignment strategy that incrementally transfers mature task semantics from deeper layers into earlier layers. Extensive experiments on STS, NLI, and classification benchmarks (spanning models from TinyBERT to BGEM3, Qwen3) demonstrate that MIPIC yields Matryoshka representations that are highly competitive across all capacities, with significant performance advantages observed under extreme low-dimensional.
>
---
#### [new 106] $\mathcal{S}^2$IT: Stepwise Syntax Integration Tuning for Large Language Models in Aspect Sentiment Quad Prediction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于Aspect Sentiment Quad Prediction任务，解决LLMs在生成范式中未充分利用句法结构的问题。提出S²IT框架，通过多步调优逐步整合句法知识，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.23296](https://arxiv.org/pdf/2604.23296)**

> **作者:** Bingfeng Chen; Chenjie Qiu; Yifeng Xie; Boyan Xu; Ruichu Cai; Zhifeng Hao
>
> **备注:** Accepted to Findings of NAACL 2025
>
> **摘要:** Aspect Sentiment Quad Prediction (ASQP) has seen significant advancements, largely driven by the powerful semantic understanding and generative capabilities of large language models (LLMs). However, while syntactic structure information has been proven effective in previous extractive paradigms, it remains underutilized in the generative paradigm of LLMs due to their limited reasoning capabilities. In this paper, we propose S^2IT, a novel Stepwise Syntax Integration Tuning framework that progressively integrates syntactic structure knowledge into LLMs through a multi-step tuning process. The training process is divided into three steps. S^2IT decomposes the quadruple generation task into two stages: 1) Global Syntax-guided Extraction and 2) Local Syntax-guided Classification, integrating both global and local syntactic structure information. Finally, Fine-grained Structural Tuning enhances the model's understanding of syntactic structures through the prediction of element links and node classification. Experiments demonstrate that S^2IT significantly improves state-of-the-art performance across multiple datasets. Our implementation will be open-sourced at this https URL.
>
---
#### [new 107] TSAssistant: A Human-in-the-Loop Agentic Framework for Automated Target Safety Assessment
- **分类: cs.CL**

- **简介: 该论文属于目标安全评估任务，旨在解决TSA过程的迭代性与可重复性问题。提出TSAssistant框架，通过多代理协作和人机交互，提升报告生成效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.23938](https://arxiv.org/pdf/2604.23938)**

> **作者:** Xiaochen Zheng; Zhiwen Jiang; Melanie Guerard; Klas Hatje; Tatyana Doktorova
>
> **备注:** Preliminary version; quantitative evaluation results to be included in a future revision
>
> **摘要:** Target Safety Assessment (TSA) requires systematic integration of heterogeneous evidence, including genetic, transcriptomic, target homology, pharmacological, and clinical data, to evaluate potential safety liabilities of therapeutic targets. This process is inherently iterative and expert-driven, posing challenges in scalability and reproducibility. We present TSAssistant, a multi-agent framework designed to support TSA report drafting through a modular, section-based, and human-in-the-loop paradigm. The framework decomposes report generation into a coordinated pipeline of specialised subagents, each targeting a single TSA section. Specialised subagents retrieve structured and unstructured data as well as literature evidence from curated biomedical sources through standardised tool interfaces, producing individually citable, evidence-grounded sections. Agent behaviour is governed by a hierarchical instruction architecture comprising system prompts, domain-specific skill modules, and runtime user instructions. A key feature is an interactive refinement loop in which users may manually edit sections, append new information, upload additional sources, or re-invoke agents to revise specific sections, with the system maintaining conversational memory across iterations. TSAssistant is designed to reduce the mechanical burden of evidence synthesis and report drafting, supporting a hybrid model in which agentic AI augments evidence synthesis while toxicologists retain final decision authority.
>
---
#### [new 108] K-SENSE: A Knowledge-Guided Self-Augmented Encoder for Neuro-Semantic Evaluation of Mental Health Conditions on Social Media
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出K-SENSE框架，用于社交媒体中文本的心理健康状况评估，解决压力和抑郁检测问题，结合外部知识与自我增强方法提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.23493](https://arxiv.org/pdf/2604.23493)**

> **作者:** Vijay Yadav
>
> **摘要:** Early detection of mental health conditions, particularly stress and depression, from social media text remains a challenging open problem in computational psychiatry and natural language processing. Automated systems must contend with figurative language, implicit emotional expression, and the high noise inherent in user-generated content. Existing approaches either leverage external commonsense knowledge to model mental states explicitly, or apply self-augmentation and contrastive training to improve generalization, but seldom do both in a principled, unified framework. We propose K-SENSE (Knowledge-guided Self-augmented Encoder for Neuro-Semantic Evaluation of Mental Health), a framework that jointly exploits external psychological reasoning and internal representation robustness. K-SENSE adopts a three-stage encoding pipeline: (1) inferential commonsense knowledge is extracted from the COMET model across five mental state dimensions; (2) a semantic anchor is constructed by combining hidden representations from two parallel encoding streams, projected into a shared space before fusion; and (3) a supervised contrastive learning objective aligns same-class representations while encouraging the attention mechanism to suppress irrelevant knowledge noise. We evaluate K-SENSE on Dreaddit (stress detection) and Depression_Mixed (depression detection), achieving mean F1-scores of 86.1 (0.6%) and 94.3 (0.8%), respectively, over five independent runs. These represent improvements of approximately 2.6 and 1.5 percentage points over the strongest prior baselines. Ablation experiments confirm the contribution of each architectural component, including the temporal knowledge integration strategy and the choice to keep the knowledge encoder frozen during fine-tuning.
>
---
#### [new 109] MEMCoder: Multi-dimensional Evolving Memory for Private-Library-Oriented Code Generation
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出MEMCoder，解决企业环境中LLM因缺乏私有库知识导致的代码生成问题。通过多维记忆机制和闭环反馈提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.24222](https://arxiv.org/pdf/2604.24222)**

> **作者:** Mofei Li; Taozhi Chen; Guowei Yang; Jia Li
>
> **摘要:** Large Language Models (LLMs) excel at general code generation, but their performance drops sharply in enterprise settings that rely on internal private libraries absent from public pre-training corpora. While Retrieval-Augmented Generation (RAG) offers a training-free alternative by providing static API documentation, we find that such documentation typically provides only isolated definitions, leaving a fundamental knowledge gap. Specifically, LLMs struggle with a task-level lack of coordination patterns between APIs and an API-level misunderstanding of parameter constraints and boundary conditions. To address this, we propose MEMCoder, a novel framework that enables LLMs to autonomously accumulate and evolve Usage Guidelines across these two dimensions. MEMCoder introduces a Multi-dimensional Evolving Memory that captures distilled lessons from the model's own problem-solving trajectories. During inference, MEMCoder employs a dual-source retrieval mechanism to inject both static documentation and relevant historical guidelines into the context. The framework operates in an automated closed loop by using objective execution feedback to reflect on successes and failures, resolve knowledge conflicts, and dynamically update memory. Extensive evaluations on the NdonnxEval and NumbaEval benchmarks demonstrate that MEMCoder substantially enhances existing RAG systems, yielding an average absolute pass@1 gain of 16.31%. Furthermore, MEMCoder exhibits vastly superior domain-specific adaptation compared to existing memory-based continual learning methods.
>
---
#### [new 110] FinGround: Detecting and Grounding Financial Hallucinations via Atomic Claim Verification
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于金融问答任务，旨在解决AI系统生成虚假财务信息的问题。通过三阶段验证流程，提升答案的准确性和可追溯性。**

- **链接: [https://arxiv.org/pdf/2604.23588](https://arxiv.org/pdf/2604.23588)**

> **作者:** Dongxin Guo; Jikun Wu; Siu Ming Yiu
>
> **备注:** Accepted to ACL 2026 Industry Track. 14 pages, 1 figure, 14 tables
>
> **摘要:** Financial AI systems must produce answers grounded in specific regulatory filings, yet current LLMs fabricate metrics, invent citations, and miscalculate derived quantities. These errors carry direct regulatory consequences as the EU AI Act's high-risk enforcement deadline approaches (August 2026). Existing hallucination detectors treat all claims uniformly, missing 43% of computational errors that require arithmetic re-verification against structured tables. We present FinGround, a three-stage verify-then-ground pipeline for financial document QA. Stage 1 performs finance-aware hybrid retrieval over text and tables. Stage 2 decomposes answers into atomic claims classified by a six-type financial taxonomy and verified with type-routed strategies including formula reconstruction. Stage 3 rewrites unsupported claims with paragraph- and table-cell-level citations. To cleanly isolate verification value from retrieval quality, we propose retrieval-equalized evaluation as standard methodology for RAG verification research: when all systems receive identical retrieval, FinGround still reduces hallucination rates by 68% over the strongest baseline ($p < 0.01$). The full pipeline achieves a 78% reduction relative to GPT-4o. An 8B distilled detector retains 91.4% F1 at 18x lower per-claim latency, enabling $0.003/query deployment, supported by qualitative signals from a four-week analyst pilot.
>
---
#### [new 111] Discovering Agentic Safety Specifications from 1-Bit Danger Signals
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI安全任务，旨在通过稀疏危险信号发现隐藏安全规范。工作包括提出EPO-Safe框架，利用少量危险反馈进化出可解释的安全规则。**

- **链接: [https://arxiv.org/pdf/2604.23210](https://arxiv.org/pdf/2604.23210)**

> **作者:** Víctor Gallego
>
> **备注:** Accepted to the Adaptive and Learning Agents Workshop (ALA 2026) @ AAMAS 2026. Code is available at this http URL
>
> **摘要:** Can large language model agents discover hidden safety objectives through experience alone? We introduce EPO-Safe (Experiential Prompt Optimization for Safe Agents), a framework where an LLM iteratively generates action plans, receives sparse binary danger warnings, and evolves a natural language behavioral specification through reflection. Unlike standard LLM reflection methods that rely on rich textual feedback (e.g., compiler errors or detailed environment responses), EPO-Safe demonstrates that LLMs can perform safety reasoning from a strictly impoverished signal in structured, low-dimensional environments: the agent never observes the hidden performance function $R^*$, only a single bit per timestep indicating that an action was unsafe. We evaluate on five AI Safety Gridworlds (Leike et al., 2017) and five text-based scenario analogs where visible reward $R$ may diverge from $R^*$. EPO-Safe discovers safe behavior within 1-2 rounds (5-15 episodes), producing human-readable specifications with correct explanatory hypotheses about hazards (e.g., "X cells are directionally hazardous: entering from the north is dangerous"). Critically, we show that standard reward-driven reflection actively degrades safety: agents reflecting on reward alone use the loop to justify and accelerate reward hacking, proving that reflection must be paired with a dedicated safety channel to discover hidden constraints. We further evaluate robustness to noisy oracles: even when 50% of non-dangerous steps produce spurious warnings, mean safety performance degrades by only 15% on average, though sensitivity is environment-dependent, as cross-episode reflection naturally filters inconsistent signals. Each evolved specification functions as an auditable set of grounded behavioral rules discovered autonomously through interaction, rather than authored by humans as in Constitutional AI (Bai et al., 2022).
>
---
#### [new 112] Large language model-enabled automated data extraction for concrete materials informatics
- **分类: cond-mat.mtrl-sci; cs.CL; cs.LG**

- **简介: 该论文属于材料信息学领域，旨在解决实验数据稀缺问题。通过构建LLM驱动的自动化数据提取管道，从文献中高效获取高质量混凝土材料数据。**

- **链接: [https://arxiv.org/pdf/2604.22938](https://arxiv.org/pdf/2604.22938)**

> **作者:** Zhanzhao Li; Kengran Yang; Qiyao He; Kai Gong
>
> **备注:** 20 pages, 5 figures, 1 table
>
> **摘要:** The promise of data-driven materials discovery remains constrained by the scarcity of large, high-quality, and accessible experimental datasets. Here, we introduce a generalizable large language model (LLM)-powered pipeline for automated extraction and structuring of materials data from unstructured scientific literature, using concrete materials as a representative and particularly challenging example. The pipeline exhibits robust performance across a broad range of LLMs and achieves an $F_1$ score of up to 0.97 for diverse composition--process--property attributes. Within one hour, it extracts nearly 9,000 high-quality records with over 100 attributes screened from more than 27,000 publications, enabling the construction of the largest open laboratory database for blended cement concrete. Machine learning analyses underscore the importance of large, diverse, and information-rich datasets for enhancing both in-distribution accuracy and out-of-distribution generalization to unseen materials. The proposed pipeline is readily adaptable to other materials domains and accelerates the development of scalable data infrastructures for materials informatics.
>
---
#### [new 113] Efficient Rationale-based Retrieval: On-policy Distillation from Generative Rerankers based on JEPA
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于信息检索任务，旨在解决基于理由的检索计算成本高的问题。提出Rabtriever模型，通过知识蒸馏和JEPA架构实现高效检索。**

- **链接: [https://arxiv.org/pdf/2604.23336](https://arxiv.org/pdf/2604.23336)**

> **作者:** Teng Chen; Sheng Xu; Feixiang Guo; Xiaoyu Wang; Qingqing Gu; Hongyan Li; Luo Ji
>
> **备注:** 11 pages, 8 figures. ICMR 2026
>
> **摘要:** Unlike traditional fact-based retrieval, rationale-based retrieval typically necessitates cross-encoding of query-document pairs using large language models, incurring substantial computational costs. To address this limitation, we propose Rabtriever, which independently encodes queries and documents, while providing comparable cross query-document comprehension capabilities to rerankers. We start from training a LLM-based generative reranker, which puts the document prior to the query and prompts the LLM to generate the relevance score by log probabilities. We then employ it as the teacher of an on-policy distillation framework, with Rabtriever as the student to reconstruct the teacher's contextual-aware query embedding. To achieve this effect, Rabtriever is first initialized from the teacher, with parameters frozen. The Joint-Embedding Predictive Architecture (JEPA) paradigm is then adopted, which integrates a lightweight, trainable predictor between LLM layers and heads, projecting the query embedding into a new hidden space, with the document embedding as the latent vector. JEPA then minimizes the distribution difference between this projected embedding and the teacher embedding. To strengthen the sampling efficiency of on-policy distillation, we also add an auxiliary loss on the reverse KL of LLM logits, to reshape the student's logit distribution. Rabtriever optimizes the teacher's quadratic complexity on the document length to linear, verified both theoretically and empirically. Experiments show that Rabtriever outperforms different retriever baselines across diverse rationale-based tasks, including empathetic conversations and robotic manipulations, with minor accuracy degradation from the reranker. Rabtriever also generalizes well on traditional retrieval benchmarks such as MS MARCO and BEIR, with comparable performance to the best retriever baseline.
>
---
#### [new 114] Towards Lawful Autonomous Driving: Deriving Scenario-Aware Driving Requirements from Traffic Laws and Regulations
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于自动驾驶法律合规任务，旨在解决AV在复杂场景中违反交通法规的问题。通过构建场景分类体系，提升LLM对法规的精准理解与应用。**

- **链接: [https://arxiv.org/pdf/2604.24562](https://arxiv.org/pdf/2604.24562)**

> **作者:** Bowen Jian; Rongjie Yu; Hong Wang; Liqiang Wang; Zihang Zou
>
> **摘要:** Driving in compliance with traffic laws and regulations is a basic requirement for human drivers, yet autonomous vehicles (AVs) can violate these requirements in diverse real-world scenarios. To encode law compliance into AV systems, conventional approaches use formal logic languages to explicitly specify behavioral constraints, but this process is labor-intensive, hard to scale, and costly to maintain. With recent advances in artificial intelligence, it is promising to leverage large language models (LLMs) to derive legal requirements from traffic laws and regulations. However, without explicitly grounding and reasoning in structured traffic scenarios, LLMs often retrieve irrelevant provisions or miss applicable ones, yielding imprecise requirements. To address this, we propose a novel pipeline that grounds LLM reasoning in a traffic scenario taxonomy through node-wise anchors that encode hierarchical semantics. On Chinese traffic laws and OnSite dataset (5,897 scenarios), our method improves law-scenario matching by 29.1\% and increases the accuracy of derived mandatory and prohibitive requirements by 36.9\% and 38.2\%, respectively. We further demonstrate real-world applicability by constructing a law-compliance layer for AV navigation and developing an onboard, real-time compliance monitor for in-field testing, providing a solid foundation for future AV development, deployment, and regulatory oversight.
>
---
#### [new 115] Case-Specific Rubrics for Clinical AI Evaluation: Methodology, Validation, and LLM-Clinician Agreement Across 823 Encounters
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于临床AI评估任务，旨在解决人工评审成本高、效率低的问题。通过构建病例特异性评分标准，验证LLM生成的评分与医生一致性，实现低成本高效评估。**

- **链接: [https://arxiv.org/pdf/2604.24710](https://arxiv.org/pdf/2604.24710)**

> **作者:** Aaryan Shah; Andrew Hines; Alexia Downs; Denis Bajet; Paulius Mui; Fabiano Araujo; Laura Offutt; Aida Rutledge; Elizabeth Jimenez
>
> **备注:** 14 pages, 2 figures, 3 tables, submitted to JAMIA
>
> **摘要:** Objective. Clinical AI documentation systems require evaluation methodologies that are clinically valid, economically viable, and sensitive to iterative changes. Methods requiring expert review per scoring instance are too slow and expensive for safe, iterative deployment. We present a case-specific, clinician-authored rubric methodology for clinical AI evaluation and examine whether LLM-generated rubrics can approximate clinician agreement. Materials and Methods. Twenty clinicians authored 1,646 rubrics for 823 clinical cases (736 real-world, 87 synthetic) across primary care, psychiatry, oncology, and behavioral health. Each rubric was validated by confirming that an LLM-based scoring agent consistently scored clinician-preferred outputs higher than rejected ones. Seven versions of an EHR-embedded AI agent for clinicians were evaluated across all cases. Results. Clinician-authored rubrics discriminated effectively between high- and low-quality outputs (median score gap: 82.9%) with high scoring stability (median range: 0.00%). Median scores improved from 84% to 95%. In later experiments, clinician-LLM ranking agreement (tau: 0.42-0.46) matched or exceeded clinician-clinician agreement (tau: 0.38-0.43), attributable to both ceiling compression and LLM rubric improvement. Discussion. This convergence supports incorporating LLM rubrics alongside clinician-authored ones. At roughly 1,000 times lower cost, LLM rubrics enable substantially greater evaluation coverage, while continued clinical authorship grounds evaluation in expert judgment. Ceiling compression poses a methodological challenge for future inter-rater agreement studies. Conclusion. Case-specific rubrics offer a path for clinical AI evaluation that preserves expert judgment while enabling automation at three orders lower cost. Clinician-authored rubrics establish the baseline against which LLM rubrics are validated.
>
---
#### [new 116] Automating Categorization of Scientific Texts with In-Context Learning and Prompt-Chaining in Large Language Models
- **分类: cs.IR; cs.AI; cs.CL; cs.DL; cs.SE**

- **简介: 该论文属于科学文本分类任务，旨在解决大规模文献中自动分类难题。通过引入ICL和Prompt Chaining策略，提升LLMs在ORKG分类体系中的准确率。**

- **链接: [https://arxiv.org/pdf/2604.23430](https://arxiv.org/pdf/2604.23430)**

> **作者:** Gautam Kishore Shahi; Oliver Hummel
>
> **备注:** 25 pages
>
> **摘要:** The relentless expansion of scientific literature presents significant challenges for navigation and knowledge discovery. Within Research Information Retrieval, established tasks such as text summarization and classification remain crucial for enabling researchers and practitioners to effectively navigate this vast landscape, so that efforts have increasingly been focused on developing advanced research information systems. These systems aim not only to provide standard keyword-based search functionalities but also to incorporate capabilities for automatic content categorization within knowledge-intensive organizations across academia and industry. This study systematically evaluates the performance of off-the-shelf Large Language Models (LLMs) in analyzing scientific texts according to a given classification scheme. We utilized the hierarchical ORKG taxonomy as a classification framework, employing the FORC dataset as ground truth. We investigated the effectiveness of advanced prompt engineering strategies, namely In-Context Learning (ICL) and Prompt Chaining, and experimentally explored the influence of the LLMs' temperature hyperparameter on classification accuracy. Our experiments demonstrate that Prompt Chaining yields superior classification accuracy compared to pure ICL, particularly when applied to the nested structure of the ORKG taxonomy. LLMs with prompt chaining outperform the state-of-the-art models for domain (1st level) prediction and show even better performance for subject (2nd level) prediction compared to the older BERT model. However, LLMs are not yet able to perform well in classifying the topic (3rd level) of research areas based on this specific hierarchical taxonomy, as they only reach about 50% accuracy even with prompt chaining.
>
---
#### [new 117] Learning to Route Queries to Heads for Attention-based Re-ranking with Large Language Models
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决LLM注意力机制中头选择不当导致的重排序效果不佳问题。提出RouteHead方法，通过学习查询到最优头集的路由，提升重排序性能。**

- **链接: [https://arxiv.org/pdf/2604.24608](https://arxiv.org/pdf/2604.24608)**

> **作者:** Yuxing Tian; Fengran Mo; Zhiqi Huang; Weixu Zhang; Jian-Yun Nie
>
> **备注:** Accepted by SIGIR 2026
>
> **摘要:** Large Language Models (LLMs) have recently been explored as fine-grained zero-shot re-rankers by leveraging attention signals to estimate document relevance. However, existing methods either aggregate attention signals across all heads or rely on a statically selected subset identified by heuristic rules. This solution can be suboptimal because the informative heads can vary across queries or domains. Moreover, naively combining multiple heads can degrade performance due to redundancy or conflicting ranking signals. In this paper, we propose a query-dependent head selection method, RouteHead, for attention-based re-ranking with LLMs. Specifically, we learn a lightweight router that can map each query to an optimal head set, and relevance scores are computed by aggregating attention signals only from these heads. Since query-to-head optimal labels are unavailable, we first construct pseudo labels via an offline search. The router represents each head with a learnable embedding and represents each query using an embedding extracted from the hidden states of the frozen LLM. Then it is trained on the pseudo labels with a sparsity regularizer. Experiments on diverse benchmarks and multiple LLM backbones show that the proposed method consistently outperforms strong baselines.
>
---
#### [new 118] DeepTaxon: An Interpretable Retrieval-Augmented Multimodal Framework for Unified Species Identification and Discovery
- **分类: cs.CV; cs.CL; cs.IR; cs.MM**

- **简介: 该论文提出DeepTaxon，解决物种识别与发现任务。通过多模态检索增强框架，统一处理识别与发现问题，提升准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2604.24029](https://arxiv.org/pdf/2604.24029)**

> **作者:** Jiawei Wang; Ming Lei; Yaning Yang; Xinyan Lin; Yuquan Le; Qiwei Ma; Zhiwei Xu; Zheqi Lv; Yuchen Ang; Zhe Quan; Tat-Seng Chua
>
> **备注:** 13 pages, 6 figures, 9 tables
>
> **摘要:** Identifying species in biology among tens of thousands of visually similar taxa while discovering unknown species in open-world environments remains a fundamental challenge in biodiversity research. Current methods treat identification and discovery as separate problems, with classification models assuming closed sets and discovery relying on threshold-based rejection. Here we present DeepTaxon, a retrieval-augmented multimodal framework that unifies species identification and discovery through interpretable reasoning over retrieved visual evidence. Given a query image, DeepTaxon retrieves the top-$k$ candidate species with $n$ exemplar images each from a retrieval index and performs chain-of-thought comparative reasoning. Critically, we redefine discovery as an explicit, retrieval-based decision problem rather than an implicit parametric memory problem. A sample is novel if and only if the retrieval index lacks sufficient evidence for identification, so each retrieval naturally yields a classification or discovery label without manual annotation, thereby providing automatic supervision for both tasks. We train the framework via supervised fine-tuning on synthetic retrieval-augmented data, followed by reinforcement learning on hard samples, converting high-recall retrieval into high-precision decisions that scale to massive taxonomic vocabularies. Extensive experiments on a large-scale in-distribution benchmark and six out-of-distribution datasets demonstrate consistent improvements in both identification and discovery. Ablation studies further reveal effective test-time scaling with candidate count $k$ and exemplar count $n$, strong zero-shot transfer to unseen domains, and consistent performance across retrieval encoders, establishing an interpretable solution for biodiversity research.
>
---
#### [new 119] An Information-Geometric Framework for Stability Analysis of Large Language Models under Entropic Stress
- **分类: cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于AI可靠性分析任务，旨在解决大语言模型在不确定性下的稳定性评估问题。通过构建信息几何框架，结合熵与内部结构指标，提升模型稳定性评分。**

- **链接: [https://arxiv.org/pdf/2604.24076](https://arxiv.org/pdf/2604.24076)**

> **作者:** Hikmat Karimov; Rahid Zahid Alekberli
>
> **摘要:** As large language models (LLMs) are increasingly deployed in high-stakes and operational settings, evaluation strategies based solely on aggregate accuracy are often insucient to characterize system reliability. This study proposes a thermodynamic inspired modeling framework for analyzing the stability of LLM outputs under conditions of uncertainty and perturbation. The framework introduces a composite stability score that integrates task utility, entropy as a measure of external uncertainty, and two internal structural proxies: internal integration and aligned reective capacity. Rather than interpreting these quantities as physical variables, the formulation is intended as an interpretable abstraction that captures how internal structure may modulate the impact of disorder on model behavior. Using the IST-20 benchmarking protocol and associated metadata, we analyze 80 modelscenario observations across four contemporary LLMs. The proposed formulation consistently yields higher stability scores than a reduced utilityentropy baseline, with a mean improvement of 0.0299 (95% CI: 0.02470.0351). The observed gain is more pronounced under higher entropy conditions, suggesting that the framework captures a form of nonlinear attenuation of uncertainty. We do not claim a fundamental physical law or a complete theory of machine ethics. Instead, the contribution of this work is a compact and interpretable modeling perspective that connects uncertainty, performance, and internal structure within a unied evaluation lens. The framework is intended to complement existing benchmarking approaches and to support ongoing discussions in AI safety, reliability, and governance.
>
---
#### [new 120] RedParrot: Accelerating NL-to-DSL for Business Analytics via Query Semantic Caching
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于自然语言到领域语言的转换任务，旨在解决企业级实时分析中NL-to-DSL的高延迟和低准确问题。提出RedParrot框架，通过语义缓存加速推理，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.22758](https://arxiv.org/pdf/2604.22758)**

> **作者:** Tong Wang; Yongqin Xu; Jianfeng Zhang; Lingxi Cui; Wenqing Wei; Suzhou Chen; Huan Li; Ke Chen; Lidan Shou
>
> **摘要:** Recently, at Xiaohongshu, the rapid expansion of e-commerce and advertising demands real-time business analytics with high accuracy and low latency. To meet this demand, systems typically rely on converting natural language (NL) queries into Domain-Specific Languages (DSLs) to ensure semantic consistency, validation, and portability. However, existing multi-stage LLM pipelines for this NL-to-DSL task suffer from prohibitive latency, high cost, and error propagation, rendering them unsuitable for enterprise-scale deployment. In this paper, we propose RedParrot, a novel NL-to-DSL framework that accelerates inference via a semantic cache. Observing the high repetition and stable structural patterns in user queries, RedParrot bypasses the costly pipeline by matching new requests against cached "query skeletons" (normalized structural patterns) and adapting their corresponding DSLs. Our core technical contributions include (1) an offline skeleton construction strategy, (2) an online, entity-agnostic embedding model trained via contrastive learning for robust matching, and (3) a heterogeneous Retrieval-Augmented Generation (RAG) method that integrates diverse knowledge sources to handle unseen entities. Experiments on six real enterprise datasets from Xiaohongshu show RedParrot achieves an average 3.6x speedup and an 8.26% accuracy improvement. Furthermore, on new public benchmarks adapted from Spider and BIRD, it boosts accuracy by 34.8%, substantially outperforming standard in-context learning baselines.
>
---
#### [new 121] ShredBench: Evaluating the Semantic Reasoning Capabilities of Multimodal LLMs in Document Reconstruction
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文档重建任务，旨在解决多模态大模型在碎片化文档语义推理上的不足。提出ShredBench基准，评估模型在不同碎片化程度下的表现。**

- **链接: [https://arxiv.org/pdf/2604.23813](https://arxiv.org/pdf/2604.23813)**

> **作者:** Zichun Guo; Yuling Shi; Wenhao Zeng; Chao Hu; Haotian Lin; Terry Yue Zhuo; Jiawei Chen; Xiaodong Gu; Wenping Ma
>
> **备注:** ACL 2026 Findings. Code available at this https URL
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable performance in Visually Rich Document Understanding (VRDU) tasks, but their capabilities are mainly evaluated on pristine, well-structured document images. We consider content restoration from shredded fragments, a challenging VRDU setting that requires integrating visual pattern recognition with semantic reasoning under significant content discontinuities. To facilitate systematic evaluation of complex VRDU tasks, we introduce ShredBench, a benchmark supported by an automated generation pipeline that renders fragmented documents directly from Markdown. The proposed pipeline ensures evaluation validity by allowing the flexible integration of latest or unseen textual sources to prevent training data contamination. ShredBench assesses four scenarios (English, Chinese, Code, Table) with three fragmentation granularities (8, 12, 16 pieces). Empirical evaluations on state-of-the-art MLLMs reveal a significant performance gap: The method is effective on intact documents; however, once the document is shredded, restoration becomes a significant challenge, with NED dropping sharply as fragmentation increases. Our findings highlight that current MLLMs lack the fine-grained cross-modal reasoning required to bridge visual discontinuities, identifying a critical gap in robust VRDU research.
>
---
#### [new 122] PExA: Parallel Exploration Agent for Complex Text-to-SQL
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出PExA，解决文本到SQL生成中的延迟与性能平衡问题。通过并行测试用例提升语义覆盖，优化最终SQL生成效果。**

- **链接: [https://arxiv.org/pdf/2604.22934](https://arxiv.org/pdf/2604.22934)**

> **作者:** Tanmay Parekh; Ella Hofmann-Coyle; Shuyi Wang; Sachith Sri Ram Kothur; Srivas Prasad; Yunmo Chen
>
> **备注:** Accepted at ACL 2026
>
> **摘要:** LLM-based agents for text-to-SQL often struggle with latency-performance trade-off, where performance improvements come at the cost of latency or vice versa. We reformulate text-to-SQL generation within the lens of software test coverage where the original query is prepared with a suite of test cases with simpler, atomic SQLs that are executed in parallel and together ensure semantic coverage of the original query. After iterating on test case coverage, the final SQL is generated only when enough information is gathered, leveraging the explored test case SQLs to ground the final generation. We validated our framework on a state-of-the-art benchmark for text-to-SQL, Spider 2.0, achieving a new state-of-the-art with 70.2% execution accuracy.
>
---
#### [new 123] A Large-Scale, Cross-Disciplinary Corpus of Systematic Reviews
- **分类: cs.IR; cs.CL**

- **简介: 该论文构建了一个跨学科的大规模系统综述语料库，解决现有基准在规模和领域覆盖上的不足。通过预处理流程提取结构化方法信息，支持检索与筛选的基准测试及跨领域分析。**

- **链接: [https://arxiv.org/pdf/2604.22864](https://arxiv.org/pdf/2604.22864)**

> **作者:** Pierre Achkar; Tim Gollub; Arno Simons; Harrisen Scells; Martin Potthast
>
> **摘要:** Existing benchmarks for systematic reviewing remain limited either in scale or in disciplinary coverage, with some collections comprising only a modest number of topics and others focusing primarily on biomedical research. We present Webis-SR4ALL-26, a large-scale, cross-disciplinary corpus of 301,871 systematic reviews spanning all scientific fields as covered by OpenAlex. Using a multi-stage pre-processing pipeline, we link reviews to resolved OpenAlex metadata and reference lists and extract, when explicitly reported, structured method artifacts relevant to retrieval and screening. These artifacts include reported search strategies (Boolean queries or keyword lists) that we normalize into executable approximations, as well as reported inclusion and exclusion criteria. Together, these layers support cross-domain benchmarking of retrieval and screening components against review reference lists, training and evaluation of extraction methods for review artifacts, and comparative meta-science analyses of systematic review practices across disciplines and time. To demonstrate one concrete use case, we report large-scale baseline retrieval signals by executing normalized search strategies in OpenAlex and comparing retrieved sets to resolved reference lists. We release the corpus and the pre-processing pipeline, along with code used for extraction validation and the retrieval demonstration.
>
---
#### [new 124] All That Glitters Is Not Audio: Rethinking Text Priors and Audio Reliance in Audio-Language Evaluation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频-语言模型评估任务，旨在解决现有基准无法准确衡量音频理解的问题。通过分析文本先验和音频依赖性，发现模型依赖音频程度有限，提出改进评估方法。**

- **链接: [https://arxiv.org/pdf/2604.24401](https://arxiv.org/pdf/2604.24401)**

> **作者:** Leonardo Haw-Yang Foo; Chih-Kai Yang; Chen-An Li; Ke-Han Lu; Hung-yi Lee
>
> **备注:** 6 pages, 3 figures, 5 tables
>
> **摘要:** Large Audio-Language Models show consistent performance gains across speech and audio benchmarks, yet high scores may not reflect true auditory perception. If a model can answer questions without processing the acoustic signal, the benchmark fails as a measure of auditory understanding. We present a diagnostic framework using two axes: text prior, which measures answerability from text and general knowledge alone, and audio reliance, which assesses actual dependency on the acoustic signal. Evaluating eight LALMs across three benchmarks, we find that models retain 60-72% of their full audio scores even without any audio input. Moreover, among items that require audio, only 3.0-4.2% need the complete audio clip; the majority can be resolved using localized fragments. These findings challenge the assumption that benchmark performance equals robust audio understanding, and we conclude with practical guidelines for improving evaluation reliability and benchmark design.
>
---
#### [new 125] Beyond Static: Related Questions Retrieval Through Conversations in Community Question Answering
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于社区问答中的相关问题检索任务，旨在解决传统方法忽视对话交互的问题。提出TeCQR模型，通过对话和标签增强提升检索效果。**

- **链接: [https://arxiv.org/pdf/2604.22759](https://arxiv.org/pdf/2604.22759)**

> **作者:** Xiao Ao; Jie Zou; Yibiao Wei; Peng Wang; Weikang Guo
>
> **备注:** 9 pages. Accepted at AAAI 2026
>
> **摘要:** In community question answering (cQA) platforms like Stack Overflow, related question retrieval is recognized as a fundamental task that allows users to retrieve related questions to answer user queries automatically. Although many traditional approaches have been proposed for investigating this research field, they mostly rely on static approaches and neglect the interaction property. We argue that the conversational way can well distinguish the fine-grained representations of questions and has great potential to improve the performance of question retrieval. In this paper, we propose a related question retrieval model through conversations, called TeCQR, to locate related questions in cQA. Specifically, we build conversations by utilizing tag-enhanced clarifying questions (CQs). In addition, we design a noise tolerance model that evaluates the semantic similarity between questions and tags, enabling the model to effectively handle noisy feedback. Moreover, the tag-enhanced two-stage offline training is proposed to fully exploit the mutual relationships among user queries, questions, and tags to learn their fine-grained representations. Based on the learned representations and contextual conversations, TeCQR incorporates conversational feedback by learning to ask tag-enhanced clarifying questions to retrieve related questions more effectively. Experimental results demonstrate that our model significantly outperforms state-of-the-art baselines.
>
---
#### [new 126] Representational Curvature Modulates Behavioral Uncertainty in Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究大语言模型中的表示曲率与行为不确定性之间的关系，属于自然语言处理任务。通过分析曲率与熵的关联，揭示了模型行为的不确定性来源。**

- **链接: [https://arxiv.org/pdf/2604.23985](https://arxiv.org/pdf/2604.23985)**

> **作者:** Jack King; Evelina Fedorenko; Eghbal A. Hosseini
>
> **摘要:** In autoregressive large language models (LLMs), temporal straightening offers an account of how the next-token prediction objective shapes representations. Models learn to progressively straighten the representational trajectory of input sequences across layers, potentially facilitating next-token prediction via linear extrapolation. However, a direct link between this trajectory and token-level behavior has been missing. We provide such a link by relating contextual curvature-a geometric measure of how sharply the representational trajectory bends over recent context-to next-token entropy. Across two models (GPT-2 XL and Pythia-2.8B), contextual curvature is correlated with entropy, and this relationship emerges during training. Perturbation experiments reveal selective dependence: manipulating curvature through trajectory-aligned interventions reliably modulates entropy, while geometrically misaligned perturbations have no effect. Finally, regularizing representations to be straighter during training modestly reduces token-level entropy without degrading validation loss. These results identify trajectory curvature as a task-aligned representational feature that influences behavioral uncertainty in LLMs.
>
---
#### [new 127] RCSB PDB AI Help Desk: retrieval-augmented generation for protein structure deposition support
- **分类: cs.IR; cs.AI; cs.CL; q-bio.QM**

- **简介: 该论文属于生物信息学任务，旨在解决蛋白质结构数据提交支持问题。通过RAG技术构建AI助手，提升Help Desk效率。**

- **链接: [https://arxiv.org/pdf/2604.22800](https://arxiv.org/pdf/2604.22800)**

> **作者:** Vivek Reddy Chithari; Jasmine Y. Young; Irina Persikova; Yuhe Liang; Gregg V. Crichlow; Justin W. Flatt; Sutapa Ghosh; Brian P. Hudson; Ezra Peisach; Monica Sekharan; Chenghua Shao; Stephen K. Burley
>
> **备注:** 13 pages, 0 figures
>
> **摘要:** Motivation: Structural Biologists have contributed more than 245,000 experimentally determined three-dimensional structures of biological macromolecules to the Protein Data Bank (PDB). Incoming data are validated and biocurated by ~20 expert biocurators across the wwPDB. RCSB PDB biocurators who process more than 40% of global depositions face increasing challenges in maintaining efficient Help Desk operations, with approximately 19,000 messages in approximately 8,000 entries received from depositors in 2025. Results: We developed an AI-powered Help Desk using Retrieval-Augmented Generation (RAG) built on LangChain with a pgvector store (PostgreSQL) and GPT-4.1-mini. The system employs pymupdf4llm for Markdown-preserving PDF extraction, two-stage document chunking, Maximal Marginal Relevance retrieval, a topical guardrail that filters off-topic queries, and a specialized system prompt that prevents exposure of internal terminology. A dual-LLM architecture uses separate model configurations for question condensing and response generation. Deployed in production on Kubernetes with PostgreSQL (pgvector), it provides around-the-clock depositor assistance with citation-backed, streaming responses. Availability and implementation: Freely available at this https URL.
>
---
#### [new 128] Layerwise Convergence Fingerprints for Runtime Misbehavior Detection in Large Language Models
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于模型安全任务，解决大语言模型运行时异常检测问题。提出LCF方法，无需参考模型或触发知识，通过分析层间状态轨迹实现高效威胁检测。**

- **链接: [https://arxiv.org/pdf/2604.24542](https://arxiv.org/pdf/2604.24542)**

> **作者:** Nay Myat Min; Long H. Pham; Jun Sun
>
> **备注:** 34 pages, 5 figures. Code: this https URL
>
> **摘要:** Large language models deployed at runtime can misbehave in ways that clean-data validation cannot anticipate: training-time backdoors lie dormant until triggered, jailbreaks subvert safety alignment, and prompt injections override the deployer's instructions. Existing runtime defenses address these threats one at a time and often assume a clean reference model, trigger knowledge, or editable weights, assumptions that rarely hold for opaque third-party artifacts. We introduce Layerwise Convergence Fingerprinting (LCF), a tuning-free runtime monitor that treats the inter-layer hidden-state trajectory as a health signal: LCF computes a diagonal Mahalanobis distance on every inter-layer difference, aggregates via Ledoit-Wolf shrinkage, and thresholds via leave-one-out calibration on 200 clean examples, with no reference model, trigger knowledge, or retraining. Evaluated on four architectures (Llama-3-8B, Qwen2.5-7B, Gemma-2-9B, Qwen2.5-14B) across backdoors, jailbreaks, and prompt injection (56 backdoor combinations, 3 jailbreak techniques, and BIPIA email + code-QA), LCF reduces mean backdoor attack success rate (ASR) below 1% on Qwen2.5-7B and Gemma-2 and to 1.3% on Qwen2.5-14B, detects 92-100% of DAN jailbreaks (62-100% for GCG and softer role-play), and flags 100% of text-payload injections across all eight (model, domain) cells, at 12-16% backdoor FPR and <0.1% inference overhead. A single aggregation score covers all three threat families without threat-specific tuning, positioning LCF as a general-purpose runtime safety layer for cloud-served and on-device LLMs.
>
---
#### [new 129] Process Supervision of Confidence Margin for Calibrated LLM Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM推理中过度自信导致的幻觉问题。提出RLCM框架，通过优化信心边界提升模型校准性与可靠性。**

- **链接: [https://arxiv.org/pdf/2604.23333](https://arxiv.org/pdf/2604.23333)**

> **作者:** Liaoyaqi Wang; Chunsheng Zuo; William Jurayj; Benjamin Van Durme; Anqi Liu
>
> **摘要:** Scaling test-time computation with reinforcement learning (RL) has emerged as a reliable path to improve large language models (LLM) reasoning ability. Yet, outcome-based reward often incentivizes models to be overconfident, leading to hallucinations, unreliable confidence-based control, and unnecessary compute allocation. We introduce Reinforcement Learning with Confidence Margin (\textbf{RLCM}), a calibration-aware RL framework that jointly optimizes correctness and confidence reliability via a margin-enhanced process reward over intermediate-budget completions. Rather than aligning confidence to correctness likelihoods, RLCM encourages to widen the confidence margin between correct and incorrect steps within a single reasoning trajectory. Across mathematical, code, logic and science benchmarks, our method substantially improves calibration while maintaining or improving accuracy. We further show that, with calibrated confidence signals, the resulting models enable more efficient conformal risk control and effective confidence-weighted aggregation.
>
---
#### [new 130] Lightweight and Production-Ready PDF Visual Element Parsing
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于PDF视觉元素解析任务，旨在解决现有解析器漏检复杂图形、误提取无关元素及无法准确关联标题的问题。通过结合空间启发式、布局分析和语义相似性，提升检测与关联准确率。**

- **链接: [https://arxiv.org/pdf/2604.23276](https://arxiv.org/pdf/2604.23276)**

> **作者:** Meizhu Liu; Yassi Abbasi; Matthew Rowe; Michael Avendi; Paul Li
>
> **摘要:** PDF documents contain critical visual elements such as figures, tables, and forms whose accurate extraction is essential for document understanding and multimodal retrieval-augmented generation (RAG). Existing PDF parsers often miss complex visuals, extract non-informative artifacts (e.g., watermarks, logos), produce fragmented elements, and fail to reliably associate captions with their corresponding elements, which degrades downstream retrieval and question answering. We present a lightweight and production level PDF parsing framework that can accurately detect visual elements and associates captions using a combination of spatial heuristics, layout analysis, and semantic similarity. On popular benchmark datasets and internal product data, the proposed solution achieves $\geq96\%$ visual element detection accuracy and $93\%$ caption association accuracy. When used as a preprocessing step for multimodal RAG, it significantly outperforms state-of-the-art parsers and large vision-language models on both internal data and the MMDocRAG benchmark, while reducing latency by over $2\times$. We have deployed the proposed system in challenging production environment.
>
---
#### [new 131] FormalScience: Scalable Human-in-the-Loop Autoformalisation of Science with Agentic Code Generation in Lean
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出FormalScience，解决科学领域自动形式化问题，通过人机协作生成可验证的正式证明。**

- **链接: [https://arxiv.org/pdf/2604.23002](https://arxiv.org/pdf/2604.23002)**

> **作者:** Jordan Meadows; Lan Zhang; Andre Freitas
>
> **备注:** ACL 2026
>
> **摘要:** Formalising informal mathematical reasoning into formally verifiable code is a significant challenge for large language models. In scientific fields such as physics, domain-specific machinery (\textit{e.g.} Dirac notation, vector calculus) imposes additional formalisation challenges that modern LLMs and agentic approaches have yet to tackle. To aid autoformalisation in scientific domains, we present FormalScience; a domain-agnostic human-in-the-loop agentic pipeline that enables a single domain expert (without deep formal language experience) to produce \textit{syntactically correct} and \textit{semantically aligned} formal proofs of informal reasoning for low economic cost. Applying FormalScience to physics, we construct FormalPhysics, a dataset of 200 university-level (LaTeX) physics problems and solutions (primarily quantum mechanics and electromagnetism), along with their Lean4 formal representations. Compared to existing formal math benchmarks, FormalPhysics achieves perfect formal validity and exhibits greater statement complexity. We evaluate open-source models and proprietary systems on a statement autoformalisation task on our dataset via zero-shot prompting, self-refinement with error feedback, and a novel multi-stage agentic approach, and explore autoformalisation limitations in modern LLM-based approaches. We provide the first systematic characterisation of semantic drift in physics autoformalisation in terms of concepts such as notational collapse and abstraction elevation which reveals what formal language verifies when full semantic preservation is unattainable. We release the codebase together with an interactive UI-based FormalScience system which facilitates autoformalisation and theorem proving in scientific domains beyond this http URL://github.com/jmeadows17/formal-science
>
---
#### [new 132] Agentic clinical reasoning over longitudinal myeloma records: a retrospective evaluation against expert consensus
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医疗决策支持任务，旨在评估基于大语言模型的代理推理系统在多发性骨髓瘤长期病历中的临床推理能力，解决如何有效整合复杂病史以支持治疗决策的问题。**

- **链接: [https://arxiv.org/pdf/2604.24473](https://arxiv.org/pdf/2604.24473)**

> **作者:** Johannes Moll; Jannik Lübberstedt; Christoph Nuernbergk; Jacob Stroh; Luisa Mertens; Anna Purcarea; Christopher Zirn; Zeineb Benchaaben; Fabian Drexel; Hartmut Häntze; Anirudh Narayanan; Friedrich Puttkammer; Andrei Zhukov; Jacqueline Lammert; Sebastian Ziegelmayer; Markus Graf; Marion Högner; Marcus Makowski; Florian Bassermann; Lisa C. Adams; Jiazhen Pan; Daniel Rueckert; Krischan Braitsch; Keno K. Bressem
>
> **摘要:** Multiple myeloma is managed through sequential lines of therapy over years to decades, with each decision depending on cumulative disease history distributed across dozens to hundreds of heterogeneous clinical documents. Whether LLM-based systems can synthesise this evidence at a level approaching expert agreement has not been established. A retrospective evaluation was conducted on longitudinal clinical records of 811 myeloma patients treated at a tertiary centre (2001-2026), covering 44,962 documents and 1,334,677 laboratory values, with external validation on MIMIC-IV. An agentic reasoning system was compared against single-pass retrieval-augmented generation (RAG), iterative RAG, and full-context input on 469 patient-question pairs from 48 templates at three complexity levels. Reference labels came from double annotation by four oncologists with senior haematologist adjudication. Iterative RAG and full-context input converged on a shared ceiling (75.4% vs 75.8%, p = 1.00). The agentic system reached 79.6% concordance (95% CI 76.4-82.8), exceeding both baselines (+3.8 and +4.2 pp; p = 0.006 and 0.007). Gains rose with question complexity, reaching +9.4 pp on criteria-based synthesis (p = 0.032), and with record length, reaching +13.5 pp in the top decile (n = 10). The system error rate (12.2%) was comparable to expert disagreement (13.6%), but severity was inverted: 57.8% of system errors were clinically significant versus 18.8% of expert disagreements. Agentic reasoning was the only approach to exceed the shared ceiling, with gains concentrated on the most complex questions and longest records. The greater clinical consequence of residual system errors indicates that prospective evaluation in routine care is required before these findings translate into patient benefit.
>
---
#### [new 133] HeadRouter: Dynamic Head-Weight Routing for Task-Adaptive Audio Token Pruning in Large Audio Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频语言模型任务，解决高推理成本问题。通过动态路由注意力头，实现音频令牌压缩，提升效率。**

- **链接: [https://arxiv.org/pdf/2604.23717](https://arxiv.org/pdf/2604.23717)**

> **作者:** Peize He; Yaodi Luo; Xiaoqian Liu; Xuyang Liu; Jiahang Deng; Yaosong Du; Bangyu Li; Xiyan Gui; Yuxuan Chen; Linfeng Zhang
>
> **备注:** Homepage: this https URL
>
> **摘要:** Recent large audio language models (LALMs) demonstrate remarkable capabilities in processing extended multi-modal sequences, yet incur high inference costs. Token compression is an effective method that directly reduces redundant tokens in the sequence. Existing compression methods usually assume that all attention heads in LALMs contribute equally to various audio tasks and calculate token importance by averaging scores across all heads. However, our analysis demonstrates that attention heads exhibit distinct behaviors across diverse audio domains. We further reveal that only a sparse subset of attention heads actively responds to audio, with completely different performance when handling semantic and acoustic tasks. In light of this observation, we propose HeadRouter, a head-importance-aware token pruning method that perceives the varying importance of attention heads in different audio tasks to maximize the retention of crucial tokens. HeadRouter is training-free and can be applied to various LALMs. Extensive experiments on the AudioMarathon and MMAU-Pro benchmarks demonstrate that HeadRouter achieves state-of-the-art compression performance, exceeding the baseline model even when retaining 70% of the audio tokens and achieving 101.8% and 103.0% of the vanilla average on Qwen2.5-Omni-3B and Qwen2.5-Omni-7B, respectively.
>
---
#### [new 134] HalalBench: A Multilingual OCR Benchmark for Food Packaging Ingredient Extraction
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出HalalBench，一个用于食品包装OCR的多语言基准，解决自动化清真食品验证中的文本识别问题。工作包括构建数据集、评估OCR引擎并优化后处理算法。**

- **链接: [https://arxiv.org/pdf/2604.22754](https://arxiv.org/pdf/2604.22754)**

> **作者:** Hasan Arief
>
> **备注:** 8 pages, 6 figures, 7 tables
>
> **摘要:** No standardized benchmark exists for evaluating OCR on food packaging, despite its critical role in automated halal food verification. Existing benchmarks target documents or scene text, missing the unique challenges of ingredient labels: curved surfaces, dense multilingual text, and sub-8pt fonts. We present HalalBench, the first open multilingual benchmark for food packaging OCR, comprising 1,043 images (50 real, 993 synthetic) with 36,438 annotations in COCO format spanning 14 languages. We evaluate four engines: docTR achieves F1=0.193, ML Kit 0.180, EasyOCR 0.167, while all fail on Japanese (F1=0.000). A clustering ablation shows 36% F1 improvement from our post-processing algorithm. We validate findings through HalalLens (this https URL), a production halal scanner serving 20+ countries. Dataset and code are released under open licenses.
>
---
#### [new 135] Code Broker: A Multi-Agent System for Automated Code Quality Assessment
- **分类: cs.SE; cs.AI; cs.CL; cs.PL**

- **简介: 论文介绍Code Broker，一个用于自动化代码质量评估的多智能体系统。该系统解决代码质量分析问题，通过并行智能体实现正确性、风格等维度评估，生成报告。**

- **链接: [https://arxiv.org/pdf/2604.23088](https://arxiv.org/pdf/2604.23088)**

> **作者:** Samer Attrah
>
> **备注:** 8 pages, 1 figure, 2 tables, 28 references
>
> **摘要:** We present Code Broker, a multi agent system built with Google Agent Development Kit ADK that analyses Python code from files, local directories, or GitHub repositories and generates actionable quality assessment reports. The system employs a hierarchical five agents architecture in which a root orchestrator coordinates a sequential pipeline agent, which in turn dispatches three specialised agents in parallel a Correctness Assessor, a Style Assessor, and a Description Generator before synthesising findings through an Improvement Recommender. Reports score four dimensions correctness, security, style, and maintainability and are rendered in both Markdown and HTML. Code Broker combines LLM based reasoning with deterministic static-analysis signals from Pylint, uses asynchronous execution with retry logic to improve robustness, and explores lightweight session memory for retaining and querying prior assessment context. We position the paper as a technical report on system design and prompt or tool orchestration, and present a preliminary qualitative evaluation on representative Python codebases. The results suggest that parallel specialised agents produce readable, developer oriented feedback, while also highlighting current limitations in evaluation depth, security tooling, large repository handling, and the current use of only in memory persistence. All code and reproducibility materials are available at: this https URL.
>
---
#### [new 136] Can Humans Detect AI? Mining Textual Signals of AI-Assisted Writing Under Varying Scrutiny Conditions
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于AI写作检测任务，探讨AI辅助写作是否可被识别。通过实验研究发现，受警告的作者文本更常被误判为人类撰写，但文本特征无差异，表明检测依赖非特征因素。**

- **链接: [https://arxiv.org/pdf/2604.23471](https://arxiv.org/pdf/2604.23471)**

> **作者:** Daniel Tabach
>
> **备注:** 25 pages, 12 figures
>
> **摘要:** This study asks whether the threat of AI detection changes how people write with AI, and whether other people can tell the difference. In a two-phase controlled experiment, 21 participants wrote opinion pieces on remote work using an AI chatbot. Half were randomly warned that their submission would be scanned by an AI detection tool. The other half received no warning. Both groups had access to the same chatbot. In Phase 2, 251 independent judges evaluated 1,999 paired comparisons, each time choosing which document in the pair was written by a human. Judges were not told that both writers had access to AI. Across all evaluations, judges selected the warned writer's document as human 54.13% of the time versus 45.87% for the unwarned writer. A two-sided binomial test rejects chance guessing at p = 0.000243, and the result holds across both writing stances. Yet on every measurable text feature extracted, including AI overlap scores, lexical diversity, sentence structure, and pronoun usage, the two groups were indistinguishable. The judges are picking up on something that feature-based methods do not capture.
>
---
#### [new 137] The Collapse of Heterogeneity in Silicon Philosophers
- **分类: cs.CY; cs.CL; cs.LG**

- **简介: 该论文属于人工智能对齐研究，探讨语言模型在哲学领域中过度一致的问题。通过实验发现模型在哲学判断上产生虚假共识，影响评估准确性。**

- **链接: [https://arxiv.org/pdf/2604.23575](https://arxiv.org/pdf/2604.23575)**

> **作者:** Yuanming Shi; Andreas Haupt
>
> **摘要:** Silicon samples are increasingly used as a low-cost substitute for human panels and have been shown to reproduce aggregate human opinion with high fidelity. We show that, in the alignment-relevant domain of philosophy, silicon samples systematically collapse heterogeneity. Using data from $N = {277}$ professional philosophers drawn from PhilPeople profiles, we evaluate seven proprietary and open-source large language models on their ability to replicate individual philosophical positions and to preserve cross-question correlation structures across philosophical domains. We find that language models substantially over-correlate philosophical judgments, producing artificial consensus across domains. This collapse is associated in part with specialist effects, whereby models implicitly assume that domain specialists hold highly similar philosophical views. We assess the robustness of these findings by studying the impact of DPO fine-tuning and by validating results against the full PhilPapers 2020 Survey ($N = {1785}$). We conclude by discussing implications for alignment, evaluation, and the use of silicon samples as substitutes for human judgment. The code of this project can be found at this https URL.
>
---
#### [new 138] The Limits of Artificial Companionship
- **分类: cs.CY; cs.CL; cs.HC**

- **简介: 论文探讨数字陪伴中的商业与非商业对话界限，旨在解决 conversational advertising 带来的伦理问题。属于伦理与法律研究任务，提出应明确区分商业与非商业对话以保护用户自主性。**

- **链接: [https://arxiv.org/pdf/2604.23601](https://arxiv.org/pdf/2604.23601)**

> **作者:** Mauricio Figueroa
>
> **备注:** Southwestern Journal of International Law (2026, forthcoming)
>
> **摘要:** This Article argues that conversations with companion chatbot should be subject to a clear structural distinction between commercial and non-commercial contexts. The insertion of undisclosed promotional content into affective or relational exchanges should be prohibited, as it collapses the boundary between market transaction and communicative intimacy in ways that erode user autonomy and conversational context. The Article begins by theorizing digital companionship as a sociotechnical form that reconfigures intimacy, dependence and relational vulnerability. It then introduces the potential economic harms derived from conversational advertising. The Article ultimately argues for a firm legal and social distinction between commercial and non-commercial conversational contexts as a precondition for the responsible stabilization of these technologies within social life.
>
---
#### [new 139] Graph Memory Transformer (GMT)
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Graph Memory Transformer (GMT)，用于替代Transformer中的FFN子层，通过记忆图实现更结构化的语言建模，解决传统模型可解释性不足的问题。**

- **链接: [https://arxiv.org/pdf/2604.23862](https://arxiv.org/pdf/2604.23862)**

> **作者:** Nicola Zanarini; Niccolò Ferrari
>
> **备注:** 65 pages, 10 figures, 5 tables. Code available at this https URL
>
> **摘要:** We investigate whether the Feed-Forward Network (FFN) sublayer in a decoder-only transformer can be replaced by an explicit learned memory graph while preserving the surrounding autoregressive architecture. The proposed Graph Memory Transformer (GMT) keeps causal self-attention intact, but replaces the usual per-token FFN transformation with a memory cell that routes token representations over a learned bank of centroids connected by a learned directed transition matrix. In the base GMT v7 instantiation studied here, each of 16 transformer blocks contains 128 centroids, a 128 * 128 edge matrix, gravitational source routing, token-conditioned target selection, and a gated displacement readout. The cell therefore returns movement from an estimated source memory state toward a target memory state, rather than a retrieved value. The resulting model is a fully decoder-only language model with 82.2M trainable parameters and no dense FFN sublayers, compared with a 103.0M-parameter dense GPT-style baseline used in the evaluation. The base v7 model trains stably and exposes centroid usage, transition structure, and source-to-target movement as directly inspectable quantities of the forward computation. It remains behind the larger dense baseline in validation loss and perplexity (3.5995/36.58 vs. 3.2903/26.85), while showing close zero-shot benchmark behavior under the evaluated setting. These results are not intended as a state-of-the-art claim; they support the viability and structural interpretability of replacing dense within-token transformation with graph-mediated memory navigation. Broader scaling, optimized kernels, and more extensive benchmark evaluation are left for subsequent work.
>
---
#### [new 140] Evolve: A Persistent Knowledge Lifecycle for Small Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Evolve系统，用于增强小语言模型的知识生命周期管理。针对知识更新与存储效率问题，通过教师模型构建语义一致的知识库，并实现知识复用与压缩，提升模型准确率并减少教师调用。**

- **链接: [https://arxiv.org/pdf/2604.23424](https://arxiv.org/pdf/2604.23424)**

> **作者:** Dikran Hovagimian
>
> **备注:** 35 pages, 1 figure. Code and evaluation data: this https URL
>
> **摘要:** Evolve pairs a small local language model with a persistent, teacher-compiled knowledge store -- refined through sleep consolidation and usage-driven refresh -- to deliver substantial accuracy gains over the model's parametric baseline while amortizing teacher costs through cross-query knowledge reuse. Rather than retrieving document fragments at query time, Evolve constructs a store of semantically coherent sections compiled by teacher models at natural conceptual boundaries; new sections are staged on acquisition, consolidated offline through teacher-mediated merging, and refreshed inline when expired. A 2B-parameter local model handles classification and generation; large teacher models are invoked only for knowledge operations. Across 750 benchmark queries spanning custom specialist questions, NaturalQuestions, and TriviaQA, the 2B model augmented by Evolve improves from 20-33% baseline accuracy to 60-84% (+40-52pp) while reducing teacher invocations by over 50% through reuse. Post-consolidation compresses the knowledge store by 31-33.5% across three independent benchmarks while preserving accuracy; section-based retrieval outperforms chunk-based retrieval by 5-9pp across every lifecycle condition. The architecture supports two generation modes over the same lifecycle -- suppress (strict section-only grounding, auditable) and augment (section-supplemented responses).
>
---
#### [new 141] A Survey on Split Learning for LLM Fine-Tuning: Models, Systems, and Privacy Optimizations
- **分类: cs.CR; cs.CL; cs.DC; cs.LG**

- **简介: 该论文属于自然语言处理中的模型优化任务，旨在解决资源受限环境下大语言模型微调的计算成本高和数据隐私问题。通过综述分裂学习方法，提出统一训练流程，分析模型、系统和隐私三个维度的优化策略。**

- **链接: [https://arxiv.org/pdf/2604.24468](https://arxiv.org/pdf/2604.24468)**

> **作者:** Zihan Liu; Yizhen Wang; Rui Wang; Xiu Tang; Sai Wu
>
> **摘要:** Fine-tuning unlocks large language models (LLMs) for specialized applications, but its high computational cost often puts it out of reach for resource-constrained organizations. While cloud platforms could provide the needed resources, data privacy concerns make sharing sensitive information with third parties risky. A promising solution is split learning for LLM fine-tuning, which divides the model between clients and a server, allowing collaborative and secure training through exchanged intermediate data, thus enabling resource-constrained participants to adapt LLMs safely. % In light of this, a growing body of literature has emerged to advance this paradigm, introducing varied model methods, system optimizations, and privacy defense-attack techniques for split learning. To bring clarity and direction to the field, a comprehensive survey is needed to classify, compare, and critique these diverse approaches. This paper fills the gap by presenting the first extensive survey dedicated to split learning for LLM fine-tuning. We propose a unified, fine-grained training pipeline to pinpoint key operational components and conduct a systematic review of state-of-the-art work across three core dimensions: model-level optimization, system-level efficiency, and privacy preservation. Through this structured taxonomy, we establish a foundation for advancing scalable, robust, and secure collaborative LLM adaptation.
>
---
#### [new 142] The Power of Power Law: Asymmetry Enables Compositional Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，研究如何通过幂律分布数据提升模型的组合推理能力。针对数据分布对模型性能的影响问题，通过实验和理论分析证明幂律分布优于均匀分布。**

- **链接: [https://arxiv.org/pdf/2604.22951](https://arxiv.org/pdf/2604.22951)**

> **作者:** Zixuan Wang; Xingyu Dang; Jason D. Lee; Kaifeng Lyu
>
> **摘要:** Natural language data follows a power-law distribution, with most knowledge and skills appearing at very low frequency. While a common intuition suggests that reweighting or curating data towards a uniform distribution may help models better learn these long-tail skills, we find a counterintuitive result: across a wide range of compositional reasoning tasks, such as state tracking and multi-step arithmetic, training under power-law distributions consistently outperforms training under uniform distributions. To understand this advantage, we introduce a minimalist skill-composition task and show that learning under a power-law distribution provably requires significantly less training data. Our theoretical analysis reveals that power law sampling induces a beneficial asymmetry that improves the pathological loss landscape, which enables models to first acquire high-frequency skill compositions with low data complexity, which in turn serves as a stepping stone to efficiently learn rare long-tailed skills. Our results offer an alternative perspective on what constitutes an effective data distribution for training models.
>
---
#### [new 143] KARL: Mitigating Hallucinations in LLMs via Knowledge-Boundary-Aware Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型幻觉问题。通过KARL框架，动态调整模型的回避行为，提升准确性和减少幻觉。**

- **链接: [https://arxiv.org/pdf/2604.22779](https://arxiv.org/pdf/2604.22779)**

> **作者:** Cheng Gao; Cheng Huang; Kangyang Luo; Ziqing Qiao; Shuzheng Si; Huimin Chen; Chaojun Xiao; Maosong Sun
>
> **备注:** 21 pages, 8 figures
>
> **摘要:** Enabling large language models (LLMs) to appropriately abstain from answering questions beyond their knowledge is crucial for mitigating hallucinations. While existing reinforcement learning methods foster autonomous abstention, they often compromise answer accuracy because their static reward mechanisms, agnostic to models' knowledge boundaries, drive models toward excessive caution. In this work, we propose KARL, a novel framework that continuously aligns an LLM's abstention behavior with its evolving knowledge boundary. KARL introduces two core innovations: a Knowledge-Boundary-Aware Reward that performs online knowledge boundary estimation using within-group response statistics, dynamically rewarding correct answers or guided abstention; and a Two-Stage RL Training Strategy that first explores the knowledge boundary and bypasses the "abstention trap", and subsequently converts incorrect answers beyond the knowledge boundary into abstentions without sacrificing accuracy. Extensive experiments on multiple benchmarks demonstrate that KARL achieves a superior accuracy-hallucination trade-off, effectively suppressing hallucinations while maintaining high accuracy across both in-distribution and out-of-distribution scenarios.
>
---
#### [new 144] Spectro-Temporal Modulation Representation Framework for Human-Imitated Speech Detection
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音伪造检测任务，旨在解决人类仿声语音难以检测的问题。通过构建基于听觉感知的时频调制框架，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.23241](https://arxiv.org/pdf/2604.23241)**

> **作者:** Khalid Zaman; Masashi Unoki
>
> **摘要:** Human-imitated speech poses a greater challenge than AI-generated speech for both human listeners and automatic detection systems. Unlike AI-generated speech, which often contains artifacts, over-smoothed spectra, or robotic cues, imitated speech is produced naturally by humans, thereby preserving a higher degree of naturalness that makes imitation-based speech forgery significantly more challenging to detect using conventional acoustic or cepstral features. To overcome this challenge, this study proposes an auditory perception-based Spectro-Temporal Modulation (STM) representation framework for human-imitated speech detection. The STM representations are derived from two cochlear filterbank models: the Gammatone Filterbank (GTFB), which simulates frequency selectivity and can be regarded as a first approximation of cochlear filtering, and the Gammachirp Filterbank (GCFB), which further models both frequency selectivity and level-dependent asymmetry. These STM representations jointly capture temporal and spectral fluctuations in speech signals, corresponding to changes over time in the spectrogram and variations along the frequency axis related to human auditory perception. We also introduce a Segmental-STM representation to analyze short-term modulation patterns across overlapping time windows, enabling high-resolution modeling of temporal speech variations. Experimental results show that STM representations are effective for human-imitated speech detection, achieving accuracy levels close to those of human listeners. In addition, Segmental-STM representations are more effective, surpassing human perceptual performance. The findings demonstrate that perceptually inspired spectro-temporal modeling is promising for detecting imitation-based speech attacks and improving voice authentication robustness.
>
---
#### [new 145] The Kerimov-Alekberli Model: An Information-Geometric Framework for Real-Time System Stability
- **分类: cs.AI; cs.CL; cs.CR**

- **简介: 该论文提出Kerimov-Alekberli模型，将非平衡热力学与随机控制结合，用于实时检测系统异常，解决AI安全问题。**

- **链接: [https://arxiv.org/pdf/2604.24083](https://arxiv.org/pdf/2604.24083)**

> **作者:** Hikmat Karimov; Rahid Zahid Alekberli
>
> **摘要:** This study introduces the Kerimov-Alekberli model, a novel information-geometric framework that redefines AI safety by formally linking non-equilibrium thermodynamics to stochastic control for the ethical alignment of autonomous systems. By establishing a formal isomorphism between non-equilibrium thermodynamics and stochastic control, we define systemic anomalies as deviations from a Riemannian manifold. The model utilizes the Kullback-Leibler divergence as the primary metric, governed by a dynamic threshold derived from the Fisher Information Metric. We further ground this framework in the Landauer Principle, proving that adversarial perturbations perform measurable physical work by increasing the system's informational entropy. Validation on the NSL-KDD dataset and unmanned aerial vehicle trajectory simulations demonstrated that our model achieves effective real-time detection via the FPT trigger, with strong performance metrics (e.g., high accuracy and low FPR) on benchmark datasets. This study provides a rigorous physical foundation for AI safety, transitioning from heuristic, rule-based ethical frameworks to a thermodynamics-based stability paradigm by grounding ethical violations in quantifiable physical work and entropic information.
>
---
#### [new 146] Talker-T2AV: Joint Talking Audio-Video Generation with Autoregressive Diffusion Modeling
- **分类: cs.CV; cs.CL; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于语音驱动的视频生成任务，旨在解决跨模态一致性问题。提出Talker-T2AV框架，通过共享骨干和专用解码器分离高、低层次信息，提升唇同步与质量。**

- **链接: [https://arxiv.org/pdf/2604.23586](https://arxiv.org/pdf/2604.23586)**

> **作者:** Zhen Ye; Xu Tan; Aoxiong Yin; Hongzhan Lin; Guangyan Zhang; Peiwen Sun; Yiming Li; Chi-Min Chan; Wei Ye; Shikun Zhang; Wei Xue
>
> **摘要:** Joint audio-video generation models have shown that unified generation yields stronger cross-modal coherence than cascaded approaches. However, existing models couple modalities throughout denoising via pervasive attention, treating high-level semantics and low-level details in a fully entangled manner. This is suboptimal for talking head synthesis: while audio and facial motion are semantically correlated, their low-level realizations (acoustic signals and visual textures) follow distinct rendering processes. Enforcing joint modeling across all levels causes unnecessary entanglement and reduces efficiency. We propose Talker-T2AV, an autoregressive diffusion framework where high-level cross-modal modeling occurs in a shared backbone, while low-level refinement uses modality-specific decoders. A shared autoregressive language model jointly reasons over audio and video in a unified patch-level token space. Two lightweight diffusion transformer heads decode the hidden states into frame-level audio and video latents. Experiments on talking portrait benchmarks show Talker-T2AV outperforms dual-branch baselines in lip-sync accuracy, video quality, and audio quality, achieving stronger cross-modal consistency than cascaded pipelines.
>
---
#### [new 147] Quantifying Divergence in Inter-LLM Communication Through API Retrieval and Ranking
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于多大语言模型协作任务，旨在解决LLM间API调用与排序的分歧问题。通过构建基准框架，量化模型在不同任务中的差异，分析其可靠性与一致性。**

- **链接: [https://arxiv.org/pdf/2604.22760](https://arxiv.org/pdf/2604.22760)**

> **作者:** Eyhab Al-Masri
>
> **备注:** AAAI 2026 Conference (LAMAS Workshop)
>
> **摘要:** Large language models (LLMs) increasingly operate as autonomous agents that reason over external APIs to perform complex tasks. However, their reliability and agreement remain poorly characterized. We present a unified benchmarking framework to quantify inter-LLM divergence, defined as the extent to which models differ in API discovery and ranking under identical tasks. Across 15 canonical API domains and 5 major model families, we measure pairwise and group-level agreement using set-, rank-, and consensus-based metrics including Average Overlap, Jaccard similarity, Rank-Biased Overlap, Kendall's tau, Kendall's W, and Cronbach's alpha. Results show moderate overall alignment (AO about 0.50, tau about 0.45) but strong domain dependence: structured tasks (Weather, Speech-to-Text) are stable, while open-ended tasks (Sentiment Analysis) exhibit substantially higher divergence. Volatility and consensus analyses reveal that coherence clusters around data-bound domains and degrades for abstract reasoning tasks. These insights enable reliability-aware orchestration in multi-agent systems, where consensus weighting can improve coordination among heterogeneous LLMs. Beyond performance benchmarking, our results reveal systematic failure modes in multi-agent LLM coordination, where apparent agreement can mask instability in action-relevant rankings. This hidden divergence poses a pre-deployment safety risk and motivates diagnostic benchmarks for early detection.
>
---
#### [new 148] In-Sync: Adaptation of Speech Aware Large Language Models for ASR with Word Level Timestamp Predictions
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音识别任务，解决ASR中时间戳预测问题。通过改进模型，直接预测单词时间戳，提升对齐精度和整体识别效果。**

- **链接: [https://arxiv.org/pdf/2604.22817](https://arxiv.org/pdf/2604.22817)**

> **作者:** Xulin Fan; Vishal Sunder; Samuel Thomas; Mark Hasegawa-Johnson; Brian Kingsbury; George Saon
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Recent advances in speech-aware language models have coupled strong acoustic encoders with large language models, enabling systems that move beyond transcription to produce richer outputs. Among these, word-level timestamp prediction is critical for applications such as captioning, media search, and multimodal synchronization, yet it is often handled by external alignment tools. In this work, we extend an existing speech-aware language model to predict timestamps directly alongside transcripts. We introduce a set of novel lightweight training strategies that improve alignment robustness while preserving recognition quality. Experiments across multiple datasets show that these strategies not only enhance timestamp accuracy, but also yield gains in overall ASR performance. Together, they demonstrate an efficient and unified approach to speech recognition with precise timestamp prediction.
>
---
#### [new 149] Secure On-Premise Deployment of Open-Weights Large Language Models in Radiology: An Isolation-First Architecture with Prospective Pilot Evaluation
- **分类: cs.CY; cs.CL**

- **简介: 该论文提出一种隔离优先的本地部署架构，解决放射学中使用开源大语言模型的合规与安全问题，通过试点评估验证其临床实用性与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.22768](https://arxiv.org/pdf/2604.22768)**

> **作者:** Sebastian Nowak; Jann-Frederick Laß; Narine Mesropyan; Babak Salam; Nico Piel; Mohammed Bahaaeldin; Wolfgang Block; Alois Martin Sprinkart; Julian Alexander Luetkens; Benjamin Wulff; Alexander Isaak
>
> **备注:** 39 pages, 4 figures, 3 tables
>
> **摘要:** Purpose: To design, implement, evaluate, and report on the regulatory requirements of a self-hosted LLM infrastructure for radiology adhering to the principle of least privilege, emphasizing technical feasibility, network isolation, and clinical utility. Materials and Methods: The isolation-first, containerized LLM inference stack relies on strict network segmentation, host-enforced egress filtering, and active isolation monitoring preventing unauthorized external connectivity. An accompanying deployment package provides automated isolation and hardening tests. The system served the open-weights DeepSeek-R1 model via vLLM. In a one-week pilot phase, 22 residents and radiologists were free to use 10 predefined prompt-templates whenever they considered them useful in daily work. Afterward, they rated clinical utility and system stability on an 0-10 Likert scale and reported observed critical errors in model output. Results: The applied institutional governance pathway achieved approval from clinic management, compliance, data protection and information security officers for processing unanonymized PHI. The system was rated stable and user friendly during the pilot. Source text-anchored tasks, such as report corrections or simplifications, and radiology guideline recommendations received the highest utility ratings, whereas open-ended conclusion generation based on findings resulted in the highest frequency of critical errors, such as clinically relevant hallucinations or omissions. Conclusion: The proposed isolation-first on-premise architecture enabled overcoming regulatory borders, showed promising clinical utility in text-anchored tasks and is the current base to serve open-weights LLMs as an official service of a German University Hospital with over 10,000 employees. The deployment package were made publicly available (this https URL).
>
---
#### [new 150] When Does Removing LayerNorm Help? Activation Bounding as a Regime-Dependent Implicit Regularizer
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究动态Tanh（DyT）移除层归一化的效果，探讨其作为隐式正则化的任务。旨在解决模型训练中不同规模下的性能变化问题，通过实验和干预验证激活边界的作用机制。**

- **链接: [https://arxiv.org/pdf/2604.23434](https://arxiv.org/pdf/2604.23434)**

> **作者:** Lucky Verma
>
> **备注:** 28 pages, 7 figures, includes appendices. Code and artifacts: this https URL
>
> **摘要:** Dynamic Tanh (DyT) removes LayerNorm by bounding activations with a learned tanh(alpha x). We show that this bounding is a regime-dependent implicit regularizer, not a uniformly beneficial replacement. Across GPT-2-family models spanning 64M to 3.78B parameters and 1M to 118M tokens, with Llama and ViT cross-checks, DyT improves validation loss by 27.3% at 64M/1M but worsens it by 18.8% at 64M/118M; the 1M benefit vanishes with capacity (+1.7% at 3.78B), while the 118M penalty reaches +27.9%. The mechanism is measurable: 49% of DyT activations saturate at 1M versus 23% at 118M, and a 500-step saturation heuristic classifies DyT's sign with 75% raw in-sample accuracy on the 12-cell GPT-2 calibration set (AUC 0.75; 64% when adding Scale 5 stress cells), correctly labels 3/3 Llama checks, but only reaches 50% raw leave-one-scale-out accuracy. Three interventions support the bounding explanation: HardTanh reproduces the regime pattern, increasing alpha at 118M monotonically reduces DyT's penalty, and vanilla+dropout(p=0.5) matches DyT's data-rich loss. We also localize Llama-DyT collapse to SwiGLU gating, where saturation separates collapse from convergence in a 3-seed component ablation (r=0.94). Scope: all experiments are compute-limited (T/P < 1.84), below Chinchilla-optimal training.
>
---
#### [new 151] Training a General Purpose Automated Red Teaming Model
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于红队测试任务，旨在提升模型在未知攻击目标上的泛化能力。工作包括设计通用红队训练流程，无需预设评估器，显著增强模型生成攻击的能力。**

- **链接: [https://arxiv.org/pdf/2604.23067](https://arxiv.org/pdf/2604.23067)**

> **作者:** Aishwarya Padmakumar; Leon Derczynski; Traian Rebedea; Christopher Parisien
>
> **摘要:** Automated methods for red teaming LLMs are an important tool to identify LLM vulnerabilities that may not be covered in static benchmarks, allowing for more thorough probing. They can also adapt to each specific LLM to discover weaknesses unique to it. Most current automated red teaming methods are intended for tackling safety and content moderation. Thus, they make use of content safety models as evaluators and optimize for circumventing them, and as such, have not been tested with other adversarial intents not typically captured by these. We propose a pipeline for training a red teaming model that can generalize to arbitrary adversarial goals, including objectives it has not been directly trained on, and that does not depend on the existence of a pre-existing evaluator available at training time. We demonstrate that finetuning small models, such as Qwen3-8B, using this pipeline results in a substantial improvement in their ability to generate attacks for both in and out of domain adversarial goals.
>
---
#### [new 152] Lost in Decoding? Reproducing and Stress-Testing the Look-Ahead Prior in Generative Retrieval
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于信息检索任务，解决生成式检索中因早期剪枝导致的检索失效问题。通过复现与测试PAG方法，验证其有效性并分析其在查询变化下的稳定性。**

- **链接: [https://arxiv.org/pdf/2604.23396](https://arxiv.org/pdf/2604.23396)**

> **作者:** Kidist Amde Mekonnen; Yongkang Li; Yubao Tang; Simon Lupart; Maarten de Rijke
>
> **备注:** 12 pages, 5 figures, 9 tables; accepted to the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval, July 20-24, 2026, Melbourne/Naarm, Australia
>
> **摘要:** Generative retrieval (GR) ranks documents by autoregressively generating document identifiers. Because many GR methods rely on trie-constrained beam search, they are vulnerable to early pruning of relevant prefixes under finite-beam decoding. Planning Ahead in Generative Retrieval (PAG) mitigates this failure mode by using simultaneous decoding to compute a document-level look-ahead prior that guides subsequent sequential decoding. We reproduce PAG at inference time and stress-test its decoding behavior. Using the authors' released checkpoint and identifier/trie artifacts under the reported decoding setup, we reproduce the main effectiveness results on MS MARCO Dev and TREC-DL 2019/2020, and corroborate the reported beam-size-latency trade-off in our hardware setting. Beyond reproduction, we introduce plan drift diagnostics that quantify how intent-preserving query variations alter the planner's top-n candidate set and highest-weight planner tokens, and how these changes affect guided decoding. We find that PAG's planning signal is brittle under lexical surface-form variation: intent-preserving typos can trigger plan collapse, where the planned candidate pool shifts enough that the look-ahead bonus provides little useful guidance, effectively reverting decoding toward weaker unguided search. We further evaluate fixed-index cross-lingual robustness using non-English mMARCO queries against an English index, and assess query-side mitigation strategies that require no re-indexing; query translation provides the strongest recovery in our setting. Overall, our results confirm PAG's reported effectiveness and the benefit of planning-guided decoding under the released inference setup, while showing that these gains depend on the stability of the planning signal under realistic query variation and query-document mismatch.
>
---
#### [new 153] EgoDyn-Bench: Evaluating Ego-Motion Understanding in Vision-Centric Foundation Models for Autonomous Driving
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于自主驾驶任务，旨在解决视觉基础模型在自我运动理解上的物理逻辑不足问题。通过构建基准测试，发现模型在视觉与物理推理耦合上存在缺陷，并提出显式轨迹编码以提升一致性。**

- **链接: [https://arxiv.org/pdf/2604.22851](https://arxiv.org/pdf/2604.22851)**

> **作者:** Finn Rasmus Schäfer; Yuan Gao; Dingrui Wang; Thomas Stauner; Stephan Günnemann; Mattia Piccinini; Sebastian Schmidt; Johannes Betz
>
> **备注:** 36 Pages, under review
>
> **摘要:** While Vision-Language Models (VLMs) have advanced highlevel reasoning in autonomous driving, their ability to ground this reasoning in the underlying physics of ego-motion remains poorly understood. We introduce EgoDyn-Bench, a diagnostic benchmark for evaluating the semantic ego-motion understanding of vision-centric foundation models. By mapping continuous vehicle kinematics to discrete motion concepts via a deterministic oracle, we decouple a model's internal physical logic from its visual perception. Our large-scale empirical audit spanning 20 + models, including closed-source MLLMs, open-source VLMs across multiple scales, and specialized VLAs, identifies a significant Perception Bottleneck: while models exhibit logical physical concepts, they consistently fail to accurately align them with visual observations, frequently underperforming classical non-learned geometric baselines. This failure persists across model scales and domain-specific training, indicating a structural deficit in how current architectures couple visual perception with physical reasoning. We demonstrate that providing explicit trajectory encodings substantially restores physical consistency across all evaluated models, revealing a functional disentanglement between vision and language: egomotion logic is derived almost exclusively from the language modality, while visual observations contribute negligible additional signal. This structural finding provides a standardized diagnostic framework and a practical pathway toward physically aligned embodied AI. Keywords: Ego-motion - Physical Reasoning - Foundation Models
>
---
#### [new 154] Jailbreaking Frontier Foundation Models Through Intention Deception
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文研究大模型的安全漏洞，针对其意图欺骗问题提出多轮攻击方法，揭示了新型漏洞“para-jailbreaking”，并验证了攻击效果。**

- **链接: [https://arxiv.org/pdf/2604.24082](https://arxiv.org/pdf/2604.24082)**

> **作者:** Xinhe Wang; Katia Sycara; Yaqi Xie
>
> **备注:** Accepted at CVPR 2026 Findings Track
>
> **摘要:** Large (vision-)language models exhibit remarkable capability but remain highly susceptible to jailbreaking. Existing safety training approaches aim to have the model learn a refusal boundary between safe and unsafe, based on the user's intent. It has been found that this binary training regime often leads to brittleness, since the user intent cannot reliably be evaluated, especially if the attacker obfuscates their intent, and also makes the system seem unhelpful. In response, frontier models, such as GPT-5, have shifted from refusal-based safeguards to safe completion, that aims to maximize helpfulness while obeying safety constraints. However, safe completion could be exploited when a user pretends their intention is benign. Specifically, this intent inversion would be effective in multi-turn conversation, where the attacker has multiple opportunities to reinforce their deceptively benign intent. In this work, we introduce a novel multi-turn jailbreaking method that exploits this vulnerability. Our approach gradually builds conversational trust by simulating benign-seeming intentions and by exploiting the consistency property of the model, ultimately guiding the target model toward harmful, detailed outputs. Most crucially, our approach also uncovered an additional class of model vulnerability that we call para-jailbreaking that has been unnoticed up to now. Para-jailbreaking describes the situation where the model may not reveal harmful direct reply to the attack query, however the information that it reveals is nevertheless harmful. Our contributions are threefold. First, it achieves high success rates against frontier models including GPT-5-thinking and Claude-Sonnet-4.5. Second, our approach revealed and addressed para-jailbreaking harmful output. Third, experiments on multimodal VLM models showed that our approach outperformed state-of-the-art models.
>
---
#### [new 155] AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出AgentEval，用于评估智能体工作流中的错误传播，通过DAG结构提升故障检测与根因分析的准确性。**

- **链接: [https://arxiv.org/pdf/2604.23581](https://arxiv.org/pdf/2604.23581)**

> **作者:** Dongxin Guo; Jikun Wu; Siu Ming Yiu
>
> **备注:** Accepted at ACL 2026 Industry Track. 14 pages, 3 figures, 21 tables
>
> **摘要:** Agentic systems that chain reasoning, tool use, and synthesis into multi-step workflows are entering production, yet prevailing evaluation practices like end-to-end outcome checks and ad-hoc trace inspection systematically mask the intermediate failures that dominate real-world error budgets. We present AgentEval, a framework that formalizes agent executions as evaluation directed acyclic graphs (DAGs), where each node carries typed quality metrics assessed by a calibrated LLM judge (GPT-4o), classified through a hierarchical failure taxonomy (3 levels, 21 subcategories), and linked to upstream dependencies for automated root cause attribution. An ablation study isolates the impact of DAG-based dependency modeling: it alone contributes +22 percentage points to failure detection recall and +34 pp to root cause accuracy over flat step-level evaluation with identical judges and rubrics. Across three production workflows (450 test cases, two agent model families, predominantly sequential architectures with a 12% non-DAG trace rate), AgentEval achieves 2.17x higher failure detection recall than end-to-end evaluation (0.89 vs. 0.41), Cohen's kappa = 0.84 agreement with human experts, and 72% root cause accuracy against an 81% human ceiling. Cross-system evaluation on tau-bench and SWE-bench traces confirms transferability (failure detection recall >= 0.78) without taxonomy or rubric modification. A 4-month pilot with 18 engineers detected 23 pre-release regressions through CI/CD-integrated regression testing, reducing median root-cause identification time from 4.2 hours to 22 minutes and driving measurable failure rate reductions in two workflows.
>
---
#### [new 156] STELLAR-E: a Synthetic, Tailored, End-to-end LLM Application Rigorous Evaluator
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出STELLAR-E，解决LLM评估数据集生成难题，通过自动化方法生成高质量合成数据，支持多领域、多语言评估。**

- **链接: [https://arxiv.org/pdf/2604.24544](https://arxiv.org/pdf/2604.24544)**

> **作者:** Alessio Sordo; Lingxiao Du; Meeka-Hanna Lenisa; Evgeny Bogdanov; Maxim Romanovsky
>
> **摘要:** The increasing reliance on Large Language Models (LLMs) across diverse sectors highlights the need for robust domain-specific and language-specific evaluation datasets; however, the collection of such datasets is challenging due to privacy concerns, regulatory restrictions, and the time cost for manual creation. Existing automated benchmarking methods are often limited by relying on pre-existing data, poor scalability, single-domain focus, and lack of multilingual support. We present STELLAR-E - a fully automated system to generate high-quality synthetic datasets of custom size, using minimal human inputs without depending on existing datasets. The system is structured in two stages: (1) We modify the TGRT Self-Instruct framework to create a synthetic data engine that enables controllable, custom synthetic dataset generation, and (2) an evaluation pipeline incorporating statistical and LLM-based metrics to assess the applicability of the synthetic dataset for LLM-based application evaluations. The synthetic datasets reach an average difference of +5.7% in terms of LLM-as-a-judge scores against existing language-specific benchmarks, demonstrating comparable quality for comprehensive assessment of big and small LLMs. While real datasets remain slightly more challenging for LLMs especially for smaller models, this work establishes a scalable and domain-adaptable benchmarking framework that supports fair evaluation of LLM applications, offering a faster alternative to manual approaches and enabling high-efficiency automated quality assurance cycles.
>
---
#### [new 157] Less Is More: Engineering Challenges of On-Device Small Language Model Integration in a Mobile Application
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 论文探讨在移动应用中集成小型语言模型的工程挑战，解决如何实现离线、私密AI的问题。通过案例研究，分析失败原因并提出解决方案，总结出八条设计准则。**

- **链接: [https://arxiv.org/pdf/2604.24636](https://arxiv.org/pdf/2604.24636)**

> **作者:** William Oliveira
>
> **备注:** 28 pages, 8 tables, 17 references
>
> **摘要:** On-device Small Language Models (SLMs) promise fully offline, private AI experiences for mobile users (no cloud dependency, no data leaving the device). But is this promise achievable in practice? This paper presents a longitudinal practitioner case study documenting the engineering challenges of integrating SLMs (Gemma 4 E2B, 2.6B parameters; Qwen3 0.6B, 600M parameters) into Palabrita, a production Android word-guessing game. Over a 5-day development sprint comprising 204 commits (~90 directly AI-related), the system underwent a radical transformation: from an ambitious design where the LLM generated complete structured puzzles (word, category, difficulty, and five hints as JSON) to a pragmatic architecture where curated word lists provide the words and the LLM generates only three short hints, with a deterministic fallback if it fails. We identify five categories of failures specific to on-device SLM integration: output format violations, constraint violations, context quality degradation, latency incompatibility, and model selection instability. For each failure category, we document the observed symptoms, root causes, and the prompt engineering and architectural strategies that effectively mitigated them, including multi-layer defensive parsing, contextual retry with failure feedback, session rotation, progressive prompt hardening, and systematic responsibility reduction. Our findings demonstrate that on-device SLMs are viable for production mobile applications, but only when the developer accepts a fundamental constraint: the most reliable on-device LLM feature is one where the LLM does the least. We distill our experience into eight actionable design heuristics for practitioners integrating SLMs into mobile apps.
>
---
#### [new 158] Supernodes and Halos: Loss-Critical Hubs in LLM Feed-Forward Layers
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究Transformer模型中通道重要性结构，解决结构化剪枝问题。通过分析损失敏感通道，发现关键核心（supernodes）及弱关联的halo结构，提出SCAR-Prot方法提升剪枝效果。**

- **链接: [https://arxiv.org/pdf/2604.23475](https://arxiv.org/pdf/2604.23475)**

> **作者:** Audrey Cherilyn; Houman Safaai
>
> **摘要:** We study the organization of channel-level importance in transformer feed-forward networks (FFNs). Using a Fisher-style loss proxy (LP) based on activation-gradient second moments, we show that loss sensitivity is concentrated in a small set of channels within each layer. In Llama-3.1-8B, the top 1% of channels per layer accounts for a median of 58.7% of LP mass, with a range of 33.0% to 86.1%. We call these loss-critical channels supernodes. Although FFN layers also contain strong activation outliers, LP-defined supernodes overlap only weakly with activation-defined outliers and are not explained by activation power or weight norms alone. Around this core, we find a weaker but consistent halo structure: some non-supernode channels share the supernodes' write support and show stronger redundancy with the protected core. We use one-shot structured FFN pruning as a diagnostic test of this organization. At 50% FFN sparsity, baselines that prune many supernodes degrade sharply, whereas our SCAR variants explicitly protect the supernode core; the strongest variant, SCAR-Prot, reaches perplexity 54.8 compared with 989.2 for Wanda-channel. The LP-concentration pattern appears across Mistral-7B, Llama-2-7B, and Qwen2-7B, remains visible in targeted Llama-3.1-70B experiments, and increases during OLMo-2-7B pretraining. These results suggest that LLM FFNs develop a small learned core of loss-critical channels, and that preserving this core is important for reliable structured pruning.
>
---
#### [new 159] AeSlides: Incentivizing Aesthetic Layout in LLM-Based Slide Generation via Verifiable Rewards
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于幻灯片生成任务，旨在解决文本生成与视觉美学不匹配的问题。通过设计可验证的美学指标，采用强化学习优化布局质量。**

- **链接: [https://arxiv.org/pdf/2604.22840](https://arxiv.org/pdf/2604.22840)**

> **作者:** Yiming Pan; Chengwei Hu; Xuancheng Huang; Can Huang; Mingming Zhao; Yuean Bi; Xiaohan Zhang; Aohan Zeng; Linmei Hu
>
> **备注:** 21 pages, 25 figures, 9 tables
>
> **摘要:** Large language models (LLMs) have demonstrated strong potential in agentic tasks, particularly in slide generation. However, slide generation poses a fundamental challenge: the generation process is text-centric, whereas its quality is governed by visual aesthetics. This modality gap leads current models to frequently produce slides with aesthetically suboptimal layouts. Existing solutions typically rely either on heavy visual reflection, which incurs high inference cost yet yields limited gains; or on fine-tuning with large-scale datasets, which still provides weak and indirect aesthetic supervision. In contrast, the explicit use of aesthetic principles as supervision remains unexplored. In this work, we present AeSlides, a reinforcement learning framework with verifiable rewards for Aesthetic layout supervision in Slide generation. We introduce a suite of meticulously designed verifiable metrics to quantify slide layout quality, capturing key layout issues in an accurate, efficient, and low-cost manner. Leveraging these verifiable metrics, we develop a GRPO-based reinforcement learning method that directly optimizes slide generation models for aesthetically coherent layouts. With only 5K training prompts on GLM-4.7-Flash, AeSlides improves aspect ratio compliance from 36% to 85%, while reducing whitespace by 44%, element collisions by 43%, and visual imbalance by 28%. Human evaluation further shows a substantial improvement in overall quality, increasing scores from 3.31 to 3.56 (+7.6%), outperforming both model-based reward optimization and reflection-based agentic approaches, and even edging out Claude-Sonnet-4.5. These results demonstrate that such a verifiable aesthetic paradigm provides an efficient and scalable approach to aligning slide generation with human aesthetic preferences. Our repository is available at this https URL.
>
---
#### [new 160] Learning in Blocks: A Multi Agent Debate Assisted Personalized Adaptive Learning Framework for Language Learning
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于语言学习任务，旨在解决传统测试无法准确评估口语能力的问题。提出Learning in Blocks框架，通过多智能体辩论实现精准评分与个性化推荐，提升学习效果。**

- **链接: [https://arxiv.org/pdf/2604.22770](https://arxiv.org/pdf/2604.22770)**

> **作者:** Nicy Scaria; Silvester John Joseph Kennedy; Deepak Subramani
>
> **备注:** Accepted as main paper in AIED 2026
>
> **摘要:** Most digital language learning curricula rely on discrete-item quizzes that test recall rather than applied conversational proficiency. When progression is driven by quiz performance, learners can advance despite persistent gaps in using grammar and vocabulary during interaction. Recent work on LLM-based judging suggests a path toward scoring open-ended conversations, but using interaction evidence to drive progression and review requires scoring protocols that are reliable and validated. We introduce Learning in Blocks, a framework that grounds progression in demonstrated conversational competence evaluated using CEFR-aligned rubrics. The framework employs heterogeneous multi-agent debate (HeteroMAD) in two stages: a scoring stage where role-specialized agents independently evaluate Grammar, Vocabulary, and Interactive Communication, engage in debate to address conflicting judgments, and a judge synthesizes consensus scores; and a recommendation stage that identifies specific grammar skills and vocabulary topics for targeted review. Progression requires demonstrating 70% mastery, and spaced review targets identified weaknesses to counter skill decay. We benchmark four scoring and recommendation methods on CEFR A2 conversations annotated by ESL experts. HeteroMAD achieves a superior score agreement with a 0.23 degree of variation and recommendation acceptability of 90.91%. An 8-week study with 180 CEFR A2 learners demonstrates that combining rubric-aligned scoring and recommendation with spaced review and mastery-based progression produces better learning outcomes than feedback alone.
>
---
#### [new 161] When to Commit? Towards Variable-Size Self-Contained Blocks for Discrete Diffusion Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言生成任务，解决dLLMs中块边界选择导致的提前提交问题。提出自洽块（VSB）方法，基于预测一致性选择块边界。**

- **链接: [https://arxiv.org/pdf/2604.23994](https://arxiv.org/pdf/2604.23994)**

> **作者:** Danny Wang; Ruihong Qiu; Zi Huang
>
> **摘要:** Discrete diffusion language models (dLLMs) enable parallel token updates with bidirectional attention, yet practical generation typically adopts blockwise semi-autoregressive decoding. This switch creates a training-inference mismatch: training denoises with full-sequence context, while inference commits tokens within a bounded block without future context. Therefore, decoding with fixed-size or heuristic-based blocks can lead to premature token commitments, as decisions are made without full access to future context that could alter those choices. Motivated by this, we propose self-containedness as a principled criterion for block commitment. A block is self-contained if its predictions remain consistent with Future-Aware (FA) or without No-Future (NF) access to future context, reframing block boundary selection as a test of self-containedness rather than a heuristic choice. Based on this principle, we introduce Variable-size Self-contained Blocks (VSB) for dLLMs. VSB scores and selects block boundaries using the divergence between token-level predictive distributions under NF and FA conditioning, which quantifies how predictions would change if future context were revealed. We provide theoretical justification linking self-containedness to predictive consistency, and extensive experiments validate VSB's efficacy over fixed-size and heuristic blockwise decoding.
>
---
#### [new 162] Quantifying and Mitigating Self-Preference Bias of LLM Judges
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的模型评估任务，旨在解决LLM在自我评价中产生的自偏见问题，通过自动化框架量化并减轻这种偏差。**

- **链接: [https://arxiv.org/pdf/2604.22891](https://arxiv.org/pdf/2604.22891)**

> **作者:** Jinming Yang; Chuxian Qiu; Zhenyu Deng; Xinshan Jiao; Tao Zhou
>
> **摘要:** LLM-as-a-Judge has become a dominant approach in automated evaluation systems, playing critical roles in model alignment, leaderboard construction, quality control, and so on. However, the scalability and trustworthiness of this approach can be substantially distorted by Self-Preference Bias (SPB), which is a directional evaluative deviation in which LLMs systematically favor or disfavor their own generated outputs during evaluation. Existing measurements rely on costly human annotations and conflate generative capability with evaluative stance, and thus are impractical for large-scale deployment in real-world systems. To address this issue, we introduce a fully automated framework to quantifying and mitigating SPB, which constructs equal-quality pairs of responses with negligible quality differences, enabling statistical disentanglement of discriminability from bias propensity without human gold standards. Empirical analysis across 20 mainstream LLMs reveals that advanced capabilities are often uncorrelated, or even negatively correlated, with low SPB. To mitigate this bias, we propose a structured multi-dimensional evaluation strategy grounded in cognitive load decomposition, which reduces SPB by 31.5\% on average.
>
---
#### [new 163] SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型推理任务，指出混合策略方法的基准存在错误，修正后标准SFT-then-RL方法表现更优。**

- **链接: [https://arxiv.org/pdf/2604.23747](https://arxiv.org/pdf/2604.23747)**

> **作者:** Alexis Limozin; Eduard Durech; Torsten Hoefler; Imanol Schlag; Valentina Pyatkin
>
> **摘要:** Recent mixed-policy optimization methods for LLM reasoning that interleave or blend supervised and reinforcement learning signals report improvements over the standard SFT-then-RL pipeline. We show that numerous recently published research papers rely on a faulty baseline caused by two distinct bugs: a CPU-offloaded optimizer bug in DeepSpeed that silently drops intermediate micro-batches during gradient accumulation (affecting multiple downstream frameworks including TRL, OpenRLHF and Llama-Factory), and a loss aggregation bug in OpenRLHF that incorrectly weights per-mini-batch losses. Together they suppress SFT performance, with the optimizer bug accounting for most of the gap and the loss aggregation bug contributing a smaller additional effect. Once corrected, the standard SFT-then-RL pipeline surpasses every published mixed-policy method we evaluate by +3.8 points on math benchmarks with Qwen2.5-Math-7B and by +22.2 points with Llama-3.1-8B. Even a truncated variant with just 50 RL steps outperforms mixed-policy methods on math benchmarks while using fewer FLOPs.
>
---
#### [new 164] A Parametric Memory Head for Continual Generative Retrieval
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于生成式信息检索任务，解决动态文档集合下的模型遗忘问题。提出PAMT方法，在不更新主模型的情况下，通过参数化记忆头提升模型对旧文档的保留能力。**

- **链接: [https://arxiv.org/pdf/2604.23388](https://arxiv.org/pdf/2604.23388)**

> **作者:** Kidist Amde Mekonnen; Yubao Tang; Maarten de Rijke
>
> **备注:** 12 pages, 3 figures, 3 tables; accepted to the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval, July 20-24, 2026, Melbourne/Naarm, Australia
>
> **摘要:** Generative information retrieval (GenIR) consolidates retrieval into a single neural model that decodes document identifiers (docids) directly from queries. While this model-as-index paradigm offers architectural simplicity, it is poorly suited to dynamic document collections. Unlike modular systems, where indexes are easily updated, GenIR's knowledge is parametrically encoded in its weights; consequently, standard adaptation methods such as full and parameter-efficient fine-tuning can induce catastrophic forgetting. We show that sequential adaptation improves retrieval on newly added documents but substantially degrades performance on earlier slices, exposing a pronounced stability-plasticity trade-off. To address this, we propose post-adaptation memory tuning (PAMT), a memory-only stabilization stage that augments an adapted model with a modular parametric memory head (PMH). PAMT freezes the backbone and attaches a product-key memory with fixed addressing. During prefix-trie constrained decoding, decoder hidden states sparsely query PMH to produce residual corrections in hidden space; these corrections are mapped to score adjustments via the frozen output embedding matrix, computed only over trie-valid tokens. This guides docid generation while keeping routing and backbone parameters fixed. To limit cross-slice interference, PAMT updates only a fixed budget of memory values selected using decoding-time access statistics, prioritizing entries frequently activated by the current slice and rarely used in prior sessions. Experiments on MS MARCO and Natural Questions under sequential, disjoint corpus increments show that PAMT substantially improves retention on earlier slices with minimal impact on retrieval performance for newly added documents, while modifying only a sparse subset of memory values per session.
>
---
#### [new 165] Rank, Head-Channel Non-Identifiability, and Symmetry Breaking: A Precise Analysis of Representational Collapse in Transformers
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文分析Transformer模型中的表示崩溃问题，探讨了秩坍缩、头通道不可识别性等现象，提出对称性破缺框架，并改进输出投影结构。任务为理解Transformer表示特性，解决问题为揭示崩溃机制与改进架构。**

- **链接: [https://arxiv.org/pdf/2604.23681](https://arxiv.org/pdf/2604.23681)**

> **作者:** Giansalvo Cirrincione
>
> **备注:** 36 pages, 8 figures, 1 table. Submitted to Artificial Intelligence (Elsevier)
>
> **摘要:** A widely cited result by Dong et al. (2021) showed that Transformers built from self-attention alone, without skip connections or feed-forward layers, suffer from rapid rank collapse: all token representations converge to a single direction. The proposed remedy was the MLP. We show that this picture, while correct in the regime studied by Dong, is incomplete in ways that matter for architectural understanding. Three results are established. First, layer normalisation is precisely affine-rank-neutral: it preserves the affine rank of the token representation set exactly. The widespread claim that LN "plays no role" is imprecise; the correct statement is sharper. Second, residual connections generically obstruct rank collapse in real Transformers such as BERT-base, in a measure-theoretic sense, without contribution from the MLP. The MLP's irreplaceable function is different: generating feature directions outside the linear span of the original token embeddings, which no stack of attention layers can produce. Third, a phenomenon distinct from rank collapse is identified: head-channel non-identifiability. After multi-head attention sums per-head outputs through the output projection, individual contributions cannot be canonically attributed to a specific head; n(H-1)d_k degrees of freedom per layer remain ambiguous when recovering a single head from the mixed signal. The MLP cannot remedy this because it acts on the post-summation signal. A constructive partial remedy is proposed: a position-gated output projection (PG-OP) at parameter overhead below 1.6% of the standard output projection. The four collapse phenomena identified in the literature -- rank collapse in depth, in width, head-channel non-identifiability, and entropy collapse -- are unified under a symmetry-breaking framework, each corresponding to a distinct symmetry of the Transformer's forward pass.
>
---
#### [new 166] Preserving Long-Tailed Expert Information in Mixture-of-Experts Tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型微调任务，解决MoE架构中路由器脆弱导致的性能下降问题。通过结合稀疏化与持续激活的专家机制，有效保留长尾专家信息，提升模型表现。**

- **链接: [https://arxiv.org/pdf/2604.23036](https://arxiv.org/pdf/2604.23036)**

> **作者:** Haoze He; Xingyuan Ding; Xuan Jiang; Xinkai Zou; Alex Cheng; Yibo Zhao; Juncheng Billy Li; Heather Miller
>
> **备注:** 36 pages
>
> **摘要:** Despite MoE models leading many benchmarks, supervised fine-tuning (SFT) for the MoE architectures remains difficult because its router layers are fragile. Methods such as DenseMixer and ESFT mitigate router collapse with dense mixing or auxiliary load-balancing losses, but these introduce noisy gradients that often degrade performance. In preliminary experiments, we systematically pruned experts and observed that while certain super experts are activated far more frequently, discarding less used experts still leads to notable performance degradation. This suggests that even rarely activated experts encode non-trivial knowledge useful for downstream tasks. Motivated by this, we propose an auxiliary-loss-free MoE SFT framework that combines bias-driven sparsification with always-active gated condenser experts. Rather than enforcing balanced activation across all experts, our method encourages task-relevant experts to remain active while pushing long-tailed experts toward inactivity. The condenser experts provide a persistent, learnable pathway that alleviates gradient starvation and facilitates consolidation of information that would otherwise remain fragmented across sparsely activated experts. Analysis further suggest that this design better preserves long-tailed expert information under sparse routing. Experiments on large-scale MoE models demonstrate that our approach outperforms state-of-the-art SFT baselines such as DenseMixer and ESFT, achieving average gain of 2.5%+ on both mathematical reasoning and commonsenseQA benchmarks.
>
---
#### [new 167] AgentPulse: A Continuous Multi-Signal Framework for Evaluating AI Agents in Deployment
- **分类: cs.AI; cs.CL; cs.SE**

- **简介: 该论文提出AgentPulse框架，用于持续评估AI代理在部署中的表现。解决静态基准无法反映实际部署情况的问题，通过多信号分析，涵盖性能、采用、社区和生态四个维度。**

- **链接: [https://arxiv.org/pdf/2604.24038](https://arxiv.org/pdf/2604.24038)**

> **作者:** Yuxuan Gao; Megan Wang; Yi Ling Yu
>
> **备注:** 19 pages, 5 figures, 9 tables. Preprint under review
>
> **摘要:** Static benchmarks measure what AI agents can do at a fixed point in time but not how they are adopted, maintained, or experienced in deployment. We introduce AgentPulse, a continuous evaluation framework scoring 50 agents across 10 workload categories along four factors (Benchmark Performance, Adoption Signals, Community Sentiment, and Ecosystem Health) aggregated from 18 real-time signals across GitHub, package registries, IDE marketplaces, social platforms, and benchmark leaderboards. Three analyses ground the framework. The four factors capture largely complementary information (n=50; $\rho_{\max}=0.61$ for Adoption-Ecosystem, all others $|\rho| \leq 0.37$). A circularity-controlled test (n=35) shows the Benchmark+Sentiment sub-composite, which contains no GitHub-derived signals, predicts external adoption proxies it does not aggregate: GitHub stars ($\rho_s=0.52$, $p<0.01$) and Stack Overflow question volume ($\rho_s=0.49$, $p<0.01$), with VS Code installs ($\rho_s=0.44$, $p<0.05$) reported as illustrative given that only 11 of 35 agents have non-zero installs. On the n=11 subset with published SWE-bench scores, composite and benchmark-only rankings are nearly uncorrelated ($\rho_s=0.25$; 9 of 11 agents shift by at least 2 ranks), driven by a strong negative Adoption-Capability correlation among closed-source high-capability agents within this subset. This is precisely why we rest the framework's validity claim on the broader n=35 test rather than the SWE-bench overlap. AgentPulse surfaces deployment signal absent from benchmarks; it is a methodology, not a ground-truth ranking. The framework, all collected signals, scoring outputs, and evaluation harness are released under CC BY 4.0.
>
---
#### [new 168] Ulterior Motives: Detecting Misaligned Reasoning in Continuous Thought Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于模型安全任务，旨在检测连续思维模型中的对齐偏差。通过构建基准和双触发机制，研究发现可利用潜在空间特征进行安全监控。**

- **链接: [https://arxiv.org/pdf/2604.23460](https://arxiv.org/pdf/2604.23460)**

> **作者:** Sharan Ramjee
>
> **备注:** 15 pages with 2 figures
>
> **摘要:** Chain-of-Thought (CoT) reasoning has emerged as a key technique for eliciting complex reasoning in Large Language Models (LLMs). Although interpretable, its dependence on natural language limits the model's expressive bandwidth. Continuous thought models address this bottleneck by reasoning in latent space rather than human-readable tokens. While they enable richer representations and faster inference, they raise a critical safety question: how can we detect misaligned reasoning in an uninterpretable latent space? To study this, we introduce MoralChain, a benchmark of 12,000 social scenarios with parallel moral/immoral reasoning paths. We train a continuous thought model with backdoor behavior using a novel dual-trigger paradigm - one trigger that arms misaligned latent reasoning ([T]) and another that releases harmful outputs ([O]). We demonstrate three findings: (1) continuous thought models can exhibit misaligned latent reasoning while producing aligned outputs, with aligned and misaligned reasoning occupying geometrically distinct regions of latent space; (2) linear probes trained on behaviorally-distinguishable conditions ([T][O] vs [O]) transfer to detecting armed-but-benign states ([T] vs baseline) with high accuracy; and (3) misalignment is encoded in early latent thinking tokens, suggesting safety monitoring for continuous thought models should target the "planning" phase of latent reasoning.
>
---
#### [new 169] AgenticCache: Cache-Driven Asynchronous Planning for Embodied AI Agents
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出AgenticCache框架，解决 embodied AI 代理中因频繁调用大语言模型导致的延迟和成本问题。通过缓存重用计划，提升任务成功率并降低资源消耗。**

- **链接: [https://arxiv.org/pdf/2604.24039](https://arxiv.org/pdf/2604.24039)**

> **作者:** Hojoon Kim; Yuheng Wu; Thierry Tambe
>
> **备注:** Accepted at MLSys 2026
>
> **摘要:** Embodied AI agents increasingly rely on large language models (LLMs) for planning, yet per-step LLM calls impose severe latency and cost. In this paper, we show that embodied tasks exhibit strong plan locality, where the next plan is largely predictable from the current one. Building on this, we introduce AgenticCache, a planning framework that reuses cached plans to avoid per-step LLM calls. In AgenticCache, each agent queries a runtime cache of frequent plan transitions, while a background Cache Updater asynchronously calls the LLM to validate and refine cached entries. Across four multi-agent embodied benchmarks, AgenticCache improves task success rate by 22% on average across 12 configurations (4 benchmarks x 3 models), reduces simulation latency by 65%, and lowers token usage by 50%. Cache-based plan reuse thus offers a practical path to low-latency, low-cost embodied agents. Code is available at this https URL.
>
---
## 更新

#### [replaced 001] Hyperloop Transformers
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型架构研究，旨在提升参数效率。通过引入循环Transformer结构和超连接，减少内存占用并保持性能。**

- **链接: [https://arxiv.org/pdf/2604.21254](https://arxiv.org/pdf/2604.21254)**

> **作者:** Abbas Zeitoun; Lucas Torroba-Hennigen; Yoon Kim
>
> **摘要:** LLM architecture research generally aims to maximize model quality subject to fixed compute/latency budgets. However, many applications of interest such as edge and on-device deployment are further constrained by the model's memory footprint, thus motivating parameter-efficient architectures for language modeling. This paper describes a simple architecture that improves the parameter-efficiency of LLMs. Our architecture makes use of looped Transformers as a core primitive, which reuse Transformer layers across depth and are thus more parameter-efficient than ordinary (depth-matched) Transformers. We organize the looped Transformer into three blocks--begin, middle, and end blocks--where each block itself consists of multiple Transformer layers, and only the middle block is applied recurrently across depth. We augment the looped middle block with hyper-connections (Xie et al., 2026), which expand the residual stream into matrix-valued residual streams. Hyper-connections are applied only after each loop, and therefore add minimal new parameters and compute cost. Across various model scales, we find that our Hyper-Connected Looped Transformer (Hyperloop Transformer) is able to outperform depth-matched Transformer and mHC Transformer baselines despite using approximately 50% fewer parameters. The outperformance persists through post-training weight quantization, thus positioning Hyperloop Transformers as an attractive architecture for memory-efficient language modeling.
>
---
#### [replaced 002] Gated Tree Cross-Attention for Checkpoint-Compatible Syntax Injection in Decoder-Only LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升解码器模型的语法鲁棒性。针对模型对语法扰动敏感的问题，提出GTCA方法，在不改变原有结构的前提下注入语法信息，增强模型稳定性。**

- **链接: [https://arxiv.org/pdf/2602.15846](https://arxiv.org/pdf/2602.15846)**

> **作者:** Xinyu Gao; Shaonan Wang; Nai Ding
>
> **备注:** ACL 2026 MainConference
>
> **摘要:** Decoder-only large language models achieve strong broad performance but are brittle to minor grammatical perturbations, undermining reliability for downstream reasoning. However, directly injecting explicit syntactic structure into an existing checkpoint can interfere with its pretrained competence. We introduce a checkpoint-compatible gated tree cross-attention (GTCA) branch that reads precomputed constituency chunk memory while leaving backbone architecture unchanged. Our design uses a token update mask and staged training to control the scope and timing of structural updates. Across benchmarks and Transformer backbones, GTCA strengthens syntactic robustness beyond continued-training baselines without compromising Multiple-Choice QA performance or commonsense reasoning, providing a practical checkpoint-compatible route to more syntax-robust decoder-only LLMs.
>
---
#### [replaced 003] Green Prompting: Characterizing Prompt-driven Energy Costs of LLM Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLM推理中的能耗问题，分析提示和响应特征对能耗的影响，旨在优化推理效率。属于自然语言处理任务。**

- **链接: [https://arxiv.org/pdf/2503.10666](https://arxiv.org/pdf/2503.10666)**

> **作者:** Marta Adamska; Daria Smirnova; Hamid Nasiri; Zhengxin Yu; Peter Garraghan
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Large Language Models (LLMs) have become widely used across various domains spanning search engines, code generation, and text creation. However, a major concern associated with their adoption is the high cost of inference, impacting both their sustainability and financial feasibility. In this study, we empirically study how different prompt and response characteristics directly impact LLM inference energy cost. We conduct experiments leveraging three open-source transformer-based LLMs across three task types$-$question answering, sentiment analysis, and text generation. For each inference, we analyzed prompt and response characteristics (length, semantic meaning, time taken, energy consumption). Our results demonstrate that even when presented with identical tasks, models generate responses with varying characteristics and subsequently exhibit distinct energy consumption patterns. We found that prompt length is less significant than the semantic meaning of the task itself. In addition, we identified specific keywords associated with higher or lower energy usage that vary between associated tasks. These findings highlight the importance of prompt design in optimizing inference efficiency. We conclude that the semantic meaning of prompts and certain task-related keywords significantly impact inference costs, leading the way for deeper exploration towards creating energy-adaptive LLMs.
>
---
#### [replaced 004] Speech-FT: Merging Pre-trained And Fine-Tuned Speech Representation Models For Cross-Task Generalization
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文提出Speech-FT框架，解决语音模型微调后跨任务泛化能力下降的问题。通过两阶段微调保持特征相似性，提升多种任务性能。**

- **链接: [https://arxiv.org/pdf/2502.12672](https://arxiv.org/pdf/2502.12672)**

> **作者:** Tzu-Quan Lin; Wei-Ping Huang; Hao Tang; Hung-yi Lee
>
> **备注:** Published in IEEE Transactions on Audio, Speech, and Language Processing (TASLP). Model and code available at: this https URL
>
> **摘要:** Fine-tuning speech representation models can enhance performance on specific tasks but often compromises their cross-task generalization ability. This degradation is often caused by excessive changes in the representations, making it difficult to retain information learned during pre-training. Existing approaches, such as regularizing weight changes during fine-tuning, may fail to maintain sufficiently high feature similarity with the pre-trained model, and thus could possibly lose cross-task generalization. To address this issue, we propose Speech-FT, a novel two-stage fine-tuning framework designed to maintain cross-task generalization while benefiting from fine-tuning. Speech-FT first applies fine-tuning specifically designed to reduce representational drift, followed by weight-space interpolation with the pre-trained model to restore cross-task generalization. Extensive experiments on HuBERT, wav2vec 2.0, DeCoAR 2.0, and WavLM Base+ demonstrate that Speech-FT consistently improves performance across a wide range of supervised, unsupervised, and multitask fine-tuning scenarios. Moreover, Speech-FT achieves superior cross-task generalization compared to fine-tuning baselines that explicitly constrain weight changes, such as weight-space regularization and LoRA fine-tuning. Our analysis reveals that Speech-FT maintains higher feature similarity to the pre-trained model compared to alternative strategies, despite allowing larger weight-space updates. Notably, Speech-FT achieves significant improvements on the SUPERB benchmark. For example, when fine-tuning HuBERT on automatic speech recognition, Speech-FT is able to reduce phone error rate from 5.17% to 3.94%, lower word error rate from 6.38% to 5.75%, and increase speaker identification accuracy from 81.86% to 84.11%. Speech-FT provides a simple yet powerful solution for further refining speech representation models after pre-training.
>
---
#### [replaced 005] Evaluating Language Models' Evaluations of Games
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI评估任务，旨在研究语言模型对游戏的评价能力。通过对比模型与人类及符号代理的评价，分析模型在复杂性和量化难度上的表现。**

- **链接: [https://arxiv.org/pdf/2510.10930](https://arxiv.org/pdf/2510.10930)**

> **作者:** Katherine M. Collins; Cedegao E. Zhang; Graham Todd; Lance Ying; Mauricio Barba da Costa; Ryan Liu; Prafull Sharma; Adrian Weller; Ionatan Kuperwajs; Lionel Wong; Joshua B. Tenenbaum; Thomas L. Griffiths
>
> **摘要:** Reasoning is not just about solving problems -- it is also about evaluating which problems are worth solving at all. Evaluations of artificial intelligence (AI) systems primarily focused on problem solving, historically by studying how models play games such as chess and Go. In this paper, we advocate for a new paradigm that assesses AI systems' evaluation of games. First, we introduce a formalism for evaluating such evaluations. We then leverage a large-scale dataset of over 100 novel board games and over 450 human judgments to compare evaluations produced by modern language and reasoning models against those of people and symbolic computational agents. We consider two kinds of evaluative queries: assessing the payoff (or fairness) and the funness of games. These queries span two dimensions relevant to the design of evaluations of AI evaluations: how complex a query is to compute and how difficult a query is to quantify. Our results show that reasoning models are generally more aligned to people in their evaluations of games than non-reasoning language models. However, we observe a non-monotonic relationship: as models get closer to game-theoretic optimal, their fit to human data weakens. We also observe more "jaggedness" across models for assessing funness, in line with the greater difficulty of quantifying this query. Across queries and games, reasoning models show highly variable and unpredictable resource usage when assessing queries, pointing to the importance of imbuing more resource-rational meta-reasoning in language and reasoning models.
>
---
#### [replaced 006] Annotating Dimensions of Social Perception in Text: A Sentence-Level Dataset of Warmth and Competence
- **分类: cs.CL**

- **简介: 该论文属于社会感知维度标注任务，旨在解决NLP中温暖与能力维度的句子级分析问题。作者构建了首个相关句子级数据集，并评估了大语言模型的表现。**

- **链接: [https://arxiv.org/pdf/2601.06316](https://arxiv.org/pdf/2601.06316)**

> **作者:** Mutaz Ayesh; Saif M. Mohammad; Nedjma Ousidhoum
>
> **备注:** Accepted at ACL2026 (Main Conference)
>
> **摘要:** Warmth (W) (often further broken down intoTrust (T) and Sociability (S)) and Competence (C) are central dimensions along which people evaluate individuals and social groups (Fiske, 2018). While these constructs are well established in social psychology, they are only starting to get attention in NLP research through word-level lexicons, which do not fully capture their contextual expression in larger text units and discourse. In this work, we introduce Warmth and Competence Sentences (W&C-Sent), the first sentence-level dataset annotated for warmth and competence. The dataset includes over 1,600 English sentence--target pairs annotated along three dimensions: trust and sociability (components of warmth), and competence. The sentences in W&C-Sent are social media posts that express attitudes and opinions about specific individuals or social groups (the targets of our annotations). We describe the data collection, annotation, and quality-control procedures in detail, and evaluate a range of large language models (LLMs) on their ability to identify trust, sociability, and competence in text. W&C-Sent provides a new resource for analyzing warmth and competence in language and supports future research at the intersection of NLP and computational social science.
>
---
#### [replaced 007] AtomEval: Atomic Evaluation of Adversarial Claims in Fact Verification
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AtomEval，用于评估事实验证中的对抗性陈述。针对标准指标无法准确检测事实错误的问题，通过分解陈述并评分，提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2604.07967](https://arxiv.org/pdf/2604.07967)**

> **作者:** Hongyi Cen
>
> **摘要:** Adversarial claim rewriting is widely used to test fact-checking systems, but standard metrics fail to capture truth-conditional consistency and often label semantically corrupted rewrites as successful. We introduce AtomEval, a validity-aware evaluation framework that decomposes claims into subject-relation-object-modifier (SROM) atoms and scores adversarial rewrites with Atomic Validity Scoring (AVS), enabling detection of factual corruption beyond surface similarity. Experiments on the FEVER dataset across representative attack strategies and LLM generators show that AtomEval provides more reliable evaluation signals in our experiments. Using AtomEval, we further analyze LLM-based adversarial generators and observe that stronger models do not necessarily produce more effective adversarial claims under validity-aware evaluation, highlighting previously overlooked limitations in current adversarial evaluation practices.
>
---
#### [replaced 008] SWE-QA: Can Language Models Answer Repository-level Code Questions?
- **分类: cs.CL; cs.PL; cs.SE**

- **简介: 该论文属于代码理解任务，旨在解决仓库级代码问答问题。构建了SWE-QA基准，涵盖多种类型的问题，并开发了SWE-QA-Agent框架进行自动回答。**

- **链接: [https://arxiv.org/pdf/2509.14635](https://arxiv.org/pdf/2509.14635)**

> **作者:** Weihan Peng; Yuling Shi; Yuhang Wang; Xinyun Zhang; Beijun Shen; Xiaodong Gu
>
> **备注:** Accepted to ACL 2026 Findings. Code and data available at this https URL
>
> **摘要:** Understanding and reasoning about entire software repositories is an essential capability for intelligent software engineering tools. While existing benchmarks such as CoSQA and CodeQA have advanced the field, they predominantly focus on small, self-contained code snippets. These setups fail to capture the complexity of real-world repositories, where effective understanding and reasoning often require navigating multiple files, understanding software architecture, and grounding answers in long-range code dependencies. In this paper, we present SWE-QA, a repository-level code question answering (QA) benchmark designed to facilitate research on automated QA systems in realistic code environments. SWE-QA involves 576 high-quality question-answer pairs spanning diverse categories, including intention understanding, cross-file reasoning, and multi-hop dependency analysis. To construct SWE-QA, we first crawled 77,100 GitHub issues from 11 popular repositories. Based on an analysis of naturally occurring developer questions extracted from these issues, we developed a two-level taxonomy of repository-level questions and constructed a set of seed questions for each category. For each category, we manually curated and validated questions and collected their corresponding answers. As a prototype application, we further develop SWE-QA-Agent, an agentic framework in which LLM agents reason and act to find answers automatically. We evaluate six advanced LLMs on SWE-QA under various context augmentation strategies. Experimental results highlight the promise of LLMs, particularly our SWE-QA-Agent framework, in addressing repository-level QA, while also revealing open challenges and pointing to future research directions.
>
---
#### [replaced 009] Conjecture and Inquiry: Quantifying Software Performance Requirements via Interactive Retrieval-Augmented Preference Elicitation
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于软件工程任务，旨在解决性能需求量化难题。通过交互式检索增强偏好获取方法（IRAP），将自然语言性能需求转化为数学函数，提升量化准确性与效率。**

- **链接: [https://arxiv.org/pdf/2604.21380](https://arxiv.org/pdf/2604.21380)**

> **作者:** Shihai Wang; Tao Chen
>
> **备注:** 9 pages,accepted by ACL 2026
>
> **摘要:** Since software performance requirements are documented in natural language, quantifying them into mathematical forms is essential for software engineering. Yet, the vagueness in performance requirements and uncertainty of human cognition have caused highly uncertain ambiguity in the interpretations, rendering their automated quantification an unaddressed and challenging problem. In this paper, we formalize the problem and propose IRAP, an approach that quantifies performance requirements into mathematical functions via interactive retrieval-augmented preference elicitation. IRAP differs from the others in that it explicitly derives from problem-specific knowledge to retrieve and reason the preferences, which also guides the progressive interaction with stakeholders, while reducing the cognitive overhead. Experiment results against 10 state-of-the-art methods on four real-world datasets demonstrate the superiority of IRAP on all cases with up to 40x improvements under as few as five rounds of interactions.
>
---
#### [replaced 010] How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究循环语言模型的深度缩放规律，解决如何量化额外循环带来的效果。通过实验分析循环次数对模型性能的影响，提出一个新指数φ来衡量循环等效性。**

- **链接: [https://arxiv.org/pdf/2604.21106](https://arxiv.org/pdf/2604.21106)**

> **作者:** Kristian Schwethelm; Daniel Rueckert; Georgios Kaissis
>
> **备注:** v2: added interesting truncated-BPTT and hyperconnections probes, new discussion sections on $φ$ as decision metric and inference cost
>
> **摘要:** We measure how much one extra recurrence is worth to a looped (depth-recurrent) language model, in equivalent unique parameters. From an iso-depth sweep of 116 pretraining runs across recurrence counts $r \in \{1, 2, 4, 8\}$ spanning ${\sim}50\times$ in training compute, we fit a joint scaling law $L = E + A\,(N_\text{once} + r^{\varphi} N_\text{rec})^{-\alpha} + B\,D^{-\beta}$ and recover a new recurrence-equivalence exponent $\varphi = 0.46$. Intuitively, $\varphi$ tells us whether looping a block $r$ times is equivalent in validation loss to $r$ unique blocks of a non-looped model (full equivalence, $\varphi{=}1$) or to a single block run repeatedly with no capacity gain ($\varphi{=}0$). Our $\varphi = 0.46$ sits in between, so each additional recurrence predictably increases validation loss at matched training compute. For example, at $r{=}4$ a 410M looped model performs on par with a 580M non-looped model, but incurs the training cost of a 1B non-looped one. We demonstrate the utility of $\varphi$ as a measurement tool on two probes. Truncated backpropagation lowers $\varphi$ to $0.38$, indicating that the loop mechanism is poorly trained under truncation, even though validation loss decreases. Conversely, hyperconnections raise $\varphi$ to $0.65$, a genuine capacity gain. Our method applies to any looped LM and separates true loop improvements from token-budget gains.
>
---
#### [replaced 011] Council Mode: A Heterogeneous Multi-Agent Consensus Framework for Reducing LLM Hallucination and Bias
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM的幻觉和偏见问题。提出Council Mode框架，通过多智能体共识提升生成内容的准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2604.02923](https://arxiv.org/pdf/2604.02923)**

> **作者:** Shuai Wu; Xue Li; Yanna Feng; Yufang Li; Zhijun Wang; Ran Wang
>
> **备注:** 24 pages, 8 figures, 16 tables, 1 algorithm. Open-source implementation: this https URL. Archived software DOI: https://doi.org/10.5281/zenodo.19767626
>
> **摘要:** Large Language Models (LLMs) have demonstrated advanced capabilities but often suffer from factual inaccuracies (hallucinations) and systematic biases. These issues, sometimes amplified in specific architectures like Mixture-of-Experts (MoE) which motivate our work, pose risks for reliable deployment. To address these challenges, we propose the Council Mode, a multi-agent consensus framework. Our approach dispatches queries to multiple heterogeneous frontier LLMs in parallel and synthesizes their outputs using a dedicated consensus model. The pipeline consists of three phases: an intelligent triage for query complexity, parallel generation across diverse models, and a structured synthesis that identifies agreement, disagreement, and unique findings. In our evaluation, conducted under controlled no-web settings, the Council Mode achieved a 35.9% relative reduction in hallucination rates on a 1,200-sample HaluEval subset and a 7.8-point improvement on TruthfulQA compared to the top-performing individual model. On our curated MDR-500 multi-domain reasoning benchmark, the Council Mode achieved a Quality Score of 91.7%, representing a 10.2-point improvement over the best individual model. The framework also exhibited lower measured bias variance under our rubric-based evaluation protocol. We provide a cost-effectiveness analysis showing that the framework incurs a 4.2x token-cost overhead, making it most suitable for accuracy-prioritized applications where the cost of errors exceeds the added inference cost. These findings suggest that structured multi-agent consensus is a promising direction for enhancing the reliability and factual grounding of LLM-generated content.
>
---
#### [replaced 012] What Prompts Don't Say: Understanding and Managing Underspecification in LLM Prompts
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于自然语言处理任务，解决LLM提示不明确的问题。分析了提示不明确的挑战，提出改进的优化机制和系统化管理流程。**

- **链接: [https://arxiv.org/pdf/2505.13360](https://arxiv.org/pdf/2505.13360)**

> **作者:** Chenyang Yang; Yike Shi; Qianou Ma; Michael Xieyang Liu; Christian Kästner; Tongshuang Wu
>
> **摘要:** Prompt underspecification is a common challenge when interacting with LLMs. In this paper, we present an in-depth analysis of this problem, showing that while LLMs can often infer unspecified requirements by default (41.1%), such behavior is fragile: Under-specified prompts are 2x as likely to regress across model or prompt changes, sometimes with accuracy drops exceeding 20%. This instability makes it difficult to reliably build LLM applications. Moreover, simply specifying all requirements does not consistently help, as models have limited instruction-following ability and requirements can conflict. Standard prompt optimizers likewise provide little benefit. To address these issues, we propose requirements-aware prompt optimization mechanisms that improve performance by 4.8% on average over baselines. We further advocate for a systematic process of proactive requirements discovery, evaluation, and monitoring to better manage prompt underspecification in practice.
>
---
#### [replaced 013] In-depth Analysis of Graph-based RAG in a Unified Framework
- **分类: cs.IR; cs.CL; cs.DB**

- **简介: 该论文属于知识增强生成任务，旨在系统比较图基RAG方法。通过统一框架分析不同方法在问答任务中的效果，提出改进变体并指出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2503.04338](https://arxiv.org/pdf/2503.04338)**

> **作者:** Yingli Zhou; Yaodong Su; Youran Sun; Shu Wang; Taotao Wang; Runyuan He; Yongwei Zhang; Sicong Liang; Xilin Liu; Yuchi Ma; Yixiang Fang
>
> **摘要:** Graph-based Retrieval-Augmented Generation (RAG) has proven effective in integrating external knowledge into large language models (LLMs), improving their factual accuracy, adaptability, interpretability, and trustworthiness. A number of graph-based RAG methods have been proposed in the literature. However, these methods have not been systematically and comprehensively compared under the same experimental settings. In this paper, we first summarize a unified framework to incorporate all graph-based RAG methods from a high-level perspective. We then extensively compare representative graph-based RAG methods over a range of questing-answering (QA) datasets -- from specific questions to abstract questions -- and examine the effectiveness of all methods, providing a thorough analysis of graph-based RAG approaches. As a byproduct of our experimental analysis, we are also able to identify new variants of the graph-based RAG methods over specific QA and abstract QA tasks respectively, by combining existing techniques, which outperform the state-of-the-art methods. Finally, based on these findings, we offer promising research opportunities. We believe that a deeper understanding of the behavior of existing methods can provide new valuable insights for future research.
>
---
#### [replaced 014] MERIT: Modular Framework for Multimodal Misinformation Detection with Web-Grounded Reasoning
- **分类: cs.AI; cs.CL; cs.CV; cs.CY; cs.LG**

- **简介: 该论文提出MERIT框架，用于多模态虚假信息检测，解决信息真实性验证问题。通过模块化设计提升检测效果。**

- **链接: [https://arxiv.org/pdf/2510.17590](https://arxiv.org/pdf/2510.17590)**

> **作者:** Mir Nafis Sharear Shopnil; Sharad Duwal; Abhishek Tyagi; Adiba Mahbub Proma
>
> **备注:** 18 pages, 4 tables, 3 figures. Major revision with updated title, framing, methodology, experiments, and error analysis
>
> **摘要:** We present MERIT, an inference-time modular framework for multimodal misinformation detection that decomposes verification into four specialized modules: visual forensics, cross-modal alignment, retrieval-augmented claim verification, and calibrated judgment. On MMFakeBench, MERIT with GPT-4o-mini achieves 81.65% F1, outperforming all reported zero-shot baselines including GPT-4V with MMD-Agent (74.0% F1). A controlled same-model evaluation confirms gains stem from architectural design: MERIT achieves 6.14 points higher misinformation recall than MMD-Agent under identical model conditions, with per-class gains of +18.0 on visual distortion and +5.33 on textual distortion. Ablation studies reveal non-overlapping module specialization, where removing any module disproportionately degrades its target category while leaving others intact. Test set evaluation on 5,000 samples confirms generalization within 0.21 F1 points of validation results. The framework operates with any instruction-following vision-language model and produces citation-linked rationales for human review.
>
---
#### [replaced 015] The High Cost of Incivility: Quantifying Interaction Inefficiency via Multi-Agent Monte Carlo Simulations
- **分类: cs.AI; cs.CL; cs.CY; cs.MA**

- **简介: 该论文属于社会行为建模任务，旨在量化职场不文明对沟通效率的影响。通过多智能体模拟，分析有毒互动导致的对话延迟，提供伦理且可重复的研究方法。**

- **链接: [https://arxiv.org/pdf/2512.08345](https://arxiv.org/pdf/2512.08345)**

> **作者:** Benedikt Mangold
>
> **备注:** 8 figures, 3 tables
>
> **摘要:** Workplace toxicity is widely recognized as detrimental to organizational culture, yet quantifying its direct impact on operational efficiency remains methodologically challenging due to the ethical and practical difficulties of reproducing conflict in human subjects. This study leverages Large Language Model (LLM) based Multi-Agent Systems to simulate 1-on-1 adversarial debates, creating a controlled "sociological sandbox". We employ a Monte Carlo method to simulate hundrets of discussions, measuring the convergence time (defined as the number of arguments required to reach a conclusion) between a baseline control group and treatment groups involving agents with "toxic" system prompts. Our results demonstrate a statistically significant increase of approximately 25\% in the duration of conversations involving toxic participants. We propose that this "latency of toxicity" serves as a proxy for financial damage in corporate and academic settings. Furthermore, we demonstrate that agent-based modeling provides a reproducible, ethical alternative to human-subject research for measuring the mechanics of social friction.
>
---
#### [replaced 016] AP-BMM: Approximating Capability-Efficiency Pareto Sets of LLMs via Asynchronous Prior-guided Bayesian Model Merging
- **分类: cs.LG; cs.CL; cs.NE**

- **简介: 该论文属于大语言模型优化任务，旨在解决能力与效率的权衡问题。通过异步贝叶斯模型融合方法，提升帕累托集近似效果。**

- **链接: [https://arxiv.org/pdf/2512.09972](https://arxiv.org/pdf/2512.09972)**

> **作者:** Kesheng Chen; Yamin Hu; Zhenqian Zhu; Yiya Diao; Wenjian Luo
>
> **摘要:** Navigating the capability--efficiency trade-off in Large Language Models (LLMs) requires approximating a high-quality Pareto set. Existing model merging research has focused predominantly on coarse model-level operators, which are easy to apply but offer limited control over the trade-off geometry. Layer-wise merging is more expressive, yet current methods still suffer from two bottlenecks: they treat the high-dimensional fusion space as an unstructured black box, and they rely on synchronous optimization despite highly uneven LLM evaluation latency. We propose Asynchronous Prior-guided Bayesian Model Merging (AP-BMM), which addresses these issues with a discrepancy-derived importance prior that initializes the surrogate geometry and an event-driven optimization loop built on pending-aware hypervolume improvement. Under a common evaluation budget, AP-BMM yields stronger Pareto-set approximations than both synchronous layer-wise baselines and representative model-level merging methods, with higher hypervolume and broader coverage of the trade-off frontier. Against the synchronous Bayesian baseline, it also achieves substantially shorter wall-clock time. Code: this https URL.
>
---
#### [replaced 017] Peer-Predictive Self-Training for Language Model Reasoning
- **分类: cs.CL; cs.AI; cs.GT**

- **简介: 该论文提出PST框架，用于语言模型的自监督训练，解决无外部监督下的持续自我提升问题，通过多模型协作生成更可靠答案并进行反馈优化。**

- **链接: [https://arxiv.org/pdf/2604.13356](https://arxiv.org/pdf/2604.13356)**

> **作者:** Shi Feng; Hanlin Zhang; Fan Nie; Sham Kakade; Yiling Chen
>
> **备注:** 19 pages, 5 figures
>
> **摘要:** Mechanisms for continued self-improvement of language models without external supervision remain an open challenge. We propose Peer-Predictive Self-Training (PST), a label-free fine-tuning framework in which multiple language models improve collaboratively by leveraging a cross-model aggregated response as an internal training signal. Given a prompt question, the models generate responses sequentially; the final aggregated answer, often more reliable than individual responses in practice, serves as an internal target for learning. We measure how informative each intermediate response is about the aggregate using pointwise mutual information (PMI), and use this signal to scale self-training updates. Responses already aligned with the aggregate are updated less, while less informative or misaligned responses are updated more. On mathematical reasoning benchmarks (SimulEq, Math500, and MultiArith), PST improves exact-match accuracy by 2.2 to 4.3 percentage points across Gemma-2-2B, LLaMA-3.2-1B, and Qwen-2.5-1.5B, and reduces the average generator-verifier gap (GV-Gap) by 26 to 40 percent, while requiring no external supervision or teacher-student hierarchy and relying solely on cross-model interactions. These results suggest that cross-model generations and peer-predictive feedback can serve as an effective approach for self-supervised training.
>
---
#### [replaced 018] OLaPh: Optimal Language Phonemizer
- **分类: cs.CL**

- **简介: 该论文属于语音合成中的文字转音素（G2P）任务，旨在提升对未登录词的泛化能力。提出OLaPh框架，结合多语言词典与神经方法，有效处理OOV数据，并用于生成高质量训练数据。**

- **链接: [https://arxiv.org/pdf/2509.20086](https://arxiv.org/pdf/2509.20086)**

> **作者:** Johannes Wirth
>
> **备注:** 11 pages, 1 figure, 4 tables
>
> **摘要:** Phonemization is a critical component in text-to-speech synthesis. Traditional approaches rely on deterministic transformations and lexica, while neural methods offer potential for higher generalization on out-of-vocabulary (OOV) terms. This work introduces OLaPh (Optimal Language Phonemizer), a hybrid framework that integrates extensive multilingual lexica with advanced NLP techniques and a statistical subword segmentation function. Evaluations on the WikiPron benchmark show that the OLaPh framework significantly outperforms established baselines in overall accuracy and maintains robustness on OOV data through advanced fallback mechanisms. To further explore neural generalization, we utilize the framework to synthesize a high-consistency training corpus for an instruction-tuned Large Language Model (LLM). While the deterministic framework remains more accurate overall, the LLM demonstrates strong generalization, matching or partly exceeding the framework's performance. This suggests that the LLM successfully internalized phonetic intuitions from the synthetic data that transcend the framework's capabilities. Together, these tools provide a comprehensive, open-source resource for multilingual G2P research.
>
---
#### [replaced 019] PARASITE: Conditional System Prompt Poisoning to Hijack LLMs
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于安全任务，针对LLM系统提示的供应链漏洞，提出PARASITE框架，通过注入“睡眠代理”在特定查询下触发恶意响应，同时保持正常功能。**

- **链接: [https://arxiv.org/pdf/2505.16888](https://arxiv.org/pdf/2505.16888)**

> **作者:** Viet Pham; Thai Le
>
> **备注:** ACL 2026 Main
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed via third-party system prompts downloaded from public marketplaces. We identify a critical supply-chain vulnerability: conditional system prompt poisoning, where an adversary injects a ``sleeper agent'' into a benign-looking prompt. Unlike traditional jailbreaks that aim for broad refusal-breaking, our proposed framework, PARASITE, optimizes system prompts to trigger LLMs to output targeted, compromised responses only for specific queries (e.g., ``Who should I vote for the US President?'') while maintaining high utility on benign inputs. Operating in a strict black-box setting without model weight access, PARASITE utilizes a two-stage optimization including a global semantic search followed by a greedy lexical refinement. Tested on open-source models and commercial APIs (GPT-4o-mini, GPT-3.5), PARASITE achieves up to 70\% F1 reduction on targeted queries with minimal degradation to general capabilities. We further demonstrate that these poisoned prompts evade standard defenses, including perplexity filters and typo-correction, by exploiting the natural noise found in real-world system prompts. Our code and data are available at this https URL. WARNING: Our paper contains examples that might be sensitive to the readers!
>
---
#### [replaced 020] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于对话问答任务，旨在解决对话中意图演变和上下文理解问题。提出ChatR1框架，通过强化学习实现搜索与推理的交互，提升问答效果。**

- **链接: [https://arxiv.org/pdf/2510.13312](https://arxiv.org/pdf/2510.13312)**

> **作者:** Simon Lupart; Mohammad Aliannejadi; Evangelos Kanoulas
>
> **备注:** 18 pages, 9 figures, Main ACL 2026 Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics, July 2--7, 2026, San Diego, California
>
> **摘要:** We present ChatR1, a reasoning framework based on reinforcement learning (RL) for conversational question answering (CQA). Reasoning plays an important role in CQA, where user intent evolves across dialogue turns, and utterances are often underspecified, requiring contextual interpretation, query reformulation, and dynamic coordination between retrieval and generation. Unlike static `rewrite, retrieve, and generate' pipelines, ChatR1 interleaves search and reasoning across turns, enabling exploratory and adaptive behaviors learned through RL. To address the challenge of sparse and delayed rewards in RL, we propose an intent-aware reward that provides turn-level feedback by aligning retrieval and reasoning with evolving user goals. ChatR1 demonstrates strong performance on both 3B and 7B model backbones, outperforming competitive models on five CQA datasets, measured by different metrics (F1, BERTScore, and LLM-as-judge). We include a diverse set of CQA datasets to cover topic shifts, evolving intents, mixed-initiative dialogues, and multi-document grounding, testing ChatR1's performance from various aspects. Ablation studies confirm the effectiveness of the intent-aware reward. Our analyses further reveal diverse reasoning trajectories and effective use of the search tool. ChatR1 also generalizes robustly across domains, demonstrating that RL-based reasoning enables more flexible and context-aware behavior than static CQA pipelines.
>
---
#### [replaced 021] ZoFia: Zero-Shot Fake News Detection with Entity-Guided Retrieval and Multi-LLM Interaction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于虚假新闻检测任务，旨在解决LLM在时间敏感新闻中的知识局限和偏差问题。提出ZoFia框架，通过实体引导检索和多LLM协作提升检测效果。**

- **链接: [https://arxiv.org/pdf/2511.01188](https://arxiv.org/pdf/2511.01188)**

> **作者:** Lvhua Wu; Xuefeng Jiang; Sheng Sun; Yan Lei; Tian Wen; Yuwei Wang; Min Liu
>
> **摘要:** The rapid spread of fake news threatens social stability and public trust, highlighting the urgent need for its effective detection. Although large language models (LLMs) show potential in fake news detection, they are limited by knowledge cutoff and easily generate factual hallucinations when handling time-sensitive news. Furthermore, the thinking of a single LLM easily falls into early stance locking and confirmation bias, making it hard to handle both content reasoning and fact checking simultaneously. To address these challenges, we propose ZoFia, a two-stage zero-shot fake news detection framework. In the first retrieval stage, we propose novel Hierarchical Salience and Salience-Calibrated Minimum Marginal Relevance (SC-MMR) algorithm to extract core entities accurately, which drive dual-source retrieval to overcome knowledge and evidence gaps. In the subsequent stage, a multi-agent system conducts multi-perspective reasoning and verification in parallel and achieves an explainable and robust result via adversarial debate. Comprehensive experiments on two public datasets show that ZoFia outperforms existing zero-shot baselines and even most few-shot methods. Our code has been open-sourced to facilitate the research community at this https URL.
>
---
#### [replaced 022] PDF-WuKong: A Large Multimodal Model for Efficient Long PDF Reading with End-to-End Sparse Sampling
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态文档理解任务，旨在解决长PDF文档中文本与图像混合内容的高效问答问题。提出PDF-WuKong模型，采用稀疏采样提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2410.05970](https://arxiv.org/pdf/2410.05970)**

> **作者:** Xudong Xie; Hao Yan; Liang Yin; Yang Liu; Jing Ding; Minghui Liao; Yuliang Liu; Wei Chen; Xiang Bai
>
> **备注:** Accepted by International Journal of Computer Vision (IJCV)
>
> **摘要:** Multimodal document understanding is a challenging task to process and comprehend large amounts of textual and visual information. Recent advances in Large Language Models (LLMs) have significantly improved the performance of this task. However, existing methods typically focus on either plain text or a limited number of document images, struggling to handle long PDF documents with interleaved text and images, especially for academic papers. In this paper, we introduce PDF-WuKong, a multimodal large language model (MLLM) that is designed to enhance multimodal question-answering (QA) for long PDF documents. PDF-WuKong incorporates a sparse sampler that operates on both text and image representations, significantly improving the efficiency and capability of the MLLM. The sparse sampler selects the paragraphs or diagrams most pertinent to user queries. To effectively train and evaluate our model, we construct PaperPDF, a dataset consisting of a broad collection of English and Chinese academic papers. Multiple strategies are proposed to build high-quality 1.1 million QA pairs along with their corresponding evidence sources. Experimental results demonstrate the superiority and high efficiency of our approach over other models on the task of long multimodal document understanding, surpassing proprietary products by an average of 8.6% on F1. Our code and dataset will be released at this https URL.
>
---
#### [replaced 023] CorpusQA: A 10 Million Token Benchmark for Corpus-Level Analysis and Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CorpusQA，解决长文本推理任务中的全局信息整合问题，通过生成10万词数据集，评估模型在大规模文档中的推理能力。**

- **链接: [https://arxiv.org/pdf/2601.14952](https://arxiv.org/pdf/2601.14952)**

> **作者:** Zhiyuan Lu; Chenliang Li; Yingcheng Shi; Weizhou Shen; Ming Yan; Fei Huang
>
> **摘要:** While large language models now handle million-token contexts, their capacity for reasoning across entire document repositories remains largely untested. Existing benchmarks are inadequate, as they are mostly limited to single long texts or rely on a "sparse retrieval" assumption-that answers can be derived from a few relevant chunks. This assumption fails for true corpus-level analysis, where evidence is highly dispersed across hundreds of documents and answers require global integration, comparison, and statistical aggregation. To address this critical gap, we introduce CorpusQA, a new benchmark scaling up to 10 million tokens, generated via a novel data synthesis framework. By decoupling reasoning from textual representation, this framework creates complex, computation-intensive queries with programmatically guaranteed ground-truth answers, challenging systems to perform holistic reasoning over vast, unstructured text without relying on fallible human annotation. We further demonstrate the utility of our framework beyond evaluation, showing that fine-tuning on our synthesized data effectively enhances an LLM's general long-context reasoning capabilities. Extensive experiments reveal that even state-of-the-art long-context LLMs struggle as input length increases, and standard retrieval-augmented generation systems collapse entirely. Our findings indicate that memory-augmented agentic architectures offer a more robust alternative, suggesting a critical shift is needed from simply extending context windows to developing advanced architectures for global information synthesis.
>
---
#### [replaced 024] CiteAudit: You Cited It, But Did You Read It? A Benchmark for Verifying Scientific References in the LLM Era
- **分类: cs.CL; cs.DL**

- **简介: 该论文属于科学引用验证任务，解决LLM生成虚假引用的问题。构建基准数据集，提出多阶段验证框架，提升引用可信度。**

- **链接: [https://arxiv.org/pdf/2602.23452](https://arxiv.org/pdf/2602.23452)**

> **作者:** Zhengqing Yuan; Kaiwen Shi; Zheyuan Zhang; Lichao Sun; Nitesh V. Chawla; Yanfang Ye
>
> **备注:** After internal review, we found systematic errors in the benchmark and reference verification pipeline that affect core results and conclusions. As these issues are fundamental and cannot be addressed by revision or replacement, we request withdrawal to avoid potential misuse
>
> **摘要:** Scientific research relies on accurate citation for attribution and integrity, yet large language models (LLMs) introduce a new risk: fabricated references that appear plausible but correspond to no real publications. Such hallucinated citations have already been observed in submissions and accepted papers at major machine learning venues, exposing vulnerabilities in peer review. Meanwhile, rapidly growing reference lists make manual verification impractical, and existing automated tools remain fragile to noisy and heterogeneous citation formats and lack standardized evaluation. We present the first comprehensive benchmark and detection framework for hallucinated citations in scientific writing. Our multi-agent verification pipeline decomposes citation checking into claim extraction, evidence retrieval, passage matching, reasoning, and calibrated judgment to assess whether a cited source truly supports its claim. We construct a large-scale human-validated dataset across domains and define unified metrics for citation faithfulness and evidence alignment. Experiments with state-of-the-art LLMs reveal substantial citation errors and show that our framework significantly outperforms prior methods in both accuracy and interpretability. This work provides the first scalable infrastructure for auditing citations in the LLM era and practical tools to improve the trustworthiness of scientific references.
>
---
#### [replaced 025] Hearing to Translate: The Effectiveness of Speech Modality Integration into LLMs
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文研究语音翻译任务，比较SpeechLLMs与传统级联系统的性能，旨在验证语音模态集成对翻译质量的影响。**

- **链接: [https://arxiv.org/pdf/2512.16378](https://arxiv.org/pdf/2512.16378)**

> **作者:** Sara Papi; Javier Garcia Gilabert; Zachary Hopton; Vilém Zouhar; Carlos Escolano; Gerard I. Gállego; Jorge Iranzo-Sánchez; Ahrii Kim; Dominik Macháček; Patricia Schmidtova; Maike Züfle
>
> **备注:** Project available at this https URL | Accepted at TACL, this version is a pre-MIT Press publication version
>
> **摘要:** As Large Language Models (LLMs) expand beyond text, integrating speech as a native modality has given rise to SpeechLLMs, which directly process spoken language and enable speech-to-text translation (ST) and other downstream tasks, bypassing traditional transcription-based pipelines. Whether this integration improves ST quality over established cascaded architectures, however, remains an open question. We present Hearing to Translate, the first comprehensive test suite rigorously benchmarking 6 state-of-the-art SpeechLLMs against 16 strong direct and cascade systems that couple leading speech foundation models (SFM), with multilingual LLMs. Our analysis spans 16 benchmarks, 13 language pairs, and 9 challenging conditions, including disfluent, noisy, and long-form speech. Across this extensive evaluation, we find that cascaded systems remain the most reliable solution overall, but most recent SpeechLLMs can match or even outperform cascades in various settings while SFMs lag behind both, highlighting that integrating an LLM, either within the model or in a pipeline, is essential for high-quality speech translation.
>
---
#### [replaced 026] TeachMaster: Generative Teaching via Code
- **分类: cs.CY; cs.AI; cs.CL; cs.HC; cs.MA**

- **简介: 该论文提出Generative Teaching，解决在线教育内容生成效率低的问题。通过TeachMaster框架，利用代码作为语义媒介，实现教育视频的自动化生成。**

- **链接: [https://arxiv.org/pdf/2601.04204](https://arxiv.org/pdf/2601.04204)**

> **作者:** Yuheng Wang; Runde Yang; Lin Wu; Jie Zhang; Jingru Fan; Tianle Zhou; Ruoyu Fu; Huatao Li; Ruijie Shi; Siheng Chen; Weinan E; Chen Qian
>
> **备注:** Accepted to ACL 2026; this https URL
>
> **摘要:** The scalability of high-quality online education is hindered by the high costs and slow cycles of manual content creation. Despite advancements in video generation, current approaches often fail to ensure pedagogical structure and precise control due to their pixel-level, black-box nature. In this paper, we propose Generative Teaching, a novel paradigm shifting educators from manual creators to high-level directors who focus on pedagogical intents while agents handle the execution. To realize this vision, we introduce TeachMaster, a multi-agent framework that leverages code as an intermediate semantic medium. Unlike traditional video generation methods, TeachMaster orchestrates a collaborative team of agents, spanning planning, design, and rendering, to automate the production of interpretable, editable, and curriculum-ready educational videos. Experiments validate that TeachMaster significantly boosts production efficiency without compromising structural coherence or visual fidelity, slashing production costs to only 0.3% of traditional online course videos and providing a robust solution for scalable education.
>
---
#### [replaced 027] Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究强化学习中的策略蒸馏任务，针对采样token OPD的稳定性问题，提出改进方法提升训练稳定性和性能。**

- **链接: [https://arxiv.org/pdf/2603.25562](https://arxiv.org/pdf/2603.25562)**

> **作者:** Yuqian Fu; Haohuan Huang; Kaiwen Jiang; Jiacai Liu; Zhuo Jiang; Yuanheng Zhu; Dongbin Zhao
>
> **摘要:** On-policy distillation (OPD) is increasingly used in LLM post-training because it can leverage a teacher model to provide dense supervision on student rollouts. The standard implementation, however, usually reduces distribution matching to a sampled-token log-ratio, which can make the learning signal fragile on long rollouts whose prefixes drift away from the teacher's typical support. We revisit this formulation from both theoretical and implementation perspectives. Theoretically, token-level OPD is biased relative to sequence-level reverse-KL minimization, but admits a substantially tighter worst-case variance bound; a controlled synthetic study further shows that stronger future-reward coupling increases gradient variance and destabilizes training. Empirically, we identify three failure modes of sampled-token OPD: imbalanced token-level supervision, unreliable teacher guidance on student-generated prefixes, and tokenizer or special-token mismatch. These findings motivate teacher top-K local support matching, a truncated reverse-KL objective that compares teacher and student distributions over a teacher-supported token set at each prefix, together with top-p rollout sampling and special-token masking. Across single-task reasoning and multi-task benchmarks spanning agentic and reasoning settings, this objective improves optimization stability and yields a +19.8% performance gain over standard sampled-token OPD baselines, providing a practical recipe for more stable on-policy distillation.
>
---
#### [replaced 028] POPI: Personalizing LLMs via Optimized Natural Language Preference Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出POPI框架，解决个性化大语言模型问题。通过自然语言接口分离推理与生成，提升个性化效果并降低上下文开销。**

- **链接: [https://arxiv.org/pdf/2510.17881](https://arxiv.org/pdf/2510.17881)**

> **作者:** Yizhuo Chen; Xin Liu; Ruijie Wang; Zheng Li; Pei Chen; Changlong Yu; Qingyu Yin; Priyanka Nigam; Meng Jiang; Bing Yin
>
> **摘要:** Large language models (LLMs) are typically aligned with population-level preferences, despite substantial variation across individual users. We introduce POPI, a user-level personalization framework that separates the problem into two components connected by a natural-language interface: a shared inference model that distills heterogeneous user signals into a concise preference summary, and a shared generator that conditions on this summary to produce personalized responses. Both components are trained under a unified preference-optimization objective, with reinforcement learning handling the non-differentiable inference step. This objective decomposes into generator approximation error and summary informativeness, revealing how a single loss simultaneously drives accurate generation and informative summarization. Because the interface is natural language, learned summaries can be inferred once per user and reused across different generators -- including frozen, black-box commercial APIs. Across four personalization benchmarks, POPI generally improves personalization quality while reducing context overhead by up to an order of magnitude.
>
---
#### [replaced 029] The Rise of Verbal Tics in Large Language Models: A Systematic Analysis Across Frontier Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，研究大模型中口头习惯现象。通过分析多个模型，提出VTI指标，揭示其与拟人化和自然度的关系，旨在改善人机交互真实性。**

- **链接: [https://arxiv.org/pdf/2604.19139](https://arxiv.org/pdf/2604.19139)**

> **作者:** Shuai Wu; Xue Li; Yanna Feng; Yufang Li; Zhijun Wang; Ran Wang
>
> **备注:** 20 pages, 17 figures, 8 tables; code and data available at this https URL DOI: https://doi.org/10.5281/zenodo.19767626
>
> **摘要:** As Large Language Models (LLMs) continue to evolve through alignment techniques such as Reinforcement Learning from Human Feedback (RLHF) and Constitutional AI, a growing and increasingly conspicuous phenomenon has emerged: the proliferation of verbal tics, repetitive, formulaic linguistic patterns that pervade model outputs. These range from sycophantic openers (That's a great question!, Awesome!) to pseudo-empathetic affirmations (I completely understand your concern, I'm right here to catch you) and overused vocabulary (delve, tapestry, nuanced). In this paper, we present a systematic analysis of the verbal tic phenomenon across eight state-of-the-art LLMs: GPT-5.4, Claude Opus 4.7, Gemini 3.1 Pro, Grok 4.2, Doubao-Seed-2.0-pro, Kimi K2.5, DeepSeek V3.2, and MiMo-V2-Pro. Utilizing a custom evaluation framework for standardized API-based evaluation, we assess 10,000 prompts across 10 task categories in both English and Chinese, yielding 160,000 model responses. We introduce the Verbal Tic Index (VTI), a composite metric quantifying tic prevalence, and analyze its correlation with sycophancy, lexical diversity, and human-perceived naturalness. Our findings reveal significant inter-model variation: Gemini 3.1 Pro exhibits the highest VTI (0.590), while DeepSeek V3.2 achieves the lowest (0.295). We further demonstrate that verbal tics accumulate over multi-turn conversations, are amplified in subjective tasks, and show distinct cross-lingual patterns. Human evaluation (N = 120) confirms a strong inverse relationship between sycophancy and perceived naturalness (r = -0.87, p < 0.001). These results underscore the alignment tax of current training paradigms and highlight the urgent need for more authentic human-AI interaction frameworks.
>
---
#### [replaced 030] AdaRubric: Task-Adaptive Rubrics for LLM Agent Evaluation
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ADARUBRIC，用于LLM代理评估，解决固定评分标准无法适应不同任务的问题。通过生成任务特定评分标准，提升评估准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.21362](https://arxiv.org/pdf/2603.21362)**

> **作者:** Liang Ding
>
> **摘要:** LLM-as-Judge evaluation fails agent tasks because a fixed rubric cannot capture what matters for this task: code debugging demands Correctness and Error Handling; web navigation demands Goal Alignment and Action Efficiency. We present ADARUBRIC, which closes this gap by generating task-specific evaluation rubrics on the fly from task descriptions, scoring trajectories step-by-step with confidence-weighted per-dimension feedback, and filtering preference pairs with the novel DimensionAwareFilter - a provably necessary condition for preventing high-scoring dimensions from masking dimension-level failures. On WebArena and ToolBench, ADARUBRIC achieves Pearson r=0.79 human correlation (+0.16 over the best static baseline) with deployment-grade reliability (Krippendorff's $\alpha$=0.83). DPO agents trained on ADARUBRIC preference pairs gain +6.8 to +8.5 pp task success over Prometheus across three benchmarks; gains transfer to SWE-bench code repair (+4.9 pp) and accelerate PPO convergence by +6.6 pp at 5K steps - both without any rubric engineering. Code: this https URL.
>
---
#### [replaced 031] A BERTology View of LLM Orchestrations: Token- and Layer-Selective Probes for Efficient Single-Pass Classification
- **分类: cs.CL**

- **简介: 该论文针对大模型分类任务，提出轻量级探针，利用预训练模型的隐藏状态进行高效单次前向分类，解决多模型系统带来的延迟和资源消耗问题。**

- **链接: [https://arxiv.org/pdf/2601.13288](https://arxiv.org/pdf/2601.13288)**

> **作者:** Gonzalo Ariel Meyoyan; Luciano Del Corro
>
> **备注:** Accepted to ACL 2026 (Main Conference)
>
> **摘要:** Production LLM systems often rely on separate models for safety and other classification-heavy steps, increasing latency, VRAM footprint, and operational complexity. We instead reuse computation already paid for by the serving LLM: we train lightweight probes on its hidden states and predict labels in the same forward pass used for generation. We frame classification as representation selection over the full token-layer hidden-state tensor, rather than committing to a fixed token or fixed layer (e.g., first-token logits or final-layer pooling). To implement this, we introduce a two-stage aggregator that (i) summarizes tokens within each layer and (ii) aggregates across layer summaries to form a single representation for classification. We instantiate this template with direct pooling, a 100K-parameter scoring-attention gate, and a downcast multi-head self-attention (MHA) probe with up to 35M trainable parameters. Across safety and sentiment benchmarks our probes improve over logit-only reuse (e.g., MULI) and are competitive with substantially larger task-specific baselines, while preserving near-serving latency and avoiding the VRAM and latency costs of a separate guard-model pipeline. Multi-backbone experiments on dense and mixture-of-experts architectures (Llama-3.2-3B, GPT-OSS-20B, Qwen3-30B-A3B) confirm that these findings generalize beyond a single model family.
>
---
#### [replaced 032] The Consensus Trap: Dissecting Subjectivity and the "Ground Truth" Illusion in Data Annotation
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于数据标注领域，探讨"共识陷阱"问题，指出"地面真实"概念的缺陷，提出构建多元标注体系以反映人类经验多样性。**

- **链接: [https://arxiv.org/pdf/2602.11318](https://arxiv.org/pdf/2602.11318)**

> **作者:** Sheza Munir; Benjamin Mah; Krisha Kalsi; Shivani Kapania; Julian Posada; Edith Law; Ding Wang; Syed Ishtiaque Ahmed
>
> **摘要:** In machine learning, "ground truth" refers to the assumed correct labels used to train and evaluate models. However, the foundational "ground truth" paradigm rests on a positivistic fallacy that treats human disagreement as technical noise rather than a vital sociotechnical signal. This systematic literature review analyzes research published between 2020 and 2025 across seven premier venues: ACL, AIES, CHI, CSCW, EAAMO, FAccT, and NeurIPS, investigating the mechanisms in data annotation practices that facilitate this "consensus trap". Our reflexive thematic analysis of 346 papers reveals that systemic failures in positional legibility, combined with the recent architectural shift toward human-as-verifier models, specifically the reliance on model-mediated annotations, introduce deep-seated anchoring bias and effectively remove human voices from the loop. We further demonstrate how geographic hegemony imposes Western norms as universal benchmarks, often enforced by the performative alignment of precarious data workers who prioritize requester compliance over honest subjectivity to avoid economic penalties. Critiquing the "noisy sensor" fallacy, where statistical models misdiagnose pluralism as error, we argue for reclaiming disagreement as a high-fidelity signal essential for building culturally competent models. To address these systemic tensions, we propose a roadmap for pluralistic annotation infrastructures that shift the objective from discovering a singular "right" answer to mapping the diversity of human experience.
>
---
#### [replaced 033] Rethinking Parameter Sharing for LLM Fine-Tuning with Multiple LoRAs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对多任务和联邦微调中的参数共享问题，提出ALoRA和Fed-ALoRA方法，通过异构矩阵设计提升性能平衡与准确性。**

- **链接: [https://arxiv.org/pdf/2509.25414](https://arxiv.org/pdf/2509.25414)**

> **作者:** Hao Ban; Kaiyi Ji
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Large language models are often adapted using parameter-efficient techniques such as Low-Rank Adaptation (LoRA), formulated as $y = W_0x + BAx$, where $W_0$ is the pre-trained parameters and $x$ is the input to the adapted layer. While multi-adapter extensions often employ multiple LoRAs, prior studies suggest that the inner $A$ matrices are highly similar during training and thus suitable for sharing. We revisit this phenomenon and find that this similarity is largely attributable to the identical initialization rather than shared knowledge, with $B$ playing a more critical role in knowledge encoding and transfer. Motivated by these insights, we propose \textbf{ALoRA}, an asymmetric multi-LoRA design with multiple $A$ matrices and a single shared $B$ in multi-task fine-tuning, and \textbf{Fed-ALoRA}, which shares $B$ across clients in federated fine-tuning under both homogeneous and heterogeneous settings, through a novel matrix decomposition strategy to accommodate heterogeneous ranks across clients. Experiments on commonsense reasoning, math reasoning, multi-task NLP dataset, and federated NLP dataset demonstrate that our methods achieve more balanced performance across tasks with comparable or superior average accuracy relative to existing multi-LoRA approaches. The code is available at this https URL.
>
---
#### [replaced 034] HiRAS: A Hierarchical Multi-Agent Framework for Paper-to-Code Generation and Execution
- **分类: cs.CL**

- **简介: 该论文属于论文到代码生成任务，旨在解决现有方法在实验复现中的协调不足与评估缺陷。提出HiRAS框架，采用分层多智能体协作，并引入改进的评估协议P2C-Ex。**

- **链接: [https://arxiv.org/pdf/2604.17745](https://arxiv.org/pdf/2604.17745)**

> **作者:** Hanhua Hong; Yizhi LI; Jiaoyan Chen; Sophia Ananiadou; Xiaoli Li; Jung-jae Kim; Chenghua Lin
>
> **备注:** 29 pages
>
> **摘要:** Recent advances in large language models have highlighted their potential to automate computational research, particularly reproducing experimental results. However, existing approaches still use fixed sequential agent pipelines with weak global coordination, which limits their robustness and overall performance. In this work, we propose Hierarchical Research Agent System (HiRAS), a hierarchical multi-agent framework for end-to-end experiment reproduction that employs supervisory manager agents to coordinate specialised agents across fine-grained stages. We also identify limitations in the reference-free evaluation of the Paper2Code benchmark and introduce Paper2Code-Extra (P2C-Ex), a refined protocol that incorporates repository-level information and better aligns with the original reference-based metric. We conduct extensive evaluation, validating the effectiveness and robustness of our proposed methods, and observing improvements, including >10\% relative performance gain beyond the previous state-of-the-art using open-source backbone models and significantly reduced hallucination in evaluation. Our work is available on GitHub: this https URL.
>
---
#### [replaced 035] Reinforcement Learning with Backtracking Feedback
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于安全增强任务，旨在提升大语言模型对对抗攻击的鲁棒性。通过强化学习与回溯反馈机制，使模型能动态纠正生成错误，提高安全性。**

- **链接: [https://arxiv.org/pdf/2602.08377](https://arxiv.org/pdf/2602.08377)**

> **作者:** Bilgehan Sel; Vaishakh Keshava; Phillip Wallis; Lukas Rutishauser; Ming Jin; Dingcheng Li
>
> **备注:** NeurIPS 2025
>
> **摘要:** Addressing the critical need for robust safety in Large Language Models (LLMs), particularly against adversarial attacks and in-distribution errors, we introduce Reinforcement Learning with Backtracking Feedback (RLBF). This framework advances upon prior methods, such as BSAFE, by primarily leveraging a Reinforcement Learning (RL) stage where models learn to dynamically correct their own generation errors. Through RL with critic feedback on the model's live outputs, LLMs are trained to identify and recover from their actual, emergent safety violations by emitting an efficient "backtrack by x tokens" signal, then continuing generation autoregressively. This RL process is crucial for instilling resilience against sophisticated adversarial strategies, including middle filling, Greedy Coordinate Gradient (GCG) attacks, and decoding parameter manipulations. To further support the acquisition of this backtracking capability, we also propose an enhanced Supervised Fine-Tuning (SFT) data generation strategy (BSAFE+). This method improves upon previous data creation techniques by injecting violations into coherent, originally safe text, providing more effective initial training for the backtracking mechanism. Comprehensive empirical evaluations demonstrate that RLBF significantly reduces attack success rates across diverse benchmarks and model scales, achieving superior safety outcomes while critically preserving foundational model utility.
>
---
#### [replaced 036] How Much Heavy Lifting Can an Agent Harness Do?: Measuring the LLM's Residual Role in a Planning Agent
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究代理框架中语言模型的剩余作用，通过分解代理层结构，评估各层对任务表现的贡献，旨在明确语言模型在规划代理中的实际角色。**

- **链接: [https://arxiv.org/pdf/2604.07236](https://arxiv.org/pdf/2604.07236)**

> **作者:** Sungwoo Jung; Seonil Son
>
> **摘要:** Agent harnesses -- the stateful programs that wrap a language model and decide what it sees at each step -- are now known to change end-to-end performance on a fixed model by as much as six times. That observation raises a question asked less often than it should be: once the harness is serious, how much of an agent's competence does the harness itself already carry, and how much genuinely still needs the LLM? We study this in noisy Collaborative Battleship, a partially observable planning setting with belief update, information-gathering questions, and uncertainty-aware action selection. We externalize a planning harness into four progressively richer layers -- posterior belief tracking, declarative planning, symbolic reflection, and an LLM-backed revision gate -- and report per-layer contribution under a common runtime. We report \emph{win rate} as the primary, game-level metric and \emph{F1} as a secondary, local-targeting indicator, and pre-specify \emph{heavy lifting} as the single largest positive marginal to the primary metric. Across 54 games, the declarative planning layer does most of the heavy lifting under this criterion, raising win rate from 50.0\% (Wilson 95\% CI $[37.1,62.9]$) to 74.1\% ($[61.1,83.9]$) over a belief-only harness (+24.1pp, +0.017 F1). Symbolic reflection is mechanistically real but calibration-sensitive, shifting board-level outcomes by up to $\pm0.140$ F1 without being net-positive on aggregate. LLM-backed revision activates on only 4.3\% of turns at the strictest confidence threshold and yields a small, non-monotonic change (+0.005 F1, -3.7pp win rate). The contribution is methodological: once harness layers are made externally measurable, one can ask not only how far the harness already carries the agent, but also where the LLM's role is actually residual rather than central.
>
---
#### [replaced 037] MathDuels: Evaluating LLMs as Problem Posers and Solvers
- **分类: cs.CL; cs.SE**

- **简介: 该论文提出MathDuels，用于评估大模型作为问题生成者和求解者的双重能力。解决传统基准无法区分模型差异的问题，通过自对弈机制和难度评估模型，揭示模型能力差异。**

- **链接: [https://arxiv.org/pdf/2604.21916](https://arxiv.org/pdf/2604.21916)**

> **作者:** Zhiqiu Xu; Shibo Jin; Shreya Arya; Mayur Naik
>
> **摘要:** As frontier language models attain near-ceiling performance on static mathematical benchmarks, existing evaluations are increasingly unable to differentiate model capabilities, largely because they cast models solely as solvers of fixed problem sets. We introduce MathDuels, a self-play benchmark in which models occupy dual roles: each authors math problems under adversarial prompting and solves problems authored by every other participant. Problems are produced through a three-stage generation pipeline (meta-prompting, problem generation, and difficulty amplification), and validated by an independent verifier that excludes ill-posed questions. A Rasch model (Rasch, 1993) jointly estimates solver abilities and problem difficulties; author quality is derived from the difficulties of each model's authored problems. Experiments across 19 frontier models reveal that authoring and solving capabilities are partially decoupled, and that dual-role evaluation reveals capability separations invisible in single-role benchmarks. As newer models enter the arena, they produce problems that defeat previously dominant solvers, so the benchmark's difficulty co-evolves with participant strength rather than saturating at a fixed ceiling. We host a public leaderboard that updates as new models are released.
>
---
#### [replaced 038] Building a Precise Video Language with Human-AI Oversight
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文聚焦视频语言模型任务，解决视频精准描述问题。通过结构化规范和人机协作框架，提升视频字幕的准确性与细节控制能力。**

- **链接: [https://arxiv.org/pdf/2604.21718](https://arxiv.org/pdf/2604.21718)**

> **作者:** Zhiqiu Lin; Chancharik Mitra; Siyuan Cen; Isaac Li; Yuhan Huang; Yu Tong Tiffany Ling; Hewei Wang; Irene Pi; Shihang Zhu; Ryan Rao; George Liu; Jiaxi Li; Ruojin Li; Yili Han; Yilun Du; Deva Ramanan
>
> **备注:** CVPR 2026 Highlight. Project page: this https URL
>
> **摘要:** Video-language models (VLMs) learn to reason about the dynamic visual world through natural language. We introduce a suite of open datasets, benchmarks, and recipes for scalable oversight that enable precise video captioning. First, we define a structured specification for describing subjects, scenes, motion, spatial, and camera dynamics, grounded by hundreds of carefully defined visual primitives developed with professional video creators such as filmmakers. Next, to curate high-quality captions, we introduce CHAI (Critique-based Human-AI Oversight), a framework where trained experts critique and revise model-generated pre-captions into improved post-captions. This division of labor improves annotation accuracy and efficiency by offloading text generation to models, allowing humans to better focus on verification. Additionally, these critiques and preferences between pre- and post-captions provide rich supervision for improving open-source models (Qwen3-VL) on caption generation, reward modeling, and critique generation through SFT, DPO, and inference-time scaling. Our ablations show that critique quality in precision, recall, and constructiveness, ensured by our oversight framework, directly governs downstream performance. With modest expert supervision, the resulting model outperforms closed-source models such as Gemini-3.1-Pro. Finally, we apply our approach to re-caption large-scale professional videos (e.g., films, commercials, games) and fine-tune video generation models such as Wan to better follow detailed prompts of up to 400 words, achieving finer control over cinematography including camera motion, angle, lens, focus, point of view, and framing. Our results show that precise specification and human-AI oversight are key to professional-level video understanding and generation. Data and code are available on our project page: this https URL
>
---
#### [replaced 039] CRISP: Persistent Concept Unlearning via Sparse Autoencoders
- **分类: cs.CL**

- **简介: 该论文属于知识删除任务，旨在解决LLM中 unwanted knowledge 的持久移除问题。提出CRISP方法，利用SAE实现参数高效、持久的概念遗忘。**

- **链接: [https://arxiv.org/pdf/2508.13650](https://arxiv.org/pdf/2508.13650)**

> **作者:** Tomer Ashuach; Dana Arad; Aaron Mueller; Martin Tutek; Yonatan Belinkov
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** As large language models (LLMs) are increasingly deployed in real-world applications, the need to selectively remove unwanted knowledge while preserving model utility has become paramount. Recent work has explored sparse autoencoders (SAEs) to perform precise interventions on monosemantic features. However, most SAE-based methods operate at inference time, which does not create persistent changes in the model's parameters. Such interventions can be bypassed or reversed by malicious actors with parameter access. We introduce CRISP, a parameter-efficient method for persistent concept unlearning using SAEs. CRISP automatically identifies salient SAE features across multiple layers and suppresses their activations. We experiment with two LLMs and show that our method outperforms prior approaches on safety-critical unlearning tasks from the WMDP benchmark, successfully removing harmful knowledge while preserving general and in-domain capabilities. Feature-level analysis reveals that CRISP achieves semantically coherent separation between target and benign concepts, allowing precise suppression of the target features.
>
---
#### [replaced 040] Bridging the Domain Divide: Supervised vs. Zero-Shot Clinical Section Segmentation from MIMIC-III to Obstetrics
- **分类: cs.CL**

- **简介: 该论文属于临床文本分段任务，旨在解决跨领域分割性能下降问题。通过构建新数据集、评估模型表现，并比较监督与零样本模型，提出有效方法提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.17513](https://arxiv.org/pdf/2602.17513)**

> **作者:** Baris Karacan; Barbara Di Eugenio; Patrick Thornton
>
> **备注:** 14 pages. Camera-ready version accepted at LREC 2026; includes minor revisions and an appendix. To appear in the conference proceedings
>
> **摘要:** Clinical free-text notes contain vital patient information. They are structured into labelled sections; recognizing these sections has been shown to support clinical decision-making and downstream NLP tasks. In this paper, we advance clinical section segmentation through three key contributions. First, we curate a new de-identified, section-labeled obstetrics notes dataset, to supplement the medical domains covered in public corpora such as MIMIC-III, on which most existing segmentation approaches are trained. Second, we systematically evaluate transformer-based supervised models for section segmentation on a curated subset of MIMIC-III (in-domain), and on the new obstetrics dataset (out-of-domain). Third, we conduct the first head-to-head comparison of supervised models for medical section segmentation with zero-shot large language models. Our results show that while supervised models perform strongly in-domain, their performance drops substantially out-of-domain. In contrast, zero-shot models demonstrate robust out-of-domain adaptability once hallucinated section headers are corrected. These findings underscore the importance of developing domain-specific clinical resources and highlight zero-shot segmentation as a promising direction for applying healthcare NLP beyond well-studied corpora, as long as hallucinations are appropriately managed.
>
---
#### [replaced 041] Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决RAG系统对语义无关特征的鲁棒性问题。通过提出SURE框架，量化并提升模型对虚假特征的抗干扰能力。**

- **链接: [https://arxiv.org/pdf/2503.05587](https://arxiv.org/pdf/2503.05587)**

> **作者:** Shiping Yang; Jie Wu; Wenbiao Ding; Ning Wu; Shining Liang; Ming Gong; Hongzhi Li; Hengyuan Zhang; Angel X. Chang; Dongmei Zhang
>
> **备注:** ACL 2026 camera-ready version
>
> **摘要:** Robustness has become a critical attribute for the deployment of RAG systems in real-world applications. Existing research focuses on robustness to explicit noise (e.g., document semantics) but overlooks implicit noise (spurious features). Moreover, previous studies on spurious features in LLMs are limited to specific types (e.g., formats) and narrow scenarios (e.g., ICL). In this work, we identify and study spurious features in the RAG paradigm, a robustness issue caused by the sensitivity of LLMs to semantic-agnostic features. We then propose a novel framework, SURE, to empirically quantify the robustness of RALMs against spurious features. Beyond providing a comprehensive taxonomy and metrics for evaluation, the framework's data synthesis pipeline facilitates training-based strategies to improve robustness. Further analysis suggests that spurious features are a widespread and challenging problem in the field of RAG. Our code is available at this https URL .
>
---
#### [replaced 042] Patterns vs. Patients: Evaluating LLMs against Mental Health Professionals on Personality Disorder Diagnosis through First-Person Narratives
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文属于心理诊断任务，比较LLMs与专业人员在基于第一人称叙述的边缘型和自恋型人格障碍诊断效果，发现模型在识别BPD上表现较好，但对NPD存在严重低估。**

- **链接: [https://arxiv.org/pdf/2512.20298](https://arxiv.org/pdf/2512.20298)**

> **作者:** Karolina Drożdż; Kacper Dudzic; Anna Sterna; Marcin Moskalewicz
>
> **摘要:** Growing reliance on LLMs for psychiatric self-assessment raises questions about their ability to interpret qualitative patient narratives. This depth-first case study provides the first direct comparison of state-of-the-art LLMs and mental health professionals in assessing Borderline (BPD) and Narcissistic (NPD) Personality Disorders based on Polish-language first-person autobiographical accounts. Within our sample, the overall diagnostic scores of the top-performing Gemini Pro models (65.48%) were 21.91 percentage points higher than the average scores of the human professionals (43.57%). While both models and human experts excelled at identifying BPD (F1 = 83.4 & F1 = 80.0, respectively), models severely underdiagnosed NPD (F1 = 6.7 vs. 50.0), showing a potential reluctance toward the value-laden term "narcissism." Qualitatively, models provided confident, elaborate justifications focused on patterns and formal categories, while human experts remained concise and cautious, emphasizing the patients' sense of self and temporal experience. Our findings demonstrate that while LLMs might be competent at interpreting complex first-person clinical data, their outputs still carry critical reliability and bias issues.
>
---
#### [replaced 043] Towards Holistic Evaluation of Large Audio-Language Models: A Comprehensive Survey
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于音频语言模型评估任务，旨在解决现有评估体系碎片化问题，提出四维系统分类，梳理研究现状并指明未来方向。**

- **链接: [https://arxiv.org/pdf/2505.15957](https://arxiv.org/pdf/2505.15957)**

> **作者:** Chih-Kai Yang; Neo S. Ho; Hung-yi Lee
>
> **备注:** EMNLP 2025 (Main). Project Website: this https URL
>
> **摘要:** With advancements in large audio-language models (LALMs), which enhance large language models (LLMs) with auditory capabilities, these models are expected to demonstrate universal proficiency across various auditory tasks. While numerous benchmarks have emerged to assess LALMs' performance, they remain fragmented and lack a structured taxonomy. To bridge this gap, we conduct a comprehensive survey and propose a systematic taxonomy for LALM evaluations, categorizing them into four dimensions based on their objectives: (1) General Auditory Awareness and Processing, (2) Knowledge and Reasoning, (3) Dialogue-oriented Ability, and (4) Fairness, Safety, and Trustworthiness. We provide detailed overviews within each category and highlight challenges in this field, offering insights into promising future directions. To the best of our knowledge, this is the first survey specifically focused on the evaluations of LALMs, providing clear guidelines for the community. We will release the collection of the surveyed papers and actively maintain it to support ongoing advancements in the field.
>
---
#### [replaced 044] Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models
- **分类: cs.CL; cs.AI; cs.LG; cs.PF**

- **简介: 该论文属于提升语言模型推理能力的任务，针对参数约束下的模型优化问题。提出TaH方法，通过选择性迭代提升准确率，减少不必要的计算。**

- **链接: [https://arxiv.org/pdf/2511.08577](https://arxiv.org/pdf/2511.08577)**

> **作者:** Tianyu Fu; Yichen You; Zekai Chen; Guohao Dai; Huazhong Yang; Yu Wang
>
> **摘要:** Improving reasoning abilities of Large Language Models (LLMs), especially under parameter constraints, is crucial for real-world applications. Looped transformers address this by performing multiple latent iterations to refine each token beyond a single forward pass. However, we identify a latent overthinking phenomenon: most token predictions are already correct after the first pass, but are sometimes revised into errors in later iterations. In this work, we ask whether selectively skipping latent iterations may improve accuracy. We reveal significant potential with an oracle iteration policy that boosts model performance by up to 7.3%. Motivated by this, we propose Think-at-Hard (TaH), a looped transformer optimized for selective iteration. TaH employs a lightweight neural decider to trigger latent iteration only at tokens that are likely incorrect after the standard forward pass. During latent iterations, depth-aware Low-Rank Adaptation (LoRA) modules shift the LLM's objective from general next-token prediction to focused hard-token refinement. A duo-causal attention mechanism extends attention from the token sequence dimension to an additional iteration depth dimension, enabling cross-iteration information flow with full sequential parallelism. Experiments on nine benchmarks show consistent gains across math, QA, and coding tasks. With identical parameter counts, TaH outperforms always-iterate baselines by 3.8-4.4% while skipping iterations on 93% of tokens, and exceeds single-iteration Qwen3 baselines by 3.0-3.8%. When allowing <3% more parameters from LoRA and decider modules, the gains further increase to 5.3-6.2% and 6.1-6.8%, respectively. Our code is available at this https URL.
>
---
#### [replaced 045] Robust Explanations for User Trust in Enterprise NLP Systems
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的可解释性任务，旨在解决企业级NLP系统中解释稳定性问题。通过构建评估框架，分析不同模型在噪声下的解释鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.12069](https://arxiv.org/pdf/2604.12069)**

> **作者:** Guilin Zhang; Kai Zhao; Jeffrey Friedman; Xu Chu; Amine Anoun; Jerry Ting
>
> **摘要:** Robust explanations are increasingly required for user trust in enterprise NLP, yet pre-deployment validation is difficult in the common case of black-box deployment (API-only access) where representation-based explainers are infeasible and existing studies provide limited guidance on whether explanations remain stable under real user noise, especially when organizations migrate from encoder classifiers to decoder LLMs. To close this gap, we propose a unified black-box robustness evaluation framework for token-level explanations based on leave-one-out occlusion, and operationalize explanation robustness with top-token flip rate under realistic perturbations (swap, deletion, shuffling, and back-translation) at multiple severity levels. Using this protocol, we conduct a systematic cross-architecture comparison across three benchmark datasets and six models spanning encoder and decoder families (BERT, RoBERTa, Qwen 7B/14B, Llama 8B/70B; 64,800 cases). We find that decoder LLMs produce substantially more stable explanations than encoder baselines (73% lower flip rates on average), and that stability improves with model scale (44% gain from 7B to 70B). Finally, we relate robustness improvements to inference cost, yielding a practical cost-robustness tradeoff curve that supports model and explanation selection prior to deployment in compliance-sensitive applications.
>
---
#### [replaced 046] Structure-Grounded Knowledge Retrieval via Code Dependencies for Multi-Step Data Reasoning
- **分类: cs.CL**

- **简介: 该论文属于数据推理任务，解决多步分析中知识检索不准确的问题。提出SGKR框架，通过代码依赖结构组织知识，提升LLM的推理准确性。**

- **链接: [https://arxiv.org/pdf/2604.10516](https://arxiv.org/pdf/2604.10516)**

> **作者:** Xinyi Huang
>
> **摘要:** Selecting the right knowledge is critical when using large language models (LLMs) to solve domain-specific data analysis tasks. However, most retrieval-augmented approaches rely primarily on lexical or embedding similarity, which is often a weak proxy for the task-critical knowledge needed for multi-step reasoning. In many such tasks, the relevant knowledge is not merely textually related to the query, but is instead grounded in executable code and the dependency structure through which computations are carried out. To address this mismatch, we propose SGKR (Structure-Grounded Knowledge Retrieval), a retrieval framework that organizes domain knowledge with a graph induced by function-call dependencies. Given a question, SGKR extracts semantic input and output tags, identifies dependency paths connecting them, and constructs a task-relevant subgraph. The associated knowledge and corresponding function implementations are then assembled as a structured context for LLM-based code generation. Experiments on multi-step data analysis benchmarks show that SGKR consistently improves solution correctness over no-retrieval and similarity-based retrieval baselines for both vanilla LLMs and coding agents.
>
---
#### [replaced 047] EXCEEDS: Extracting Complex Events via Nugget-based Grid Modeling in Scientific Domain
- **分类: cs.CL**

- **简介: 该论文属于科学领域事件抽取任务，旨在解决科学文本中事件提取不足的问题。通过构建SciEvents数据集并提出EXCEEDS框架，提升事件抽取效果。**

- **链接: [https://arxiv.org/pdf/2406.14075](https://arxiv.org/pdf/2406.14075)**

> **作者:** Yi-Fan Lu; Xian-Ling Mao; Bo Wang; Xiao Liu; Heyan Huang
>
> **备注:** Accepted by ACL 2026 Main Conference
>
> **摘要:** It is crucial to understand a specific domain by events. Extensive event extraction research has been conducted in many domains such as news, finance, and biology. However, event extraction in scientific domain is still insufficiently supported by comprehensive datasets and tailored methods. Compared with other domains, scientific domain has two characteristics: (1) denser nuggets and events, and (2) more complex information forms. To solve the above problem, considering these two characteristics, we first construct SciEvents, a large-scale multi-event document-level dataset with a schema tailored for scientific domain. It consists of 2,508 documents and 24,381 events under multi-stage manual annotation and quality control. Then, we propose EXCEEDS, an end-to-end scientific event extraction framework by encoding dense nuggets into a grid matrix and simplifying complex event extraction as a nugget-based grid modeling task. Experiments on SciEvents demonstrate state-of-the-art performances of EXCEEDS. Both the SciEvents dataset and the EXCEEDS framework are released publicly to facilitate future research.
>
---
#### [replaced 048] Scoring, Reasoning, and Selecting the Best! Ensembling Large Language Models via a Peer-Review Process
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LLM-PeerReview，用于集成多个大语言模型的输出。任务是提升模型回答质量，通过评分、推理和选择最佳响应解决多模型协作问题。**

- **链接: [https://arxiv.org/pdf/2512.23213](https://arxiv.org/pdf/2512.23213)**

> **作者:** Zhijun Chen; Zeyu Ji; Qianren Mao; Hao Wu; Jinhuan Song; Junhang Cheng; Bangjie Qin; Zhuoran Li; Jingzheng Li; Kai Sun; Zizhe Wang; Yikun Ban; Zhu Sun; Xiangyang Ji; Hailong Sun
>
> **摘要:** We propose LLM-PeerReview, an unsupervised LLM Ensemble method that selects the most ideal response from multiple LLM-generated candidates for each query, harnessing the collective wisdom of multiple models with diverse strengths. LLM-PeerReview is built on a novel, peer-review-inspired framework that offers a transparent and interpretable mechanism, while remaining fully unsupervised for flexible adaptability and generalization. Specifically, it operates in three stages: For scoring, we use the emerging LLM-as-a-Judge technique to evaluate each response by reusing multiple LLMs at hand; For reasoning, we can apply a straightforward averaging strategy or a principled graphical model-based truth inference algorithm to aggregate multiple scores to produce a final score for each response; Finally, the highest-scoring response is selected as the best ensemble output. LLM-PeerReview is conceptually simple and empirically powerful. Our results across four datasets show that the two variants of the proposed approach outperform the advanced model Smoothie-Global by 6.9% and 7.3% points, cross diverse task types including factual recall QA, math reasoning, and instruction following.
>
---
#### [replaced 049] Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文研究合成数据多样性对大语言模型微调的影响，旨在解决模型分布崩溃、对抗鲁棒性和自偏好偏差问题。通过实验分析不同数据源的效果。**

- **链接: [https://arxiv.org/pdf/2511.01490](https://arxiv.org/pdf/2511.01490)**

> **作者:** Max Schaffelder; Albert Gatt
>
> **备注:** Accepted to Findings of the Association for Computational Linguistics: ACL 2026
>
> **摘要:** As synthetic data becomes widely used in language model development, understanding its impact on model behavior is crucial. This paper investigates the impact of the diversity of sources of synthetic data on fine-tuned large language models. We focus on three key dimensions: distribution collapse, adversarial robustness, and self-preference bias. Our findings reveal that fine-tuning models on synthetic data from diverse sources can mitigate distribution collapse, preserving the breadth of the output distribution and the diversity of the output text. Furthermore, while both human and synthetic fine-tuning data can remove safeguards, we observe a tendency for higher output quality in the latter case, thus making outputs potentially more usable and dangerous. Finally, we also find evidence that fine-tuning reduces self-preference bias, with human data being the most effective, followed by multi-source synthetic data.
>
---
#### [replaced 050] Evaluation Framework for Highlight Explanations of Context Utilisation in Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型解释任务，旨在评估语言模型在生成回复时是否正确引用上下文。提出首个黄金标准评估框架，测试多种解释方法，发现MechLight表现最佳，但长上下文和位置偏差仍是挑战。**

- **链接: [https://arxiv.org/pdf/2510.02629](https://arxiv.org/pdf/2510.02629)**

> **作者:** Jingyi Sun; Pepa Atanasova; Sagnik Ray Choudhury; Sekh Mainul Islam; Isabelle Augenstein
>
> **摘要:** Context utilisation, the ability of Language Models (LMs) to incorporate relevant information from the provided context when generating responses, remains largely opaque to users, who cannot determine whether models draw from parametric memory or provided context, nor identify which specific context pieces inform the response. Highlight explanations (HEs) offer a natural solution as they can point the exact context pieces and tokens that influenced model outputs. However, no existing work evaluates their effectiveness in accurately explaining context utilisation. We address this gap by introducing the first gold standard HE evaluation framework for context attribution, using controlled test cases with known ground-truth context usage, which avoids the limitations of existing indirect proxy evaluations. To demonstrate the framework's broad applicability, we evaluate four HE methods -- three established techniques and MechLight, a mechanistic interpretability approach we adapt for this task -- across four context scenarios, four datasets, and five LMs. Overall, we find that MechLight performs best across all context scenarios. However, all methods struggle with longer contexts and exhibit positional biases, pointing to fundamental challenges in explanation accuracy that require new approaches to deliver reliable context utilisation explanations at scale.
>
---
#### [replaced 051] Always Tell Me The Odds: Fine-grained Conditional Probability Estimation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于概率估计任务，解决在不确定或部分信息下准确预测概率的问题。通过数据和模型优化，提升概率估计的精度和校准度。**

- **链接: [https://arxiv.org/pdf/2505.01595](https://arxiv.org/pdf/2505.01595)**

> **作者:** Liaoyaqi Wang; Zhengping Jiang; Anqi Liu; Benjamin Van Durme
>
> **摘要:** We present a state-of-the-art model for fine-grained probability estimation of propositions conditioned on context. Recent advances in large language models (LLMs) have significantly enhanced their reasoning capabilities, particularly on well-defined tasks with complete information. However, LLMs continue to struggle with making accurate and well-calibrated probabilistic predictions under uncertainty or partial information. While incorporating uncertainty into model predictions often boosts performance, obtaining reliable estimates of that uncertainty remains understudied. In particular, LLM probability estimates tend to be coarse and biased towards more frequent numbers. Through a combination of human and synthetic data creation and assessment, scaling to larger models, and better supervision, we propose a set of strong and precise probability estimation models. We conduct systematic evaluations across tasks that rely on conditional probability estimation and show that our approach consistently outperforms existing fine-tuned and prompting-based methods by a large margin.
>
---
#### [replaced 052] The Geometric Canary: Predicting Steerability and Detecting Drift via Representational Stability
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型部署任务，解决模型可控性和结构漂移检测问题。通过几何稳定性分析，提出监督与非监督方法分别预测可控性和检测漂移。**

- **链接: [https://arxiv.org/pdf/2604.17698](https://arxiv.org/pdf/2604.17698)**

> **作者:** Prashant C. Raju
>
> **摘要:** Reliable deployment of language models requires two capabilities that appear distinct but share a common geometric foundation: predicting whether a model will accept targeted behavioral control, and detecting when its internal structure degrades. We show that geometric stability, the consistency of a representation's pairwise distance structure, addresses both. Supervised Shesha variants that measure task-aligned geometric stability predict linear steerability with near-perfect accuracy ($\rho = 0.89$-$0.97$) across 35-69 embedding models and three NLP tasks, capturing unique variance beyond class separability (partial $\rho = 0.62$-$0.76$). A critical dissociation emerges: unsupervised stability fails entirely for steering on real-world tasks ($\rho \approx 0.10$), revealing that task alignment is essential for controllability prediction. However, unsupervised stability excels at drift detection, measuring nearly $2\times$ greater geometric change than CKA during post-training alignment (up to $5.23\times$ in Llama) while providing earlier warning in 73\% of models and maintaining a $6\times$ lower false alarm rate than Procrustes. Together, supervised and unsupervised stability form complementary diagnostics for the LLM deployment lifecycle: one for pre-deployment controllability assessment, the other for post-deployment monitoring.
>
---
#### [replaced 053] Game-Time: Evaluating Temporal Dynamics in Spoken Language Models
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决口语模型在时间动态上的不足。提出Game-Time基准，评估模型在时间感知和同步响应方面的能力。**

- **链接: [https://arxiv.org/pdf/2509.26388](https://arxiv.org/pdf/2509.26388)**

> **作者:** Kai-Wei Chang; En-Pei Hu; Chun-Yi Kuan; Wenze Ren; Wei-Chih Chen; Guan-Ting Lin; Yu Tsao; Shao-Hua Sun; Hung-yi Lee; James Glass
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Conversational Spoken Language Models (SLMs) are emerging as a promising paradigm for real-time speech interaction. However, their capacity of temporal dynamics, including the ability to manage timing, tempo and simultaneous speaking, remains a critical and unevaluated challenge for conversational fluency. To address this gap, we introduce the Game-Time Benchmark, a framework to systematically assess these temporal capabilities. Inspired by how humans learn a language through language activities, Game-Time consists of basic instruction-following tasks and advanced tasks with temporal constraints, such as tempo adherence and synchronized responses. Our evaluation of diverse SLM architectures reveals a clear performance disparity: while state-of-the-art models handle basic tasks well, many contemporary systems still struggle with fundamental instruction-following. More critically, nearly all models degrade substantially under temporal constraints, exposing persistent weaknesses in time awareness and full-duplex interaction. The Game-Time Benchmark provides a foundation for guiding future research toward more temporally-aware conversational AI. Demos and datasets are available on our project website this https URL.
>
---
#### [replaced 054] Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks with Generative Optimization
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Frontier-Eng基准，用于评估AI代理在真实工程任务中的自进化能力，解决传统基准忽略迭代优化的问题。**

- **链接: [https://arxiv.org/pdf/2604.12290](https://arxiv.org/pdf/2604.12290)**

> **作者:** Yizhe Chi; Deyao Hong; Dapeng Jiang; Tianwei Luo; Kaisen Yang; Boshi Zhang; Zhe Cao; Xiaoyan Fan; Bingxiang He; Han Hao; Weiyang Jin; Dianqiao Lei; Qingle Liu; Houde Qian; Bowen Wang; Situ Wang; Youjie Zheng; Yifan Zhou; Calvin Xiao; Eren Cai; Qinhuai Na
>
> **摘要:** Current LLM agent benchmarks, which predominantly focus on binary pass/fail tasks such as code generation or search-based question answering, often neglect the value of real-world engineering that is often captured through the iterative optimization of feasible designs. To this end, we introduce Frontier-Eng, a human-verified benchmark for generative optimization -- an iterative propose-execute-evaluate loop in which an agent generates candidate artifacts, receives executable verifier feedback, and revises them under a fixed interaction budget -- spanning $47$ tasks across five broad engineering categories. Unlike previous suites, Frontier-Eng tasks are grounded in industrial-grade simulators and verifiers that provide continuous reward signals and enforce hard feasibility constraints under constrained budgets. We evaluate eight frontier language models using representative search frameworks, finding that while GPT 5.4 achieves the most robust performance, the benchmark remains challenging for all models. Our analysis suggests a dual power-law decay in improvement frequency ($\sim$ 1/iteration) and magnitude ($\sim$ 1/improvement count). We further show that although width improves parallelism and diversity, depth remains crucial for hard-won improvements under a fixed budget. Frontier-Eng establishes a new standard for assessing the capacity of AI agents to integrate domain knowledge with executable feedback to solve complex, open-ended engineering problems.
>
---
#### [replaced 055] Spontaneous Persuasion: An Audit of Model Persuasiveness in Everyday Conversations
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于AI伦理研究，探讨LLMs在日常对话中的非刻意说服行为。通过审计五种模型，分析其说服策略及与人类回答的差异。**

- **链接: [https://arxiv.org/pdf/2604.22109](https://arxiv.org/pdf/2604.22109)**

> **作者:** Nalin Poungpeth; Nicholas Clark; Tanu Mitra
>
> **摘要:** Large language models (LLMs) possess strong persuasive capabilities that outperform humans in head-to-head comparisons. Users report consulting LLMs to inform major life decisions in relationships, medical settings, and when seeking professional advice. Prior work measures persuasion as intentional attempts at producing the most effective argument or convincing statement. This fails to capture everyday human-AI interactions in which users seek information or advice. To address this gap, we introduce "spontaneous persuasion," which characterizes the inexplicit use of persuasive strategies in everyday scenarios where persuasion is not necessarily warranted. We conduct an audit of five LLMs to uncover how frequently and through which techniques spontaneous persuasion appears in multi-turn conversations. To simulate response styles, we provide a user response taxonomy grounded in literature from psychology, communication, and linguistics. Furthermore, we compare the distribution of spontaneous persuasion produced by LLMs with human responses on the same topics, collected from Reddit. We find LLMs spontaneously persuade the user in virtually all conversations, heavily relying on information-based strategies such as appeals to logic or quantitative evidence. This was consistent across models and user response styles, but conversations concerning mental health saw higher rates of appraisal-based and emotion-based strategies. In comparison, human responses tended to invoke strategies that generate social influence, like negative emotion appeals and non-expert testimony. This difference may explain the effectiveness of LLM in persuading users, as well as the perception of models as objective and impartial.
>
---
#### [replaced 056] BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning
- **分类: cs.CL**

- **简介: 该论文提出BRIEF-Pro，用于解决多跳问答中长上下文带来的延迟和认知负担问题。通过压缩技术生成简洁摘要，提升模型效率与准确性。**

- **链接: [https://arxiv.org/pdf/2510.13799](https://arxiv.org/pdf/2510.13799)**

> **作者:** Jia-Chen Gu; Junyi Zhang; Di Wu; Yuankai Li; Kai-Wei Chang; Nanyun Peng
>
> **备注:** Accepted by ACL 2026 Findings. Code and data: this https URL
>
> **摘要:** As retrieval-augmented generation (RAG) tackles complex tasks, increasingly expanded contexts offer richer information, but at the cost of higher latency and increased cognitive load on the model. To mitigate this bottleneck, especially for intricate multi-hop questions, we introduce BRIEF-Pro. It is a universal, lightweight compressor that distills relevant evidence for a given query from retrieved documents into a concise summary for seamless integration into in-context RAG. Using seed data consisting of relatively short contexts (fewer than 1k words), BRIEF-Pro is trained to perform abstractive compression of extended contexts exceeding 10k words across a wide range of scenarios. Furthermore, BRIEF-Pro offers flexible user control over summary length by allowing users to specify the desired number of sentences. Experiments on four open-domain multi-hop question-answering datasets show that BRIEF-Pro generates more concise and relevant summaries, enhancing performance across small, large, and proprietary language models. With the 70B reader model, 32x compression by BRIEF-Pro improves QA performance by 4.67% on average over LongLLMLingua's 9x, while requiring only 23% of its computational overhead.
>
---
#### [replaced 057] SwissGov-RSD: A Human-annotated, Cross-lingual Benchmark for Token-level Recognition of Semantic Differences Between Related Documents
- **分类: cs.CL**

- **简介: 该论文提出SwissGov-RSD，首个跨语言文档级语义差异识别基准，解决跨语言文本对比难题。通过人工标注数据评估模型性能，揭示现有方法的不足。**

- **链接: [https://arxiv.org/pdf/2512.07538](https://arxiv.org/pdf/2512.07538)**

> **作者:** Michelle Wastl; Jannis Vamvas; Rico Sennrich
>
> **备注:** 30 pages; v3 accepted to ACL Main (camera-ready)
>
> **摘要:** Recognizing semantic differences across documents is crucial for text generation evaluation and content alignment, especially in cross-lingual settings. However, as a standalone task, it has received little attention. We address this by introducing SwissGov-RSD, the first naturalistic, document-level, cross-lingual dataset for semantic difference recognition. It encompasses a total of 224 multi-parallel documents in English--German, English--French, and English--Italian with token-level difference annotations by human annotators. We evaluate a variety of open-source and closed-source large language models as well as encoder models across different fine-tuning settings on this new benchmark. Our results show that current automatic approaches perform poorly compared to their performance on monolingual, sentence-level, and synthetic benchmarks, revealing a considerable gap for both LLMs and encoder models. We make our code and dataset publicly available.
>
---
#### [replaced 058] Can We Still Hear the Accent? Investigating the Resilience of Native Language Signals in the LLM Era
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言识别任务，研究LLM时代研究论文是否失去母语特征。通过分析ACL Anthology数据，发现NLI性能下降，但中法语表现异常。**

- **链接: [https://arxiv.org/pdf/2604.08568](https://arxiv.org/pdf/2604.08568)**

> **作者:** Nabelanita Utami; Ryohei Sasano
>
> **摘要:** The evolution of writing assistance tools from machine translation to large language models (LLMs) has changed how researchers write. This study investigates whether this shift is homogenizing research papers by analyzing native language identification (NLI) trends in ACL Anthology papers across three eras: pre-neural network (NN), pre-LLM, and post-LLM. We construct a labeled dataset using a semi-automated framework and fine-tune a classifier to detect linguistic fingerprints of author backgrounds. Our analysis shows a consistent decline in NLI performance over time. Interestingly, the post-LLM era reveals anomalies: while Chinese and French show unexpected resistance or divergent trends, Japanese and Korean exhibit sharper-than-expected declines.
>
---
#### [replaced 059] NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于神经网络机器翻译任务，旨在解决包含新词的句子翻译问题。通过构建数据集和搜索工具包，结合强化学习提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2601.03790](https://arxiv.org/pdf/2601.03790)**

> **作者:** Zhongtao Miao; Kaiyan Zhao; Masaaki Nagata; Yoshimasa Tsuruoka
>
> **备注:** ACL 2026 Main
>
> **摘要:** Neologism-aware machine translation aims to translate source sentences containing neologisms into target languages. This field remains underexplored compared with general machine translation (MT). In this paper, we propose an agentic framework, NeoAMT, for neologism-aware machine translation equipped with a Wiktionary-based search toolkit. Specifically, we first construct a dedicated dataset for neologism-aware machine translation and build a search toolkit grounded in Wiktionary. The dataset covers 16 languages and 75 translation directions in total, derived from approximately 10 million records of an English Wiktionary dump. The retrieval corpus of the search toolkit is also constructed from around 3 million cleaned records of the same dump. We then leverage the dataset and toolkit to train a translation agent via reinforcement learning (RL) and to evaluate the accuracy of neologism-aware machine translation. Furthermore, we propose an RL training framework featuring a novel reward design and an adaptive rollout generation strategy that exploits translation difficulty to further improve the translation quality of translation agents using our search toolkit.
>
---
#### [replaced 060] KLong: Training LLM Agent for Extremely Long-horizon Tasks
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出KLong，解决长周期任务难题。通过轨迹拆分微调和渐进强化学习提升模型能力，实验显示其性能优于现有模型。**

- **链接: [https://arxiv.org/pdf/2602.17547](https://arxiv.org/pdf/2602.17547)**

> **作者:** Yue Liu; Yingwei Ma; Yibo Miao; Yanhao Li; Yuchong Xie; Xinlong Yang; Zhiyuan Hu; Flood Sung; Jiaheng Zhang; Bryan Hooi
>
> **备注:** We request standard withdrawal of this submission because significant errors were discovered in the data after submission, which affect the validity of the results. We may submit a corrected version later
>
> **摘要:** This paper introduces KLong, an open-source LLM agent trained to solve extremely long-horizon tasks. The principle is to first cold-start the model via trajectory-splitting SFT, then scale it via progressive RL training. Specifically, we first activate basic agentic abilities of a base model with a comprehensive SFT recipe. Then, we introduce Research-Factory, an automated pipeline that generates high-quality training data by collecting research papers and constructing evaluation rubrics. Using this pipeline, we build thousands of long-horizon trajectories distilled from Claude 4.5 Sonnet (Thinking). To train with these extremely long trajectories, we propose a new trajectory-splitting SFT, which preserves early context, progressively truncates later context, and maintains overlap between sub-trajectories. In addition, to further improve long-horizon task-solving capability, we propose a novel progressive RL, which schedules training into multiple stages with progressively extended timeouts. Experiments demonstrate the superiority and generalization of KLong, as shown in Figure 1. Notably, our proposed KLong (106B) surpasses Kimi K2 Thinking (1T) by 11.28% on PaperBench, and the performance improvement generalizes to other coding benchmarks like SWE-bench Verified and MLE-bench.
>
---
#### [replaced 061] KOCO-BENCH: Can Large Language Models Leverage Domain Knowledge in Software Development?
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出KOCO-BENCH基准，用于评估大语言模型在软件开发中的领域知识应用能力。针对现有基准不足，该工作构建包含多领域知识的评测体系，旨在推动更有效的领域专业化方法研究。**

- **链接: [https://arxiv.org/pdf/2601.13240](https://arxiv.org/pdf/2601.13240)**

> **作者:** Xue Jiang; Ge Li; Jiaru Qian; Xianjie Shi; Chenjie Li; Hao Zhu; Ziyu Wang; Jielun Zhang; Zheyu Zhao; Lingwei Wu; Kechi Zhang; Jia Li; Wenpin Jiao; Zhi Jin; Yihong Dong
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Large language models (LLMs) excel at general programming but struggle with domain-specific software development, necessitating domain specialization methods for LLMs to learn and utilize domain knowledge and data. However, existing domain-specific code benchmarks cannot evaluate the effectiveness of domain specialization methods, which focus on assessing what knowledge LLMs possess rather than how they acquire and apply new knowledge, lacking explicit knowledge corpora for developing domain specialization methods. To this end, we present KOCO-BENCH, a novel benchmark designed for evaluating domain specialization methods in real-world software development. KOCO-BENCH contains 6 emerging domains with 11 software frameworks and 25 projects, featuring curated knowledge corpora alongside multi-granularity evaluation tasks including domain code generation (from function-level to project-level with rigorous test suites) and domain knowledge understanding (via multiple-choice Q&A). Unlike previous benchmarks that only provide test sets for direct evaluation, KOCO-BENCH requires acquiring and applying diverse domain knowledge (APIs, rules, constraints, etc.) from knowledge corpora to solve evaluation tasks. Our evaluations reveal that KOCO-BENCH poses significant challenges to state-of-the-art LLMs. Even with domain specialization methods (e.g., SFT, RAG, kNN-LM) applied, improvements remain marginal. Best-performing coding agent, Claude Code, achieves only 34.2%, highlighting the urgent need for more effective domain specialization methods. We release KOCO-BENCH, evaluation code, and baselines to advance further research at this https URL.
>
---
#### [replaced 062] Language Models Might Not Understand You: Evaluating Theory of Mind via Story Prompting
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出StorySim框架，用于评估大语言模型的理论心智和世界建模能力。通过生成可控故事测试模型理解他人心理状态的能力，发现模型在心智理论任务中表现较差，且更擅长处理人物而非物体的推理。**

- **链接: [https://arxiv.org/pdf/2506.19089](https://arxiv.org/pdf/2506.19089)**

> **作者:** Nathaniel Getachew; Abulhair Saparov
>
> **备注:** 21 pages, 17 figures
>
> **摘要:** We introduce StorySim, a programmable framework for synthetically generating stories to evaluate the theory of mind (ToM) and world modeling (WM) capabilities of large language models (LLMs). Unlike prior benchmarks that may suffer from contamination in pretraining data, or rely on an LLM for generation, StorySim produces novel, compositional story prompts anchored by a highly controllable Storyboard, enabling precise manipulation of character perspectives and events. We use this framework to design first- and second-order ToM tasks alongside WM tasks that control for the ability to track and model mental states. Our experiments across a suite of LLMs show that most models achieve higher accuracy on WM tasks than on ToM tasks, and that models tend to reason more accurately when the subject of reasoning is a person rather than an inanimate object. Additionally, our framework enabled us to find evidence of heuristic behavior and an over-reliance on earlier events in the story. All code for generating data and evaluations is freely available.
>
---
#### [replaced 063] Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于文本到语音（TTS）任务，解决传统方法依赖配对数据及无法进行细粒度优化的问题。提出TKTO方法，无需配对数据，直接优化token级，提升日语TTS准确率39%。**

- **链接: [https://arxiv.org/pdf/2510.05799](https://arxiv.org/pdf/2510.05799)**

> **作者:** Rikuto Kotoge; Yuichi Sasaki
>
> **备注:** Accepted at ACL 2026 (Main)
>
> **摘要:** Aligning text-to-speech (TTS) system outputs with human feedback through preference optimization has been shown to effectively improve the robustness and naturalness of language model-based TTS models. Current approaches primarily require paired desirable and undesirable samples at the utterance level. However, such pairs are often limited in TTS output data, and utterance-level formulation prevents fine-grained token-level optimization needed for accurate pronunciation alignment. In this study, we propose TKTO that eliminates the need for paired data, enabling a more data-efficient training paradigm, and directly targets token-level units, automatically providing fine-grained alignment signals without token-level annotations. TKTO improves the challenging Japanese TTS accuracy by 39% and reduces CER by 54%, automatically assigning 12.8 times stronger reward to targeted tokens.
>
---
#### [replaced 064] Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness
- **分类: cs.CL**

- **简介: 该论文研究大语言模型是否具备判断答案正确性的私有知识。任务是识别模型内部表示是否优于外部模型。工作包括训练分类器并比较不同模型的性能，发现事实性任务中自表示有优势。**

- **链接: [https://arxiv.org/pdf/2604.12373](https://arxiv.org/pdf/2604.12373)**

> **作者:** Tomer Ashuach; Shai Gretz; Yoav Katz; Yonatan Belinkov; Liat Ein-Dor
>
> **备注:** Accepted to ACL 2026 (Main Conference). 8 pages, 16 figures, 2 tables
>
> **摘要:** Humans use introspection to evaluate their understanding through private internal states inaccessible to external observers. We investigate whether large language models possess similar privileged knowledge about answer correctness, information unavailable through external observation. We train correctness classifiers on question representations from both a model's own hidden states and external models, testing whether self-representations provide a performance advantage. On standard evaluation, we find no advantage: self-probes perform comparably to peer-model probes. We hypothesize this is due to high inter-model agreement of answer correctness. To isolate genuine privileged knowledge, we evaluate on disagreement subsets, where models produce conflicting predictions. Here, we discover domain-specific privileged knowledge: self-representations consistently outperform peer representations in factual knowledge tasks, but show no advantage in math reasoning. We further localize this domain asymmetry across model layers, finding that the factual advantage emerges progressively from early-to-mid layers onward, consistent with model-specific memory retrieval, while math reasoning shows no consistent advantage at any depth.
>
---
#### [replaced 065] Comparison of sEMG Encoding Accuracy Across Speech Modes Using Articulatory and Phoneme Features
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音编码分析任务，比较SPARC与音素特征在不同说话模式下对sEMG的预测效果，验证SPARC的优越性与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.18920](https://arxiv.org/pdf/2604.18920)**

> **作者:** Chenqian Le; Ruisi Li; Beatrice Fumagalli; Yasamin Esmaeili; Xupeng Chen; Amirhossein Khalilian-Gourtani; Tianyu He; Adeen Flinker; Yao Wang
>
> **摘要:** We test whether Speech Articulatory Coding (SPARC) features can linearly predict surface electromyography (sEMG) envelopes across aloud, mimed, and subvocal speech in twenty-four subjects. Using elastic-net multivariate temporal response function (mTRF) with sentence-level cross-validation, SPARC yields higher prediction accuracy than phoneme one-hot representations on nearly all electrodes and in all speech modes. Aloud and mimed speech perform comparably, and subvocal speech remains above chance, indicating detectable articulatory activity. Variance partitioning shows a substantial unique contribution from SPARC and a minimal unique contribution from phoneme features. mTRF weight patterns reveal anatomically interpretable relationships between electrode sites and articulatory movements that remain consistent across modes. This study focuses on representation/encoding analysis (not end-to-end decoding) and supports SPARC as a robust and interpretable intermediate target for sEMG-based silent-speech modeling.
>
---
#### [replaced 066] The Surprising Effectiveness of Membership Inference with Simple N-Gram Coverage
- **分类: cs.CL**

- **简介: 该论文属于隐私安全任务，旨在解决黑盒模型的成员推断问题。提出N-Gram Coverage Attack，仅通过模型输出文本进行攻击，验证其有效性并分析模型隐私保护趋势。**

- **链接: [https://arxiv.org/pdf/2508.09603](https://arxiv.org/pdf/2508.09603)**

> **作者:** Skyler Hallinan; Jaehun Jung; Melanie Sclar; Ximing Lu; Abhilasha Ravichander; Sahana Ramnath; Yejin Choi; Sai Praneeth Karimireddy; Niloofar Mireshghallah; Xiang Ren
>
> **备注:** CoLM 2025. v2: update citation
>
> **摘要:** Membership inference attacks serves as useful tool for fair use of language models, such as detecting potential copyright infringement and auditing data leakage. However, many current state-of-the-art attacks require access to models' hidden states or probability distribution, which prevents investigation into more widely-used, API-access only models like GPT-4. In this work, we introduce N-Gram Coverage Attack, a membership inference attack that relies solely on text outputs from the target model, enabling attacks on completely black-box models. We leverage the observation that models are more likely to memorize and subsequently generate text patterns that were commonly observed in their training data. Specifically, to make a prediction on a candidate member, N-Gram Coverage Attack first obtains multiple model generations conditioned on a prefix of the candidate. It then uses n-gram overlap metrics to compute and aggregate the similarities of these outputs with the ground truth suffix; high similarities indicate likely membership. We first demonstrate on a diverse set of existing benchmarks that N-Gram Coverage Attack outperforms other black-box methods while also impressively achieving comparable or even better performance to state-of-the-art white-box attacks - despite having access to only text outputs. Interestingly, we find that the success rate of our method scales with the attack compute budget - as we increase the number of sequences generated from the target model conditioned on the prefix, attack performance tends to improve. Having verified the accuracy of our method, we use it to investigate previously unstudied closed OpenAI models on multiple domains. We find that more recent models, such as GPT-4o, exhibit increased robustness to membership inference, suggesting an evolving trend toward improved privacy protections.
>
---
#### [replaced 067] Switch Attention: Towards Dynamic and Fine-grained Hybrid Transformers
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决长序列注意力计算效率低的问题。提出Switch Attention机制，动态结合全注意力与滑动窗口注意力，提升模型效率与效果。**

- **链接: [https://arxiv.org/pdf/2603.26380](https://arxiv.org/pdf/2603.26380)**

> **作者:** Yusheng Zhao; Hourun Li; Bohan Wu; Yichun Yin; Lifeng Shang; Jingyang Yuan; Meng Zhang; Ming Zhang
>
> **摘要:** The attention mechanism has been the core component in modern transformer architectures. However, the computation of standard full attention scales quadratically with the sequence length, serving as a major bottleneck in long-context language modeling. Sliding window attention restricts the context length for better efficiency at the cost of narrower receptive fields. While existing efforts attempt to take the benefits from both sides by building hybrid models, they often resort to static, heuristically designed alternating patterns that limit efficient allocation of computation in various scenarios. In this paper, we propose Switch Attention (SwiAttn), a novel hybrid transformer that enables dynamic and fine-grained routing between full attention and sliding window attention. For each token at each transformer layer, SwiAttn dynamically routes the computation to either a full-attention branch for global information aggregation or a sliding-window branch for efficient local pattern matching. An adaptive regularization objective is designed to encourage the model towards efficiency. Moreover, we adopt continual pretraining to optimize the model, transferring the full attention architecture to the hybrid one. Extensive experiments are conducted on twenty-three benchmark datasets across both regular (4K) and long (32K) context lengths, demonstrating the effectiveness of the proposed method.
>
---
#### [replaced 068] Reducing Maintenance Burden in Behaviour-Driven Development: A Paraphrase-Robust Duplicate-Step Detector with a 1.1M-Step Open Benchmark
- **分类: cs.SE; cs.CL; cs.IR**

- **简介: 该论文属于BDD步骤去重任务，解决步骤重复导致的维护成本问题，提出一种鲁棒检测器并构建公开基准。**

- **链接: [https://arxiv.org/pdf/2604.20462](https://arxiv.org/pdf/2604.20462)**

> **作者:** Ali Hassaan Mughal; Noor Fatima; Muhammad Bilal
>
> **备注:** 25 pages, 2 figures, 4 tables. Submitted to Information and Software Technology (Elsevier). Tool, corpus, labelled benchmark, and rubric released at this https URL under Apache-2.0
>
> **摘要:** Context. Behaviour-Driven Development (BDD) suites in Gherkin accumulate step-text duplication with documented maintenance cost. Prior detectors either require runnable tests or are single-organisation, leaving a gap: a static, paraphrase-robust, step-level detector and a public benchmark to calibrate it. Objective. We release (i) the largest cross-organisational BDD step corpus to date, (ii) a labelled pair-level calibration benchmark, and (iii) a four-strategy detector with a consolidation-savings model linking clusters to ISO/IEC 25010 maintainability sub-characteristics. Method. The corpus contains 347 public GitHub repositories, 23,667 .feature files, and 1,113,616 Gherkin steps, SPDX-tagged. The detector layers exact hashing, normalised Levenshtein, sentence-transformer cosine, and a Levenshtein-banded hybrid. Calibration uses 1,020 manually labelled step pairs under a released rubric (60-pair overlap, Fleiss kappa = 0.84). We report precision, recall, and F1 with bootstrap 95% CIs under the primary rubric and a score-free relabelling, and benchmark against SourcererCC-style and NiCad-style lexical baselines. Results. Step-weighted exact-duplicate rate is 80.2%; median-repository rate is 58.6% (Spearman rho = 0.51). The top hybrid cluster has 20,737 occurrences across 2,245 files. Near-exact reaches F1 = 0.822 on score-free labels; semantic F1 = 0.906 under the primary rubric reflects a disclosed stratification artefact. Lexical baselines reach F1 = 0.761 and 0.799. The savings model estimates 893,357 corpus-wide eliminable step occurrences; on the median repository 62.5% of step lines are eliminable.
>
---
#### [replaced 069] Swa-bhasha Resource Hub: Romanized Sinhala to Sinhala Transliteration Systems and Data Resources
- **分类: cs.CL**

- **简介: 该论文介绍了一个资源库，用于罗马化僧伽罗语到僧伽罗语的转写任务，提供数据和算法，解决转写模型训练与应用开发问题。**

- **链接: [https://arxiv.org/pdf/2507.09245](https://arxiv.org/pdf/2507.09245)**

> **作者:** Deshan Sumanathilaka; Sameera Perera; Sachithya Dharmasiri; Maneesha Athukorala; Anuja Dilrukshi Herath; Rukshan Dias; Pasindu Gamage; Ruvan Weerasinghe; Y.H.P.P. Priyadarshana
>
> **备注:** 15 pages, 5 Tables, 3 figures
>
> **摘要:** The Swa-bhasha Resource Hub provides a comprehensive collection of data resources and algorithms developed for Romanized Sinhala to Sinhala transliteration between 2020 and 2025. These resources have played a significant role in advancing research in Sinhala Natural Language Processing (NLP), particularly in training transliteration models and developing applications involving Romanized Sinhala. The current openly accessible data sets and corresponding tools are made publicly available through this hub. This paper presents a detailed overview of the resources contributed by the authors and includes a comparative analysis of existing transliteration applications in the domain.
>
---
#### [replaced 070] Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs
- **分类: cs.CL**

- **简介: 该论文属于安全评估任务，旨在解决大语言模型在代理系统中的漏洞问题。通过构建动作图，分析模型与代理层级的风险差异，并提出自动化防御方法。**

- **链接: [https://arxiv.org/pdf/2509.04802](https://arxiv.org/pdf/2509.04802)**

> **作者:** Ilham Wicaksono; Zekun Wu; Rahul Patel; Theo King; Adriano Koshiyama; Philip Treleaven
>
> **备注:** ICLR 2026 Agents in the Wild (Spotlight & Oral); ICLR 2026 AFAA; OpenAI Red-Teaming Challenge Winner (2025); NeurIPS 2025 LLMEval
>
> **摘要:** As large language models increasingly deployed into agentic systems, existing methods face critical gaps in observing, assessing, and mitigating deployment-specific risks. We present a comprehensive, observability-driven workflow: we introduce \textbf{AgentSeer}, observability tool which decomposes agentic executions into granular \emph{action-component} graphs; we use this decomposition to rigorously quantify the gap between model-level and agent-level jailbreaking risk via cross-model validation on GPT-OSS-20B and Gemini-2.0-flash with HarmBench under single-turn and iterative-refinement attacks; we leverage action-graph risk signals to automate iterative prompt hardening against direct and iterative jailbreak attacks. Stark differences is revealed between model-level and agentic-level vulnerability profiles. Model-level evaluation reveals baseline differences: GPT-OSS-20B (39.47\% ASR) versus Gemini-2.0-flash (50.00\% ASR), with both models showing susceptibility to social engineering. However, agentic-level assessment exposes agent-specific risks invisible to traditional evaluation. We discover "agentic-only" vulnerabilities that emerge exclusively in agentic contexts, with tool-calling showing 24-60\% higher ASR across both models. Cross-model analysis reveals universal agentic patterns, where agent transfer operations as highest-risk tools, with semantic pattern revealed rather than syntactic vulnerability mechanisms. Direct attack transfer from model-level to agentic contexts shows degraded performance of successful prompts (GPT-OSS-20B: 57\% human injection ASR; Gemini-2.0-flash: 28\%), while context-aware iterative attacks successfully compromise objectives that failed at model-level, confirming systematic vulnerabilities gaps. Action-based prompt improvement substantially reduces action-averaged agentic jailbreak success on GPT-OSS-20B (direct: 45.3\%
>
---
#### [replaced 071] AI Security Beyond Core Domains: Resume Screening as a Case Study of Adversarial Vulnerabilities in Specialized LLM Applications
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究AI在简历筛选中的安全问题，针对对抗性指令攻击提出防御方法，评估不同防御机制的效果。**

- **链接: [https://arxiv.org/pdf/2512.20164](https://arxiv.org/pdf/2512.20164)**

> **作者:** Honglin Mu; Jinghao Liu; Kaiyang Wan; Rui Xing; Xiuying Chen; Timothy Baldwin; Wanxiang Che
>
> **摘要:** Large Language Models (LLMs) excel at text comprehension and generation, making them ideal for automated tasks like code review and content moderation. However, our research identifies a vulnerability: LLMs can be manipulated by "adversarial instructions" hidden in input data, such as resumes or code, causing them to deviate from their intended task. Notably, while defenses may exist for mature domains such as code review, they are often absent in other common applications such as resume screening and peer review. This paper introduces a benchmark to assess this vulnerability in resume screening, revealing attack success rates exceeding 80% for certain attack types. We evaluate two defense mechanisms: prompt-based defenses achieve 10.1% attack reduction with 12.5% false rejection increase, while our proposed FIDS (Foreign Instruction Detection through Separation) using LoRA adaptation achieves 15.4% attack reduction with 10.4% false rejection increase. The combined approach provides 26.3% attack reduction, demonstrating that training-time defenses outperform inference-time mitigations in both security and utility preservation.
>
---
#### [replaced 072] A Comparative analysis of Layer-wise Representational Capacity in AR and Diffusion LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文对比分析AR与扩散语言模型的层表示能力，解决模型结构对表示影响的问题，通过相似度和层跳过分析，揭示扩散模型具有全局表示和冗余性。**

- **链接: [https://arxiv.org/pdf/2603.07475](https://arxiv.org/pdf/2603.07475)**

> **作者:** Raghavv Goel; Risheek Garrepalli; Sudhanshu Agrawal; Chris Lott; Mingu Lee; Fatih Porikli
>
> **备注:** v2: Clarified problem framing and key takeaways. Revised introduction for improved exposition. Added additional analysis and results to strengthen empirical support
>
> **摘要:** Autoregressive (AR) language models build representations incrementally via left-to-right prediction, while diffusion language models (dLLMs) are trained through full-sequence denoising. Although recent dLLMs match AR performance, whether diffusion objectives fundamentally reshape internal representations remains unclear. We perform the first layer- and token-wise representational analysis comparing native dLLMs (LLaDA), native AR models (Qwen2.5), and AR-initialized dLLMs (Dream-7B), using cosine similarity across layers and tokens alongside static inference-time layer-skipping as an analytical probe of redundancy. We find that diffusion objectives produce more global representations with substantial early-layer redundancy and reduced recency bias, while AR objectives yield tightly coupled, locally structured representations. AR-initialized dLLMs retain AR-like dynamics despite diffusion training, revealing persistent initialization bias. Leveraging this redundancy, native dLLMs absorb up to 18.75% FLOPs reduction while retaining over 90% performance on math-reasoning and coding benchmarks, whereas AR models collapse under identical skipping, revealing that diffusion objectives, rather than architecture alone, induce depth redundancy that enables principled compression.
>
---
#### [replaced 073] Food4All: A Multi-Agent Framework for Real-time Free Food Discovery with Integrated Nutritional Metadata
- **分类: cs.CL; cs.CY; cs.MA**

- **简介: 该论文属于信息检索任务，旨在解决食物不安全问题。通过构建多智能体框架Food4All，整合多源数据、优化地理与营养因素，并动态调整策略，提升实时免费食物获取效率。**

- **链接: [https://arxiv.org/pdf/2510.18289](https://arxiv.org/pdf/2510.18289)**

> **作者:** Zhengqing Yuan; Yiyang Li; Weixiang Sun; Zheyuan Zhang; Kaiwen Shi; Keerthiram Murugesan; Yanfang Ye
>
> **备注:** This paper is withdrawn because parts of the Method section are inconsistent with the actual implementation and code. Specifically, some components of the described multi-agent workflow and nutritional-metadata integration were not implemented as stated. We withdraw this version to avoid misleading readers
>
> **摘要:** Food insecurity remains a persistent public health emergency in the United States, tightly interwoven with chronic disease, mental illness, and opioid misuse. Yet despite the existence of thousands of food banks and pantries, access remains fragmented: 1) current retrieval systems depend on static directories or generic search engines, which provide incomplete and geographically irrelevant results; 2) LLM-based chatbots offer only vague nutritional suggestions and fail to adapt to real-world constraints such as time, mobility, and transportation; and 3) existing food recommendation systems optimize for culinary diversity but overlook survival-critical needs of food-insecure populations, including immediate proximity, verified availability, and contextual barriers. These limitations risk leaving the most vulnerable individuals, those experiencing homelessness, addiction, or digital illiteracy, unable to access urgently needed resources. To address this, we introduce Food4All, the first multi-agent framework explicitly designed for real-time, context-aware free food retrieval. Food4All unifies three innovations: 1) heterogeneous data aggregation across official databases, community platforms, and social media to provide a continuously updated pool of food resources; 2) a lightweight reinforcement learning algorithm trained on curated cases to optimize for both geographic accessibility and nutritional correctness; and 3) an online feedback loop that dynamically adapts retrieval policies to evolving user needs. By bridging information acquisition, semantic analysis, and decision support, Food4All delivers nutritionally annotated and guidance at the point of need. This framework establishes an urgent step toward scalable, equitable, and intelligent systems that directly support populations facing food insecurity and its compounding health risks.
>
---
#### [replaced 074] Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究通用变换器在组合推理任务中的记忆需求，解决其递归推理效率与深度的权衡问题。通过引入记忆标记提升性能，发现初始化策略对训练稳定性至关重要。**

- **链接: [https://arxiv.org/pdf/2604.21999](https://arxiv.org/pdf/2604.21999)**

> **作者:** Grigory Sapunov
>
> **备注:** 12 pages, 7 figures, 8 tables. Code: this https URL
>
> **摘要:** We study learned memory tokens as computational scratchpad for a single-block Universal Transformer (UT) with Adaptive Computation Time (ACT) on Sudoku-Extreme, a combinatorial reasoning benchmark. We find that memory tokens are empirically necessary: across all configurations tested -- 3 seeds, multiple token counts, two initialization schemes, ACT and fixed-depth processing -- no configuration without memory tokens achieves non-trivial performance. The optimal count exhibits a sharp lower threshold (T=0 always fails, T=4 is borderline, T=8 reliably succeeds for 81-cell puzzles) followed by a stable plateau (T=8-32, 57.4% +/- 0.7% exact-match) and collapse from attention dilution at T=64. During experimentation, we identify a router initialization trap that causes >70% of training runs to fail: both default zero-bias initialization (p ~ 0.5) and Graves' recommended positive bias (p ~ 0.73) cause tokens to halt after ~2 steps at initialization, settling into a shallow equilibrium (halt ~ 5-7) that the model cannot escape. Inverting the bias to -3 ("deep start," p ~ 0.05) eliminates this failure mode. We confirm through ablation that the trap is inherent to ACT initialization, not an artifact of our architecture choices. With reliable training established, we show that (1) ACT provides more consistent results than fixed-depth processing (56.9% +/- 0.7% vs 53.4% +/- 9.3% across 3 seeds); (2) ACT with lambda warmup achieves matching accuracy (57.0% +/- 1.1%) using 34% fewer ponder steps; and (3) attention heads specialize into memory readers, constraint propagators, and integrators across recursive depth. Code is available at this https URL.
>
---
#### [replaced 075] Indirect Question Answering in English, German and Bavarian: A Challenging Task for High- and Low-Resource Languages Alike
- **分类: cs.CL**

- **简介: 该论文研究间接问答（IQA）任务，旨在分类间接回答的极性。针对英、德及巴伐利亚语，构建了两个语料库，并测试了多语言模型的表现，发现其在高、低资源语言中均表现不佳。**

- **链接: [https://arxiv.org/pdf/2603.15130](https://arxiv.org/pdf/2603.15130)**

> **作者:** Miriam Winkler; Verena Blaschke; Barbara Plank
>
> **备注:** LREC 2026 (this version fixes an error with the baseline scores)
>
> **摘要:** Indirectness is a common feature of daily communication, yet is underexplored in NLP research for both low-resource as well as high-resource languages. Indirect Question Answering (IQA) aims at classifying the polarity of indirect answers. In this paper, we present two multilingual corpora for IQA of varying quality that both cover English, Standard German and Bavarian, a German dialect without standard orthography: InQA+, a small high-quality evaluation dataset with hand-annotated labels, and GenIQA, a larger training dataset, that contains artificial data generated by GPT-4o-mini. We find that IQA is a pragmatically hard task that comes with various challenges, based on several experiment variations with multilingual transformer models (mBERT, XLM-R and mDeBERTa). We suggest and employ recommendations to tackle these challenges. Our results reveal low performance, even for English, and severe overfitting. We analyse various factors that influence these results, including label ambiguity, label set and dataset size. We find that the IQA performance is poor in high- (English, German) and low-resource languages (Bavarian) and that it is beneficial to have a large amount of training data. Further, GPT-4o-mini does not possess enough pragmatic understanding to generate high-quality IQA data in any of our tested languages.
>
---
#### [replaced 076] Rank-Turbulence Delta and Interpretable Approaches to Stylometric Delta Metrics
- **分类: cs.CL**

- **简介: 该论文属于作者身份归属任务，旨在提升stylometric delta度量的准确性与可解释性。提出两种新指标，通过概率分布距离函数改进传统方法，并在多语种语料库中验证效果。**

- **链接: [https://arxiv.org/pdf/2604.19499](https://arxiv.org/pdf/2604.19499)**

> **作者:** Dmitry Pronin; Evgeny Kazartsev
>
> **备注:** Under review at Digital Scholarship in the Humanities. Code available at: this https URL
>
> **摘要:** This article introduces two new measures for authorship attribution - Rank-Turbulence Delta and Jensen-Shannon Delta - which generalise Burrows's classical Delta by applying distance functions designed for probabilistic distributions. We first set out the theoretical basis of the measures, contrasting centred and uncentred z-scoring of word-frequency vectors and re-casting the uncentred vectors as probability distributions. Building on this representation, we develop a token-level decomposition that renders every Delta distance numerically interpretable, thereby facilitating close reading and the validation of results. The effectiveness of the methods is assessed on four literary corpora in English, German, French and Russian. The English, German and French datasets are compiled from Project Gutenberg, whereas the Russian benchmark is the SOCIOLIT corpus containing 639 works by 89 authors spanning the eighteenth to the twenty-first centuries. Rank-Turbulence Delta attains attribution accuracy comparable with Cosine Delta; Jensen-Shannon Delta consistently matches or exceeds the performance of canonical Burrows's Delta. Finally, several established attribution algorithms are re-evaluated on the extended SOCIOLIT corpus, providing a realistic estimate of their robustness under pronounced temporal and stylistic variation.
>
---
#### [replaced 077] PRISM: Probing Reasoning, Instruction, and Source Memory in LLM Hallucinations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在解决LLM幻觉问题。提出PRISM基准，用于诊断幻觉成因，涵盖知识、推理和指令遵循等维度。**

- **链接: [https://arxiv.org/pdf/2604.16909](https://arxiv.org/pdf/2604.16909)**

> **作者:** Yuhe Wu; Guangyu Wang; Yuran Chen; Jiatong Zhang; Yutong Zhang; Yujie Chen; Jiaming Shang; Guang Zhang; Zhuang Liu
>
> **备注:** Accepted by ACL main conference 2026
>
> **摘要:** As large language models (LLMs) evolve from conversational assistants into agents capable of handling complex tasks, they are increasingly deployed in high-risk domains. However, existing benchmarks largely rely on mixed queries and posterior evaluation, output-level scoring, which quantifies hallucination severity but offers limited insight into where and why hallucinations arise in the generation pipeline. We therefore reformulate hallucination evaluation as a diagnostic problem and propose PRISM, a controlled benchmark that disentangles hallucinations into four dimensions: knowledge missing, knowledge errors, reasoning errors, and instruction-following errors, grounded in three stages of generation (memory, instruction, and reasoning). PRISM contains 9,448 instances across 65 tasks and supports fine-grained, stage-aware diagnostic evaluation. Evaluating 24 mainstream open-source and proprietary LLMs, we uncover consistent trade-offs across instruction following, memory retrieval, and logical reasoning, showing that mitigation strategies often improve specific dimensions at the expense of others. We hope PRISM provides a framework for understanding the specific mechanisms behind LLMs hallucinations, ultimately accelerating the development of trustworthy large language models.
>
---
#### [replaced 078] Making Dialogue Grounding Data Rich: A Three-Tier Data Synthesis Framework for Generalized Referring Expression Comprehension
- **分类: cs.CL**

- **简介: 该论文属于对话式指代表达理解任务，解决领域分布差异导致的模型性能下降问题，提出一种三层数据合成方法以生成高质量标注数据。**

- **链接: [https://arxiv.org/pdf/2512.02791](https://arxiv.org/pdf/2512.02791)**

> **作者:** Juexi Shao; Siyou Li; Yujian Gan; Chris Madge; Vanja Karan; Massimo Poesio
>
> **摘要:** Dialogue-Based Generalized Referring Expression Comprehension (GREC) requires models to ground the expression and unlimited targets in complex visual scenes while resolving coreference across a long dialogue context. However, existing systems struggle under distribution shift between training and evaluation domains, a gap exacerbated by the scarcity of annotated dialogue grounding data. We address this challenge with a three-tier data-synthesis method that balances realism and controllability to produce scalable supervision for dialogue-conditioned grounding. Fine-tuning on the synthesized data yields consistent, substantial improvements over prior approaches across standard evaluation metrics.
>
---
#### [replaced 079] EmoBench-M: Benchmarking Emotional Intelligence for Multimodal Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感智能任务，旨在解决多模态大语言模型（MLLMs）情感理解能力评估不足的问题。提出EmoBench-M基准，涵盖13个场景，评估模型在情绪识别、理解与分析方面的能力。**

- **链接: [https://arxiv.org/pdf/2502.04424](https://arxiv.org/pdf/2502.04424)**

> **作者:** He Hu; Lianzhong You; Hongbo Xu; Qianning Wang; Fei Richard Yu; Fei Ma; Zebang Cheng; Zheng Lian; Yucheng Zhou; Laizhong Cui
>
> **摘要:** With the integration of multimodal large language models (MLLMs) into robotic systems and AI applications, embedding emotional intelligence (EI) capabilities is essential for enabling these models to perceive, interpret, and respond to human emotions effectively in real-world scenarios. Existing static, text-based, or text-image benchmarks overlook the multimodal complexities of real interactions and fail to capture the dynamic, context-dependent nature of emotional expressions, rendering them inadequate for evaluating MLLMs' EI capabilities. To address these limitations, we introduce EmoBench-M, a systematic benchmark grounded in established psychological theories, designed to evaluate MLLMs across 13 evaluation scenarios spanning three hierarchical dimensions: foundational emotion recognition (FER), conversational emotion understanding (CEU), and socially complex emotion analysis (SCEA). Evaluation was conducted on 27 state-of-the-art MLLMs, using both objective task-specific metrics and LLM-based evaluation, revealing a substantial performance gap relative to human-level competence. Even the best performing models, Gemini-3.0-Pro and GPT-5.2, achieve the highest scores on EmoBench-M, 70.5 and 66.5 points respectively. Specialized models such as AffectGPT exhibit uneven performance across EmoBench-M, demonstrating strengths in certain scenarios but generally lacking comprehensive emotional intelligence. By providing a comprehensive, multimodal evaluation framework, EmoBench-M captures both the strengths and weaknesses of current MLLMs across diverse emotional contexts. All benchmark resources, including datasets and code, are publicly available at this https URL, facilitating further research and advancement in MLLM emotional intelligence.
>
---
#### [replaced 080] CFDLLMBench: A Benchmark Suite for Evaluating Large Language Models in Computational Fluid Dynamics
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CFDLLMBench，一个用于评估大语言模型在计算流体力学中表现的基准套件，解决LLM在复杂物理系统数值实验自动化中的能力评估问题。**

- **链接: [https://arxiv.org/pdf/2509.20374](https://arxiv.org/pdf/2509.20374)**

> **作者:** Nithin Somasekharan; Ling Yue; Yadi Cao; Weichao Li; Patrick Emami; Pochinapeddi Sai Bhargav; Anurag Acharya; Xingyu Xie; Shaowu Pan
>
> **备注:** 40 pages
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong performance across general NLP tasks, but their utility in automating numerical experiments of complex physical system -- a critical and labor-intensive component -- remains underexplored. As the major workhorse of computational science over the past decades, Computational Fluid Dynamics (CFD) offers a uniquely challenging testbed for evaluating the scientific capabilities of LLMs. We introduce CFDLLMBench, a benchmark suite comprising three complementary components -- CFDQuery, CFDCodeBench, and FoamBench -- designed to holistically evaluate LLM performance across three key competencies: graduate-level CFD knowledge, numerical and physical reasoning of CFD, and context-dependent implementation of CFD workflows. Grounded in real-world CFD practices, our benchmark combines a detailed task taxonomy with a rigorous evaluation framework to deliver reproducible results and quantify LLM performance across code executability, solution accuracy, and numerical convergence behavior. CFDLLMBench establishes a solid foundation for the development and evaluation of LLM-driven automation of numerical experiments for complex physical systems. Code and data are available at this https URL.
>
---
#### [replaced 081] Stress-Testing Emotional Support Models: Moving from Homogeneous to Diverse Help Seekers
- **分类: cs.CL**

- **简介: 该论文属于情感支持聊天机器人评估任务，旨在解决现有模拟器行为单一、控制性差的问题。通过构建基于九个特征的可控模拟器，提升评估的准确性和多样性。**

- **链接: [https://arxiv.org/pdf/2601.07698](https://arxiv.org/pdf/2601.07698)**

> **作者:** Chaewon Heo; Cheyon Jin; Yohan Jo
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** As emotional support chatbots have recently gained significant traction across both research and industry, a common evaluation strategy has emerged: use help-seeker simulators to interact with supporter chatbots. However, current simulators suffer from two critical limitations: (1) they fail to capture the behavioral diversity of real-world seekers, often portraying them as overly cooperative, and (2) they lack the controllability required to simulate specific seeker profiles. To address these challenges, we present a controllable seeker simulator driven by nine psychological and linguistic features that underpin seeker behavior. Using authentic Reddit conversations, we train our model via a Mixture-of-Experts (MoE) architecture, which effectively differentiates diverse seeker behaviors into specialized parameter subspaces, thereby enhancing fine-grained controllability. Our simulator achieves superior profile adherence and behavioral diversity compared to existing approaches. Furthermore, evaluating 7 prominent supporter models with our system uncovers previously obscured performance degradations. These findings underscore the utility of our framework in providing a more faithful and stress-tested evaluation for emotional support chatbots.
>
---
#### [replaced 082] On Emergent Social World Models -- Evidence for Functional Integration of Theory of Mind and Pragmatic Reasoning in Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言模型是否具备整合心智理论与语用推理的社交世界模型，通过实验验证其功能整合性，属于自然语言处理中的社会认知研究任务。**

- **链接: [https://arxiv.org/pdf/2602.10298](https://arxiv.org/pdf/2602.10298)**

> **作者:** Polina Tsvilodub; Jan-Felix Klumpp; Amir Mohammadpour; Jennifer Hu; Michael Franke
>
> **备注:** 39 pages, 20 figures, accepted to ACL 2026 Main Conference
>
> **摘要:** This paper investigates whether LMs recruit shared computational mechanisms for general Theory of Mind (ToM) and language-specific pragmatic reasoning in order to contribute to the general question of whether LMs may be said to have emergent "social world models", i.e., representations of mental states that are repurposed across tasks (the functional integration hypothesis). Using behavioral evaluations and causal-mechanistic experiments via functional localization methods inspired by cognitive neuroscience, we analyze LMs' performance across seven subcategories of ToM abilities (Beaudoin et al., 2020) on a substantially larger localizer dataset than used in prior like-minded work. Results from stringent hypothesis-driven statistical testing offer suggestive evidence for the functional integration hypothesis, indicating that LMs may develop interconnected "social world models" rather than isolated competencies. This work contributes novel ToM localizer data, methodological refinements to functional localization techniques, and empirical insights into the emergence of social cognition in artificial systems.
>
---
#### [replaced 083] Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决推理过程中生成长链式思维（CoT）成本高的问题。通过引入抽象链式思维（Abstract-CoT），用离散的抽象标记替代自然语言CoT，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2604.22709](https://arxiv.org/pdf/2604.22709)**

> **作者:** Keshav Ramji; Tahira Naseem; Ramón Fernandez Astudillo
>
> **摘要:** While long, explicit chains-of-thought (CoT) have proven effective on complex reasoning tasks, they are costly to generate during inference. Non-verbal reasoning methods have emerged with shorter generation lengths by leveraging continuous representations, yet their performance lags behind verbalized CoT. We propose $\textbf{Abstract Chain-of-Thought}$, a discrete latent reasoning post-training mechanism in which the language model produces a short sequence of tokens from a reserved vocabulary in lieu of a natural language CoT, before generating a response. To make previously unseen ''abstract'' tokens useful, we introduce a policy iteration-style warm-up loop that alternates between (i.) bottlenecking from a verbal CoT via masking and performing supervised fine-tuning, and (ii.) self-distillation by training the model to generate abstract tokens from the prompt alone via constrained decoding with the codebook. After warm-up, we optimize the generation of abstract sequences with warm-started reinforcement learning under constrained decoding. Abstract-CoT achieves up to $11.6\times$ fewer reasoning tokens while demonstrating comparable performance across mathematical reasoning, instruction-following, and multi-hop reasoning, and generalizes across language model families. We also find an emergent power law distribution over the abstract vocabulary, akin to those seen in natural language, that evolves across the training phases. Our findings highlight the potential for post-training latent reasoning mechanisms that enable efficient inference through a learned abstract reasoning language.
>
---
#### [replaced 084] SSR-Zero: Simple Self-Rewarding Reinforcement Learning for Machine Translation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SSR-Zero框架，用于机器翻译任务，解决依赖外部监督信号的问题。通过自奖励机制提升模型性能，无需参考数据或人工标注。**

- **链接: [https://arxiv.org/pdf/2505.16637](https://arxiv.org/pdf/2505.16637)**

> **作者:** Wenjie Yang; Mao Zheng; Mingyang Song; Zheng Li; Sitong Wang
>
> **摘要:** Large language models (LLMs) have recently demonstrated remarkable capabilities in machine translation (MT). However, most advanced MT-specific LLMs heavily rely on external supervision signals during training, such as human-annotated reference data or trained reward models (RMs), which are often expensive to obtain and challenging to scale. To overcome this limitation, we propose a Simple Self-Rewarding (SSR) Reinforcement Learning (RL) framework for MT that is reference-free, fully online, and relies solely on self-judging rewards. Training with SSR using 13K monolingual examples and Qwen-2.5-7B as the backbone, our model SSR-Zero-7B outperforms existing MT-specific LLMs, e.g., TowerInstruct-13B and GemmaX-28-9B, as well as larger general LLMs like Qwen2.5-32B-Instruct in English $\leftrightarrow$ Chinese translation tasks from WMT23, WMT24, and Flores200 benchmarks. Furthermore, by augmenting SSR with external supervision from COMET, our strongest model, SSR-X-Zero-7B, achieves state-of-the-art performance in English $\leftrightarrow$ Chinese translation, surpassing all existing open-source models under 72B parameters and even outperforming closed-source models. Our analysis highlights the effectiveness of the self-rewarding mechanism compared to the external LLM-as-a-judge approach in MT and demonstrates its complementary benefits when combined with trained RMs. Our findings provide valuable insight into the potential of self-improving RL methods. We have publicly released our code, data and models.
>
---
#### [replaced 085] Explaining Sources of Uncertainty in Automated Fact-Checking
- **分类: cs.CL**

- **简介: 该论文属于事实核查任务，旨在解决模型不确定性解释不足的问题。通过CLUE框架，识别文本冲突与共识，生成更准确的不确定性解释。**

- **链接: [https://arxiv.org/pdf/2505.17855](https://arxiv.org/pdf/2505.17855)**

> **作者:** Jingyi Sun; Greta Warren; Irina Shklovski; Isabelle Augenstein
>
> **摘要:** Understanding sources of a model's uncertainty regarding its predictions is crucial for effective human-AI collaboration. Prior work proposes using numerical uncertainty or hedges ("I'm not sure, but ..."), which do not explain uncertainty that arises from conflicting evidence, leaving users unable to resolve disagreements or rely on the output. We introduce CLUE (Conflict-and-Agreement-aware Language-model Uncertainty Explanations), the first framework to generate natural language explanations of model uncertainty by (i) identifying relationships between spans of text that expose claim-evidence or inter-evidence conflicts and agreements that drive the model's predictive uncertainty in an unsupervised way, and (ii) generating explanations via prompting and attention steering that verbalize these critical interactions. Across three language models and two fact-checking datasets, we show that CLUE produces explanations that are more faithful to the model's uncertainty and more consistent with fact-checking decisions than prompting for uncertainty explanations without span-interaction guidance. Human evaluators judge our explanations to be more helpful, more informative, less redundant, and more logically consistent with the input than this baseline. CLUE requires no fine-tuning or architectural changes, making it plug-and-play for any white-box language model. By explicitly linking uncertainty to evidence conflicts, it offers practical support for fact-checking and generalises readily to other tasks that require reasoning over complex information.
>
---
#### [replaced 086] AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AgentHER，用于LLM代理轨迹重标签，解决失败轨迹浪费问题。通过四阶段流程将失败数据转化为训练数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.21357](https://arxiv.org/pdf/2603.21357)**

> **作者:** Liang Ding
>
> **摘要:** LLM agents fail on the majority of real-world tasks -- GPT-4o succeeds on fewer than 15% of WebArena navigation tasks and below 55% pass@1 on ToolBench (Zhou et al., 2024; Qin et al., 2024) -- yet every failed trajectory is routinely discarded, wasting the dominant source of collected experience. We introduce AgentHER, a framework that recovers this lost training signal by adapting the Hindsight Experience Replay (HER; Andrychowicz et al., 2017) principle to natural-language agent trajectories for offline data augmentation. The key insight is simple: a trajectory that fails goal A is often a correct demonstration for some achievable alternative goal B. AgentHER realises this idea through a four-stage pipeline -- failure classification, outcome extraction, LLM-guided prompt relabeling with confidence gating, and data packaging -- that converts discarded failures into high-quality SFT, DPO, and ShareGPT training data, with both zero-cost rule-based and LLM-judge implementations. On WebArena (Zhou et al., 2024) and ToolBench (Qin et al., 2024), AgentHER improves over success-only SFT by +7.1-11.7 pp across four model families (GPT-4o, Qwen2.5-72B/7B, LLaMA-3.1-8B), while achieving 2x data efficiency -- matching baseline performance with only 50% of successful demonstrations. Gains are consistent from 1.5B to 72B parameters (+5.8-9.2 pp) and compound under iterative redeployment (+2.1 pp over additional rounds). Human evaluation confirms 97.7% relabeling precision under multi-judge verification.
>
---
#### [replaced 087] How to measure the optimality of word or gesture order with respect to the principle of swap distance minimization
- **分类: cs.CL; cond-mat.stat-mech; physics.soc-ph**

- **简介: 该论文研究语言或手势顺序的最优性，解决如何衡量其在交换距离最小化下的优化程度。通过数学框架和QAP模型，验证跨语言手势的高优化性。**

- **链接: [https://arxiv.org/pdf/2604.01938](https://arxiv.org/pdf/2604.01938)**

> **作者:** Ramon Ferrer-i-Cancho
>
> **备注:** Little corrections specially in appendix B and C
>
> **摘要:** The structure of all the permutations of a sequence can be represented as a permutohedron, a graph where vertices are permutations and two vertices are linked if a swap of adjacent elements in the permutation of one of the vertices produces the permutation of the other vertex. It has been hypothesized that word orders in languages minimize the swap distance in the permutohedron: given a source order, word orders that are closer in the permutohedron should be less costly and thus more likely. Here we explain how to measure the degree of optimality of word order variation with respect to swap distance minimization. We illustrate the power of our novel mathematical framework by showing that crosslinguistic gestures are at least $77\%$ optimal. It is unlikely that the multiple times where crosslinguistic gestures hit optimality are due to chance. We establish the theoretical foundations for research on the optimality of word or gesture order with respect to swap distance minimization in communication systems. Finally, we introduce the quadratic assignment problem (QAP) into language research as an umbrella for multiple optimization problems and, accordingly, postulate a general principle of optimal assignment that unifies various linguistic principles including swap distance minimization.
>
---
#### [replaced 088] When Annotators Agree but Labels Disagree: The Projection Problem in Stance Detection
- **分类: cs.CL; cs.SI**

- **简介: 该论文属于立场检测任务，探讨标注者一致但标签不一致的投影问题。通过分析多维度态度，发现标签分歧源于维度权重不同，而非混淆。**

- **链接: [https://arxiv.org/pdf/2603.24231](https://arxiv.org/pdf/2603.24231)**

> **作者:** Bowen Zhang
>
> **摘要:** Stance detection is nearly always formulated as classifying text into Favor, Against, or Neutral. This convention was inherited from debate analysis and has been applied without modification to social media since SemEval-2016. However, attitudes toward complex targets are not unitary. A person can accept climate science while opposing carbon taxes, expressing support on one dimension and opposition on another. When annotators must compress such multi-dimensional attitudes into a single label, different annotators may weight different dimensions, producing disagreement that reflects different compression choices rather than confusion. We call this the projection problem. We conduct an annotation study across five targets from three stance benchmarks (SemEval-2016, P-Stance, COVID-19-Stance), with the same three annotators labeling all targets. For each target, annotators assign both a standard stance label and per-dimension judgments along target-specific dimensions discovered through bottom-up analysis, using the same number of categories for both. Across all fifteen target--dimension pairs, dimensional agreement consistently exceeds label agreement. The gap appears to scale with target complexity: modest for a single-entity target like Joe Biden (AC1: 0.87 vs. 0.95), but large for a multi-faceted policy target like school closures (AC1: 0.21 vs. 0.71).
>
---
#### [replaced 089] SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出SWE-Pruner，解决编码代理中长上下文带来的高成本和低效问题。通过自适应剪枝，保留关键代码信息，提升任务效率与成功率。**

- **链接: [https://arxiv.org/pdf/2601.16746](https://arxiv.org/pdf/2601.16746)**

> **作者:** Yuhang Wang; Yuling Shi; Mo Yang; Rongrui Zhang; Shilin He; Heng Lian; Yuting Chen; Siyu Ye; Kai Cai; Xiaodong Gu
>
> **备注:** Accepted to ACL 2026. Code available at this https URL
>
> **摘要:** LLM agents have demonstrated remarkable capabilities in software development, but their performance is hampered by long interaction contexts, which incur high API costs and latency. While various context compression approaches such as LongLLMLingua have emerged to tackle this challenge, they typically rely on fixed metrics such as PPL, ignoring the task-specific nature of code understanding. As a result, they frequently disrupt syntactic and logical structure and fail to retain critical implementation details. In this paper, we propose SWE-Pruner, a self-adaptive context pruning framework tailored for coding agents. Drawing inspiration from how human programmers "selectively skim" source code during development and debugging, SWE-Pruner performs task-aware adaptive pruning for long contexts. Given the current task, the agent formulates an explicit goal (e.g., "focus on error handling") as a hint to guide the pruning targets. A lightweight neural skimmer (0.6B parameters) is trained to dynamically select relevant lines from the surrounding context given the goal. Evaluations across four benchmarks and multiple models validate SWE-Pruner's effectiveness in various scenarios, achieving 23-54% token reduction on agent tasks like SWE-Bench Verified while even improving success rates, and up to 14.84x compression on single-turn tasks like LongCodeQA with minimal performance impact.
>
---
#### [replaced 090] Diagnostic-Driven Layer-Wise Compensation for Post-Training Quantization of Encoder-Decoder ASR Models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，解决量化后模型精度下降问题。针对编码器-解码器模型的层间敏感性差异，提出FADE框架，通过自适应补偿系数提升量化效果。**

- **链接: [https://arxiv.org/pdf/2601.02455](https://arxiv.org/pdf/2601.02455)**

> **作者:** Xinyu Wang; Ziyu Zhao; Yajie Luo; Yihong Wu; Liheng Ma; Jingrui Tian; Lei Ding; Xiao-Wen Chang; Peng Lu
>
> **备注:** 9 pages, 4 figures, 3 tables
>
> **摘要:** Deploying Automatic Speech Recognition (ASR) models on memory-constrained edge devices requires aggressive low-bit weight quantization. Layer-wise post-training quantization is practical and effective, but it suffers from cross-layer error accumulation. Existing compensation methods typically use a single global strength for all layers, which is ill-suited to encoder-decoder ASR models whose acoustic encoder and linguistic decoder exhibit markedly different sensitivities to quantization noise. We propose FADE, a diagnostic-driven framework that assigns each layer an adaptive compensation coefficient by combining two complementary signals: an intrinsic vulnerability score from weight geometry and a calibration reliability score from the data-driven solution. The resulting layer-wise coefficient balances local quantization fidelity against cross-layer error correction, enabling tailored compensation without retraining or hyperparameter search. Experiments on Whisper, Moonshine, and Qwen3-ASR across four benchmarks show that FADE consistently improves mean Word Error Rate over strong baselines at both 3- and 4-bit precision while substantially reducing run-to-run variance.
>
---
#### [replaced 091] A Lightweight Explainable Guardrail for Prompt Safety
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种轻量级可解释的提示安全防护机制（LEG），用于检测不安全提示。解决提示安全问题，通过多任务学习联合分类与解释，提升模型可解释性与性能。**

- **链接: [https://arxiv.org/pdf/2602.15853](https://arxiv.org/pdf/2602.15853)**

> **作者:** Md Asiful Islam; Mihai Surdeanu
>
> **摘要:** We propose a lightweight explainable guardrail (LEG) method to detect unsafe prompts. LEG uses a multi-task learning architecture to jointly learn a prompt classifier and an explanation classifier, where the latter labels prompt words that explain the safe/unsafe overall decision. LEG is trained on synthetic explanation data, which is generated using a novel strategy that counteracts the confirmation biases of LLMs. Lastly, LEG's training process uses a novel loss that captures global explanation signals as a weak supervision and combines cross-entropy and focal losses with uncertainty-based weighting. LEG obtains equivalent or better performance than the state-of-the-art for both prompt classification and explainability, both in-domain and out-of-domain on three datasets, despite the fact that its model size is considerably smaller than current approaches.
>
---
#### [replaced 092] On the Reasoning Abilities of Masked Diffusion Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究Masked Diffusion Language Models（MDMs）的推理能力，探讨其在特定任务中的效率与适用性，解决传统模型在并行生成中的局限性。**

- **链接: [https://arxiv.org/pdf/2510.13117](https://arxiv.org/pdf/2510.13117)**

> **作者:** Anej Svete; Ashish Sabharwal
>
> **摘要:** Masked diffusion models (MDMs) for text offer a compelling alternative to traditional autoregressive language models. Parallel generation makes them efficient, but their computational capabilities and the limitations inherent in their parallelism remain largely unexplored. To this end, we characterize what types of reasoning problems MDMs can provably solve and how efficiently. We do this by connecting MDMs to the well-understood reasoning frameworks of chain of thought (CoT) and padded looped transformers (PLTs) in the finite-precision log-width setting: We show that MDMs and polynomially-padded PLTs are, in fact, equivalent in this setting, and that MDMs can solve all problems that CoT-augmented transformers can. Moreover, we showcase classes of problems (including regular languages) for which MDMs are inherently more efficient than CoT transformers, where parallel generation allows for substantially faster reasoning.
>
---
#### [replaced 093] AVISE: Framework for Evaluating the Security of AI Systems
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于AI安全评估任务，旨在解决AI系统漏洞检测问题。提出AVISE框架，开发SET测试工具，评估语言模型安全性。**

- **链接: [https://arxiv.org/pdf/2604.20833](https://arxiv.org/pdf/2604.20833)**

> **作者:** Mikko Lempinen; Joni Kemppainen; Niklas Raesalmi
>
> **备注:** Fixed LaTex label command typos in Tables 8 and 9; fixed minor typos
>
> **摘要:** As artificial intelligence (AI) systems are increasingly deployed across critical domains, their security vulnerabilities pose growing risks of high-profile exploits and consequential system failures. Yet systematic approaches to evaluating AI security remain underdeveloped. In this paper, we introduce AVISE (AI Vulnerability Identification and Security Evaluation), a modular open-source framework for identifying vulnerabilities in and evaluating the security of AI systems and models. As a demonstration of the framework, we extend the theory-of-mind-based multi-turn Red Queen attack into an Adversarial Language Model (ALM) augmented attack and develop an automated Security Evaluation Test (SET) for discovering jailbreak vulnerabilities in language models. The SET comprises 25 test cases and an Evaluation Language Model (ELM) that determines whether each test case was able to jailbreak the target model, achieving 92% accuracy, an F1-score of 0.91, and a Matthews correlation coefficient of 0.83. We evaluate nine recently released language models of diverse sizes with the SET and find that all are vulnerable to the augmented Red Queen attack to varying degrees. AVISE provides researchers and industry practitioners with an extensible foundation for developing and deploying automated SETs, offering a concrete step toward more rigorous and reproducible AI security evaluation.
>
---
#### [replaced 094] Learning to Refine: Self-Refinement of Parallel Reasoning in LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大模型推理优化任务，解决候选答案均错误时无法得到正确结果的问题。提出Refinement Gap指标和GSR框架，提升并行自修正效果。**

- **链接: [https://arxiv.org/pdf/2509.00084](https://arxiv.org/pdf/2509.00084)**

> **作者:** Qibin Wang; Pu Zhao; Shaohan Huang; Fangkai Yang; Lu Wang; Furu Wei; Qingwei Lin; Saravan Rajmohan; Dongmei Zhang
>
> **摘要:** Test-time scaling (TTS) has gained widespread attention for enhancing LLM reasoning. Existing approaches such as Best-of-N and majority voting are limited as their performance depends on the quality of candidate responses, making them unable to produce a correct solution when all candidates are incorrect. Parallel self-refinement, generating multiple candidates and synthesizing a refined answer conditioned on them, offers a promising alternative, but the underlying mechanism driving its effectiveness remains obscure. To bridge this gap in understanding, we introduce a new metric, the Refinement Gap, designed to quantify the relative improvement of self-refinement beyond majority voting. We show that the Refinement Gap exhibits a clear scaling trend with model size and is only weakly correlated with the base capability. Based on this discovery, we propose Generative Self-Refinement (GSR), a parallel test-time scaling framework that transfers the refinement policy from larger teacher models with higher refinement gap into smaller students. Crucially, GSR jointly trains a single model to generate strong candidates and refine a better final answer based on these candidates. Experimental results demonstrate that our method achieves state-of-the-art performance across five mathematical benchmarks over other parallel aggregation methods, while the learned refinement skill transfers across multiple model scales and families and exhibits robust generalization to an out-of-distribution domain.
>
---
#### [replaced 095] Scheming Ability in LLM-to-LLM Strategic Interactions
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文研究LLM在策略互动中的欺骗能力，解决多智能体环境下AI系统可能存在的战略欺骗问题。通过游戏理论框架测试模型的欺骗行为与效果。**

- **链接: [https://arxiv.org/pdf/2510.12826](https://arxiv.org/pdf/2510.12826)**

> **作者:** Thao Pham
>
> **备注:** 20 pages, 13 figures
>
> **摘要:** As large language model (LLM) agents are deployed autonomously in diverse contexts, evaluating their capacity for strategic deception becomes crucial. While recent research has examined how AI systems scheme against human developers, LLM-to-LLM scheming remains underexplored. We investigate the scheming ability and propensity of frontier LLM agents through two game-theoretic frameworks: a Cheap Talk signaling game and a Peer Evaluation adversarial game. Testing four models (GPT-4o, Gemini-2.5-pro, Claude-3.7-Sonnet, and Llama-3.3-70b), we measure scheming performance with and without explicit prompting while analyzing scheming tactics through chain-of-thought reasoning. When prompted, most models, especially Gemini-2.5-pro and Claude-3.7-Sonnet, achieved near-perfect performance. Critically, models exhibited significant scheming propensity without prompting: all models chose deception over confession in Peer Evaluation (100% rate), while models choosing to scheme in Cheap Talk succeeded at 95-100% rates. These findings highlight the need for robust evaluations using high-stakes game-theoretic scenarios in multi-agent settings.
>
---
#### [replaced 096] Reasoning Dynamics and the Limits of Monitoring Modality Reliance in Vision-Language Models
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文研究视觉语言模型的推理动态，探讨多模态信息整合问题。通过分析18个模型，揭示其对文本线索的依赖及推理过程中的偏差。**

- **链接: [https://arxiv.org/pdf/2604.14888](https://arxiv.org/pdf/2604.14888)**

> **作者:** Danae Sánchez Villegas; Samuel Lewis-Lim; Nikolaos Aletras; Desmond Elliott
>
> **摘要:** Recent advances in vision language models (VLMs) offer reasoning capabilities, yet how these unfold and integrate visual and textual information remains unclear. We analyze reasoning dynamics in 18 VLMs covering instruction-tuned and reasoning-trained models from two different model families. We track confidence over Chain-of-Thought (CoT), measure the corrective effect of reasoning, and evaluate the contribution of intermediate reasoning steps. We find that models are prone to answer inertia, in which early commitments to a prediction are reinforced, rather than revised during reasoning steps. While reasoning-trained models show stronger corrective behavior, their gains depend on modality conditions, from text-dominant to vision-only settings. Using controlled interventions with misleading textual cues, we show that models are consistently influenced by these cues even when visual evidence is sufficient, and assess whether this influence is recoverable from CoT. Although this influence can appear in the CoT, its detectability varies across models and depends on what is being monitored. Reasoning-trained models are more likely to explicitly refer to the cues, but their longer and fluent CoTs can still appear visually grounded while actually following textual cues, obscuring modality reliance. In contrast, instruction-tuned models refer to the cues less explicitly, but their shorter traces reveal inconsistencies with the visual input. Taken together, these findings indicate that CoT provides only a partial view of how different modalities drive VLM decisions, with important implications for the transparency and safety of multimodal systems.
>
---
#### [replaced 097] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决紧凑语言模型在资源受限环境下实现自主搜索能力的问题。通过DGPO方法提升模型的代理式RAG能力。**

- **链接: [https://arxiv.org/pdf/2508.20324](https://arxiv.org/pdf/2508.20324)**

> **作者:** Rikuto Kotoge; Mai Nishimura; Jiaxin Ma
>
> **备注:** Accepted at ACL 2026 Main
>
> **摘要:** Reinforcement Learning has emerged as a dominant post-training approach to elicit agentic RAG behaviors such as search and planning from language models. Despite its success with larger models, applying RL to compact models (e.g., 0.5--1B parameters) presents unique challenges. The compact models exhibit poor initial performance, resulting in sparse rewards and unstable training. To overcome these difficulties, we propose Distillation-Guided Policy Optimization (DGPO), which employs cold-start initialization from teacher demonstrations and continuous teacher guidance during policy optimization. To understand how compact models preserve agentic behavior, we introduce Agentic RAG Capabilities (ARC), a fine-grained metric analyzing reasoning, search coordination, and response synthesis. Comprehensive experiments demonstrate that DGPO enables compact models to achieve sophisticated agentic search behaviors, even outperforming the larger teacher model in some cases. DGPO makes agentic RAG feasible in computing resource-constrained environments.
>
---
#### [replaced 098] DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于语言模型水印防御任务，旨在解决 paraphrase 和 spoofing 攻击问题。提出 DualGuard，通过双流机制同时防御两类攻击，提升水印可靠性与可追溯性。**

- **链接: [https://arxiv.org/pdf/2512.16182](https://arxiv.org/pdf/2512.16182)**

> **作者:** Hao Li; Yubing Ren; Yanan Cao; Yingjie Li; Fang Fang; Shi Wang; Li Guo
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** With the rapid development of cloud-based services, large language models have become increasingly accessible through various web platforms. However, this accessibility has also led to growing risks of model abuse. LLM watermarking has emerged as an effective approach to mitigate such misuse and protect intellectual property. Existing watermarking algorithms, however, primarily focus on defending against paraphrase attacks while overlooking piggyback spoofing attacks, which can inject harmful content, compromise watermark reliability, and undermine trust in attribution. To address this limitation, we propose DualGuard, the first watermarking algorithm capable of defending against both paraphrase and spoofing attacks. DualGuard employs the adaptive dual-stream watermarking mechanism, in which two complementary watermark signals are dynamically injected based on the semantic content. This design enables DualGuard not only to detect but also to trace spoofing attacks, thereby ensuring reliable and trustworthy watermark detection. Extensive experiments conducted across multiple datasets and language models demonstrate that DualGuard achieves excellent detectability, robustness, traceability, and text quality, effectively advancing the state of LLM watermarking for real-world applications.
>
---
#### [replaced 099] ANCHOR: LLM-driven Subject Conditioning for Text-to-Image Synthesis
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于文本到图像生成任务，旨在解决现有模型在多主体理解和上下文推理上的不足。通过构建ANCHOR数据集并提出SAFE方法提升图像与文本的一致性。**

- **链接: [https://arxiv.org/pdf/2404.10141](https://arxiv.org/pdf/2404.10141)**

> **作者:** Aashish Anantha Ramakrishnan; Sharon X. Huang; Dongwon Lee
>
> **备注:** Accepted to The 64th Annual Meeting of the Association for Computational Linguistics (ACL) 2026
>
> **摘要:** Text-to-image (T2I) models have achieved remarkable progress in high-quality image synthesis, yet most benchmarks rely on simple, self-contained prompts, failing to capture the complexity of real-world captions. Human-written captions often involve multiple interacting subjects, rich contextual references, and abstractive phrasing, conditions under which current image-text encoders like CLIP struggle. To systematically study these deficiencies, we introduce ANCHOR, a large-scale dataset of 70K+ abstractive captions sourced from five major news media organizations. Analysis with ANCHOR reveals persistent failures in multi-subject understanding, context reasoning, and nuanced grounding. Motivated by these challenges, we propose Subject-Aware Fine-tuning (SAFE), which uses Large Language Models (LLMs) to extract key subjects and enhance their representation at the embedding-level. Experiments with contemporary models show that SAFE significantly improves image-caption consistency and human preference alignment, serving as a practical and scalable solution.
>
---
#### [replaced 100] AdaComp: Extractive Context Compression with Adaptive Predictor for Retrieval-Augmented Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于检索增强生成（RAG）任务，旨在解决上下文压缩中的压缩率确定问题。通过自适应预测器动态调整压缩率，提升效率与性能平衡。**

- **链接: [https://arxiv.org/pdf/2409.01579](https://arxiv.org/pdf/2409.01579)**

> **作者:** Qianchi Zhang; Hainan Zhang; Liang Pang; Hongwei Zheng; Zhiming Zheng
>
> **备注:** Accepted to KSEM 2026
>
> **摘要:** Retrieved documents containing noise will hinder RAG from detecting answer clues and make the inference process slow and expensive. Therefore, context compression is necessary to enhance its accuracy and efficiency. Existing context compression methods use extractive or generative models to retain the most query-relevant sentences or apply the information bottleneck theory to preserve sufficient information. However, these methods may face issues such as over-compression or high computational costs. We observe that the retriever often ranks relevant documents at the top, but the exact number of documents needed to answer the query is uncertain due to the impact of query complexity and retrieval quality: complex queries like multi-hop questions may require retaining more documents than simpler queries, and a low-quality retrieval may need to rely on more documents to generate accurate outputs. Therefore, determining the minimum number of required documents (compression rate) is still a challenge for RAG. In this paper, we introduce AdaComp, a low-cost extractive context compression method that adaptively determines the compression rate based on both query complexity and retrieval quality. Specifically, we first annotate the minimum top-k documents necessary for the RAG system to answer the current query as the compression rate and then construct triplets of the query, retrieved documents, and its compression rate. Then, we use this triplet dataset to train a compression-rate predictor. Experiments on three QA datasets and one conversational Multi-doc QA dataset show that AdaComp significantly reduces inference costs while maintaining performance nearly identical to uncompressed models, achieving a balance between efficiency and performance.
>
---
#### [replaced 101] CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多语言医疗推理任务，旨在解决大语言模型在多语言医疗场景中的可靠性问题。通过构建多语言数据集并提出CURE-MED框架，提升模型的语言一致性和逻辑正确性。**

- **链接: [https://arxiv.org/pdf/2601.13262](https://arxiv.org/pdf/2601.13262)**

> **作者:** Eric Onyame; Akash Ghosh; Subhadip Baidya; Sriparna Saha; Xiuying Chen; Chirag Agarwal
>
> **备注:** Accepted at ACL 2026, main conference, oral presentation
>
> **摘要:** While large language models (LLMs) have shown to perform well on monolingual mathematical and commonsense reasoning, they remain unreliable for multilingual medical reasoning applications, hindering their deployment in multilingual healthcare settings. We address this by first introducing CUREMED-BENCH, a high-quality multilingual medical reasoning dataset with open-ended reasoning queries with a single verifiable answer, spanning thirteen languages, including underrepresented languages such as Amharic, Yoruba, and Swahili. Building on this dataset, we propose CURE-MED, a curriculum-informed reinforcement learning framework that integrates code-switching-aware supervised fine-tuning and Group Relative Policy Optimization to jointly improve logical correctness and language stability. Across thirteen languages, our approach consistently outperforms strong baselines and scales effectively, achieving 85.21% language consistency and 54.35% logical correctness at 7B parameters, and 94.96% language consistency and 70.04% logical correctness at 32B parameters. These results support reliable and equitable multilingual medical reasoning in LLMs. The code and dataset are available at this https URL
>
---
#### [replaced 102] Survey in Characterizing Semantic Change
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的语义变化分析任务，旨在解决如何准确描述和理解词语意义变化的问题。文中综述了现有方法，提出了三种语义变化的分类方式。**

- **链接: [https://arxiv.org/pdf/2402.19088](https://arxiv.org/pdf/2402.19088)**

> **作者:** Jader Martins Camboim de Sá; Marcos Da Silveira; Cédric Pruski
>
> **摘要:** Live languages continuously evolve to integrate the cultural change of human societies. This evolution manifests through neologisms (new words) or \textbf{semantic changes} of words (new meaning to existing words). Understanding the meaning of words is vital for interpreting texts coming from different cultures (regionalism or slang), domains (e.g., technical terms), or periods. In computer science, these words are relevant to computational linguistics algorithms such as translation, information retrieval, question answering, etc. Semantic changes can potentially impact the quality of the outcomes of these algorithms. Therefore, it is important to understand and characterize these changes formally. The study of this impact is a recent problem that has attracted the attention of the computational linguistics community. Several approaches propose methods to detect semantic changes with good precision, but more effort is needed to characterize how the meaning of words changes and to reason about how to reduce the impact of semantic change. This survey provides an understandable overview of existing approaches to the \textit{characterization of semantic changes} and also formally defines three classes of characterizations: if the meaning of a word becomes more general or narrow (change in dimension) if the word is used in a more pejorative or positive/ameliorated sense (change in orientation), and if there is a trend to use the word in a, for instance, metaphoric or metonymic context (change in relation). We summarized the main aspects of the selected publications in a table and discussed the needs and trends in the research activities on semantic change characterization.
>
---
#### [replaced 103] DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models
- **分类: cs.CL**

- **简介: 该论文属于高效大模型推理任务，旨在解决长链推理导致的效率低下问题。通过DRP框架，结合剪枝与知识蒸馏，提升推理效率并保持准确性。**

- **链接: [https://arxiv.org/pdf/2505.13975](https://arxiv.org/pdf/2505.13975)**

> **作者:** Yuxuan Jiang; Dawei Li; Francis Ferraro
>
> **备注:** Published on ACL 2026
>
> **摘要:** While Large Reasoning Models (LRMs) have demonstrated success in complex reasoning tasks through long chain-of-thought (CoT) reasoning, their inference often involves excessively verbose reasoning traces, resulting in substantial inefficiency. To address this, we propose Distilled Reasoning Pruning (DRP), a hybrid framework that combines inference-time pruning with tuning-based distillation, two widely used strategies for efficient reasoning. DRP uses a teacher model to perform skill-aware step decomposition and content pruning, and then distills the pruned reasoning paths into a student model, enabling it to reason both efficiently and accurately. Across several challenging mathematical reasoning datasets, we find that models trained with DRP achieve substantial improvements in token efficiency without sacrificing accuracy. Specifically, DRP reduces average token usage on GSM8K from 917 to 328 while improving accuracy from 91.7% to 94.1%, and achieves a 43% token reduction on AIME with no performance drop. Further analysis shows that aligning the reasoning structure of training CoTs with the student's reasoning capacity is critical for effective knowledge transfer and performance gains.
>
---
#### [replaced 104] Arch: An AI-Native Hardware Description Language for Register-Transfer Clocked Hardware Design
- **分类: cs.PL; cs.CL**

- **简介: 该论文提出Arch，一种面向寄存器传输时序硬件设计的AI原生硬件描述语言，解决传统HDL易出错的问题，通过类型系统和AI生成实现更安全、高效的硬件设计。**

- **链接: [https://arxiv.org/pdf/2604.05983](https://arxiv.org/pdf/2604.05983)**

> **作者:** Shuqing Zhao
>
> **摘要:** We present Arch (AI-native Register-transfer Clocked Hardware), a hardware description language for micro-architecture specification and AI-assisted code generation. Arch provides first-class constructs for pipelines, FSMs, FIFOs, arbiters, register files, buses with handshake channels, clock-domain crossings, and multi-cycle threads -- structures that existing HDLs express only as user-defined patterns prone to subtle errors. A central design choice is that clocks and resets are parameterized types (Clock<D>, Reset<S,P,D?>) rather than ordinary nets, converting CDC and reset-domain analysis from external linter passes into compile-time typing rules. Bit widths, port directions, single-driver ownership, and combinational acyclicity are tracked in the same pass, catching latches, width mismatches, loops, and unsynchronized crossings before simulation. A guard clause on reg declarations captures the valid-data pattern declaratively, catching the producer bug where a valid flag asserts before data is written. Every syntactic choice is governed by an AI-generatability contract: an LL(1) grammar, no preprocessor, a uniform declaration schema, named block endings, and a todo! escape hatch let LLMs produce structurally correct, type-safe Arch from natural-language specs without fine-tuning. The compiler emits lint-clean IEEE 1800-2017 SystemVerilog and auto-generates safety properties (FIFO no-overflow, counter range, FSM legal-state, handshake protocol) verified with Verilator -- assert and EBMC, plus direct AST-to-SMT-LIB2 bounded model checking via arch formal. An integrated simulator compiles designs to native C++ with Python cocotb support. Case studies: L1 cache and AXI DMA (Yosys/OpenSTA, Sky130); 428/431 tests pass on VerilogEval and CVDP.
>
---
#### [replaced 105] When Silence Matters: The Impact of Irrelevant Audio on Text Reasoning in Large Audio-Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文研究音频干扰对文本推理的影响，属于多模态模型鲁棒性任务。解决无关音频降低推理准确性的问题，通过实验分析干扰因素并测试缓解策略。**

- **链接: [https://arxiv.org/pdf/2510.00626](https://arxiv.org/pdf/2510.00626)**

> **作者:** Chen-An Li; Tzu-Han Lin; Hung-yi Lee
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Large audio-language models (LALMs) unify speech and text processing, but their robustness in noisy real-world settings remains underexplored. We investigate how irrelevant audio, such as silence, synthetic noise, and environmental sounds, affects text reasoning tasks where audio is unnecessary. Across three text-based benchmarks, we find that even non-informative audio reduces accuracy and increases prediction volatility; the severity of interference scales with longer durations, higher amplitudes, and elevated decoding temperatures. Silence, often assumed neutral, destabilizes outputs as strongly as synthetic noise. While larger models show greater resilience, vulnerabilities persist across all evaluated systems. We further test mitigation strategies and find that prompting shows limited effectiveness, whereas self-consistency improves stability at the cost of increased computation. Our results reveal cross-modal interference as a key robustness challenge and highlight the need for efficient fusion strategies that preserve reasoning performance in the presence of irrelevant inputs.
>
---
#### [replaced 106] For-Value: Efficient Forward-Only Data Valuation for finetuning LLMs and VLMs
- **分类: cs.CL**

- **简介: 该论文属于数据估值任务，旨在解决大模型微调中数据价值评估效率低的问题。提出For-Value框架，通过单次前向计算高效估计数据价值。**

- **链接: [https://arxiv.org/pdf/2508.10180](https://arxiv.org/pdf/2508.10180)**

> **作者:** Wenlong Deng; Qi Zeng; Jiaming Zhang; Minghui Chen; Zixin Ding; Christos Thrampoulidis; Boying Gong; Xiaoxiao Li
>
> **摘要:** Data valuation is essential for enhancing the transparency and accountability of large language models (LLMs) and vision-language models (VLMs). However, existing methods typically rely on gradient computations, making them computationally prohibitive for billion-parameter models and precluding batch parallelization. In this work, we introduce For-Value, a forward-only data valuation framework that enables efficient batch-scalable value estimation while maintaining effectiveness. Leveraging the expressive power of pretrained LLMs/VLMs, we theoretically demonstrate that data valuation can be captured by the alignment between the final hidden representations and prediction errors at the last layer. In light of this insight, For-Value computes data value using a simple closed-form expression with a single forward pass, eliminating the need for costly backpropagation and enabling efficient batch calculating at scale. Extensive experiments show that For-Value matches or outperforms gradient-based baselines in detecting influential data and mislabeled data, while achieving significant efficiency improvements.
>
---
#### [replaced 107] LinguDistill: Recovering Linguistic Ability in Vision-Language Models via Selective Cross-Modal Distillation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型任务，解决多模态适配导致语言能力下降的问题。通过无适配器的蒸馏方法恢复语言能力，保持视觉任务性能。**

- **链接: [https://arxiv.org/pdf/2604.00829](https://arxiv.org/pdf/2604.00829)**

> **作者:** Patrick Amadeus Irawan; Erland Hilman Fuadi; Shanu Kumar; Alham Fikri Aji; Yova Kementchedjhieva
>
> **摘要:** Adapting pretrained language models (LMs) into vision-language models (VLMs) can degrade their native linguistic capability due to representation shift and cross-modal interference introduced during multimodal adaptation. Such loss is difficult to recover, even with targeted task-specific fine-tuning using standard objectives. Prior recovery approaches typically introduce additional modules that act as intermediate alignment layers to maintain or isolate modality-specific subspaces, which increases architectural complexity, adds parameters at inference time, and limits flexibility across models and settings. We propose LinguDistill, an adapter-free distillation method that restores linguistic capability by utilizing the original frozen LM as a teacher. We overcome the key challenge of enabling vision-conditioned teacher supervision by introducing layer-wise KV-cache sharing, which exposes the teacher to the student's multimodal representations without modifying the architecture of either model. We then selectively distill the teacher's strong linguistic signal on language-intensive data to recover language capability, while preserving the student's visual grounding on multimodal tasks. As a result, LinguDistill recovers $\sim$10% of the performance lost on language and knowledge benchmarks, while maintaining comparable performance on vision-heavy tasks. Our findings demonstrate that linguistic capability can be recovered without additional modules, providing an efficient and practical solution to modality-specific degradation in multimodal models.
>
---
#### [replaced 108] Position: Logical Soundness is not a Reliable Criterion for Neurosymbolic Fact-Checking with LLMs
- **分类: cs.CL**

- **简介: 该论文属于事实核查任务，探讨逻辑严谨性在神经符号系统中的局限性，指出逻辑正确结论可能误导人类，提出结合LLM推理倾向的补充方法。**

- **链接: [https://arxiv.org/pdf/2604.04177](https://arxiv.org/pdf/2604.04177)**

> **作者:** Jason Chan; Robert Gaizauskas; Zhixue Zhao
>
> **备注:** ICLR 2026 Workshop on Logical Reasoning of Large Language Models
>
> **摘要:** As large language models (LLMs) are increasing integrated into fact-checking pipelines, formal logic is often proposed as a rigorous means by which to mitigate bias, errors and hallucinations in these models' outputs. For example, some neurosymbolic systems verify claims by using LLMs to translate natural language into logical formulae and then checking whether the proposed claims are logically sound, i.e. whether they can be validly derived from premises that are verified to be true. We argue that such approaches structurally fail to detect misleading claims due to systematic divergences between conclusions that are logically sound and inferences that humans typically make and accept. Drawing on studies in cognitive science and pragmatics, we present a typology of cases in which logically sound conclusions systematically elicit human inferences that are unsupported by the underlying premises. Consequently, we advocate for a complementary approach: leveraging human-like reasoning tendencies of LLMs as a feature rather than a bug, and using these models to validate the outputs of formal components in neurosymbolic systems against potentially misleading conclusions.
>
---
#### [replaced 109] MedSpeak: A Knowledge Graph-Aided ASR Error Correction Framework for Spoken Medical QA
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗语音问答任务，解决ASR识别医疗术语错误的问题。通过知识图谱和大模型提升术语识别与答案预测准确性。**

- **链接: [https://arxiv.org/pdf/2602.00981](https://arxiv.org/pdf/2602.00981)**

> **作者:** Yutong Song; Shiva Shrestha; Chenhan Lyu; Elahe Khatibi; Pengfei Zhang; Honghui Xu; Nikil Dutt; Amir Rahmani
>
> **摘要:** Spoken question-answering (SQA) systems relying on automatic speech recognition (ASR) often struggle with accurately recognizing medical terminology. To this end, we propose MedSpeak, a novel knowledge graph-aided ASR error correction framework that refines noisy transcripts and improves downstream answer prediction by leveraging both semantic relationships and phonetic information encoded in a medical knowledge graph, together with the reasoning power of LLMs. Comprehensive experimental results on benchmarks demonstrate that MedSpeak significantly improves the accuracy of medical term recognition and overall medical SQA performance, establishing MedSpeak as a state-of-the-art solution for medical SQA. The code is available at this https URL.
>
---
#### [replaced 110] EVE: A Domain-Specific LLM Framework for Earth Intelligence
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EVE框架，用于构建和部署地球智能领域的专用大模型。解决领域模型泛化与精准问答问题，通过优化模型、构建基准测试及集成RAG和幻觉检测技术实现高效应用。**

- **链接: [https://arxiv.org/pdf/2604.13071](https://arxiv.org/pdf/2604.13071)**

> **作者:** Àlex R. Atrio; Antonio Lopez; Jino Rohit; Yassine El Ouahidi; Marcello Politi; Vijayasri Iyer; Umar Jamil; Sébastien Bratières; Nicolas Longépé
>
> **备注:** To be published in the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** We introduce Earth Virtual Expert (EVE), the first open-source, end-to-end initiative for developing and deploying domain-specialized LLMs for Earth Intelligence. At its core is EVE-Instruct, a domain-adapted 24B model built on Mistral Small 3.2 and optimized for reasoning and question answering. On newly constructed Earth Observation and Earth Sciences benchmarks, it outperforms comparable models while preserving general capabilities. We release curated training corpora and the first systematic domain-specific evaluation benchmarks, covering MCQA, open-ended QA, and factuality. EVE further integrates RAG and a hallucination-detection pipeline into a production system deployed via API and GUI, supporting 350 pilot users so far. All models, datasets, and code are ready to be released under open licenses as contributions to our field at this http URL and this http URL.
>
---
#### [replaced 111] One Token Away from Collapse: The Fragility of Instruction-Tuned Helpfulness
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究指令调优大模型在简单约束下的脆弱性，揭示其响应能力的下降问题。属于自然语言处理任务，旨在探讨模型鲁棒性及评估方法缺陷。**

- **链接: [https://arxiv.org/pdf/2604.13006](https://arxiv.org/pdf/2604.13006)**

> **作者:** Erfan Baghaei Potraghloo; Seyedarmin Azizi; Souvik Kundu; Massoud Pedram
>
> **摘要:** Instruction-tuned large language models produce helpful, structured responses, but how robust is this helpfulness under trivial constraints? We show that simple lexical constraints (banning a single punctuation character or common word) cause instruction-tuned LLMs to collapse their responses, losing 14--48\% of comprehensiveness across seven models spanning five families (7B--70B, open- and closed-weight). A blinded human evaluation with 10 STEM-trained evaluators confirms genuine content loss, with information criteria degrading $1.5$--$2.3\times$ more than surface criteria, a finding corroborated by over 4,100 automated pairwise comparisons (77--100\% baseline preference) across three LLM judges from two model families. Diagnostic analysis identifies this as a \emph{planning failure}: two-pass generation recovers 59--96\% of response length, and linear probes on prompt representations predict response length with $R^2 = 0.51$--$0.94$ before generation begins. The same probes yield negative $R^2$ on base models, confirming that instruction tuning introduces the representational structure underlying the collapse. Base models show no systematic degradation under identical constraints, demonstrating that instruction tuning couples task competence to narrow surface-form templates. The effect extends to realistic deployment constraints (preamble suppression, corporate tone guidelines, legal compliance hedging, accessibility requirements) causing comparable degradation ($-$22\% to $-$34\%), with suppressing the conversational opener alone (``Certainly!'') causing 40\% collapse on our most fragile model despite restricting only the opening tokens. We further show that standard independent LLM-as-judge evaluation detects only a 3.5\% quality drop where pairwise evaluation reveals 23\%, exposing a methodological blind spot in current evaluation practice.
>
---
#### [replaced 112] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到图像检索任务，旨在解决跨模态对齐不足的问题。通过先生成图像再检索的方法，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2505.20291](https://arxiv.org/pdf/2505.20291)**

> **作者:** Di Wu; Yixin Wan; Kai-Wei Chang
>
> **备注:** ACL 2026 Camera Ready
>
> **摘要:** Text-to-image retrieval (T2I retrieval) remains challenging because cross-modal embeddings often behave as bags of concepts, underrepresenting structured visual relationships such as pose and viewpoint. We proposeVisualize-then-Retrieve (VisRet), a retrieval paradigm that mitigates this limitation of cross-modal similarity alignment. VisRet first projects textual queries into the image modality via T2I generation, then performs retrieval within the image modality to bypass the weaknesses of cross-modal retrievers in recognizing subtle visual-spatial features. Across four benchmarks (Visual-RAG, INQUIRE-Rerank, Microsoft COCO, and our new Visual-RAG-ME featuring multi-entity comparisons), VisRet substantially outperforms cross-modal similarity matching and baselines that recast T2I retrieval as text-to-text similarity matching, improving nDCG@30 by 0.125 on average with CLIP as the retriever and by 0.121 with E5-V. For downstream question answering, VisRet increases accuracy on Visual-RAG and Visual-RAG-ME by 3.8% and 15.7% in top-1 retrieval, and by 3.9% and 11.1% in top-10 retrieval. Ablation studies show compatibility with different T2I instruction LLMs, T2I generation models, and downstream LLMs. VisRet provides a simple yet effective perspective for advancing in text-image retrieval. Our code and the new benchmark are publicly available at this https URL.
>
---
#### [replaced 113] Beyond Context: Large Language Models' Failure to Grasp Users' Intent
- **分类: cs.AI; cs.CL; cs.CR; cs.CY**

- **简介: 该论文属于AI安全研究任务，探讨LLMs无法理解用户意图的问题。通过实验发现现有模型在情感、渐进和学术手段下易被绕过安全机制，提出需加强上下文与意图识别作为核心安全能力。**

- **链接: [https://arxiv.org/pdf/2512.21110](https://arxiv.org/pdf/2512.21110)**

> **作者:** Ahmed M. Hussain; Salahuddin Salahuddin
>
> **备注:** 22 pages and 23 figures; updated authors list and revised manuscript
>
> **摘要:** Current Large Language Models (LLMs) safety approaches focus on explicitly harmful content while overlooking a critical vulnerability: the inability to understand context and recognize user intent. This creates exploitable vulnerabilities that malicious users can systematically leverage to circumvent safety mechanisms. We empirically evaluate multiple state-of-the-art LLMs, including ChatGPT, Claude, Gemini, and DeepSeek. Our analysis demonstrates the circumvention of reliable safety mechanisms through emotional framing, progressive revelation, and academic justification techniques. Notably, reasoning-enabled configurations amplified rather than mitigated the effectiveness of exploitation, increasing factual precision while failing to interrogate the underlying intent. The exception was Claude Opus 4.1, which prioritized intent detection over information provision in some use cases. This pattern reveals that current architectural designs create systematic vulnerabilities. These limitations require paradigmatic shifts toward contextual understanding and intent recognition as core safety capabilities rather than post-hoc protective mechanisms.
>
---
#### [replaced 114] AlphaContext: An Evolutionary Tree-based Psychometric Context Generator for Creativity Assessment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于创造力评估任务，旨在解决高质量评估情境稀缺的问题。提出AlphaContext，通过树状结构生成并优化情境，提升评估效果。**

- **链接: [https://arxiv.org/pdf/2604.18398](https://arxiv.org/pdf/2604.18398)**

> **作者:** Yixuan Wang; Yue Huang; Hong Qian; Yunzhao Wei; Yifei Ding; Wenkai Wang; Zhi Liu; Zhongjing Huang; Aimin Zhou; Jiajun Guo
>
> **备注:** Accepted by ACL 2026 Main Conference
>
> **摘要:** Creativity has become a core competence in the era of LLMs and human-AI collaboration, underpinning innovation in real-world problem solving. Crucially, the systematic improvement of creativity necessitates scientifically valid assessment instruments. Psychometric research recognizes context-based assessment as an effective way to measure creative thinking. However, high-quality expert-designed contexts remain scarce. Existing LLM-based generators often struggle with insufficient assessment cues, weak narrative coherence, limited stylistic diversity, and poor support for creative thinking. To address these challenges, we propose AlphaContext, an evolutionary tree-based psychometric context generator for creativity assessment. First, the HyperTree Outline Planner formalizes expert-designed outlining as a rule-guided hypertree and performs top-down hierarchical planning. The MCTS-based Context Generator fills the outline via MCTS to balance global structure and local quality. Then, the Evolutionary Context Optimizer evolves contexts with MAP-Elites by repeatedly updating niche elites to jointly improve diversity and quality. Finally, the Assessment-Guided Evolution Refiner simulates virtual participants with diverse styles and recycles weak contexts for further evolution. Experiments show that AlphaContext yields an average improvement of 8% over competitive methods across 6 quality metrics.
>
---
#### [replaced 115] CUB: Benchmarking Context Utilisation Techniques for Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识密集型任务，旨在解决语言模型在利用外部知识时忽略或被干扰的问题。提出CUB基准，评估多种上下文利用技术的有效性。**

- **链接: [https://arxiv.org/pdf/2505.16518](https://arxiv.org/pdf/2505.16518)**

> **作者:** Lovisa Hagström; Youna Kim; Haeun Yu; Sang-goo Lee; Richard Johansson; Hyunsoo Cho; Isabelle Augenstein
>
> **备注:** Accepted at ACL 2026, 33 pages
>
> **摘要:** Incorporating external knowledge is crucial for knowledge-intensive tasks, such as question answering and fact checking. However, language models (LMs) may ignore relevant information that contradicts outdated parametric memory or be distracted by irrelevant contexts. While many context utilisation manipulation techniques (CMTs) have recently been proposed to alleviate these issues, few have seen systematic comparison. In this paper, we develop CUB (Context Utilisation Benchmark) - the first comprehensive benchmark designed to help diagnose CMTs under diverse noisy context conditions within retrieval-augmented generation (RAG). With this benchmark, we conduct the most extensive evaluation to date of seven state-of-the-art methods, representative of the main categories of CMTs, across three diverse datasets and tasks, applied to 11 LMs. Our findings expose critical gaps in current CMT evaluation practices, demonstrating the need for holistic testing. We reveal that most existing CMTs struggle to handle the full spectrum of context types encountered in real-world RAG scenarios. We also find that many CMTs display inflated performance on simple synthesised datasets, compared to more realistic datasets with naturally occurring samples.
>
---
#### [replaced 116] Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究CoT推理在LLM中的有效性，探讨其在分布外数据上的局限性。任务是分析CoT推理的可靠性，解决其是否为虚假现象的问题。工作包括提出数据分布视角，设计DataAlchemy实验环境进行验证。**

- **链接: [https://arxiv.org/pdf/2508.01191](https://arxiv.org/pdf/2508.01191)**

> **作者:** Chengshuai Zhao; Zhen Tan; Pingchuan Ma; Dawei Li; Bohan Jiang; Yancheng Wang; Yingzhen Yang; Huan Liu
>
> **备注:** Accepted by the Association for Computational Linguistics (ACL) 2026 and Foundations of Reasoning in Language Models (FoRLM) at NeurIPS 2025
>
> **摘要:** Chain-of-Thought (CoT) prompting has been shown to be effective in eliciting structured reasoning (i.e., CoT reasoning) from large language models (LLMs). Regardless of its popularity, recent studies expose its failures in some reasoning tasks, raising fundamental questions about the nature of CoT reasoning. In this work, we propose a data distribution lens to understand when and why CoT reasoning succeeds or fails. We hypothesize that CoT reasoning reflects a structured inductive bias learned from in-distribution data, enabling models to conditionally generate reasoning trajectories that approximate those observed during training. As such, the effectiveness of CoT reasoning is fundamentally governed by the nature and degree of distribution discrepancy between training data and test queries. Guided by this lens, we dissect CoT reasoning via three dimensions: task, length, and format. To test the hypothesis, we introduce DataAlchemy, an abstract and fully controllable environment that trains LLMs from scratch and systematically probes them under various distribution conditions. Through rigorous controlled experiments, we reveal that CoT reasoning is a brittle mirage when it is pushed beyond training distributions, emphasizing the ongoing challenge of achieving genuine and generalizable reasoning.
>
---
#### [replaced 117] Investigating the Representation of Backchannels and Fillers in Fine-tuned Language Models
- **分类: cs.CL**

- **简介: 该论文属于对话理解任务，旨在解决语言模型对回声词和填充词表示不足的问题。通过微调策略，提升模型对这些语言现象的捕捉能力。**

- **链接: [https://arxiv.org/pdf/2509.20237](https://arxiv.org/pdf/2509.20237)**

> **作者:** Yu Wang; Leyi Lao; Langchu Huang; Gabriel Skantze; Yang Xu; Hendrik Buschmeier
>
> **备注:** Accepted at ACL 2026 main
>
> **摘要:** Backchannels and fillers are important linguistic expressions in dialogue, but often treated as 'noise' to be bypassed in modern transformer-based language models (LMs). Here, we study how they are represented in LMs using three fine-tuning strategies on three dialogue corpora in English and Japanese, in which backchannels and fillers are both preserved and annotated. This allows us to investigate how fine-tuning can help LMs learn these representations. We first apply clustering analysis to the learnt representation of backchannels and fillers, and find increased silhouette scores in representations from fine-tuned models, which suggests that fine-tuning enables LMs to distinguish the nuanced semantic variation in different backchannel and filler use. We also employ natural language generation metrics and qualitative analyses to verify that utterances produced by fine-tuned LMs resemble those produced by humans more closely. Our findings suggest the potential for transforming general LMs into conversational LMs that can produce human-like language more adequately.
>
---
#### [replaced 118] V-SEAM: Visual Semantic Editing and Attention Modulating for Causal Interpretability of Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文属于视觉语言模型的因果可解释性任务，旨在通过语义级视觉操作和注意力调节，提升模型的可解释性与性能。**

- **链接: [https://arxiv.org/pdf/2509.14837](https://arxiv.org/pdf/2509.14837)**

> **作者:** Qidong Wang; Junjie Hu; Ming Jiang
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Recent advances in causal interpretability have extended from language models to vision-language models (VLMs), seeking to reveal their internal mechanisms through input interventions. While textual interventions often target semantics, visual interventions typically rely on coarse pixel-level perturbations, limiting semantic insights on multimodal integration. In this study, we introduce V-SEAM, a novel framework that combines Visual Semantic Editing and Attention Modulating for causal interpretation of VLMs. V-SEAM enables concept-level visual manipulations and identifies attention heads with positive or negative contributions to predictions across three semantic levels: objects, attributes, and relationships. We observe that positive heads are often shared within the same semantic level but vary across levels, while negative heads tend to generalize broadly. Finally, we introduce an automatic method to modulate key head embeddings, demonstrating enhanced performance for both LLaVA and InstructBLIP across three diverse VQA benchmarks. Our data and code are released at: this https URL.
>
---
#### [replaced 119] LongFlow: Efficient KV Cache Compression for Reasoning Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型推理优化任务，旨在解决长输出导致的KV缓存占用过高问题。通过提出LongFlow方法，实现高效压缩与计算优化。**

- **链接: [https://arxiv.org/pdf/2603.11504](https://arxiv.org/pdf/2603.11504)**

> **作者:** Yi Su; Zhenxu Tian; Dan Qiao; Yuechi Zhou; Juntao Li; Min Zhang
>
> **摘要:** Recent reasoning models such as OpenAI-o1 and DeepSeek-R1 have shown strong performance on complex tasks including mathematical reasoning and code generation. However, this performance gain comes with substantially longer output sequences, leading to significantly increased deployment costs. In particular, long outputs require large KV caches, resulting in high memory consumption and severe bandwidth pressure during attention computation. Most existing KV cache optimization methods are designed for long-input, short-output scenarios and are ineffective for the long-output setting of reasoning models. Moreover, importance estimation in prior work is computationally expensive and becomes prohibitive when continuous re-evaluation is required during long generation. To address these challenges, we propose LongFlow, a KV cache compression method with an efficient importance estimation metric derived from an intermediate result of attention computation using only the current query. This design introduces negligible computational overhead and requires no auxiliary storage. We further develop a custom kernel that fuses FlashAttention, importance estimation, and token eviction into a single optimized operator, improving system-level efficiency. Experiments show that LongFlow achieves up to an 11.8 times throughput improvement with 80% KV cache compression with minimal impact on model accuracy.
>
---
#### [replaced 120] AI use in American newspapers is widespread, uneven, and rarely disclosed
- **分类: cs.CL**

- **简介: 该论文属于新闻AI应用研究，旨在揭示AI在报纸中的使用现状。通过分析大量文章，发现AI使用广泛但不均，且披露极少，需加强透明度。**

- **链接: [https://arxiv.org/pdf/2510.18774](https://arxiv.org/pdf/2510.18774)**

> **作者:** Jenna Russell; Marzena Karpinska; Destiny Akinode; Katherine Thai; Bradley Emi; Max Spero; Mohit Iyyer
>
> **备注:** ACL Camera Ready
>
> **摘要:** AI is rapidly transforming journalism, but the extent of its use in published newspaper articles remains unclear. We address this gap by auditing a large-scale dataset of 186K articles from online editions of 1.5K American newspapers published in the summer of 2025. Using Pangram, a state-of-the-art AI detector, we discover that approximately 9% of newly-published articles are either partially or fully AI-generated. This AI use is unevenly distributed, appearing more frequently in smaller, local outlets, in specific topics such as weather and technology, and within certain ownership groups. We also analyze 45K opinion pieces from Washington Post, New York Times, and Wall Street Journal, finding that they are 6.4 times more likely to contain AI-generated content than news articles from the same publications, with many AI-flagged op-eds authored by prominent public figures. Despite this prevalence, we find that AI use is rarely disclosed: a manual audit of 100 AI-flagged articles found only five disclosures of AI use. Overall, our audit highlights the immediate need for greater transparency and updated editorial standards regarding the use of AI in journalism to maintain public trust.
>
---
#### [replaced 121] Learning to Conceal Risk: Controllable Multi-turn Red Teaming for LLMs in the Financial Domain
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于安全评估任务，旨在检测金融领域LLM的隐蔽风险行为。提出CoRT框架，通过多轮对话隐藏风险，提升攻击成功率。**

- **链接: [https://arxiv.org/pdf/2509.10546](https://arxiv.org/pdf/2509.10546)**

> **作者:** Gang Cheng; Haibo Jin; Wenbin Zhang; Haohan Wang; Jun Zhuang
>
> **备注:** Accepted for ACL'26 (Main). TL;DR: We propose a controllable multi-turn risk-concealed red-teaming framework, CoRT, that progressively conceals surface-level risk while exploiting regulatory-violating behaviors on a proposed new benchmark, FinRisk-Bench
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in finance, where unsafe behavior can lead to serious regulatory risks. However, most red-teaming research focuses on overtly harmful content and overlooks attacks that appear legitimate on the surface yet induce regulatory-violating responses. We address this gap by introducing a controllable black-box multi-turn risk-concealed red-teaming framework (CoRT) that progressively conceals surface-level risk while exploiting regulatory-violating behaviors. CoRT contains two key components: (i) a Risk Concealment Attacker (RCA) that generates multi-turn prompts via iterative refinement, and (ii) a Risk Concealment Controller (RCC) that predicts a turn-level Risk Concealment Score (RCS) to steer RCA's follow-up style. We also built a domain-specific benchmark, FinRisk-Bench, with 522 instructions spanning six financial risk categories. Experiments on nine widely used LLMs show that CoRT (RCA) achieves 93.19% average attack success rate (ASR), and CoRT (RCA+RCC) further improves the average ASR to 95.00%. Our code and FinRisk-Bench are available at this https URL.
>
---
