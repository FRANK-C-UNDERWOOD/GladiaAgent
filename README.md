# GladiaAgent: Your Predictive Dialog Companion

## 1. 简介 (Introduction)

**GladiaAgent** (亦称为 Predictive Dialog Agent - PDA) 是一个先进的对话式人工智能框架，其核心设计理念源于认知科学中的**预测编码 (Predictive Coding)** 理论。本项目旨在构建一个不仅仅能流畅对话，更能理解、学习并动态适应对话上下文的智能体。它致力于模拟类人对话的某些方面，例如基于经验的预期、对新奇信息的敏感度以及持续的知识积累。

与传统的基于固定脚本或简单检索的聊天机器人不同，GladiaAgent 的目标是：
*   **深度理解与推理：** 通过整合预测编码机制和大型语言模型 (LLM) 的能力，实现更深层次的语义理解和初步的推理。
*   **动态知识演化：** 智能体在交互过程中动态构建和更新其核心知识库，优先记忆那些具有“信息增益”或“预测意外”的内容。
*   **上下文感知交互：** 能够有效利用对话历史、已形成的知识以及当前对话的细微差别，生成高度相关且连贯的响应。

本项目探索了将认知理论与现代人工智能技术（如深度学习、NLP和LLM）相结合的可能性，以期创造出更智能、更自适应的对话体验。

## 2. 核心特性 (Core Features)

GladiaAgent 具备一系列使其在对话智能体领域独树一帜的特性：

*   🧠 **基于预测编码的智能决策:**
    *   核心的预测编码模块 (`PredictiveCodingAgent`) 评估输入信息（如从对话中提取的三元组）的“新奇性”或“重要性”。
    *   系统根据预测误差（衡量输入与智能体内部模型的匹配程度）来决定是否将新知识整合到其长期记忆中，从而实现更高效的学习。

*   📚 **动态演化的核心知识库:**
    *   在对话过程中，智能体从用户输入中提取关键信息（知识三元组）。
    *   这些信息经过评估后，有选择地存储在其核心知识库 (`knowledge_base_vectors`) 中，形成一个不断增长和优化的知识图谱。
    *   该知识库为智能体提供了事实依据和上下文参考。

*   💬 **多轮对话连贯性与深度记忆:**
    *   通过 `DialogHistoryBuffer` 精心管理对话历史。
    *   结合动态知识库的检索，确保在长时间的多轮对话中保持上下文连贯性和话题焦点。

*   💡 **上下文感知的响应生成:**
    *   `PredictiveDialogAgent (PDA)` 模块在生成响应前，会整合当前用户输入、完整的对话历史、从核心知识库中检索到的相关记忆，以及一个概念上的“思维链”。
    *   这种丰富的上下文被构建成精确的提示，指导强大的 DeepSeek 大型语言模型生成高度相关、信息丰富且符合当前对话场景的回答。

*   🖥️ **双模式交互界面:**
    *   提供直观的**图形用户界面 (GUI)** (`gladia_gui.py`)，方便用户进行交互和观察智能体的行为。
    *   同时支持传统的**命令行界面 (CLI)**，适用于开发者调试或在无图形环境下运行。

*   🔗 **模块化与可扩展设计:**
    *   系统采用模块化架构，主要功能（如预测编码、对话管理、知识集成、LLM接口）被封装在独立的组件中。
    *   这种设计便于未来的功能扩展、模块替换或针对特定应用的定制。

*   🛠️ **集成 DeepSeek LLM:**
    *   利用先进的 DeepSeek 大型语言模型进行复杂自然语言理解（如从非结构化文本中提取结构化的知识三元组）和高质量自然语言生成（对话响应）。

*   💾 **记忆持久化:**
    *   智能体的核心知识库和部分模型状态（如 `seRNN`）可以被保存到磁盘，并在下次启动时加载，实现了学习成果的持久化。

## 3. 技术架构 (Technical Architecture)

GladiaAgent 的架构设计旨在将认知启发的功能模块与强大的深度学习模型相结合，形成一个协同工作的智能对话系统。

### 3.1 整体概念流程 (Conceptual Flow)

GladiaAgent 的核心数据流和模块交互可以概括如下：

```mermaid
graph TD
    A[User Input (GUI/CLI)] --> B(Text Embedding Module);
    B --> UserInputEmbedded[User Input Embedded];

    UserInputEmbedded --> PDA[PredictiveDialogAgent PDA];
    A --> TripleExtractPath{Parallel Path for Knowledge};
    TripleExtractPath --> D[Triple Extraction (LLM)];
    D --> F[Triple Text];
    F --> G(Text Embedding Module);
    G --> H[Triple Embeddings];
    H --> I[seRNN Module];
    I --> J[Enhanced Triple Embeddings];
    J --> K[PredictiveCodingAgent PC];
    K -- Prediction Loss > Threshold --> L[Store in Core Knowledge Base];
    K -- Prediction Loss <= Threshold --> M(Discard/Ignore Triple);

    PDA --> N(Query Embedding Generation);
    N --> O[Query Core Knowledge Base];
    L --> O;

    O -- Retrieved Knowledge --> PDA;
    P[Dialog History] --> PDA;

    PDA -- Constructed Prompt --> Q[DeepSeek LLM];
    Q -- LLM Response Stream --> R(Format & Display Response);
    R --> A;

    subgraph CoreCognitiveLoop [Core Cognitive Loop / Knowledge Storage]
        D; F; G; H; I; J; K; L; M;
    end

    subgraph DialogManagement [Dialog Management & Generation]
        PDA; N; O; P; Q; R;
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style Q fill:#lightgrey,stroke:#333,stroke-width:2px
    style L fill:#ccffcc,stroke:#333,stroke-width:2px
    style K fill:#lightblue,stroke:#333,stroke-width:2px
    style I fill:#lightblue,stroke:#333,stroke-width:2px
```

**图例与说明:**

*   **矩形框 (e.g., `User Input`)**: 表示数据、状态或外部实体。
*   **圆角矩形框 (e.g., `Text Embedding Module`)**: 表示处理模块或组件。
*   **菱形框 (e.g., `Input Text`)**: 表示数据分支点或中间数据形态。
*   **箭头 (`-->`)**: 表示数据流或控制流方向。
*   **Subgraph (e.g., `Core Cognitive Loop`)**: 表示一组功能上相关的模块。

**流程解读:**

1.  **用户输入 (`User Input`)** 进入系统，首先由 **文本嵌入模块 (`Text Embedding Module`)** 处理，生成初步的文本表示 (`UserInputEmbedded`)。
2.  该嵌入表示一方面直接送入 **PredictiveDialogAgent (PDA)** 用于即时对话处理；另一方面，原始用户输入会并行触发**知识提取路径 (`Parallel Path for Knowledge`)**：
    *   原始文本送入 **三元组提取模块 (`Triple Extraction (LLM)`)**，利用大型语言模型从文本中抽取出结构化的三元组知识 (`Triple Text`)。
3.  提取出的三元组文本再次经过 **文本嵌入模块 (`Text Embedding Module`)** 转换为向量形式 (`Triple Embeddings`)。
4.  这些三元组嵌入被送入 **seRNN 模块 (`seRNN Module`)** 进行深度处理和特征增强，得到 `Enhanced Triple Embeddings`。
5.  **核心认知循环 (`Core Cognitive Loop / Knowledge Storage`)**:
    *   增强后的三元组嵌入由 **PredictiveCodingAgent (PC)** 进行评估。PC Agent 计算其预测损失。
    *   如果预测损失**高于**预设阈值，表明该三元组具有新奇性或重要价值，其向量表示将被**存入核心知识库 (`Store in Core Knowledge Base`)**。
    *   如果预测损失**低于或等于**阈值，则该三元组被认为信息量不足或已存在相似知识，将被**忽略 (`Discard/Ignore Triple`)**。
6.  **对话管理与生成 (`Dialog Management & Generation`)**:
    *   **PredictiveDialogAgent (PDA)** 是对话管理的核心。它接收用户输入的嵌入表示，并生成用于查询的嵌入 (`Query Embedding Generation`)。
    *   利用此查询嵌入，PDA 从**核心知识库 (`Core Knowledge Base`)** 中检索相关历史知识。**对话历史 (`Dialog History`)** 也作为上下文输入到 PDA。
    *   PDA 整合所有这些信息（当前输入、检索到的知识、对话历史），构建一个全面的上下文**提示 (`Constructed Prompt`)**。
    *   该提示被发送给 **DeepSeek LLM**。
    *   LLM 生成的响应以**流式 (`LLM Response Stream`)** 返回，经过格式化后**展示给用户 (`Format & Display Response`)**，并反馈到新的用户输入循环中。


### 3.2 主要模块说明 (Key Modules)

以下是构成 GladiaAgent 核心功能的关键模块：

*   **`IntegratedSystem` (`integrated_system.py`)**
    *   **作用:** 作为系统的中央协调器，`IntegratedSystem` 负责初始化所有其他核心模块、适配器和配置。它定义并管理主要的处理流水线，控制数据如何在不同模块间流动。
    *   **功能:**
        *   加载和保存核心知识库及 `seRNN` 模型状态。
        *   提供知识库查询接口。
        *   调用 LLM 进行三元组提取。
        *   管理将新知识存入知识库的决策过程（基于 PC 模块的输出，通过 `store_triple_with_pc` 方法）。
        *   将用户对话请求路由到 `PredictiveDialogAgent`。

*   **`PredictiveDialogAgent (PDA)` (`PDA.py`)**
    *   **作用:** 对话管理的核心引擎，负责处理用户与智能体的直接交互。
    *   **功能:**
        *   维护对话历史 (`DialogHistoryBuffer`) 和概念上的“思维链”。
        *   为用户输入生成嵌入（通过调用 `IntegratedSystem` 的嵌入功能），并利用这些嵌入从核心知识库中检索最相关的上下文信息。
        *   智能构建包含当前输入、对话历史、检索到的知识以及其他上下文信号（如名义上的预测误差）的系统提示。在当前版本的 `PDA.py` 中，“预测误差” (`current_prediction_error`) 主要作为提示的一部分，其动态更新机制可能需要进一步开发或明确；它旨在概念上指导LLM当前对话的“意外程度”。
        *   通过异步客户端与配置的 DeepSeek 大型语言模型进行通信，以生成自然、连贯且上下文恰当的对话响应。

*   **`PredictiveCodingAgent (PC)` (`PredictiveCoding.py`)**
    *   **作用:** 实现预测编码理论的核心计算逻辑。该模块的主要职责是评估输入数据相对于智能体内部模型的新奇性或意外性。
    *   **功能:**
        *   包含一个编码器（将输入，如来自 `seRNN` 的384维嵌入，编码到较低维度）和多个连续时间神经元 (`ContinuousTimeNeuron`) 及解码器。
        *   通过其 `forward` 方法处理输入，计算预测损失 (prediction loss) 和重构损失 (reconstruction loss)。
        *   `IntegratedSystem` 利用此模块计算出的 `prediction_loss` 来判断一个新提取的三元组（在通过 `seRNN` 处理后）是否足够“令人惊讶”或“信息丰富”，从而决定是否将其存储到核心知识库中。

*   **`seRNN (Spatially-Embedded Recurrent Neural Network)` (`seRNN.py`)**
    *   **作用:** 一个增强型的循环神经网络，它在传统 RNN 的基础上加入了空间嵌入的概念和预测编码层，旨在更好地捕捉序列数据中的复杂依赖关系和结构信息。
    *   **功能:**
        *   处理输入的时间序列数据（例如，由三元组文本转换而来的嵌入向量序列）。
        *   其内部包含 `PredictiveCodingLayer` 用于计算预测误差，`SelectiveGatingMechanism` 以及 `SpatialEmbeddingLayer` 来处理空间位置信息。
        *   输出经过处理的序列表示，这些表示随后可能被 `PredictiveCodingAgent` 用作输入，以进行更高级的“新奇性”评估。

*   **文本嵌入 (Text Embedding)**
    *   **技术:** 主要使用 `sentence-transformers` 库中的预训练模型（例如 `paraphrase-MiniLM-L6-v2`）。
    *   **作用:** 将所有文本数据（包括用户的原始输入、从文本中提取的三元组的主谓宾部分等）转换为固定维度（本项目中为384维）的密集向量表示。这些嵌入是后续所有深度学习模块处理的基础。

*   **大语言模型 (LLM - DeepSeek Integration)**
    *   **技术:** 通过 `openai` Python 库与 DeepSeek API 进行交互。
    *   **作用:**
        1.  **知识提取:** 在 `IntegratedSystem` 中，LLM 被用于从用户提供的非结构化文本中精确提取结构化的知识三元组（主体-关系-客体）。
        2.  **对话生成:** 在 `PredictiveDialogAgent` 中，LLM 根据精心构建的上下文提示生成最终的用户回复，确保对话的自然流畅和信息丰富度。

### 3.3 辅助与实验性模块 (Supporting & Experimental Modules)

这些模块虽然存在于代码库中，但在当前核心运行流程中的作用可能不是主要的，或者代表了早期的设计思路及未来可能的扩展方向。

*   **`TN (Tensor Network)` (`TN.py`)**
    *   **设计用途:** 该模块 (`TripleCompressor`, `TensorNetworkLayer`) 旨在提供一种基于张量网络的方法来学习和压缩三元组的表示，可能直接从三元组的组成部分（实体、关系ID）生成嵌入。
    *   **当前状态:** 在 `IntegratedSystem` 的当前主要流程中，三元组的嵌入是通过将其文本化后由 `SentenceTransformer` 生成的，而非直接使用 `TN.py` 中的组件。因此，`TN.py` 更像是一个备选的或实验性的三元组表示学习模块。

*   **`Adapters` (`adapters.py`)**
    *   **`SeRNNAdapter`, `PredictiveAdapter`:** 这些类为它们各自包装的核心模块 (`seRNN`, `PredictiveCodingAgent`) 提供了一个简化的接口或特定的数据转换逻辑，使得模块间的集成更为平滑。
    *   **`PDAAdapter`:** 包含了一套更复杂的逻辑，将预测编码机制更紧密地整合到对话处理和三元组提取流程中。它代表了一个不同的集成策略，但在当前 `main.py` 驱动的核心流程中并未显式激活。

## 4. 核心机制详解 (Core Mechanisms Explained)

GladiaAgent 的智能行为主要通过以下两大核心机制实现：动态知识库构建和上下文感知的对话生成。

### 4.1 动态知识库构建 (Dynamic Knowledge Base Construction)

GladiaAgent 的一个关键特性是其能够根据对话交互动态地学习和积累知识。这个过程并非简单地存储所有信息，而是有选择性地将“有价值”的信息融入其核心知识库。这里的“价值”是通过预测编码机制来评估的。

**流程概述：**

1.  **三元组提取 (Triple Extraction):**
    *   当用户提供输入时，系统首先尝试从中提取结构化的知识，通常表现为（主语 Subject, 谓语 Predicate, 宾语 Object）这样的三元组。
    *   此任务由 `IntegratedSystem` 调用 DeepSeek LLM 完成，LLM 被指示从文本中识别并抽取出这些关系。

2.  **初步嵌入 (Initial Embedding):**
    *   提取出的每个三元组被转换成一个单一的文本字符串 (例如, "主体 谓语 宾语")。
    *   然后，`SentenceTransformer` (如 `paraphrase-MiniLM-L6-v2`) 将这个文本字符串编码为一个384维的密集向量嵌入。这个嵌入是三元组的初步语义表示。

3.  **时空特征增强 (`seRNN` Processing):**
    *   初步的三元组嵌入被传递给 `seRNN` 模块。
    *   `seRNN` (Spatially-Embedded Recurrent Neural Network) 对这些嵌入进行进一步处理，旨在捕捉更深层次的语义特征以及（概念上的）时空关系，输出一个“增强的”或“情境化的”嵌入表示。

4.  **新奇性评估 (`PredictiveCodingAgent`):**
    *   `seRNN` 处理后的增强嵌入被送入 `PredictiveCodingAgent` (PC Agent)。
    *   PC Agent 的核心功能是将其内部的预测与当前输入（即增强嵌入）进行比较。这个比较的结果量化为一个**预测损失 (prediction loss)**。
    *   高预测损失意味着输入对于智能体当前的内部模型来说是“意外的”、“新奇的”或“信息丰富的”。低预测损失则表示输入与智能体的预期相符，可能已知或冗余。

5.  **选择性存储 (Selective Storage):**
    *   `IntegratedSystem` 模块根据 PC Agent 计算出的 `prediction_loss` 作出决策。
    *   在 `config.py` 中定义了一个**存储阈值** (`storage_threshold`，例如 0.65)。
    *   如果一个三元组经过上述流程后，其对应的 `prediction_loss` **大于** 此阈值，系统就认为这个三元组携带了足够的新信息，值得被记住。
    *   此时，该三元组的向量表示 (通常是 `seRNN` 的输出向量) 连同其文本键 (一个唯一标识三元组的字符串) 一起被存储到核心知识库 (`knowledge_base_vectors`) 中。
    *   如果预测损失低于阈值，则该三元组被认为不够新奇，不会被添加到核心知识库中，从而避免了知识库的冗余膨胀。

通过这个机制，GladiaAgent 能够优先学习那些挑战其现有知识或填补其知识空白的信息，模拟了生物学习中对意外刺激的关注。

### 4.2 对话生成流程 (Dialog Generation Flow)

GladiaAgent 的对话生成能力依赖于 `PredictiveDialogAgent (PDA)` 模块与大型语言模型 (DeepSeek LLM) 的紧密协作，并充分利用了系统积累的知识和对话历史。

**流程概述：**

1.  **接收用户输入:**
    *   `PDA` 模块接收到来自用户通过 GUI 或 CLI 发送的最新消息。

2.  **构建即时上下文:**
    *   **对话历史管理:** `PDA` 内部的 `DialogHistoryBuffer` 记录了最近的多轮对话内容（用户和智能体的发言）。
    *   **思维链记录 (Conceptual):** `DialogHistoryBuffer` 还维护一个“思维链”，记录了智能体在生成响应过程中的一些中间思考步骤或决策点（这更多是一个概念性的辅助，用于构建更全面的提示）。

3.  **核心知识库检索:**
    *   用户的当前输入被 `SentenceTransformer` 转换为嵌入向量（通过 `IntegratedSystem`）。
    *   `PDA` 使用这个查询向量，通过 `IntegratedSystem` 的 `query_core_knowledge_base` 方法，在核心知识库中检索最相似（即最相关）的已存储知识三元组。
    *   检索到的三元组（通常是 Top-K 结果）及其相似度得分，将作为重要的背景知识。

4.  **构建 LLM 系统提示 (System Prompt Construction):**
    *   这是对话生成的关键步骤。`PDA` 精心构建一个全面的系统提示，该提示会提供给 DeepSeek LLM。这个提示通常包含以下部分：
        *   **角色设定:** 定义智能体（例如，歌蕾蒂娅，实验室科研助手）的身份、语气和专业领域。
        *   **完整的对话历史:** 前几轮的用户提问和智能体回答。
        *   **思维链记录:** 智能体之前的思考路径摘要。
        *   **关联记忆:** 从核心知识库中检索到的与当前输入最相关的知识三元组，通常会格式化为易于LLM理解的形式。
        *   **认知状态 (Conceptual):** 包含一个名义上的“预测误差”值 (`current_prediction_error`)。在当前版本的 `PDA.py` 中，此值似乎主要作为提示的一部分，其动态更新机制可能需要进一步开发或明确；它旨在概念上指导LLM当前对话的“意外程度”。
        *   **响应要求:** 对LLM生成的回答提出具体要求，如回答的长度、专业性、特定结尾等。

5.  **与 DeepSeek LLM 交互:**
    *   构建好的系统提示和用户的最新输入一起被发送到 DeepSeek API。
    *   `PDA` 使用异步HTTP客户端 (`AsyncOpenAI`) 与 LLM 进行通信，以支持流式响应。

6.  **流式响应与呈现:**
    *   LLM 开始生成响应。响应内容以数据流的形式逐块返回。
    *   `PDA` 接收这些数据块，并实时地将它们传递给用户界面（GUI 或 CLI），用户可以看到文字逐字或逐句出现，提升了交互的即时感。
    *   完整的响应也被记录到对话历史中，为下一轮对话做准备。

通过这种方式，GladiaAgent 不仅仅是简单地将用户输入传递给LLM，而是通过动态检索和整合内部知识、维护对话状态，为LLM提供了丰富的、高度情境化的输入，从而引导LLM生成更准确、更深入、更个性化的回答。

## 5. 本地部署与运行 (Local Deployment and Execution)

本节将指导您如何在本地环境部署并运行 GladiaAgent。

### 5.1 获取项目代码
首先，您需要克隆项目地址。
```bash
git clone [项目地址]
cd [文件名]
```

### 5.2 系统需求

在开始安装之前，请确保您的系统满足以下基本要求：

*   **操作系统兼容性:**
    *   Windows 10/11
    *   理论上应兼容其他主流现代桌面操作系统

### 5.3 环境设置与安装 (Environment Setup & Installation)

#### 5.3.1 Python 版本
本项目建议使用 **Python 3.11 或 Python 3.12**。您可以在终端通过 `python --version` 或 `python3 --version` 命令检查您的 Python 版本。

#### 5.3.2 创建虚拟环境
为了保持项目依赖的隔离并避免与系统或其他项目的包产生冲突，建议您在 Python 虚拟环境中安装 GladiaAgent。

*   **使用 `venv` (Python 内置):**
    ```bash
    # 在项目根目录下创建虚拟环境 (例如，名为 venv_gladia)
    python -m venv venv_gladia

    # 激活虚拟环境
    # Windows (CMD/PowerShell):
    .\venv_gladia\Scripts\activate
    # macOS/Linux (bash/zsh):
    source venv_gladia/bin/activate
    ```

#### 5.3.3 安装依赖 (Dependencies)
        
        pip install -r requirements.txt
        
如果遇到冲突，您可能需要手动编辑 `requirements.txt`，选择合适的版本，或者逐个解决。
等待系统安装完成后run main.py文件。

## 6. 项目文件结构 (Project File Structure)

以下是项目主要文件和目录的简要说明：

*   `main.py`: 项目的入口点。处理命令行参数，初始化并启动集成系统，以及选择运行 GUI 或 CLI 模式。
*   `config.py`: 包含所有模块和系统行为的全局配置参数，如模型维度、学习率、API设置等。
*   `integrated_system.py`: `IntegratedSystem` 类的实现，是整个系统的核心协调器，管理模块间的交互、数据处理流水线和知识库。
*   `PDA.py`: `PredictiveDialogAgent` 类的实现，负责对话管理、上下文构建和与 LLM 的交互。
*   `PredictiveCoding.py`: `PredictiveCodingAgent` 和 `ContinuousTimeNeuron` 类的实现，构成了预测编码机制的核心。
*   `seRNN.py`: `SeRNN` 模块的实现，用于处理序列数据的时空嵌入。
*   `TN.py`: 包含 `TripleCompressor` 和 `TensorNetworkLayer`，用于实验性的三元组嵌入和压缩，当前未在主流程中激活。
*   `adapters.py`: 包含各个模块的适配器类，用于促进模块间的接口兼容和数据转换。
*   `gladia_gui.py`: 图形用户界面 (GUI) 的实现，可能基于 PyQt5。
*   `requirements.txt`: 原始的项目依赖列表。
*   `requirements_cleaned.txt`: 一个经过清理和版本协调的依赖列表，建议用于安装。
*   `core_sernn_memory/`: (程序运行时自动创建) 用于持久化存储核心知识库 (`knowledge_base.pt`) 和 `seRNN` 模型状态 (`sernn_model_state.pth`) 的目录。
*   `README.md`: 本文件，提供项目概览、设置指南和使用说明。

## 7. 未来工作 (Future Work) - 简述

*   **动态预测误差更新:** 完善 `PDA.py` 中 `current_prediction_error` 的动态计算和利用机制，使其能更真实地反映对话的“意外程度”并指导LLM。。
*   **PDAAdapter集成:** 进一步开发和测试 `PDAAdapter.py` 中的高级集成逻辑，并考虑将其作为可选的对话处理流程。
*   **更细致的记忆管理:** 引入更复杂的记忆衰减、遗忘和泛化机制。
*   **用户评估与反馈:** 构建用户评估框架，收集反馈以持续改进。

## 8. 贡献 (Contributing)
如果您对改进 GladiaAgent 感兴趣，欢迎通过 GitHub Issues 提交问题或建议。对于代码贡献，请先发起讨论或提出 Issue。

