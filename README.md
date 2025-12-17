# 多模态机器学习实验2 - 本地AI智能文献与图像管理助手

## 1. 项目简介与核心功能
本项目实现一个本地论文库的管理与多模态检索工具，统一通过 `main.py` 提供命令行入口，支持：

- 对单篇 / 批量 PDF 论文附加主题标签（topics），并根据主题将论文加入本地论文库中
- 论文语义搜索，实现 **Text → Paper** 的语义检索
- 以文搜图，实现 **Text → Image** 的跨模态检索

适用场景：本地论文管理、按主题归档、快速定位“相关论文/图片”。

## 2. 项目结构
```text
.
├── main.py                     # 主文件
├── papers/                     # 本地论文库（PDF）
├── images/                     # 本地图片库
├── embeddings/                 # 向量库
│   ├── papers_chroma/          # 论文文本向量库（ChromaDB）
│   └── images_chroma/          # 图片向量库（ChromaDB）
├── README.md
└── environment.yml             # Conda 环境配置
```

## 3. 环境配置与依赖安装
Python 版本推荐：**Python 3.8 +**

在项目根目录执行：
```bash
conda env create -f environment.yml
```

## 4. 使用说明
### 4.1 将单篇文献加入本地论文库
```bash
python main.py add_paper <path> --topics "Topic1,Topic2"
```
说明：
- `<path>`：PDF 文件路径；
- `--topics`：以逗号分隔的主题标签字符串。

### 4.2 将批量文献加入本地论文库
```bash
python main.py organize_papers --root-dir <path> --topics "Topic1,Topic2"
```
说明：
- `<path>`：需要批量处理的 PDF 文件夹。

### 4.3 在本地论文库中搜索文献
```bash
python main.py search_paper <query>
```
说明：
- `<query>`：自然语言提问，如`"What is diffusion model"`。

### 4.4 在本地图片库中搜索图片
```bash
python main.py search_image <query>
```
说明：
- `<query>`：自然语言提问，如`"sunset over the sea"`。

## 5. 技术选型说明
- **PDF 解析与图片抽取：PyMuPDF（fitz）**
  - 用途：从 PDF 中提取文本内容
  - 优点：速度快、对 PDF 支持成熟

- **文本向量：Sentence-Transformers**
  - 用途：对论文文本（或摘要/分段文本）生成语义向量，用于 **Text → Paper** 检索
  - 模型：`sentence-transformers/all-MiniLM-L6-v2`（轻量、效果稳定，适合本地运行）

- **跨模态向量：CLIP**
  - 用途：将文本与图片编码到同一嵌入空间，用于 **Text → Image** 检索
  - 模型：`openai/clip-vit-base-patch32`

- **向量数据库：ChromaDB**
  - 用途：存储向量与元数据，支持 Top-K 相似度查询
  - 设计：论文文本与论文图片分别存入不同 collection，便于独立检索与增量更新

## 6. 演示




