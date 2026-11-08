# AI Infra Atlas

一个围绕 Qwen3-Triton、DFuzz、DeepOpFuzz 组织的中文 AI Infra 教程网站。

**54 篇教程 · 9 个模块 · 109 道自测题 · 61 个术语入口 · 4 个理论计算器 · 6 个实验脚本。**

## 打开网站

解压后直接双击 `index.html`。正文、公式、搜索、计算器与自测都不依赖网络，不需要 npm 或启动开发服务器。

建议使用较新版本的 Chrome、Edge、Firefox 或 Safari。数学公式使用浏览器原生 MathML，不使用远程 CDN 和额外字体文件。手机文件管理器的内置预览可能不执行 JavaScript，请改用浏览器或静态托管链接。

也可在该文件夹运行：

```bash
python -m http.server 8000 --bind 127.0.0.1
```

然后打开 `http://127.0.0.1:8000`。

## 目录与功能

- 左侧章节目录、章首先修链接、章末后续用途、右侧页内目录。
- 全文搜索，快捷键 `Ctrl+K` / `Command+K` 或 `/`。
- 全局知识地图、五条分阶段路线、模块与重点章节流程图。
- 项目档案、术语索引与带出处的章节阅读。
- GEMM / KV Cache / FSDP 状态 / Ring All-Reduce 理论计算器。
- 折叠自测答案、收藏、已读标记、本地笔记、JSON 备份导入/导出。
- 浅色/深色、移动端目录、打印样式、单章 Markdown 导出。
- 4 个 CPU / NumPy 实验、2 个需要自行配置 GPU 环境的示例。

## 修改内容和重新构建

```
content/*.md              # 人工可编辑的正文
content/catalog.json      # 目录、前置章节、来源与题目
content/sources.json      # 来源清单与证据边界
content/structure.json    # 章节组与模块说明
content/roadmap.json      # 分阶段路线、模块流程图和章节流程图
content/glossary.json     # 术语与正文对应关系
assets/site.css           # 响应式样式
assets/app.js             # 不依赖框架的交互代码
assets/template.html      # HTML 模板
labs/*.py                 # 教学练习
build.py                  # 构建脚本
index.html                # 已构建、可离线打开的完整站点
```

修改后使用 **Python 3.10+ 与 Pandoc** 构建：

```bash
python build.py
```

不需要 Python 第三方构建库。实验 01–04 依赖 NumPy，与网站构建分开。增加文章时更新 `catalog.json`，并提供同名 Markdown；所有前置章节、来源和站内链接会在构建时检查。

## 部署到 GitHub Pages

本交付没有创建远端仓库、修改你的 GitHub 项目或启用公网服务。

一种不需要云端构建的部署方式：

1. 新建你允许公开的仓库，将本目录的文件上传并提交。
2. 在仓库 Settings → Pages 选择从分支发布，选相应分支与根目录 `/`。
3. 使用 GitHub 返回的 Pages 地址。站点使用相对资源和 hash 路由，不依赖域名根路径。

也可以只发布 `index.html` 与 `.nojekyll`；源码、教程Markdown与练习已嵌入HTML，网页功能仍可使用。原始练习脚本可由网页按钮下载。

GitHub 的可用发布方式和权限可能随账户/组织设置而变化；请以仓库当前设置为准。

## 隐私、研究稿件与使用边界

- 不打包上传的 PDF、简历、模型权重、Token 数据、私有日志或大 checkpoint。
- 网站文字包含你提供的论文方法、结果和项目报告信息。**公开前请确认研究内容允许发布**；匿名评审稿不应在未经确认的情况下与个人身份关联公开。
- 本教程不是项目的完整可复现 artifact。Qwen 的原始 `artifacts/` 多为未公开本地文件；链接缺失不表示本站独立验证过它们。
- 性能数字按报告口径记录，本站没有重新运行 GPU benchmark。论文结果不等于当前 upstream 状态。
- 理论章节与扩展知识不自动构成你的项目贡献。正文中的逻辑依赖顺序不冒充实际研发时间线。
- DFuzz 的 0.01 是其训练 loss 轨迹阈值，不是通用算子容差；DeepOpFuzz 的 30 根因回放与独立历史效率实验不混用。
- 外部来源链接需要联网；本站不包含第三方文章全文或第三方字体文件。
- 阅读进度和笔记只保存在本浏览器 `localStorage`，不上传。不同地址和浏览器可能分别存储；定期导出 JSON 备份。

## 版本

内容基于 Qwen3-Triton `10be3d87`、DeepOpFuzz artifact `c69ead1b`、用户提供的 DFuzz / DeepOpFuzz 论文稿件，以及章节列出的官方资料。

网页交付版本：1.0。细节可继续随项目版本更新，但更新实验数字时必须同步版本、baseline、shape、dtype、计时边界和验证范围。
