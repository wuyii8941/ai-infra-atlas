# 交付检查记录

本记录只针对网站与教学脚本，不是三个研究项目的GPU实验复现报告。

## 内容与构建

- 54篇Markdown正文，109道自测题，61条术语，83项参考来源。
- 构建时检查章节ID唯一性、先修链接、站内文章链接、来源ID和实验脚本路径。
- 所有正文编译为原生MathML；正文、CSS、JavaScript、脚本下载内容与全文搜索数据均嵌入index.html。
- 没有第三方运行时CDN、网络字体、远端图片或遥测脚本。
- 未打包用户PDF、简历、模型权重、语料或私有实验日志。

## 浏览器检查

使用Playwright和Chromium，桌面1440×1000、移动390×844：

- 54篇文章路由和标题匹配通过。
- 两种宽度下未发现页面级水平溢出。
- 全文搜索、先修链接、目录、明暗主题、折叠题目通过。
- 四种计算器检查：默认KV为56MiB；36层为72MiB；M128的示例算术强度约113.778 FLOPs/Byte；单rank Ring流量为0。
- 已读、收藏、笔记与JSON序列化逻辑通过。
- 页面加载时外部resource请求数为0；没有JavaScript页面异常。

测试环境的受管Chromium禁止file://及localhost导航，因此采用把完整HTML载入浏览器文档的方式测试渲染与交互。其about:blank不提供真实localStorage，存储逻辑使用同接口的内存模拟验证。没有把该模拟宣称为真实浏览器磁盘持久化复测。正常浏览器下使用localStorage；若浏览器不允许，网站会显示提示并允许导出备份。

## 实验脚本

01_layout.py、02_online_softmax.py、03_sharding.py、04_precision.py已在CPU/NumPy环境执行并通过断言。

05_cuda_benchmark.py、06_triton_add.py仅完成Python语法检查，GPU路径未执行；脚本会在缺少依赖或CUDA时明确SKIP。它们是教学代码，不属于原仓库实验数字的复现。

## 未进行的工作

没有部署到公网或创建远端GitHub仓库；没有重新运行Qwen/DFuzz/DeepOpFuzz的GPU研究实验；没有全面巡检所有外部链接的未来可用性。官方动态文档内容可能更新，应按自己的运行版本核对。
