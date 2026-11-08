/* AI Infra Atlas: all content is embedded, no network requests or third-party runtime. */
'use strict';
(() => {
const DB = JSON.parse(document.getElementById('atlas-data').textContent);
const articles = DB.articles, byId = Object.fromEntries(articles.map(a => [a.id,a]));
const sections = DB.sections, sec = Object.fromEntries(sections.map(s=>[s.id,s]));
const {paths, sections: moduleFlows, chapters: chapterFlows} = DB.roadmap;
const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const escape = v => String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const icons = {
 search:'<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
 arrow:'<path d="M4 12h15m-6-6 6 6-6 6"/>', book:'<path d="M12 5c-3-2-7-2-10 0v15c3-2 7-2 10 0 3-2 7-2 10 0V5c-3-2-7-2-10 0Zm0 0v15"/>',
 menu:'<path d="M4 6h16M4 12h16M4 18h16"/>',moon:'<path d="M20 15A9 9 0 0 1 9 4a9 9 0 1 0 11 11Z"/>',sun:'<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>',
 check:'<path d="m5 12 4 4L20 5"/>',bookmark:'<path d="M6 3h12v19l-6-4-6 4Z"/>',copy:'<rect x="8" y="8" width="12" height="13" rx="2"/><path d="M15 8V3H3v13h5"/>',
 down:'<path d="M12 3v12m-5-5 5 5 5-5M4 17v4h16v-4"/>',map:'<path d="m3 5 6-2 6 2 6-2v16l-6 2-6-2-6 2V5Zm6-2v16m6-14v16"/>',close:'<path d="m5 5 14 14M19 5 5 19"/>',external:'<path d="M14 3h7v7m0-7-11 11M10 4H4v16h16v-6"/>',grid:'<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',print:'<path d="M6 8V3h12v5M6 17H3V8h18v9h-3M6 14h12v8H6Z"/>'
};
function icon(name){return `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${icons[name]||icons.book}</svg>`}
let state={read:[],saved:[],notes:{},theme:'light',last:'orientation'},storageAvailable=true;
try{state={...state,...JSON.parse(localStorage.getItem('infra-atlas-v1')||'{}')}}catch(e){storageAvailable=false}
if(!Array.isArray(state.read))state.read=[];if(!Array.isArray(state.saved))state.saved=[];if(!state.notes||typeof state.notes!=='object')state.notes={};
state.read=state.read.filter(x=>byId[x]);state.saved=state.saved.filter(x=>byId[x]);
function persist(){try{localStorage.setItem('infra-atlas-v1',JSON.stringify(state))}catch(e){storageAvailable=false}}
function toast(message){$('#toast').textContent=message;$('#toast').classList.add('show');clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('#toast').classList.remove('show'),2600)}
function setTheme(theme){state.theme=theme;document.documentElement.dataset.theme=theme;$('#theme-toggle').innerHTML=icon(theme==='dark'?'sun':'moon');$('#theme-toggle').setAttribute('aria-label',theme==='dark'?'切换浅色主题':'切换深色主题');persist()}
function articleLink(id,label){const a=byId[id];return a?`<a href="#article/${id}">${escape(label||a.title)}</a>`:''}
function flowDiagram({title, steps}) {
  return `<figure class="atlas-flow"><figcaption>${escape(title)}</figcaption><ol>${steps.map(({label,detail,article},i)=>`
    <li><span class="step-number">${String(i+1).padStart(2,'0')}</span><strong>${article ? articleLink(article,label) : escape(label)}</strong><p>${escape(detail)}</p></li>`).join('')}</ol></figure>`;
}
function systemMap() {
  const trunk = [
    ['tensor-layout','张量与计算','用形状、步长和类型描述数据，从点积和字节数建立成本账本。'],
    ['transformer','模型前向','Embedding 产生隐藏状态，Decoder Block 逐层变换，LM Head 输出 logits。']
  ];
  const branches = [
    ['#map/kernel','推理执行','已有 logits 产生首个 Token。后续 Decode 读取并更新 KV，再生成下一个 Token。'],
    ['#map/training','训练更新','logits 与标签得到 Loss。沿反向图求梯度，再更新权重与优化器状态。'],
    ['#map/compiler','编译实现','从前向或训练程序提取计算图，再调度算子、生成或调用 Kernel。']
  ];
  return `<section class="system-map" aria-label="模型执行与验证总图">
    <div class="map-trunk">${trunk.map(([id,title,detail])=>`<div class="map-node"><strong>${articleLink(id,title)}</strong><p>${detail}</p></div>`).join('')}</div>
    <div class="map-connector">模型前向的两种用途与执行实现</div>
    <div class="map-branches">${branches.map(([href,title,detail])=>`<a class="map-node" href="${href}"><strong>${title}</strong><p>${detail}</p></a>`).join('')}</div>
    <div class="map-connector">为各条路径建立可比较的基线</div>
    <div class="map-node map-evidence"><strong>验证性能与正确性</strong><p>${articleLink('profiling','测量时间与数据搬运')} · ${articleLink('oracles','比较数值和执行语义')} · <a href="#projects">核对项目证据</a></p></div>
  </section>`;
}
function routeCards() {
  return paths.map(p=>`<a class="path-card" href="#map/${p.id}"><div class="card-kicker">${escape(p.label)}</div><h3>${escape(p.name)}</h3><p>${escape(p.desc)}</p><div class="card-footer"><span>${p.stages.length} 个阶段</span>${icon('arrow')}</div></a>`).join('');
}
function learningPath(path) {
  const items = path.stages.flatMap(stage=>stage.items);
  const included = new Set(items);
  return `<section class="path-group" id="path-${path.id}"><div class="card-kicker">${escape(path.label)}</div><h2>${escape(path.name)}</h2><p>${escape(path.desc)}</p>
    <ol class="route-stages">${path.stages.map(({title,transition,items:stageItems},i)=>{
      const extra = [...new Set(stageItems.flatMap(id=>byId[id].prereq))].filter(id=>!included.has(id));
      return `<li><h3><span class="step-number">${String(i+1).padStart(2,'0')}</span>${escape(title)}</h3><p>${escape(transition)}</p>
        ${extra.length?`<details class="route-prereqs"><summary>本阶段补读 ${extra.length} 篇</summary><ul>${extra.map(id=>`<li>${articleLink(id)}</li>`).join('')}</ul></details>`:''}
        <ol class="route-chapters">${stageItems.map(id=>`<li class="${state.read.includes(id)?'done':''}">${articleLink(id)}${state.read.includes(id)?'<span class="small">已读</span>':''}</li>`).join('')}</ol></li>`;
    }).join('')}</ol></section>`;
}
function chapterConnections(a) {
  const next = articles.filter(other=>other.prereq.includes(a.id));
  return `<section class="chapter-connections"><h2>使用本章知识的后续章节</h2>
    ${next.length?`<ul>${next.map(other=>`<li>${articleLink(other.id)}</li>`).join('')}</ul>`:''}
    <p><a href="#map/section-${a.section}">查看本模块的流程与目录 →</a></p></section>`;
}
function footer(){return `<footer class="footer"><span>AI INFRA ATLAS · 项目驱动的系统学习</span><span>离线正文 · 本地进度 · 固定版本证据 <a href="#sources">资料与边界 ↗</a></span></footer>`}
function breadcrumb(text){return `<div class="breadcrumbs"><a href="#home">Atlas</a><span>/</span><span>${escape(text)}</span></div>`}
function routeInfo(){const [type='home',id='',heading='']=location.hash.slice(1).split('/');return {type:type||'home',id,heading}}
let activeArticle=null,observer=null;
function sidebar(){
const info=routeInfo();
$('#sidebar').innerHTML=`<div class="sidebar-head"><span>学习目录</span><span>${articles.length} 篇</span></div><a href="#home" class="side-home">${icon('grid')}知识库首页</a>`+
sections.map(s=>{const list=articles.filter(a=>a.section===s.id);const open=(byId[info.id]?.section===s.id)||['start','foundations'].includes(s.id);return `<details class="side-group" ${open?'open':''}><summary><span class="sec-number">${s.number}</span>${escape(s.title)}</summary><div class="side-links">${list.map(a=>`<a class="side-link ${info.type==='article'&&info.id===a.id?'active':''}" href="#article/${a.id}" ${info.id===a.id?'aria-current="page"':''}><span>${escape(a.title.split('：')[0])}</span>${state.read.includes(a.id)?'<span class="read-dot" title="已读">✓</span>':''}</a>`).join('')}</div></details>`}).join('')+
`<div class="sidebar-foot"><div class="sidebar-utilities"><a href="#map">学习地图</a><a href="#projects">项目档案</a><a href="#lab">计算实验室</a><a href="#glossary">术语索引</a><a href="#sources">资料与边界</a><a href="#notebook">阅读记录与笔记</a></div><div class="progress-track"><div class="progress-fill" style="width:${state.read.length/articles.length*100}%"></div></div><span>${state.read.length} / ${articles.length} 已读</span><p>计算原理 → 源码实现 → 实验证据</p><span>版本 ${DB.version} · 中文教程</span></div>`;
}
function currentNav(type){$$('.topnav a').forEach(a=>a.classList.toggle('active',a.dataset.route===type));}
function home(){
const continueId=byId[state.last]?state.last:'orientation';
return `<div class="page home-page"><section class="hero"><div><div class="eyebrow">A practical atlas of AI systems</div><h1>从一次模型执行，<br>建立 AI Infra <em>知识地图</em></h1><p class="hero-desc">先沿张量、前向计算和输出建立共同基础，再沿推理、训练与编译三条线追踪数据和状态。</p><div class="hero-actions"><a class="btn btn-primary" href="#article/${continueId}">${state.read.length?'继续学习':'开始学习'} ${icon('arrow')}</a><a class="btn" href="#map">${icon('map')}查看学习地图</a></div><div class="method-strip"><span>逐步推导</span><span>真实源码</span><span>证据边界</span></div></div></section>
<div class="stats"><div class="stat"><strong>${articles.length}</strong><span>篇连续教程</span></div><div class="stat"><strong>${DB.quizCount}</strong><span>道自测与解析</span></div><div class="stat"><strong>${DB.glossary.length}</strong><span>个术语入口</span></div><div class="stat"><strong>06</strong><span>份动手实验</span></div></div>
<div class="section-head"><h2>模型执行与验证总图</h2><a href="#map">查看完整学习地图 →</a></div>${systemMap()}
<div class="section-head"><h2>分阶段阅读路线</h2></div><div class="path-cards">${routeCards()}</div>
<div class="section-head"><h2>逐层建立知识体系</h2><a href="#article/gemm-flops">试读：矩阵乘法的计算量推导 →</a></div><div class="curriculum-grid">${sections.filter(s=>s.id!=='start').map(s=>`<a class="curriculum-card" href="#map/section-${s.id}"><span class="curriculum-num">${s.number}</span><div><h3>${s.title}</h3><p>${s.summary}</p></div><span class="count">${articles.filter(a=>a.section===s.id).length} 篇</span></a>`).join('')}</div>
<div class="section-head"><h2>由真实项目连接起来</h2><a href="#projects">进入项目档案 →</a></div><div class="path-cards">${[['project-qwen','Qwen3-Triton','推理、训练、全分片：分别说明做法、指标与适用范围。'],['project-dfuzz','DFuzz / Megatron','44 项训练优化的数值评估，两阶段搜索与根因实验。'],['project-deepop','DeepOpFuzz','从 operation target 到 artifact，理解两类真实 silent bugs。']].map(([id,t,p])=>`<a class="path-card" href="#article/${id}"><div class="card-kicker">PROJECT DOSSIER</div><h3>${t}</h3><p>${p}</p><div class="card-footer"><span>证据 · 代码 · 局限</span>${icon('arrow')}</div></a>`).join('')}</div>
<div class="note-box"><strong>数字有出处，结论有边界。</strong> 教学推导不冒充实测；论文结果按原实验口径呈现。Qwen 源码固定到 <code>10be3d87</code>，本站没有重新运行项目 GPU benchmark，也未打包论文、简历、权重或未公开日志。</div>${footer()}</div>`;
}
function sourceCard(id){const s=DB.sources[id];if(!s)return '';return `<div class="resource" id="source-${id}"><span>${escape(s.kind)}</span>${s.url?`<a href="${escape(s.url)}" target="_blank" rel="noopener noreferrer">${escape(s.title)} ↗</a>`:`<strong>${escape(s.title)}</strong>`}<p>${escape(s.note)}</p></div>`}
function article(id){const a=byId[id];if(!a)return notFound();activeArticle=a;state.last=id;persist();
const index=articles.indexOf(a),prev=articles[index-1],next=articles[index+1];
return `<div class="page"><div class="article-layout"><article class="article-main">${breadcrumb(sec[a.section].title)}<header class="article-header"><div class="eyebrow">CHAPTER ${String(index+1).padStart(2,'0')} / ${sec[a.section].number}</div><h1>${escape(a.title)}</h1><p class="summary">${escape(a.summary)}</p><div class="meta-row"><span class="badge badge-kind">${escape(a.kind)}</span><span class="badge">${['','入门','进阶','深入'][a.level]||'进阶'}</span><span class="badge">${a.charCount.toLocaleString()} 字符</span><span class="badge">${a.checks.length} 道自测</span>${a.tags.slice(0,4).map(t=>`<span class="badge">${escape(t)}</span>`).join('')}</div></header>${a.prereq.length?`<div class="prereq-box"><span>先修知识</span>${a.prereq.map(p=>articleLink(p,byId[p]?.title.split('：')[0])).join(' · ')}</div>`:'<div class="prereq-box"><span>从这里开始</span>从模型的输入和输出开始。</div>'}
<div class="article-actions"><button class="btn btn-small" id="read-toggle">${icon(state.read.includes(id)?'check':'book')}${state.read.includes(id)?'已标记读完':'标记读完'}</button><button class="btn btn-small" id="save-toggle">${icon('bookmark')}${state.saved.includes(id)?'已收藏':'收藏章节'}</button><button class="btn btn-small" id="export-md">${icon('down')}Markdown</button><button class="btn btn-small" id="print-article">${icon('print')}打印</button></div>${chapterFlows[id]?flowDiagram(chapterFlows[id]):''}<div class="article-body">${a.html}</div>
${a.lab?`<section class="callout-lab"><strong>动手实验 · ${escape(a.lab.split('/').pop())}</strong><p>${a.lab.match(/0[56]/)?'需要自己的 CUDA / PyTorch / Triton 环境。本站未运行 GPU 实验。':'CPU / NumPy 教学实验，已在交付环境执行检查。它验证数学与索引，不复现 GPU 性能。'}</p><button class="btn btn-small" data-download-lab="${a.lab}">${icon('down')}下载实验脚本</button> <a href="#lab">打开理论计算实验室 →</a></section>`:''}
${chapterConnections(a)}
<section id="self-check"><h2>理解检查</h2><p class="small">先独立推导或口述，再展开解析。尝试写出前提、步骤与结论。</p>${a.checks.map((q,i)=>`<details class="quiz"><summary><span class="quiz-num">Q${String(i+1).padStart(2,'0')}</span>${escape(q.q)}</summary><div class="quiz-answer">${escape(q.a)}</div></details>`).join('')}</section>
<section id="article-sources"><h2>来源与阅读边界</h2><p class="small">项目结论以对应报告为限；通用理论来自原始论文与官方文档。动态文档不替代实验锁定版本。</p><div class="resource-list">${a.sources.map(sourceCard).join('')}</div></section>
<section class="notes-panel" id="article-notes"><h2>自己的推导与笔记</h2><p class="small">只保存在当前浏览器。请用“阅读记录”中的导出功能备份；清理浏览器数据会删除它们。</p><textarea class="notes-area" id="note-input" aria-label="章节笔记" placeholder="写下仍然不理解的一个环节，或用自己的话解释本章…">${escape(state.notes[id]||'')}</textarea><small id="note-status">${storageAvailable?'本地自动保存':'当前浏览器存储不可用，可导出阅读记录备份'}</small></section>
<div class="chapter-nav">${prev?`<a href="#article/${prev.id}"><small>← 目录上一章</small>${escape(prev.title)}</a>`:'<a href="#map"><small>回到路线</small>查看完整学习地图</a>'}${next?`<a href="#article/${next.id}"><small>目录下一章 →</small>${escape(next.title)}</a>`:'<a href="#map"><small>完成这一轮</small>回到知识地图，自查薄弱环节</a>'}</div>${footer()}</article><aside class="toc" aria-label="本页目录"><div class="toc-title">ON THIS PAGE</div><nav>${a.toc.map(h=>`<a href="#article/${id}/${h.id}" data-toc="${h.id}" class="${h.level===3?'sub':''}">${escape(h.title)}</a>`).join('')}<a href="#article/${id}/self-check" data-toc="self-check">理解检查</a><a href="#article/${id}/article-sources" data-toc="article-sources">来源与边界</a></nav><div class="toc-footer"><a href="#notebook">阅读记录与笔记 →</a></div></aside></div></div>`;
}
function mapPage(){
  return `<div class="page">${breadcrumb('学习地图')}<div class="eyebrow">KNOWLEDGE MAP</div>
  <h1>AI Infra 学习地图</h1><p class="page-intro">从张量和模型前向建立共同基础，再跟踪推理、训练与编译中的数据流和状态变化。每条路线给出阶段目标、阅读顺序与补读章节。</p>
  ${systemMap()}
  <div class="section-head"><h2>按目标选择阅读路线</h2></div><div class="path-cards">${routeCards()}</div>
  <nav class="path-items module-jumps" aria-label="直接查看模块流程">${sections.filter(s=>moduleFlows[s.id]).map(s=>`<a class="path-item" href="#map/section-${s.id}"><span class="order">${s.number}</span>${escape(s.title)}</a>`).join('')}</nav>
  ${paths.map(learningPath).join('')}
  <div class="section-head"><h2>模块流程与完整目录</h2><span class="small">${articles.length} 篇 · ${sections.length} 个模块</span></div>
  ${sections.map(s=>`<section class="topic-section" id="section-${s.id}"><div class="eyebrow">PART ${s.number}</div><h2>${escape(s.title)}</h2><p class="section-caption">${escape(s.summary)}</p>
  ${moduleFlows[s.id]?flowDiagram(moduleFlows[s.id]):''}
  <div class="chapter-index">${articles.filter(a=>a.section===s.id).map(a=>`<a class="chapter-item" href="#article/${a.id}"><strong>${state.read.includes(a.id)?'✓ ':''}${escape(a.title)}</strong><p>${escape(a.summary)}</p></a>`).join('')}</div></section>`).join('')}${footer()}</div>`;
}
function projectsPage(){return `<div class="page">${breadcrumb('项目档案')}<div class="eyebrow">PROJECT DOSSIERS</div><h1>从真实工作出发，<br>让知识有落点。</h1><p class="page-intro">每个项目分别整理问题、方法、实现、结果和边界。不能用一个 speedup 概括不同实验，也不能把扩展知识写成个人贡献。</p><div class="project-grid">${[
{title:'Qwen3-Triton',label:'PERFORMANCE / SYSTEMS',desc:'Kernel、推理 Runtime、训练状态与全分片基础设施。',metric:'3 条工作流',caption:'推理 / 单卡训练 / 多卡分片，不是同一个 benchmark。',id:'project-qwen',links:['qk-fusion','cuda-graph','tied-grad','block-fsdp']},
{title:'DFuzz',label:'NUMERICAL RELIABILITY',desc:'配置搜索与输入搜索，研究训练算子优化的数值偏差。',metric:'16 / 44',caption:'论文阈值下的偏差案例；12 项此前未被记录，不等价16个代码缺陷。',id:'project-dfuzz',links:['floating-point','dfuzz-config','dfuzz-input','dfuzz-megatron']},
{title:'DeepOpFuzz',label:'COMPILER CORRECTNESS',desc:'Operation-guided generation、artifact observation 与多类 oracle。',metric:'86 bugs',caption:'论文统计：54 silent、32 crash，43 fixed；不是最新上游状态追踪。',id:'project-deepop',links:['inductor','deepop-method','pool-bug','backward-bug']}
].map(p=>`<section class="project-card"><div class="project-top"><div class="card-kicker">${p.label}</div><h2>${p.title}</h2><p>${p.desc}</p></div><div class="project-body"><div class="project-measure">${p.metric}</div><div class="metric-caption">${p.caption}</div><a class="btn btn-primary" href="#article/${p.id}">阅读完整档案 ${icon('arrow')}</a><div class="project-links">${p.links.map(id=>articleLink(id)).join('')}</div></div></section>`).join('')}</div><div class="note-box"><strong>你做了什么，与系统通常怎么做，是两层事实。</strong><br>Megatron 是 DFuzz 的实际测试对象；FSDP 是 Qwen 仓库的真实实现分支。通用 TP、PP、Serving、量化扩展章节会明确标注，不据此增添简历成果。</div><a class="btn" href="#article/interview">项目连环追问与证据复盘 ${icon('arrow')}</a>${footer()}</div>`}
function glossaryPage(){return `<div class="page">${breadcrumb('术语索引')}<div class="eyebrow">TERMS IN CONTEXT</div><h1>按概念查阅术语</h1><p class="page-intro">先查定义，再沿章节链接阅读公式、实现和应用。</p><input class="filter-input" id="glossary-filter" type="search" placeholder="搜索术语、中文解释或关键词，例如 memory、梯度、alias…" aria-label="过滤术语"><div class="glossary-grid" id="glossary-results">${glossaryCards('')}</div>${footer()}</div>`}
function glossaryCards(query){const q=query.trim().toLowerCase();const found=DB.glossary.filter(t=>(t.term+t.zh+t.definition).toLowerCase().includes(q));return found.length?found.map(t=>`<section class="term-card"><h3>${escape(t.term)}</h3><span class="zh">${escape(t.zh)}</span><p>${escape(t.definition)}</p><a href="#article/${t.article}">读懂这个概念 →</a></section>`).join(''):'<div class="empty">没有匹配术语。试试全站搜索。</div>'}
function field(name,label,value,step=1){return `<div class="field"><label for="${name}">${label}</label><input type="number" id="${name}" value="${value}" min="${step}" step="${step}" inputmode="decimal"></div>`}
function labPage(){return `<div class="page">${breadcrumb('计算实验室')}<div class="eyebrow">INTERACTIVE THEORY LAB</div><h1>用输入规模推导系统开销</h1><p class="page-intro">调整张量形状、缓存长度或并行度，观察理论计算量、显存和通信量。每项计算都链接到公式推导。</p><div class="calc-grid">
<section class="calc-card"><div class="card-kicker">01 / GEMM</div><h2>计算量与算术强度</h2><p>A[M,K] × B[K,N] → C[M,N]。假设输入各读一次、输出写一次，忽略 bias、cache重读与workspace。</p><div class="calc-fields">${field('gemm-m','M · 输入行数',1)}${field('gemm-n','N · 输出列数',2048)}${field('gemm-k','K · 归约维度',2048)}${field('gemm-b','每个输入/输出元素字节',2)}${field('gemm-peak','假设峰值 · TFLOP/s',100,0.1)}${field('gemm-bw','假设内存带宽 · GB/s',500,0.1)}</div><div class="calc-result" id="gemm-result" aria-live="polite"></div><div class="calc-formula">F ≈ 2MNK; Q = b(MK+KN+MN); AI=F/Q</div><p>峰值和带宽为教学占位值，不是 A6000 规格。Roofline 不包含 launch 开销。</p><a href="#article/gemm-flops">逐步推导 2MNK →</a></section>
<section class="calc-card"><div class="card-kicker">02 / KV CACHE</div><h2>历史 K/V 存储量</h2><p>按紧凑KV heads计算，不包括padding、分配器、workspace与额外预留capacity。</p><div class="calc-fields">${field('kv-batch','Batch',1)}${field('kv-seq','缓存 Token 数',512)}${field('kv-layers','层数 L',28)}${field('kv-heads','KV Heads',8)}${field('kv-dim','Head Dimension',128)}${field('kv-bytes','字节 / 元素',2)}</div><div class="calc-result" id="kv-result" aria-live="polite"></div><div class="calc-formula">Bytes = 2 × B × S × L × Hkv × D × b</div><p>2代表K和V。默认使用1.7B项目结构，不混用8B的36层配置。</p><a href="#article/gqa-mla">GQA 与缓存公式 →</a></section>
<section class="calc-card"><div class="card-kicker">03 / FULL SHARD</div><h2>训练状态显存账本</h2><p>假设FP32 master和两份FP32 Adam状态，每参数12字节；可另外计FP32本地梯度。</p><div class="calc-fields">${field('fsdp-p','参数量 · Billion',8.19073536,0.00000001)}${field('fsdp-world','World Size',4)}<div class="field"><label for="fsdp-grad">本地梯度状态</label><select id="fsdp-grad"><option value="4">计入 FP32 梯度 (+4B)</option><option value="0">不计梯度</option></select></div></div><div class="calc-result" id="fsdp-result" aria-live="polite"></div><div class="calc-formula">State/rank = P × (12 + grad_bytes) / N</div><p>不含当前完整单元、激活、重算峰值、NCCL临时区；不能代替35.76GiB的项目实测。</p><a href="#article/block-fsdp">沿 BlockFSDP 的生命周期阅读 →</a></section>
<section class="calc-card"><div class="card-kicker">04 / RING ALL-REDUCE</div><h2>每卡流量与理想时间</h2><p>固定消息S，假设均匀全双工链路。发送和接收各一份，不能混叫“总字节”。</p><div class="calc-fields">${field('ring-n','Rank 数 N',8)}${field('ring-s','Tensor 大小 · MiB',128,0.1)}${field('ring-bw','假设有效链路 · GB/s',50,0.1)}${field('ring-alpha','每轮延迟 α · μs',5,0.1)}</div><div class="calc-result" id="ring-result" aria-live="polite"></div><div class="calc-formula">T ≈ 2(N−1)α + 2(N−1)S/(N×BW)</div><p>忽略拓扑、拥塞、分块与规约计算；NCCL不保证始终使用Ring。</p><a href="#article/collectives">用具体数组理解 collective →</a></section>
</div><div class="section-head"><h2>可下载的动手实验</h2><span class="small">4 个 CPU 实验 + 2 个可选 GPU 实验</span></div><div class="chapter-index">${Object.entries(DB.labs).map(([p,c],i)=>`<div class="chapter-item"><strong>${escape(p.split('/').pop())}</strong><p>${i<4?'Python + NumPy；已完成 CPU 执行检查。':'需要兼容 CUDA / PyTorch / Triton；本站未执行 GPU 路径。'}</p><button class="btn btn-small" data-download-lab="${p}" style="margin-top:12px">${icon('down')}下载脚本</button></div>`).join('')}</div>${footer()}</div>`}
function validNumbers(ids,integers=[]){const o={};for(const id of ids){const n=Number($('#'+id).value);if(!Number.isFinite(n)||n<=0||n>1e12||(integers.includes(id)&&!Number.isInteger(n)))return null;o[id]=n}return o}
const num=(n,d=2)=>n.toLocaleString('zh-CN',{maximumFractionDigits:d,minimumFractionDigits:0});
const row=(title,result)=>`<div class="result-row"><span>${title}</span><strong>${result}</strong></div>`;
function updateCalculators(){
let x=validNumbers(['gemm-m','gemm-n','gemm-k','gemm-b','gemm-peak','gemm-bw'],['gemm-m','gemm-n','gemm-k','gemm-b']);
if(x){const M=x['gemm-m'],N=x['gemm-n'],K=x['gemm-k'],F=2*M*N*K,Q=x['gemm-b']*(M*K+K*N+M*N),ai=F/Q,limit=Math.min(x['gemm-peak'],ai*x['gemm-bw']/1000);$('#gemm-result').innerHTML=row('约定计算量',num(F/1e9,4)+' GFLOPs')+row('理想逻辑流量',num(Q/2**20,3)+' MiB')+row('Arithmetic intensity',num(ai,3)+' FLOPs/B')+row('理论吞吐上界',num(limit,3)+' TFLOP/s')+`<small>${ai*x['gemm-bw']/1000<x['gemm-peak']?'该模型的带宽上界更低':'该模型的计算上界更低'}，不是实测瓶颈判定。</small>`}else $('#gemm-result').textContent='请输入正数；矩阵维度和字节数应为整数。';
x=validNumbers(['kv-batch','kv-seq','kv-layers','kv-heads','kv-dim','kv-bytes'],['kv-batch','kv-seq','kv-layers','kv-heads','kv-dim','kv-bytes']);
if(x){const size=2*Object.values(x).reduce((p,v)=>p*v,1);$('#kv-result').innerHTML=row('KV 内容容量',num(size/2**20,3)+' MiB')+row('换算 GiB',num(size/2**30,4)+' GiB')+row('每序列每 token',num(size/x['kv-batch']/x['kv-seq']/1024,3)+' KiB')}else $('#kv-result').textContent='所有参数需要是正整数。';
x=validNumbers(['fsdp-p','fsdp-world'],['fsdp-world']);
if(x){const P=x['fsdp-p']*1e9,N=x['fsdp-world'],g=Number($('#fsdp-grad').value);$('#fsdp-result').innerHTML=row('Master + m + v / rank',num(P*12/N/2**30,3)+' GiB')+row('选定状态合计 / rank',num(P*(12+g)/N/2**30,3)+' GiB')+row('完整 BF16 权重内容',num(P*2/2**30,3)+' GiB')}else $('#fsdp-result').textContent='参数量必须为正数，world size必须为正整数。';
x=validNumbers(['ring-n','ring-s','ring-bw','ring-alpha'],['ring-n']);
if(x){const N=x['ring-n'],S=x['ring-s']*2**20,send=2*(N-1)/N*S,lat=2*(N-1)*x['ring-alpha']/1000,bw=send/(x['ring-bw']*1e9)*1000;$('#ring-result').innerHTML=row('每 rank 发送量',num(send/2**20,3)+' MiB')+row('接收量（另计）',num(send/2**20,3)+' MiB')+row('延迟项 / 带宽项',num(lat,3)+' / '+num(bw,3)+' ms')+row('理想模型总时间',num(lat+bw,3)+' ms')}else $('#ring-result').textContent='请输入正数，rank数必须为整数。';
}
function sourcesPage(){const types=[...new Set(Object.values(DB.sources).map(s=>s.kind))];return `<div class="page">${breadcrumb('资料与边界')}<div class="eyebrow">SOURCES & EVIDENCE</div><h1>把出处和不确定性一起留下。</h1><p class="page-intro">${Object.keys(DB.sources).length} 条来源索引，连接原始论文、官方文档和固定版本项目代码。章节末尾给出对应来源；外部链接需要联网，正文、数学公式和练习无需联网。</p><div class="note-box"><strong>阅读边界</strong><br>本网站为学习教程，不是完整实验复现包。用户论文按提供稿件解释，未收录PDF；GPU性能来自项目报告，未在此重新测量。官方动态文档仅作概念与API参考，不能自动替换历史实验版本。发布前请确认研究稿件和项目结果允许公开。</div><div class="note-box"><strong>对来源本身保留判断</strong><br>DFuzz的阈值是论文实验约定，不能推出BF16存在通用的10⁻⁶噪声底；预算搜索不证明全局最坏值。DeepOpFuzz的30根因回放与48个历史bug效率实验分开呈现。Bug草稿中的解释与上游最终修复不是同一种证据。</div><input id="source-filter" class="filter-input" type="search" aria-label="过滤来源" placeholder="按标题、项目、文档或文件名查找…"><div id="source-results">${types.map(t=>`<section class="source-group"><h2>${escape(t)}</h2><div class="source-grid">${Object.values(DB.sources).filter(s=>s.kind===t).map(s=>sourceCard(s.id)).join('')}</div></section>`).join('')}</div>${footer()}</div>`}
function notebookPage(){const noted=Object.keys(state.notes).filter(id=>byId[id]&&state.notes[id]?.trim());return `<div class="page">${breadcrumb('阅读记录')}<div class="eyebrow">YOUR LOCAL NOTEBOOK</div><h1>留下自己的推导。</h1><p class="page-intro">阅读记录和笔记仅保存在此浏览器，不上传到服务器。不同文件地址、设备或浏览器之间不自动同步；可手动导出、导入 JSON 备份。</p><div class="toolbar"><button class="btn btn-primary" id="export-state">${icon('down')}导出阅读记录</button><button class="btn" id="import-state">导入备份</button><input id="state-file" type="file" accept="application/json,.json" hidden><button class="btn" id="reset-state">清空本地记录</button></div><div class="stats"><div class="stat"><strong>${state.read.length}</strong><span>章节已读</span></div><div class="stat"><strong>${state.saved.length}</strong><span>收藏章节</span></div><div class="stat"><strong>${noted.length}</strong><span>篇个人笔记</span></div><div class="stat"><strong>${Math.round(state.read.length/articles.length*100)}%</strong><span>阅读进度</span></div></div><h2>收藏章节</h2><div class="saved-list">${state.saved.length?state.saved.map(id=>`<a class="chapter-item" href="#article/${id}"><strong>${escape(byId[id].title)}</strong><p>${escape(byId[id].summary)}</p></a>`).join(''):'<div class="empty">在章节顶部点击“收藏章节”，把需要反复阅读的内容放在这里。</div>'}</div><h2>笔记</h2>${noted.length?noted.map(id=>`<section class="card" style="margin:12px 0"><h3>${articleLink(id)}</h3><p style="white-space:pre-wrap;font-size:13px;color:var(--muted)">${escape(state.notes[id])}</p></section>`).join(''):'<div class="empty">每篇章节底部都有笔记区。用自己的话重述一次，胜过再看一遍定义。</div>'}${footer()}</div>`}
function notFound(){return `<div class="page"><div class="eyebrow">NOT FOUND</div><h1>这一页没有找到。</h1><p>链接中的章节标识不在当前版本中。</p><a class="btn btn-primary" href="#map">回到完整目录</a></div>`}
function downloadFile(name,text,type='text/plain;charset=utf-8'){const blob=new Blob([text],{type});const u=URL.createObjectURL(blob);const a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),2000)}
function enhanceArticle(){
$$('.article-body pre').forEach(p=>{const raw=p.textContent;const lang=p.querySelector('code')?.className.match(/language-([\w+-]+)/)?.[1]||p.className.split(' ')[0]||'code';p.dataset.copy='true';const tag=document.createElement('span');tag.className='code-language';tag.textContent=lang.toUpperCase();p.appendChild(tag);const b=document.createElement('button');b.className='code-copy';b.innerHTML=icon('copy')+'复制';b.setAttribute('aria-label','复制代码');b.addEventListener('click',()=>copy(raw));p.appendChild(b)});
$$('.article-body table').forEach(t=>{const w=document.createElement('div');w.className='table-wrap';t.parentNode.insertBefore(w,t);w.appendChild(t)});
if(observer)observer.disconnect();
observer=new IntersectionObserver(items=>{const visible=items.filter(x=>x.isIntersecting);if(visible.length){const id=visible[0].target.id;$$('[data-toc]').forEach(a=>a.classList.toggle('active',a.dataset.toc===id))}},{rootMargin:'-100px 0px -65% 0px'});
$$('.article-body h2,.article-body h3,#self-check,#article-sources').forEach(h=>observer.observe(h));
}
async function copy(text){try{if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(text)}else{const t=document.createElement('textarea');t.value=text;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();if(!document.execCommand('copy'))throw Error();t.remove()}toast('代码已复制')}catch(e){toast('浏览器不允许自动复制，请手动选择代码。')}}
function scrollRoute(info){let el=null;if(info.type==='article'&&info.heading)el=document.getElementById(decodeURIComponent(info.heading));else if(info.type==='map'&&info.id)el=document.getElementById(info.id.startsWith('section-')?info.id:'path-'+info.id);if(el)el.scrollIntoView({behavior:'auto',block:'start'});else window.scrollTo({top:0,behavior:'auto'})}
function render(){
const info=routeInfo();if(info.type==='article'&&activeArticle?.id===info.id&&info.heading){scrollRoute(info);return}
activeArticle=null;if(observer)observer.disconnect();
const renderers={home:home,map:mapPage,article:()=>article(info.id),projects:projectsPage,lab:labPage,glossary:glossaryPage,sources:sourcesPage,notebook:notebookPage};
$('#main').innerHTML=(renderers[info.type]||notFound)();sidebar();currentNav(info.type);closeMenu();
document.title=(activeArticle?activeArticle.title:({home:'从公式到代码，从代码到证据',map:'学习地图',projects:'项目档案',lab:'计算实验室',glossary:'术语索引',sources:'资料与边界',notebook:'阅读记录'}[info.type]||'页面'))+' · AI Infra Atlas';
if(activeArticle)enhanceArticle();if(info.type==='lab')updateCalculators();wirePage();requestAnimationFrame(()=>scrollRoute(info));
}
function wirePage(){
$('#glossary-filter')?.addEventListener('input',e=>$('#glossary-results').innerHTML=glossaryCards(e.target.value));
$('#source-filter')?.addEventListener('input',e=>{const q=e.target.value.toLowerCase();$$('.source-group').forEach(g=>{let count=0;g.querySelectorAll('.resource').forEach(r=>{const ok=r.textContent.toLowerCase().includes(q);r.hidden=!ok;if(ok)count++});g.hidden=count===0})});
$$('.calc-fields input,.calc-fields select').forEach(e=>e.addEventListener('input',updateCalculators));
$$('[data-download-lab]').forEach(b=>b.addEventListener('click',()=>{const path=b.dataset.downloadLab;downloadFile(path.split('/').pop(),DB.labs[path]);toast('已生成实验脚本')}));
$('#read-toggle')?.addEventListener('click',()=>{const id=activeArticle.id;state.read=state.read.includes(id)?state.read.filter(x=>x!==id):[...state.read,id];persist();$('#read-toggle').innerHTML=icon(state.read.includes(id)?'check':'book')+(state.read.includes(id)?'已标记读完':'标记读完');sidebar();toast(state.read.includes(id)?'已保存阅读进度':'已取消已读标记')});
$('#save-toggle')?.addEventListener('click',()=>{const id=activeArticle.id;state.saved=state.saved.includes(id)?state.saved.filter(x=>x!==id):[...state.saved,id];persist();$('#save-toggle').innerHTML=icon('bookmark')+(state.saved.includes(id)?'已收藏':'收藏章节')});
$('#export-md')?.addEventListener('click',()=>downloadFile(activeArticle.id+'.md','# '+activeArticle.title+'\n\n'+activeArticle.raw+'\n\n## 来源\n'+activeArticle.sources.map(id=>{const s=DB.sources[id];return '- '+s.title+(s.url?' '+s.url:'')+'\n  '+s.note}).join('\n'),'text/markdown;charset=utf-8'));
$('#print-article')?.addEventListener('click',()=>window.print());
$('#note-input')?.addEventListener('input',e=>{state.notes[activeArticle.id]=e.target.value;persist();$('#note-status').textContent=storageAvailable?'已保存到当前浏览器':'当前浏览器存储不可用，请导出备份'});
$('#export-state')?.addEventListener('click',()=>downloadFile('infra-atlas-progress.json',JSON.stringify({version:1,exported:new Date().toISOString(),state},null,2),'application/json'));
$('#import-state')?.addEventListener('click',()=>$('#state-file').click());
$('#state-file')?.addEventListener('change',async e=>{try{const f=e.target.files[0];if(!f)return;if(f.size>5e6)throw Error('备份文件过大');const d=JSON.parse(await f.text());const x=d.state;if(!x||!Array.isArray(x.read)||!Array.isArray(x.saved)||typeof x.notes!=='object')throw Error('不支持的备份格式');const notes={};for(const [id,n]of Object.entries(x.notes)){if(byId[id]&&typeof n==='string')notes[id]=n.slice(0,100000)}state={...state,read:x.read.filter(id=>byId[id]),saved:x.saved.filter(id=>byId[id]),notes};persist();render();toast('阅读记录已导入')}catch(err){toast('导入失败：'+err.message)}});
$('#reset-state')?.addEventListener('click',()=>{if(confirm('将清空本浏览器的已读、收藏和笔记。请先导出备份。继续吗？')){state={...state,read:[],saved:[],notes:{},last:'orientation'};persist();render();toast('本地记录已清空')}});
}
function closeMenu(){$('#sidebar').classList.remove('open');$('#mobile-shade').classList.remove('visible');$('#menu-toggle').setAttribute('aria-expanded','false')}
function searchText(query){const q=query.trim().toLowerCase();if(!q)return articles.filter(a=>['gemm-flops','qk-fusion','block-fsdp','dfuzz-config','inductor'].includes(a.id)).map(a=>({a,score:1}));const tokens=q.split(/\s+/).filter(Boolean);return articles.map(a=>{const title=(a.title+' '+a.tags.join(' ')).toLowerCase(),summary=a.summary.toLowerCase(),body=a.plain.toLowerCase();let score=0;for(const t of tokens){if(title.includes(t))score+=16;if(summary.includes(t))score+=7;if(body.includes(t))score+=2;else if(!title.includes(t)&&!summary.includes(t))return {a,score:0}}return {a,score}}).filter(x=>x.score>0).sort((a,b)=>b.score-a.score).slice(0,18)}
function highlight(text,q){if(!q.trim())return escape(text);const escaped=escape(text);const words=q.trim().split(/\s+/).map(w=>w.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).filter(Boolean);try{return escaped.replace(new RegExp('('+words.map(escape).join('|')+')','gi'),'<mark>$1</mark>')}catch(e){return escaped}}
function doSearch(){const q=$('#search-input').value;const results=searchText(q);$('#search-results').innerHTML=results.length?results.map(({a})=>{const pos=q.trim()?a.plain.toLowerCase().indexOf(q.trim().split(/\s+/)[0].toLowerCase()):-1;const snippet=pos>=0?a.plain.slice(Math.max(0,pos-35),pos+110):a.summary;return `<a class="search-result" href="#article/${a.id}"><small>${sec[a.section].title} · ${a.kind}</small><strong>${highlight(a.title,q)}</strong><p>${highlight(snippet,q)}${pos>=0?'…':''}</p></a>`}).join(''):'<div class="empty">没有匹配章节。试试中文术语或更短的关键词。</div>';$('#search-count').textContent=q.trim()?`${results.length} 个相关章节`:'推荐起点';$$('.search-result').forEach(a=>a.addEventListener('click',()=>$('#search-dialog').close()))}
function openSearch(){const dialog=$('#search-dialog');if(!dialog.open)dialog.showModal();doSearch();setTimeout(()=>$('#search-input').focus(),10)}
$('#search-open').innerHTML=icon('search')+'<span class="search-word">搜索知识库</span><kbd>⌘ K</kbd>';
$('#menu-toggle').innerHTML=icon('menu');$('#search-close').innerHTML=icon('close');
$('#search-open').addEventListener('click',openSearch);$('#search-close').addEventListener('click',()=>$('#search-dialog').close());$('#search-input').addEventListener('input',doSearch);
$('#search-dialog').addEventListener('click',e=>{if(e.target===e.currentTarget){const r=e.currentTarget.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.currentTarget.close()}});
$('#search-input').addEventListener('keydown',e=>{if(e.key==='ArrowDown'){e.preventDefault();$('.search-result')?.focus()}if(e.key==='Enter'){$('.search-result')?.click()}});
$('#search-results').addEventListener('keydown',e=>{if(!['ArrowDown','ArrowUp'].includes(e.key))return;e.preventDefault();const list=$$('.search-result');const i=list.indexOf(document.activeElement);list[(i+(e.key==='ArrowDown'?1:-1)+list.length)%list.length]?.focus()});
document.addEventListener('keydown',e=>{const typing=['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName);if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();openSearch()}else if(e.key==='/'&&!typing){e.preventDefault();openSearch()}if(e.key==='Escape')closeMenu()});
$('#menu-toggle').addEventListener('click',()=>{const open=$('#sidebar').classList.toggle('open');$('#mobile-shade').classList.toggle('visible',open);$('#menu-toggle').setAttribute('aria-expanded',String(open))});$('#mobile-shade').addEventListener('click',closeMenu);
$('#theme-toggle').addEventListener('click',()=>setTheme(state.theme==='dark'?'light':'dark'));
$('.skip').addEventListener('click',e=>{e.preventDefault();$('#main').focus();window.scrollTo({top:0,behavior:'auto'})});
window.addEventListener('hashchange',render);
setTheme(state.theme==='dark'?'dark':'light');render();
})();
