#!/usr/bin/env python3
"""Build an entirely self-contained tutorial site.

Requires Python 3.10+ and Pandoc installed on PATH. Browser use has no
build-time or network dependency. Run from any directory: python build.py
"""
from __future__ import annotations
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def strip_markup(value: str) -> str:
    value = re.sub(r'<annotation\b[^>]*>.*?</annotation>', '', value, flags=re.S)
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', value))).strip()

def convert_markdown(raw: str) -> tuple[str, list[dict]]:
    proc = subprocess.run(
        ['pandoc', '--from=markdown+tex_math_dollars+raw_html', '--to=html5',
         '--mathml', '--no-highlight', '--wrap=none'],
        input=raw, text=True, capture_output=True, check=True, timeout=30,
    )
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr)
    toc = []
    def replace_header(match: re.Match) -> str:
        level, content = int(match.group(1)), match.group(2)
        identity = 'h' + str(len(toc)+1).zfill(2)
        toc.append(dict(id=identity, title=strip_markup(content), level=level))
        return f'<h{level} id="{identity}">{content}</h{level}>'
    body = re.sub(r'<h([23])\b[^>]*>(.*?)</h\1>', replace_header, proc.stdout, flags=re.S)
    # Keep native MathML block equations inside a scrollable visual panel.
    body = re.sub(r'<p>(<math display="block".*?</math>)</p>', r'<div class="math display">\1</div>', body, flags=re.S)
    # Open external links safely. Internal hash routes remain in the same document.
    body = re.sub(r'<a href="(https?://[^"]+)"', r'<a target="_blank" rel="noopener noreferrer" href="\1"', body)
    return body, toc

def main() -> None:
    if not shutil.which('pandoc'):
        raise SystemExit('Pandoc is required only for rebuilding. The delivered index.html already works offline.')
    content = ROOT/'content'
    catalog = json.loads((content/'catalog.json').read_text(encoding='utf-8'))
    structure = json.loads((content/'structure.json').read_text(encoding='utf-8'))
    sources = json.loads((content/'sources.json').read_text(encoding='utf-8'))
    glossary = json.loads((content/'glossary.json').read_text(encoding='utf-8'))
    section_order = {s['id']: i for i,s in enumerate(structure['sections'])}
    catalog.sort(key=lambda a: section_order[a['section']])
    ids = {a['id'] for a in catalog}
    if len(ids)!=len(catalog): raise SystemExit('Duplicate article ID')
    for a in catalog:
        raw = (content/(a['id']+'.md')).read_text(encoding='utf-8')
        a['raw'] = raw
        a['html'],a['toc'] = convert_markdown(raw)
        a['plain'] = strip_markup(a['html'])
        a['charCount'] = len(re.sub(r'\s+','',a['plain']))
        missing=set(a['prereq'])-ids
        if missing: raise SystemExit(f"Missing prerequisite for {a['id']}: {missing}")
        missing=set(a['sources'])-sources.keys()
        if missing: raise SystemExit(f"Missing source: {missing}")
        for link in re.findall(r'#article/([a-z0-9-]+)',raw):
            if link not in ids:raise SystemExit(f'Unknown internal link {link}')
        if a['lab'] and not (ROOT/a['lab']).is_file():raise SystemExit(f"Missing lab {a['lab']}")
    for g in glossary:
        if g['article'] not in ids:raise SystemExit('Invalid glossary target')
    for p in structure['paths']:
        if not set(p['items'])<=ids:raise SystemExit('Invalid learning path')
    labs={str(p.relative_to(ROOT)):p.read_text(encoding='utf-8') for p in sorted((ROOT/'labs').glob('*.py'))}
    payload=dict(articles=catalog,sections=structure['sections'],paths=structure['paths'],
                 sources=sources,glossary=glossary,labs=labs,
                 quizCount=sum(len(a['checks']) for a in catalog),version='1.0')
    data=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    template=(ROOT/'assets/template.html').read_text(encoding='utf-8')
    page=template.replace('@@CSS@@',(ROOT/'assets/site.css').read_text(encoding='utf-8')).replace('@@DATA@@',data).replace('@@JS@@',(ROOT/'assets/app.js').read_text(encoding='utf-8'))
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    (ROOT/'.nojekyll').touch()
    print(f'Built {len(catalog)} chapters / {payload["quizCount"]} questions / {len(sources)} sources / {len(page.encode())/1024:.0f} KiB')

if __name__=='__main__':main()
