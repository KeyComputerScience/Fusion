"""Deterministic vector framework drawing matching the reconstructed methods."""
from pathlib import Path
import json
import base64
from io import BytesIO
from html import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import argparse
parser = argparse.ArgumentParser(description='Draw the exact framework using Times New Roman.')
parser.add_argument('--font-dir', required=True, help='Licensed Times New Roman TTF folder')
parser.add_argument('--output', default='fig')
args = parser.parse_args()
OUT = Path(args.output)
OUT.mkdir(parents=True, exist_ok=True)
PDF = OUT / 'information_fusion_framework.pdf'
SVG = OUT / 'information_fusion_framework.svg'
PNG = OUT / 'information_fusion_framework.png'
W, H = 824.0, 508.0

font_files = {
    'regular': Path(args.font_dir) / 'times.ttf',
    'bold': Path(args.font_dir) / 'timesbd.ttf',
    'italic': Path(args.font_dir) / 'timesi.ttf',
}
embedded = all(p.is_file() for p in font_files.values())
if not embedded:
    raise FileNotFoundError('Actual Times New Roman font files are required; no substitute is used.')
names = {'regular': 'TNR', 'bold': 'TNR-Bold', 'italic': 'TNR-Italic'}
ps_names = {'regular': 'TimesNewRomanPSMT', 'bold': 'TimesNewRomanPS-BoldMT', 'italic': 'TimesNewRomanPS-ItalicMT'}
base_names = {'regular': 'Times-Roman', 'bold': 'Times-Bold', 'italic': 'Times-Italic'}
for style, font_name in names.items():
    if embedded:
        pdfmetrics.registerFont(TTFont(font_name, str(font_files[style])))
    else:
        names[style] = base_names[style]

c = canvas.Canvas(str(PDF), pagesize=(W, H), pageCompression=1, initialFontName=names['regular'])
c.setTitle('Information fusion framework for adaptive edge AI services')
c.setSubject('Multi-source evidence alignment, reliability weighting, drift fusion, source disagreement and feedback-driven resource coordination')
c.setAuthor('')
c.setCreator('Vector diagram')
c.setFillColorRGB(1, 1, 1)
c.rect(0, 0, W, H, stroke=0, fill=1)
c.setFillColorRGB(0, 0, 0)
c.setStrokeColorRGB(0, 0, 0)
svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    '<title>Information fusion framework for adaptive edge AI services</title>',
    '<desc>Completed-window workload, operating-state and performance evidence is aligned, normalized and fused using source reliability. Fused drift and source disagreement inform joint retraining and inference decisions. Edge-cloud outcomes update source reliability and decision proxies.</desc>',
    '<style>text{font-family:"Times New Roman",Times,serif;fill:#000} .line{stroke:#000;fill:none}</style>',
    f'<rect x="0" y="0" width="{W}" height="{H}" fill="#fff"/>',
]
checks = []

def line(x1, y1, x2, y2, width=1.0, dashed=False):
    c.setLineWidth(width)
    c.setDash(3.5, 2.7) if dashed else c.setDash()
    c.line(x1, H-y1, x2, H-y2)
    dash = ' stroke-dasharray="3.5 2.7"' if dashed else ''
    svg.append(f'<line class="line" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke-width="{width}"{dash}/>')
    c.setDash()

def rect(x, y, w, h, width=1.0):
    c.setLineWidth(width)
    c.setFillColorRGB(1, 1, 1)
    c.rect(x, H-y-h, w, h, stroke=1, fill=1)
    c.setFillColorRGB(0, 0, 0)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#fff" stroke="#000" stroke-width="{width}"/>')

def text(x, y, s, size=13.0, style='regular', anchor='middle', max_width=None):
    font = names[style]
    length = pdfmetrics.stringWidth(s, font, size)
    if max_width is not None:
        checks.append({'text':s,'width':round(length,2),'limit':max_width,'fits':length <= max_width})
        if length > max_width:
            raise ValueError(f'Text exceeds its box: {s!r} ({length} > {max_width})')
    c.setFont(font, size)
    c.setFillColorRGB(0, 0, 0)
    if anchor == 'middle':
        c.drawCentredString(x, H-y, s)
    elif anchor == 'end':
        c.drawRightString(x, H-y, s)
    else:
        c.drawString(x, H-y, s)
    weight = ' font-weight="bold"' if style == 'bold' else ''
    italic = ' font-style="italic"' if style == 'italic' else ''
    svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"{weight}{italic}>{escape(s)}</text>')

def arrow(points, dashed=False, width=1.15, head=5.5):
    for a,b in zip(points, points[1:]):
        line(*a,*b,width=width,dashed=dashed)
    a, b = points[-2], points[-1]
    dx, dy = b[0]-a[0], b[1]-a[1]
    norm = (dx*dx+dy*dy)**0.5
    ux, uy = dx/norm, dy/norm
    vx, vy = -uy, ux
    tip = b
    left = (b[0]-head*ux+head*0.48*vx,b[1]-head*uy+head*0.48*vy)
    right = (b[0]-head*ux-head*0.48*vx,b[1]-head*uy-head*0.48*vy)
    path = c.beginPath()
    path.moveTo(tip[0],H-tip[1]);path.lineTo(left[0],H-left[1]);path.lineTo(right[0],H-right[1]);path.close()
    c.setFillColorRGB(0,0,0);c.drawPath(path,fill=1,stroke=0)
    svg.append(f'<polygon points="{tip[0]},{tip[1]} {left[0]},{left[1]} {right[0]},{right[1]}" fill="#000"/>')

def panel(x, y, w, h, title):
    rect(x,y,w,h,1.25)
    text(x+w/2,y+21,title,14.8,'bold',max_width=w-16)
    line(x+1,y+32,x+w-1,y+32,width=0.85)

def block(x, y, w, h, title, body, title_size=13.4, body_size=13.0):
    rect(x,y,w,h,0.9)
    text(x+w/2,y+18.5,title,title_size,'bold',max_width=w-12)
    start = y+34.8 if len(body) == 3 else y+37
    leading = 13.4 if len(body) == 3 else 15.2
    for i,s in enumerate(body):
        text(x+w/2,start+i*leading,s,body_size,max_width=w-12)

text(W/2,22,'Multi-Source Information Fusion for Adaptive Edge AI Services',17.0,'bold',max_width=W-28)

panel(12,44,198,322,'1. Heterogeneous sources')
panel(227,44,310,322,'2. Reliability-aware evidence fusion')
panel(554,44,258,322,'3. Retraining-inference coordination')

block(23,87,176,62,'Workload evidence',[
    'Task arrival / workload mix',
    'Historical and current regimes',
],body_size=12.8)
block(23,163,176,62,'Operating-state evidence',[
    'Queues / utilization / bandwidth',
    'Available edge-cloud resources',
],title_size=13.0,body_size=12.5)
block(23,239,176,62,'Performance evidence',[
    'Reward / service completion',
    'Deadline loss / SLA outcomes',
],body_size=12.7)
rect(23,317,176,32,0.9)
text(111,338,'Completed windows only',13.0,'italic',max_width=164)

block(240,86,284,48,'Window alignment and normalization',[
    'Common time scale; missing-source masks',
],title_size=13.5,body_size=13.0)
arrow([(382,134),(382,145)])
block(240,145,284,70,'Source evidence extraction',[
    'd1: workload-regime divergence (JSD)',
    'd2: normalized operating-state shift',
    'd3: normalized reward degradation',
],body_size=13.0)
arrow([(382,215),(382,226)])
block(240,226,284,71,'Reliability-weighted fusion',[
    'Source reliability: support, freshness and noise',
    'Normalize reliability over available sources',
    'Fuse evidence using adaptive source weights',
],body_size=12.9)
arrow([(382,297),(382,308)])
block(240,308,284,45,'Fused drift and source disagreement',[
    'Weighted mean and weighted variance',
],title_size=13.5,body_size=13.0)

block(567,86,232,48,'SLA, priorities and resource budgets',[
    'Deadline / availability / throughput',
],title_size=13.2,body_size=13.0)
arrow([(683,134),(683,146)])
block(567,146,232,69,'Decision-value estimation',[
    'Delayed policy-recovery value',
    'Current inference-service value',
    'Drift and disagreement inform decisions',
],body_size=12.9)
arrow([(683,215),(683,227)])
block(567,227,232,71,'Joint resource coordination',[
    'Exact search for 5 x 6 profiles',
    'AO for larger configuration sets',
    'Disagreement-aware profile selection',
],body_size=12.9)
arrow([(683,298),(683,308)])
block(567,308,232,45,'Selected execution profiles',[
    'Retraining intensity and inference mode',
],body_size=12.9)

for y in (118,194,270):
    line(199,y,217,y,1.0)
line(217,110,217,270,1.0)
arrow([(217,110),(240,110)])
arrow([(524,330),(545,330),(545,182),(567,182)])

rect(12,390,525,81,1.25)
text(274.5,410,'4. Outcome feedback and fusion calibration',14.4,'bold',max_width=505)
text(274.5,433,'Task completion / deadline loss / queues / retraining cost',13.3,max_width=499)
text(274.5,454,'Update source reliability and calibrate decision-value proxies',13.3,max_width=499)

rect(554,390,258,81,1.25)
text(683,410,'Edge-cloud service pool',14.4,'bold',max_width=242)
text(683,432,'Edge nodes 1, 2, ..., M and cloud',13.2,max_width=238)
text(683,449,'Colocated inference and policy updates',13.0,max_width=238)
text(683,465,'Shared memory, compute and budget',12.9,max_width=238)

arrow([(683,353),(683,390)])
arrow([(554,434),(537,434)])
arrow([(111,390),(111,366)],dashed=True)
arrow([(382,390),(382,366)],dashed=True)

line(18,491,43,491,1.1)
text(48,495,'Evidence / decision / execution',11.8,anchor='start')
line(251,491,276,491,1.1,dashed=True)
text(281,495,'Completed feedback',11.8,anchor='start')

c.showPage();c.save()
if embedded:
    from fontTools import subset
    from fontTools.ttLib import TTFont as FTFont
    face_rules = []
    used_text = '\n'.join(svg)
    for style, path in font_files.items():
        font = FTFont(path)
        opts = subset.Options()
        opts.name_IDs = [0,1,2,3,4,5,6]
        opts.name_languages = [0x409]
        reducer = subset.Subsetter(options=opts)
        reducer.populate(text=used_text)
        reducer.subset(font)
        font.flavor = 'woff'
        data = BytesIO();font.save(data)
        payload = base64.b64encode(data.getvalue()).decode('ascii')
        weight = '700' if style == 'bold' else '400'
        slant = 'italic' if style == 'italic' else 'normal'
        face_rules.append("@font-face{font-family:'Times New Roman';font-weight:"+weight+";font-style:"+slant+";src:url('data:font/woff;base64,"+payload+"') format('woff');}")
    svg.insert(4, '<defs><style>' + ''.join(face_rules) + '</style></defs>')
svg.append('</svg>')
SVG.write_text('\n'.join(svg),encoding='utf-8')

import fitz
doc=fitz.open(PDF)
doc[0].get_pixmap(dpi=600,alpha=False).save(PNG)
preview=OUT / 'information_fusion_framework_preview.png'
doc[0].get_pixmap(matrix=fitz.Matrix(2.3,2.3),alpha=False).save(preview)

qa = {
    'pdf':str(PDF),'svg':str(SVG),'png':str(PNG),
    'font_family':'Times New Roman',
    'font_embedded':embedded,
    'font_note': 'Actual Times New Roman embedded',
    'canvas_points':[W,H],
    'pdf_pages':len(doc),
    'text_fit_checks':checks,
}
(OUT / 'information_fusion_figure_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in qa.items() if k!='text_fit_checks'},ensure_ascii=False))
