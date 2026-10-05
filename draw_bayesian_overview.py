"""Deterministic SVG and PNG rendering of the manuscript overview.

The publication LaTeX figure is self-contained TikZ. This matching diagram
uses only SVG primitives and Pillow, with no generative rendering.
"""
from pathlib import Path
from html import escape
import math
import re
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
SCALE=2
UNIT=120
WIDTH,HEIGHT=2120,1340
X0,Y0=WIDTH/2,1130
COLORS={
    "ink":"#253449","blue":"#326A9A","amber":"#AA6423","green":"#327866",
    "paleBlue":"#EFF5FA","paleAmber":"#FBF4EB","paleGreen":"#EFF7F3",
}
FONTROOT=Path("/System/Library/Fonts/Supplemental")
REGULAR=FONTROOT/"Arial Unicode.ttf"
BOLD=FONTROOT/"Arial Bold.ttf"
canvas=Image.new("RGB",(WIDTH*SCALE,HEIGHT*SCALE),"white")
draw=ImageDraw.Draw(canvas)
svg=[
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
    '<title>Decision-specific Bayesian fusion for paid model updates</title>',
    '<desc>Joint action-error archive, conditional block posterior, convex source weights, corrected admission gate, fixed service leases, complete return audit and delayed calibration of all issued forks.</desc>',
    '<rect width="100%" height="100%" fill="white"/>',
]
checks=[]

def point(x,y):
    return X0+UNIT*x,Y0-UNIT*y

def font(size,bold=False):
    return ImageFont.truetype(str(BOLD if bold else REGULAR),round(size*SCALE))

def text_runs(text,size,bold=False):
    """Render simple math scripts without relying on missing Unicode glyphs."""
    runs=[]
    start=0
    for match in re.finditer(r"([\^_])([A-Za-z0-9]+)",text):
        if match.start()>start:
            runs.append((text[start:match.start()],size,0))
        runs.append((match.group(2),size*.7,-size*.38 if match.group(1)=="^" else size*.27))
        start=match.end()
    if start<len(text):
        runs.append((text[start:],size,0))
    return [(t,s,dy,draw.textlength(t,font=font(s,bold))/SCALE) for t,s,dy in runs]

def label(x,y,text,size=28,bold=False,color="ink",rotation=0):
    px,py=point(x,y)
    fill=COLORS.get(color,color)
    f=font(size,bold)
    if rotation:
        bounds=draw.textbbox((0,0),text,font=f)
        tile=Image.new("RGBA",(bounds[2]-bounds[0]+18*SCALE,bounds[3]-bounds[1]+18*SCALE),(255,255,255,255))
        td=ImageDraw.Draw(tile)
        td.text((9*SCALE-bounds[0],9*SCALE-bounds[1]),text,font=f,fill=fill)
        tile=tile.rotate(rotation,expand=True)
        canvas.paste(tile,(round(px*SCALE-tile.width/2),round(py*SCALE-tile.height/2)),tile)
        tr=f' transform="rotate({-rotation} {px} {py})"'
    else:
        runs=text_runs(text,size,bold)
        cursor=px-sum(run[3] for run in runs)/2
        for token,run_size,dy,run_width in runs:
            tx=cursor+run_width/2
            draw.text((round(tx*SCALE),round((py+dy)*SCALE)),token,
                      font=font(run_size,bold),fill=fill,anchor="mm")
            svg.append(
                f'<text x="{tx:.2f}" y="{py+dy:.2f}" text-anchor="middle" dominant-baseline="central"'
                f' font-family="Arial, Helvetica, sans-serif" font-size="{run_size}"'
                f' font-weight="{700 if bold else 400}" fill="{fill}">{escape(token)}</text>'
            )
            cursor+=run_width
        return
    svg.append(
        f'<text x="{px:.2f}" y="{py:.2f}" text-anchor="middle" dominant-baseline="central"'
        f' font-family="Arial, Helvetica, sans-serif" font-size="{size}"'
        f' font-weight="{700 if bold else 400}" fill="{fill}"{tr}>{escape(text)}</text>'
    )

def box(x,y,w,h,title,lines,fill="paleBlue",body_size=28,title_size=30):
    cx,cy=point(x,y)
    l,t=cx-w*UNIT/2,cy-h*UNIT/2
    draw.rounded_rectangle(
        (round(l*SCALE),round(t*SCALE),round((l+w*UNIT)*SCALE),round((t+h*UNIT)*SCALE)),
        radius=10*SCALE,fill=COLORS[fill],outline=COLORS["ink"],width=2*SCALE
    )
    svg.append(f'<rect x="{l:.2f}" y="{t:.2f}" width="{w*UNIT:.2f}" height="{h*UNIT:.2f}" rx="10" fill="{COLORS[fill]}" stroke="{COLORS["ink"]}" stroke-width="2"/>')
    line_height=body_size*1.24
    total_height=title_size+12+len(lines)*line_height
    first_y=y+(total_height/2-title_size/2)/UNIT
    label(x,first_y,title,title_size,True)
    body_start=first_y-(title_size/2+12+body_size/2)/UNIT
    for i,text in enumerate(lines):
        line_width=sum(run[3] for run in text_runs(text,body_size))
        assert line_width < w*UNIT-18, (title,text,line_width,w*UNIT)
        label(x,body_start-i*line_height/UNIT,text,body_size)
    title_width=draw.textlength(title,font=font(title_size,True))/SCALE
    assert title_width < w*UNIT-18,(title,title_width,w*UNIT)
    assert total_height < h*UNIT-12,(title,total_height,h*UNIT)
    checks.append({"title":title,"within_box":True})

def arrow(points,color="ink",dashed=False,width=2.5):
    pts=[point(*p) for p in points]
    fill=COLORS[color]
    dash_attr=' stroke-dasharray="10 7"' if dashed else ""
    svg.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in pts)}" fill="none" stroke="{fill}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>')
    for a,b in zip(pts,pts[1:]):
        if dashed:
            dx,dy=b[0]-a[0],b[1]-a[1]
            length=math.hypot(dx,dy)
            for start in range(0,math.ceil(length),17):
                end=min(start+10,length)
                q1=(a[0]+dx*start/length,a[1]+dy*start/length)
                q2=(a[0]+dx*end/length,a[1]+dy*end/length)
                draw.line(tuple(v*SCALE for q in [q1,q2] for v in q),fill=fill,width=round(width*SCALE))
        else:
            draw.line(tuple(v*SCALE for q in [a,b] for v in q),fill=fill,width=round(width*SCALE))
    a,b=pts[-2:]
    angle=math.atan2(b[1]-a[1],b[0]-a[0])
    length,half=13,5.6
    base=(b[0]-length*math.cos(angle),b[1]-length*math.sin(angle))
    poly=[b,(base[0]+half*math.sin(angle),base[1]-half*math.cos(angle)),
          (base[0]-half*math.sin(angle),base[1]+half*math.cos(angle))]
    draw.polygon([(round(x*SCALE),round(y*SCALE)) for x,y in poly],fill=fill)
    svg.append(f'<polygon points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in poly)}" fill="{fill}"/>')

# Short Unicode equations in SVG/PNG correspond to full TeX in the native figure.
label(0,9.2,"Decision-specific Bayesian fusion for paid model updates",34,True)
box(0,8.4,16.1,1.12,"Observable decision information",[
    "Source forecasts p, context u, source mask, current candidate/reference contrast b, current values h",
    "Empty mask: keep reference before inference     |     Single source: unit influence",
],body_size=26,title_size=29)
box(-5.5,6.45,4.6,2.28,"1. Joint action-error archive",[
    "Immutable forecasts, features, masks",
    "Attach audit labels after arrival",
    "Reproject z = (p − Y)^T b",
    "Original mask covers current mask",
    "No imputed cross-source errors",
],body_size=25.5,title_size=27.5)
box(0,6.45,4.6,2.28,"2. Conditional block posterior",[
    "Positive context/age masses a",
    "P ~ Dirichlet(a₁, …, a_G)",
    "μ: mean action-error correction",
    "Σ: predictive joint covariance",
    "U = B/(A + 1): correction uncertainty",
],fill="paleAmber",body_size=25.5,title_size=27.5)
box(5.5,6.45,4.6,2.28,"3. Convex source influence",[
    "Capped simplex; prefix prior π",
    "min  τ KL(w || π)",
    "+ λ w^T(Σ + ηU)w / (2r)",
    "Shared predictive scale r",
],body_size=26,title_size=28)
box(-4.2,3.35,7.15,2.46,"7. Delayed calibration of all issued forks",[
    "Save forecast, scale, issued q and model versions",
    "Include rejected candidates; wait for complete labels",
    "R = (ĝ − g_true) / s",
    "q_new = max{0, q + γ(I{R > q_issued} − α)}",
    "Updated q applies to subsequent decisions",
],fill="paleGreen",body_size=26.5,title_size=29)
box(4.2,3.35,7.15,2.46,"4. Corrected value and deployment gate",[
    "ĝ = N w^T(h − μ)",
    "s = N √(w^T U w + 10⁻⁴)     (raw U)",
    "L = ĝ − 5 − qs;     admit when L > ε",
    "Admit only a feasible, complete lease",
],fill="paleAmber",body_size=28,title_size=29)
box(-5.5,.4,4.6,1.9,"Exogenous delayed labels",[
    "Arrive after forecast issuance",
    "Available regardless of admission",
    "Audit/lease maturity are separate",
],fill="paleGreen",body_size=25.5,title_size=28)
box(0,.4,4.6,1.9,"5. Paid lease or keep reference",[
    "Admit: H = 4 fixed-model windows",
    "Restore reference at closure",
    "Reject: retain shared reference",
],body_size=25.5,title_size=26.5)
box(5.5,.4,4.6,1.9,"6. Complete return audit",[
    "After complete outcome labels mature",
    "Served correctness minus all fees",
    "D = g_true − ℓ_cand − 3",
    "J − J_ref = ∑ a_j D_j",
],fill="paleGreen",body_size=25.5,title_size=28)

arrow([(-5.5,7.84),(-5.5,7.59)])
arrow([(0,7.84),(0,7.59)])
arrow([(-3.2,6.45),(-2.3,6.45)])
arrow([(2.3,6.45),(3.2,6.45)],"blue")
arrow([(5.5,5.31),(5.5,4.58)],"blue",width=3)
label(5.76,4.96,"w",24,True,"blue")
arrow([(0,5.31),(0,4.95),(3.0,4.95),(3.0,4.58)],"amber",width=3)
label(1.45,5.13,"μ, raw U",23,True,"amber")
arrow([(8.05,8.4),(8.32,8.4),(8.32,3.35),(7.775,3.35)])
label(8.32,5.3,"Current h",24,False,rotation=90)
arrow([(.625,3.9),(-.625,3.9)])
label(0,4.14,"issued",21)
arrow([(-.625,2.92),(.625,2.92)],"green",True)
label(0,2.66,"next q",21,True,"green")
arrow([(4.2,2.12),(4.2,1.65),(0,1.65),(0,1.35)])
label(2.4,1.86,"admit / retain",23)
arrow([(2.3,.4),(3.2,.4)])
arrow([(-5.5,1.35),(-5.5,2.12)],"green",True)
arrow([(-7.8,.4),(-8.32,.4),(-8.32,6.45),(-7.8,6.45)],"green",True)
label(-8.32,4.0,"Arrived audit labels",24,False,"green",rotation=90)
arrow([(-5.5,-.55),(-5.5,-.85),(5.5,-.85),(5.5,-.55)],"green",True)
label(0,-.67,"Matured lease outcomes",23,True,"green")
label(0,-1.3,"Solid: issued information and executed decisions     |     Dashed: completed delayed feedback",24)

svg.append("</svg>")
OUT.joinpath("bayesian_fusion_overview.svg").write_text("\n".join(svg))
canvas.save(OUT/"bayesian_fusion_overview.png",dpi=(300,300))
import json
OUT.joinpath("bayesian_fusion_overview_layout_audit.json").write_text(
    json.dumps({"png_size":canvas.size,"svg_viewbox":[WIDTH,HEIGHT],"text_fit":checks,
                "future_labels_enter_current_gate":False,
                "calibration_includes_rejected_forks":True},indent=2)
)
print("Saved SVG, high-resolution PNG, and text-fit audit.")
