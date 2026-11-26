#!/usr/bin/env python3
"""Render KAM's original vector-inspired hero and narrated attention explainer."""

import argparse
from functools import lru_cache
from math import exp, sqrt
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont

SIZE = 1280, 720
BG = "#0c1425"
PANEL = "#131f35"
INK = "#f1f6ff"
SUB = "#b5c3d8"
MINT = "#75efd0"
BORDER = "#304059"
KEYS = [(-1.55, .85, "#ffb19a"), (-.75, -1.05, "#fed78c"), (.95, .8, "#ada8fb"), (1.65, -.47, "#87beee")]
CHAPTERS = (
    (7, "Move the query", "Move the query and its new neighbors carry more of the attention."),
    (7, "Narrow the bandwidth", "A smaller bandwidth makes the same differences in distance count more."),
    (8, "Change the metric", "A diagonal metric can make one direction matter more than another."),
    (8, "Two sources of memory", "KAM can retrieve from keys in the current context and a separate bank of learned supports."),
)


@lru_cache(maxsize=None)
def font(size, bold=False, mono=False):
    family = "DejaVu Sans Mono" if mono else "DejaVu Sans"
    name = findfont(FontProperties(family=family, weight="bold" if bold else "normal"))
    return ImageFont.truetype(name, size)


def text(draw, position, value, size, color=INK, bold=False, mono=False):
    draw.text(position, value, fill=color, font=font(size, bold, mono), anchor="lt")


def geometry(draw, q, sigma, metric, region):
    x0, y0, x1, y1 = region
    draw.rounded_rectangle(region, radius=24, fill="#0e192c", outline=BORDER, width=2)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    scale = min((x1-x0)/5.4, (y1-y0)/3.6)

    def project(x, y):
        return cx+x*scale, cy-y*scale

    for i in range(-5, 6):
        x, _ = project(i/2, 0)
        draw.line((x, y0+17, x, y1-17), fill="#26354c")
    for i in range(-3, 4):
        _, y = project(0, i/2)
        draw.line((x0+17, y, x1-17, y), fill="#26354c")
    px, py = project(*q)
    for factor in (1.65, 1.23, .81, .4):
        rx, ry = factor*sigma*scale/sqrt(metric), factor*sigma*scale
        draw.ellipse((px-rx,py-ry,px+rx,py+ry), outline="#39786f", width=2)
    scores = [-(metric*(x-q[0])**2+(y-q[1])**2)/(2*sigma**2) for x,y,_ in KEYS]
    transformed = [exp(s-max(scores)) for s in scores]
    weights = [v/sum(transformed) for v in transformed]
    for i, ((x,y,color), weight) in enumerate(zip(KEYS, weights, strict=True)):
        nx, ny = project(x, y)
        draw.line((px,py,nx,ny),fill="#354760",width=2)
        if weight > .08:
            draw.line((px,py,nx,ny),fill=color,width=1+round(5*weight))
        radius = 7 + 4*weight
        draw.ellipse((nx-radius,ny-radius,nx+radius,ny+radius),fill=color)
        text(draw, (nx+15,ny-20), f"K{i+1}", 17, color, True, True)
    draw.ellipse((px-10,py-10,px+10,py+10),fill=MINT,outline=BG,width=3)
    text(draw,(px+17,py+11),"q",20,MINT,True,True)
    return weights


def hero(out):
    image = Image.new("RGB", (1280,490), BG)
    d = ImageDraw.Draw(image)
    d.rectangle((0,486,1280,490),fill=MINT)
    text(d,(58,52),"KAM  /  RESEARCH",20,MINT,True,True)
    text(d,(58,119),"Kernel Adaptive",62,bold=True)
    text(d,(58,189),"Memory",79,bold=True)
    text(d,(62,303),"How should a model search its memory?",25,SUB)
    text(d,(62,391),"Radial attention  ·  Learned supports",20,MINT)
    geometry(d, (.25,.06), .88, 1.8, (708,44,1238,441))
    image.save(out)


def support_map(draw):
    cards = ((65, 232, 514, 387, "CONTEXT ATTENTION", "Keys come from the input", "#ffb19a"),
             (65, 415, 514, 570, "SUPPORT ATTENTION", "Keys and values are learned", MINT))
    for x0,y0,x1,y1,title,body,color in cards:
        draw.rounded_rectangle((x0,y0,x1,y1),radius=20,fill=PANEL,outline=BORDER,width=2)
        text(draw,(x0+30,y0+38),title,22,color,True,True)
        text(draw,(x0+30,y0+87),body,24)
    draw.line((274,390,274,403),fill="#ffb19a",width=4)
    draw.polygon(((266,402),(282,402),(274,413)),fill="#ffb19a")
    draw.line((531,493,618,493,618,392,679,392),fill=MINT,width=4)
    draw.polygon(((678,383),(697,392),(678,401)),fill=MINT)
    draw.rounded_rectangle((712,316,1216,475),radius=22,fill=PANEL,outline=BORDER,width=2)
    text(draw,(746,352),"Update the token state",30,bold=True)
    text(draw,(746,412),"Each read can add a residual update.",23,SUB)


def frame(chapter, progress):
    image = Image.new("RGB", SIZE, BG)
    d = ImageDraw.Draw(image)
    title, caption = CHAPTERS[chapter][1:]
    text(d,(65,42),f"KAM  /  ATTENTION GEOMETRY   {chapter+1} OF {len(CHAPTERS)}",20,MINT,True,True)
    text(d,(65,91),title,49,bold=True)
    if chapter == 3:
        support_map(d)
    else:
        u = min(1,max(0,(progress-.12)/.72))
        t = u*u*(3-2*u)
        q = (-.9 + 1.35*t, .16 + .22*t) if chapter == 0 else (.45,.38)
        sigma = .95-.53*t if chapter == 1 else (.95 if chapter == 0 else .42)
        metric = 1+2.3*t if chapter == 2 else 1.0
        weights = geometry(d,q,sigma,metric,(65,177,802,588))
        d.rounded_rectangle((823,177,1215,588),radius=24,fill=PANEL,outline=BORDER,width=2)
        text(d,(855,208),"SOFTMAX WEIGHTS",20,MINT,True,True)
        for i, ((_,_,color),weight) in enumerate(zip(KEYS,weights,strict=True)):
            y = 259 + 53*i
            text(d,(854,y),f"K{i+1}",19,color,True,True)
            d.rounded_rectangle((898,y+3,1105,y+15),radius=6,fill="#2a3a52")
            d.rounded_rectangle((898,y+3,898+round(207*weight),y+15),radius=6,fill=color)
            text(d,(1117,y),f"{weight:5.1%}",17,INK,False,True)
        text(d,(855,501),f"σ {sigma:.2f}     horizontal weight {metric:.2f}",17,SUB,False,True)
    d.rounded_rectangle((64,625,1216,678),radius=13,fill="#142239")
    text(d,(84,640),caption,21)
    return image


def run_ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def render(out, spoken):
    out.mkdir(parents=True,exist_ok=True)
    hero(out/"kam-cover.png")
    frame(3,.6).save(out/"memory-paths.png")
    with TemporaryDirectory() as work:
        temp = Path(work)
        fps = 24
        mp4 = out/"radial-attention.mp4"
        muted = temp/"silent.mp4"
        command = ["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24",
                   "-s","1280x720","-r",str(fps),"-i","-","-c:v","libx264","-preset","fast",
                   "-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(muted)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as child:
            for i,(duration,_,_) in enumerate(CHAPTERS):
                for j in range(duration*fps):
                    child.stdin.write(frame(i,j/(duration*fps-1)).tobytes())
            child.stdin.close()
            if child.wait():
                raise RuntimeError("ffmpeg video encoding failed")
        if spoken:
            for i,(duration,_,line) in enumerate(CHAPTERS):
                clip = temp/f"word-{i}.aiff"
                subprocess.run(["say","-v","Samantha","-r","170","-o",str(clip),line],check=True)
                measured = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(clip), "-af", "volumedetect",
                                           "-f", "null", "-"], capture_output=True, text=True, check=True)
                peak = re.search(r"max_volume: (-?\d+(?:\.\d+)?) dB", measured.stderr)
                if peak is None or float(peak[1]) < -45:
                    raise RuntimeError(f"Narration is silent for chapter {i + 1}")
                run_ffmpeg("-i",str(clip),"-af","adelay=420,apad","-t",str(duration),"-ar","44100",
                           "-ac","1",str(temp/f"sound-{i}.wav"))
            listing = temp/"audio.txt"
            listing.write_text("".join(f"file sound-{i}.wav\n" for i in range(len(CHAPTERS))))
            run_ffmpeg("-i",str(muted),"-f","concat","-safe","0","-i",str(listing),"-c:v","copy","-c:a","aac",
                       "-b:a","96k","-movflags","+faststart",str(mp4))
        else:
            muted.replace(mp4)
        run_ffmpeg("-i",str(mp4),"-t","7","-vf",
                   "fps=11,scale=760:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=80[p];[b][p]paletteuse",
                   "-loop","0",str(out/"attention-preview.gif"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--narrate",action="store_true",help="Use the macOS Samantha voice")
    args=parser.parse_args()
    render(args.output,args.narrate)


if __name__ == "__main__":
    main()
