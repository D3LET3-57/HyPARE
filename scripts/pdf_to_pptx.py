#!/usr/bin/env python3
"""Convert a rendered deck PDF into a PPTX and embed the inference GIF."""

import argparse
import shutil
import tempfile
from pathlib import Path

import pymupdf
from pptx import Presentation
from pptx.util import Inches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--gif", type=Path, required=True)
    parser.add_argument("--examples-gif", type=Path, default=None)
    parser.add_argument("--ood-gif", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--live-text", default="Live Inference Demo")
    parser.add_argument("--examples-text", default="More DFS Examples")
    parser.add_argument("--ood-text", default="OOD Prediction")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(args.pdf)
    slide_width = Inches(13.333)
    slide_height = Inches(7.5)
    page_width = doc[0].rect.width
    page_height = doc[0].rect.height
    scale_x = slide_width / page_width
    scale_y = slide_height / page_height

    presentation = Presentation()
    presentation.slide_width = slide_width
    presentation.slide_height = slide_height
    blank = presentation.slide_layouts[6]

    temporary = Path(tempfile.mkdtemp(prefix="pptx_pages_"))
    try:
        for index, page in enumerate(doc):
            image_path = temporary / f"page_{index + 1:02d}.png"
            page.get_pixmap(dpi=args.dpi).save(image_path)
            slide = presentation.slides.add_slide(blank)
            slide.shapes.add_picture(
                str(image_path), 0, 0, width=slide_width, height=slide_height
            )

            page_text = page.get_text()
            embedded_gif = None
            note = None
            if args.live_text in page_text:
                embedded_gif = args.gif
                note = "Animated GIF: play this slide in Slide Show mode."
            elif args.examples_gif and args.examples_text in page_text:
                embedded_gif = args.examples_gif
                note = "Animated GIF cycles through three examples; play in Slide Show mode."
            elif args.ood_gif and args.ood_text in page_text:
                embedded_gif = args.ood_gif
                note = "OOD prediction GIF: play in Slide Show mode."
            if embedded_gif is None:
                continue

            rects = []
            for image in page.get_images(full=True):
                rects.extend(page.get_image_rects(image[0]))
            if not rects:
                continue
            image_rect = max(rects, key=lambda rect: rect.width * rect.height)
            left = int(image_rect.x0 * scale_x)
            top = int(image_rect.y0 * scale_y)
            width = int(image_rect.width * scale_x)
            height = int(image_rect.height * scale_y)
            slide.shapes.add_picture(
                str(embedded_gif), left, top, width=width, height=height
            )
            slide.notes_slide.notes_text_frame.text = note
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    presentation.save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
