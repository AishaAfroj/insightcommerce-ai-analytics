"""Generate a polished architecture diagram PNG using Pillow."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "architecture.png"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a stable macOS font for deterministic local rendering."""

    return ImageFont.truetype(FONT_BOLD if bold and Path(FONT_BOLD).exists() else FONT, size)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    detail: str,
    fill: str,
    outline: str,
) -> None:
    """Draw one labeled architecture component."""

    draw.rounded_rectangle(rect, radius=22, fill=fill, outline=outline, width=3)
    x1, y1, x2, y2 = rect
    draw.text((x1 + 22, y1 + 18), title, font=font(26, True), fill="#0F172A")
    draw.multiline_text(
        (x1 + 22, y1 + 58), detail, font=font(18), fill="#334155", spacing=5
    )


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    """Draw a directional connector."""

    draw.line([start, end], fill="#64748B", width=5)
    x, y = end
    sx, sy = start
    if abs(x - sx) >= abs(y - sy):
        points = [(x, y), (x - 14 if x > sx else x + 14, y - 9), (x - 14 if x > sx else x + 14, y + 9)]
    else:
        points = [(x, y), (x - 9, y - 14 if y > sy else y + 14), (x + 9, y - 14 if y > sy else y + 14)]
    draw.polygon(points, fill="#64748B")


def main() -> None:
    """Create the architecture image used in README and the final report."""

    canvas = Image.new("RGB", (1800, 1120), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((55, 45, 1745, 160), radius=26, fill="#0F172A")
    draw.text((95, 70), "InsightCommerce system architecture", font=font(43, True), fill="white")
    draw.text(
        (97, 122),
        "Verified data → safe AI query → interactive decision intelligence",
        font=font(21),
        fill="#BFDBFE",
    )

    rounded_box(draw, (70, 235, 390, 365), "Inputs", "Kaggle CSV\nQuestions + filters", "#DBEAFE", "#60A5FA")
    rounded_box(draw, (70, 470, 390, 600), "Data preparation", "Schema validation\nQuality profile", "#E0F2FE", "#38BDF8")
    rounded_box(draw, (70, 705, 390, 835), "Processed data", "Zstd Parquet\n18 analytics fields", "#CCFBF1", "#2DD4BF")

    rounded_box(draw, (530, 235, 880, 365), "Streamlit UI", "5 sections\nGlobal filters", "#EDE9FE", "#8B5CF6")
    rounded_box(draw, (530, 470, 880, 600), "Schema + memory", "Prompt contract\nLast 5 turns", "#F3E8FF", "#A855F7")
    rounded_box(draw, (530, 705, 880, 835), "LLM planner", "Ollama / OpenAI\nOffline demo fallback", "#FAE8FF", "#D946EF")

    rounded_box(draw, (1020, 235, 1370, 365), "DuckDB", "Parameterized filters\n2.5 ms median", "#DCFCE7", "#22C55E")
    rounded_box(draw, (1020, 470, 1370, 600), "Safety gate", "SELECT-only AST\n5 s / 5,000 rows", "#FEF3C7", "#F59E0B")
    rounded_box(draw, (1020, 705, 1370, 835), "Advanced analytics", "Demand forecast\nIsolation Forest", "#FFEDD5", "#FB923C")

    rounded_box(draw, (1495, 235, 1730, 420), "Insights", "Grounded result\nPreset analysis", "#FFE4E6", "#FB7185")
    rounded_box(draw, (1495, 500, 1730, 685), "Visuals", "8 chart types\nAuto + override", "#FCE7F3", "#F472B6")
    rounded_box(draw, (1495, 765, 1730, 950), "Exports", "PDF · DOCX\nCSV · HTML · PNG", "#E0E7FF", "#818CF8")

    arrow(draw, (390, 300), (530, 300))
    arrow(draw, (230, 365), (230, 470))
    arrow(draw, (230, 600), (230, 705))
    arrow(draw, (390, 770), (1020, 300))
    arrow(draw, (705, 365), (705, 470))
    arrow(draw, (705, 600), (705, 705))
    arrow(draw, (880, 770), (1020, 535))
    arrow(draw, (1195, 470), (1195, 365))
    arrow(draw, (1370, 300), (1495, 325))
    arrow(draw, (1370, 330), (1495, 585))
    arrow(draw, (1370, 770), (1495, 855))

    draw.rounded_rectangle((350, 1010, 1450, 1075), radius=20, fill="#E2E8F0")
    draw.text(
        (405, 1028),
        "Privacy boundary: synthetic name/email fields stay out of AI prompts and general exports",
        font=font(22, True),
        fill="#334155",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, format="PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
