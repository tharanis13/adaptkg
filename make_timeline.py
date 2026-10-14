from PIL import Image, ImageDraw, ImageFont

W, H = 1400, 820
img = Image.new('RGB', (W, H), '#f3f3f3')
d = ImageDraw.Draw(img)

header_blue = '#c9d9e8'
cell_fill = '#f5f5f5'
line = '#2a2a2a'
text = '#1a1a1a'
months = ['June', 'July', 'August', 'September', 'October']
phases = [
    'Planning & Problem Identification',
    'Literature Study & Requirement Analysis',
    'Dataset Collection (MetaQA)',
    'Data Preprocessing',
    'Entity Detection & Feature Engineering',
    'Hop Classifier Development',
    'Adaptive Retrieval & Graph Traversal',
    'Triple Ranking & Answer Generation',
    'Backend API Development',
    'Frontend Dashboard Development',
    'Testing & Performance Evaluation',
    'Final Improvements & Integration',
    'Report Preparation & Submission',
    'Final Viva',
]

# Try common fonts; fallback to default if not present.
font_candidates = [
    'C:/Windows/Fonts/calibri.ttf',
    'C:/Windows/Fonts/arial.ttf',
    'C:/Windows/Fonts/tahoma.ttf',
]

def load_font(size):
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()

header_font = load_font(30)
small_font = load_font(17)
label_font = load_font(15)

# Title box
header_box = (300, 15, 1100, 65)
d.rounded_rectangle(header_box, radius=8, fill=header_blue, outline='#333333', width=2)
d.text((520, 20), 'PROJECT TIMELINE', fill='#111111', font=header_font)

# Layout
x0, y0 = 60, 90
label_w = 300
col_w = 200
row_h = 42

# Column headers
for i, m in enumerate(months):
    xx = x0 + label_w + i * col_w + 10
    d.rounded_rectangle((xx, y0, xx + col_w - 10, y0 + 38), radius=2, outline=line, width=1, fill='#e7e7e7')
    d.text((xx + 18, y0 + 8), m, fill=text, font=label_font)

# Left-side phase labels and grid
for i, phase in enumerate(phases):
    yy = y0 + 40 + i * row_h
    d.rounded_rectangle((x0, yy, x0 + label_w, yy + row_h - 2), radius=2, outline=line, width=1, fill=cell_fill)
    d.text((x0 + 10, yy + 8), f'{i + 1}. {phase}', fill=text, font=small_font)
    for j in range(len(months)):
        xx = x0 + label_w + j * col_w + 10
        d.rounded_rectangle((xx, yy, xx + col_w - 10, yy + row_h - 2), radius=2, outline='#b5b5b5', width=1, fill='#f8f8f8')

# Activity bars by row
bar_defs = [
    (0, 0, 1, '#f1b9cb'),
    (0, 0, 1, '#f1b9cb'),
    (1, 0, 2, '#f2c199'),
    (1, 0, 2, '#e9d9a8'),
    (2, 0, 2, '#b8d7b6'),
    (2, 0, 3, '#c1dff4'),
    (2, 0, 3, '#d4c6ea'),
    (2, 0, 3, '#f0b7d9'),
    (3, 0, 4, '#bce0f7'),
    (3, 0, 4, '#d8c8ff'),
    (4, 0, 4, '#f6d6a9'),
    (4, 0, 4, '#d8d8d8'),
    (3, 0, 4, '#d8bcb7'),
    (4, 0, 4, '#d9b3ff'),
]

for idx, phase in enumerate(phases):
    yy = y0 + 40 + idx * row_h + 6
    start_m, start_offset, end_m, color = bar_defs[idx]
    x_start = x0 + label_w + start_m * col_w + 18 + start_offset * 20
    x_end = x0 + label_w + (end_m + 1) * col_w + 4
    # For the final row, place a shorter bar near the end.
    if idx == len(phases) - 1:
        x_start = x0 + label_w + 4 * col_w + 20
        x_end = x0 + label_w + 5 * col_w + 20
    if idx == 0:
        x_start = x0 + label_w + 10
        x_end = x0 + label_w + 2 * col_w - 5
    if x_end > x0 + label_w + len(months) * col_w:
        x_end = x0 + label_w + len(months) * col_w - 8
    d.rounded_rectangle((x_start, yy, x_end, yy + 20), radius=8, fill=color, outline='#444444', width=1)

# Bottom legend row (matching sample style)
legend_y = 700
legend_x = 120
legend_labels = ['Project Work', 'Project Development', 'Implementation & Testing', 'Report Preparation & Submission', 'Final Viva']
legend_colors = ['#f0b7c9', '#d9b3ff', '#d7e9f7', '#c9d7a8', '#c4b3e6']
for i, label in enumerate(legend_labels):
    x = legend_x + i * 210
    d.rounded_rectangle((x, legend_y, x + 150, legend_y + 25), radius=6, fill=legend_colors[i], outline='#333333', width=1)
    d.text((x + 8, legend_y + 5), label, fill='#111111', font=label_font)

out = r'c:\sem7pro\project_timeline.png'
img.save(out)
print(f'Created: {out}')
