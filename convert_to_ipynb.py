"""
Script untuk mengkonversi 04_colab_notebook.py menjadi 04_colab_notebook.ipynb
"""

import json
import re

# Baca file .py
with open('04_colab_notebook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Pisahkan per cell berdasarkan komentar separator
# Pattern: # ===...=== (baris yang berisi banyak = dan merupakan header cell)
cell_pattern = re.compile(
    r'# =+\n# CELL \d+:.*?\n# =+\n',
    re.MULTILINE
)

# Split berdasarkan header cell
parts = cell_pattern.split(content)
headers = cell_pattern.findall(content)

# Hapus bagian kosong di awal jika ada
cells_raw = [p.strip() for p in parts if p.strip()]

# Buat struktur notebook
def make_code_cell(source_lines):
    """Buat satu code cell dari list baris kode"""
    # Pastikan setiap baris diakhiri \n kecuali baris terakhir
    lines = []
    for i, line in enumerate(source_lines):
        if i < len(source_lines) - 1:
            lines.append(line + '\n')
        else:
            lines.append(line)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

def make_markdown_cell(text):
    """Buat satu markdown cell"""
    lines = text.split('\n')
    source = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            source.append(line + '\n')
        else:
            source.append(line)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

# Definisi cell berdasarkan CELL header
cell_definitions = [
    ("CELL 1: SETUP & IMPORTS", "markdown"),
    ("CELL 2: IMPORTS & CONFIGURATION", "markdown"),
    ("CELL 3: HELPER FUNCTIONS", "markdown"),
    ("CELL 4: DATASET SPLITTING", "markdown"),
    ("CELL 5: CUSTOM DATASET & TRANSFORMS", "markdown"),
    ("CELL 6: CREATE DATASETS & DATALOADERS", "markdown"),
    ("CELL 7: BUILD MODEL", "markdown"),
    ("CELL 8: TRAINING SETUP", "markdown"),
    ("CELL 9: TRAINING LOOP", "markdown"),
    ("CELL 10: LOAD BEST MODEL & EVALUATE", "markdown"),
    ("CELL 11: VISUALIZATIONS", "markdown"),
    ("CELL 12: CONFUSION MATRIX", "markdown"),
    ("CELL 13: SAVE MODEL & CONFIG", "markdown"),
    ("CELL 14: PIPELINE COMPLETE", "markdown"),
    ("CELL 15: PREDICTION PADA GAMBAR BARU (OPTIONAL)", "markdown"),
]

# Rebuild: parse ulang content menjadi cells
# Strategi: split by "# ===...===\n# CELL N: ...\n# ===...==="
full_separator = re.compile(
    r'\n?# =+\n# (CELL \d+[^\n]*)\n# =+\n',
    re.MULTILINE
)

splits = full_separator.split(content)
# splits[0] = konten sebelum cell pertama (biasanya kosong)
# splits[1] = nama cell 1
# splits[2] = isi cell 1
# splits[3] = nama cell 2
# splits[4] = isi cell 2
# dst...

notebook_cells = []

# Iterasi berpasangan (nama, isi)
for i in range(1, len(splits), 2):
    cell_name = splits[i].strip()
    cell_body = splits[i+1].strip() if i+1 < len(splits) else ""

    if not cell_body:
        continue

    # Tambahkan markdown header
    notebook_cells.append(make_markdown_cell(f"## {cell_name}"))

    # Tambahkan code cell
    code_lines = cell_body.split('\n')
    notebook_cells.append(make_code_cell(code_lines))

# Struktur .ipynb lengkap
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "name": "Rice Disease Classification - MobileNetV2"
        }
    },
    "cells": notebook_cells
}

# Simpan sebagai .ipynb
output_path = '04_colab_notebook.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✓ Berhasil dibuat: {output_path}")
print(f"  Total cells: {len(notebook_cells)} (markdown + code)")
