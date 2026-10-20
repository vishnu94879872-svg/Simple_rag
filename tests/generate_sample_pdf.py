"""Script to generate a comprehensive sample PDF book with headings, text, tables, code, and images."""

import io
from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw

def create_sample_pdf(output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open()

    # --- Page 1: Chapter 1 - Foundations ---
    page1 = doc.new_page(width=595, height=842) # A4
    page1.insert_text((50, 70), "Chapter 1: Foundations of Quantum Computing", fontsize=18, fontname="helv", color=(0, 0, 0))
    page1.insert_text((50, 110), "Section 1.1: Introduction to Qubits", fontsize=14, fontname="helv", color=(0.1, 0.1, 0.1))
    
    p1_body = (
        "Quantum computing harnesses the phenomena of quantum mechanics to solve complex problems. "
        "A qubit is defined as the fundamental unit of quantum information, analogous to the classical bit. "
        "Unlike classical bits which can only be 0 or 1, a qubit can exist in a superposition of both states. "
        "Superposition allows quantum algorithms to process exponentially large state spaces simultaneously."
    )
    rect1 = pymupdf.Rect(50, 130, 545, 230)
    page1.insert_textbox(rect1, p1_body, fontsize=11, fontname="helv")

    # Add bold key term definition
    p1_bold_def = (
        "Quantum entanglement is defined as a physical phenomenon where pairs or groups of particles interact "
        "in ways such that the quantum state of each particle cannot be described independently."
    )
    rect1_def = pymupdf.Rect(50, 240, 545, 320)
    page1.insert_textbox(rect1_def, p1_bold_def, fontsize=11, fontname="Helvetica-Bold")

    # Add a code block
    code_text = (
        "import numpy as np\n"
        "def hadamard_gate():\n"
        "    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])\n"
        "state = hadamard_gate() @ np.array([1, 0])\n"
        "print('Superposition state:', state)"
    )
    rect_code = pymupdf.Rect(50, 340, 545, 430)
    page1.draw_rect(pymupdf.Rect(45, 335, 550, 435), color=(0.8, 0.8, 0.8), fill=(0.95, 0.95, 0.95))
    page1.insert_textbox(rect_code, code_text, fontsize=10, fontname="Courier")

    # Add a list
    list_text = (
        "Key Properties of Quantum Bits:\n"
        "• State Superposition allows concurrent exploration of solutions.\n"
        "• Quantum Entanglement links qubits regardless of physical distance.\n"
        "• Quantum Interference amplifies correct probability amplitudes."
    )
    rect_list = pymupdf.Rect(50, 450, 545, 550)
    page1.insert_textbox(rect_list, list_text, fontsize=11, fontname="Helvetica")

    # Add an image to Page 1
    img = Image.new("RGB", (300, 150), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 280, 130], outline=(255, 255, 255), width=3)
    d.text((40, 60), "Bloch Sphere Representation", fill=(255, 255, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()
    
    img_rect = pymupdf.Rect(50, 580, 350, 730)
    page1.insert_image(img_rect, stream=img_bytes)

    # --- Page 2: Chapter 2 - Algorithms & Data ---
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 70), "Chapter 2: Quantum Algorithms and Complexity", fontsize=18, fontname="Helvetica-Bold")
    page2.insert_text((50, 110), "Section 2.1: Performance Comparison", fontsize=14, fontname="Helvetica-Bold")

    # Draw Table on Page 2
    # PyMuPDF table detection looks for lines and cell grids
    x_start, y_start = 50, 140
    row_height = 30
    col_widths = [150, 150, 180]
    headers = ["Algorithm", "Classical Complexity", "Quantum Complexity"]
    data = [
        ["Shor's Algorithm", "Exp(O(n^(1/3)))", "Polynomial O(n^3)"],
        ["Grover's Search", "O(N)", "O(sqrt(N))"],
        ["Deutsch-Jozsa", "O(2^(n-1))", "O(1)"]
    ]
    all_rows = [headers] + data

    for row_idx, row in enumerate(all_rows):
        y = y_start + row_idx * row_height
        x = x_start
        for col_idx, cell_text in enumerate(row):
            w = col_widths[col_idx]
            cell_rect = pymupdf.Rect(x, y, x + w, y + row_height)
            page2.draw_rect(cell_rect, color=(0.4, 0.4, 0.4), width=1)
            font_to_use = "Helvetica-Bold" if row_idx == 0 else "Helvetica"
            page2.insert_text((x + 8, y + 20), cell_text, fontsize=10, fontname=font_to_use)
            x += w

    # Paragraph below table
    p2_desc = (
        "As demonstrated in the comparison table, quantum algorithms deliver quadratic to exponential speedups. "
        "Shor's algorithm poses a direct challenge to modern RSA cryptographic infrastructure."
    )
    rect2 = pymupdf.Rect(50, 300, 545, 420)
    page2.insert_textbox(rect2, p2_desc, fontsize=11, fontname="helv")

    # Set Document TOC Outline: [[lvl, title, pagenum]]
    toc = [
        [1, "Chapter 1: Foundations of Quantum Computing", 1],
        [2, "Section 1.1: Introduction to Qubits", 1],
        [1, "Chapter 2: Quantum Algorithms and Complexity", 2],
        [2, "Section 2.1: Performance Comparison", 2]
    ]
    doc.set_toc(toc)
    doc.metadata = {"title": "Quantum Computing Principles", "author": "Dr. Antigravity"}

    doc.save(str(output_path))
    doc.close()
    return output_path

if __name__ == "__main__":
    out = Path("data/books/sample_quantum.pdf")
    create_sample_pdf(out)
    print(f"Sample PDF created at {out}")
