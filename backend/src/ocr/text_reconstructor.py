from typing import List, Dict, Optional


def reconstruct_text(ocr_blocks: List[Dict]) -> str:
    """
    Join OCR blocks into readable text in reading order.

    Blocks must contain x0, y0, x1, y1 keys (preserved by run_ocr).
    Blocks are already spatially sorted by run_ocr() — this function
    additionally groups them into lines by vertical proximity and
    inserts paragraph breaks for large vertical gaps.

    Teaching note:
        OCR returns blocks in the order the model found them — not
        necessarily top-to-bottom. This sort ensures multi-column and
        header/footer text ends up in the right reading position.
    """
    if not ocr_blocks:
        return ""

    # Blocks should already be sorted by run_ocr(), but re-sort defensively
    # Bucket rows with 20px tolerance for minor slant/misalignment
    sorted_blocks = sorted(ocr_blocks, key=lambda b: (round(b.get("y0", 0) / 20) * 20, b.get("x0", 0)))

    lines: List[str] = []
    current_line_words: List[str] = []
    current_line_y: Optional[float] = None
    prev_line_y: Optional[float] = None

    for block in sorted_blocks:
        y_center = (block.get("y0", 0) + block.get("y1", 0)) / 2

        if current_line_y is None:
            current_line_y = y_center
            current_line_words.append(block["text"])
        elif abs(y_center - current_line_y) < 15:
            # Same line — words are within 15px vertically
            current_line_words.append(block["text"])
        else:
            # New line — flush current line
            completed_line = " ".join(current_line_words)
            lines.append(completed_line)

            # Large vertical gap (>40px) = paragraph break
            if prev_line_y is not None and (y_center - prev_line_y) > 40:
                lines.append("")  # blank line between paragraphs

            prev_line_y = current_line_y
            current_line_y = y_center
            current_line_words = [block["text"]]

    # Flush last line
    if current_line_words:
        lines.append(" ".join(current_line_words))

    return "\n".join(lines)

