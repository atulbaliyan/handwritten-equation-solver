from PIL import Image

from ocr_engine import extract_best_expression
from solver import solve_math

IMG = "/home/atul-baliyan/Downloads/p0cywtsg.png"

img = Image.open(IMG).convert("RGB")
best, cands = extract_best_expression(img)
print("BEST:", best)
print("CANDIDATES:", cands[:20])

res = solve_math(best)
print("MODE:", res.mode)
print("NORMALIZED:", res.normalized)
print("SOLUTIONS:", res.solutions)
