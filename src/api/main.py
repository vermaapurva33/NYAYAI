from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
import shutil
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
import io
import random  # <--- NEW IMPORT

try:
    from src.ocr.pdf_to_images import pdf_to_images
except ImportError:
    print("WARNING: Could not import pdf_to_images.")

app = FastAPI()

@app.post("/detect-mistakes")
async def detect_mistakes(
    file: UploadFile = File(...), 
    page_number: int = Form(...) 
):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        temp_pdf_path = temp_path / "uploaded.pdf"
        output_img_dir = temp_path / "images"
        
        # Save file to disk
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run OCR
        try:
            image_paths = pdf_to_images(temp_pdf_path, output_img_dir)
        except Exception as e:
            return Response(content=f"OCR Error: {str(e)}", status_code=500)

        # Select Page
        page_index = page_number - 1
        if page_index < 0 or page_index >= len(image_paths):
            return Response(content="Page number out of range", status_code=400)
            
        target_image_path = image_paths[page_index]

        # --- DRAWING LOGIC STARTS HERE ---
        with Image.open(target_image_path) as img:
            img = img.convert("RGBA")
            overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            
            width, height = img.size

            # ====================================================
            # CURRENT: RANDOM DEMO MODE
            # ====================================================
            # Randomly decide to show 2 or 3 mistakes
            num_mistakes = random.randint(2, 3)

            for _ in range(num_mistakes):
                # Generate random box coordinates ensuring they fit in the page
                x1 = random.randint(50, width - 200)
                y1 = random.randint(50, height - 100)
                x2 = x1 + random.randint(100, 300) # Random width
                y2 = y1 + random.randint(30, 80)   # Random height (line of text)

                draw.rectangle(
                    [x1, y1, x2, y2], 
                    fill=(255, 0, 0, 60),   # Transparent Red
                    outline=(255, 0, 0, 255), 
                    width=3
                )
            # ====================================================
            #in future to be changed with calling the model api
            # Merge and Save
            final_img = Image.alpha_composite(img, overlay)
            img_byte_arr = io.BytesIO()
            final_img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)