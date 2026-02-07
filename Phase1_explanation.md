# OCR Pipeline (Phase 1)

Here we will explain everything line by line in a systematic manner.

#Start with Pipeline.py

It imports a bunch of stuff, lets take a look.

1. dataclass : it is basically a class only , they automatically generate boiler plate for functions like __inti__, __repr__ .....
2. Path: Path represents a filesystem path , also offers
methods to do system calls on path objects. 
3. Optional, Dict, List: Dict and List we know
Optional :- Optional[X] is equivalent to Union[X, None]. We can specify a type or multiple type or no type at all.
4. hashlib: we use to make a 256 bit hash value for our files names.

Then we start by importing our files, the files will we explained in the process.


Then we initialise a data class named page.
## Page
it has 
1. page number
2. image path: the path where its image is stored, it is of the form
    data-> temp -> doc_name_hash->pages.
3. preprocessed path: the path where the pre-processed images are stored.
4. Layout: 
5. ocr_blocks:
6. text : extracted text will be stored here and can be referenced.
7. text_layer : 
8. Confidence :
9. failed : 
10. error : 

Now we make a class named document
##  Document
it has 
1. id : the hashed id
2. input_path : the path of the doc, generally in raw
3. work_dir : the temporary directory to store the info about the docs
4. self.pages: it is a dictionary, 
5. meta : 
6. status : 

Now the actual work start, this is the main OCRPipeline.

### __init__
1. use_layout:
2. use_postprocess:
3. has_gpu:

### run 
This function turn the raw doc into an object of document class.

1. path : the path of the raw doc
2. _setup_document : this is a function which return the doc as an object of document class with attributes path, work_dir(the temp one) and the doc id.
3. _extract_pages : we pass the doc and it extract pages and store it inside the temp directory.
4. _preprocess :
5. _find_layout :
6. _ocr :
7. _rebuild_text :
8. _cleanup :

### check GPU
This function return true or false to the __inti__ section of the main pipeline,based on if gpu is available

### _doc_hash
Converts the doc in an 256 bit id and return the last 16 bit.

## Stages 
### _setup_document
As the name suggest, it sets up to document to be processed in a much smoother way.
It checks if the file path is valid, checks if it is supported or not, createst the temp working directory for the particular doc using its 16 bit hashed code and then finally return the doc object to the run function where it stores this in a variable name doc.

### _extract_pages
This function is used to extract pages from the pdf. It calls the pdf_to _image function in the src module. 

The pdf_to_image fun:
first validates the paths ans the file types and if it is readable or not. It uses pdftoppm to convert each doc into a list of pages.
It accepts pdf_path which is the input path of the doc, the actual path converted into a Path type value.
It also accepts an output_path which is the pages_dir inside the temp->doc_id directory. It stores the images here in png format. one page = one image.

It runs a command in which is written inside cmd, stores the log in pdftoppm.log file for debugging.

It returns a list of pages to the _extract_pages function.

Then this function iterates over this list and converts every page into a page object.

### _extract_text_layer
This fun accepts the doc object as input and return nothing.
It tries to extract text layer using the function extract_page_text present inside the src module.
It then modify the attributes of pages of the doc based on the text.

extract_page_text fun:
it accepts the doc input path and page_no as input and return Optional[str].
it uses fitz or (PyMuPDF). Here we check if the page contains sufficient characters to be classified for skipping ocr.

### _preprocess 
This fun is called by run to preprocess the doc. It creates a new output dir named preprocessed inside temp->doc_id.

It prepares to pages for ocr.

For each page in doc , if hte page has a text layer or it had failed previously(probably in previous runs) then just continue, no need to preprocess it. It stores the page with the same name as in pages directory.

it calls preprocess_image fun which accepts the page.image_path and out and returns a path as output path.

img = cv2.imread(str(input_path),cv2.IMREAD_GRAYSCALE) : reads the image from disk, and converts it into black and white which is standard for ocr.

it finaly saves the image in the given output directory and retuns the path of the imagem which is stored in page.processed path.

### _find_layout
Inside the run function, if use_layout is true, then only this fun is called.
This function accepts the whole doc as imput and detects layout on each page.
It iterates over all the pages, it the page had failed or it contains text or its processed path field is blank , we skip that and continue with other pages.

It calls detect_layout function from the src module.
It accepts a input which is the processed_image_path of the page. It returns a list of dictionaries which contains bounding boxes, which is stored in page.layout.
Here the detector is paddleocr.

### _ocr
It is called by run fun for every doc.
If the page has failed anytime or the page has a text layer or it does nothave the processed path then we skip those pages.

This fun called run_ocr which accepts preprocessed path and page.layout as input.

run_ocr : 
here the engine is paddleocr, we check for both languages hindi and english.

runs ocr using paddleocr and stores the results, it then return a list of dict of text and confidence which is stored in page.ocr_blocks.







