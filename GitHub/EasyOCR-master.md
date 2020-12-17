https://github.com/JaidedAI/EasyOCR
pip3 install easyocr

> import easyocr
> reader = easyocr.Reader(['ch_sim','en'])
> result = reader.readtext('chinese.jpg')