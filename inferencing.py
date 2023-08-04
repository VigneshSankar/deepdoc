from pathlib import Path
import deepdoctection as dd
import os

dd.print_model_infos(add_description=False,add_config=False,add_categories=False)

categories=dd.ModelCatalog.get_profile("doctr/db_resnet50/pt/db_resnet50-ac60cadc.pt").categories
path_weights=dd.ModelDownloadManager.maybe_download_weights_and_configs("doctr/db_resnet50/pt/db_resnet50-ac60cadc.pt")

#Doctr

categories_tr=dd.ModelCatalog.get_profile("doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt").categories
print(dd.ModelCatalog.get_profile("doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt").model_wrapper)
path_weights_tr=dd.ModelDownloadManager.maybe_download_weights_and_configs("doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt")

#Tesseract
# tess_ocr_config_path = dd.get_configs_dir_path() / "dd/conf_tesseract.yaml"  # This file will be in your .cache if you ran the analyzer before.
# # Otherwise make sure to copy the file from 'configs/conf_tesseract.yaml'
# tesseract_ocr = dd.TesseractOcrDetector(tess_ocr_config_path.as_posix())

b=1
text_line_predictor = dd.DoctrTextlineDetector("db_resnet50", path_weights, categories)
layout = dd.ImageLayoutService(text_line_predictor,
                            to_image=True)     # ImageAnnotation created from this service will get a nested image
                                                # defined by the bounding boxes of its annotation. This is helpful
                                                # if you want to call a service only on the region of the
                                                # ImageAnnotation

text_recognizer = dd.DoctrTextRecognizer("crnn_vgg16_bn", path_weights_tr)
text = dd.TextExtractionService(text_recognizer, extract_from_roi="word") # text recognition on the region of word
                                                                        # ImageAnnotation
analyzer = dd.DoctectionPipe(pipeline_component_list=[layout, text])      # defining the pipeline


file_path = "/home/vignesh/pathsetter/ocr/LARES-20230730T005059Z-001/LARES/Good/T-40411/T-40411b.pdf"

df = analyzer.analyze(path=file_path)
df.reset_state()
dp = next(iter(df))
print(dp)
a=1