import deepdoctection as dd
from matplotlib import pyplot as plt
import os
os.environ['TESSDATA_PREFIX'] = '/home/vignesh/pathsetter/ocr/LARES-20230730T005059Z-001/'

analyzer = dd.get_dd_analyzer()  # instantiate the built-in analyzer similar to the Hugging Face space demo
analyzer.get_pipeline_info()

df = analyzer.analyze(path="/home/vignesh/pathsetter/ocr/LARES-20230730T005059Z-001/LARES/Good/T-40411/")  # setting up pipeline
df.reset_state()                 # Trigger some initialization

doc = iter(df)
page = next(doc)

image = page.viz()
plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(image)
plt.show()
a=1