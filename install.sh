pip install -q git+https://github.com/huggingface/transformers.git
pip install -q pyyaml==5.1
pip install -q torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -q 'git+https://github.com/facebookresearch/detectron2.git'
pip install -q datasets
pip install huggingface-hub==0.2.1
sudo apt install tesseract-ocr
pip install pytesseract
sudo apt update && sudo apt install -y poppler-utils
pip install pdf2image
pip install keybert
pip install spacy
pip install pytextrank
sudo apt-get -y install libreoffice