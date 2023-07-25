'''

conda create --prefix env python = 3.8 -y
conda activate env
python install setup.py <!--to treat local folder as a pkg-->
pip install -r requirements.txt
pip install ipykernel

'''
