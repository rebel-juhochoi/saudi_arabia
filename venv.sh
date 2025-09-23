uv venv -p 3.12 --clear
source .venv/bin/activate

echo "installing requirements"
uv pip install -r requirements.txt  
echo "requirements installed"

echo "installing rebel-compiler"
uv pip install -i https://pypi.rbln.ai/simple/ rebel-compiler==0.8.3
echo "rebel-compiler installed"

echo "venv created"
