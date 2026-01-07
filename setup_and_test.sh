#!/bin/bash
# Script para configurar ambiente e testar YourTTS Lightning com LJSpeech

set -e  # Parar em caso de erro

echo "================================================================================"
echo "YourTTS Lightning - Setup e Teste com LJSpeech"
echo "================================================================================"

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar se LJSpeech existe
LJSPEECH_PATH="/home/fred/Projetos/DATASETS/LJSpeech-1.1"
if [ ! -d "$LJSPEECH_PATH" ]; then
    echo -e "${RED}✗ LJSpeech não encontrado em: $LJSPEECH_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✓ LJSpeech encontrado em: $LJSPEECH_PATH${NC}"

# Verificar se conda está instalado
if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗ Conda não encontrado. Instale o Miniconda ou Anaconda primeiro.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Conda encontrado${NC}"

# Criar ambiente conda
echo ""
echo "================================================================================"
echo "1. Criando ambiente conda 'yourtts'"
echo "================================================================================"

# Remover ambiente existente se houver
conda env remove -n yourtts -y 2>/dev/null || true

# Criar novo ambiente
conda create -n yourtts python=3.9 pip -y

echo -e "${GREEN}✓ Ambiente conda criado${NC}"

# Ativar ambiente
echo ""
echo "================================================================================"
echo "2. Ativando ambiente e instalando dependências"
echo "================================================================================"

# Usar source para ativar (funciona melhor em scripts)
eval "$(conda shell.bash hook)"
conda activate yourtts

echo -e "${GREEN}✓ Ambiente ativado: $(which python)${NC}"
echo "Python version: $(python --version)"

# Instalar PyTorch
echo ""
echo "Instalando PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Instalar dependências do TTS
echo ""
echo "Instalando dependências do TTS..."
pip install -r requirements.txt

# Instalar TTS em modo desenvolvimento
echo ""
echo "Instalando TTS em modo desenvolvimento..."
pip install -e .

echo -e "${GREEN}✓ Todas as dependências instaladas${NC}"

# Verificar instalação
echo ""
echo "================================================================================"
echo "3. Verificando instalação"
echo "================================================================================"

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pytorch_lightning as pl; print(f'Lightning: {pl.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo -e "${GREEN}✓ Verificação concluída${NC}"

# Executar testes
echo ""
echo "================================================================================"
echo "4. Executando testes automatizados"
echo "================================================================================"

python test_training.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Todos os testes passaram!${NC}"
else
    echo -e "${RED}✗ Alguns testes falharam${NC}"
    exit 1
fi

# Teste rápido com LJSpeech
echo ""
echo "================================================================================"
echo "5. Teste rápido com LJSpeech (modo debug - 5 batches)"
echo "================================================================================"

python TTS/bin/train_yourtts_lightning.py \
    --config_path ljspeech_test_config.json \
    --debug

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Teste com LJSpeech bem-sucedido!${NC}"
else
    echo -e "${RED}✗ Teste com LJSpeech falhou${NC}"
    exit 1
fi

# Resumo final
echo ""
echo "================================================================================"
echo "✅ SETUP COMPLETO!"
echo "================================================================================"
echo ""
echo "O ambiente 'yourtts' foi criado e testado com sucesso!"
echo ""
echo "Para usar:"
echo "  1. Ativar ambiente:"
echo "     conda activate yourtts"
echo ""
echo "  2. Treinar com LJSpeech (teste - 10 épocas):"
echo "     python TTS/bin/train_yourtts_lightning.py \\"
echo "         --config_path ljspeech_test_config.json \\"
echo "         --gpus 1 \\"
echo "         --precision 16-mixed"
echo ""
echo "  3. Monitorar com TensorBoard:"
echo "     tensorboard --logdir outputs/"
echo ""
echo "  4. Para treinamento completo, edite ljspeech_test_config.json:"
echo "     - Aumentar 'epochs' para 1000+"
echo "     - Ajustar 'batch_size' conforme sua GPU"
echo ""
echo "================================================================================"
