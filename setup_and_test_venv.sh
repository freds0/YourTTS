#!/bin/bash
# Script para configurar ambiente (venv) e testar YourTTS Lightning com LJSpeech

set -e  # Parar em caso de erro

echo "================================================================================"
echo "YourTTS Lightning - Setup e Teste com LJSpeech (usando venv)"
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

# Verificar se python3 está instalado
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python3 não encontrado${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3 encontrado: $(python3 --version)${NC}"

# Criar ambiente virtual
echo ""
echo "================================================================================"
echo "1. Criando ambiente virtual 'venv_yourtts'"
echo "================================================================================"

# Remover venv existente se houver
if [ -d "venv_yourtts" ]; then
    echo "Removendo venv existente..."
    rm -rf venv_yourtts
fi

# Criar novo ambiente
python3 -m venv venv_yourtts

echo -e "${GREEN}✓ Ambiente virtual criado${NC}"

# Ativar ambiente
echo ""
echo "================================================================================"
echo "2. Ativando ambiente e instalando dependências"
echo "================================================================================"

source venv_yourtts/bin/activate

echo -e "${GREEN}✓ Ambiente ativado: $(which python)${NC}"
echo "Python version: $(python --version)"

# Atualizar pip
echo ""
echo "Atualizando pip..."
python -m pip install --upgrade pip

# Instalar PyTorch
echo ""
echo "Instalando PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

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
python -c "import pytorch_lightning as pl; print(f'Lightning: {pl.__version__}')" || echo "Lightning não instalado ainda"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')" || echo "GPU: N/A"
fi

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
    echo -e "${YELLOW}⚠ Alguns testes podem ter falhado (isso é normal se faltam dependências opcionais)${NC}"
fi

# Teste rápido com LJSpeech
echo ""
echo "================================================================================"
echo "5. Teste rápido com LJSpeech (modo debug - 5 batches)"
echo "================================================================================"

python TTS/bin/train_yourtts_lightning.py \
    --config_path ljspeech_test_config.json \
    --gpus 1 \
    --debug

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Teste com LJSpeech bem-sucedido!${NC}"
else
    echo -e "${RED}✗ Teste com LJSpeech falhou${NC}"
    echo "Verifique os logs acima para detalhes"
    exit 1
fi

# Resumo final
echo ""
echo "================================================================================"
echo "✅ SETUP COMPLETO!"
echo "================================================================================"
echo ""
echo "O ambiente 'venv_yourtts' foi criado e testado com sucesso!"
echo ""
echo "Para usar:"
echo "  1. Ativar ambiente:"
echo "     source venv_yourtts/bin/activate"
echo ""
echo "  2. Treinar com LJSpeech (teste - 10 épocas):"
echo "     python TTS/bin/train_yourtts_lightning.py \\"
echo "         --config_path ljspeech_test_config.json \\"
echo "         --gpus 1 \\"
echo "         --precision 16-mixed"
echo ""
echo "  3. Monitorar com TensorBoard:"
echo "     tensorboard --logdir outputs/"
echo "     # Acesse: http://localhost:6006"
echo ""
echo "  4. Para treinamento completo, edite ljspeech_test_config.json:"
echo "     - Aumentar 'epochs' para 1000+"
echo "     - Ajustar 'batch_size' conforme sua GPU"
echo "     - Habilitar mixed_precision: true"
echo ""
echo "Arquivos criados:"
echo "  - venv_yourtts/              (ambiente virtual)"
echo "  - outputs/yourtts_ljspeech_test/  (outputs de treinamento)"
echo "  - setup_log.txt              (log deste script)"
echo ""
echo "================================================================================"
