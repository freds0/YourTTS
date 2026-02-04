#!/bin/bash
# Script para preparar o dataset LJSpeech para treinamento com YourTTS Lightning

set -e  # Exit on error

echo "================================================================================"
echo "    YourTTS Lightning - Preparação do Dataset LJSpeech"
echo "================================================================================"
echo

# Configurações
DATASET_PATH="/home/fred/Projetos/DATASETS/LJSpeech-1.1/"
EMBEDDINGS_FILE="${DATASET_PATH}/speakers.pth"
SPEAKER_ENCODER_CHECKPOINT="https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
SPEAKER_ENCODER_CONFIG="https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

echo "📁 Dataset Path: $DATASET_PATH"
echo "📄 Embeddings File: $EMBEDDINGS_FILE"
echo

# Verificar se o dataset existe
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ ERRO: Dataset não encontrado em $DATASET_PATH"
    echo
    echo "Por favor, faça o download do LJSpeech dataset:"
    echo "  wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    echo "  tar -xvjf LJSpeech-1.1.tar.bz2 -C /home/fred/Projetos/DATASETS/"
    echo
    exit 1
fi

echo "✓ Dataset encontrado"
echo

# Verificar se os embeddings já foram computados
if [ -f "$EMBEDDINGS_FILE" ]; then
    echo "✓ Speaker embeddings já existem: $EMBEDDINGS_FILE"
    echo "  Pulando computação de embeddings."
    echo
else
    echo "⏳ Computando speaker embeddings..."
    echo "  Isso pode levar alguns minutos dependendo do tamanho do dataset."
    echo

    # Ativar conda environment
    #source /opt/anaconda3/etc/profile.d/conda.sh
    #conda activate yourtts

    # Computar embeddings
    python TTS/bin/compute_embeddings.py \
        "$SPEAKER_ENCODER_CHECKPOINT" \
        "$SPEAKER_ENCODER_CONFIG" \
        "$EMBEDDINGS_FILE" \
        --dataset_path "$DATASET_PATH" \
        --dataset_name "ljspeech" \
        --formatter_name "ljspeech" \
        --meta_file_train "metadata.csv"

    if [ $? -eq 0 ]; then
        echo
        echo "✓ Speaker embeddings computados com sucesso!"
        echo "  Arquivo salvo em: $EMBEDDINGS_FILE"
    else
        echo
        echo "❌ ERRO ao computar speaker embeddings"
        exit 1
    fi
fi

echo
echo "================================================================================"
echo "✅ Preparação Concluída!"
echo "================================================================================"
echo
echo "Próximos passos:"
echo
echo "1. Verificar configuração:"
echo "   cat config_ljspeech.json"
echo
echo "2. Treinar com Lightning:"
echo "   python train_lightning.py --config config_ljspeech.json --output_dir ./outputs_ljspeech"
echo
echo "3. Visualizar treinamento:"
echo "   tensorboard --logdir outputs_ljspeech/lightning_logs"
echo
echo "================================================================================"
