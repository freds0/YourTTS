# Quick Start - YourTTS Lightning

Guia rápido para começar a usar o YourTTS com PyTorch Lightning.

## 🚀 Setup Rápido (Recomendado)

### Opção 1: Setup Automatizado com venv

```bash
# Execute o script de setup (instala tudo e testa)
./setup_and_test_venv.sh
```

Isso vai:
1. ✅ Criar ambiente virtual Python
2. ✅ Instalar todas as dependências
3. ✅ Executar testes automatizados
4. ✅ Testar treinamento com LJSpeech (5 batches)

**Tempo estimado**: 10-15 minutos (dependendo da conexão)

### Opção 2: Setup Manual

```bash
# 1. Criar ambiente virtual
python3 -m venv venv_yourtts
source venv_yourtts/bin/activate

# 2. Instalar PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Instalar TTS
pip install -e .

# 5. Testar instalação
python test_training.py
```

---

## 📊 Teste com LJSpeech

O dataset LJSpeech já está configurado em:
```
/home/fred/Projetos/DATASETS/LJSpeech-1.1/
```

### Teste Rápido (Debug Mode)

```bash
python TTS/bin/train_yourtts_lightning.py \
    --config_path ljspeech_test_config.json \
    --gpus 1 \
    --debug
```

Isso executa apenas 5 batches para verificar se tudo funciona.

### Treinamento Real (10 épocas para teste)

```bash
python TTS/bin/train_yourtts_lightning.py \
    --config_path ljspeech_test_config.json \
    --gpus 1 \
    --precision 16-mixed
```

### Monitorar com TensorBoard

Em outro terminal:
```bash
source venv_yourtts/bin/activate
tensorboard --logdir outputs/
```

Acesse: http://localhost:6006

---

## 📁 Arquivos Importantes

| Arquivo | Descrição |
|---------|-----------|
| `ljspeech_test_config.json` | Config para teste com LJSpeech |
| `test_training.py` | Testes automatizados |
| `setup_and_test_venv.sh` | Setup completo automatizado |
| `README_YOURTTS.md` | Documentação principal |
| `MIGRATION_GUIDE.md` | Guia de migração |
| `TESTING_GUIDE.md` | Guia de testes detalhado |

---

## ✅ Checklist Pós-Setup

Depois do setup, verifique:

```bash
# Ativar ambiente
source venv_yourtts/bin/activate

# Verificar PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Verificar Lightning
python -c "import pytorch_lightning as pl; print(f'Lightning: {pl.__version__}')"

# Verificar CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Verificar GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Executar testes
python test_training.py
```

Saída esperada:
```
PyTorch: 2.1.0+cu118
Lightning: 2.0.0
CUDA: True
GPU: NVIDIA GeForce RTX 3090 (ou sua GPU)

[Testes...]
🎉 TODOS OS TESTES PASSARAM! 🎉
```

---

## 🎯 Próximos Passos

### 1. Experimente com Diferentes Configurações

Edite `ljspeech_test_config.json`:

```json
{
    "batch_size": 32,           // ← Aumentar se tiver GPU grande
    "epochs": 1000,             // ← Para treinamento completo
    "mixed_precision": true,    // ← Economiza memória
    "lr_gen": 2e-4,             // ← Ajustar learning rate
    "use_phonemes": true        // ← Usar fonemas (melhor qualidade)
}
```

### 2. Multi-GPU

Se você tem múltiplas GPUs:

```bash
python TTS/bin/train_yourtts_lightning.py \
    --config_path ljspeech_test_config.json \
    --gpus 2 \                    # ← Número de GPUs
    --strategy ddp \              # ← Distributed Data Parallel
    --precision 16-mixed
```

### 3. Treinar com Seus Próprios Dados

1. Organize seus dados:
```
meu_dataset/
├── wavs/
│   ├── audio001.wav
│   ├── audio002.wav
│   └── ...
└── metadata.csv
```

2. Formato do metadata.csv:
```
wavs/audio001.wav|Text for audio 1.|speaker_name
wavs/audio002.wav|Text for audio 2.|speaker_name
```

3. Crie config:
```json
{
    "datasets": [
        {
            "name": "meu_dataset",
            "path": "/path/to/meu_dataset/",
            "meta_file_train": "metadata.csv",
            "formatter": "ljspeech",
            "language": "pt-br"
        }
    ]
}
```

4. Treine:
```bash
python TTS/bin/train_yourtts_lightning.py \
    --config_path meu_config.json \
    --gpus 1
```

---

## 🐛 Resolução de Problemas Comuns

### Erro: "CUDA out of memory"

```json
// Reduzir batch_size
{"batch_size": 16}  // ou 8
```

Ou usar gradient accumulation:
```bash
--accumulate_grad_batches 2
```

### Erro: "No module named 'pytorch_lightning'"

```bash
source venv_yourtts/bin/activate
pip install pytorch-lightning>=2.0.0
```

### Treinamento muito lento

1. Habilitar mixed precision:
   ```bash
   --precision 16-mixed
   ```

2. Verificar se GPU está sendo usada:
   ```bash
   nvidia-smi
   # GPU utilization deve estar >80%
   ```

3. Aumentar num_workers:
   ```json
   {"num_loader_workers": 8}
   ```

### Loss = NaN

1. Reduzir learning rate:
   ```json
   {"lr_gen": 1e-4, "lr_disc": 1e-4}
   ```

2. Verificar gradient clipping:
   ```json
   {"grad_clip": 1000.0}
   ```

---

## 📚 Documentação Completa

Para mais detalhes, consulte:

- [README_YOURTTS.md](README_YOURTTS.md) - Overview completo
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Guia de testes detalhado
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Migração de código antigo
- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - O que foi mudado

---

## 💡 Dicas

1. **Sempre use mixed precision** (`--precision 16-mixed`) - economiza ~50% de memória

2. **Monitor TensorBoard** - visualize losses, áudio gerado, alinhamentos

3. **Comece pequeno** - teste com poucos dados primeiro (--debug mode)

4. **Salve checkpoints regularmente** - já configurado automaticamente

5. **Use DDP para multi-GPU** - muito mais rápido que DP

---

## ✨ Estrutura Final do Projeto

```
TTS/
├── venv_yourtts/                          # Ambiente virtual
├── outputs/
│   └── yourtts_ljspeech_test/
│       ├── checkpoints/                   # Checkpoints salvos
│       ├── lightning_logs/                # Logs TensorBoard
│       └── phoneme_cache/                 # Cache de fonemas
├── TTS/
│   ├── tts/
│   │   ├── models/
│   │   │   ├── vits.py                    # Modelo VITS/YourTTS
│   │   │   └── vits_lightning.py          # Lightning Module
│   │   └── datasets/
│   │       └── yourtts_datamodule.py      # DataModule
│   └── bin/
│       └── train_yourtts_lightning.py     # Script de treinamento
├── ljspeech_test_config.json              # Config de teste
├── test_training.py                       # Testes automatizados
├── setup_and_test_venv.sh                 # Setup automatizado
└── README_YOURTTS.md                      # Doc principal
```

---

## 🎉 Pronto para Começar!

```bash
# 1. Setup
./setup_and_test_venv.sh

# 2. Ativar ambiente
source venv_yourtts/bin/activate

# 3. Treinar
python TTS/bin/train_yourtts_lightning.py \
    --config_path ljspeech_test_config.json \
    --gpus 1 \
    --precision 16-mixed

# 4. Monitorar
tensorboard --logdir outputs/
```

**Boa sorte com o treinamento! 🚀🎤**

---

*Criado em: 2026-01-02*
*Branch: yourtts-lightning-refactor*
