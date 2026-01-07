# Guia de Testes - YourTTS Lightning

Este guia explica como testar o código de treinamento do YourTTS com PyTorch Lightning.

## 📋 Índice

1. [Pré-requisitos](#pré-requisitos)
2. [Testes Rápidos](#testes-rápidos)
3. [Teste Completo Automatizado](#teste-completo-automatizado)
4. [Teste de Treinamento Real](#teste-de-treinamento-real)
5. [Troubleshooting](#troubleshooting)

---

## Pré-requisitos

### 1. Instalar Dependências

```bash
# Criar ambiente virtual (recomendado)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Ou instalar manualmente
pip install torch torchaudio
pip install pytorch-lightning>=2.0.0
pip install tensorboard>=2.13.0
pip install -r requirements.txt
```

### 2. Verificar Instalação

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import pytorch_lightning as pl; print(f'Lightning: {pl.__version__}')"
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

Saída esperada:
```
PyTorch: 2.1.0 (ou superior)
Lightning: 2.0.0 (ou superior)
CUDA: True (se você tem GPU)
```

---

## Testes Rápidos

### Teste 1: Importar Módulos

```python
# test_imports.py
from TTS.tts.models.vits_lightning import YourTTSLightningModule
from TTS.tts.datasets.yourtts_datamodule import YourTTSDataModule
from TTS.tts.configs.vits_config import VitsConfig
import pytorch_lightning as pl

print("✅ Todos os imports funcionaram!")
```

Execute:
```bash
python3 test_imports.py
```

### Teste 2: Criar Modelo

```python
# test_model.py
from TTS.tts.models.vits_lightning import YourTTSLightningModule
from TTS.tts.configs.vits_config import VitsConfig

# Criar config mínima
config = VitsConfig()
config.num_chars = 100
config.model_args.num_speakers = 1
config.model_args.use_speaker_embedding = False
config.model_args.use_d_vector_file = False

# Criar modelo
model = YourTTSLightningModule(config)
print(f"✅ Modelo criado com {sum(p.numel() for p in model.parameters()):,} parâmetros")
```

Execute:
```bash
python3 test_model.py
```

### Teste 3: Forward Pass

```python
# test_forward.py
import torch
from TTS.tts.models.vits_lightning import YourTTSLightningModule
from TTS.tts.configs.vits_config import VitsConfig

config = VitsConfig()
config.num_chars = 100
config.model_args.num_speakers = 1
config.model_args.use_speaker_embedding = False
config.model_args.use_d_vector_file = False

model = YourTTSLightningModule(config)
model.eval()

# Criar batch sintético
batch_size = 2
tokens = torch.randint(0, 100, (batch_size, 50))
token_lens = torch.tensor([50, 50])
spec = torch.randn(batch_size, 513, 100)
spec_lens = torch.tensor([100, 100])
waveform = torch.randn(batch_size, 1, 25600)

# Forward pass
with torch.no_grad():
    outputs = model.vits_model.forward(
        tokens, token_lens, spec, spec_lens, waveform, aux_input={}
    )

print(f"✅ Forward pass funcionou! Output shape: {outputs['model_outputs'].shape}")
```

Execute:
```bash
python3 test_forward.py
```

---

## Teste Completo Automatizado

Use o script `test_training.py` fornecido:

```bash
python3 test_training.py
```

Este script executa 6 testes automáticos:

1. ✅ **Imports** - Verifica se todos os módulos necessários podem ser importados
2. ✅ **Criação do Modelo** - Testa se o YourTTSLightningModule pode ser criado
3. ✅ **Forward Pass** - Verifica se o modelo pode processar dados
4. ✅ **Training Step** - Testa se o passo de treinamento funciona
5. ✅ **Config Loading** - Verifica se configs JSON podem ser carregados
6. ✅ **Checkpoint** - Testa save/load de checkpoints

### Saída Esperada

```
================================================================================
TESTE DE TREINAMENTO YOURTTS LIGHTNING
================================================================================
PyTorch: 2.1.0
CUDA disponível: True
CUDA device: NVIDIA GeForce RTX 3090
================================================================================

================================================================================
TESTE 1: Verificando imports...
================================================================================
✓ PyTorch Lightning: 2.0.0
✓ YourTTSLightningModule importado
✓ YourTTSDataModule importado
✓ VitsConfig importado

✅ Todos os imports funcionaram!

[... outros testes ...]

================================================================================
RESUMO DOS TESTES
================================================================================
✅ PASSOU - imports
✅ PASSOU - model_creation
✅ PASSOU - forward_pass
✅ PASSOU - training_step
✅ PASSOU - config_loading
✅ PASSOU - checkpoint

================================================================================

🎉 TODOS OS TESTES PASSARAM! 🎉

Você pode prosseguir com o treinamento usando:
python TTS/bin/train_yourtts_lightning.py --config_path <seu_config.json>
```

---

## Teste de Treinamento Real

### Opção 1: Modo Debug (Rápido)

Teste com apenas 5 batches para verificar se tudo funciona:

```bash
python3 TTS/bin/train_yourtts_lightning.py \
    --config_path recipes/vctk/yourtts/yourtts_lightning_config.json \
    --debug
```

Isso executa:
- 5 batches de treino
- 5 batches de validação
- Sem salvamento de checkpoints
- Apenas para verificar se o código roda

### Opção 2: Teste com Dados Sintéticos

Crie um pequeno dataset sintético para teste:

```python
# create_synthetic_dataset.py
import os
import numpy as np
import soundfile as sf

# Criar diretório
os.makedirs("datasets/test_dataset/wavs", exist_ok=True)

# Criar arquivos de áudio sintéticos
for i in range(10):
    audio = np.random.randn(16000).astype(np.float32) * 0.1
    sf.write(f"datasets/test_dataset/wavs/sample_{i:03d}.wav", audio, 16000)

# Criar metadata
with open("datasets/test_dataset/metadata.csv", "w") as f:
    for i in range(10):
        f.write(f"wavs/sample_{i:03d}.wav|This is test sentence {i}.|speaker_0\n")

print("✅ Dataset sintético criado em datasets/test_dataset/")
```

Execute:
```bash
python3 create_synthetic_dataset.py
```

Depois teste o treinamento:

```bash
python3 TTS/bin/train_yourtts_lightning.py \
    --config_path test_config.json \
    --gpus 1 \
    --debug
```

### Opção 3: Teste com LJSpeech (Dataset Real Pequeno)

```bash
# Baixar LJSpeech (2.6 GB)
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2

# Criar subset pequeno (100 amostras) para teste
head -100 LJSpeech-1.1/metadata.csv > LJSpeech-1.1/metadata_train.csv
tail -10 LJSpeech-1.1/metadata.csv > LJSpeech-1.1/metadata_val.csv
```

Configuração para LJSpeech:

```json
{
    "datasets": [
        {
            "name": "ljspeech",
            "path": "LJSpeech-1.1/",
            "meta_file_train": "metadata_train.csv",
            "meta_file_val": "metadata_val.csv",
            "formatter": "ljspeech"
        }
    ]
}
```

Treinar:
```bash
python3 TTS/bin/train_yourtts_lightning.py \
    --config_path ljspeech_config.json \
    --gpus 1 \
    --precision 16-mixed
```

### Opção 4: Teste de Fine-tuning

Teste carregar um checkpoint existente:

```bash
# Se você tem um checkpoint antigo
python3 TTS/bin/train_yourtts_lightning.py \
    --config_path config.json \
    --restore_path old_checkpoint.pth \
    --debug
```

---

## Verificações Durante o Treinamento

### 1. Verificar Logs

```bash
# Abrir TensorBoard
tensorboard --logdir outputs/

# Acesse: http://localhost:6006
```

O que verificar:
- ✅ **Losses diminuindo**: `train/gen_loss`, `train/disc_loss`
- ✅ **Sem NaN**: Nenhuma métrica deve ser NaN
- ✅ **GPU utilizada**: Uso de GPU > 0%
- ✅ **Áudio gerado**: Samples de áudio na aba "Audio"
- ✅ **Alinhamentos**: Devem ficar diagonais

### 2. Verificar Checkpoints

```bash
ls -lh outputs/checkpoints/

# Deve mostrar:
# yourtts-epoch=0000-val_loss=X.XX.ckpt
# yourtts-epoch=0001-val_loss=X.XX.ckpt
# last.ckpt
```

### 3. Verificar GPU

```bash
# Durante o treinamento, em outro terminal:
nvidia-smi

# Ou monitorar continuamente:
watch -n 1 nvidia-smi
```

Esperado:
- GPU utilization: 80-100%
- Memory usado: Depende do batch_size

### 4. Verificar Performance

Métricas esperadas (após algumas épocas):

| Métrica | Valor Esperado |
|---------|----------------|
| gen_loss | < 50 (inicial: ~100-200) |
| disc_loss | ~1-5 |
| mel_loss | < 10 (inicial: ~20-40) |
| kl_loss | ~1-3 |
| duration_loss | ~5-15 |
| feature_loss | ~5-15 |

---

## Troubleshooting

### Erro: "CUDA out of memory"

**Solução 1**: Reduzir batch_size
```json
{
    "batch_size": 16,  // Em vez de 32
    "eval_batch_size": 8
}
```

**Solução 2**: Usar gradient accumulation
```bash
python3 TTS/bin/train_yourtts_lightning.py \
    --config_path config.json \
    --accumulate_grad_batches 2
```

**Solução 3**: Mixed precision
```bash
--precision 16-mixed
```

### Erro: "No module named 'pytorch_lightning'"

```bash
pip install pytorch-lightning>=2.0.0
```

### Erro: Loss = NaN

**Causas comuns**:
1. Learning rate muito alto
   ```json
   {"lr_gen": 1e-4, "lr_disc": 1e-4}  // Reduzir de 2e-4
   ```

2. Gradient clipping desabilitado
   ```json
   {"grad_clip": 1000.0}  // Adicionar
   ```

3. Dados corrompidos
   - Verificar se áudios contêm valores válidos
   - Normalização correta

### Erro: "Discriminator loss exploding"

Ajustar learning rate do discriminador:
```json
{
    "lr_disc": 1e-4,  // Menor que generator
    "lr_gen": 2e-4
}
```

### Treinamento muito lento

**Verificações**:
1. Usar DDP em vez de DP:
   ```bash
   --strategy ddp  # Correto
   # vs
   --strategy dp   # Lento
   ```

2. Habilitar mixed precision:
   ```bash
   --precision 16-mixed
   ```

3. Aumentar num_workers:
   ```json
   {"num_loader_workers": 8}  // Depende da CPU
   ```

4. Desabilitar validação frequente:
   ```python
   trainer = pl.Trainer(
       check_val_every_n_epoch=5  # Validar a cada 5 épocas
   )
   ```

### Erro ao carregar checkpoint antigo

```python
# Modo compatibilidade
model = YourTTSLightningModule.load_from_checkpoint(
    "old_checkpoint.pth",
    strict=False,  # Ignorar keys ausentes
)
```

---

## Checklist Final

Antes de treinar o modelo completo, verifique:

- [ ] Todos os testes em `test_training.py` passaram
- [ ] Config JSON está correto e completo
- [ ] Dataset está preparado e acessível
- [ ] D-vectors computados (se usando `use_d_vector_file: true`)
- [ ] GPU disponível e funcional
- [ ] Espaço em disco suficiente (~50GB para checkpoints)
- [ ] TensorBoard acessível
- [ ] Modo debug funcionou sem erros

Se tudo ✅, você está pronto para treinar!

```bash
python3 TTS/bin/train_yourtts_lightning.py \
    --config_path recipes/vctk/yourtts/yourtts_lightning_config.json \
    --gpus 2 \
    --precision 16-mixed \
    --strategy ddp
```

---

## Recursos Adicionais

- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Migração de código antigo
- [README_YOURTTS.md](README_YOURTTS.md) - Documentação principal
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)

**Boa sorte com o treinamento! 🚀**
