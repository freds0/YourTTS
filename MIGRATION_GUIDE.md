# YourTTS PyTorch Lightning Migration Guide

Este guia explica como usar o novo sistema de treinamento com PyTorch Lightning e as mudanças feitas no repositório.

## 📋 Índice

1. [O que mudou](#o-que-mudou)
2. [Instalação](#instalação)
3. [Treinamento com PyTorch Lightning](#treinamento-com-pytorch-lightning)
4. [Migração de Checkpoints Antigos](#migração-de-checkpoints-antigos)
5. [Comparação: Antigo vs Novo](#comparação-antigo-vs-novo)
6. [Resolução de Problemas](#resolução-de-problemas)

---

## O que mudou

### ✅ Mantido (YourTTS Essencial)

**Modelos:**
- ✅ VITS/YourTTS (`TTS/tts/models/vits.py`)
- ✅ HiFiGAN Vocoder (`TTS/vocoder/models/hifigan_*.py`)
- ✅ Speaker Encoder (`TTS/encoder/`)

**Funcionalidades YourTTS:**
- ✅ Multi-speaker com d-vectors
- ✅ Speaker Consistency Loss (SCL)
- ✅ Treinamento multilíngue
- ✅ Todas as capacidades de síntese

### ❌ Removido

**Modelos TTS (11 modelos):**
- ❌ Tacotron, Tacotron2
- ❌ GlowTTS
- ❌ ForwardTTS (FastSpeech, SpeedySpeech, FastPitch)
- ❌ AlignTTS, NeuralHMM, Overflow
- ❌ DelightfulTTS
- ❌ XTTS, Bark, Tortoise

**Vocoders (10+ modelos):**
- ❌ MelGAN, MultiBand MelGAN
- ❌ Parallel WaveGAN
- ❌ UnivNet
- ❌ WaveGrad, WaveRNN

**Outros:**
- ❌ Voice Conversion (TTS/vc/)
- ❌ Layers específicos de modelos removidos
- ❌ Configs de modelos removidos

### ➕ Adicionado

**PyTorch Lightning:**
- ➕ `YourTTSLightningModule` - Módulo Lightning completo
- ➕ `YourTTSDataModule` - DataModule para dados
- ➕ `train_yourtts_lightning.py` - Script de treinamento moderno
- ➕ Suporte a multi-GPU com DDP
- ➕ Mixed precision automático
- ➕ TensorBoard integrado
- ➕ Checkpointing avançado

---

## Instalação

### 1. Instalar Dependências

```bash
# Instalar PyTorch Lightning e dependências
pip install -r requirements.txt

# Ou instalar apenas as novas dependências
pip install pytorch-lightning>=2.0.0 tensorboard>=2.13.0
```

### 2. Verificar Instalação

```python
import pytorch_lightning as pl
import torch

print(f"PyTorch: {torch.__version__}")
print(f"PyTorch Lightning: {pl.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## Treinamento com PyTorch Lightning

### Configuração Rápida

1. **Prepare seus dados** (mesmo formato de antes)

```
datasets/
└── VCTK-Corpus/
    ├── wav48/
    │   ├── p225/
    │   ├── p226/
    │   └── ...
    └── metadata.csv
```

2. **Configure o arquivo JSON** (veja `recipes/vctk/yourtts/yourtts_lightning_config.json`)

```json
{
    "model": "vits",
    "output_path": "outputs/yourtts_experiment/",
    "batch_size": 32,
    "epochs": 1000,
    "model_args": {
        "use_d_vector_file": true,
        "d_vector_dim": 512,
        "use_speaker_encoder_as_loss": true,
        "num_layers_text_encoder": 10,
        ...
    },
    ...
}
```

3. **Treine o modelo**

```bash
# Single GPU
python TTS/bin/train_yourtts_lightning.py \
    --config_path recipes/vctk/yourtts/yourtts_lightning_config.json

# Multi-GPU (4 GPUs com DDP)
python TTS/bin/train_yourtts_lightning.py \
    --config_path recipes/vctk/yourtts/yourtts_lightning_config.json \
    --gpus 4 \
    --strategy ddp

# Com mixed precision (recomendado)
python TTS/bin/train_yourtts_lightning.py \
    --config_path recipes/vctk/yourtts/yourtts_lightning_config.json \
    --gpus 2 \
    --precision 16-mixed
```

### Opções de Linha de Comando

```bash
python TTS/bin/train_yourtts_lightning.py --help

Opções principais:
  --config_path PATH              Caminho para o arquivo de configuração JSON (obrigatório)
  --gpus N                        Número de GPUs (default: 1)
  --precision {32,16-mixed,bf16-mixed}  Precisão de treinamento (default: 16-mixed)
  --strategy STRATEGY             Estratégia (auto, ddp, deepspeed, etc.)
  --resume_from_checkpoint PATH   Retomar treinamento de checkpoint Lightning
  --restore_path PATH             Carregar checkpoint antigo para fine-tuning
  --debug                         Modo debug (5 batches apenas)
```

### Monitoramento com TensorBoard

```bash
# Durante o treinamento, abra outro terminal
tensorboard --logdir outputs/yourtts_experiment/

# Acesse: http://localhost:6006
```

Você verá:
- Curvas de loss (discriminator, generator, total)
- Learning rates
- Amostras de áudio geradas
- Gráficos de alinhamento
- Uso de recursos

---

## Migração de Checkpoints Antigos

### Carregar Checkpoint Antigo em Lightning

O sistema mantém **compatibilidade total** com checkpoints antigos:

```python
from TTS.tts.models.vits_lightning import YourTTSLightningModule
from TTS.tts.configs.vits_config import VitsConfig

# Opção 1: Usar diretamente
model = YourTTSLightningModule.load_from_checkpoint(
    "path/to/old_checkpoint.pth",
    config=config,
)

# Opção 2: Via script de treinamento (fine-tuning)
python TTS/bin/train_yourtts_lightning.py \
    --config_path configs/yourtts_config.json \
    --restore_path path/to/old_checkpoint.pth
```

### Formato de Checkpoints

**Checkpoint Lightning (novo):**
```python
{
    'state_dict': {...},          # Pesos do modelo (formato Lightning)
    'model': {...},               # Pesos do modelo (formato antigo - compatibilidade)
    'config': VitsConfig,         # Configuração completa
    'epoch': int,                 # Época atual
    'global_step': int,           # Step global
    'optimizer_states': [...],    # Estados dos otimizadores
    'lr_schedulers': [...],       # Estados dos schedulers
    'speaker_manager': SpeakerManager,  # Se aplicável
    'language_manager': LanguageManager, # Se aplicável
}
```

**Checkpoint Antigo (Trainer):**
```python
{
    'model': {...},               # Pesos do modelo
    'config': VitsConfig,         # Configuração
    'step': int,                  # Step de treinamento
    # ... outros campos do Trainer
}
```

### Converter Checkpoint Antigo para Novo

Se você quiser converter explicitamente:

```python
import torch
from TTS.tts.models.vits_lightning import YourTTSLightningModule
from TTS.tts.configs.vits_config import VitsConfig

# Carregar checkpoint antigo
old_checkpoint = torch.load("old_checkpoint.pth")
config = old_checkpoint['config']

# Criar modelo Lightning
model = YourTTSLightningModule(config)

# Carregar pesos
model.vits_model.load_state_dict(old_checkpoint['model'])

# Salvar como checkpoint Lightning
trainer.save_checkpoint("new_checkpoint.ckpt")
```

---

## Comparação: Antigo vs Novo

### Script de Treinamento

**Antigo (Trainer):**
```python
from trainer import Trainer, TrainerArgs
from TTS.tts.models.vits import Vits

config = VitsConfig()
model = Vits.init_from_config(config)

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path="outputs/",
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
```

**Novo (PyTorch Lightning):**
```python
import pytorch_lightning as pl
from TTS.tts.models.vits_lightning import YourTTSLightningModule
from TTS.tts.datasets.yourtts_datamodule import YourTTSDataModule

config = VitsConfig()
model = YourTTSLightningModule(config)
datamodule = YourTTSDataModule(config, train_samples, eval_samples)

trainer = pl.Trainer(
    max_epochs=1000,
    gpus=2,
    precision="16-mixed",
    strategy="ddp",
)

trainer.fit(model, datamodule)
```

### Multi-GPU Training

**Antigo:**
```bash
# Necessário usar CUDA_VISIBLE_DEVICES e scripts customizados
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 TTS/bin/train_tts.py --config config.json
```

**Novo:**
```bash
# Simples e direto
python TTS/bin/train_yourtts_lightning.py \
    --config_path config.json \
    --gpus 4 \
    --strategy ddp
```

### Checkpointing

**Antigo:**
```python
# Salvamento manual em intervalos fixos
if step % 5000 == 0:
    save_checkpoint(model, step)
```

**Novo:**
```python
# Salvamento automático com monitoramento
checkpoint_callback = ModelCheckpoint(
    monitor='val/loss',
    save_top_k=3,
    mode='min',
    filename='yourtts-{epoch:04d}-{val/loss:.4f}'
)
# Salva automaticamente os 3 melhores checkpoints!
```

### Learning Rate Scheduling

**Antigo:**
```python
# Manual no training loop
scheduler.step()
```

**Novo:**
```python
# Automático via configure_optimizers
def configure_optimizers(self):
    optimizer = AdamW(...)
    scheduler = ExponentialLR(optimizer, gamma=0.999875)
    return [optimizer], [scheduler]
# Lightning gerencia tudo!
```

---

## Resolução de Problemas

### Erro: "No module named 'pytorch_lightning'"

```bash
pip install pytorch-lightning>=2.0.0
```

### Erro: "CUDA out of memory"

**Soluções:**
1. Reduzir `batch_size` no config
2. Usar gradient accumulation:
   ```bash
   python train_yourtts_lightning.py \
       --config_path config.json \
       --accumulate_grad_batches 2
   ```
3. Usar mixed precision:
   ```bash
   --precision 16-mixed
   ```

### Checkpoints muito grandes

Lightning salva otimizadores e schedulers. Para salvar apenas o modelo:

```python
# Em vez de:
torch.save(checkpoint, 'file.ckpt')

# Use:
torch.save({'state_dict': model.state_dict()}, 'file.ckpt')
```

### Treinamento lento com multi-GPU

Certifique-se de usar DDP (não DP):

```bash
--strategy ddp  # ✅ Correto (mais rápido)
# vs
--strategy dp   # ❌ Lento (evitar)
```

### Validação muito frequente

Ajuste a frequência de validação:

```python
trainer = pl.Trainer(
    check_val_every_n_epoch=5,  # Validar a cada 5 épocas
    # ou
    val_check_interval=0.5,     # Validar 2x por época
)
```

### Incompatibilidade de checkpoint

Se encontrar erro ao carregar checkpoint:

```python
# Opção 1: Carregar com strict=False
model.load_state_dict(checkpoint['state_dict'], strict=False)

# Opção 2: Extrair apenas os pesos do modelo VITS
model.vits_model.load_state_dict(checkpoint['model'])
```

---

## Próximos Passos

1. **Teste o treinamento** com seus dados
2. **Compare resultados** com o método antigo (devem ser idênticos)
3. **Aproveite features do Lightning**:
   - Early stopping
   - Gradient clipping automático
   - Profiling e otimização
   - Logging avançado

4. **Experimente com DeepSpeed** para modelos grandes:
   ```bash
   python train_yourtts_lightning.py \
       --config_path config.json \
       --strategy deepspeed_stage_2
   ```

---

## Recursos Adicionais

- **PyTorch Lightning Docs**: https://lightning.ai/docs/pytorch/stable/
- **YourTTS Paper**: https://arxiv.org/abs/2112.02418
- **VITS Paper**: https://arxiv.org/abs/2106.06103

---

## Suporte

Para problemas ou dúvidas:
1. Verifique este guia
2. Consulte os logs do TensorBoard
3. Revise a configuração JSON
4. Abra uma issue no repositório

**Boa sorte com seu treinamento YourTTS! 🚀**
