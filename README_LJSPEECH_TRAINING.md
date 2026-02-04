# Treinamento YourTTS com LJSpeech usando PyTorch Lightning

Guia completo para treinar o YourTTS no dataset LJSpeech usando a implementação Lightning.

## 📋 Pré-requisitos

1. **Dataset LJSpeech baixado:**
   ```bash
   wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
   tar -xvjf LJSpeech-1.1.tar.bz2 -C /home/fred/Projetos/DATASETS/
   ```

2. **PyTorch Lightning instalado:**
   ```bash
   conda activate yourtts
   pip install pytorch-lightning
   ```

3. **Ambiente yourtts configurado** (conda environment)

## 🚀 Início Rápido (3 Passos)

### 1. Preparar Dataset
Execute o script de preparação para computar os speaker embeddings:

```bash
./prepare_ljspeech.sh
```

**O que este script faz:**
- Verifica se o dataset existe
- Computa speaker embeddings (d-vectors) se necessário
- Salva embeddings em `/home/fred/Projetos/DATASETS/LJSpeech-1.1/speakers.pth`

**Tempo estimado:** 5-10 minutos (apenas na primeira vez)

### 2. Treinar Modelo
```bash
python train_lightning.py \
    --config config_ljspeech.json \
    --output_dir ./outputs_ljspeech
```

### 3. Visualizar Treinamento
Em outro terminal:
```bash
tensorboard --logdir outputs_ljspeech/lightning_logs
```

Acesse: http://localhost:6006

## 📝 Configuração (config_ljspeech.json)

A configuração foi criada baseada no recipe original em `recipes/ljspeech/train_yourtts_ljspeech.py`.

### Principais Parâmetros:

**Modelo:**
- Arquitetura: VITS com modificações YourTTS
- Speaker embeddings: d-vectors de 512 dimensões
- Text encoder: 10 camadas (mais profundo que VITS padrão)
- Decoder: ResNet blocks tipo 2

**Treinamento:**
- Batch size: 2 (ajuste conforme sua GPU)
- Learning rate: 0.0001 (generator: 0.0002, discriminator: 0.0002)
- Optimizer: AdamW
- Scheduler: ExponentialLR (gamma: 0.999875)
- Gradient clipping: [1000, 1000]

**Áudio:**
- Sample rate: 22050 Hz
- FFT size: 1024
- Hop length: 256
- Mel bins: 80

**Losses:**
- KL loss: α=1.0
- Generator loss: α=1.0
- Discriminator loss: α=1.0
- Feature matching: α=1.0
- Mel reconstruction: α=45.0 (principal)
- Duration: α=1.0
- Speaker encoder: α=9.0

**Dataset:**
- Mín. comprimento texto: 5 caracteres
- Máx. comprimento texto: 500 caracteres
- Mín. comprimento áudio: 3 segundos
- Máx. comprimento áudio: 20 segundos

## 🔧 Ajustes para Sua GPU

### GPU com Pouca Memória (<8GB)

Edite `config_ljspeech.json`:

```json
{
  "batch_size": 1,          // Reduzir batch size
  "eval_batch_size": 1,
  "mixed_precision": true,  // Usar 16-bit precision
  "num_loader_workers": 4,  // Reduzir workers
  "batch_group_size": 24    // Reduzir group size
}
```

Ou use gradient accumulation em `train_lightning.py`:
```python
trainer = pl.Trainer(
    accumulate_grad_batches=4  # Simula batch_size=4
)
```

### GPU com Muita Memória (>16GB)

```json
{
  "batch_size": 8,
  "eval_batch_size": 8,
  "num_loader_workers": 16,
  "batch_group_size": 96
}
```

### Multi-GPU

Edite `train_lightning.py`, linha ~341:

```python
trainer = pl.Trainer(
    devices=[0, 1, 2, 3],  # IDs das GPUs
    strategy='ddp'          # Distributed Data Parallel
)
```

## 📊 Monitoramento

### TensorBoard

```bash
tensorboard --logdir outputs_ljspeech/lightning_logs
```

**Métricas disponíveis:**
- `train/loss_total` - Loss total (discriminador + gerador)
- `train/loss_disc` - Loss do discriminador
- `train/loss_gen` - Loss do gerador
- `train/gen_loss_kl` - KL divergence
- `train/gen_loss_mel` - Reconstrução de mel-spectrogram
- `train/gen_loss_duration` - Duration predictor
- `train/lr_gen` - Learning rate do gerador
- `train/lr_disc` - Learning rate do discriminador
- Métricas de validação (`val/...`)

### Logs em Tempo Real

```bash
tail -f outputs_ljspeech/lightning_logs/version_X/metrics.csv
```

## ⏱️ Tempo de Treinamento

**Estimativas (baseadas em experiência com YourTTS):**

| GPU | Batch Size | Steps/Second | Tempo para 100k steps |
|-----|------------|--------------|----------------------|
| RTX 3090 (24GB) | 8 | ~1.5 | ~18 horas |
| RTX 3080 (10GB) | 4 | ~1.2 | ~23 horas |
| RTX 2080 Ti (11GB) | 4 | ~1.0 | ~28 horas |
| V100 (32GB) | 16 | ~2.0 | ~14 horas |

**Nota:** YourTTS geralmente precisa de 100k-200k steps para convergir.

## 🎯 Checkpoints e Avaliação

### Salvamento de Checkpoints

Por padrão, checkpoints são salvos a cada 10000 steps:

```
outputs_ljspeech/checkpoints/
├── vits-epoch=000-train_loss_total=12.3456.ckpt
├── vits-epoch=001-train_loss_total=10.1234.ckpt
└── last.ckpt
```

### Carregar Checkpoint

Em `train_lightning.py`:

```python
# Carregar de checkpoint específico
model = VitsLightningModule.load_from_checkpoint(
    'outputs_ljspeech/checkpoints/vits-epoch=010.ckpt',
    config=config,
    ap=ap
)

# Ou continuar treinamento automaticamente
trainer.fit(model, train_loader, eval_loader, ckpt_path='path/to/checkpoint.ckpt')
```

### Avaliar Modelo

Após o treinamento, você pode gerar amostras dos test_sentences:

```python
# No final do treinamento, o Lightning salva automaticamente
# Você pode carregar e usar o modelo para inferência
```

## 🐛 Troubleshooting

### Erro: "speakers.pth not found"

```bash
# Re-execute o script de preparação
./prepare_ljspeech.sh
```

### Erro: "CUDA out of memory"

1. Reduza batch_size no config
2. Ative mixed_precision
3. Use gradient accumulation
4. Feche outros programas usando GPU

```bash
# Ver uso de GPU
nvidia-smi

# Limpar cache CUDA (em Python)
import torch
torch.cuda.empty_cache()
```

### Erro: "No audio files found"

Verifique o caminho do dataset no `config_ljspeech.json`:

```json
{
  "datasets": [
    {
      "path": "/home/fred/Projetos/DATASETS/LJSpeech-1.1/",  // Correto
      "meta_file_train": "metadata.csv"
    }
  ]
}
```

### Warning: "stft with return_complex=False is deprecated"

Este é apenas um warning do PyTorch. Não afeta o treinamento.

### Loss não está convergindo

1. Verifique se os embeddings foram computados corretamente
2. Verifique se o dataset está no caminho correto
3. Reduza learning rate se loss explodir
4. Aumente batch_size se possível (mais estável)
5. Verifique logs para erros de carregamento de áudio

## 📈 Resultados Esperados

### Após 50k steps:
- Loss total: ~15-20
- Loss mel: ~3-5
- Áudio começando a ter estrutura, mas ainda com ruído

### Após 100k steps:
- Loss total: ~10-15
- Loss mel: ~2-3
- Áudio inteligível, com boa prosódia

### Após 200k steps:
- Loss total: ~8-12
- Loss mel: ~1.5-2.5
- Áudio de alta qualidade, indistinguível do original

## 🔄 Comparação com Original

Para comparar com o treinamento original:

```bash
# Lightning
python train_lightning.py --config config_ljspeech.json --output_dir ./outputs_lightning

# Original
python TTS/bin/train_tts.py --config_path recipes/ljspeech/train_yourtts_ljspeech.py
```

**Diferenças esperadas:**
- ✅ Mesmas losses
- ✅ Mesma convergência
- ✅ Mesma qualidade de áudio
- ⚡ Melhor logging no Lightning (TensorBoard mais rico)
- ⚡ Checkpointing mais flexível no Lightning

## 📚 Próximos Passos

Após treinar com sucesso:

1. **Experimente com outros datasets:**
   - VCTK (multi-speaker inglês)
   - Common Voice (multilingual)
   - Seu próprio dataset

2. **Ajuste hiperparâmetros:**
   - Learning rate
   - Batch size
   - Alphas das losses

3. **Ative features avançadas:**
   - Speaker Consistency Loss (SCL)
   - Language embeddings (multilingual)
   - Use pre-trained checkpoint

4. **Deploy:**
   - Exporte para ONNX
   - Use TorchScript
   - Crie API de inferência

## 🔗 Referências

- **YourTTS Paper:** https://arxiv.org/abs/2112.02418
- **VITS Paper:** https://arxiv.org/abs/2106.06103
- **LJSpeech Dataset:** https://keithito.com/LJ-Speech-Dataset/
- **PyTorch Lightning:** https://lightning.ai/docs/pytorch/stable/

---

**Criado em:** 2026-02-03
**Atualizado em:** 2026-02-03
**Versão:** 1.0
