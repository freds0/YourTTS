# Guia para Iniciar o Treinamento - YourTTS Lightning

## 📋 Passo a Passo Completo

### 1. Computar Speaker Embeddings (OBRIGATÓRIO - primeira vez apenas)

Os embeddings de speaker precisam ser computados antes do treinamento. Isso leva **5-10 minutos** e é feito apenas uma vez.

```bash
# Opção A: Usar o script Python
python compute_embeddings_ljspeech.py

# Opção B: Via linha de comando
python TTS/bin/compute_embeddings.py \
    model_se.pth.tar \
    config_se.json \
    /home/fred/Projetos/DATASETS/LJSpeech-1.1/speakers.pth \
    --dataset_path /home/fred/Projetos/DATASETS/LJSpeech-1.1/ \
    --formatter_name ljspeech
```

**Aguarde até ver:**
```
✓ Embeddings computed successfully!
✓ Saved to: /home/fred/Projetos/DATASETS/LJSpeech-1.1/speakers.pth
✓ Total embeddings: 13100
```

### 2. Verificar se Embeddings Foram Criados

```bash
ls -lh /home/fred/Projetos/DATASETS/LJSpeech-1.1/speakers.pth
# Deve mostrar um arquivo de ~59MB

# Verificar formato
python -c "
import torch
d = torch.load('/home/fred/Projetos/DATASETS/LJSpeech-1.1/speakers.pth')
print(f'Total: {len(d)} embeddings')
print(f'Sample keys: {list(d.keys())[:3]}')
"
# Deve mostrar: ljspeech#wavs/LJ...
```

### 3. Treinar do Zero

```bash
python train_lightning.py \
    --config config_ljspeech.json \
    --output_dir ./outputs_ljspeech
```

### 4. Treinar com Checkpoint Pré-treinado (Fine-tuning)

```bash
python train_lightning.py \
    --config config_ljspeech_finetuned.json \
    --output_dir ./outputs_finetuned \
    --restore_path /home/fred/Projetos/yourtts-du_en_fr_ge_it_pl_ptbr_sp-November-08-2022_12+19PM-0cbaa0f6/checkpoint_1020000.pth
```

### 5. Monitorar Treinamento

Em outro terminal:

```bash
# TensorBoard
tensorboard --logdir outputs_ljspeech/lightning_logs

# Ou para fine-tuning
tensorboard --logdir outputs_finetuned/lightning_logs
```

Acesse: http://localhost:6006

##⚠️ Solução de Problemas

### Erro: `KeyError: 'ljspeech#wavs/LJ...'`

**Causa:** Os embeddings não foram computados ou estão em formato errado.

**Solução:**
```bash
# 1. Deletar embeddings antigos
rm /home/fred/Projetos/DATASETS/LJSpeech-1.1/speakers.pth

# 2. Recomputar
python compute_embeddings_ljspeech.py

# 3. Verificar formato
python -c "
import torch
d = torch.load('/home/fred/Projetos/DATASETS/LJSpeech-1.1/speakers.pth')
print(list(d.keys())[0])  # Deve começar com 'ljspeech#'
"
```

### Erro: `AttributeError: 'NoneType' object has no attribute 'use_phonemes'`

**Causa:** Tokenizer não foi inicializado.

**Solução:** Já corrigido no código. Se persistir, atualize train_lightning.py.

### Erro: `KeyError: 'spec_lens'`

**Causa:** Batch não foi processado com `format_batch_on_device`.

**Solução:** Já corrigido no código. Se persistir, atualize train_lightning.py.

### Erro: `CUDA out of memory`

**Solução:**
```json
// Em config_ljspeech.json
{
  "batch_size": 1,  // Reduzir de 2 para 1
  "mixed_precision": true  // Ativar 16-bit
}
```

## 📊 Status do Treinamento

### Como Verificar se Está Treinando

```bash
# Ver processo
ps aux | grep train_lightning.py

# Ver GPU
nvidia-smi

# Ver logs
tail -f outputs_ljspeech/lightning_logs/version_0/metrics.csv
```

### Métricas Esperadas

**Primeiros Steps:**
- Loss total: ~500-1000 (treino do zero) ou ~50-100 (fine-tuning)
- Loss diminuindo gradualmente

**Após 10k steps:**
- Loss total: ~100-200 (treino do zero) ou ~30-50 (fine-tuning)
- Áudio começando a ficar inteligível

**Após 50k steps:**
- Loss total: ~30-50
- Áudio de boa qualidade

## ✅ Checklist Antes de Treinar

- [ ] Dataset LJSpeech baixado em `/home/fred/Projetos/DATASETS/LJSpeech-1.1/`
- [ ] Speaker embeddings computados (`speakers.pth` existe e tem ~59MB)
- [ ] Embeddings verificados (chaves começam com `ljspeech#`)
- [ ] PyTorch Lightning instalado (`pip install pytorch-lightning`)
- [ ] Conda environment `yourtts` ativado
- [ ] Config file escolhido (`config_ljspeech.json` ou `config_ljspeech_finetuned.json`)
- [ ] GPU disponível (`nvidia-smi` funciona)

## 🚀 Comando Final

Depois que os embeddings forem computados (aguarde o script `compute_embeddings_ljspeech.py` terminar):

```bash
# Ativar ambiente
conda activate yourtts

# Treinar do zero
python train_lightning.py \
    --config config_ljspeech.json \
    --output_dir ./outputs_ljspeech

# OU fine-tuning com checkpoint
python train_lightning.py \
    --config config_ljspeech_finetuned.json \
    --output_dir ./outputs_finetuned \
    --restore_path /home/fred/Projetos/yourtts-du_en_fr_ge_it_pl_ptbr_sp-November-08-2022_12+19PM-0cbaa0f6/checkpoint_1020000.pth
```

## 📚 Documentação Adicional

- **QUICKSTART.md** - Guia rápido geral
- **README_LIGHTNING.md** - Documentação técnica
- **README_LJSPEECH_TRAINING.md** - Guia específico LJSpeech
- **FINETUNING_GUIDE.md** - Guia de fine-tuning
- **FILES_CREATED.md** - Lista completa de arquivos

---

**Atualizado:** 2026-02-03
**Status:** Computando embeddings... aguarde 5-10 minutos
**Próximo Passo:** Executar `python train_lightning.py ...` após embeddings prontos
