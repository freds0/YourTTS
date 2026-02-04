# Guia de Fine-tuning - YourTTS com Checkpoint Pré-treinado

Este guia mostra como fazer fine-tuning do YourTTS partindo de um checkpoint pré-treinado.

## 📦 Seu Checkpoint

```
Checkpoint: /home/fred/Projetos/yourtts-du_en_fr_ge_it_pl_ptbr_sp-November-08-2022_12+19PM-0cbaa0f6/checkpoint_1020000.pth
Tipo: YourTTS Multilingual (Alemão, Inglês, Francês, Alemão, Italiano, Polonês, Português-BR, Espanhol)
Steps: 1,020,000
```

## 🎯 Vantagens do Fine-tuning

✅ **Convergência muito mais rápida** (5-10x menos steps)
✅ **Melhor qualidade inicial** (já sabe falar)
✅ **Menos dados necessários** (pode funcionar com datasets pequenos)
✅ **Transfer learning eficaz** (conhecimento multilíngue ajuda)

## 🚀 Método 1: Via Linha de Comando (Recomendado)

### Passo 1: Preparar Dataset

```bash
# Se ainda não fez
./prepare_ljspeech.sh
```

### Passo 2: Treinar com Checkpoint

```bash
python train_lightning.py \
    --config config_ljspeech.json \
    --output_dir ./outputs_finetuned \
    --restore_path /home/fred/Projetos/yourtts-du_en_fr_ge_it_pl_ptbr_sp-November-08-2022_12+19PM-0cbaa0f6/checkpoint_1020000.pth
```

**Saída esperada:**
```
================================================================================
Loading checkpoint from: /home/fred/Projetos/yourtts-du_en_fr_ge_it_pl_ptbr_sp-November-08-2022_12+19PM-0cbaa0f6/checkpoint_1020000.pth
================================================================================

✓ Loaded model weights from 'model' key
✓ Successfully restored model from checkpoint
================================================================================
```

## 📝 Método 2: Via Config JSON

Use o config já criado:

```bash
python train_lightning.py \
    --config config_ljspeech_finetuned.json \
    --output_dir ./outputs_finetuned
```

O arquivo `config_ljspeech_finetuned.json` já contém:
```json
{
  "restore_path": "/home/fred/Projetos/yourtts-du_en_fr_ge_it_pl_ptbr_sp-November-08-2022_12+19PM-0cbaa0f6/checkpoint_1020000.pth",
  ...
}
```

## ⚙️ Ajustes Recomendados para Fine-tuning

### 1. Learning Rate Menor

No fine-tuning, use learning rate **5-10x menor**:

```json
{
  "lr": 0.00001,      // Original: 0.0001
  "lr_gen": 0.00001,  // Original: 0.0002
  "lr_disc": 0.00001  // Original: 0.0002
}
```

**Por quê?** O modelo já está treinado, você quer apenas ajustar, não recomeçar.

### 2. Salvar Checkpoints Mais Frequentemente

```json
{
  "save_step": 2500,    // Original: 10000
  "print_step": 10      // Original: 25
}
```

**Por quê?** Fine-tuning converge rápido, você quer capturar o melhor momento.

### 3. Menos Epochs

```json
{
  "epochs": 100  // Original: 1000
}
```

**Por quê?** Com checkpoint pré-treinado, 10-20 epochs podem ser suficientes.

## 📊 Monitoramento

### TensorBoard

```bash
tensorboard --logdir outputs_finetuned/lightning_logs
```

**O que observar:**
- Loss deve começar **baixa** (~50-100) em vez de alta (~500-1000)
- Loss deve cair rapidamente nos primeiros 5k steps
- Qualidade de áudio deve ser boa desde o início

### Verificar Progresso

```bash
# Ver últimas métricas
tail -f outputs_finetuned/lightning_logs/version_0/metrics.csv

# Ver checkpoints salvos
ls -lh outputs_finetuned/checkpoints/
```

## 🎯 Expectativas de Convergência

### Com Checkpoint Pré-treinado:

| Métrica | Sem Checkpoint | Com Checkpoint |
|---------|---------------|----------------|
| Steps até áudio inteligível | 50k-100k | 5k-10k |
| Steps até qualidade ótima | 150k-200k | 20k-50k |
| Loss inicial | ~500 | ~50-100 |
| Loss final | ~8-12 | ~8-12 |
| Tempo de treino (RTX 3090) | ~18h | ~2-4h |

## 🔧 Troubleshooting

### Erro: "Checkpoint keys don't match"

O checkpoint pode ter formato diferente. O script tenta 3 formatos:

1. `checkpoint['model']` (formato TTS original) ✅ Seu caso
2. `checkpoint['state_dict']` (formato Lightning)
3. `checkpoint` direto (state_dict puro)

Se falhar, verifique as chaves:

```python
import torch
ckpt = torch.load('checkpoint_1020000.pth', map_location='cpu')
print(ckpt.keys())
```

### Loss Muito Alta no Início

Se a loss inicial for alta (~500) como sem checkpoint:
- Verifique se o checkpoint foi carregado (veja mensagem de confirmação)
- Verifique se o caminho está correto
- Verifique se a arquitetura do modelo é compatível

### Loss Não Diminui

Se a loss não diminuir durante fine-tuning:
- **Reduza o learning rate** (tente 0.00001 ou 0.000005)
- Verifique se os dados estão carregando corretamente
- Verifique se não há erro de formato de áudio

### Overfitting Rápido

Se o modelo overfitar rápido (loss de treino cai, mas validação sobe):
- **Reduza epochs** (tente 20-50)
- Use data augmentation
- Aumente batch_size se possível

## 💡 Dicas Avançadas

### 1. Congelar Algumas Camadas

Para fine-tuning mais conservador, você pode congelar o encoder:

```python
# Em train_lightning.py, após carregar o checkpoint
for name, param in model.model.named_parameters():
    if 'text_encoder' in name:
        param.requires_grad = False
```

### 2. Usar Learning Rate Scheduler Diferente

Para fine-tuning, considere usar scheduler mais agressivo:

```json
{
  "lr_scheduler_gen": "CosineAnnealingLR",
  "lr_scheduler_gen_params": {
    "T_max": 50000,
    "eta_min": 1e-6
  }
}
```

### 3. Gradual Unfreezing

Comece com encoder congelado, depois descongele após alguns epochs:

```python
# Epoch 0-5: Só decoder
# Epoch 5-10: Decoder + flow
# Epoch 10+: Tudo
```

## 📈 Comparação: Treino do Zero vs Fine-tuning

### Treino do Zero (config_ljspeech.json)

```bash
python train_lightning.py \
    --config config_ljspeech.json \
    --output_dir ./outputs_from_scratch
```

**Características:**
- ❌ Lento (150k-200k steps)
- ❌ Loss inicial alta (~500)
- ❌ Precisa de muito dado
- ✅ Modelo específico para seu dataset

### Fine-tuning (config_ljspeech_finetuned.json)

```bash
python train_lightning.py \
    --config config_ljspeech_finetuned.json \
    --output_dir ./outputs_finetuned
```

**Características:**
- ✅ Rápido (20k-50k steps)
- ✅ Loss inicial baixa (~50-100)
- ✅ Funciona com menos dados
- ✅ Aproveita conhecimento multilíngue

## 🎤 Exemplo Completo

```bash
# 1. Preparar dataset
./prepare_ljspeech.sh

# 2. Fine-tuning com checkpoint
python train_lightning.py \
    --config config_ljspeech_finetuned.json \
    --output_dir ./outputs_finetuned \
    --restore_path /home/fred/Projetos/yourtts-du_en_fr_ge_it_pl_ptbr_sp-November-08-2022_12+19PM-0cbaa0f6/checkpoint_1020000.pth

# 3. Monitorar
tensorboard --logdir outputs_finetuned/lightning_logs

# 4. Testar checkpoint intermediário (após 10k steps)
python test_model.py \
    --checkpoint outputs_finetuned/checkpoints/vits-epoch=005.ckpt \
    --text "Teste de síntese de voz"
```

## 📚 Referências

- **YourTTS Paper:** https://arxiv.org/abs/2112.02418 (Section 4.2: Transfer Learning)
- **Fine-tuning Best Practices:** https://lightning.ai/docs/pytorch/stable/advanced/transfer_learning.html

---

**Criado em:** 2026-02-03
**Checkpoint:** checkpoint_1020000.pth (1.02M steps)
**Objetivo:** Fine-tuning para LJSpeech
**Status:** ✅ Pronto para uso
