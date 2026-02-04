# Guia Weights & Biases (W&B) - YourTTS Lightning

Guia completo para usar Weights & Biases no treinamento do YourTTS com PyTorch Lightning.

## 🎯 Por Que Usar W&B?

✅ **Melhor que TensorBoard:**
- Interface web moderna e intuitiva
- Comparação fácil entre múltiplos experimentos
- Acesso remoto de qualquer lugar
- Compartilhamento de resultados com equipe
- Logs de sistema (GPU, CPU, memória) automáticos
- Alertas e notificações
- Relatórios automáticos

✅ **Funcionalidades Extras:**
- Salva código automaticamente
- Versionamento de modelos
- Hyperparameter sweeps (busca automática)
- Colaboração em equipe
- API para análise programática

## 📦 Instalação

### 1. Instalar W&B

```bash
conda activate yourtts
pip install wandb
```

### 2. Login no W&B

```bash
wandb login
```

Isso abrirá seu navegador. Você precisará:
- Criar conta gratuita em https://wandb.ai (se não tiver)
- Copiar sua API key
- Colar no terminal

**Alternativa offline:**
```bash
# Modo offline (sem subir para cloud)
wandb offline
```

## 🚀 Como Usar

### Opção 1: Habilitar W&B no Config Existente

Edite `config_ljspeech.json`:

```json
{
  ...
  "use_wandb": true,
  "wandb_project": "yourtts-ljspeech",
  "wandb_entity": "seu-username",
  "wandb_tags": ["yourtts", "ljspeech", "tts", "portuguese"]
}
```

Depois treine normalmente:

```bash
python train_lightning.py \
    --config config_ljspeech.json \
    --output_dir ./outputs_wandb
```

### Opção 2: Usar Config com W&B Pré-configurado

```bash
python train_lightning.py \
    --config config_ljspeech_wandb.json \
    --output_dir ./outputs_wandb
```

### Opção 3: Fine-tuning com W&B

```bash
python train_lightning.py \
    --config config_ljspeech_wandb.json \
    --output_dir ./outputs_finetuned_wandb \
    --restore_path /path/to/checkpoint_1020000.pth
```

## ⚙️ Configurações W&B

### Parâmetros no Config JSON

```json
{
  "use_wandb": true,              // Habilitar W&B (false = TensorBoard)
  "wandb_project": "my-project",  // Nome do projeto
  "wandb_entity": "my-team",      // Seu username ou team (null = default)
  "wandb_tags": ["tag1", "tag2"]  // Tags para organizar runs
}
```

### Exemplo de Configuração Completa

```json
{
  "run_name": "YourTTS-LJSpeech-v1",
  "use_wandb": true,
  "wandb_project": "yourtts-experiments",
  "wandb_entity": "meu-username",
  "wandb_tags": [
    "yourtts",
    "ljspeech",
    "portuguese",
    "batch-size-4",
    "lr-0.0002"
  ]
}
```

## 📊 O Que é Logado Automaticamente

### Métricas de Treinamento

- `train/loss_total` - Loss total (discriminador + gerador)
- `train/loss_disc` - Loss do discriminador
- `train/loss_gen` - Loss do gerador
- `train/disc_loss_disc` - Componentes da loss do discriminador
- `train/gen_loss_kl` - KL divergence
- `train/gen_loss_mel` - Reconstrução de mel-spectrogram
- `train/gen_loss_feat` - Feature matching
- `train/gen_loss_duration` - Duration predictor
- `train/lr_gen` - Learning rate do gerador
- `train/lr_disc` - Learning rate do discriminador

### Métricas de Validação

- `val/loss_total`
- `val/loss_disc`
- `val/loss_gen`
- E todos os componentes individuais

### Sistema

- GPU usage (%)
- GPU memory (MB)
- CPU usage (%)
- RAM usage (MB)
- Disk I/O
- Network I/O

### Código e Modelo

- Código Python usado (salvo automaticamente)
- Checkpoints do modelo (se `log_model=True`)
- Git commit hash (se em repositório git)
- Diff de código desde último commit

## 🔍 Visualizando Resultados

### Durante o Treinamento

1. O script imprimirá o link do W&B:
   ```
   wandb: 🚀 View run at https://wandb.ai/username/project/runs/abc123
   ```

2. Abra o link no navegador

3. Você verá em tempo real:
   - Gráficos de loss
   - Métricas do sistema
   - Logs do console
   - Código usado

### Após o Treinamento

Acesse: https://wandb.ai/seu-username/seu-projeto

Você pode:
- Comparar múltiplos runs
- Criar relatórios
- Exportar dados
- Compartilhar com equipe

## 📈 Comparando Experimentos

### Comparar Diferentes Configs

Execute múltiplos treinamentos:

```bash
# Experimento 1: Batch size 2
python train_lightning.py --config config1.json --output_dir ./exp1

# Experimento 2: Batch size 4
python train_lightning.py --config config2.json --output_dir ./exp2

# Experimento 3: Learning rate diferente
python train_lightning.py --config config3.json --output_dir ./exp3
```

No W&B:
1. Vá para o projeto
2. Selecione os runs
3. Clique em "Compare"
4. Veja gráficos lado a lado

### Tabela de Comparação

| Run | Batch Size | LR | Loss Final | Tempo |
|-----|------------|----|-----------:|-------|
| exp1 | 2 | 0.0002 | 12.3 | 18h |
| exp2 | 4 | 0.0002 | 11.8 | 10h |
| exp3 | 4 | 0.0001 | 10.5 | 12h |

W&B gera essa tabela automaticamente!

## 🔔 Alertas e Notificações

### Configurar Alertas

No W&B web interface:

1. Vá para o run
2. Clique em "Alerts"
3. Configure:
   - "Notify me when loss < 15"
   - "Notify me when training completes"
   - "Notify me when GPU usage < 50%"

Você receberá:
- Email
- Notificação no app móvel
- Slack (se configurado)

## 📱 App Móvel

Baixe o app W&B:
- iOS: https://apps.apple.com/app/wandb/id1462693001
- Android: https://play.google.com/store/apps/details?id=com.wandb.wandb

Monitore seu treinamento do celular! 📱

## 💡 Dicas Avançadas

### 1. Adicionar Métricas Customizadas

Em `train_lightning.py`, no `training_step`:

```python
# Logar métricas extras
self.log('custom/my_metric', custom_value)
```

W&B capturará automaticamente.

### 2. Logar Imagens/Áudios

```python
import wandb

# No training_step ou validation_step
if batch_idx % 100 == 0:
    # Logar espectrograma
    self.logger.experiment.log({
        "spectrograms": wandb.Image(spec_image),
        "audio": wandb.Audio(waveform, sample_rate=22050)
    })
```

### 3. Salvar Artefatos

```python
# Salvar checkpoint como artifact
artifact = wandb.Artifact('model-checkpoint', type='model')
artifact.add_file('checkpoint.pth')
self.logger.experiment.log_artifact(artifact)
```

### 4. Hyperparameter Sweeps

Crie `sweep_config.yaml`:

```yaml
program: train_lightning.py
method: bayes
metric:
  name: val/loss_total
  goal: minimize
parameters:
  lr_gen:
    values: [0.0001, 0.0002, 0.0003]
  batch_size:
    values: [2, 4, 8]
```

Execute:

```bash
wandb sweep sweep_config.yaml
wandb agent your-sweep-id
```

W&B testará todas as combinações automaticamente!

### 5. Modo Offline

Se não quiser subir dados para cloud:

```bash
# Antes de treinar
export WANDB_MODE=offline

python train_lightning.py --config config_ljspeech_wandb.json
```

Depois sincronize:

```bash
wandb sync outputs_wandb/wandb/
```

## 🆚 W&B vs TensorBoard

| Feature | W&B | TensorBoard |
|---------|-----|-------------|
| Interface | Web moderna | Web básica |
| Acesso remoto | ✅ Sim | ❌ Precisa VPN/túnel |
| Comparação de runs | ✅ Fácil | ⚠️ Limitado |
| Logs de sistema | ✅ Automático | ❌ Não |
| Compartilhamento | ✅ Link simples | ❌ Difícil |
| Versionamento | ✅ Automático | ❌ Manual |
| Alertas | ✅ Email/Slack | ❌ Não |
| App móvel | ✅ iOS/Android | ❌ Não |
| Custo | Gratuito (100GB) | Gratuito |
| Offline | ✅ Sim | ✅ Sim |

## 🎓 Recursos de Aprendizado

- **Docs:** https://docs.wandb.ai/
- **Tutoriais:** https://wandb.ai/site/tutorials
- **Exemplos:** https://github.com/wandb/examples
- **Lightning + W&B:** https://docs.wandb.ai/guides/integrations/lightning

## 🔧 Troubleshooting

### Erro: "wandb: ERROR Failed to launch TensorBoard"

**Solução:** Ignore, não afeta treinamento.

### Erro: "wandb.errors.UsageError: api_key not configured"

**Solução:**
```bash
wandb login
# Ou
export WANDB_API_KEY=your-key-here
```

### W&B Muito Lento

**Solução:**
```bash
# Reduzir frequência de logging
# Em train_lightning.py
trainer = pl.Trainer(
    log_every_n_steps=100  # Ao invés de 25
)
```

### Não Quer Usar W&B

**Solução:**
```json
// config.json
{
  "use_wandb": false  // Volta para TensorBoard
}
```

## ✅ Checklist para Usar W&B

- [ ] W&B instalado (`pip install wandb`)
- [ ] Login feito (`wandb login`)
- [ ] Config atualizado (`"use_wandb": true`)
- [ ] Projeto nomeado (`"wandb_project": "..."`)
- [ ] Tags adicionadas (opcional mas recomendado)
- [ ] Treinamento iniciado
- [ ] Link do W&B copiado
- [ ] Monitorando no navegador

## 🎯 Exemplo Completo

```bash
# 1. Setup
conda activate yourtts
pip install wandb
wandb login

# 2. Treinar com W&B
python train_lightning.py \
    --config config_ljspeech_wandb.json \
    --output_dir ./outputs_wandb

# 3. Abrir link que aparece no terminal
# Exemplo: https://wandb.ai/username/yourtts-ljspeech/runs/abc123

# 4. Monitorar em tempo real!
```

---

**Criado em:** 2026-02-03
**Versão:** 1.0
**Status:** ✅ Pronto para uso
**Wandb Version:** 0.16.x+
