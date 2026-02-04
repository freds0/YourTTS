# YourTTS - PyTorch Lightning Implementation

Este repositório contém uma implementação alternativa do treinamento do YourTTS usando PyTorch Lightning, mantendo a mesma lógica do treinamento original.

## Arquivos Criados

### 1. `train_lightning.py`
Script principal de treinamento usando PyTorch Lightning.

**Características principais:**
- **VitsLightningModule**: Módulo Lightning que encapsula o modelo VITS
- **Otimização Manual**: Usa `automatic_optimization = False` para controlar os dois otimizadores (discriminador e gerador)
- **Reusa Código Existente**: Chama diretamente `model.train_step()` e `model.eval_step()` do VITS original
- **Mesma Lógica**: Mantém exatamente a mesma ordem de operações do treinamento original

**Estrutura do training_step:**
```python
def training_step(self, batch, batch_idx):
    # 1. Passo do Discriminador (optimizer_idx=0)
    outputs_disc, loss_dict_disc = self.model.train_step(batch, self.criterion, optimizer_idx=0)
    optimizer_disc.zero_grad()
    self.manual_backward(loss_dict_disc['loss'])
    torch.nn.utils.clip_grad_norm_(disc_params, grad_clip)
    optimizer_disc.step()

    # 2. Passo do Gerador (optimizer_idx=1)
    outputs_gen, loss_dict_gen = self.model.train_step(batch, self.criterion, optimizer_idx=1)
    optimizer_gen.zero_grad()
    self.manual_backward(loss_dict_gen['loss'])
    torch.nn.utils.clip_grad_norm_(gen_params, grad_clip)
    optimizer_gen.step()

    # 3. Atualizar learning rate schedulers
    scheduler_disc.step()
    scheduler_gen.step()

    return total_loss
```

**Uso:**
```bash
# Ativar ambiente conda
conda activate yourtts

# Treinar com configuração existente
python train_lightning.py --config /path/to/config.json --output_dir ./outputs_lightning

# Treinar com configuração customizada
python train_lightning.py --config my_config.json --output_dir ./my_training
```

### 2. `test_training_comparison.py`
Script de teste para comparar um step de treinamento entre as implementações original e Lightning.

**O que o teste faz:**
1. Cria uma configuração mínima de teste
2. Inicializa o modelo VITS original
3. Inicializa o modelo Lightning com os mesmos pesos
4. Cria um batch dummy com dados aleatórios (mas consistentes via seed)
5. Executa um step de treinamento completo em ambos
6. Compara:
   - Valores de loss (discriminador, gerador, total)
   - Tensores de saída (waveform gerado, latentes, etc.)
   - Diferenças relativas e absolutas

**Uso:**
```bash
# Teste com configuração mínima (padrão)
python test_training_comparison.py

# Teste com configuração existente
python test_training_comparison.py --config /path/to/config.json
```

**Exemplo de saída esperada:**
```
================================================================================
YourTTS Training Comparison Test
================================================================================

Using device: cuda

Creating minimal test config...
Config loaded successfully

Initializing original VITS model...
Original model initialized with 89724076 parameters

Initializing Lightning model...
Copying weights from original to Lightning model...
Weights copied successfully

Creating dummy batch...
Batch created with:
  - Tokens: torch.Size([2, 50])
  - Waveform: torch.Size([2, 1, 66150])
  - D-vectors: torch.Size([2, 512])

Running original training step...
✓ Original training step completed
  - Discriminator loss: {...}
  - Generator loss: {...}

Running Lightning training step...
✓ Lightning training step completed
  - Total loss: ...

================================================================================
Comparison Results
================================================================================

Comparing losses:
----------------------------------------
Original total loss: 123.456789
Lightning total loss: 123.456790

Loss difference: 1.234e-06
Relative difference: 0.001%

Comparing model outputs:
----------------------------------------
✓ Generated waveform:
   Max diff: 1.234e-06
   Mean diff: 5.678e-07

✓ Latent z_p:
   Max diff: 2.345e-06
   Mean diff: 1.234e-06

================================================================================
✓ TEST PASSED: Original and Lightning implementations are equivalent!
================================================================================
```

## Arquitetura do Treinamento VITS

### Dual Optimizer Strategy

O VITS usa duas etapas de otimização por batch:

#### **Passo 1: Discriminador (optimizer_idx=0)**
- Forward pass completo do gerador
- Computa scores do discriminador (real vs fake)
- Calcula loss do discriminador
- **Cacheia outputs** para uso no próximo passo
- Atualiza apenas pesos do discriminador

#### **Passo 2: Gerador (optimizer_idx=1)**
- Reutiliza outputs cacheados
- Computa mel-spectrogram do áudio gerado
- Calcula todas as losses do gerador:
  - KL divergence (VAE loss)
  - Generator adversarial loss
  - Feature matching loss
  - Mel reconstruction loss
  - Duration predictor loss
  - Speaker encoder loss (opcional)
- Atualiza apenas pesos do gerador

### Componentes de Loss

**Discriminador:**
- `loss_disc`: Mean squared error entre scores real/fake
- Alpha padrão: 1.0

**Gerador:**
- `loss_kl`: KL divergence (VAE) - alpha: 1.0
- `loss_gen`: Adversarial generator loss - alpha: 1.0
- `loss_feat`: Feature matching - alpha: 1.0
- `loss_mel`: L1 mel reconstruction - alpha: 45.0
- `loss_duration`: Duration predictor - alpha: 1.0
- `loss_spk_encoder`: Speaker consistency (opcional) - alpha: 9.0

## Vantagens da Implementação Lightning

1. **Compatibilidade Total**: Usa os métodos `train_step()` e `eval_step()` existentes
2. **Sem Modificações**: Não altera nenhum código existente do TTS
3. **Callbacks Integrados**: Checkpointing, logging, early stopping prontos
4. **Multi-GPU Fácil**: Adicione `devices=[0, 1]` e `strategy='ddp'`
5. **16-bit Training**: Configure `precision='16-mixed'`
6. **Progress Bar**: Barra de progresso automática
7. **TensorBoard**: Logger integrado para visualização
8. **Resumo de Modelo**: Visualização automática da arquitetura

## Diferenças em Relação ao Original

### Mantido Igual:
- ✓ Ordem exata dos passos de treinamento
- ✓ Mesmas funções de loss
- ✓ Mesmo processamento de batch
- ✓ Mesmos hiperparâmetros
- ✓ Mesma arquitetura de modelo
- ✓ Mesmos otimizadores e schedulers
- ✓ Mesmo gradient clipping

### Diferenças (apenas infraestrutura):
- Framework de treinamento (Trainer → Lightning Trainer)
- Sistema de logging (custom → TensorBoard Logger)
- Sistema de callbacks (custom → Lightning Callbacks)
- Salvamento de checkpoints (custom → ModelCheckpoint)

## Requisitos

```bash
# Instalar PyTorch Lightning
conda activate yourtts
pip install pytorch-lightning

# Ou adicionar ao requirements.txt:
pytorch-lightning>=2.0.0
```

## Extensões Futuras

Com PyTorch Lightning, é fácil adicionar:

1. **Multi-GPU Training:**
```python
trainer = pl.Trainer(
    devices=[0, 1, 2, 3],
    strategy='ddp'
)
```

2. **Mixed Precision:**
```python
trainer = pl.Trainer(
    precision='16-mixed'  # ou 'bf16-mixed'
)
```

3. **Early Stopping:**
```python
from pytorch_lightning.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val/loss_total',
    patience=10,
    mode='min'
)

trainer = pl.Trainer(callbacks=[early_stop])
```

4. **Learning Rate Finder:**
```python
from pytorch_lightning.tuner import Tuner

tuner = Tuner(trainer)
lr_finder = tuner.lr_find(model, train_loader)
model.learning_rate = lr_finder.suggestion()
```

5. **Gradient Accumulation:**
```python
trainer = pl.Trainer(
    accumulate_grad_batches=4  # Simula batch 4x maior
)
```

## Estrutura de Diretórios

```
YourTTS-lightning/
├── train_lightning.py           # Script de treinamento Lightning
├── test_training_comparison.py  # Teste de comparação
├── README_LIGHTNING.md          # Esta documentação
├── TTS/                         # Código original (não modificado)
│   ├── tts/
│   │   ├── models/
│   │   │   └── vits.py         # Modelo VITS original
│   │   ├── layers/
│   │   │   └── losses.py       # Funções de loss
│   │   └── configs/
│   │       └── vits_config.py  # Configurações
│   └── utils/
└── outputs_lightning/           # Outputs do treinamento Lightning
    ├── checkpoints/
    ├── lightning_logs/
    └── tensorboard/
```

## Troubleshooting

### Erro: "No module named 'pytorch_lightning'"
```bash
conda activate yourtts
pip install pytorch-lightning
```

### Erro: "CUDA out of memory"
- Reduza `batch_size` na configuração
- Use `precision='16-mixed'` para economizar memória
- Use `accumulate_grad_batches` para simular batches maiores

### Erro: "AttributeError: 'AudioProcessor' object has no attribute 'stft'"
- Use as funções `wav_to_spec` e `spec_to_mel` do módulo `vits.py`
- Veja o teste de comparação para exemplo

## Validação

O script `test_training_comparison.py` valida que:
1. Os outputs do modelo são idênticos (dentro de tolerância numérica)
2. As losses são calculadas da mesma forma
3. A ordem de operações é a mesma
4. Os gradientes são aplicados corretamente

Execute o teste regularmente para garantir que as modificações mantêm a equivalência.

## Contribuindo

Para adicionar novas funcionalidades:
1. Mantenha a compatibilidade com o código original
2. Não modifique arquivos em `TTS/`
3. Adicione testes em `test_training_comparison.py`
4. Documente mudanças neste README

## Licença

Mesmo que o projeto original YourTTS.

## Contato

Para questões sobre a implementação Lightning, abra uma issue no repositório.
