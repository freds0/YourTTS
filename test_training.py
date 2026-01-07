#!/usr/bin/env python3
"""Script de teste para verificar se o treinamento YourTTS Lightning está funcionando.

Este script executa testes básicos para validar:
1. Imports funcionam corretamente
2. Modelo pode ser criado
3. Forward pass funciona
4. DataModule funciona
5. Training step funciona
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Adicionar TTS ao path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Teste 1: Verificar se todos os imports funcionam."""
    print("\n" + "="*80)
    print("TESTE 1: Verificando imports...")
    print("="*80)

    try:
        import pytorch_lightning as pl
        print(f"✓ PyTorch Lightning: {pl.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch Lightning não instalado: {e}")
        return False

    try:
        from TTS.tts.models.vits_lightning import YourTTSLightningModule
        print("✓ YourTTSLightningModule importado")
    except ImportError as e:
        print(f"✗ Erro ao importar YourTTSLightningModule: {e}")
        return False

    try:
        from TTS.tts.datasets.yourtts_datamodule import YourTTSDataModule
        print("✓ YourTTSDataModule importado")
    except ImportError as e:
        print(f"✗ Erro ao importar YourTTSDataModule: {e}")
        return False

    try:
        from TTS.tts.configs.vits_config import VitsConfig
        print("✓ VitsConfig importado")
    except ImportError as e:
        print(f"✗ Erro ao importar VitsConfig: {e}")
        return False

    print("\n✅ Todos os imports funcionaram!\n")
    return True


def test_model_creation():
    """Teste 2: Verificar se o modelo pode ser criado."""
    print("\n" + "="*80)
    print("TESTE 2: Criando modelo...")
    print("="*80)

    try:
        from TTS.tts.models.vits_lightning import YourTTSLightningModule
        from TTS.tts.configs.vits_config import VitsConfig

        # Criar configuração mínima
        config = VitsConfig()

        # Configuração mínima para teste
        config.num_chars = 100
        config.model_args.num_speakers = 1
        config.model_args.use_speaker_embedding = False
        config.model_args.use_d_vector_file = False
        config.model_args.init_discriminator = True

        print("Criando modelo...")
        model = YourTTSLightningModule(config)

        print(f"✓ Modelo criado com sucesso")
        print(f"  - Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Device: {model.device}")

        print("\n✅ Modelo criado com sucesso!\n")
        return True, model, config

    except Exception as e:
        print(f"✗ Erro ao criar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_forward_pass(model, config):
    """Teste 3: Verificar se forward pass funciona."""
    print("\n" + "="*80)
    print("TESTE 3: Testando forward pass...")
    print("="*80)

    try:
        # Criar batch sintético
        batch_size = 2
        max_text_len = 50
        max_spec_len = 100

        # Dados sintéticos
        tokens = torch.randint(0, config.num_chars, (batch_size, max_text_len))
        token_lens = torch.tensor([max_text_len, max_text_len])

        spec = torch.randn(batch_size, config.model_args.out_channels, max_spec_len)
        spec_lens = torch.tensor([max_spec_len, max_spec_len])

        waveform = torch.randn(batch_size, 1, max_spec_len * config.audio.hop_length)

        print(f"Batch sintético criado:")
        print(f"  - tokens: {tokens.shape}")
        print(f"  - spec: {spec.shape}")
        print(f"  - waveform: {waveform.shape}")

        # Forward pass
        print("\nExecutando forward pass...")
        with torch.no_grad():
            outputs = model.vits_model.forward(
                tokens,
                token_lens,
                spec,
                spec_lens,
                waveform,
                aux_input={},
            )

        print(f"✓ Forward pass bem-sucedido")
        print(f"  - Output waveform: {outputs['model_outputs'].shape}")
        print(f"  - z shape: {outputs['z'].shape}")
        print(f"  - z_p shape: {outputs['z_p'].shape}")

        print("\n✅ Forward pass funcionando!\n")
        return True

    except Exception as e:
        print(f"✗ Erro no forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(model, config):
    """Teste 4: Verificar se training step funciona."""
    print("\n" + "="*80)
    print("TESTE 4: Testando training step...")
    print("="*80)

    try:
        # Criar batch sintético completo
        batch_size = 2
        max_text_len = 50
        max_spec_len = 100

        batch = {
            'tokens': torch.randint(0, config.num_chars, (batch_size, max_text_len)),
            'token_lens': torch.tensor([max_text_len, max_text_len]),
            'linear': torch.randn(batch_size, config.model_args.out_channels, max_spec_len),
            'mel_lengths': torch.tensor([max_spec_len, max_spec_len]),
            'waveform': torch.randn(batch_size, 1, max_spec_len * config.audio.hop_length),
            'mel': torch.randn(batch_size, config.audio.num_mels, max_spec_len),
            'd_vectors': None,
            'speaker_ids': None,
            'language_ids': None,
        }

        print("Configurando otimizadores...")
        model.configure_optimizers()

        print("Executando training step...")
        # Note: O training step precisa de manual optimization
        # Aqui apenas testamos se o código roda
        model.automatic_optimization = False

        # Simular os otimizadores manualmente
        from itertools import chain
        opt_disc = torch.optim.AdamW(model.vits_model.disc.parameters(), lr=2e-4)
        opt_gen = torch.optim.AdamW(
            chain(
                model.vits_model.text_encoder.parameters(),
                model.vits_model.posterior_encoder.parameters(),
                model.vits_model.flow.parameters(),
                model.vits_model.waveform_decoder.parameters(),
                model.vits_model.duration_predictor.parameters(),
            ),
            lr=2e-4
        )

        # Monkey patch para teste
        model.optimizers = lambda: [opt_disc, opt_gen]
        model.manual_backward = lambda x: x.backward()

        result = model.training_step(batch, 0)

        print(f"✓ Training step executado")
        print(f"  - Loss: {result['loss']:.4f}")

        print("\n✅ Training step funcionando!\n")
        return True

    except Exception as e:
        print(f"✗ Erro no training step: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Teste 5: Verificar se config pode ser carregado."""
    print("\n" + "="*80)
    print("TESTE 5: Testando carregamento de config...")
    print("="*80)

    try:
        from TTS.tts.configs.vits_config import VitsConfig

        config_path = "recipes/vctk/yourtts/yourtts_lightning_config.json"

        if Path(config_path).exists():
            print(f"Carregando config: {config_path}")
            config = VitsConfig()
            config.load_json(config_path)

            print(f"✓ Config carregado com sucesso")
            print(f"  - Model: {config.model}")
            print(f"  - Batch size: {config.batch_size}")
            print(f"  - Epochs: {config.epochs}")
            print(f"  - Use d-vectors: {config.model_args.use_d_vector_file}")
            print(f"  - Text encoder layers: {config.model_args.num_layers_text_encoder}")

            print("\n✅ Config carregado!\n")
            return True
        else:
            print(f"⚠️  Config não encontrado: {config_path}")
            print("   (Isso é normal se você ainda não criou o arquivo)")
            return True

    except Exception as e:
        print(f"✗ Erro ao carregar config: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_compatibility():
    """Teste 6: Verificar compatibilidade com checkpoints."""
    print("\n" + "="*80)
    print("TESTE 6: Testando salvamento/carregamento de checkpoint...")
    print("="*80)

    try:
        from TTS.tts.models.vits_lightning import YourTTSLightningModule
        from TTS.tts.configs.vits_config import VitsConfig
        import tempfile

        # Criar modelo
        config = VitsConfig()
        config.num_chars = 100
        config.model_args.num_speakers = 1
        config.model_args.use_speaker_embedding = False
        config.model_args.use_d_vector_file = False
        config.model_args.init_discriminator = True

        model = YourTTSLightningModule(config)

        # Salvar checkpoint
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
            checkpoint_path = f.name

        print(f"Salvando checkpoint em: {checkpoint_path}")

        # Simular salvamento de checkpoint
        checkpoint = {
            'state_dict': model.state_dict(),
            'model': model.vits_model.state_dict(),
            'config': config,
            'epoch': 0,
            'global_step': 0,
        }

        torch.save(checkpoint, checkpoint_path)
        print("✓ Checkpoint salvo")

        # Carregar checkpoint
        print("Carregando checkpoint...")
        loaded_model = YourTTSLightningModule.load_from_checkpoint(
            checkpoint_path,
            config=config,
        )

        print("✓ Checkpoint carregado")

        # Limpar
        import os
        os.remove(checkpoint_path)

        print("\n✅ Checkpoint save/load funcionando!\n")
        return True

    except Exception as e:
        print(f"✗ Erro com checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Executar todos os testes."""
    print("\n" + "="*80)
    print("TESTE DE TREINAMENTO YOURTTS LIGHTNING")
    print("="*80)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("="*80)

    results = {}

    # Teste 1: Imports
    results['imports'] = test_imports()
    if not results['imports']:
        print("\n❌ Testes interrompidos devido a falha nos imports")
        return False

    # Teste 2: Criação do modelo
    success, model, config = test_model_creation()
    results['model_creation'] = success
    if not success:
        print("\n❌ Testes interrompidos devido a falha na criação do modelo")
        return False

    # Teste 3: Forward pass
    results['forward_pass'] = test_forward_pass(model, config)

    # Teste 4: Training step
    results['training_step'] = test_training_step(model, config)

    # Teste 5: Config loading
    results['config_loading'] = test_config_loading()

    # Teste 6: Checkpoint compatibility
    results['checkpoint'] = test_checkpoint_compatibility()

    # Resumo
    print("\n" + "="*80)
    print("RESUMO DOS TESTES")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"{status} - {test_name}")

    all_passed = all(results.values())

    print("="*80)
    if all_passed:
        print("\n🎉 TODOS OS TESTES PASSARAM! 🎉")
        print("\nVocê pode prosseguir com o treinamento usando:")
        print("python TTS/bin/train_yourtts_lightning.py --config_path <seu_config.json>")
    else:
        print("\n⚠️  ALGUNS TESTES FALHARAM")
        print("Verifique os erros acima antes de prosseguir com o treinamento.")
    print("\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
