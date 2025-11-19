import torch
from ptls.frames.tabformer.tabformer_module import TabformerPretrainModule
from ptls.nn.trx_encoder.trx_encoder import TrxEncoder
from ptls.nn.trx_encoder.tabformer_feature_encoder import TabFormerFeatureEncoder
from ptls.nn import TransformerEncoder
from ptls.data_load.padded_batch import PaddedBatch


def test_tabformer_training_step():
    """Test that training_step works with proper broadcasting"""
    # Setup trx_encoder with embeddings
    trx_encoder = TrxEncoder(
        embeddings={
            'feature1': {'in': 10, 'out': 8},
            'feature2': {'in': 20, 'out': 8},
            'feature3': {'in': 15, 'out': 8},
        },
    )
    
    # Setup feature encoder
    feature_encoder = TabFormerFeatureEncoder(
        n_cols=3,  # 3 features
        emb_dim=8,
        transf_feedforward_dim=32,
        n_heads=2,
        n_layers=1,
    )
    
    # Setup seq encoder
    seq_encoder = TransformerEncoder(input_size=24)  # 3 features * 8 dims
    
    # Create TabformerPretrainModule
    model = TabformerPretrainModule(
        trx_encoder=trx_encoder,
        feature_encoder=feature_encoder,
        seq_encoder=seq_encoder,
        total_steps=100,
        max_lr=0.001,
        mask_prob=0.15,
    )
    
    # Create a batch
    B, T = 4, 10
    batch = PaddedBatch(
        payload={
            'feature1': torch.randint(0, 9, (B, T)),
            'feature2': torch.randint(0, 19, (B, T)),
            'feature3': torch.randint(0, 14, (B, T)),
        },
        length=torch.randint(5, T, (B,)),
    )
    
    # Test training step
    loss = model.training_step(batch, 0)
    assert loss is not None
    assert loss.item() >= 0


def test_tabformer_validation_step():
    """Test that validation_step works with proper broadcasting"""
    # Setup trx_encoder with embeddings
    trx_encoder = TrxEncoder(
        embeddings={
            'feature1': {'in': 10, 'out': 8},
            'feature2': {'in': 20, 'out': 8},
        },
    )
    
    # Setup feature encoder
    feature_encoder = TabFormerFeatureEncoder(
        n_cols=2,  # 2 features
        emb_dim=8,
        transf_feedforward_dim=32,
        n_heads=2,
        n_layers=1,
    )
    
    # Setup seq encoder
    seq_encoder = TransformerEncoder(input_size=16)  # 2 features * 8 dims
    
    # Create TabformerPretrainModule
    model = TabformerPretrainModule(
        trx_encoder=trx_encoder,
        feature_encoder=feature_encoder,
        seq_encoder=seq_encoder,
        total_steps=100,
        max_lr=0.001,
        mask_prob=0.15,
    )
    
    # Create a batch
    B, T = 4, 10
    batch = PaddedBatch(
        payload={
            'feature1': torch.randint(0, 9, (B, T)),
            'feature2': torch.randint(0, 19, (B, T)),
        },
        length=torch.randint(5, T, (B,)),
    )
    
    # Test validation step
    model.validation_step(batch, 0)
    # If we get here without error, the test passes
    assert True
