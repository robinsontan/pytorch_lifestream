import torch
from ptls.frames.tabformer import TabformerPretrainModule
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn import TrxEncoder, TransformerEncoder
from ptls.nn.trx_encoder.tabformer_feature_encoder import TabFormerFeatureEncoder


def test_tabformer_dtype_fix():
    """Test that TabFormer correctly handles int32 labels by converting to int64."""
    # Setup trx_encoder with embeddings
    trx_encoder = TrxEncoder(
        embeddings={
            'cat_feature_1': {'in': 10, 'out': 8},
            'cat_feature_2': {'in': 10, 'out': 8},
        },
    )
    
    # Setup feature encoder
    feature_encoder = TabFormerFeatureEncoder(
        n_cols=2,
        emb_dim=8,
        n_heads=2,
        n_layers=1,
    )
    
    # Setup seq_encoder
    seq_encoder = TransformerEncoder(
        input_size=16,  # 2 features * 8 emb_dim
        n_layers=1,
        n_heads=2,
        dim_hidden=32,
    )
    
    # Create TabFormer module
    model = TabformerPretrainModule(
        trx_encoder=trx_encoder,
        feature_encoder=feature_encoder,
        seq_encoder=seq_encoder,
        total_steps=100,
        max_lr=0.001,
        mask_prob=0.15,
    )
    
    # Create a batch with int32 data (simulating the real-world scenario)
    batch_size = 4
    seq_len = 10
    
    # Create features with int32 dtype (this is the root cause of the error)
    cat_feature_1 = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.int32)
    cat_feature_2 = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.int32)
    
    batch = PaddedBatch(
        payload={
            'cat_feature_1': cat_feature_1,
            'cat_feature_2': cat_feature_2,
        },
        length=torch.full((batch_size,), seq_len, dtype=torch.long),
    )
    
    # This should not raise an error about int32 vs int64
    # The fix converts labels to .long() in loss_tabformer method
    try:
        loss = model.training_step(batch, 0)
        assert loss is not None
        assert torch.isfinite(loss).item()
        print(f"Test passed! Loss: {loss.item()}")
    except RuntimeError as e:
        if "not implemented for 'Int'" in str(e):
            raise AssertionError(
                f"Data type error not fixed: {e}\n"
                "The labels should be converted to torch.long before passing to CrossEntropyLoss"
            )
        raise


def test_tabformer_with_long_dtype():
    """Test that TabFormer works correctly with int64 labels."""
    # Setup trx_encoder with embeddings
    trx_encoder = TrxEncoder(
        embeddings={
            'cat_feature_1': {'in': 10, 'out': 8},
            'cat_feature_2': {'in': 10, 'out': 8},
        },
    )
    
    # Setup feature encoder
    feature_encoder = TabFormerFeatureEncoder(
        n_cols=2,
        emb_dim=8,
        n_heads=2,
        n_layers=1,
    )
    
    # Setup seq_encoder
    seq_encoder = TransformerEncoder(
        input_size=16,  # 2 features * 8 emb_dim
        n_layers=1,
        n_heads=2,
        dim_hidden=32,
    )
    
    # Create TabFormer module
    model = TabformerPretrainModule(
        trx_encoder=trx_encoder,
        feature_encoder=feature_encoder,
        seq_encoder=seq_encoder,
        total_steps=100,
        max_lr=0.001,
        mask_prob=0.15,
    )
    
    # Create a batch with int64 data (already correct)
    batch_size = 4
    seq_len = 10
    
    # Create features with int64 dtype (already correct)
    cat_feature_1 = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
    cat_feature_2 = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
    
    batch = PaddedBatch(
        payload={
            'cat_feature_1': cat_feature_1,
            'cat_feature_2': cat_feature_2,
        },
        length=torch.full((batch_size,), seq_len, dtype=torch.long),
    )
    
    # This should work without any issues
    loss = model.training_step(batch, 0)
    assert loss is not None
    assert torch.isfinite(loss).item()
    print(f"Test passed! Loss: {loss.item()}")
