import numpy as np
import pytest
import torch

from sqpilot.model import masked_mean
from sqpilot.smoke import feature_cache


def test_masked_pooling_averages_four_layers_over_only_valid_frames():
    layers = [torch.tensor([[[1.+layer], [3.+layer], [999.]]]) for layer in range(4)]
    assert masked_mean(layers,torch.tensor([[True,True,False]])).tolist() == [[3.5]]
    with pytest.raises(ValueError,match="Invalid"):
        masked_mean(layers,torch.tensor([[False,False,False]]))


def test_feature_cache_rejects_stale_key_and_preserves_prediction_input(tmp_path):
    path = tmp_path/"features.npz"
    values = np.arange(2*768,dtype=np.float32).reshape(2,768)
    fresh,hit = feature_cache(path,"same",2,lambda:values)
    assert not hit
    cached,hit = feature_cache(path,"same",2,lambda:pytest.fail("cache should be read"))
    assert hit and np.array_equal(fresh,cached)
    with pytest.raises(ValueError,match="Stale"):
        feature_cache(path,"changed",2,lambda:values)


def test_stock_and_feature_frame_padding_match_on_tiny_wavlm():
    from transformers import WavLMConfig, WavLMModel, Wav2Vec2FeatureExtractor
    from sqpilot.model import feature, padded_frames

    torch.set_num_threads(2)
    torch.manual_seed(101)
    model = WavLMModel(WavLMConfig(
        hidden_size=48, num_hidden_layers=4, num_attention_heads=3, intermediate_size=64,
        conv_dim=(16,)*7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4,
        apply_spec_augment=False,
    )).eval()
    extractor = Wav2Vec2FeatureExtractor(do_normalize=False,return_attention_mask=True)
    signal = np.random.default_rng(101).standard_normal(6400).astype(np.float32)
    expected = feature(model,extractor,signal)
    np.testing.assert_allclose(feature(model,extractor,signal,pad=640),expected,rtol=0,atol=0)
    np.testing.assert_allclose(padded_frames(model,extractor,signal,5),expected,rtol=1e-4,atol=1e-5)
