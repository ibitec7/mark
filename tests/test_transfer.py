import pytest
import torch

model = torch.load("models/hydra_23layers.pt", weights_only=False)
guider = torch.load("models/hydra_guider_12layers.pt", weights_only=False)

model_dict = torch.load("models/hydra_bert_23layers.pt", weights_only=False)
model_dict = model_dict["state"]["model"]

@pytest.xfail(reason="Some weights have been truncated and new weights introduced beyond the pre-trained model")
def test_embedding_transfer():
    """
    Test if the embedding layer weights are correctly transferred.
    """
    assert torch.all(model["hydra.embeddings.LayerNorm.weight"] == model_dict["model.bert.embeddings.LayerNorm.weight"]), "LayerNorm weights do not match!"
    assert torch.all(model["hydra.embeddings.LayerNorm.bias"] == model_dict["model.bert.embeddings.LayerNorm.bias"]), "LayerNorm biases do not match!"
    assert torch.all(model["hydra.embeddings.word_embeddings.weight"] == model_dict["model.bert.embeddings.word_embeddings.weight"]), "Word embeddings do not match!"
    assert torch.all(model["hydra.embeddings.token_type_embeddings.weight"] == model_dict["model.bert.embeddings.token_type_embeddings.weight"]), "Token type embeddings do not match!"

@pytest.xfail(reason="Some weights have been truncated and new weights introduced beyond the pre-trained model")
def test_encoder_layer_transfer():
    """
        Test if the encoder layer weights are correctly transferred. Which contains multiple deep layers
        of the Hydra SSM mixer.
    """

    for layer_idx in range(0, 23):
        assert torch.all(model[f"hydra.encoder.layer.{layer_idx}.layer.mixer.conv1d.bias"] == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.conv1d.bias"]), f"Conv1D bias for layer {layer_idx} does not match!"
        assert torch.all(model[f"hydra.encoder.layer.{layer_idx}.layer.mixer.conv1d.weight"] == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.conv1d.weight"]), f"Conv1D weight for layer {layer_idx} does not match!"

        assert torch.all(model[f"hydra.encoder.layer.{layer_idx}.layer.mixer.in_proj.weight"] == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.in_proj.weight"]), f"In projection weight for layer {layer_idx} does not match!"

        assert torch.all(model[f"hydra.encoder.layer.{layer_idx}.layer.mixer.A_log"] == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.A_log"]), f"A_log for layer {layer_idx} does not match!"

        assert torch.all(model[f"hydra.encoder.layer.{layer_idx}.layer.mixer.D"] == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.D"]), f"D for layer {layer_idx} does not match!"

        assert torch.all(model[f"hydra.encoder.layer.{layer_idx}.layer.mixer.dt_bias"] == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.dt_bias"]), f"dt_bias for layer {layer_idx} does not match!"

        assert torch.all(model[f"hydra.encoder.layer.{layer_idx}.layer.mixer.fc_D.weight"] == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.fc_D.weight"]), f"fc_D weight for layer {layer_idx} does not match!"

        assert torch.all(model[f"hydra.encoder.layer.{layer_idx}.layer.mixer.norm.weight"] == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.norm.weight"]), f"Norm weight for layer {layer_idx} does not match!"

        assert torch.all(model[f"hydra.encoder.layer.{layer_idx}.layer.mixer.out_proj.weight"] == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.out_proj.weight"]), f"Out projection weight for layer {layer_idx} does not match!"

        assert torch.all(model[f"hydra.encoder.layer.{layer_idx}.layer.norm.bias"] == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.norm.bias"]), f"Norm bias for layer {layer_idx} does not match!"
        assert torch.all(model[f"hydra.encoder.layer.{layer_idx}.layer.norm.weight"] == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.norm.weight"]), f"Norm weight for layer {layer_idx} does not match!"

@pytest.xfail(reason="Some weights have been truncated and new weights introduced beyond the pre-trained model")
def test_decoder_transfer():
    """
        Test if the CLS prediction head decoder weights are correctly transferred.
    """
    assert torch.all(model["cls.predictions.decoder.bias"] == model_dict["model.cls.predictions.decoder.bias"]), "Prediction decoder bias does not match!"
    assert torch.all(model["cls.predictions.decoder.weight"] == model_dict["model.cls.predictions.decoder.weight"]), "Prediction decoder weight does not match!"

def test_transform_layer_transfer():
    """
        Test if the CLS prediction head transform layer weights are correctly transferred.
    """

    assert torch.all(model["cls.predictions.transform.LayerNorm.bias"] == model_dict["model.cls.predictions.transform.LayerNorm.bias"]), "Prediction transform LayerNorm bias does not match!"
    assert torch.all(model["cls.predictions.transform.LayerNorm.weight"] == model_dict["model.cls.predictions.transform.LayerNorm.weight"]), "Prediction transform LayerNorm weight does not match!"
    assert torch.all(model["cls.predictions.transform.dense.bias"] == model_dict["model.cls.predictions.transform.dense.bias"]), "Prediction transform dense bias does not match!"
    assert torch.all(model["cls.predictions.transform.dense.weight"] == model_dict["model.cls.predictions.transform.dense.weight"]), "Prediction transform dense weight does not match!"

@pytest.xfail(reason="Some weights have been truncated and new weights introduced beyond the pre-trained model")
def test_guider_embedding_transfer():
    """
        Test if the Guider model's embedding layer weights are correctly transferred.
    """
    assert torch.all(guider["embeddings.LayerNorm.weight"] == model_dict["model.bert.embeddings.LayerNorm.weight"]), "Guider LayerNorm weights do not match!"
    assert torch.all(guider["embeddings.LayerNorm.bias"] == model_dict["model.bert.embeddings.LayerNorm.bias"]), "Guider LayerNorm biases do not match!"

    assert torch.all(guider["embeddings.word_embeddings.weight"] == model_dict["model.bert.embeddings.word_embeddings.weight"]), "Guider word embeddings do not match!"
    assert torch.all(guider["embeddings.token_type_embeddings.weight"] == model_dict["model.bert.embeddings.token_type_embeddings.weight"]), "Guider token type embeddings do not match!"

@pytest.xfail(reason="Some weights have been truncated and new weights introduced beyond the pre-trained model")
def test_guider_encoder_layer_transfer():
    """
        Test if the Guider model's encoder layer weights are correctly transferred.
    """

    for idx in range(0, 12):
        # Assert the Convolution weights match
        conv_weight = guider[f"encoder.layer.{idx}.layer.mixer.conv1d.weight"]
        conv_bias = guider[f"encoder.layer.{idx}.layer.mixer.conv1d.bias"]

        out_c, in_c, _ = conv_weight.shape

        assert torch.all(conv_bias == model_dict[f"model.bert.encoder.layer.{idx}.layer.mixer.conv1d.bias"][:out_c]), f"Guider encoder layer {idx} mixer conv1d bias does not match!"
        assert torch.all(conv_weight == model_dict[f"model.bert.encoder.layer.{idx}.layer.mixer.conv1d.weight"][:out_c, :in_c, :]), f"Guider encoder layer {idx} mixer conv1d weight does not match!"

        index = guider[f"encoder.layer.{idx}.layer.mixer.in_proj.weight"].shape
        assert torch.all(guider[f"encoder.layer.{idx}.layer.mixer.in_proj.weight"] == model_dict[f"model.bert.encoder.layer.{idx}.layer.mixer.in_proj.weight"][:index[0]]), f"Guider encoder layer {idx} in_proj weight does not match!"

        assert torch.all(guider[f"encoder.layer.{idx}.layer.mixer.A_log"] == model_dict[f"model.bert.encoder.layer.{idx}.layer.mixer.A_log"]), f"Guider encoder layer {idx} mixer A_log does not match!"
        assert torch.all(guider[f"encoder.layer.{idx}.layer.mixer.D"] == model_dict[f"model.bert.encoder.layer.{idx}.layer.mixer.D"]), f"Guider encoder layer {idx} mixer D does not match!"

        assert torch.all(guider[f"encoder.layer.{idx}.layer.mixer.dt_bias"] == model_dict[f"model.bert.encoder.layer.{idx}.layer.mixer.dt_bias"]), f"Guider encoder layer {idx} mixer dt_bias does not match!"

        index = guider[f"encoder.layer.{idx}.layer.mixer.out_proj.weight"].shape
        assert torch.all(guider[f"encoder.layer.{idx}.layer.mixer.out_proj.weight"] == model_dict[f"model.bert.encoder.layer.{idx}.layer.mixer.out_proj.weight"][:index[0]]), f"Guider encoder layer {idx} mixer out_proj.weight does not match!"

def test_guider_transform_layer_transfer():
    """
        Test if the Guider model's transform layer weights are correctly transferred.
    """
    assert torch.all(guider["prediction.transform.LayerNorm.bias"] == model_dict["model.cls.predictions.transform.LayerNorm.bias"]), "Guider prediction transform LayerNorm bias does not match!"
    assert torch.all(guider["prediction.transform.LayerNorm.weight"] == model_dict["model.cls.predictions.transform.LayerNorm.weight"]), "Guider prediction transform LayerNorm weight does not match!"

    assert torch.all(guider["prediction.transform.dense.bias"] == model_dict["model.cls.predictions.transform.dense.bias"]), "Guider prediction transform dense bias does not match!"
    assert torch.all(guider["prediction.transform.dense.weight"] == model_dict["model.cls.predictions.transform.dense.weight"]), "Guider prediction transform dense weight does not match!"