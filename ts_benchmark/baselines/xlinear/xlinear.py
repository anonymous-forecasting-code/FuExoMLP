import torch

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.xlinear.models.xlinear_model import XLinearModel


MODEL_HYPER_PARAMS = {
    "enc_in": 1,
    "d_model": 512,
    "t_ff": 1024,
    "c_ff": 16,
    "features": "MS",
    "usenorm": True,
    "embed_dropout": 0.0,
    "head_dropout": 0.0,
    "t_dropout": 0.0,
    "c_dropout": 0.0,
    "batch_size": 16,
    "lradj": "type1",
    "lr": 0.0001,
    "num_epochs": 10,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 3,
    "fusion_method": "mlp",
    "mlp_hidden_dims": 128,
}


class XLinear(DeepForecastingModelBase):
    """
    XLinear adapter for covariate forecasting.

    The source XLinear model expects exogenous channels first and the target
    channel last. This benchmark passes target channels first and historical
    exogenous channels after them, so the adapter reorders the history before
    calling the model. Future exogenous values are enabled by default through
    DeepForecastingModelBase's fusion_method path.
    """

    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "XLinear"

    def _init_model(self):
        self.config.usenorm = bool(getattr(self.config, "norm", self.config.usenorm))
        self.config.features = str(getattr(self.config, "features", "MS")).upper()
        if self.config.features != "M" and getattr(self.config, "series_dim", 1) != 1:
            raise ValueError(
                "XLinear with exogenous inputs supports one endogenous target channel. "
                "Use target_channel [-1] or set features='M' for all-channel forecasting."
            )
        return XLinearModel(self.config)

    def _reorder_input_for_xlinear(self, input):
        if self.config.features == "M":
            return input

        series_dim = self.config.series_dim
        target_history = input[:, :, :series_dim]
        exog_history = input[:, :, series_dim:]
        if exog_history.shape[-1] == 0:
            return target_history
        return torch.cat([exog_history, target_history], dim=-1)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        output = self.model(self._reorder_input_for_xlinear(input))
        return {"output": output}
