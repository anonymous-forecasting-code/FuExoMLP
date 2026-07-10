import math

import torch
import torch.nn as nn

from ts_benchmark.baselines.deep_forecasting_model_base import (
    DeepForecastingModelBase,
)


MODEL_HYPER_PARAMS = {
    "batch_size": 64,
    "lr": 0.001,
    "lradj": "type3",
    "num_epochs": 50,
    "num_workers": 0,
    "patience": 5,
    "loss": "MAE",
    "norm": True,
    "hidden_dim": 256,
    "history_hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.1,
    "individual": 0,
    "etp_model": "etp",
    "scfm_state_len": 0,
    "scfm_hidden_dim": None,
    "scfm_state_hidden_dim": None,
    "scfm_num_layers": None,
    "scfm_dropout": None,
    "sfta_mode": "adaptive_ratio",
    "model_variant": "auto",
    "future_mark_dim": None,
    "etp_fusion_alpha": 0.5,
    "sfta_ratio": 0.5,
    "task_name": "short_term_forecast",
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "d_model": 128,
    "d_ff": 256,
    "e_layers": 1,
    "factor": 3,
    "n_heads": 4,
    "patch_len": 8,
    "stride": 8,
    "activation": "gelu",
    "output_attention": 0,
    "hist_exog_mode": "none",
    "hist_exog_seq_len": None,
    "hist_exog_hidden_dim": None,
    "hist_exog_num_heads": 4,
}


def _make_mlp(input_dim, hidden_dim, output_dim, num_layers, dropout):
    layers = []
    in_dim = int(input_dim)
    for _ in range(max(int(num_layers), 1)):
        layers.append(nn.Linear(in_dim, int(hidden_dim)))
        layers.append(nn.GELU())
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        in_dim = int(hidden_dim)
    layers.append(nn.Linear(in_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _valid_num_heads(hidden_dim, requested_heads):
    heads = max(1, int(requested_heads))
    hidden_dim = int(hidden_dim)
    while heads > 1 and hidden_dim % heads != 0:
        heads -= 1
    return heads


def _mark_dim_from_freq(freq):
    key = str(freq or "h").lower()[0]
    return {
        "h": 4,
        "t": 5,
        "s": 6,
        "m": 1,
        "a": 1,
        "y": 1,
        "w": 2,
        "d": 3,
        "b": 3,
    }.get(key, 4)


def _future_mark(target_mark, pred_len, expected_dim):
    expected_dim = (
        target_mark.shape[-1] if expected_dim is None else int(expected_dim)
    )
    mark = target_mark[:, -int(pred_len) :, :]
    if mark.shape[-1] > expected_dim:
        return mark[:, :, :expected_dim]
    if mark.shape[-1] < expected_dim:
        padding = mark.new_zeros(
            mark.shape[0], mark.shape[1], expected_dim - mark.shape[-1]
        )
        mark = torch.cat([mark, padding], dim=-1)
    return mark


def _periodic_mark_dim(freq, base_dim):
    key = str(freq or "h").lower()[0]
    if key in {"s", "t", "h"}:
        return 8
    if key in {"d", "b"}:
        return 6
    if key == "w":
        return 4
    if key in {"m", "a", "y"}:
        return 2
    return int(base_dim) * 2


def _mark_value(mark, index, source_scale):
    if mark.shape[-1] <= index:
        return mark.new_zeros(mark.shape[:-1])
    return (mark[..., index] + 0.5) * float(source_scale)


def _append_periodic_pair(parts, phase):
    parts.append(torch.sin(2.0 * math.pi * phase).unsqueeze(-1))
    parts.append(torch.cos(2.0 * math.pi * phase).unsqueeze(-1))


def _periodic_future_mark(target_mark, pred_len, expected_dim, freq):
    mark = _future_mark(target_mark, pred_len, expected_dim)
    key = str(freq or "h").lower()[0]
    parts = []

    if key in {"s", "t"}:
        offset = 1 if key == "s" else 0
        second = (
            _mark_value(mark, 0, 59.0)
            if key == "s"
            else mark.new_zeros(mark.shape[:-1])
        )
        minute = _mark_value(mark, offset, 59.0)
        hour = _mark_value(mark, offset + 1, 23.0)
        time_of_day = (hour + minute / 60.0 + second / 3600.0) / 24.0
        _append_periodic_pair(parts, time_of_day)
        _append_periodic_pair(parts, _mark_value(mark, offset + 2, 6.0) / 7.0)
        _append_periodic_pair(parts, _mark_value(mark, offset + 3, 30.0) / 31.0)
        _append_periodic_pair(parts, _mark_value(mark, offset + 4, 365.0) / 365.0)
    elif key == "h":
        _append_periodic_pair(parts, _mark_value(mark, 0, 23.0) / 24.0)
        _append_periodic_pair(parts, _mark_value(mark, 1, 6.0) / 7.0)
        _append_periodic_pair(parts, _mark_value(mark, 2, 30.0) / 31.0)
        _append_periodic_pair(parts, _mark_value(mark, 3, 365.0) / 365.0)
    elif key in {"d", "b"}:
        _append_periodic_pair(parts, _mark_value(mark, 0, 6.0) / 7.0)
        _append_periodic_pair(parts, _mark_value(mark, 1, 30.0) / 31.0)
        _append_periodic_pair(parts, _mark_value(mark, 2, 365.0) / 365.0)
    elif key == "w":
        _append_periodic_pair(parts, _mark_value(mark, 0, 30.0) / 31.0)
        _append_periodic_pair(parts, _mark_value(mark, 1, 52.0) / 53.0)
    elif key in {"m", "a", "y"}:
        _append_periodic_pair(parts, _mark_value(mark, 0, 11.0) / 12.0)
    else:
        phase = mark + 0.5
        parts = [
            torch.sin(2.0 * math.pi * phase),
            torch.cos(2.0 * math.pi * phase),
        ]
    return torch.cat(parts, dim=-1)


class _EndogenousTrajectoryProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pred_len = int(config.pred_len)
        self.series_dim = int(config.series_dim)
        self.net = _make_mlp(
            int(config.seq_len) * self.series_dim,
            int(config.hidden_dim),
            self.pred_len * self.series_dim,
            int(config.num_layers),
            float(config.dropout),
        )

    def forward(self, history):
        output = self.net(history.reshape(history.shape[0], -1))
        return output.reshape(
            history.shape[0], self.pred_len, self.series_dim
        )


class _FutureExogenousPointMapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.series_dim = int(config.series_dim)
        self.exog_dim = int(config.exog_dim)
        self.net = _make_mlp(
            self.exog_dim,
            int(config.hidden_dim),
            self.series_dim,
            int(config.num_layers),
            float(config.dropout),
        )

    def forward(self, future_exog):
        batch, horizon, exog_dim = future_exog.shape
        output = self.net(future_exog.reshape(batch * horizon, exog_dim))
        return output.reshape(batch, horizon, self.series_dim)


class _WindowConcatenationMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pred_len = int(config.pred_len)
        self.series_dim = int(config.series_dim)
        self.exog_dim = int(config.exog_dim)
        self.freq = getattr(config, "freq", "h")
        mark_dim = getattr(config, "future_mark_dim", None)
        self.future_mark_dim = (
            _mark_dim_from_freq(self.freq)
            if mark_dim is None
            else int(mark_dim)
        )
        self.periodic_mark_dim = _periodic_mark_dim(self.freq, self.future_mark_dim)
        input_dim = (
            int(config.seq_len) * self.series_dim
            + self.pred_len * self.exog_dim
            + self.pred_len * self.periodic_mark_dim
        )
        self.net = _make_mlp(
            input_dim,
            int(config.hidden_dim),
            self.pred_len * self.series_dim,
            int(config.num_layers),
            float(config.dropout),
        )

    def forward(self, history, future_exog, target_mark):
        batch = history.shape[0]
        mark = _periodic_future_mark(
            target_mark,
            self.pred_len,
            self.future_mark_dim,
            self.freq,
        )
        features = torch.cat(
            [
                history.reshape(batch, -1),
                future_exog.reshape(batch, -1),
                mark.reshape(batch, -1),
            ],
            dim=-1,
        )
        output = self.net(features)
        return output.reshape(batch, self.pred_len, self.series_dim)


class _StateConditionedFutureMapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pred_len = int(config.pred_len)
        self.series_dim = int(config.series_dim)
        self.exog_dim = int(config.exog_dim)
        self.scfm_state_len = max(0, int(config.scfm_state_len))
        mark_dim = getattr(config, "future_mark_dim", None)
        self.future_mark_dim = (
            _mark_dim_from_freq(getattr(config, "freq", "h"))
            if mark_dim is None
            else int(mark_dim)
        )
        self.history_encoder = None
        input_dim = self.exog_dim + self.future_mark_dim
        if self.scfm_state_len > 0:
            history_hidden_dim = int(config.history_hidden_dim)
            self.history_encoder = _make_mlp(
                self.scfm_state_len * self.series_dim,
                history_hidden_dim,
                history_hidden_dim,
                1,
                float(config.dropout),
            )
            input_dim += history_hidden_dim
        self.net = _make_mlp(
            input_dim,
            int(config.hidden_dim),
            self.series_dim,
            int(config.num_layers),
            float(config.dropout),
        )

    def forward(self, history, future_exog, target_mark):
        batch, horizon, _ = future_exog.shape
        features = [future_exog]
        if self.history_encoder is not None:
            scfm_mapper_history = history
            if self.scfm_state_len < history.shape[1]:
                scfm_mapper_history = history[:, -self.scfm_state_len :, :]
            state = self.history_encoder(
                scfm_mapper_history.reshape(scfm_mapper_history.shape[0], -1)
            )
            features.append(state.unsqueeze(1).expand(-1, horizon, -1))
        features.append(_future_mark(target_mark, self.pred_len, self.future_mark_dim))
        features = torch.cat(features, dim=-1)
        output = self.net(features.reshape(batch * horizon, features.shape[-1]))
        return output.reshape(batch, horizon, self.series_dim)


class _SelectiveFutureTimeAdapter(nn.Module):
    def __init__(
        self,
        input_dim,
        mark_dim,
        hidden_dim,
        output_dim,
        num_layers,
        dropout,
        ratio,
    ):
        super().__init__()
        self.ratio = float(ratio)
        self.mapper = _make_mlp(
            int(input_dim) + int(mark_dim),
            int(hidden_dim),
            int(output_dim),
            int(num_layers),
            float(dropout),
        )

    def forward(self, features, periodic_mark, base_output=None):
        adapted = self.mapper(torch.cat([features, periodic_mark], dim=-1))
        if base_output is None or self.ratio >= 1:
            return adapted
        if self.ratio <= 0:
            return base_output
        return (1.0 - self.ratio) * base_output + self.ratio * adapted


class _FuExoMLPCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pred_len = int(config.pred_len)
        self.series_dim = int(config.series_dim)
        self.exog_dim = int(config.exog_dim)
        self.scfm_state_len = max(0, int(config.scfm_state_len))
        self.etp_model = str(config.etp_model).lower()
        self.sfta_mode = str(config.sfta_mode).lower()
        self.hist_exog_mode = str(getattr(config, "hist_exog_mode", "none")).lower()
        self.etp_fusion_alpha = float(config.etp_fusion_alpha)
        self.sfta_ratio = float(config.sfta_ratio)
        self.freq = getattr(config, "freq", "h")

        if self.etp_model not in {"none", "etp"}:
            raise ValueError(
                "etp_model must be one of: none, etp"
            )
        if self.sfta_mode not in {
            "none",
            "periodic",
            "adaptive_ratio",
        }:
            raise ValueError(
                "sfta_mode must be one of: none, periodic, adaptive_ratio"
            )
        if self.hist_exog_mode not in {
            "none",
            "plain",
            "crosslinear",
            "ca",
            "crossattn",
            "gate",
            "gated_crossattn",
        }:
            raise ValueError(
                "hist_exog_mode must be one of: none, plain, crosslinear, "
                "ca, crossattn, gate, gated_crossattn"
            )

        self.etp = None
        if self.etp_model == "etp":
            self.etp = _EndogenousTrajectoryProjector(config)

        scfm_hidden = (
            int(config.hidden_dim)
            if config.scfm_hidden_dim is None
            else int(config.scfm_hidden_dim)
        )
        history_hidden = (
            scfm_hidden
            if config.scfm_state_hidden_dim is None
            else int(config.scfm_state_hidden_dim)
        )
        scfm_layers = (
            int(config.num_layers)
            if config.scfm_num_layers is None
            else int(config.scfm_num_layers)
        )
        scfm_dropout = (
            float(config.dropout)
            if config.scfm_dropout is None
            else float(config.scfm_dropout)
        )

        self.history_encoder = None
        scfm_input_dim = self.exog_dim
        if self.scfm_state_len > 0:
            self.history_encoder = _make_mlp(
                self.scfm_state_len * self.series_dim,
                history_hidden,
                history_hidden,
                1,
                scfm_dropout,
            )
            scfm_input_dim += history_hidden

        self.hist_exog_encoder = None
        self.hist_exog_token_proj = None
        self.hist_exog_query_proj = None
        self.hist_exog_attention = None
        self.hist_exog_gate = None
        self.hist_exog_seq_len = 0
        if self.hist_exog_mode != "none":
            requested_hist_len = getattr(config, "hist_exog_seq_len", None)
            requested_hist_len = (
                int(config.seq_len)
                if requested_hist_len is None
                else int(requested_hist_len)
            )
            self.hist_exog_seq_len = max(
                1, min(int(config.seq_len), requested_hist_len)
            )
            hist_exog_hidden = (
                scfm_hidden
                if getattr(config, "hist_exog_hidden_dim", None) is None
                else int(config.hist_exog_hidden_dim)
            )
            if self.hist_exog_mode in {"plain", "crosslinear"}:
                self.hist_exog_encoder = _make_mlp(
                    self.hist_exog_seq_len * self.exog_dim,
                    hist_exog_hidden,
                    hist_exog_hidden,
                    1,
                    scfm_dropout,
                )
            else:
                self.hist_exog_token_proj = nn.Linear(
                    self.exog_dim, hist_exog_hidden
                )
                self.hist_exog_query_proj = nn.Linear(
                    self.series_dim, hist_exog_hidden
                )
                self.hist_exog_attention = nn.MultiheadAttention(
                    hist_exog_hidden,
                    _valid_num_heads(
                        hist_exog_hidden,
                        getattr(config, "hist_exog_num_heads", 4),
                    ),
                    batch_first=True,
                    dropout=scfm_dropout,
                )
                if self.hist_exog_mode in {"gate", "gated_crossattn"}:
                    self.hist_exog_gate = nn.Sequential(
                        nn.Linear(hist_exog_hidden * 2, hist_exog_hidden),
                        nn.GELU(),
                        nn.Linear(hist_exog_hidden, hist_exog_hidden),
                        nn.Sigmoid(),
                    )
            scfm_input_dim += hist_exog_hidden

        base_mark_dim = (
            _mark_dim_from_freq(self.freq)
            if config.future_mark_dim is None
            else int(config.future_mark_dim)
        )
        self.future_mark_dim = base_mark_dim
        self.periodic_mark_dim = _periodic_mark_dim(
            self.freq, base_mark_dim
        )

        self.scfm_mapper = None
        self.sfta = None
        if self.sfta_mode == "periodic":
            self.scfm_mapper = _make_mlp(
                scfm_input_dim + self.periodic_mark_dim,
                scfm_hidden,
                self.series_dim,
                scfm_layers,
                scfm_dropout,
            )
        else:
            self.scfm_mapper = _make_mlp(
                scfm_input_dim,
                scfm_hidden,
                self.series_dim,
                scfm_layers,
                scfm_dropout,
            )
            if self.sfta_mode == "adaptive_ratio":
                self.sfta = _SelectiveFutureTimeAdapter(
                    scfm_input_dim,
                    self.periodic_mark_dim,
                    scfm_hidden,
                    self.series_dim,
                    scfm_layers,
                    scfm_dropout,
                    self.sfta_ratio,
                )

    def _history_state(self, history):
        if self.history_encoder is None:
            return None
        scfm_mapper_history = history[:, -self.scfm_state_len :, :]
        return self.history_encoder(
            scfm_mapper_history.reshape(scfm_mapper_history.shape[0], -1)
        )

    def _hist_exog_state(self, history, historical_exog):
        if self.hist_exog_mode == "none" or historical_exog is None:
            return None
        if historical_exog.shape[-1] <= 0:
            return None
        hist_exog = historical_exog[:, -self.hist_exog_seq_len :, :]
        if self.hist_exog_encoder is not None:
            return self.hist_exog_encoder(hist_exog.reshape(hist_exog.shape[0], -1))

        tokens = self.hist_exog_token_proj(hist_exog)
        query = self.hist_exog_query_proj(history.mean(dim=1)).unsqueeze(1)
        context, _weights = self.hist_exog_attention(query, tokens, tokens)
        context = context.squeeze(1)
        if self.hist_exog_gate is not None:
            gate = self.hist_exog_gate(torch.cat([query.squeeze(1), context], dim=-1))
            context = context * gate
        return context

    def _etp_forward(
        self, history, input_mark, target_mark, decoder_input
    ):
        if self.etp is None:
            return None
        return self.etp(history)

    def _scfm_mapper_forward(self, history, target_mark, future_exog, historical_exog=None):
        batch, horizon, _ = future_exog.shape
        features = [future_exog]
        history_state = self._history_state(history)
        if history_state is not None:
            features.append(
                history_state.unsqueeze(1).expand(-1, horizon, -1)
            )
        hist_exog_state = self._hist_exog_state(history, historical_exog)
        if hist_exog_state is not None:
            features.append(
                hist_exog_state.unsqueeze(1).expand(-1, horizon, -1)
            )
        features = torch.cat(features, dim=-1)
        flat_features = features.reshape(batch * horizon, -1)

        if self.sfta_mode == "none":
            output = self.scfm_mapper(flat_features)
        else:
            mark = _periodic_future_mark(
                target_mark,
                self.pred_len,
                self.future_mark_dim,
                self.freq,
            ).reshape(batch * horizon, -1)
            if self.sfta_mode == "periodic":
                output = self.scfm_mapper(
                    torch.cat([flat_features, mark], dim=-1)
                )
            elif self.sfta_ratio <= 0:
                output = self.scfm_mapper(flat_features)
            elif self.sfta_ratio >= 1:
                output = self.sfta(flat_features, mark)
            else:
                no_time = self.scfm_mapper(flat_features)
                output = self.sfta(flat_features, mark, no_time)
        return output.reshape(batch, horizon, self.series_dim)

    def forward(
        self, history, input_mark, target_mark, future_exog, decoder_input, historical_exog=None
    ):
        scfm_output = self._scfm_mapper_forward(
            history, target_mark, future_exog, historical_exog
        )
        etp_output = self._etp_forward(
            history, input_mark, target_mark, decoder_input
        )
        if etp_output is None:
            return scfm_output
        return (
            self.etp_fusion_alpha * etp_output
            + (1.0 - self.etp_fusion_alpha) * scfm_output
        )


class FuExoMLP(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "FuExoMLP"

    def _init_model(self):
        if int(getattr(self.config, "series_dim", 1)) != 1:
            raise ValueError(
                "FuExoMLP requires multi-to-one forecasting with "
                'strategy_args target_channel=[-1].'
            )
        self.config.exog_dim = (
            int(self.config.input_dim) - int(self.config.series_dim)
        )
        if self.config.exog_dim <= 0:
            raise ValueError(
                "FuExoMLP requires known future exogenous variables."
            )
        variant = str(getattr(self.config, "model_variant", "auto")).lower()
        if variant in {
            "futureexogpointmlp",
            "future_exog_point_mlp",
            "future_exog_point",
        }:
            self.config.model_variant = "future_exog_point_mlp"
            return _FutureExogenousPointMapper(self.config)
        if variant in {
            "windowconcatmlp",
            "window_concat_mlp",
            "full_window_concat_mlp",
        }:
            self.config.model_variant = "window_concat_mlp"
            return _WindowConcatenationMLP(self.config)
        if variant in {
            "state_conditioned_future_mapper",
            "scfm",
        }:
            self.config.model_variant = "state_conditioned_future_mapper"
            return _StateConditionedFutureMapper(self.config)
        if variant in {
            "fuexomlp",
            "paper",
        }:
            self.config.etp_model = "etp"
            self.config.sfta_mode = "adaptive_ratio"
            self.config.model_variant = "fuexomlp"
        elif variant != "auto":
            raise ValueError(f"Unknown FuExoMLP model_variant: {variant}")
        if variant == "auto":
            self.config.model_variant = "fuexomlp"
        return _FuExoMLPCore(self.config)

    def _process(
        self,
        input,
        target,
        input_mark,
        target_mark,
        exog_future=None,
    ):
        if exog_future is None or exog_future.shape[-1] == 0:
            raise ValueError(
                "FuExoMLP requires known future exogenous variables."
            )
        history = input[:, :, : self.config.series_dim]
        historical_exog = input[:, :, self.config.series_dim :]
        if getattr(self.config, "model_variant", "auto") == "future_exog_point_mlp":
            return {"output": self.model(exog_future)}
        if getattr(self.config, "model_variant", "auto") == "window_concat_mlp":
            return {"output": self.model(history, exog_future, target_mark)}
        if getattr(self.config, "model_variant", "auto") == "state_conditioned_future_mapper":
            return {"output": self.model(history, exog_future, target_mark)}
        decoder_input = torch.zeros_like(
            target[
                :,
                -self.config.horizon :,
                : self.config.series_dim,
            ]
        ).float()
        output = self.model(
            history,
            input_mark,
            target_mark,
            exog_future,
            decoder_input,
            historical_exog,
        )
        return {"output": output}
