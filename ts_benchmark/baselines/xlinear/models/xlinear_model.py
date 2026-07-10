import torch
from torch import nn


class XLinearModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.channel = configs.enc_in
        self.t_ff = configs.t_ff
        self.c_ff = configs.c_ff
        self.norm = configs.usenorm
        self.embed_dropout = configs.embed_dropout
        self.head_dropout = configs.head_dropout
        self.t_dropout = configs.t_dropout
        self.c_dropout = configs.c_dropout
        self.feature = configs.features

        if self.feature == "M":
            self.backbone = ForecastMulti(
                self.seq_len,
                self.d_model,
                self.channel,
                self.t_ff,
                self.c_ff,
                self.t_dropout,
                self.c_dropout,
                self.embed_dropout,
            )
        else:
            self.backbone = ForecastWithExogenous(
                self.seq_len,
                self.d_model,
                self.channel,
                self.t_ff,
                self.c_ff,
                self.t_dropout,
                self.c_dropout,
                self.embed_dropout,
            )
        self.head = nn.Sequential(
            nn.Dropout(self.head_dropout),
            nn.Linear(2 * self.d_model, self.pred_len),
        )

    def forecast_multi(self, x_enc):
        if self.norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc = self.backbone(x_enc)
        dec_out = self.head(enc).permute(0, 2, 1)

        if self.norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )

        return dec_out

    def forecast_exogenous(self, x_enc):
        if self.norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc = self.backbone(x_enc)
        dec_out = self.head(enc).permute(0, 2, 1)

        if self.norm:
            dec_out = dec_out * stdev[:, 0, -1:].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )
            dec_out = dec_out + means[:, 0, -1:].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )

        return dec_out

    def forward(self, x_enc):
        if self.feature == "M":
            return self.forecast_multi(x_enc)
        return self.forecast_exogenous(x_enc)


class ForecastMulti(nn.Module):
    def __init__(
        self,
        seq_len,
        d_model,
        channel,
        t_ff,
        c_ff,
        t_dropout,
        c_dropout,
        embed_dropout,
    ):
        super().__init__()
        self.d_model = d_model
        self.channel = channel
        self.projection = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.Dropout(embed_dropout),
        )

        self.global_token = nn.Parameter(torch.ones([1, channel, d_model]))
        self.en_attention = GatingBlock(2 * d_model, t_ff, t_dropout)
        self.ex_attention = GatingBlock(2 * channel, c_ff, c_dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        emb = self.projection(x)

        global_token = self.global_token.repeat([batch_size, 1, 1])
        en_emb = torch.cat([emb, global_token], dim=-1)
        en_attn = self.en_attention(en_emb)

        origin_attn = en_attn[:, :, : self.d_model]
        global_attn = en_attn[:, :, self.d_model :]

        ex_emb = torch.cat([emb, global_attn], dim=1)
        ex_attn = self.ex_attention(ex_emb.permute(0, 2, 1))

        global_part = ex_attn[:, :, self.channel :]
        enc = torch.cat([origin_attn, global_part.permute(0, 2, 1)], dim=-1)
        return enc


class ForecastWithExogenous(nn.Module):
    def __init__(
        self,
        seq_len,
        d_model,
        channel,
        t_ff,
        c_ff,
        t_dropout,
        c_dropout,
        embed_dropout,
    ):
        super().__init__()
        self.d_model = d_model
        self.channel = channel
        self.projection = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.Dropout(embed_dropout),
        )

        self.global_token = nn.Parameter(torch.ones([1, 1, d_model], dtype=torch.float))
        self.temporal = GatingBlock(2 * d_model, t_ff, t_dropout)
        self.cross = GatingBlock(channel, c_ff, c_dropout)

    def forward(self, x):
        # XLinear expects the target/endogenous series to be the last channel.
        batch_size = x.shape[0]
        embed = self.projection(x)
        endo = embed[:, -1:, :]
        exog = embed[:, :-1, :]

        global_token = self.global_token.repeat([batch_size, 1, 1])
        endo_with_global = torch.cat([endo, global_token], dim=-1)

        endo_attn = self.temporal(endo_with_global)
        origin_attn = endo_attn[:, :, : self.d_model]
        global_part = endo_attn[:, :, self.d_model :]

        exog_with_global = torch.cat([exog, global_part], dim=1)
        exog_attn = self.cross(exog_with_global.permute(0, 2, 1))

        global_part = exog_attn[:, :, -1:]
        enc = torch.cat([origin_attn, global_part.permute(0, 2, 1)], dim=-1)
        return enc


class GatingBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weight = self.weight(x)
        return x * weight
