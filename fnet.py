import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dhidden):
        super().__init__()
        self.dense_1 = nn.Linear(dhidden, 4*dhidden) #d_feedforward is 4*d_model
        self.dense_2 = nn.Linear(4*dhidden, dhidden)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.gelu(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, dhidden=512):
        super().__init__()
        self.LayerNorm1 = nn.LayerNorm(dhidden)
        self.feedforward = FeedForward(dhidden)
        self.LayerNorm2 = nn.LayerNorm(dhidden)

    def forward(self, x):

        # DFT along hidden dimension followed by DFT along sequence dimension. Only keep real part of the result
        x_fft = torch.real(torch.fft.fft(torch.fft.fft(x).T).T)
        x = self.LayerNorm1(x + x_fft)

        x_ff = self.feedforward(x)
        x = self.LayerNorm2(x + x_ff)
        return x


class Fnet(nn.Module):
    def __init__(self, N, dhidden):
        super().__init__()
        self.encoder_layers = nn.ModuleList([EncoderBlock(dhidden) for _ in range(N)])
        self.dense = nn.Linear(dhidden, dhidden)

    def forward(self, x):

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        x = self.dense(x)
        return x

if __name__ == "__main__":
    model = Fnet(N=2, dhidden=32)
    model = model.train(False)
    x = torch.randn((2, 8, 32))
    y = model(x)




        

        








