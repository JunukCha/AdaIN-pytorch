import torch
import torch.nn as nn

from lib.utils.adain import process_adain
from lib.models.encoder import vgg19
from lib.models.decoder import Decoder

class AdaINStyleTransfer(nn.Module):
    def __init__(self, encoder, decoder):
        super(AdaINStyleTransfer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.enc_layers = nn.Sequential(
            nn.Sequential(*self.encoder.features[:2]), # relu1_1
            nn.Sequential(*self.encoder.features[2:7]), # relu2_1
            nn.Sequential(*self.encoder.features[7:12]), # relu3_1
            nn.Sequential(*self.encoder.features[12:21]), # relu4_1
        )


    def encode_w_intermediate(self, image):
        outputs = []
        output = image
        for enc_layer in self.enc_layers:
            output = enc_layer(output)
            outputs.append(output)
        return outputs
    
    def encode(self, image):
        output = image
        for enc_layer in self.enc_layers:
            output = enc_layer(output)
        return output

    def forward(self, content, style):
        content_feat = self.encode(content)
        style_feats = self.encode_w_intermediate(style)
        t = process_adain(content_feat, style_feats[-1])
        out = self.decoder(t)
        return out, t, style_feats

    def generate(self, content, style):
        content_feat = self.encode(content)
        style_feat = self.encode(style)
        t = process_adain(content_feat, style_feat)
        out = self.decoder(t)
        return out    
        
def get_styletransfer():
    encoder = vgg19
    decoder = Decoder()
    return AdaINStyleTransfer(encoder, decoder)