def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def process_adain(content_feat, style_feat, eps=1e-5):
    content_mean, content_std = calc_mean_std(content_feat, eps)
    style_mean, style_std = calc_mean_std(style_feat, eps)
    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean