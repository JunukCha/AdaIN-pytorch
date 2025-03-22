import os
import torch
import torch.optim as optim
import torchvision

from lib.datasets.data import get_dataloader
from lib.models.styletransfer import get_styletransfer
from lib.utils.img import get_train_transform, get_test_transform, unnormalize
from lib.utils.loss import calc_content_loss, calc_style_loss

def main():
    st_net = get_styletransfer()
    st_net.cuda()
    transform = get_train_transform()
    transform = get_test_transform()
    train_dataloader = get_dataloader(root="data", transform=transform, batch_size=8, shuffle=True, num_data=1000)
    val_dataloader = get_dataloader(root="data", transform=transform, batch_size=8, shuffle=False, num_data=1000)
    test_dataloader = get_dataloader(root="data", transform=transform, batch_size=8, shuffle=False, num_data=1000, test=True)

    optimizer = optim.Adam(st_net.decoder.parameters(), lr=1e-4)
    
    os.makedirs("output/ckpts", exist_ok=True)
    os.makedirs("output/images", exist_ok=True)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for epoch in range(160):
        st_net.train()
        total_loss = 0.0
        total_c_loss = 0.0
        total_s_loss = 0.0
        total_samples = 0

        for step, (content_img, style_img) in enumerate(train_dataloader, 1):
            content_img = content_img.cuda()
            style_img = style_img.cuda()
            batch_size = content_img.size(0)

            output_img, t, style_feats = st_net(content_img, style_img)
            output_img_feats = st_net.encode_w_intermediate(output_img)

            content_loss = calc_content_loss(output_img_feats[-1], t)
            style_loss = sum(calc_style_loss(output_img_feats[i], style_feats[i]) for i in range(4))

            loss = content_loss + 10 * style_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_c_loss += content_loss.item() * batch_size
            total_s_loss += style_loss.item() * batch_size
            total_samples += batch_size
            avg_loss = total_loss / total_samples
            avg_c_loss = total_c_loss / total_samples
            avg_s_loss = total_s_loss / total_samples

            print(f"[Step {step}] Avg Loss: {avg_loss:.4f} | Avg Content Loss: {avg_c_loss:.4f} | Avg Style Loss: {avg_s_loss:.4f}")
        
        torch.save(
            st_net.state_dict(), 
            f"output/ckpts/st_net_{epoch+1}.pth", 
        )

        # if (epoch + 1) % 2 == 0:
        st_net.eval()
        with torch.no_grad():
            for step, (content_img, style_img) in enumerate(val_dataloader, 1):
                content_img = content_img.cuda()
                style_img = style_img.cuda()
                output_img = st_net.generate(content_img, style_img)

                content_img = unnormalize(content_img, mean, std)
                style_img = unnormalize(style_img, mean, std)
                output_img = unnormalize(output_img, mean, std)

                content_list = [img for img in content_img]
                style_list = [img for img in style_img]
                output_list = [img for img in output_img]

                combined_tensor = torch.stack(content_list + style_list + output_list, dim=0)
                save_path = f"output/images/output_val_{epoch+1}.png"
                torchvision.utils.save_image(combined_tensor, save_path)
                print(f"Saved image to {save_path}")
                break

            for step, (content_img, style_img) in enumerate(test_dataloader, 1):
                content_img = content_img.cuda()
                style_img = style_img.cuda()
                output_img = st_net.generate(content_img, style_img)

                content_img = unnormalize(content_img, mean, std)
                style_img = unnormalize(style_img, mean, std)
                output_img = unnormalize(output_img, mean, std)

                content_list = [img for img in content_img]
                style_list = [img for img in style_img]
                output_list = [img for img in output_img]

                combined_tensor = torch.stack(content_list + style_list + output_list, dim=0)
                save_path = f"output/images/output_test_{epoch+1}.png"
                torchvision.utils.save_image(combined_tensor, save_path)
                print(f"Saved image to {save_path}")
                break


if __name__ == '__main__':
    main()