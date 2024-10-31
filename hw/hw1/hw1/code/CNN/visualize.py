import torch
from torchvision import models, transforms
from PIL import Image
from pytorch_cnn_visualizations.src.gradcam import GradCam
from torchsummary import summary
import medmnist
from medmnist import INFO, Evaluator
import os
from medmnist import INFO, Evaluator
import matplotlib.pyplot as plt

def save_feature_maps(feature_maps, output_dir):
    for i, fmap in enumerate(feature_maps):
        layer_dir = os.path.join(output_dir, f'layer_{i + 1}')
        os.makedirs(layer_dir, exist_ok=True)

        num_features = fmap.shape[1]  
        target_size = (128, 128)
        weight, height = target_size

        for j in range(min(3, num_features)):
                image = fmap[0, j].cpu().detach().numpy()

                plt.imshow(image, cmap='viridis')
                # axis off
                plt.axis('off')
                
                file_name = os.path.join(layer_dir, f'feature_map_channel_{j + 1}.png')
                plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
                plt.close()


if __name__ == '__main__':
    
    # model preparation
    ckpt_path = './checkpoints_backup/best_model.pt'
    model = torch.load(ckpt_path)
    summary(model, (3, 64, 64))
    
    # data preparation
    data_path = './data'
    data_flag = 'pathmnist'
    info = INFO[data_flag]
    download = False
    img_idx = 0
    DataClass = getattr(medmnist, info['python_class'])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    test_dataset = DataClass(root=data_path, split='test', transform=data_transform, size=64, download=download)
        
    image, label = test_dataset[img_idx]
    x = image.unsqueeze(0).to('cuda')

    # save conv feature
    result_path = f'./results/image_{img_idx}'
    os.makedirs(result_path, exist_ok=True)
    feature_maps = []
    
    # save original image
    to_pil = transforms.ToPILImage()
    image = to_pil(image * 0.5 + 0.5)
    
    # resize to (128, 128)
    image = image.resize((369, 369))
    
    image.save(os.path.join(result_path, 'original_image.png'))
    
    for name, module in model.named_children():
        
        print(f'module: {name:15} ', end='')
        print('input:{:20}' .format(str(tuple(x.shape))), end='')
        x = module(x)
        print('output:', tuple(x.shape))
        feature_maps.append(x)
        
        if name == 'avgpool':
            # x = x.view(x.size(0), -1)
            break
            
    save_feature_maps(feature_maps, result_path)
    