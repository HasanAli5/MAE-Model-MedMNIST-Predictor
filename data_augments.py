from torch.utils.data import default_collate
from torchvision.transforms import v2
import torchvision.transforms.v2 as transforms
from collections.abc import Sequence
import torch
from torch import nn

class MixerCollate():
    def __init__(self,n_labels):
        self.mixer_augment = v2.RandomChoice([
            v2.MixUp(alpha=0.8,num_classes=n_labels),
            v2.CutMix(alpha=1.0,num_classes=n_labels)
        ])
    
    def collate_fn(self,batch):
        images,labels = default_collate(batch)

        if labels.ndim > 1 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        return self.mixer_augment(images,labels)

class MeanRandomFilling(nn.Module):
    """
    Adapts the Existing RandomErasing v2 transform
    to instead fill the mean value
    """

    def __init__(self,
            p: float = 0.5,
            scale: Sequence[float] = (0.02, 0.33),
            ratio: Sequence[float] = (0.3, 3.3),
            inplace: bool = False):

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.inplace = inplace
        

        
        super().__init__()

    def forward(self,image:torch.Tensor):
        # ....., c, h, w
        image_mean = image.mean(dim=[-2,-1]).tolist()

        random_erasing = transforms.RandomErasing(
            p=self.p,
            scale=self.scale,
            ratio=self.ratio,
            value=image_mean,
            inplace=self.inplace
        )
        return random_erasing(image)

class GaussianBlurPatches(nn.Module):
    """
    Adapts the Existing RandomErasing v2 transform
    to instead fill the mean value
    """

    def __init__(self,
            kernel_size: int | Sequence[int],
            sigma: int | float | Sequence[float] = (0.1, 2),
            p: float = 0.5,
            scale: Sequence[float] = (0.02, 0.33),
            ratio: Sequence[float] = (0.3, 3.3),
            inplace: bool = False,
            blend: bool = True,
            mask_kernel_size: int | Sequence[int] = -1,
            mask_sigma: int | float | Sequence[float]= -1,
            patches:int = 1):
        super().__init__()
        
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.inplace = inplace
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.blend = blend

        mask_scaler = 3

        if blend:
            if mask_kernel_size == -1:
                if isinstance(kernel_size,(tuple,list)):
                    mask_kernel_size = [int(k//mask_scaler) for k in kernel_size]
                    mask_kernel_size = [(k if k % 2 != 0 else k + 1) for k in mask_kernel_size]
                else:
                    mask_kernel_size = int(kernel_size//mask_scaler)
                    mask_kernel_size = mask_kernel_size if mask_kernel_size % 2 != 0 else mask_kernel_size + 1
            if mask_sigma == -1:
                if isinstance(sigma,(tuple,list)):
                    mask_sigma = [int(round(s/mask_scaler)) for s in sigma]
                else:
                    mask_sigma = int(round(sigma/mask_scaler))

        self.mask_kernel_size = mask_kernel_size
        self.mask_sigma = mask_sigma

        self.patches = patches

        self.random_erasing = transforms.RandomErasing(
            p=1.0,
            scale=self.scale,
            ratio=self.ratio,
            value=0,
            inplace=self.inplace
        )
        
    def forward(self,image:torch.Tensor):
        h = image.size(-2)
        w = image.size(-1)

        mask = torch.ones(size=(1,h,w),device=image.device,dtype=image.dtype)

        success = False
        for _ in range(self.patches):
            if self.p>torch.rand(size=(1,))[0]:
                mask = self.random_erasing(mask)
                success = True

        if not success:
            return image

        if self.blend:
            mask = transforms.functional.gaussian_blur(mask,kernel_size=self.mask_kernel_size,sigma=self.mask_sigma)

        blurred_image = transforms.functional.gaussian_blur(image,kernel_size=self.kernel_size,sigma=self.sigma)

        return (image*mask)+(blurred_image*(1-mask)) 