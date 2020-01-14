import torch
import torch.nn as nn
import copy


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class Hook:
  # hooks into the output of a specified layer upon forward pass
  def __init__(self, layer):
    self.hook = layer.register_forward_hook(self._hook_function)
  
  def _hook_function(self, layer, input, output):
    self.output = output
  
  def close(self):
    self.hook.remove()

def get_vgg_with_hooks(vgg, layers, normalization=True, device='cuda'):
  """ Returns a simplified VGG model with decreased depth, with hooks attached to 
      specifil outputs. 

      Args:
          vgg - any of the vgg model's features; e.g. torchvision.models.vgg19().features
          layers - list of layer names where hooks should be attached.
      
      Returns:
          model - a simplified model with layers from 'vgg' only up to the last hooked layer,
          which is the last layer in the 'layers' list
  """
  vgg = copy.deepcopy(vgg).eval()
  hooks = []
  if normalization:
    normalization = Normalization(torch.tensor([0.485, 0.456, 0.406], device=device),
                                  torch.tensor([0.229, 0.224, 0.225], device=device))
    model = nn.Sequential(normalization)
  else:
    model = nn.Sequential()

  i = 0  # increment every time we see a conv
  for layer in vgg.children():
      if isinstance(layer, nn.Conv2d):
          i += 1
          name = 'conv_{}'.format(i)
      elif isinstance(layer, nn.ReLU):
          name = 'relu_{}'.format(i)
          layer = nn.ReLU(inplace=False)
      elif isinstance(layer, nn.MaxPool2d):
          name = 'pool_{}'.format(i)
      elif isinstance(layer, nn.BatchNorm2d):
          name = 'bn_{}'.format(i)
      else:
          raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

      model.add_module(name, layer)

      # attach hooks
      if name in layers:
          hooks.append(Hook(layer))
      
      if name == layers[-1]:
        break

  return model.to(device), hooks