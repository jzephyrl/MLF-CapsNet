import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.nn.functional as F

from capsule_conv_layer import CapsuleConvLayer
from capsule_layer import CapsuleLayer


class CapsuleNetwork(nn.Module):
    def __init__(self,
                 eeg_width,
                 eeg_height,
                 eeg_channels,
                 conv_inputs,
                 conv_outputs,
                 num_primary_units,
                 primary_unit_size,
                 num_output_units,
                 output_unit_size):
        super(CapsuleNetwork, self).__init__()
        self.primary_units=num_primary_units
        self.reconstructed_eeg_count = 0
       
        self.eeg_channels = eeg_channels
        self.eeg_width = eeg_width
        self.eeg_height = eeg_height
        #进行一维卷积 输入[B,1,128,32]=》[B,256,120,24]???
        self.conv1 = CapsuleConvLayer(in_channels=conv_inputs,
                                      out_channels=conv_outputs,
                                      kernel_size=9,
                                      stride=2)
        self.conv2= CapsuleConvLayer(in_channels=512,
                                      out_channels=conv_outputs,
                                      kernel_size=1,
                                      stride=1)
        self.primary = CapsuleLayer(in_units=0,
                                    in_channels=conv_outputs,
                                    num_units=num_primary_units,
                                    unit_size=primary_unit_size,
                                    use_routing=False)

        self.digits = CapsuleLayer(in_units=num_primary_units,
                                   in_channels=primary_unit_size,
                                   num_units=num_output_units,
                                   unit_size=output_unit_size,
                                   use_routing=True)
        

        reconstruction_size = eeg_width * eeg_height * eeg_channels
        self.reconstruct0 = nn.Linear(num_output_units*output_unit_size, int((reconstruction_size * 2) / 3))
        self.reconstruct1 = nn.Linear(int((reconstruction_size * 2) / 3), int((reconstruction_size * 3) / 2))
        self.reconstruct2 = nn.Linear(int((reconstruction_size * 3) / 2), reconstruction_size)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, batch= data.x,data.batch
        x=x.reshape(x.size(0),1,x.size(1),x.size(2))
        # print("self.conv1(x):",self.conv1(x).size()) [35, 256, 60, 12]
        conv1=self.conv1(x)
        primary=self.primary(conv1) 
        # print(primary.size())[35, 256, 60, 12]
        primary_conv=self.conv2(primary).view(x.size(0),self.primary_units, -1)
        primary_conv=CapsuleLayer.squash(primary_conv)
        # print("primary_conv:",primary_conv.shape)
        out_caps=self.digits(primary_conv)
        # print("out_caps:",out_caps.size())
        return out_caps

    def loss(self, output, target, size_average=True):
        # return self.margin_loss(input, target, size_average) + self.reconstruction_loss(eegs, input, size_average)
        return self.margin_loss(output, target, size_average) 

    def margin_loss(self, input, target, size_average=True):
        # print("input:",input.size())
        batch_size = input.size(0)

        # ||vc|| from the paper.
        v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))
        # print("v_mag :",v_mag.size())
        # Calculate left and right max() terms from equation 4 in the paper.
        zero = Variable(torch.zeros(1)).cuda()
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2

        # This is equation 4 from the paper.
        loss_lambda = 0.5
        T_c = target
        # L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        # 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)

        if size_average:
            L_c = L_c.mean()
        # print("L_c:",L_c)
        return L_c

    # def reconstruction_loss(self, eegs, input, size_average=True):
    #     # Get the lengths of capsule outputs.
    #     v_mag = torch.sqrt((input**2).sum(dim=2))

    #     # Get index of longest capsule output.
    #     _, v_max_index = v_mag.max(dim=1)
    #     v_max_index = v_max_index.data

    #     # Use just the winning capsule's representation (and zeros for other capsules) to reconstruct input image.
    #     batch_size = input.size(0)
    #     all_masked = [None] * batch_size
    #     for batch_idx in range(batch_size):
    #         # Get one sample from the batch.
    #         input_batch = input[batch_idx]

    #         # Copy only the maximum capsule index from this batch sample.
    #         # This masks out (leaves as zero) the other capsules in this sample.
    #         batch_masked = Variable(torch.zeros(input_batch.size())).cuda()
    #         batch_masked[v_max_index[batch_idx]] = input_batch[v_max_index[batch_idx]]
    #         all_masked[batch_idx] = batch_masked

    #     # Stack masked capsules over the batch dimension.
    #     masked = torch.stack(all_masked, dim=0)

    #     # Reconstruct input image.
    #     masked = masked.view(input.size(0), -1)
    #     output = self.relu(self.reconstruct0(masked))
    #     output = self.relu(self.reconstruct1(output))
    #     output = self.sigmoid(self.reconstruct2(output))
    #     output = output.view(-1, self.eeg_channels, self.eeg_height, self.eeg_width)

    #     # Save reconstructed images occasionally.
    #     if self.reconstructed_eeg_count % 10 == 0:
    #         if output.size(1) == 2:
    #             # handle two-channel images
    #             zeros = torch.zeros(output.size(0), 1, output.size(2), output.size(3))
    #             output_eeg = torch.cat([zeros, output.data.cpu()], dim=1)
    #         else:
    #             # assume RGB or grayscale
    #             output_eeg = output.data.cpu()
    #         vutils.save_eeg(output_eeg, "reconstruction.png")
    #     self.reconstructed_eeg_count += 1

    #     # The reconstruction loss is the sum squared difference between the input image and reconstructed image.
    #     # Multiplied by a small number so it doesn't dominate the margin (class) loss.
    #     error = (output - eegs).view(output.size(0), -1)
    #     error = error**2
    #     error = torch.sum(error, dim=1) * 0.0005

    #     # Average over batch
    #     if size_average:
    #         error = error.mean()

    #     return error

