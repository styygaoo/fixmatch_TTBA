import torch
from torch.utils.data import DataLoader
from get_vKITTI import VKITTI
from model_loader import load_model
from transforms import ToTensor
from torchvision import transforms as T
import torch.optim as optim
from data_processing import unpack_and_move, inverse_depth_norm
from config_model_TTA import configure_model, collect_params
import ttba
import time
from metrics import AverageMeter, Result
import gc


transformation = T.ToTensor()
trans = T.Compose([T.ToTensor()])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
maxDepth = 80
My_to_tensor = ToTensor(test=True, maxDepth=maxDepth)

# Load pre-trained model
model_original = load_model('GuideDepth', '/HOMES/yigao/KITTI_2_VKITTI/KITTI_Half_GuideDepth.pth')

# Load model parameter to be fine-tuned during test phase
model = configure_model(model_original)
params, param_names = collect_params(model)


# Prepare test dataloader for TTA
testset = VKITTI('/HOMES/yigao/KITTI/vkitti_testset_test/test', (192, 640))
testset_loader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)       # , drop_last=True

#
# # Define loss function and optimizer for fine-tuning
# optimizer = optim.Adam(params, lr=0.00000001, betas=(0.9, 0.999), weight_decay=0.0)
# optimizer = optim.Adam(params, lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0)
# optimizer = optim.SGD(params, lr=0.0000003)
optimizer = optim.SGD(params, lr=0.0001)
# adapt the given model to make it adaptive for test data
adapted_model = ttba.TTBA(model, optimizer)
adapted_model.cuda()
#
#
average_meter = AverageMeter()

for epoch in range(10):
    for i, data in enumerate(testset_loader):
        t0 = time.time()
        images, weaks, strongs, gts = data

        for b in range(weaks.shape[0]):
            packed_data = {'image': images[b], 'weak': weaks[b], 'strong': strongs[b], 'depth': gts[b]}
            data = My_to_tensor(packed_data)
            image, weak, strong, gt = unpack_and_move(data)
            # image, gt = data['image'], data['depth']
            image = image.unsqueeze(0)
            weak = weak.unsqueeze(0)
            strong = strong.unsqueeze(0)
            gt = gt.unsqueeze(0)
            if b >= 1:
                batched_images = torch.cat((batched_images, batched_image))
                batched_weaks = torch.cat((batched_weaks, batched_weak))
                batched_strongs = torch.cat((batched_strongs, batched_strong))
                batched_gts = torch.cat((batched_gts, batched_gt))
            else:
                batched_images = image
                batched_weaks = weak
                batched_strongs = strong
                batched_gts = gt
            batched_image = image
            batched_weak = weak
            batched_strong = strong
            batched_gt = gt


        data_time = time.time() - t0
        t0 = time.time()
        prediction = adapted_model(batched_images, batched_weaks, batched_strongs)
        prediction = prediction.detach()    # memory management
        prediction = inverse_depth_norm(prediction)
        gpu_time = time.time() - t0

        result = Result()
        result.evaluate(prediction.data, batched_gts.data)
        average_meter.update(result, gpu_time, data_time, images.size(0))

# Report
avg = average_meter.average()
current_time = time.strftime('%H:%M', time.localtime())
print('\n*\n'
      'RMSE={average.rmse:.3f}\n'
      'MAE={average.mae:.3f}\n'
      'Delta1={average.delta1:.3f}\n'
      'Delta2={average.delta2:.3f}\n'
      'Delta3={average.delta3:.3f}\n'
      'REL={average.absrel:.3f}\n'
      'Lg10={average.lg10:.3f}\n'
      't_GPU={time:.3f}\n'.format(
    average=avg, time=avg.gpu_time))




