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



transformation = T.ToTensor()
trans = T.Compose([T.ToTensor()])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
maxDepth = 80
My_to_tensor = ToTensor(test=True, maxDepth=maxDepth)

# Load pre-trained model
model_original = load_model('GuideDepth', '/HOMES/yigao/KITTI_2_VKITTI/KITTI_Half_GuideDepth.pth')
# model_original.eval().cuda()

# Load model parameter to be fine-tuned during test phase
model = configure_model(model_original)
params, param_names = collect_params(model)


# Prepare test dataloader for TTA
testset = VKITTI('/HOMES/yigao/KITTI/vkitti_testset_test/test', (192, 640))
# testset = VKITTI('/HOMES/yigao/Downloads/eval_testset/NYU_Testset', 'full')
testset_loader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)       # , drop_last=True


# Define loss function and optimizer for fine-tuning
optimizer = optim.Adam(params, lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0)
# optimizer = optim.SGD(params, lr=0.001, weight_decay=0.0)
# oering the given model to make it adaptive for test data
adapted_model = ttba.TTBA(model, optimizer)
adapted_model.cuda()


average_meter = AverageMeter()
for i, data in enumerate(testset_loader):
    t0 = time.time()
    weaks, strongs, gts = data

    for b in range(weaks.shape[0]):
        # print(b)
        packed_data = {'weak': weaks[b], 'strong': strongs[b], 'depth': gts[b]}
        data = My_to_tensor(packed_data)
        weak, strong, gt = unpack_and_move(data)
        # image, gt = data['image'], data['depth']
        weak = weak.unsqueeze(0)
        strong = strong.unsqueeze(0)
        gt = gt.unsqueeze(0)
        if b >= 1:
            batched_weaks = torch.cat((batched_weaks, batched_weak))
            batched_strongs = torch.cat((batched_strongs, batched_strong))
            batched_gts = torch.cat((batched_gts, batched_gt))
        else:
            batched_weaks = weak
            batched_strongs = strong
            batched_gts = gt
        batched_weak = weak
        batched_strong = strong
        batched_gt = gt


    data_time = time.time() - t0
    t0 = time.time()

    inv_prediction_weaks, inv_prediction_strongs = adapted_model(batched_weaks, batched_strongs)
    predictions_weaks = inverse_depth_norm(inv_prediction_weaks)

    gpu_time = time.time() - t0

    result = Result()
    result.evaluate(predictions_weaks.data, batched_gts.data)
    average_meter.update(result, gpu_time, data_time, weaks.size(0))

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




