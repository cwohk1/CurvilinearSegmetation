import os
import torch
import torchsummary
from model import UNet16
from dataloader import *
from torch import optim
from tqdm import tqdm

TRAIN_IMG = "./crack_segmentation_dataset/train/images"
TRAIN_MASK = "./crack_segmentation_dataset/train/masks"
TEST_IMG = "./crack_segmentation_dataset/test/images"
TEST_MASK = "./crack_segmentation_dataset/test/masks"
LEARNING_RATE = 0.0001
BATCH_SIZE = 8

def evaluate(model, testloader, loss):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item()
    loss = test_loss/len(testloader)
    return loss

def train(model, trainloader, lr, loss_ftn, epoch, testloader = None, threshold=0.5, train_history = [], test_history = [], working_dir = "./"):
    model_dir = os.path.join(working_dir, "models")
    optimizer = optim.Adam(model.parameters(), lr = lr)
    print("Training %d images"%len(trainloader))
    for e in tqdm(range(1, epoch+1)):
        print()
        torch.cuda.empty_cache()       # GPU 메모리 정리
        model.train()
        running_loss = 0.0
        for idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_ftn(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if idx % (len(trainloader)//5)== 0:
            #     print("\t epoch : %d, batch : %d, training_loss : %.4f"%(e, idx, running_loss/(idx+1)))
        train_history.append(running_loss/len(trainloader))              # 배치별 손실값의 평균을 저장
        if testloader is not None:
            test_loss = evaluate(model, testloader, loss_ftn)
            test_history.append(test_loss)
            if test_loss <= threshold:# save model if loss if less than threshold
                threshold = test_loss
                if not os.path.exists(model_dir): os.makedirs(model_dir)
                torch.save({'epoch': e,
                            'train_history': train_history,
                            "test_history" : test_history,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }, model_dir+'/unet_%d_%d.pt'%(e, int(test_loss*1000)))
        print("\n[%d] training loss: %.3f"%(e, running_loss / len(trainloader)))
        if testloader is not None: print("[%d] test loss: %.3f"%(e, test_loss))
    return train_history, test_history

if torch.cuda.is_available(): device = torch.device("cuda:0")
elif torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")

if __name__ == "__main__":
    #criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.BCELoss()
    unet = UNet16(num_classes=1, pretrained = True).to(device)
    train_transform = TrainImageTransforms()
    test_transform = TestImageTransforms()
    mask_transforms = MaskTransforms()
    trainset = CrackDataSet(image_dir=TRAIN_IMG, mask_dir=TRAIN_IMG, image_transforms=train_transform, mask_transforms=mask_transforms)
    testset = CrackDataSet(image_dir=TEST_IMG, mask_dir=TEST_MASK, image_transforms=test_transform, mask_transforms=mask_transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
    testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False)
    train_history, test_history = train(unet, trainloader, LEARNING_RATE, criterion, 30, testloader)