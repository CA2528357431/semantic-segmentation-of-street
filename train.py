import torch
import os
import h5py
from DataLoader import DataSet
from torch.utils.data import DataLoader
import torch.nn as nn
from styler import Unet

BATCH_SIZE = 16
EPOCH = 100
LR = 0.0005
current_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = current_dir + '/data/driving_train_data.h5'
test_dir = current_dir + '/data/driving_test_data.h5'
train_dataset = DataSet(path=train_dir,
                        split='train',
                        overwrite=False,
                        transform=True
                        )
test_dataset = DataSet(path=test_dir,
                       split='test',
                       overwrite=False,
                       transform=False
                       )
rgb, seg = train_dataset[0]
print(rgb.shape)
print(seg.shape)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True)
print(len(train_dataloader))
print(len(test_dataloader))

cuda = torch.device("cuda")

model = Unet().to(cuda)
loss_func = nn.CrossEntropyLoss()
optim = torch.optim.AdamW(model.parameters(), lr=LR)
num_steps = len(train_dataloader) * EPOCH
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_steps)
# warmup_scheduler = warmup.UntunedLinearWarmup(optim)

model.to(cuda)
loss_func.to(cuda)

li1 = []
li2 = []

print("start")
for epoch in range(EPOCH):

    print(epoch)
    torch.cuda.empty_cache()
    # log.section_header('####################EPOCH:%d####################' % epoch)

    valid_step = 0
    train_step = 0
    test_step = 0
    total_train_loss = 0
    total_valid_loss = 0
    total_test_loss = 0
    iteration = 0
    model.train()

    for data in train_dataloader:
        inputs, targets = data
        # print(inputs.shape)
        # -----------------MODEL INPUT--------------------
        inputs = inputs.to(cuda)
        targets = targets.to(cuda)

        iteration = iteration + 1
        if iteration <= 700:
            train_step = train_step + 1

            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            total_train_loss += float(loss)
            loss.backward()

            optim.step()
            # with warmup_scheduler.dampening():
            #     lr_scheduler.step()

            # if train_step % 140 == 0:
            #     log.title('--------------------EPOCH:%d train_step:%d--------------------' % (epoch, train_step))
            #     log.data('loss:', loss.item())

        else:
            # log.section_header('####################VALID####################')
            valid_step = valid_step + 1
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                loss = loss_func(outputs, targets)
                total_valid_loss += float(loss)
                # if valid_step % 11 == 0:
                #     log.title('--------------------EPOCH:%d valid_step:%d--------------------' % (epoch, valid_step))
                #     log.data('loss:', loss.item())
    li1.append(total_train_loss)
    print(total_train_loss, train_step)

    with torch.no_grad():
        # log.section_header('####################TEST####################')
        for data in test_dataloader:
            test_step = test_step + 1
            inputs, targets = data
            inputs = inputs.to(cuda)
            targets = targets.to(cuda)

            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            total_test_loss += float(loss)

    li2.append(total_test_loss)
    print(total_test_loss, test_step)

    # avg_train_Closs = total_train_loss / train_step
    # log.data('train_loss:', avg_train_Closs)
    # writer.add_scalar('train_loss', avg_train_Closs, epoch)
    # avg_valid_Closs = total_valid_loss / valid_step
    # log.data('valid_loss:', avg_valid_Closs)
    # writer.add_scalar('valid_loss', avg_valid_Closs, epoch)
    # avg_test_Closs = total_test_loss / test_step
    # log.data('test_loss:', avg_test_Closs)
    # writer.add_scalar('test_loss', avg_test_Closs, epoch)

    torch.save(model, './model_{}.pth'.format(epoch))
    # log.title('模型已保存')
torch.cuda.empty_cache()
# writer.close()

with open("f1.txt", "w") as f:
    for x in li1:
        f.write(str(x)+"\n")

with open("f2.txt", "w") as f:
    for x in li2:
        f.write(str(x)+"\n")