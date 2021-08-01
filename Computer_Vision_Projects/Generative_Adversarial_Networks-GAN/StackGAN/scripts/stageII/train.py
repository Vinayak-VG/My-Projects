import pandas as pd
import numpy as np
import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.utils import save_image
from PIL import Image
import PIL
import pickle
from torch.utils.data import Dataset
from glob import glob
import time
import gc
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from dataloader import *
from model import *

trainset2 = CUB_Dataset_2()
trainloader2 = DataLoader(trainset2, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

StageII_Gen = StageII_GAN_Gen(DownSample2, ResidualBlock, UpSampling2, Conditioning_Augmentation_StageII).to(device)
StageII_Gen = StageII_Gen.apply(weights_init)
StageII_Dis = StageII_GAN_Dis(DownSample3).to(device)
StageII_Dis = StageII_Dis.apply(weights_init)
sbert_model = SentenceTransformer('paraphrase-mpnet-base-v2')

batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epoch_D1losses = []             
epoch_G1losses = []
epoch_D2losses = []             
epoch_G2losses = []
epoch_Real_Score = []
epoch_Fake_Score = []
epoch_Generator_Score = []

epochs = 600
lrG = 0.0002
lrD = 0.0002

optimizerD2 = torch.optim.Adam(StageII_Dis.parameters(), lr=lrD, betas=(0.5,0.999))
optimizerG2 = torch.optim.Adam(StageII_Gen.parameters(), lr=lrG, betas=(0.5,0.999))

BCEloss = nn.BCELoss()

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def train_StageII_Dis(real_images, wrong_images, stageI_img, text, optimizer):

    optimizer.zero_grad()

    real_images = real_images.to(device)
    text = text.to(device)
    real_pred = StageII_Dis(real_images, text)
    real_targets = torch.ones(real_images.size(0),1)
    real_pred = real_pred.to(device)
    real_targets = real_targets.to(device)
    real_loss = BCEloss(real_pred, real_targets)
    real_score = torch.mean(real_pred).item()

    fake_images, mu, logvar = StageII_Gen(stageI_img, text)

    fake_pred1 = StageII_Dis(fake_images, text)
    fake_targets1 = torch.zeros(fake_images.size(0),1)
    fake_pred1 = fake_pred1.to(device)
    fake_targets1 = fake_targets1.to(device)
    fake_loss1 = BCEloss(fake_pred1, fake_targets1)
    fake_score1 = torch.mean(fake_pred1).item()

    wrong_images = wrong_images.to(device)
    fake_pred2 = StageII_Dis(wrong_images, text)
    fake_targets2 = torch.zeros(wrong_images.size(0),1)
    fake_pred2 = fake_pred2.to(device)
    fake_targets2 = fake_targets2.to(device)
    fake_loss2 = BCEloss(fake_pred2, fake_targets2)
    fake_score2 = torch.mean(fake_pred2).item()

    discriminator_loss = (fake_loss1 + fake_loss2)/2 + real_loss
    discriminator_loss.backward()
    optimizer.step()

    return discriminator_loss.item(), real_score, (fake_score1 + fake_score2)/2


def train_StageII_Gen(gen1_image, text, optimizer):

    optimizer.zero_grad()
    
    gen1_image = gen1_image.to(device)
    text = text.to(device)
    generator_images, mu, logvar = StageII_Gen(gen1_image, text)
    generator_pred = StageII_Dis(generator_images, text)
    generator_targets = torch.ones(batch_size, 1)
    generator_pred = generator_pred.to(device)
    generator_targets = generator_targets.to(device)
    gen_bin_loss = BCEloss(generator_pred, generator_targets)
    generator_score = torch.mean(generator_pred).item()
    kl_loss = KL_loss(mu, logvar)

    generator_loss = gen_bin_loss + 2*kl_loss
    generator_loss.backward()
    optimizer.step()

    return generator_loss.item(), generator_score

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# if os.path.isfile("/content/drive/MyDrive/StackGAN/checkpoints/StageII_Dis_GAN_GPT2.pt"):
#     checkpointD2 = torch.load('/content/drive/MyDrive/StackGAN/checkpoints/StageII_Dis_GAN_GPT2.pt')
#     StageII_Dis.load_state_dict(checkpointD2['model_state_dict'])
#     StageII_Dis.to(device)
#     optimizerD2.load_state_dict(checkpointD2['optimizer_state_dict'])
#     epoch = checkpointD2['epoch']
#     best_D2loss = checkpointD2['loss']

# if os.path.isfile("/content/drive/MyDrive/StackGAN/checkpoints/StageII_Gen_GAN_GPT2.pt"):
#     checkpointG2 = torch.load('/content/drive/MyDrive/StackGAN/checkpoints/StageII_Gen_GAN_GPT2.pt')
#     StageII_Gen.load_state_dict(checkpointG2['model_state_dict'])
#     StageII_Gen.to(device)
#     optimizerG2.load_state_dict(checkpointG2['optimizer_state_dict'])
#     epoch = checkpointG2['epoch']
#     best_G2loss = checkpointG2['loss']

def save_samples(index1, stageI_img, text, show=True):
    fake_images, a, b = StageII_Gen(stageI_img, text)
    fake_images = fake_images[0:4,:,:,:]
    fake_fname = 'generated-images-{}.png'.format(index1)
    save_image((fake_images), os.path.join("/content/drive/MyDrive/GAN Images/Birds/birds-6", fake_fname), nrow=2)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

''' Stage2 Training'''

for epoch in range(epochs):
    
    start_time = time.monotonic()
    
    print(f"Epoch: {epoch + 1}")
    train_D2Loss_batch = []
    train_G2Loss_batch = []
    train_real_score = []
    train_fake_score = []
    train_generator_score = []

    for idx,(stageI_img, real_images, text) in enumerate(trainloader2):
        '''If you want to use HuggingFace Sentence Transformer'''
        text = list(text)
        embedding = []
        for i in range(len(text)):
            my_file = open(text[i], "r")
            content = my_file.read()
            embedding.append(content)
        emb = sbert_model.encode(embedding)
        emb = torch.from_numpy(emb)

        '''If you want to use GPT-2'''
        # text = list(text)
        # emb = torch.empty((0, 768))
        # for i in range(len(text)):
        #     my_file = open(text[i], "r")
        #     content = my_file.read()
        #     encoded_input = tokenizer(content, return_tensors='pt')
        #     output = model(**encoded_input)
        #     words_embs = output[0].transpose(1, 2).contiguous()
        #     sent_emb = words_embs[ :, :, -1 ].contiguous()
        #     emb = torch.cat((emb, sent_emb), dim = 0)
        #     # del encoded_input, output, words_embs, sent_emb
        # emb = emb.detach()

        wrong_images = torch.flip(real_images, [0])
        discriminator_loss, real_score, fake_score = train_StageII_Dis(real_images, wrong_images, stageI_img, emb, optimizerD2)
        generator_loss, generator_score = train_StageII_Gen(stageI_img, emb, optimizerG2)
        train_D2Loss_batch.append(discriminator_loss)
        train_G2Loss_batch.append(generator_loss)
        train_real_score.append(real_score)
        train_fake_score.append(fake_score)
        train_generator_score.append(generator_score)

        if (idx+1)%180 == 0:
            emb_save = emb[0:4, :]
            stageI_img_save = stageI_img[0:4, :, :, :]
            save_samples(epoch+1+246, stageI_img_save, emb_save, show=False)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

    if (epoch+1)%100 == 0:
        lrG = lrG/2
        lrD = lrD/2
        print(f"Learning Rate Halved: {lrG} {lrD}")

    epoch_D2losses.append(sum(train_D2Loss_batch)/len(trainloader2))
    epoch_G2losses.append(sum(train_G2Loss_batch)/len(trainloader2))
    epoch_Real_Score.append(sum(train_real_score)/len(trainloader2))
    epoch_Fake_Score.append(sum(train_fake_score)/len(trainloader2))
    epoch_Generator_Score.append(sum(train_generator_score)/len(trainloader2))

    torch.save({
        'epoch': epoch,
        'model_state_dict': StageII_Dis.state_dict(),
        'optimizer_state_dict': optimizerD2.state_dict(),
        'loss': epoch_D2losses[-1],
        }, '/content/drive/MyDrive/StackGAN/checkpoints/StageII_Dis_GAN.pt')

    torch.save({
        'epoch': epoch,
        'model_state_dict': StageII_Gen.state_dict(),
        'optimizer_state_dict': optimizerG2.state_dict(),
        'loss': epoch_G2losses[-1],
        }, '/content/drive/MyDrive/StackGAN/checkpoints/StageII_Gen_GAN.pt')

    #print(f"Epoch {epoch + 1} Training Over")
    print(f"Discriminator Epoch Loss: {epoch_D2losses[-1]:.5f}   Generator Epoch Loss: {epoch_G2losses[-1]:.5f}   Real Score: {epoch_Real_Score[-1]:.5f}   Fake Score: {epoch_Fake_Score[-1]:.5f}   Generator Score: {epoch_Generator_Score[-1]:.5f}")

    end_time = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    print("\n\n\n TIME TAKEN FOR THE EPOCH: {} mins and {} seconds".format(epoch_mins, epoch_secs))
    

print("OVERALL TRAINING COMPLETE")