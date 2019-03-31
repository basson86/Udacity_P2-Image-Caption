import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.h = None
    
    def forward(self, features, captions):
        
        captions = captions.view(captions.size(0),-1)
        emb_cap = self.word_embed(captions)
        
        features = features.view(features.size(0),1,-1)
                
        x = torch.cat((features,emb_cap),1)
        y, self.h =self.lstm(x, self.h)
        
        y = self.fc(y[:,:-1,:])

        
        return y
        
        
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        vocab_idxes = [0]
        start_emb = torch.LongTensor(torch.tensor([0])).cuda()
        start_emb = self.word_embed(start_emb).view(1,1,-1)    
        x = torch.cat((inputs,start_emb),1)
        for i in range(max_len):
            y,h = self.lstm(x)
            out = self.fc(y)
                        
            word = out[:,-1,:].cpu()
            
            prob, idx = word.topk(1)
                        
            #print(prob, idx)
            
            idx = idx.numpy().squeeze()
            vocab_idxes.append(idx.item())
                      
            if idx.item()==1:
                break
            
           
            new_word = torch.LongTensor(idx).cuda().view(1,-1)
            new_word = self.word_embed(new_word).view(1,1,-1)
            #x = torch.cat((x,new_word),1)
            x = new_word
                         
        return vocab_idxes