# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(30000, 300)
        nn.init.xavier_uniform_(self.l1.weight)
        self.l2 = nn.Linear(300, 300)
        nn.init.xavier_uniform_(self.l2.weight)
        self.l3 = nn.Linear(300, 128)
        nn.init.xavier_uniform_(self.l3.weight)

    def forward(self, x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = F.tanh(self.l3(x))
        return x

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    for q, p, n1, n2, n3, n4 in zip(query, doc_p, doc_n1, doc_n2, doc_n3, doc_n4):
    #query: Query sample, [batch size, 30000(trigram_dimension)]
    #doc_p: Doc Positive sample, [batch size, 30000(trigram_dimension)]
    #doc_n1, doc_n2, doc_n3, doc_n4: Doc Negative sample, [batch size, 30000(trigram_dimension)]
        out_q = net(q) #[batch size, 128]
        out_p = net(p) #[batch size, 128]
        out_n1 = net(n1) #[batch size, 128]
        out_n2 = net(n2) #[batch size, 128]
        out_n3 = net(n3) #[batch size, 128]
        out_n4 = net(n4) #[batch size, 128]

        batch_size = q.shape[0]
        labels = torch.tensor([0]*batch_size)

        #Relevance measured by cosine similarity
        cos_qp = torch.cosine_similarity(out_q, out_p, dim=1)#[batch size]
        cos_qp = cos_qp.unsqueeze(0).T#[batch size, 1]

        cos_qn1 = torch.cosine_similarity(out_q, out_n1, dim=1)#[batch size]
        cos_qn1 = cos_qn1.unsqueeze(0).T#[batch size, 1]

        cos_qn2 = torch.cosine_similarity(out_q, out_n2, dim=1)#[batch size]
        cos_qn2 = cos_qn2.unsqueeze(0).T#[batch size, 1]

        cos_qn3 = torch.cosine_similarity(out_q, out_n3, dim=1)#[batch size]
        cos_qn3 = cos_qn3.unsqueeze(0).T#[batch size, 1]

        cos_qn4 = torch.cosine_similarity(out_q, out_n4, dim=1)#[batch size]
        cos_qn4 = cos_qn4.unsqueeze(0).T#[batch size, 1]

        cos_uni = torch.cat((cos_qp, cos_qn1, cos_qn2, cos_qn3, cos_qn4), 1)#[batch size, 5]

        #posterior probability computed by softmax
        loss = criterion(cos_uni, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()