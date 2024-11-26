import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMO(nn.Module):
    def __init__(self, feat_dim=2, m_real=0.5, m_fake=0.2, alpha=20.0, num_centers=20):

        super(SAMO, self).__init__()
        self.feat_dim = feat_dim
        self.num_centers = num_centers
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha
        self.center = torch.eye(self.feat_dim)[:self.num_centers]
        self.softplus = nn.Softplus()

    def forward(self, x, labels, spk=None, enroll=None, attractor=0):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.center, p=2, dim=1).to(x.device)
        scores = x @ w.transpose(0, 1)
        maxscores, _ = torch.max(scores, dim=1, keepdim=True)

        if attractor == 1:
            # for all target only data, speaker-specific center scores
            tmp_w = torch.stack([enroll[id] for id in spk])
            tmp_w = F.normalize(tmp_w, p=2, dim=1).to(x.device)
            final_scores = torch.sum(x * tmp_w, dim=1).unsqueeze(-1)
            # calculate emb_loss by adjusting scores
            maxscores[labels == 0] = self.m_real - final_scores[labels == 0]
        else:
            # every sample is using maxscore with all centers
            final_scores = maxscores.clone()
            maxscores[labels == 0] = self.m_real - maxscores[labels == 0]

        maxscores[labels == 1] = maxscores[labels == 1] - self.m_fake
        emb_loss = self.softplus(self.alpha * maxscores).mean()

        return emb_loss, final_scores.squeeze(1)

    def inference(self, x, labels, spk, enroll, attractor=0):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).

            Able to deal with samples without enrollment in scenario args.target=0
        """
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.center, p=2, dim=1).to(x.device)
        scores = x @ w.transpose(0, 1)
        maxscores, _ = torch.max(scores, dim=1, keepdim=True)

        if attractor == 1:
            # modify maxscore if it has a speaker center
            final_scores = maxscores.clone()
            for idx in range(len(spk)):
                if spk[idx] in enroll:
                    tmp_w = F.normalize(enroll[spk[idx]], p=2, dim=0).to(x.device)
                    final_scores[idx] = x[idx] @ tmp_w
            # calculate emb_loss by adjusting scores
            maxscores[labels == 0] = self.m_real - final_scores[labels == 0]
        else:
            # use maxscore for all samples with all centers
            final_scores = maxscores.clone()
            maxscores[labels == 0] = self.m_real - maxscores[labels == 0]

        maxscores[labels == 1] = maxscores[labels == 1] - self.m_fake
        emb_loss = self.softplus(self.alpha * maxscores).mean()

        return emb_loss, final_scores.squeeze(1)
