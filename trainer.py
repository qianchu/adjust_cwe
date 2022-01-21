import torch
import scipy
from numpy.linalg import inv

class LinearProjection(torch.nn.Module):
    def __init__(self, D_in,  D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(LinearProjection, self).__init__()
        self.linear = torch.nn.Linear(D_in, D_out,bias=False)
        # self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        project=self.linear(x)
        # h_relu = self.linear1(x).clamp(min=0)
        # y_pred = self.linear2(h_relu)
        return project


class Trainer():
    def __init__(self, D_in_src, D_in_tgt, D_out, device):

        self.device = device
        self.D_in_src = D_in_src
        self.D_in_tgt = D_in_tgt
        self.D_out = D_out
        self.src_mean = None
        self.tgt_mean = None
        self.init_src_model = None
        self.model_src = LinearProjection(self.D_in_src, self.D_out).to(self.device)
        self.model_tgt = LinearProjection(self.D_in_tgt, self.D_out).to(self.device)
        self.reg_ratio=0

    def exact_model_ortho_update(self, X_src, X_tgt):

        A = X_src
        B = X_tgt
        W = self.model_src.linear.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W.to(self.device)))



    def apply_model(self, src_embs, tgt_embs):
        with torch.no_grad():
            self.model_src.eval()
            self.model_tgt.eval()
            src_embs = self.model_src(src_embs.to(self.device))
            if self.model_type in ['mim']:
                tgt_embs = self.model_tgt(tgt_embs.to(self.device))
        return src_embs, tgt_embs

    def exact_model_update(self, X_src, X_tgt, model_replace):
        X_src = X_src.cpu().numpy()
        X_tgt = X_tgt.cpu().numpy()
        W = inv(X_src.transpose().dot(X_src)).dot(X_src.transpose()).dot(X_tgt)
        W = torch.from_numpy(W.T)
        W_model = model_replace.linear.weight.data
        print(W_model.size(), W.size())
        assert W_model.size() == W.size()
        W_model.copy_(W.type_as(W_model))
    
    def mim_model_update(self, X_src, X_tgt):
        # model

        avg = (X_src + X_tgt) / 2

        self.exact_model_update(X_src, self.reg_ratio * X_src + (1 - self.reg_ratio) * avg, self.model_src)
        self.exact_model_update(X_tgt, self.reg_ratio * X_tgt + (1 - self.reg_ratio) * avg, self.model_tgt)

