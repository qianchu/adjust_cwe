from variable_names import *
import torch
from helpers import *
# from evaluation_script import *
from numpy.linalg import inv
import scipy
from collections import defaultdict

def reload_best_muse(exp_path,emb_dim):
    """
    Reload the best mapping.
    """
    mapping = torch.nn.Linear(emb_dim, emb_dim, bias=False)
    path = exp_path
    # path = os.path.join(exp_path, 'best_mapping.pth')
    print('* Reloading the best model from %s ...' % path)
    # reload the model
    assert os.path.isfile(path)
    load_from_weight(path,mapping)
    return mapping

def load_from_weight(src_path,model_src_linear):
    assert os.path.isfile(src_path)
    if src_path.endswith('npy'):
        exist_w=np.load(src_path)
    else:
        exist_w=torch.load(src_path)
    to_reload = torch.from_numpy(exist_w)
    W = model_src_linear.weight.data
    assert to_reload.size() == W.size()
    W.copy_(to_reload.type_as(W))

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
    def __init__(self, D_in_src, D_in_tgt, D_out, args, device, tag, cluster_flag):

        self.device = device
        self.base_embed = args.base_embed
        self.model_tgt = None
        self.optimizer_tgt = None
        self.D_in_src = D_in_src
        self.D_in_tgt = D_in_tgt
        self.D_out = D_out
        self.model_type = args.model
        self.lr = args.lr
        self.norm = args.norm
        # assert self.norm in [CENTER, NORMALIZE, '']
        self.best_src = {}
        self.best_tgt = {}
        self.model_build(args.src_path)
        self.src_mean = None
        self.tgt_mean = None
        self.src_lg = args.src_lg
        self.eval_data = {}
        self.init_src_model = None
        self.reg_ratio = args.reg
        self.tag = tag
        self.cluster = cluster_flag
        self.lg = args.lg
        self.tgt_lg=args.tgt_lg
        self.src=args.src
        self.tgt=args.tgt

    def init_eye(self):
        W_src = self.model_src.linear.weight.data
        to_reload = torch.eye(W_src.size()[0])
        assert to_reload.size() == W_src.size()
        W_src.copy_(to_reload.type_as(W_src))

        if self.model_type == BIDIRECT_MODEL:
            W_tgt = self.model_tgt.linear.weight.data
            to_reload = torch.eye(W_tgt.size()[0])
            assert to_reload.size() == W_tgt.size()
            W_tgt.copy_(to_reload.type_as(W_tgt))

    def model_build(self, src_path):

        assert self.model_type in [BIDIRECT_MODEL, UNIDIRECT_MODEL, MUSE, EXACT, EXACT_ORTHO, MIM]
        if self.model_type == BIDIRECT_MODEL or self.model_type == MIM:
            self.model_tgt = LinearProjection(self.D_in_tgt, self.D_out)
            self.model_tgt.to(self.device)
            self.optimizer_tgt = torch.optim.SGD(self.model_tgt.parameters(), lr=self.lr)

        self.model_src = LinearProjection(self.D_in_src, self.D_out)

        self.model_src.to(self.device)
        if src_path != None and self.model_type != MUSE:
            if src_path == 'eye':
                self.init_eye()
            else:
                try:
                    self.model_src.load_state_dict(torch.load(src_path))

                except:
                    print('load from weight')
                    load_from_weight(src_path, self.model_src.linear)

        self.criterion = torch.nn.MSELoss()
        # self.criterion=torch.nn.CosineEmbeddingLoss()
        self.optimizer_src = torch.optim.SGD(self.model_src.parameters(), lr=self.lr)

    def model_update(self, X_src, X_tgt, epoch):
        X_src = X_src.type('torch.FloatTensor').to(self.device)
        X_tgt = X_tgt.type('torch.FloatTensor').to(self.device)
        # with torch.no_grad():
        #     print('loss init', self.criterion(self.model_src(X_src), X_tgt.detach()))
        if self.model_type == BIDIRECT_MODEL:
            self.bidirect_model_update(X_src, X_tgt, epoch)
        elif self.model_type == UNIDIRECT_MODEL:
            self.unidirect_model_update(X_src, X_tgt)
        elif self.model_type == MIM:
            self.mim_model_update(X_src, X_tgt)
        elif self.model_type == EXACT:
            self.exact_model_update(X_src, X_tgt, self.model_src)
            # self.model_type=UNIDIRECT_MODEL
        elif self.model_type == EXACT_ORTHO:
            self.exact_model_ortho_update(X_src, X_tgt)
            # self.model_type=UNIDIRECT_MODEL

    def apply_init_src_model(self, X_src):
        with torch.no_grad():
            self.init_src_model.eval()
            X_src = self.init_src_model(X_src.type('torch.FloatTensor').to(self.device))
        return X_src

    def unidirect_model_update(self, X_src, X_tgt):
        # model
        project_src = self.model_src(X_src)

        # updating target projection
        # loss_src = self.criterion(project_src, X_tgt.detach(),torch.ones(len(project_src)))
        loss_src = self.criterion(project_src, X_tgt.detach())
        loss_reg = self.criterion(project_src, X_src.detach())
        loss_total = 0.2 * loss_src + 0.8 * loss_reg
        self.optimizer_src.zero_grad()
        loss_total.backward()
        self.optimizer_src.step()

        print('loss: ', loss_total)

    def mim_model_update(self, X_src, X_tgt):
        # model

        avg = (X_src + X_tgt) / 2

        self.exact_model_update(X_src, self.reg_ratio * X_src + (1 - self.reg_ratio) * avg, self.model_src)
        self.exact_model_update(X_tgt, self.reg_ratio * X_tgt + (1 - self.reg_ratio) * avg, self.model_tgt)

        # project_src = self.model_src(X_src)
        # loss_src = self.criterion(project_src, X_tgt.detach())
        # loss_reg = self.criterion(project_src, X_src.detach())
        # loss_total = 0.2 * loss_src + 0.8 * loss_reg
        # self.optimizer_src.zero_grad()
        # loss_total.backward()
        # self.optimizer_src.step()

        # print('loss: ', loss_total)

    def update_source(self, project_src, gold, optimizer, X_src):
        loss_src = self.criterion(project_src, gold)
        print('mapping loss', loss_src)
        loss_reg = self.criterion(project_src, X_src.detach())
        print('regularisation loss with weight', loss_reg, self.reg_ratio)
        loss = (1.0 - self.reg_ratio) * loss_src + self.reg_ratio * loss_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    # def update_target(self, project_tgt, gold):
    #     loss_tgt=self.update_source(project_tgt,gold)
    #     return loss_tgt
    # loss_tgt = self.criterion(project_tgt, gold)
    #
    # self.optimizer_tgt.zero_grad()
    # loss_tgt.backward()
    # self.optimizer_tgt.step()

    def exact_model_update(self, X_src, X_tgt, model_replace):
        X_src_orig = X_src
        X_tgt_orig = X_tgt
        X_src = X_src.cpu().numpy()
        X_tgt = X_tgt.cpu().numpy()
        W = inv(X_src.transpose().dot(X_src)).dot(X_src.transpose()).dot(X_tgt)
        W = torch.from_numpy(W.T)
        W_model = model_replace.linear.weight.data
        print(W_model.size(), W.size())
        assert W_model.size() == W.size()
        W_model.copy_(W.type_as(W_model))
        with torch.no_grad():
            print('loss mapped after exact', self.criterion(model_replace(X_src_orig), X_tgt_orig.detach()))

    def exact_model_ortho_update(self, X_src, X_tgt):
        A = X_src
        B = X_tgt
        W = self.model_src.linear.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W.to(self.device)))
        print('loss mapped after exact ortho', self.criterion(self.model_src(X_src), X_tgt.detach()))

    def bidirect_model_update(self, X_src, X_tgt, epoch):
        # model

        project_src = self.model_src(X_src)
        project_tgt = self.model_tgt(X_tgt)
        avg = (X_src + X_tgt) / 2

        # self.exact_model_update(X_src,avg,)

        # updating source projection
        # if epoch<10:
        #     print ('update source with target, update target proj')
        #     self.update_source(project_src,X_tgt)
        #     self.update_target(project_tgt, project_src.detach())

        # self.exact_model_update(X_src,avg,self.model_src)
        # self.exact_model_update(X_tgt,avg,self.model_tgt)
        if int(epoch / 2) % 2 == 1:
            print('update source')
            self.exact_model_update(X_src, project_tgt.detach(), self.model_src)
        else:
            print('update target')
            self.exact_model_update(X_tgt, project_src.detach(), self.model_tgt)
        # if int(epoch/2)%2==1:
        #     print ('update source with target proj')
        #     loss_src=self.update_source(project_src,avg.detach(),self.optimizer_src,X_src)
        #     # print ('debug',X_src[0][0],X_tgt[0][0],avg[0][0])
        #     print('src proj loss total', loss_src)
        # else:
        #     print ('update target with source proj')
        #     loss_tgt=self.update_source(project_tgt, avg.detach(),self.optimizer_tgt,X_tgt)
        #     print ('tgt proj loss total', loss_tgt)

        # self.optimizer_src.zero_grad()
        # loss_src.backward()
        # self.optimizer_src.step()
        #
        # # updating target projection
        # # if epoch<3 or epoch>6:
        # loss_tgt = self.criterion(project_tgt, project_src.detach())
        # # else:
        # #     loss_tgt = self.criterion(project_tgt, project_tgt.detach())
        #
        #
        #
        #
        # self.optimizer_tgt.zero_grad()
        # loss_tgt.backward()
        # self.optimizer_tgt.step()
        #
        # print(loss_src, loss_tgt)

    def apply_model_en(self, test_src, test_tgt):
        with torch.no_grad():
            self.model_src.eval()
            self.model_tgt.eval()
            if self.src:
                if self.init_src_model != None:
                    test_src = self.apply_init_src_model(test_src)
                    test_tgt = self.apply_init_src_model(test_tgt)

                test_src = self.model_src(test_src.to(self.device))
                test_tgt = self.model_src(test_tgt.to(self.device))
            elif self.tgt:
                if self.model_type in [BIDIRECT_MODEL, MIM]:
                    test_src = self.model_tgt(test_src.to(self.device))
                    test_tgt = self.model_tgt(test_tgt.to(self.device))
        return test_src, test_tgt

    def eval_bdi(self, args, testset, orig_flag=''):
        print('=======evaluating bdi: ', testset)
        with torch.no_grad():
            self.bdi_test(args.test_data_dir, args.base_embed, testset, args.lg, args.avg_flag, args.error_output,
                          orig_flag, args.context_num)
            # self.eval_simlex(args.test_data_dir)

    def eval_bdi_mt(self, args, testset, orig_flag=''):
        print('=======evaluating bdi: ', testset)
        with torch.no_grad():
            self.bdi_test_mt(args.test_data_dir, args.base_embed, testset, args.lg, args.avg_flag, args.error_output,
                             orig_flag)
            # self.eval_simlex(args.test_data_dir)

    # def eval_bdi_sent(self,args,testset):
    #     with torch.no_grad():
    #         self.bdi_test_sen(args.test_data_dir,args.base_embed,testset)
    # def eval_simlex(self,test_data_dir):
    #     test_simlexf = os.path.join(os.path.dirname(test_data_dir), './vocab/SimLex-999.txt')
    #     emb_zhs, word2index_zh, index2word_zh, emb_ens, word2index_en, index2word_en = self.eval_data['bdi']
    #     if self.src == 'en':
    #         emb_ens = self.model_src(torch.from_numpy(emb_ens)).to(self.device)).cpu().detach().numpy()
    #     cos=[]
    #     scores=[]
    #     for line in open(test_simlexf):
    #         word_0=line.split()[0]
    #         word_1=line.split()[1]
    #         score=line.split()[3]
    #         if word_0 in word2index_en and word_1 in word2index_en:
    #             cos.append(1 - cosine(emb_ens[word2index_en[word_0]], emb_ens[word2index_en[word_1]]))
    #             scores.append(float(score))
    #     print (cos)
    #     print (scores)
    #     print ('total number of words in simlex found: {0}'.format(len(scores)))
    #     rho = spearmanr(scores, cos)[0]
    #     print (rho)

    def src_tgt_output_bdi(self, emb_zhs, word2index_zh, index2word_zh, emb_ens, word2index_en, index2word_en):
        if self.src_lg == 'en':
            src_embs = emb_ens
            tgt_embs = emb_zhs
            src_index2word = index2word_en
            src_word2index = word2index_en

            tgt_index2word = index2word_zh
            tgt_word2index = word2index_zh

            # emb_ens=self.model_src(torch.from_numpy(np.vstack(emb_ens)).to(self.device)).cpu().detach().numpy()
        elif self.src_lg in ['zh', 'es', 'de']:
            src_embs = emb_zhs
            tgt_embs = emb_ens
            src_index2word = index2word_zh
            src_word2index = word2index_zh
            tgt_index2word = index2word_en
            tgt_word2index = word2index_en
            # emb_zhs=self.model_src(torch.from_numpy(np.vstack(emb_zhs)).to(self.device)).cpu().detach().numpy()
        src_embs = src_embs.to(self.device)
        tgt_embs = tgt_embs.to(self.device)
        return src_embs, tgt_embs, src_word2index, src_index2word, tgt_word2index, tgt_index2word

    def src_tgt_output_bdi_w(self, zh_w, en_w):
        if self.src_lg == 'en':
            src_w = en_w
            tgt_w = zh_w
        elif self.src_lg in ['zh', 'es', 'de']:
            src_w = zh_w
            tgt_w = en_w
        return src_w, tgt_w

    # def update_cbdi_base(self,emb_zhs, word2index_zh, index2word_zh, emb_ens, word2index_en, index2word_en,test_f):
    #     for line in

    def apply_model(self, src_embs, tgt_embs):
        with torch.no_grad():
            self.model_src.eval()
            self.model_tgt.eval()
            if self.init_src_model != None:
                src_embs = self.apply_init_src_model(src_embs)

            src_embs = self.model_src(src_embs.to(self.device))
            if self.model_type in [BIDIRECT_MODEL, MIM]:
                tgt_embs = self.model_tgt(tgt_embs.to(self.device))
        return src_embs, tgt_embs



    def cls_fix(self, testset, zh_w, en_w):
        if testset in [NP_CLS, NP_CLS_WSD, NP_CLS_WSD_type, NP_CLS_type]:
            zh_w = zh_w.split('|&|')[0].split('::')[0]
        if testset in [NP_CLS_type, NP_CLS_WSD_type]:
            en_w = en_w.split('|&|')[0].split('::')[0]
        return zh_w, en_w

    def test2lst(self, test, lexsubcontext=None):
        w_list = []
        with open(test, 'r') as f:
            for line in f:
                fields = line.strip().split('\t')
                w = produce_key(fields[0])
                if lexsubcontext:
                    w_list.append((w, fields[1], fields[3]))
                else:
                    w_list.append(w)
        return sorted(list(set(w_list)))



    #
    # def data_mean(self, args, batch=None):
    #     if self.norm == CENTER:
    #         print('calculate data mean..')
    #         for ch_data, en_data, ch_data_second, en_data_second in produce_crossling_batch(args, batch):
    #             data_src = torch.from_numpy(np.vstack(en_data))
    #             data_tgt = torch.from_numpy(np.vstack(ch_data))
    #
    #             src_mean = normalize_embeddings(data_src, args.norm)
    #             tgt_mean = normalize_embeddings(data_tgt, args.norm)
    #             self.src_mean = src_mean
    #             self.tgt_mean = tgt_mean
    #             break