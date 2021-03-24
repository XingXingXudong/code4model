# English-Chinese translation example.
import time
import torch
from torch.utils import data

from dataset import generate_nmt_batches, NMTDataset
from transformer import make_model, NoamOpt, SimpleLossCompute, LabelSmoothing

def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model(batch['x_source'], batch['x_target'], batch['src_mask'], batch['tgt_mask'])
        loss = loss_compute(out, batch['y_target'], batch['ntokens'])
        total_loss += loss
        tokens += loss
        total_tokens += batch['ntokens']

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f." %
                  (i, loss / batch['ntokens'], total_tokens / elapsed))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens

if __name__ == "__main__":
    dataset = NMTDataset.load_dataset_and_make_vectorizer(
        # "/home/liuxd/home/NLP/PyTorchNLPBook/code4model/data/translation2019zh_train-df_100000.csv"
        "/home/liuxd/home/NLP/PyTorchNLPBook/code4model/data/translation2019zh_train-df_70w.csv"
    )
    src_vocab_size = len(dataset.get_vectorizer().source_vocab)
    tgt_vocab_size = len(dataset.get_vectorizer().target_vocab)
    padding_idx = dataset.get_vectorizer().target_vocab.lookup_token('<MASK>')
    criterion = LabelSmoothing(size=tgt_vocab_size, padding_idx=0, smoothing=0.1)
    criterion.cuda()
    model = make_model(src_vocab_size, tgt_vocab_size, 6)
    model.cuda()
    model_opt = NoamOpt(
        model.src_embed[0].d_model, 1, 8000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98),
                         eps=1e-9))
    loss_compute = SimpleLossCompute(model.generator, criterion, model_opt)

    # train
    model.train()
    for epcho in range(10):
        data_iter = generate_nmt_batches(dataset, 16, device="cuda")
        run_epoch(data_iter, model, loss_compute)