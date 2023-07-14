import torch
from pytorch_pretrained_bert import BertTokenizer,  BertModel


def Bert2vector(input_ids):

    model = BertModel.from_pretrained('bert-base-chinese')
    model.eval()
    model.cuda()

    with torch.no_grad():
        encoded_layers, pooled_output = model(input_ids)
        return pooled_output


if __name__ == '__main__':
    input_ids = torch.LongTensor([[1, 2]])
    print(Bert2vector(input_ids).shape)
