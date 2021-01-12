import torch
from model import BertClassifier
from transformers import BertTokenizer, BertConfig
from train import get_bert_input
import pandas as pd

labels = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']

bert_config = BertConfig.from_pretrained('chinese_wwm_pytorch')
bert_config.num_labels = len(labels)
model = BertClassifier(bert_config)
model.load_state_dict(torch.load('./best_model_on_trainset.pkl', map_location=torch.device('cpu')))

tokenizer = BertTokenizer(vocab_file='chinese_wwm_pytorch/vocab.txt')

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model = torch.nn.DataParallel(model, device_ids=[2])
model.to(device)

def predict_text(text):
    input_id, attention_mask, token_type_id = get_bert_input(text, tokenizer)

    input_id = torch.tensor([input_id], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)
    token_type_id = torch.tensor([token_type_id], dtype=torch.long)

    predicted = model(
        input_id,
        attention_mask,
        token_type_id,
    )
    pred_label = torch.argmax(predicted, dim=1)
    # print(labels[pred_label])
    return labels[pred_label]

# print('新闻类别分类')
test_data = pd.read_csv('./dataset/unlabeled_data.csv',encoding='utf-8')
test_data['class_label'] = test_data['content'].apply(lambda x:predict_text(x))
print(test_data)
# test_data.drop()
test_data.to_csv('res2.csv',encoding='utf-8',index=False)
