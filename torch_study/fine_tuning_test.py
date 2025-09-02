import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset


# 定义数据集
class TextDataset(Dataset):
    def __init__(self, _texts, _labels, _tokenizer, max_len):
        self.texts = _texts
        self.labels = _labels
        self.tokenizer = _tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# 定义模型
class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(bert_model.config.hidden_size, num_classes)

    def forward(self, _input_ids, _attention_mask):
        outputs = self.bert(input_ids=_input_ids, attention_mask=_attention_mask)
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[1]
        output = self.dropout(pooled_output)
        return self.out(output)


# 数据准备
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
texts = ["Example text 1", "Example text 2"]
labels = [0, 1]
dataset = TextDataset(texts, labels, tokenizer, max_len=128)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 模型、优化器和学习率调度器
model = BertClassifier(BertModel.from_pretrained('bert-base-uncased'), num_classes=2)
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(dataloader) * 3  # 3 epochs
num_warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

# 训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 保存模型
# 保存模型和相关配置
model_info = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'num_classes': 2,
    'max_len': 128,
    'model_name': 'bert-base-uncased'
}
torch.save(model_info, 'bert_finetuned.pth')
