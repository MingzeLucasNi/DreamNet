import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from sleepnet import TextCNN  # Import the TextCNN from the model.py
import argparse
import torch.nn as nn

def tokenize_data(batch):
    encodings = tokenizer.batch_encode_plus(
        batch['text'],
        truncation=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',  # This will make the tokenizer return PyTorch tensors.
    )
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'label': batch['label']
    }


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def parse_args():
    parser = argparse.ArgumentParser(description="Train a TextCNN model using the AG News dataset.")
    
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs to train.")
    parser.add_argument("--model_save_path", default="./textcnn_bert_agnews.pth", type=str, help="Path to save the trained model.")
    parser.add_argument("--dataset", default="ag_news", type=str, help="Dataset to use for training and evaluation.")

    
    return parser.parse_args()

def main():
    args = parse_args()

    # Load 'ag_news' dataset
    dataset = load_dataset(args.dataset)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_data = dataset['train'].map(tokenize_data, batched=True)
    test_data = dataset['test'].map(tokenize_data, batched=True)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(num_classes=4, vocab_size=tokenizer.vocab_size, embed_size=768, num_filters=32, filter_sizes=[2, 3, 4], dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # print(batch['input_ids'][0].shape)
            # print(type(batch['input_ids'][0]))
            input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
            # # print(input_ids.shape)
            # attention_mask = torch.stack(batch['attention_mask'],dim=1).to(device)

            # input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)

            labels = batch['label'].to(device)
            
            outputs = model(input_ids)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # Optionally, print progress during the epoch.
            if (batch_idx + 1) % 100 == 0:  # Print every 100 batches.
                print(f"Epoch [{epoch + 1}/{args.epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)

        avg_val_loss = total_val_loss / len(test_loader)
        val_accuracy = correct_predictions.double() / len(test_loader.dataset)

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print('-' * 60)
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n")
        
        model_path = args.model_save_path
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()