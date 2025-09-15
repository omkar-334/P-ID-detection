import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class OCRTrainer:
    """
    Trainer for attention-based OCR models (Prediction='Attn')
    """

    def __init__(self, wrapper, converter, lr=0.001, device="cpu"):
        self.wrapper = wrapper
        self.model = wrapper.model
        self.converter = converter
        self.device = device
        self.model.to(self.device)

        # Attention models typically use CrossEntropy
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is usually the padding
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc_word": [],
            "val_acc_char": [],
        }
        self.best_val_loss = float("inf")  # Track best validation loss

    def train_one_epoch(self, loader):
        self.model.train()
        running_loss = 0

        for images, texts in tqdm(loader, desc="Training"):
            images = images.to(self.device)

            targets, _ = self.converter.encode(texts)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(images, text=None, is_train=True)

            # Teacher forcing: shift targets
            preds = preds[:, :-1, :].contiguous()
            targets = targets[:, 1:].contiguous()

            # Flatten for CrossEntropyLoss
            loss = self.criterion(preds.view(-1, preds.size(-1)), targets.view(-1))
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        running_loss = 0
        total_chars = 0
        correct_chars = 0
        total_words = 0
        correct_words = 0

        with torch.no_grad():
            for images, texts in loader:
                images = images.to(self.device)
                targets, _ = self.converter.encode(texts)
                targets = targets.to(self.device)

                preds = self.model(images, text=None, is_train=False)
                preds = preds[:, :-1, :].contiguous()
                targets_shifted = targets[:, 1:].contiguous()

                # Loss
                loss = self.criterion(
                    preds.view(-1, preds.size(-1)), targets_shifted.view(-1)
                )
                running_loss += loss.item()

                # --- Metrics ---
                _, pred_indices = preds.max(2)
                pred_texts = self.converter.decode(pred_indices)

                for pred_text, true_text in zip(pred_texts, texts):
                    total_words += 1
                    if pred_text.strip() == true_text.strip():
                        correct_words += 1

                    min_len = min(len(pred_text), len(true_text))
                    correct_chars += sum(
                        1 for i in range(min_len) if pred_text[i] == true_text[i]
                    )
                    total_chars += max(len(pred_text), len(true_text))

        word_acc = correct_words / total_words if total_words > 0 else 0.0
        char_acc = correct_chars / total_chars if total_chars > 0 else 0.0

        return running_loss / len(loader), word_acc, char_acc

    def fit(self, train_loader, val_loader=None, epochs=10):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_acc_word, val_acc_char = (None, None, None)
            if val_loader:
                val_loss, val_acc_word, val_acc_char = self.validate(val_loader)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint()
                    print(f"Saved best OCR checkpoint at epoch {epoch + 1}")

                self.history["val_loss"].append(val_loss)
                self.history["val_acc_word"].append(val_acc_word)
                self.history["val_acc_char"].append(val_acc_char)

            self.history["train_loss"].append(train_loss)

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Word Acc: {val_acc_word:.4f} | Char Acc: {val_acc_char:.4f}"
                if val_loader
                else f"Train Loss: {train_loss:.4f}"
            )

    def save_checkpoint(self, name="best_ocr_model"):
        """Save current model using the wrapper"""
        self.wrapper.model = self.model
        return self.wrapper.save_model(name=name)

    def predict(self, images):
        """Run inference on new images"""
        self.model.eval()
        if not isinstance(images, list):
            images = [images]

        images = [img.to(self.device) for img in images]
        preds_texts = []

        with torch.no_grad():
            for img in images:
                img = img.unsqueeze(0) if img.ndim == 3 else img  # batch dimension
                preds = self.model(img, text=None, is_train=False)
                _, pred_indices = preds[:, :-1, :].max(2)
                text = self.converter.decode(pred_indices)
                preds_texts.append(text[0] if isinstance(text, list) else text)

        return preds_texts
