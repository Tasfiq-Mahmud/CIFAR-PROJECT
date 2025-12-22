from pyexpat import model
import torch
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self, model, train_data, val_data, loss_function, optimizer,scheduler=None):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_loss_data = []
        self.val_loss_data = []
        self.train_acc_data = []
        self.val_acc_data = []
        self.scheduler = scheduler

    def enable_dropout(self):
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def train(self,epochs,n_models=1,save=None):
        for epoch in range(epochs):
            total_acc_train = 0
            total_acc_val = 0
            total_loss_train = 0
            total_loss_val = 0
            self.model.train()
            for img, label in self.train_data:
                self.optimizer.zero_grad()
                results = self.model(img)
                khoti = self.loss_function(results, label)
                khoti.backward()
                self.optimizer.step()
                acc = (torch.argmax(results, axis=1) == label).sum().item()
                total_acc_train += acc
                total_loss_train += khoti.item()
            
            self.model.eval()

            if n_models > 1:
                self.enable_dropout()
            for val_img, val_labels in self.val_data:
                with torch.no_grad():
                    ensemble_outputs = torch.zeros((val_img.size(0), n_models, 10))
                    for i in range(n_models):
                        outputs = self.model(val_img)
                        ensemble_outputs[:, i, :] = outputs
                    avg_outputs = ensemble_outputs.mean(dim=1).to(val_labels.device)
                    val_khoti = self.loss_function(avg_outputs, val_labels)
                    total_loss_val += val_khoti.item()
                    val_acc = (torch.argmax(avg_outputs, axis=1) == val_labels).sum().item()
                    total_acc_val += val_acc

            self.train_loss_data.append(round(total_loss_train / 1000, 4))
            self.val_loss_data.append(round(total_loss_val / 1000, 4))
            self.train_acc_data.append(round(total_acc_train / (len(self.train_data.dataset)) * 100, 4))
            self.val_acc_data.append(round(total_acc_val / (len(self.val_data.dataset)) * 100, 4))
            s=f'''Epoch {epoch+1}/{epochs}, Train Loss: {round(total_loss_train/100, 4)} Train Accuracy {round((total_acc_train)/len(self.train_data.dataset) * 100, 4)}
                      Validation Loss: {round(total_loss_val/100, 4)} Validation Accuracy: {round((total_acc_val)/len(self.val_data.dataset) * 100, 4)}'''
            s+="="*25
            try:
                if save:
                    with open(save, 'a') as file:
                        file.write(s+"\n\n")
                        print(f"Successfully saved the string to {save}")
                else:
                    print(s)
            except IOError as e:
                print(f"An error occurred while writing the file: {e}")
            if self.scheduler:
                self.scheduler.step()


    def plot_metrics(self,save=None):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        axs[0].plot(self.train_loss_data, label='Training Loss')
        axs[0].plot(self.val_loss_data, label='Validation Loss')
        axs[0].set_title('Training and Validation Loss over Epochs')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        axs[1].plot(self.train_acc_data, label='Training Accuracy')
        axs[1].plot(self.val_acc_data, label='Validation Accuracy')
        axs[1].set_title('Training and Validation Accuracy over Epochs')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()

        if save:
            plt.savefig(save)
        else:
            plt.show()