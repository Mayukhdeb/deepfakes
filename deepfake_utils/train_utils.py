import autoencoder

class deepfake_trainer():
    def __init__(self, model, train_loader_a, train_loader_b, checkpoint_path = None):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))
    
        self.train_loader_a = train_loader_a
        self.train_loader_b = train_loader_b


    
    def train(self, num_epochs, checkpoint_path = None, save_path = "model.pth"):
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))

        ## train loop here 
