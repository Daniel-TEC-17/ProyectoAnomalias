def load_config(config_path):
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class Trainer:
    def __init__(self, model, data_loader, optimizer, criterion, config):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch in self.data_loader:
            inputs, targets = batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(self.data_loader)
        print(f'Epoch [{epoch + 1}/{self.config["num_epochs"]}], Loss: {average_loss:.4f}')

    def train(self):
        for epoch in range(self.config['num_epochs']):
            self.train_one_epoch(epoch)

def main():
    from models.model import MyModel
    from data.loader import get_data_loader
    from utils.helpers import get_optimizer, get_criterion

    config = load_config('configs/default.yaml')
    model = MyModel()
    data_loader = get_data_loader(config['data'])
    optimizer = get_optimizer(model.parameters(), config['optimizer'])
    criterion = get_criterion(config['loss'])

    trainer = Trainer(model, data_loader, optimizer, criterion, config)
    trainer.train()

if __name__ == '__main__':
    main()