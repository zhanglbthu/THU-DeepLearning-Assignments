import torch
import torch.nn as nn
import torch.optim as optim
import pickle

with open("car_racing_data_32x32_120.pkl", "rb") as f:
    data = pickle.load(f)

class WorldModel(nn.Module):
    def __init__(self, action_size, hidden_size, output_size):
        super(WorldModel, self).__init__()
        pass

    def forward(self, state, action, hidden=None):
        """
        TODO: Implement your model here. You should implement two models, one using built-in lstm layers and one implemented by yourself.

        Forward pass for the WorldModel.
        Args:
            state: Tensor of shape [batch_size, 3, 32, 32] (current RGB image at this time step).
            action: Tensor of shape [batch_size, action_size] (3-dimensional action vector).
            hidden: Tuple of hidden states for LSTM, (h_t, c_t), each of shape [1, batch_size, hidden_size].
            
        Returns:
            next_state_pred: Tensor of shape [batch_size, output_size] (flattened next state prediction, 3*32*32).
            hidden: Updated hidden state tuple (h_t, c_t) for the LSTM.
        """
        pass

class WorldModelDataLoader:
    def __init__(self, data, batch_size, sequence_length, device):
        self.data = data
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device

        # 拆分数据为 train, valid, test 集合
        split_train = int(0.8 * len(self.data))
        split_valid = int(0.1 * len(self.data))
        self.train_data = self.data[:split_train]
        self.valid_data = self.data[split_train:split_train + split_valid]
        self.test_data = self.data[split_train + split_valid:]

        self.set_train()

    def set_train(self):
        self.current_data = self.train_data
        self.index = 0
        self.sub_index = 0  # 子序列的起始索引

    def set_valid(self):
        self.current_data = self.valid_data
        self.index = 0
        self.sub_index = 0

    def set_test(self):
        self.current_data = self.test_data
        self.index = 0
        self.sub_index = 0

    def get_batch(self):
        states, actions = [], []
        batch_data = self.current_data[self.index: self.index + self.batch_size]

        for sequence in batch_data:
            state_seq = [torch.tensor(step[0]) for step in sequence[self.sub_index:self.sub_index + self.sequence_length]]
            action_seq = [torch.tensor(step[1]) for step in sequence[self.sub_index:self.sub_index + self.sequence_length]]
            if len(state_seq) < self.sequence_length:
                pad_len = self.sequence_length - len(state_seq)
                state_seq += [torch.zeros_like(state_seq[0])] * pad_len
                action_seq += [torch.zeros_like(action_seq[0])] * pad_len

            states.append(torch.stack(state_seq))
            actions.append(torch.stack(action_seq))

        self.sub_index += self.sequence_length
        if self.sub_index >= len(self.current_data[self.index]):
            self.index += self.batch_size  
            self.sub_index = 0  
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)

        end_flag = self.index >= len(self.current_data)

        return states, actions, end_flag


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size = 3
hidden_size = 128
output_size = 32 * 32 * 3
batch_size = 16
sequence_length = 10
num_epochs = 50
learning_rate = 3e-4


data_loader = WorldModelDataLoader(data, batch_size, sequence_length, device)
model = WorldModel(action_size=action_size, hidden_size=hidden_size, output_size=output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train(num_epochs=50):
    best_val_loss = float('inf') 

    for epoch in range(num_epochs):
        model.train()
        data_loader.set_train()
        total_train_loss = 0
        total_train_samples = 0

        while True:
            states, actions, end_flag = data_loader.get_batch()
            batch_size_actual = states.size(0)
            # Initialize the hidden state (h_0, c_0) for the LSTM, resetting it for each new batch
            hidden = (torch.zeros(1, batch_size_actual, hidden_size).to(device),
                      torch.zeros(1, batch_size_actual, hidden_size).to(device))
            # Loop through each time step in the sequence
            for t in range(sequence_length - 1):
                current_state = states[:, t]
                action = actions[:, t]
                next_state = states[:, t + 1].view(batch_size_actual, -1)

                next_state_pred, hidden = model(current_state, action, hidden)
                loss = criterion(next_state_pred, next_state)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                hidden = tuple([h.detach() for h in hidden])

                total_train_loss += loss.item()
                total_train_samples += 1

            if end_flag:
                break

        avg_train_loss = total_train_loss / total_train_samples
        val_loss = evaluate()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "world_model_best.pth")
            print("Best model saved.")


def evaluate():
    model.eval()
    data_loader.set_valid()
    total_val_loss = 0
    total_val_samples = 0

    with torch.no_grad():
        while True:
            states, actions, end_flag = data_loader.get_batch()
            batch_size_actual = states.size(0)

            hidden = (torch.zeros(1, batch_size_actual, hidden_size).to(device),
                      torch.zeros(1, batch_size_actual, hidden_size).to(device))

            for t in range(sequence_length - 1):
                current_state = states[:, t]
                action = actions[:, t]
                next_state = states[:, t + 1].view(batch_size_actual, -1)

                next_state_pred, hidden = model(current_state, action, hidden)
                loss = criterion(next_state_pred, next_state)

                total_val_loss += loss.item()
                total_val_samples += 1

            if end_flag:
                break

    avg_val_loss = total_val_loss / total_val_samples
    return avg_val_loss

def test():
    model.eval()
    data_loader.set_test()
    total_test_loss = 0
    total_test_samples = 0

    with torch.no_grad():
        while True:
            states, actions, end_flag = data_loader.get_batch()
            batch_size_actual = states.size(0)

            hidden = (torch.zeros(1, batch_size_actual, hidden_size).to(device),
                      torch.zeros(1, batch_size_actual, hidden_size).to(device))

            for t in range(sequence_length - 1):
                current_state = states[:, t]
                action = actions[:, t]
                next_state = states[:, t + 1].view(batch_size_actual, -1)
                next_state_pred, hidden = model(current_state, action, hidden)
                loss = criterion(next_state_pred, next_state)

                total_test_loss += loss.item()
                total_test_samples += 1

            if end_flag:
                break

    avg_test_loss = total_test_loss / total_test_samples
    print(f"Test Loss: {avg_test_loss:.4f}")

train()
evaluate()
test()