import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import matplotlib.pyplot as plt

with open("data/car_racing_data_32x32_120.pkl", "rb") as f:
    data = pickle.load(f)

class WorldModel(nn.Module):
    def __init__(self, action_size, hidden_size, output_size, use_builtin_lstm=True):
        super(WorldModel, self).__init__()
        
        self.hidden_size = hidden_size
        
        # CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # -> [batch_size, 16, 16, 16]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> [batch_size, 32, 8, 8]
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.state_size = 32 * 8 * 8
        
        # Action embedding
        self.action_fc = nn.Linear(action_size, hidden_size)
        
        # LSTM layer
        if use_builtin_lstm:
            print("Using built-in LSTM")
            self.lstm = nn.LSTM(self.state_size + hidden_size, hidden_size, batch_first=True)
        else:
            print("Using custom LSTM")
            self.lstm = CustomLSTM(self.state_size + hidden_size, hidden_size)
            
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

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
        batch_size = state.size(0)
        
        # Encode the current state
        state_encoded = self.encoder(state)
        
        # Encode the action
        action_encoded = F.relu(self.action_fc(action))
        
        # Concatenate the state and action
        lstm_input = torch.cat((state_encoded, action_encoded), dim=-1).unsqueeze(1)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size)
            
        lstm_output, hidden = self.lstm(lstm_input, hidden)
        
        # Decode the next state
        next_state_pred = self.fc(lstm_output.squeeze(1))
        return next_state_pred, hidden
            
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))
        
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.Wx = nn.Linear(input_size, 4 * hidden_size)
        self.Wh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, hidden):
        '''
        x: [batch_size, 1, input_size]
        hidden: [[1, batch_size, hidden_size], [1, batch_size, hidden_size]]
        '''
        h_t, c_t = hidden
        # reshape hidden states to batch first
        h_t = h_t.permute(1, 0, 2)
        c_t = c_t.permute(1, 0, 2)
        
        gates = self.Wx(x) + self.Wh(h_t)
        
        i, f, o, g = torch.chunk(gates, 4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c_t + i * g
        h_next = o * torch.tanh(c_next)
        
        # reshape hidden states back to time first
        lstm_output = h_next
        c_next = c_next.permute(1, 0, 2)
        h_next = h_next.permute(1, 0, 2)
                
        return lstm_output, (h_next, c_next)

class WorldModelDataLoader:
    def __init__(self, data, batch_size, sequence_length, device):
        self.data = data
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device

        # 拆分数据为 train, valid, test 集合
        split_train = int(0.8 * len(self.data)) # 96
        split_valid = int(0.1 * len(self.data)) # 12
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
num_epochs = 100
learning_rate = 3e-4

data_loader = WorldModelDataLoader(data, batch_size, sequence_length, device)
model = WorldModel(action_size=action_size, hidden_size=hidden_size, output_size=output_size, use_builtin_lstm=False).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = f"checkpoints/{current_time}"
os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter(log_dir=checkpoint_dir)

def train(num_epochs=50):
    best_val_loss = float('inf') 

    for epoch in range(num_epochs):
        model.train()
        data_loader.set_train()
        total_train_loss = 0
        total_train_samples = 0

        while True:
            states, actions, end_flag = data_loader.get_batch()
            '''
            states: [batch_size, sequence_length, 3, 32, 32]
            actions: [batch_size, sequence_length, action_size]
            '''
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

        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/Val", val_loss, epoch + 1)
        
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

def test_rollout(rollout_steps=9, quan_eval=False):
    model.eval()
    data_loader.set_test()
    
    total_teacher_forcing_loss = 0
    total_autoregressive_loss = 0
    total_samples = 0
    
    ground_truth_frames = []
    teacher_forcing_frames = []
    autoregressive_frames = []
    
    with torch.no_grad():
        while True:
            states, actions, end_flag = data_loader.get_batch()

            batch_size_actual = states.size(0)
            
            hidden = (torch.zeros(1, 1, hidden_size).to(device),
                        torch.zeros(1, 1, hidden_size).to(device))

            current_state = states[0, 0].unsqueeze(0) # [1, 3, 32, 32]
            
            teacher_hidden = hidden
            autoregressive_hidden = hidden
            
            teacher_pred_frames = []
            
            if quan_eval:
                rollout_steps = sequence_length - 1
            
            # Teacher forcing Rollout
            for t in range(rollout_steps):
                
                action = actions[0, t].unsqueeze(0) # [1, action_size]
                next_state = states[0, t + 1].unsqueeze(0) # [1, 3, 32, 32]

                next_state_pred, teacher_hidden = model(current_state, action, teacher_hidden)
                loss = criterion(next_state_pred, next_state.view(1, -1))
                total_teacher_forcing_loss += loss.item()
                
                # * Update the current state
                current_state = next_state
                
                if not quan_eval:
                    ground_truth_frames.append(next_state.view(3, 32, 32).cpu().numpy())
                    teacher_pred_frames.append(next_state_pred.view(3, 32, 32).cpu().numpy())
                
            teacher_forcing_frames.extend(teacher_pred_frames)
            
            # Autoregressive Rollout
            current_state = states[0, 0].unsqueeze(0)
            autoregressive_pred_frames = []
            for t in range(rollout_steps):
                action = actions[0, t].unsqueeze(0)
                next_state = states[0, t + 1].unsqueeze(0)
                
                next_state_pred, autoregressive_hidden = model(current_state, action, autoregressive_hidden)
                loss = criterion(next_state_pred, next_state.view(1, -1))
                total_autoregressive_loss += loss.item()
                
                # * Update the current state
                current_state = next_state_pred.view(1, 3, 32, 32)
                
                if not quan_eval:
                    autoregressive_pred_frames.append(next_state_pred.view(3, 32, 32).cpu().numpy())
            
            autoregressive_frames.extend(autoregressive_pred_frames)
            
            total_samples += rollout_steps
            
            if end_flag:
                break
        
        # Compute the average loss
        avg_teacher_forcing_loss = total_teacher_forcing_loss / total_samples
        avg_autoregressive_loss = total_autoregressive_loss / total_samples
        print(f"Teacher Forcing Loss: {avg_teacher_forcing_loss:.4f}")
        print(f"Autoregressive Loss: {avg_autoregressive_loss:.4f}")
        
        # Visualize the rollouts
        if not quan_eval:
            visualize_rollouts(ground_truth_frames, teacher_forcing_frames, autoregressive_frames)
        
def visualize_rollouts(ground_truth, teacher_forcing, autoregressive):

    steps = len(teacher_forcing)
    print(f"Visualizing rollouts for {steps} steps")
    
    save_dir = "rollouts"
    print(f"Saving rollouts to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(steps, 3, figsize=(5, 15))

    column_labels = ["Ground Truth", "Teacher Forcing", "Autoregressive"]
    for col in range(3):
        axes[0, col].set_title(column_labels[col], fontsize=12, pad=10)

    for row in range(steps):
        axes[row, 0].imshow(ground_truth[row].transpose(1, 2, 0))  # Ground Truth
        axes[row, 1].imshow(teacher_forcing[row].transpose(1, 2, 0))  # Teacher Forcing
        axes[row, 2].imshow(autoregressive[row].transpose(1, 2, 0))  # Autoregressive

        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.001, hspace=0.2)

    plt.savefig(f"{save_dir}/rollouts_vertical.png")
    print(f"Rollout visualization saved to {save_dir}/rollouts_vertical.png")
    plt.show()

    
# train()
# evaluate()
# test()

ckpt_path = 'world_model_best.pth'
model.load_state_dict(torch.load(ckpt_path))
# test()
test_rollout(rollout_steps=10, quan_eval=True)