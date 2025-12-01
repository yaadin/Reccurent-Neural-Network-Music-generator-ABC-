import torch
import torch.nn as nn
import torch.optim as optim
import mitdeeplearning as mdl
import numpy as np
import os
from tqdm import tqdm


COMET_API_KEY = ""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), "Please enable GPU from runtime settings"


with open("songs.txt", "r", encoding="utf-8") as f:
    songs_joined = f.read()

vocab = sorted(set(songs_joined))


note_vector = {u: i for i, u in enumerate(vocab)}
vector_char = np.array(vocab)

def vectorize_string(string):
    return np.array([note_vector[c] for c in string])

vectorized_songs = vectorize_string(songs_joined)

def get_batch(vectorized_songs, seq_length, batch_size):
    n = vectorized_songs.shape[0] - 1
    idx = np.random.choice(n - seq_length, batch_size)

    input_batch = [vectorized_songs[i : i + seq_length] for i in idx]
    output_batch = [vectorized_songs[i+1 : i + seq_length+1] for i in idx]
    x_batch = torch.tensor(input_batch, dtype=torch.long)
    y_batch = torch.tensor(output_batch, dtype=torch.long)

    return x_batch, y_batch

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device))

    def forward(self, x, state=None, return_state=False):
        x = self.embedding(x)
        if state is None:
            state = self.init_hidden(x.size(0), x.device)
        out, state = self.lstm(x, state)
        out = self.fc(out)
        return out if not return_state else (out, state)


cross_entropy = nn.CrossEntropyLoss()
def compute_loss(labels, logits):
    batched_labels = labels.view(-1)
    batched_logits = logits.view(-1, logits.size(-1))
    return cross_entropy(batched_logits, batched_labels)


vocab_size = len(vocab)
params = dict(
    num_training_iterations=3000,
    batch_size=8,
    seq_length=100,
    learning_rate=5e-3,
    embedding_dim=256,
    hidden_size=1024,
)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
os.makedirs(checkpoint_dir, exist_ok=True)


####################################################################################################################################################################################
def create_experiment():
    import comet_ml
    experiment = comet_ml.Experiment(api_key=COMET_API_KEY, project_name="rnn_music")
    for param, value in params.items():
        experiment.log_parameter(param, value)
    return experiment

def train_model():
    model = LSTMModel(vocab_size, params["embedding_dim"], params["hidden_size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    history = []
    plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
    experiment = create_experiment()

    if hasattr(tqdm, "_instances"): tqdm._instances.clear()
    for iter in tqdm(range(params["num_training_iterations"])):
        x_batch, y_batch = get_batch(vectorized_songs, params["seq_length"], params["batch_size"])
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        model.train()
        optimizer.zero_grad()
        y_hat = model(x_batch)
        loss = compute_loss(y_batch, y_hat)
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        plotter.plot(history)
        experiment.log_metric("loss", loss.item(), step=iter)

        if iter % 100 == 0:
            torch.save(model.state_dict(), checkpoint_prefix)

    torch.save(model.state_dict(), checkpoint_prefix)
    experiment.flush()


if __name__ == "__main__":
    train_model()
