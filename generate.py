from music import LSTMModel, device, vectorize_string, note_vector, vocab_size, params
import torch
from tqdm import tqdm
import numpy as np
import os


model = LSTMModel(vocab_size, params["embedding_dim"], params["hidden_size"])
checkpoint_path = os.path.join("training_checkpoints", "my_ckpt")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()


def generate_text(model, start_string, generation_length=1000):
    input_idx = vectorize_string(start_string)
    input_idx = torch.tensor([input_idx], dtype=torch.long).to(device)

    state = model.init_hidden(input_idx.size(0), device)
    text_generated = []

    if hasattr(tqdm, "_instances"): tqdm._instances.clear()
    for _ in tqdm(range(generation_length)):
        predictions, state = model(input_idx, state, return_state=True)
        print(predictions)
        predictions = predictions[:, -1, :] 
        input_idx = torch.multinomial(torch.softmax(predictions, dim=-1), num_samples=1)
        text_generated.append(note_vector[input_idx.item()])

    return start_string + "".join(text_generated)

start_string = "X"
generated_text = generate_text(model, start_string, generation_length=2000)
print(generated_text)

with open("generated_songs.abc", "w") as f:
    f.write(generated_text)
