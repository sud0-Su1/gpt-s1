from data_api import get_train_valid_data, get_batch, get_vocab_size, generate_string
from neural_network.bigram import BigramLanguageModel

import torch

import argparse
import os


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, valid_data, block_size=block_size, batch_size=batch_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == '__main__':
    # Take input from the user command line
    parser = argparse.ArgumentParser(description='Generative Language Model using Bigram')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--generate', action='store_true', help='Generate text from the model')
    args = parser.parse_args()

    if not args.train and not args.generate:
        print('Please specify either --train or --generate')
        exit()

    # set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # get the data
    train_data, valid_data = get_train_valid_data()
    vocab_size = get_vocab_size()

    batch_size = 16 # how many independent sequences will we process in parallel?
    block_size = 32 # what is the maximum context length for predictions?
    max_iters = 5000
    eval_interval = 100
    learning_rate = 1e-3
    eval_iters = 200
    n_embed = 64
    n_head = 4
    n_layer = 4
    dropout = 0.0

    model = BigramLanguageModel(vocab_size=vocab_size, n_embed=n_embed, block_size=block_size, n_layer=n_layer, n_head=n_head, dropout=dropout, device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)    

    if args.train:
        # Number of parameters in the model
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f'Number of parameters: {num_params} million\n\n')
        print('Training the model')

        for iter in range(max_iters):
            # Every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # Sample a batch of data
            xb, yb = get_batch('train', train_data, valid_data, block_size=block_size, batch_size=batch_size)

            # Evaluate the loss
            logits, loss = model(xb, yb)

            # Backpropagate loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()    

        # Save the model
        torch.save(model.state_dict(), 'model.pt')

    elif args.generate:
        print('Generating text from the model')
        print('-' * 20)

        # Check if the model exists
        if not os.path.exists('model.pt'):
            print('Model not found. Please train the model first.')
            exit()

        # Load the model
        model.load_state_dict(torch.load('model.pt'))
        model.eval()

        # Generate from the model
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_ints = model.generate(context, max_new_tokens=2000)[0].tolist()
        generated_string = generate_string(generated_ints)
        
        # Save the generated text
        with open('generated_text.txt', 'w') as f:
            f.write(generated_string)

        print(generated_string)
