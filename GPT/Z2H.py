import torch # we use PyTorch: https://pytorch.org
import torch.nn as nn
from datetime import datetime
import os
from torch.nn import functional as F

# hyper params
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cpu' if torch.backends.mps.is_available() else 'cpu'
print("device is:" , device)
eval_iters = 200
n_embd = 384 # number of embeddings
n_head = 6
n_layer = 6
dropout = 0.2

output_dir = "./savedModels/"

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# read in the file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

# print(text[:1000])

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"vocab is of size = {vocab_size} and looks like this:")
print(''.join(chars))

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("Fhii there"))
print(decode(encode("hii there")))

# let's now encode the entire text dataset and store it into a torch.Tensor

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
# print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
print("Training data size:", "\t", n, "\n", "Validation data size:", "\t", len(data)-n, "\n")

train_data[:block_size+1]

torch.tensor([18, 47, 56, 57, 58,  1, 15, 47, 58])

# quick test train
x = train_data[:block_size]
y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when input is {context} the target: {target}")

# batch and run
torch.manual_seed(1337)


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target: {target}")


# let's tensor it up
print(xb) # our input to the transformer

torch.manual_seed(1337)

# The head module!
class Head(nn.Module):
    """ one head of self attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape 
        k = self.key(x) # [B, T, C]
        q = self.query(x) # [B, T, C]

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) # fill with 0s in tril
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

# Mooooooore dots (attention heads)
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
           out = torch.cat([h(x) for h in self.heads], dim=-1) # dim = -1 means concat on the channel dimension
           out = self.dropout(self.proj(out))
           return out

class FeedForward2(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_emdb: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward2(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # # classic MHA and a forward
        # x = self.sa(x)
        # x = self.ffwd(x)

        # addition / skip / residual pathways added in
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BatchNorm1d:
    """ normalize the batches to have unit variance """

    def __init__(self, dim ,eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffer (trained with a running 'momentum update')
        self.running_mean = torch.ones(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        #calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = x.var(0, keepdim=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta

        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class LayerNormIThink:
    """ normalize the batches to have unit variance """

    def __init__(self, dim ,eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        # self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # # buffer (trained with a running 'momentum update')
        # self.running_mean = torch.ones(dim)
        # self.running_var = torch.ones(dim)

    def __call__(self, x):
        #calculate the forward pass
        # if self.training:
        xmean = x.mean(1, keepdim=True) # batch mean
        xvar = x.var(1, keepdim=True) # batch variance
        # else:
        #     xmean = self.running_mean
        #     xvar = self.running_var
        
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta

        # # update the buffers
        # if self.training:
        #     with torch.no_grad():
        #         self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        #         self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

# create a massive lookup for prediction, n x n. 
# This will only use the character to predict, but not prior characters. This isn't too bad,
# but clearly, there's more to do.
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # # set up some self blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        # # set up some self blocks
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd),
        # )

        # final LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)

        # create a linear layer at the head from embeddings to vocab size
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # # self.sa_head = Head(n_embd) # single head attention
        # self.sa_heads = MultiHeadAttention(4, n_embd // 4) # ie, 4 heads of 8-dimension self-attention
        # self.ffwd = FeedForward2(n_embd)

    def forward(self, idx, targets=None):
        
        # get out B and T
        B, T = idx.shape 

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T,C
        x = tok_emb + pos_emb # (B,T,C) - add the position and token embeddings together!
        x = self.blocks(x) # apply one head of self attention! (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        # if you aren't given targets, we'll just get the logits
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # loss party over here!
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, loss = self(idx_cond)
            
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# Apparently this is a great language model choice
model = BigramLanguageModel()
m = model.to(device) # set the correct device for optimization!
print("model moved to", device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

# let's not do this now - unoptimized!
# print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


# # create a PyTorch optimizer
# optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# for steps in range(max_iters): # increase number of steps for good results... 
    
#     # sample a batch of data
#     xb, yb = get_batch('train')

#     # evaluate the loss
#     logits, loss = m(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# print(loss.item())

# # generate from the model
# context = torch.zeros((1,1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate) # -4 is the rule of thumb, but for small, go -3

for iter in range(max_iters): # increase number of steps for good results... 
    
    # every once in a while, evalutate the loss on train and val sets
    if iter % eval_iters == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f} val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"loss.item() = {loss.item()}")

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=3000)[0].tolist()))

# print the latest version (tokens increase = better)
#print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))


# save the model to disk for later - using dict for fun
datetime_detail = datetime.now().strftime("%d-%m-%Y-%H.%M.%S")
outfile = os.path.join(output_dir, datetime_detail + ".pt")
torch.save(m.state_dict(), outfile)

