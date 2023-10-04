
import torch # we use PyTorch: https://pytorch.org
import torch.nn.functional as F
import torch.nn as nn

torch.manual_seed(1337)
B, T, C = 4, 8, 2 #batch, time, channel
x = torch.randn(B,T,C)
x.shape 

torch.Size([4, 8 , 2])

xbow = torch.zeros((B, T, C))

# basic version with for loop
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev, 0)


# let's use matrices!
weiOld = torch.tril(torch.ones(T,T))
weiOld = weiOld / weiOld.sum(1, keepdim=True)
# print(wei)
xbow2 = weiOld @ x # (wei = B, T,T) @ (B, T, C) ------> B, T, C
print(f"xbow and xbow2 match? {torch.allclose(xbow, xbow2)}")

tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
print(f"wei = {wei}")
wei = F.softmax(wei, dim=-1)
print(f"wei and weiOld match? {torch.allclose(wei, weiOld)}")
print(wei)
xbow3 = wei @ x

print(f"xbow and xbow3 match? {torch.allclose(xbow, xbow3)}")

# print("x[0]")
# print(x[0])

# print("xbow[0]")
# print(xbow[0])


# # we can simplify this operation with matrices!
# torch.manual_seed(42)
# # a = torch.ones(3,3) # this is the basic version
# a = torch.tril(torch.ones(3,3)) # lower triangle! This lets us pluck each time slice
# a = a / torch.sum(a, 1, keepdim=True) # normalize so all rows add to one
# b = torch.randint(0,10,(3,2)).float()
# c = a @ b
# print('a=')
# print(a)
# print("--")
# print("b=")
# print(b)
# print("--")
# print("c=")
# print(c)

# version 4 self attention!
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channel
x = torch.randn(B,T,C) # fill it with rand to begin

# let's see a single HEAD perform self attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x) # (B, T, 16)
q = query(x) # (B, T, 16)



wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) -----> (B, T, T) 



tril = torch.tril(torch.ones(T,T))
#wei = torch.zeros((T,T))
# removing this next line will allow everything to look at everything (good for sentiment analysis)
wei = wei.masked_fill(tril==0, float('-inf')) 
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v

# out = wei @ x

print(out.shape)
print(wei[0]) 
print(k.var())

print("fun here")

testTensor = torch.tensor([0.1, -.2, .3, -.2, .5])
print(torch.softmax(testTensor, dim=-1))
print(torch.softmax(testTensor*8, dim=-1))

