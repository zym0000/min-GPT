import torch
from model import BigramLanguageModel, block_size, batch_size, device
from data import train_data, val_data, decode  # ← 从 data.py 导入

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y

torch.manual_seed(1337)
m = BigramLanguageModel().to(device)

total_params = sum(p.numel() for p in m.parameters())
print(f"模型总参数量: {total_params:,}")
print(f"使用设备: {device}")

optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)

print("开始训练...")

for step in range(5000):
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)
    
    logits, loss = m(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if step % 500 == 0:
        print(f"step {step:5d} | loss: {loss.item():.4f}")

# ============ 生成文本 ============
print("\n训练完成，生成文本：")
print("=" * 50)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = m.generate(context, max_new_tokens=1000)
print(decode(generated[0].tolist()))