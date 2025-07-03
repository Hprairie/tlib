import torch
import triton
import triton.language as tl
import triton_lib as tlib



@triton.jit
def add(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < tl.num_programs(axis=0) * BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # print("Hello", x.shape)
    import pdb; pdb.set_trace()
    x = x[None, :]
    x = tlib.ops.rearrange("h j -> j h", x)
    print("Mid", x.shape)
    tlib.ops.rearrange("h j -> (h j)", x)
    print("Done", x.shape)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add_vectors(x: torch.Tensor, y: torch.Tensor):
    BLOCK_SIZE = 32
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    add[grid](x, y, output, BLOCK_SIZE=BLOCK_SIZE)
    return output

if __name__ == '__main__':
    x = torch.rand(size=(1024,), device='cuda', dtype=torch.float32)
    y = torch.rand(size=(1024,), device='cuda', dtype=torch.float32)
    output = add_vectors(x, y)
    print(output)