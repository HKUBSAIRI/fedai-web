import glob, time, torch, tiktoken
from ChronoGPT_fp32 import ChronoGPT, extract_response
torch.set_num_threads(4)
def load(year):
    d = glob.glob(f'hf/hub/models--manelalab--chrono-gpt-instruct-v1-{year}1231/snapshots/*/')[0]
    cfg = torch.load(d + 'config.pt', map_location='cpu')
    m = ChronoGPT(**cfg); m.load_state_dict(torch.load(d + 'pytorch_model.bin', map_location='cpu')); m = m.float().eval()
    return m
tok = tiktoken.get_encoding('gpt2')
