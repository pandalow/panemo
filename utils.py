
from torch import Tensor

def find_subsequence(sequence, subseq, tokenizer):
    """
    在 sequence 中查找子序列 subseq，若找到则返回其起始位置，否则 -1。
    通过 decode 校验，防止分词器特殊字符导致的误匹配。
    :param sequence: List[int]，输入文本的 token id 列表
    :param subseq: List[int]，标签的 token id 列表
    :param tokenizer: 分词器，用于 decode 验证
    :return: int (起始位置) or -1
    """
    n = len(sequence)
    m = len(subseq)
    if m == 0 or n < m:
        return -1

    target_str = tokenizer.decode(subseq)  # 标签的真实字符串

    for start in range(n - m + 1):
        # 先看是否原始 id 相同
        if sequence[start:start + m] == subseq:
            # 再 decode 校验
            decoded = tokenizer.decode(sequence[start:start + m])
            if decoded == target_str:
                return start
    return -1



def pool_subwords(hidden_states: Tensor, start: int, length: int, pool_type: str="mean") -> Tensor:
    """
    对子词区域做 mean / max 池化:
    :param hidden_states: [seq_len, hidden_size]
    :param start: int, 子词起始位置
    :param length: int, 子词序列长度
    :param pool_type: "mean" or "max"
    :return: [hidden_size]
    """
    if start < 0 or length <= 0:
        # 返回一个全零向量，表示该标签在文本中没有出现
        return torch.zeros(hidden_states.size(-1), device=hidden_states.device)

    sub = hidden_states[start:start+length]  # [length, hidden_size]
    if pool_type == "mean":
        return sub.mean(dim=0)
    elif pool_type == "max":
        return sub.max(dim=0).values
    else:
        raise ValueError("Unknown pool_type")