def build_shift_table(pattern):
    m = len(pattern)
    shift = {c: m for c in set(pattern)}
    for i in range(m - 1):
        shift[pattern[i]] = m - 1 - i
    return shift


def horspool_search(text, pattern):
    m = len(pattern)
    n = len(text)
    shift = build_shift_table(pattern)
    i = m - 1
    while i < n:
        k = 0
        while k < m and pattern[m - 1 - k] == text[i - k]:
            k += 1
        if k == m:
            return i - m + 1  # Match found
        else:
            i += shift.get(text[i], m)
    return -1  # No match
