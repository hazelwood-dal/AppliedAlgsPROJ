# Python3 Program for Bad Character Heuristic
# of Boyer Moore String Matching Algorithm
# Adapted by Dalton Hazelwood
# Originally from GeeksforGeeks (https://www.geeksforgeeks.org/boyer-moore-algorithm-for-pattern-searching/)

NO_OF_CHARS = 256


def bm_search(txt, pat):
    m = len(pat)
    n = len(txt)
    if m == 0:
        return 0
    badChar = {}
    for i in range(m):
        badChar[pat[i]] = i
    s = 0
    while s <= n - m:
        j = m - 1
        while j >= 0 and pat[j] == txt[s + j]:
            j -= 1
        if j < 0:
            return s
        next_char = txt[s + j] if s + j < n else None
        shift = j - badChar.get(next_char, -1) if next_char else 1
        s += max(1, shift)
    return -1


def main():
    txt = "ABAAABCD"
    pat = "ABC"
    print(f"Pattern occurs at index: {bm_search(txt, pat)}")


if __name__ == '__main__':
    main()
