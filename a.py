# import numpy as np

# a = input()
# d =[[i, a.count(i)] for i in a]
# for j, i in enumerate(d):
#     if i[1] == 1:
#         if(bin(j)[-2] == "1"):
#             print("yes")
#         else:
#             print("no")
#         break


# a = int(input())
# b = input().split()
# b = [int(i) for i in b]

# w = [-1 for i in range(a)]
# for i in range(a):
#     c = 0
#     x = b[i]
#     for j in range(i + 1, a):
#         if b[j] > x and c == 0:
#             x = b[j]
#             c += 1
#         elif b[j] > x and c == 1:
#             w[i] = b[j]
#             break


# print(w)


def find_numbers_with_xor(n):
    if n == 0:
        return (0, 0)

    msb_position = 0
    temp = n
    while temp:
        msb_position += 1
        temp >>= 1

    a = 1 << (msb_position - 1)
    b = a ^ n

    return (a, b)


s = [
    14,
    72,
    0,
    31,
    92,
    28,
    61,
    50,
    93,
    66,
    75,
    46,
    98,
    34,
    11,
    77,
    30,
    78,
    69,
    19,
    23,
    84,
    36,
    52,
    4,
    48,
    42,
    25,
    95,
    70,
    17,
    73,
    40,
    24,
    59,
    83,
    32,
    54,
    76,
    94,
    82,
    9,
    85,
    21,
    67,
    37,
    29,
    7,
    41,
    88,
    2,
    81,
    86,
    45,
    18,
    90,
    10,
    20,
    12,
    26,
    62,
    3,
    53,
    57,
    49,
    79,
    71,
    55,
    16,
    63,
    96,
    38,
    33,
    60,
    64,
    35,
    39,
    22,
    47,
    56,
    74,
    27,
    58,
    8,
    51,
    43,
    68,
    89,
    87,
    5,
    15,
    91,
    80,
    44,
    99,
    65,
    6,
    1,
    13,
    97,
]

n = len(s)
result = find_numbers_with_xor(s[0])

option1 = [result[0], result[1]]
option2 = [result[1], result[0]]
for i in range(1, n):
    option1.append(option1[i] ^ s[i])
    option2.append(option2[i] ^ s[i])

x = sorted(option1)
y = sorted(option2)
for i in range(n + 1):
    if x[i] != i + 1:
        for j in option2:
            print(j, end=" ")
        break
    elif y[i] != i + 1:
        for j in option1:
            print(j, end=" ")
        break
