a = "8 6 78 78 81 13 17 44 30 67 1 74 100 6 36 47 98 124 50 119 100 115 39 3 55 51 3 41 48 111 41 56 113 89 65 122 41 9 63 115 45 127 118 35 54 117 80 77 74 99 59 57 104 62 19 1 91 81 69 73 83 109 110 91 98 83 28 91 108 124 67 35 5 36 24 88 123 92 74 101 93 23 12 54 62 13 38 98 59 108 105 102 61 109 65 34 99 101 100 105 8"
a = a.split()
s = []
for i in range(len(a) - 1):
    s.append(int(a[i]) ^ int(a[i + 1]))

print(s)
