import sys
sys.path.append('.')
from v2.core.arbitrary_number import ArbitraryNumber

pi = ArbitraryNumber.pi(50)
print(f"Pi result: {pi}")
print(f"String contains 3.14159: {'3.14159' in str(pi)}")
print(f"Pi string representation: '{str(pi)}'")
