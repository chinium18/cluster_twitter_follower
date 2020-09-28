import pandas as pd

columns = ['aa', 'bb', 'cc', 'dd']
a = pd.DataFrame([], columns=columns)

a.loc['aa','aa'] = 1 
print(a)
