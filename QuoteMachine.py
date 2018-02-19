import time
import pandas as pd
import gdax
client = gdax.PublicClient()
a = client.get_product_historic_rates('BTC-USD', granularity=60*60*24)
b = pd.DataFrame(a)
b.columns = [ 'time', 'low', 'high', 'open', 'close', 'volume' ] 

print(b)
print time.strftime('%Y-%m-%d',time.gmtime(1481155200) )
