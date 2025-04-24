import sqlite3
import requests
from bs4 import BeautifulSoup
import os

os.remove('symbols_bist.db')
conn = sqlite3.connect('symbols_bist.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS stocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    title TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

url = "https://tr.tradingview.com/markets/stocks-turkey/market-movers-active/"
headers = {'User-Agent': 'Mozilla/5.0'}

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    stock_links = soup.find_all('a', class_='tickerName-GrtoTeat')
    
    for stock in stock_links:
        symbols = stock.text.strip()
        title = stock.get('title', '')
        cursor.execute('''
        INSERT OR REPLACE INTO stocks (symbol, title)
        VALUES (?, ?)
        ''', (symbols, title))
    
    conn.commit()
    conn.close()
    print("Stocks updated successfully.")

except Exception as e:
    print(e)

