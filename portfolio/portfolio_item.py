
#портфель: bank total_profit day_profit securities['name':count,'buy':n 'cur_price':n 'profit':n% ]
#над ним совершаются сделки и они детально выводятся 

# эта часть не сделана 
class Portfolio:
    def __init__(self, initial_bank=0):
        self.bank = initial_bank
        self.total_profit = 0
        self.day_profit = 0
        self.securities = {}
        self.transactions = []

    def add_funds(self, amount):
        self.bank += amount
        self.transactions.append(f"Added funds: {amount} | New bank balance: {self.bank}")

    def buy_security(self, timestamp,name, count, buy_price):
        total_cost = count * buy_price
        if total_cost > self.bank:
            print("Not enough funds to buy the security.")
            return
        self.bank -= total_cost
        if name in self.securities:
            self.securities[name]['count'] += count
        else:
            self.securities[name] = {'count': count, 'buy_price': buy_price, 'cur_price': buy_price, 'profit': 0}
        self.transactions.append(f"Bought {timestamp} {count} of {name} at {buy_price} each | New bank balance: {self.bank}")

    def sell_security(self,timestamp, name, count, sell_price):
        if name not in self.securities or self.securities[name]['count'] < count:
            print("Not enough securities to sell.")
            return
        self.securities[name]['count'] -= count
        revenue = count * sell_price
        self.bank += revenue
        profit = revenue - (count * self.securities[name]['buy_price'])
        self.total_profit += profit
        self.day_profit += profit
        if self.securities[name]['count'] == 0:
            del self.securities[name]
        self.transactions.append(f"Sold {timestamp}, {count} of {name} at {sell_price} each | New bank balance: {self.bank} | Profit: {profit}")

    def update_prices(self, name, new_price):
        if name in self.securities:
            self.securities[name]['cur_price'] = new_price
            buy_price = self.securities[name]['buy_price']
            self.securities[name]['profit'] = ((new_price - buy_price) / buy_price) * 100

    def print_portfolio(self):
        print(f"Bank balance: {self.bank}")
        print(f"Total profit: {self.total_profit}")
        print(f"Day profit: {self.day_profit}")
        print("Securities:")
        for name, data in self.securities.items():
            print(f"{name}: Count: {data['count']}, Buy Price: {data['buy_price']}, Current Price: {data['cur_price']}, Profit: {data['profit']}%")
        print("Transactions:")
        for transaction in self.transactions:
            print(transaction)


