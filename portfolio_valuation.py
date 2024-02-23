

class TheoreticalPortfolioValuation:

    def __init__(self, loan_amount: float, spot_price: float, delta: float) -> None:


        self.loan_amount = loan_amount
        self.delta = delta

        self.base_amount = loan_amount * (1-delta)
        self.quote_amount = loan_amount * (delta) * spot_price

        return None
    
    def update(self, spot_price: float, delta: float) -> None:

        self.base_amount -= (self.loan_amount) * (delta - self.delta)
        self.quote_amount += (self.loan_amount) * (delta - self.delta) * spot_price
        self.delta = delta

    def balance_in_quote(self, spot_price: float) -> None:

        return self.quote_amount + self.base_amount * spot_price
