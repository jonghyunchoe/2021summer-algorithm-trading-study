import numpy as np 
import utils 

class Agent:
    STATE_DIM = 2 

    TRADING_CHARGE = 0.0015

    TRADING_TAX = 0.0025 
    
    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2

    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NEW_ACTIONS = len(ACTIONS)

    def __init__(self, environment, min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=.05):
        self.environment = environment 

        self.min_trading_unit = min_trading_unit 
        self.max_trading_unit = max_trading_unit 

        self.delayed_reward_threshold = delayed_reward_threshold 

        self.initial_balance = 0
        self.balance = 0
        self.num_stocks = 0
        self.portfolio_value = 0
        self.base_portfolio_value = 0
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.profitloss = 0
        self.base_profitloss = 0
        self.exploration_base = 0

        self.ratio_hold = 0
        self.ratio_portfolio_value = 0
    
    def reset(self):
        self.balance = self.initial_balance 
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance 
        self.base_portfolio_value = self.initial_balance 
        self.num_buy = 0 
        self.num_sell = 0
        self.ratio_hold = 0 
        self.ratio_portfolio_value = 0
    
    def reset_exploration(self):
        self.exploration_base = 0.5 + np.random.rand() / 2
    
    def set_balance(self, balance):
        self.inital_balance = balance 

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price()
        )
        self.ratio_portfolio_value = (
            self.portfolio_value / self.base_portfolio_value 
        )
        return (
            self.ratio_hold, 
            self.ratio_portfolio_value 
        )
    
    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0

        pred = pred_policy 
        if pred is None:
            pred = pred_value 
        
        if pred is None:
            epsilon = 1
        else:
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1
        
        if np.random.rand() < epsilon:
            exploration = True 
            if np.random.rand() < self.exploration_base:
                action = self.ACTION_BUY 
            else:
                action = np.random.randint(self.NUM_ACTIONS - 1) + 1 
        else:
            exploration = False 
            action = np.argmax(pred) 
        confidence = .5 
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])
        
        return action, confidence, exploration 
    
    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False 
        elif action == Agent.ACTION_SELL:
            if self.num_stocks <= 0:
                return False 
        return True 
    
    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit 
        added_trading = max(min(int(confidence * (self.max_trading_unit - self.min_trading_unit)), self.max_trading_unit - self.min_trading_unit), 0)
        return self.min_trading_unit + added_trading 
    
    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD 
        
        curr_price = self.environment.get_price()

        self.immediate_reward = 0
    
        if action == Agent.ACTION_BUY:
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                self.balance - curr_price * (1 + self.TRADING_CHARGE) \
                    * trading_unit
            )

            if balance < 0:
                trading_unit = max(
                    min(
                        int(self.balance / (
                            curr_price * (1 + self.TRADING_CHARGE)
                        )), self.max_trading_unit
                    ),
                    self.min_trading_unit 
                )
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) \
                * trading_unit    
            if invest_amount > 0:
                self.balance -= invest_amount 
                self.num_stocks += trading_unit 
                self.num_buy += 1
        
        elif action == Agent.ACTION_SELL:
            trading_unit = self.decide_trading_unit(confidence)
            trading_unit = min(trading_unit, self.num_stocks)
            invest_amount = curr_price * (
                1- (self.TRADING_TAX + self.TRADING_CHARGE)) \
                    * trading_unit 
            if invest_amount > 0:
                self.num_stocks -= trading_unit 
                self.balance += invest_amount 
                self.num_sell += 1
        
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1
        
        self.portfolio_value = self.balance + curr_price * self.num_stocks 
        self.profitloss = ((self.portfolio_value - self.initial_blaance) / self.initial_balance)
        
