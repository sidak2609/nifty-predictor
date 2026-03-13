NIFTY50_SYMBOLS = {
    "^NSEI":         "Nifty 50 Index",
    "ADANIENT.NS":   "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "ASIANPAINT.NS": "Asian Paints",
    "AXISBANK.NS":   "Axis Bank",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "BEL.NS":        "Bharat Electronics",
    "BHARTIARTL.NS": "Bharti Airtel",
    "BRITANNIA.NS":  "Britannia",
    "CIPLA.NS":      "Cipla",
    "COALINDIA.NS":  "Coal India",
    "DIVISLAB.NS":   "Divi's Laboratories",
    "DRREDDY.NS":    "Dr. Reddy's",
    "EICHERMOT.NS":  "Eicher Motors",
    "GRASIM.NS":     "Grasim Industries",
    "HCLTECH.NS":    "HCL Technologies",
    "HDFCBANK.NS":   "HDFC Bank",
    "HDFCLIFE.NS":   "HDFC Life",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "HINDALCO.NS":   "Hindalco",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ICICIBANK.NS":  "ICICI Bank",
    "INDUSINDBK.NS": "IndusInd Bank",
    "INFY.NS":       "Infosys",
    "ITC.NS":        "ITC",
    "JSWSTEEL.NS":   "JSW Steel",
    "KOTAKBANK.NS":  "Kotak Mahindra Bank",
    "LT.NS":         "Larsen & Toubro",
    "M&M.NS":        "Mahindra & Mahindra",
    "MARUTI.NS":     "Maruti Suzuki",
    "NESTLEIND.NS":  "Nestle India",
    "NTPC.NS":       "NTPC",
    "ONGC.NS":       "ONGC",
    "POWERGRID.NS":  "Power Grid",
    "RELIANCE.NS":   "Reliance Industries",
    "SBIN.NS":       "State Bank of India",
    "SBILIFE.NS":    "SBI Life Insurance",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "SUNPHARMA.NS":  "Sun Pharma",
    "TATACONSUM.NS": "Tata Consumer",
    "TATASTEEL.NS":  "Tata Steel",
    "TCS.NS":        "TCS",
    "TECHM.NS":      "Tech Mahindra",
    "TITAN.NS":      "Titan",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "WIPRO.NS":      "Wipro",
}

NIFTY_INDEX = "^NSEI"

# NSE market hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MIN = 30

# Model config
LOOKBACK_PERIODS = 20       # candles of history used as features
PREDICTION_HORIZON = 1      # predict 1 candle ahead (10 min)
MIN_CONFIDENCE = 0.60       # only show predictions above this confidence
HIGH_CONFIDENCE = 0.75      # show as "high confidence"
