CREATE TABLE IF NOT EXISTS paper_trades (
    id SERIAL PRIMARY KEY,
    trade_id INTEGER,
    timestamp TIMESTAMPTZ,
    direction TEXT,
    entry_price DECIMAL,
    stop_loss DECIMAL,
    take_profit DECIMAL,
    position_size DECIMAL,
    risk_amount DECIMAL,
    confidence INTEGER,
    reasoning TEXT,
    status TEXT,
    exit_price DECIMAL,
    exit_time TIMESTAMPTZ,
    pnl DECIMAL,
    close_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
