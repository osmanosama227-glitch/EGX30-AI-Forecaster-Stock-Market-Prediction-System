EGX30 AI Forecaster 📈

A modern financial forecasting dashboard for Egyptian stock market companies listed in the EGX30 index using Monte Carlo Simulation and Geometric Brownian Motion (GBM).

Built with:

Python
Streamlit
Plotly
NumPy
Pandas
yFinance
🚀 Features
📊 Real-time EGX30 stock data
📈 Interactive historical price charts
🎲 Monte Carlo stock price simulation
🧠 Geometric Brownian Motion (GBM) forecasting
📅 12-month future prediction
📉 Confidence interval visualization
⚡ Modern dark finance dashboard UI
📋 Forecast tables & statistics
🔍 Automatic working ticker filtering
📊 Volatility & return analysis
🧮 Mathematical Model

The project uses the Geometric Brownian Motion model:

S
t+1
	​

=S
t
	​

e
(μ−
2
σ
2
	​

)+σZ
t
	​


Where:

S
t
	​

 = current stock price
μ = expected return
σ = volatility
Z
t
	​

 = random normal shock

The system simulates thousands of possible future stock price paths using Monte Carlo simulation.

🏗️ Technologies Used
Backend
Python
Web Framework
Streamlit
Data Analysis
NumPy
Pandas
Visualization
Plotly
Financial Data
yFinance API
📂 Project Structure
EGX30-AI-Forecaster/
│
├── egx30_forecaster.py
├── requirements.txt
├── README.md
└── screenshots/
⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/EGX30-AI-Forecaster.git

Move into the project folder:

cd EGX30-AI-Forecaster

Install dependencies:

pip install -r requirements.txt
▶️ Run the Application
streamlit run egx30_forecaster.py
📊 Dashboard Includes
Historical stock prices
Future forecast chart
Confidence intervals
Forecast tables
Statistical metrics
Return distribution analysis
📈 Forecasting Process
Download historical stock data
Calculate logarithmic returns
Estimate drift and volatility
Generate Monte Carlo simulations
Forecast future stock prices
Visualize predictions interactively
📸 Screenshots

Add your screenshots here:

screenshots/dashboard.png
screenshots/forecast.png
screenshots/statistics.png
🎓 Academic Purpose

This project was developed as part of a Stochastic Processes / Financial Forecasting academic project.

It demonstrates:

stochastic modeling
financial mathematics
probabilistic forecasting
quantitative finance concepts
🔮 Future Improvements
GARCH volatility modeling
ARIMA forecasting
Prophet integration
Ensemble forecasting
AI sentiment analysis
Real-time streaming prices
Portfolio optimization
Candlestick charts
👨‍💻 Author

Osman Osama
Computer Science Student
Egyptian Chinese University

⚠️ Disclaimer

This project is for educational and research purposes only.

Financial forecasts are probabilistic estimates based on historical data and should not be considered investment advice.
