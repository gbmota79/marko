import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime

st.set_page_config(layout="wide")

st.title("🌎 Otimizador de Carteiras com ETFs da B3")

# === 1. SUGESTÃO DE RETORNOS ESPERADOS (em %) ===
retornos_sugeridos = {
    'LFTS11.SA': 14.25,
    'IRFM11.SA': 14.75,
    'IMAB11.SA': 17.00,
    'DEBB11.SA': 15.00,
    'GOLD11.SA': 12.00,
    'DIVO11.SA': 18.00,
    'BOVA11.SA': 20.00,
    'IVVB11.SA': 16.00,
    'HASH11.SA': 40.00
}

# === 2. ENTRADA INTERATIVA DE RETORNOS ===
df_retorno = pd.DataFrame(retornos_sugeridos.values(),
                          index=retornos_sugeridos.keys(),
                          columns=["Retorno Esperado (%)"])

st.subheader("1. Ajuste os retornos esperados (em %):")
retornos_editados = st.data_editor(df_retorno, use_container_width=True)

# Converte para vetor de retornos esperados anualizados (em decimal)
mu = (retornos_editados["Retorno Esperado (%)"].values / 100).astype(float)
tickers = list(retornos_sugeridos.keys())

# === 3. COLETA DE DADOS HISTÓRICOS ===
st.subheader("2. Coletando dados históricos dos ETFs...")
start_date = '2018-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
data = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
data = data.dropna()
returns = np.log(data / data.shift(1)).dropna()

# === 4. ESTATÍSTICAS ===
returns_filtered = returns[tickers]
cov = returns_filtered.cov() * 252
vol = returns_filtered.std() * np.sqrt(252)
corr = returns_filtered.corr()

# === 5. MOSTRA ESTATÍSTICAS BASE ===
st.subheader("3. Estatísticas Históricas dos Ativos")
df_stats = pd.DataFrame({
    "Retorno Esperado (%)": (mu * 100).round(2),
    "Volatilidade Histórica (%)": (vol * 100).round(2)
}, index=tickers)
st.dataframe(df_stats, use_container_width=True)

st.write("\n**Matriz de Correlação (mapa de calor):**")
fig_corr, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0, linewidths=0.5, ax=ax)
plt.title("Matriz de Correlação entre ETFs")
st.pyplot(fig_corr)

# === 6. OTIMIZAÇÃO ===
def get_max_return_for_vol(target_vol):
    n = len(tickers)
    init_w = np.ones(n) / n
    bounds = [(0, 1) for _ in range(n)]
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: np.sqrt(np.dot(w.T, np.dot(cov, w))) - target_vol}
    ]
    result = minimize(lambda w: -np.dot(w, mu), init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else None

def get_max_sharpe_portfolio():
    n = len(tickers)
    init_w = np.ones(n) / n
    bounds = [(0, 1) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    result = minimize(lambda w: -np.dot(w, mu) / np.sqrt(np.dot(w.T, np.dot(cov, w))), init_w,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def compute_efficient_frontier(points=100):
    min_vol = 0.01
    max_vol = np.sqrt(np.dot(np.ones(len(mu)) / len(mu), np.dot(cov, np.ones(len(mu)) / len(mu)))) * 2.5
    vol_grid = np.linspace(min_vol, max_vol, points)
    frontier = []
    for target_vol in vol_grid:
        weights = get_max_return_for_vol(target_vol)
        if weights is not None:
            port_return = np.dot(weights, mu)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            frontier.append((port_vol, port_return, weights))
    return frontier

# === 7. SLIDER DE VOLATILIDADE ===
st.subheader("4. Defina a volatilidade-alvo para a sua carteira")
vol_target = st.slider("Volatilidade alvo (%)", min_value=1.0, max_value=25.0, value=5.0, step=0.5) / 100

# === 8. CÁLCULOS DE CARTEIRAS ===
frontier = compute_efficient_frontier()
target_weights = get_max_return_for_vol(vol_target)
max_sharpe_weights = get_max_sharpe_portfolio()

target_ret = np.dot(target_weights, mu)
target_vol = np.sqrt(np.dot(target_weights.T, np.dot(cov, target_weights)))
sharpe_ret = np.dot(max_sharpe_weights, mu)
sharpe_vol = np.sqrt(np.dot(max_sharpe_weights.T, np.dot(cov, max_sharpe_weights)))

# === 9. GRÁFICO ===
st.subheader("5. Fronteira Eficiente")
vols, rets, _ = zip(*frontier)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(vols, rets, label='Fronteira Eficiente', color='black')
ax.scatter(target_vol, target_ret, color='blue', marker='X', s=150, label='Carteira Vol-alvo')
ax.scatter(sharpe_vol, sharpe_ret, color='red', marker='*', s=200, label='Carteira Sharpe Máximo')
ax.set_xlabel("Volatilidade Anual")
ax.set_ylabel("Retorno Esperado Anual")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# === 10. TABELAS COM PESOS ===
st.subheader("6. Carteiras Otimizadas")
st.markdown("**🌍 Carteira com Volatilidade-Alvo:**")
df_target = pd.DataFrame(target_weights, index=tickers, columns=["Peso (%)"]).round(4) * 100
df_target.loc["Retorno Esperado da Carteira"] = [target_ret * 100]
st.dataframe(df_target)

st.markdown("**🌟 Carteira com Maior Sharpe:**")
df_sharpe = pd.DataFrame(max_sharpe_weights, index=tickers, columns=["Peso (%)"]).round(4) * 100
df_sharpe.loc["Retorno Esperado da Carteira"] = [sharpe_ret * 100]
st.dataframe(df_sharpe)
```

Este é o esqueleto completo do app no Streamlit. Ele está pronto para rodar como `app.py` ou direto com:

```bash
streamlit run app.py
```

Se quiser, posso te guiar nos próximos passos para implantar, customizar layout ou exportar os resultados. Vamos nessa?
