import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, confusion_matrix, 
                            classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as generai
from pathlib import Path

# Configuração inicial
st.set_page_config(page_title="Análise de Consumo de Energia", layout="wide")
st.title("🔍 Análise de Consumo de Energia com Detecção de Fraude")

# Configuração da API do Google Generative AI
generai.configure(api_key="AIzaSyBHouRPqa8LLjU96nEPk6UJBgswH66OJjY")  # Substitua pela sua chave API

# Função para encontrar arquivo no Desktop
def find_csv_file():
    # Caminhos comuns para o Desktop em diferentes sistemas operacionais
    desktop_paths = [
        Path.home() / "Desktop",
        Path.home() / "Área de Trabalho",  # Para sistemas em português
        Path.home() / "Escritorio",        # Para sistemas em espanhol
    ]
    
    for desktop in desktop_paths:
        if desktop.exists():
            for file in desktop.glob("dados_consumo*.csv"):
                return str(file)
    return None

# Função para carregar dados
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df
            return df
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {str(e)}")
            return None
    
    # Tenta encontrar automaticamente no Desktop
    auto_file = find_csv_file()
    if auto_file:
        try:
            df = pd.read_csv(auto_file)
            st.session_state['df'] = df
            st.success(f"Arquivo encontrado automaticamente: {auto_file}")
            return df
        except Exception as e:
            st.error(f"Erro ao ler arquivo automático: {str(e)}")
    
    st.warning("Nenhum arquivo encontrado. Por favor, faça upload do arquivo.")
    return None

# Função para análise com IA generativa
def analyze_with_ai(data_summary, metrics):
    model = generai.GenerativeModel("gemini-1.5-flash-latest")
    
    prompt = f"""
    Você é um especialista em análise de dados de consumo de energia e detecção de fraudes. 
    Analise os seguintes dados e métricas:

    **Resumo dos dados:**
    {data_summary}

    **Métricas do modelo:**
    {metrics}

    Sua análise deve conter:
    1. Padrões de consumo de energia identificados
    2. Interpretação profissional das métricas de avaliação
    3. Análise da matriz de confusão
    4. Recomendações para melhorar a detecção
    5. Períodos de maior risco de fraude
    
    Formate a resposta em markdown com títulos e bullet points.
    """
    
    response = model.generate_content(prompt)
    return response.text

# Função principal
def main():
    st.sidebar.header("Configurações de Arquivo")
    
    # Opção 1: Upload manual
    uploaded_file = st.sidebar.file_uploader(
        "Carregar arquivo CSV", 
        type=["csv"],
        help="Selecione o arquivo dados_consumo.csv"
    )
    
    # Opção 2: Seleção automática do Desktop
    if st.sidebar.button("Buscar automaticamente no Desktop"):
        auto_file = find_csv_file()
        if auto_file:
            st.sidebar.success(f"Arquivo encontrado: {auto_file}")
        else:
            st.sidebar.warning("Nenhum arquivo encontrado no Desktop")
    
    # Carrega os dados (do upload ou da sessão)
    df = st.session_state.get('df', None)
    if uploaded_file or not df:
        df = load_data(uploaded_file)
    
    if df is not None:
        st.sidebar.header("Visualização")
        show_raw_data = st.sidebar.checkbox("Mostrar dados brutos")
        
        if show_raw_data:
            st.subheader("Dados de Consumo")
            st.dataframe(df)
        
        # Análise exploratória
        st.subheader("📊 Análise Exploratória")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Consumo Médio Diário**")
            st.line_chart(df.set_index('data')['consumo_medio_diario'])
        
        with col2:
            st.write("**Consumo Mínimo Noturno**")
            st.bar_chart(df.set_index('data')['consumo_minimo_noturno'])
        
        # Pré-processamento
        features = df.iloc[:, 1:25]  # Colunas h1-h24
        target = df['status_fraude']
        
        # Divisão treino-teste
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.3, random_state=42
        )
        
        # Modelagem
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Exibição de métricas
        st.subheader("📈 Métricas de Avaliação do Modelo")
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("Acurácia", f"{accuracy:.2%}")
            st.metric("Precisão", f"{precision:.2%}")
            st.metric("Recall", f"{recall:.2%}")
        
        with metrics_col2:
            st.write("**Matriz de Confusão**")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Normal', 'Fraude'], 
                        yticklabels=['Normal', 'Fraude'])
            plt.ylabel('Verdadeiro')
            plt.xlabel('Predito')
            st.pyplot(fig)
        
        st.write("**Relatório de Classificação**")
        st.text(classification_report(y_test, y_pred))
        
        # Análise com IA generativa
        if st.button("🧠 Obter Análise Avançada com Gemini"):
            with st.spinner("Analisando dados com Gemini 1.5 Flash..."):
                try:
                    # Preparar resumo dos dados
                    data_summary = df.describe().to_string()
                    
                    # Preparar métricas
                    metrics = f"""
                    - Acurácia: {accuracy:.2%}
                    - Precisão: {precision:.2%}
                    - Recall: {recall:.2%}
                    - Matriz de Confusão: \n{cm}
                    - Relatório: \n{classification_report(y_test, y_pred)}
                    """
                    
                    # Chamar a API do Google
                    analysis = analyze_with_ai(data_summary, metrics)
                    
                    st.subheader("📝 Análise com Gemini 1.5 Flash")
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"Erro na análise com Gemini: {str(e)}")
        
        # Seção de interpretação
        st.subheader("🔍 Guia de Interpretação")
        with st.expander("Como interpretar essas métricas?"):
            st.markdown("""
            **Acurácia** (Accuracy):  
            > Porcentagem total de previsões corretas. Útil para conjuntos balanceados, mas pode enganar em casos desbalanceados.

            **Precisão** (Precision):  
            > Dos alertas de fraude emitidos, quantos eram realmente fraudes. Alta precisão significa poucos falsos positivos.

            **Recall** (Sensibilidade):  
            > Das fraudes reais existentes, quantas foram detectadas. Alto recall significa poucos falsos negativos.

            **Matriz de Confusão**:
            - **TP** (True Positive): Fraudes detectadas corretamente
            - **FP** (False Positive): Consumos normais erroneamente classificados como fraude
            - **TN** (True Negative): Consumos normais corretamente identificados
            - **FN** (False Negative): Fraudes que passaram despercebidas

            **Trade-off Importante**:  
            > Aumentar a precisão (reduzir FP) geralmente reduz o recall (aumenta FN), e vice-versa.
            """)

if __name__ == "__main__":
    main()
