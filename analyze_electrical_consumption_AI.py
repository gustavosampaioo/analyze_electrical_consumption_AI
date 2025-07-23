import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, confusion_matrix, 
                           classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração inicial
st.set_page_config(page_title="Análise de Consumo de Energia", layout="wide")
st.title("🔍 Análise de Consumo de Energia com Detecção de Fraude")

# Função para encontrar arquivo no Desktop
def find_csv_file():
    desktop_paths = [
        Path.home() / "Desktop",
        Path.home() / "Área de Trabalho",
        Path.home() / "Escritorio",
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

# Função para criar modelo neural (usando scikit-learn)
def create_nn_model():
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2
    )
    return model

# Função principal
def main():
    st.sidebar.header("Configurações de Arquivo")
    
    # Opção de upload
    uploaded_file = st.sidebar.file_uploader(
        "Carregar arquivo CSV", 
        type=["csv"],
        help="Selecione o arquivo dados_consumo.csv"
    )
    
    # Carrega os dados
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
        
        # Pré-processamento avançado
        features = df.iloc[:, 1:25]  # Colunas h1-h24
        target = df['status_fraude']
        
        # Normalização dos dados
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Divisão treino-validação-teste
        X_train, X_temp, y_train, y_temp = train_test_split(
            features_scaled, target, test_size=0.4, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Treinar modelos
        st.subheader("🤖 Modelos de Detecção")
        
        # Modelo Random Forest
        st.write("#### Random Forest")
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        # Modelo Neural Network (MLP)
        st.write("#### Rede Neural (MLP)")
        nn_model = create_nn_model()
        nn_model.fit(X_train, y_train)
        y_pred_nn = nn_model.predict(X_test)
        
        # Avaliação dos modelos
        st.subheader("📈 Métricas de Avaliação")
        
        # Métricas RF
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        precision_rf = precision_score(y_test, y_pred_rf)
        recall_rf = recall_score(y_test, y_pred_rf)
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        
        # Métricas NN
        accuracy_nn = accuracy_score(y_test, y_pred_nn)
        precision_nn = precision_score(y_test, y_pred_nn)
        recall_nn = recall_score(y_test, y_pred_nn)
        cm_nn = confusion_matrix(y_test, y_pred_nn)
        
        # Exibição comparativa
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Random Forest**")
            st.metric("Acurácia", f"{accuracy_rf:.2%}")
            st.metric("Precisão", f"{precision_rf:.2%}")
            st.metric("Recall", f"{recall_rf:.2%}")
            
            st.write("**Matriz de Confusão**")
            fig, ax = plt.subplots()
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'Fraude'],
                        yticklabels=['Normal', 'Fraude'])
            plt.ylabel('Verdadeiro')
            plt.xlabel('Predito')
            st.pyplot(fig)
        
        with col2:
            st.write("**Rede Neural (MLP)**")
            st.metric("Acurácia", f"{accuracy_nn:.2%}")
            st.metric("Precisão", f"{precision_nn:.2%}")
            st.metric("Recall", f"{recall_nn:.2%}")
            
            st.write("**Matriz de Confusão**")
            fig, ax = plt.subplots()
            sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens',
                        xticklabels=['Normal', 'Fraude'],
                        yticklabels=['Normal', 'Fraude'])
            plt.ylabel('Verdadeiro')
            plt.xlabel('Predito')
            st.pyplot(fig)
        
        # Seção de interpretação
        st.subheader("🔍 Guia de Interpretação")
        with st.expander("Como interpretar essas métricas?"):
            st.markdown("""
            **Acurácia** (Accuracy):  
            > Porcentagem total de previsões corretas. Útil para conjuntos balanceados.

            **Precisão** (Precision):  
            > Dos alertas de fraude emitidos, quantos eram realmente fraudes.

            **Recall** (Sensibilidade):  
            > Das fraudes reais existentes, quantas foram detectadas.

            **Matriz de Confusão**:
            - **TP** (True Positive): Fraudes detectadas corretamente
            - **FP** (False Positive): Consumos normais classificados como fraude
            - **TN** (True Negative): Consumos normais corretamente identificados
            - **FN** (False Negative): Fraudes não detectadas
            """)

if __name__ == "__main__":
    main()
