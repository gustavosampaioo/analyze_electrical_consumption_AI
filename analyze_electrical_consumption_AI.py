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
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# Configuração inicial
st.set_page_config(page_title="Análise de Consumo de Energia", layout="wide")
st.title("🔍 Análise de Consumo de Energia com Detecção de Fraude")

# Configuração da API do Google Generative AI
generai.configure(api_key="AIzaSyBHouRPqa8LLjU96nEPk6UJBgswH66OJjY")  # Substitua pela sua chave API

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

# Função para criar modelo neural
def create_nn_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])
    return model

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
        
        # Modelo Neural Network
        st.write("#### Rede Neural")
        nn_model = create_nn_model(X_train.shape[1])
        history = nn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            verbose=0
        )
        
        y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
        
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
            st.write("**Rede Neural**")
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
        
        # Curvas de aprendizado
        st.subheader("📚 Curvas de Aprendizado (Rede Neural)")
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        ax[0].plot(history.history['accuracy'], label='Treino')
        ax[0].plot(history.history['val_accuracy'], label='Validação')
        ax[0].set_title('Acurácia')
        ax[0].legend()
        
        ax[1].plot(history.history['loss'], label='Treino')
        ax[1].plot(history.history['val_loss'], label='Validação')
        ax[1].set_title('Loss')
        ax[1].legend()
        
        st.pyplot(fig)
        
        # Análise com IA generativa
        if st.button("🧠 Obter Análise Avançada com Gemini"):
            with st.spinner("Analisando dados com Gemini 1.5 Flash..."):
                try:
                    data_summary = df.describe().to_string()
                    
                    metrics = f"""
                    **Random Forest:**
                    - Acurácia: {accuracy_rf:.2%}
                    - Precisão: {precision_rf:.2%}
                    - Recall: {recall_rf:.2%}
                    
                    **Rede Neural:**
                    - Acurácia: {accuracy_nn:.2%}
                    - Precisão: {precision_nn:.2%}
                    - Recall: {recall_nn:.2%}
                    """
                    
                    analysis = analyze_with_ai(data_summary, metrics)
                    
                    st.subheader("📝 Análise com Gemini 1.5 Flash")
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"Erro na análise com Gemini: {str(e)}")

if __name__ == "__main__":
    main()
