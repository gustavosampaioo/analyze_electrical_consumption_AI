import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix, 
                           classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as generai
from fpdf import FPDF
import base64
from datetime import datetime
import tempfile
import os

# Configuração inicial
st.set_page_config(page_title="Análise de Consumo de Energia", layout="wide")
st.title("🔍 Análise de Consumo de Energia com Detecção de Fraude")

# Configuração da API do Google Generative AI
generai.configure(api_key="AIzaSyBHouRPqa8LLjU96nEPk6UJBgswH66OJjY")

# Classe para gerar PDF
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Relatório de Análise de Consumo de Energia', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')
    
    def add_section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1)
        self.ln(5)
    
    def add_content(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, text)
        self.ln()
    
    def add_image(self, image_path, width=190):
        self.image(image_path, x=10, w=width)
        self.ln(5)

# Função para plotar matriz de confusão com legendas
def plot_confusion_matrix_with_labels(cm, title, cmap):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=['Normal (TN/FP)', 'Fraude (FN/TP)'],
                yticklabels=['Normal (TN/FN)', 'Fraude (FP/TP)'])
    
    plt.text(0.5, -0.3, 
             "TN: Consumos normais corretamente identificados\n"
             "FP: Consumos normais classificados como fraude\n"
             "FN: Fraudes não detectadas\n"
             "TP: Fraudes detectadas corretamente",
             ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.title(title)
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.tight_layout()

# Função para plotar métricas comparativas
def plot_metrics_comparison(metrics_rf, metrics_nn):
    labels = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
    rf_values = [metrics_rf['accuracy'], metrics_rf['precision'], 
                 metrics_rf['recall'], metrics_rf['f1']]
    nn_values = [metrics_nn['accuracy'], metrics_nn['precision'], 
                 metrics_nn['recall'], metrics_nn['f1']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, rf_values, width, label='Random Forest', color='skyblue')
    rects2 = ax.bar(x + width/2, nn_values, width, label='Rede Neural', color='lightgreen')
    
    ax.set_ylabel('Pontuação')
    ax.set_title('Comparação de Métricas entre Modelos')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Adicionar valores nas barras
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2%}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

# Função para gerar PDF
def generate_pdf(data_info, metrics, gemini_analysis, interpretation, df, cm_rf, cm_nn, metrics_comparison):
    pdf = PDF()
    pdf.add_page()
    
    # Cabeçalho
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Relatório Completo de Análise', 0, 1, 'C')
    pdf.ln(10)
    
    # Data e hora
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", 0, 1)
    pdf.ln(10)
    
    # Seção 1: Dados Analisados
    pdf.add_section_title("1. Dados Analisados")
    pdf.add_content(data_info)
    
    # Gráficos de consumo
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.figure(figsize=(8, 4))
        plt.plot(df.set_index('data')['consumo_medio_diario'])
        plt.title('Consumo Médio Diário')
        plt.xlabel('Data')
        plt.ylabel('Consumo (kWh)')
        plt.tight_layout()
        plt.savefig(tmpfile.name, dpi=100)
        plt.close()
        pdf.add_image(tmpfile.name)
        os.unlink(tmpfile.name)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.figure(figsize=(8, 4))
        plt.plot(df.set_index('data')['consumo_maximo_diario'])
        plt.title('Consumo Máximo Diário')
        plt.xlabel('Data')
        plt.ylabel('Consumo (kWh)')
        plt.tight_layout()
        plt.savefig(tmpfile.name, dpi=100)
        plt.close()
        pdf.add_image(tmpfile.name)
        os.unlink(tmpfile.name)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.figure(figsize=(8, 4))
        plt.plot(df.set_index('data')['desvio_padrao_diario'])
        plt.title('Variação Diária (Desvio Padrão)')
        plt.xlabel('Data')
        plt.ylabel('Desvio Padrão (kWh)')
        plt.tight_layout()
        plt.savefig(tmpfile.name, dpi=100)
        plt.close()
        pdf.add_image(tmpfile.name)
        os.unlink(tmpfile.name)
    
    # Seção 2: Métricas do Modelo
    pdf.add_section_title("2. Métricas do Modelo")
    pdf.add_content(metrics)
    
    # Gráfico de comparação de métricas
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        metrics_comparison.savefig(tmpfile.name, dpi=100, bbox_inches='tight')
        plt.close()
        pdf.add_image(tmpfile.name)
        os.unlink(tmpfile.name)
    
    # Matrizes de Confusão
    pdf.add_section_title("Matrizes de Confusão")
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plot_confusion_matrix_with_labels(cm_rf, "Matriz de Confusão - Random Forest", 'Blues')
        plt.savefig(tmpfile.name, dpi=100, bbox_inches='tight')
        plt.close()
        pdf.image(tmpfile.name, x=10, w=90)
        os.unlink(tmpfile.name)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plot_confusion_matrix_with_labels(cm_nn, "Matriz de Confusão - Rede Neural", 'Greens')
        plt.savefig(tmpfile.name, dpi=100, bbox_inches='tight')
        plt.close()
        pdf.image(tmpfile.name, x=110, w=90)
        pdf.ln(10)
        os.unlink(tmpfile.name)
    
    # Seção 3: Análise Avançada
    pdf.add_section_title("3. Análise Avançada (Gemini)")
    pdf.add_content(gemini_analysis)
    
    # Seção 4: Interpretação
    pdf.add_section_title("4. Guia de Interpretação")
    pdf.add_content(interpretation)
    
    return pdf

# Função para criar link de download do PDF
def create_download_link(pdf, filename):
    b64 = base64.b64encode(pdf.output(dest='S').encode('latin-1')).decode('latin-1')
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Clique para baixar o relatório PDF</a>'

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
    uploaded_file = st.sidebar.file_uploader("Carregar arquivo CSV", type=["csv"])
    
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
        
        # Gráficos na primeira linha
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Consumo Médio Diário**")
            st.line_chart(df.set_index('data')['consumo_medio_diario'])
        
        with col2:
            st.write("**Consumo Máximo Diário**")
            st.line_chart(df.set_index('data')['consumo_maximo_diario'])
        
        # Gráficos na segunda linha
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("**Consumo Mínimo Noturno**")
            st.bar_chart(df.set_index('data')['consumo_minimo_noturno'])
        
        with col4:
            st.write("**Variação Diária (Desvio Padrão)**")
            st.line_chart(df.set_index('data')['desvio_padrao_diario'])
        
        # Pré-processamento
        features = df.iloc[:, 1:25]
        target = df['status_fraude']
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            features_scaled, target, test_size=0.4, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Treinar modelos
        st.subheader("🤖 Modelos de Detecção")
        
        # Random Forest
        st.write("#### Random Forest")
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        # Rede Neural
        st.write("#### Rede Neural (MLP)")
        nn_model = create_nn_model()
        nn_model.fit(X_train, y_train)
        y_pred_nn = nn_model.predict(X_test)
        
        # Avaliação
        st.subheader("📈 Métricas de Avaliação")
        
        # Cálculo das métricas
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        precision_rf = precision_score(y_test, y_pred_rf)
        recall_rf = recall_score(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf)
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        
        accuracy_nn = accuracy_score(y_test, y_pred_nn)
        precision_nn = precision_score(y_test, y_pred_nn)
        recall_nn = recall_score(y_test, y_pred_nn)
        f1_nn = f1_score(y_test, y_pred_nn)
        cm_nn = confusion_matrix(y_test, y_pred_nn)
        
        # Dicionários com as métricas para plotagem
        metrics_rf = {
            'accuracy': accuracy_rf,
            'precision': precision_rf,
            'recall': recall_rf,
            'f1': f1_rf
        }
        
        metrics_nn = {
            'accuracy': accuracy_nn,
            'precision': precision_nn,
            'recall': recall_nn,
            'f1': f1_nn
        }
        
        # Gráfico de comparação de métricas
        st.write("#### Comparação de Métricas")
        fig = plot_metrics_comparison(metrics_rf, metrics_nn)
        st.pyplot(fig)
        
        # Exibição detalhada por modelo
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Random Forest**")
            st.metric("Acurácia", f"{accuracy_rf:.2%}")
            st.metric("Precisão", f"{precision_rf:.2%}")
            st.metric("Recall", f"{recall_rf:.2%}")
            st.metric("F1-Score", f"{f1_rf:.2%}")
            
            st.write("**Matriz de Confusão**")
            plot_confusion_matrix_with_labels(cm_rf, "Matriz de Confusão - Random Forest", 'Blues')
            st.pyplot(plt.gcf())
            plt.close()
        
        with col2:
            st.write("**Rede Neural (MLP)**")
            st.metric("Acurácia", f"{accuracy_nn:.2%}")
            st.metric("Precisão", f"{precision_nn:.2%}")
            st.metric("Recall", f"{recall_nn:.2%}")
            st.metric("F1-Score", f"{f1_nn:.2%}")
            
            st.write("**Matriz de Confusão**")
            plot_confusion_matrix_with_labels(cm_nn, "Matriz de Confusão - Rede Neural", 'Greens')
            st.pyplot(plt.gcf())
            plt.close()
        
        # Análise com IA
        gemini_analysis = None
        if st.button("🧠 Obter Análise Avançada com Gemini"):
            with st.spinner("Analisando dados com Gemini 1.5 Flash..."):
                try:
                    data_summary = df.describe().to_string()
                    
                    metrics = f"""
                    **Random Forest:**
                    - Acurácia: {accuracy_rf:.2%}
                    - Precisão: {precision_rf:.2%}
                    - Recall: {recall_rf:.2%}
                    - F1-Score: {f1_rf:.2%}
                    - Matriz de Confusão: \n{cm_rf}
                    
                    **Rede Neural:**
                    - Acurácia: {accuracy_nn:.2%}
                    - Precisão: {precision_nn:.2%}
                    - Recall: {recall_nn:.2%}
                    - F1-Score: {f1_nn:.2%}
                    - Matriz de Confusão: \n{cm_nn}
                    
                    **Relatório de Classificação (RF):**
                    \n{classification_report(y_test, y_pred_rf)}
                    """
                    
                    gemini_analysis = analyze_with_ai(data_summary, metrics)
                    st.session_state['gemini_analysis'] = gemini_analysis
                    
                    st.subheader("📝 Análise com Gemini 1.5 Flash")
                    st.markdown(gemini_analysis)
                except Exception as e:
                    st.error(f"Erro na análise com Gemini: {str(e)}")
        
        # Interpretação
        st.subheader("🔍 Guia de Interpretação")
        interpretation = """
        **Acurácia** (Accuracy):  
        > Porcentagem total de previsões corretas. Fórmula: (TP + TN) / (TP + TN + FP + FN)

        **Precisão** (Precision):  
        > Dos alertas de fraude emitidos, quantos eram realmente fraudes. Fórmula: TP / (TP + FP)

        **Recall** (Sensibilidade):  
        > Das fraudes reais existentes, quantas foram detectadas. Fórmula: TP / (TP + FN)

        **F1-Score**:  
        > Média harmônica entre Precisão e Recall. Útil quando há desbalanceamento de classes.
        > Fórmula: 2 * (Precision * Recall) / (Precision + Recall)

        **Matriz de Confusão**:
        - **TP** (True Positive): Fraudes detectadas corretamente
        - **FP** (False Positive): Consumos normais classificados como fraude
        - **TN** (True Negative): Consumos normais corretamente identificados
        - **FN** (False Negative): Fraudes não detectadas
        """
        with st.expander("Como interpretar essas métricas?"):
            st.markdown(interpretation)
        
        # Gerar PDF
        if st.button("📄 Gerar Relatório PDF"):
            with st.spinner("Gerando relatório..."):
                try:
                    data_info = f"""
                    Resumo dos dados analisados:
                    {df.describe().to_string()}
                    
                    Total de registros: {len(df)}
                    Período coberto: {df['data'].min()} a {df['data'].max()}
                    """
                    
                    metrics_info = f"""
                    **Métricas do Random Forest:**
                    - Acurácia: {accuracy_rf:.2%}
                    - Precisão: {precision_rf:.2%}
                    - Recall: {recall_rf:.2%}
                    - F1-Score: {f1_rf:.2%}
                    
                    **Métricas da Rede Neural:**
                    - Acurácia: {accuracy_nn:.2%}
                    - Precisão: {precision_nn:.2%}
                    - Recall: {recall_nn:.2%}
                    - F1-Score: {f1_nn:.2%}
                    """
                    
                    gemini_content = st.session_state.get('gemini_analysis', "Nenhuma análise Gemini foi gerada ainda.")
                    
                    pdf = generate_pdf(
                        data_info=data_info,
                        metrics=metrics_info,
                        gemini_analysis=gemini_content,
                        interpretation=interpretation,
                        df=df,
                        cm_rf=cm_rf,
                        cm_nn=cm_nn,
                        metrics_comparison=fig
                    )
                    
                    st.markdown(create_download_link(pdf, "relatorio_analise_energia.pdf"), unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Erro ao gerar relatório: {str(e)}")

if __name__ == "__main__":
    main()
