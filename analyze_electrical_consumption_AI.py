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
import google.generativeai as genai
from fpdf import FPDF
import base64
from datetime import datetime
import os
import tempfile

# Configuração inicial
st.set_page_config(page_title="Análise de Consumo de Energia", layout="wide")
st.title("🔍 Análise de Consumo de Energia com Detecção de Fraude")

# Configuração da API do Google Generative AI
genai.configure(api_key="SUA_CHAVE_API_AQUI")  # Substitua pela sua chave API

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
    
    def add_image(self, image_path, w=180):
        self.image(image_path, x=10, w=w)
        self.ln()

# Funções auxiliares
def save_figure(fig):
    """Salva uma figura matplotlib temporariamente"""
    temp_dir = tempfile.mkdtemp()
    image_path = os.path.join(temp_dir, 'temp_figure.png')
    fig.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return image_path

def create_download_link(pdf, filename):
    """Cria link de download para o PDF"""
    b64 = base64.b64encode(pdf.output(dest='S').encode('latin-1')).decode('latin-1')
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Baixar Relatório PDF</a>'

def find_csv_file():
    """Encontra arquivo CSV no Desktop"""
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

def load_data(uploaded_file=None):
    """Carrega os dados do arquivo CSV"""
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

def create_nn_model():
    """Cria modelo de rede neural"""
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2
    )

def analyze_with_ai(data_summary, metrics):
    """Analisa dados com Gemini"""
    model = genai.GenerativeModel("gemini-pro")
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

def generate_pdf_report(data_info, metrics, gemini_analysis, interpretation, images):
    """Gera o relatório PDF completo"""
    pdf = PDF()
    pdf.add_page()
    
    # Cabeçalho
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Relatório Completo de Análise', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", 0, 1)
    pdf.ln(10)
    
    # Seção 1: Dados Analisados
    pdf.add_section_title("1. Dados Analisados")
    pdf.add_content(data_info)
    
    # Gráficos exploratórios
    pdf.add_section_title("1.1 Análise Exploratória")
    for img in images[:2]:
        pdf.add_image(img)
    
    # Seção 2: Métricas
    pdf.add_section_title("2. Métricas dos Modelos")
    pdf.add_content(metrics)
    
    # Matrizes de confusão
    pdf.add_section_title("2.1 Matrizes de Confusão")
    for img in images[2:]:
        pdf.add_image(img)
    
    # Seção 3: Análise Gemini
    pdf.add_section_title("3. Análise Avançada")
    pdf.add_content(gemini_analysis)
    
    # Seção 4: Interpretação
    pdf.add_section_title("4. Guia de Interpretação")
    pdf.add_content(interpretation)
    
    return pdf

def main():
    """Função principal"""
    st.sidebar.header("Configurações")
    uploaded_file = st.sidebar.file_uploader("Carregar CSV", type=["csv"])
    
    # Carrega dados
    df = st.session_state.get('df', None)
    if uploaded_file or not df:
        df = load_data(uploaded_file)
    
    if df is not None:
        # Análise Exploratória
        st.header("📊 Análise Exploratória")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Consumo Médio Diário**")
            fig1, ax1 = plt.subplots()
            df.set_index('data')['consumo_medio_diario'].plot(ax=ax1)
            st.pyplot(fig1)
        
        with col2:
            st.write("**Consumo Mínimo Noturno**")
            fig2, ax2 = plt.subplots()
            df.set_index('data')['consumo_minimo_noturno'].plot(kind='bar', ax=ax2)
            st.pyplot(fig2)
        
        # Pré-processamento
        features = df.iloc[:, 1:25]
        target = df['status_fraude']
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)
        
        # Modelagem
        st.header("🤖 Modelos de Detecção")
        
        # Random Forest
        st.subheader("Random Forest")
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        # Rede Neural
        st.subheader("Rede Neural (MLP)")
        nn_model = create_nn_model()
        nn_model.fit(X_train, y_train)
        y_pred_nn = nn_model.predict(X_test)
        
        # Avaliação
        st.header("📈 Métricas de Avaliação")
        
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
        
        # Exibição
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Acurácia RF", f"{accuracy_rf:.2%}")
            st.metric("Precisão RF", f"{precision_rf:.2%}")
            st.metric("Recall RF", f"{recall_rf:.2%}")
            
            fig3, ax3 = plt.subplots()
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Fraude'], 
                       yticklabels=['Normal', 'Fraude'])
            st.pyplot(fig3)
        
        with col2:
            st.metric("Acurácia NN", f"{accuracy_nn:.2%}")
            st.metric("Precisão NN", f"{precision_nn:.2%}")
            st.metric("Recall NN", f"{recall_nn:.2%}")
            
            fig4, ax4 = plt.subplots()
            sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens',
                       xticklabels=['Normal', 'Fraude'],
                       yticklabels=['Normal', 'Fraude'])
            st.pyplot(fig4)
        
        # Análise Gemini
        st.header("🧠 Análise Avançada")
        if st.button("Gerar Análise com Gemini"):
            with st.spinner("Analisando..."):
                try:
                    data_summary = df.describe().to_string()
                    metrics = f"""
                    Random Forest:
                    - Acurácia: {accuracy_rf:.2%}
                    - Precisão: {precision_rf:.2%}
                    - Recall: {recall_rf:.2%}
                    
                    Rede Neural:
                    - Acurácia: {accuracy_nn:.2%}
                    - Precisão: {precision_nn:.2%}
                    - Recall: {recall_nn:.2%}
                    """
                    analysis = analyze_with_ai(data_summary, metrics)
                    st.session_state['gemini_analysis'] = analysis
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"Erro: {str(e)}")
        
        # Guia de Interpretação
        st.header("🔍 Guia de Interpretação")
        interpretation = """
        **Acurácia**: Porcentagem de previsões corretas\n
        **Precisão**: Fraudes detectadas corretamente\n
        **Recall**: Fraudes reais identificadas\n
        **Matriz de Confusão**:
        - TP: Verdadeiros positivos
        - FP: Falsos positivos
        - TN: Verdadeiros negativos
        - FN: Falsos negativos
        """
        st.markdown(interpretation)
        
        # Geração do PDF
        st.header("📄 Gerar Relatório")
        if st.button("Criar Relatório PDF"):
            with st.spinner("Gerando PDF..."):
                try:
                    # Salvar figuras
                    image_paths = [
                        save_figure(fig1),
                        save_figure(fig2),
                        save_figure(fig3),
                        save_figure(fig4)
                    ]
                    
                    # Conteúdo do relatório
                    data_info = f"""
                    Dados analisados:
                    - Registros: {len(df)}
                    - Período: {df['data'].min()} a {df['data'].max()}
                    - Variáveis: {', '.join(df.columns[1:25])}
                    """
                    
                    metrics_info = f"""
                    **Random Forest**
                    - Acurácia: {accuracy_rf:.2%}
                    - Precisão: {precision_rf:.2%}
                    - Recall: {recall_rf:.2%}
                    
                    **Rede Neural**
                    - Acurácia: {accuracy_nn:.2%}
                    - Precisão: {precision_nn:.2%}
                    - Recall: {recall_nn:.2%}
                    """
                    
                    gemini_content = st.session_state.get('gemini_analysis', "Nenhuma análise Gemini gerada")
                    
                    # Gerar PDF
                    pdf = generate_pdf_report(
                        data_info=data_info,
                        metrics=metrics_info,
                        gemini_analysis=gemini_content,
                        interpretation=interpretation,
                        images=image_paths
                    )
                    
                    # Download
                    st.markdown(create_download_link(pdf, "relatorio_consumo_energia.pdf"), unsafe_allow_html=True)
                    
                    # Limpar arquivos
                    for img in image_paths:
                        if os.path.exists(img):
                            os.remove(img)
                    os.rmdir(os.path.dirname(image_paths[0]))
                    
                except Exception as e:
                    st.error(f"Erro ao gerar PDF: {str(e)}")

if __name__ == "__main__":
    main()
