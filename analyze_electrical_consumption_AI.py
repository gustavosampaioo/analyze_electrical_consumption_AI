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
import google.generativeai as generai
from fpdf import FPDF
import base64
from datetime import datetime
import os
import tempfile

# Configuração inicial
st.set_page_config(page_title="Análise de Consumo de Energia", layout="wide")
st.title("🔍 Análise de Consumo de Energia com Detecção de Fraude")

# Configuração da API do Google Generative AI
generai.configure(api_key="AIzaSyBHouRPqa8LLjU96nEPk6UJBgswH66OJjY")  # Substitua pela sua chave API

# Classe para gerar PDF com gráficos
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

# Função para salvar figura temporariamente
def save_figure(fig):
    temp_dir = tempfile.mkdtemp()
    image_path = os.path.join(temp_dir, 'temp_figure.png')
    fig.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return image_path

# Função para gerar PDF
def generate_pdf(data_info, metrics, gemini_analysis, interpretation, images):
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
    
    # Gráficos de análise exploratória
    pdf.add_section_title("Gráficos de Análise Exploratória")
    for img in images[:2]:  # Primeiros dois gráficos
        pdf.add_image(img)
    
    # Seção 2: Métricas do Modelo
    pdf.add_section_title("2. Métricas do Modelo")
    pdf.add_content(metrics)
    
    # Matrizes de confusão
    pdf.add_section_title("Matrizes de Confusão")
    for img in images[2:]:  # Últimos dois gráficos
        pdf.add_image(img)
    
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

# [...] (mantenha todas as outras funções existentes até a função principal)

def main():
    # [...] (mantenha todo o código anterior até a seção de gráficos)
    
        # Exibição comparativa
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Random Forest**")
            st.metric("Acurácia", f"{accuracy_rf:.2%}")
            st.metric("Precisão", f"{precision_rf:.2%}")
            st.metric("Recall", f"{recall_rf:.2%}")
            
            st.write("**Matriz de Confusão**")
            fig_rf, ax_rf = plt.subplots()
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'Fraude'],
                        yticklabels=['Normal', 'Fraude'])
            plt.ylabel('Verdadeiro')
            plt.xlabel('Predito')
            st.pyplot(fig_rf)
        
        with col2:
            st.write("**Rede Neural (MLP)**")
            st.metric("Acurácia", f"{accuracy_nn:.2%}")
            st.metric("Precisão", f"{precision_nn:.2%}")
            st.metric("Recall", f"{recall_nn:.2%}")
            
            st.write("**Matriz de Confusão**")
            fig_nn, ax_nn = plt.subplots()
            sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens',
                        xticklabels=['Normal', 'Fraude'],
                        yticklabels=['Normal', 'Fraude'])
            plt.ylabel('Verdadeiro')
            plt.xlabel('Predito')
            st.pyplot(fig_nn)
        
        # Gráficos de análise exploratória
        fig_consumo, ax_consumo = plt.subplots()
        df.set_index('data')['consumo_medio_diario'].plot(ax=ax_consumo)
        ax_consumo.set_title('Consumo Médio Diário')
        
        fig_noturno, ax_noturno = plt.subplots()
        df.set_index('data')['consumo_minimo_noturno'].plot(kind='bar', ax=ax_noturno)
        ax_noturno.set_title('Consumo Mínimo Noturno')
        
        # [...] (mantenha o restante do código até a geração do PDF)
        
        # Geração do relatório PDF
        if st.button("📄 Gerar Relatório PDF"):
            with st.spinner("Gerando relatório..."):
                try:
                    # Salvar todas as figuras temporariamente
                    image_paths = [
                        save_figure(fig_consumo),
                        save_figure(fig_noturno),
                        save_figure(fig_rf),
                        save_figure(fig_nn)
                    ]
                    
                    # Preparar dados para o PDF
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
                    
                    **Métricas da Rede Neural:**
                    - Acurácia: {accuracy_nn:.2%}
                    - Precisão: {precision_nn:.2%}
                    - Recall: {recall_nn:.2%}
                    
                    **Matriz de Confusão (RF):**
                    {cm_rf}
                    
                    **Matriz de Confusão (RN):**
                    {cm_nn}
                    """
                    
                    # Obter análise do Gemini se existir
                    gemini_content = st.session_state.get('gemini_analysis', "Nenhuma análise Gemini foi gerada ainda.")
                    
                    # Gerar PDF
                    pdf = generate_pdf(
                        data_info=data_info,
                        metrics=metrics_info,
                        gemini_analysis=gemini_content,
                        interpretation=interpretation,
                        images=image_paths
                    )
                    
                    # Criar link de download
                    st.markdown(create_download_link(pdf, "relatorio_analise_energia.pdf"), unsafe_allow_html=True)
                    
                    # Limpar arquivos temporários
                    for img_path in image_paths:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    os.rmdir(os.path.dirname(image_paths[0]))
                    
                except Exception as e:
                    st.error(f"Erro ao gerar relatório: {str(e)}")
                    # Tentar limpar arquivos temporários em caso de erro
                    try:
                        for img_path in image_paths:
                            if os.path.exists(img_path):
                                os.remove(img_path)
                        os.rmdir(os.path.dirname(image_paths[0]))
                    except:
                        pass

if __name__ == "__main__":
    main()
