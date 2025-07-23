import tempfile
import os

# Modifique a função generate_pdf para incluir os gráficos
def generate_pdf(data_info, metrics, gemini_analysis, interpretation, df, cm_rf, cm_nn):
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
    
    # Adicionar gráficos de consumo
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile1:
        # Gráfico de Consumo Médio Diário
        plt.figure(figsize=(8, 4))
        plt.plot(df.set_index('data')['consumo_medio_diario'])
        plt.title('Consumo Médio Diário')
        plt.xlabel('Data')
        plt.ylabel('Consumo')
        plt.tight_layout()
        plt.savefig(tmpfile1.name, dpi=100)
        plt.close()
        
        pdf.image(tmpfile1.name, x=10, w=190)
        pdf.ln(5)
        os.unlink(tmpfile1.name)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile2:
        # Gráfico de Consumo Mínimo Noturno
        plt.figure(figsize=(8, 4))
        plt.bar(df.set_index('data').index, df.set_index('data')['consumo_minimo_noturno'])
        plt.title('Consumo Mínimo Noturno')
        plt.xlabel('Data')
        plt.ylabel('Consumo')
        plt.tight_layout()
        plt.savefig(tmpfile2.name, dpi=100)
        plt.close()
        
        pdf.image(tmpfile2.name, x=10, w=190)
        pdf.ln(10)
        os.unlink(tmpfile2.name)
    
    # Seção 2: Métricas do Modelo
    pdf.add_section_title("2. Métricas do Modelo")
    pdf.add_content(metrics)
    
    # Adicionar matrizes de confusão
    pdf.add_section_title("Matrizes de Confusão")
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile3:
        # Matriz de Confusão Random Forest
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Fraude'],
                    yticklabels=['Normal', 'Fraude'])
        plt.title('Matriz de Confusão - Random Forest')
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        plt.tight_layout()
        plt.savefig(tmpfile3.name, dpi=100)
        plt.close()
        
        pdf.image(tmpfile3.name, x=10, w=90)
        os.unlink(tmpfile3.name)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile4:
        # Matriz de Confusão Rede Neural
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Normal', 'Fraude'],
                    yticklabels=['Normal', 'Fraude'])
        plt.title('Matriz de Confusão - Rede Neural')
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        plt.tight_layout()
        plt.savefig(tmpfile4.name, dpi=100)
        plt.close()
        
        pdf.image(tmpfile4.name, x=110, w=90)
        pdf.ln(10)
        os.unlink(tmpfile4.name)
    
    # Seção 3: Análise Avançada
    pdf.add_section_title("3. Análise Avançada (Gemini)")
    pdf.add_content(gemini_analysis)
    
    # Seção 4: Interpretação
    pdf.add_section_title("4. Guia de Interpretação")
    pdf.add_content(interpretation)
    
    return pdf

# Na parte do código onde você gera o relatório PDF, modifique a chamada para incluir os parâmetros adicionais:
if st.button("📄 Gerar Relatório PDF"):
    with st.spinner("Gerando relatório..."):
        try:
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
            """
            
            # Obter análise do Gemini se existir
            gemini_content = st.session_state.get('gemini_analysis', "Nenhuma análise Gemini foi gerada ainda.")
            
            # Gerar PDF com os parâmetros adicionais
            pdf = generate_pdf(
                data_info=data_info,
                metrics=metrics_info,
                gemini_analysis=gemini_content,
                interpretation=interpretation,
                df=df,
                cm_rf=cm_rf,
                cm_nn=cm_nn
            )
            
            # Criar link de download
            st.markdown(create_download_link(pdf, "relatorio_analise_energia.pdf"), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Erro ao gerar relatório: {str(e)}")
