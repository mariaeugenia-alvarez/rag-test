# RAG PDF - Proof of Concept

Sistema RAG (Retrieval-Augmented Generation) para procesar y consultar documentos PDF.

## Características

- Carga y procesa múltiples PDFs
- Divide documentos en chunks optimizados
- Genera embeddings con OpenAI
- Almacena vectores en Pinecone
- Búsqueda semántica de información
- Generación de respuestas contextualizadas

## Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar API key
cp .env.example .env
# Editar .env con tu API key de OpenAI
```

## Uso

### 1. Indexar PDFs

```bash
python index_pdfs.py --pdf_dir ./pdfs
```

### 2. Hacer consultas

```bash
python query_rag.py "¿Cuál es el tema principal del documento?"
```

### 3. Usar el sistema de forma interactiva

```bash
python demo.py
```

## Estructura

- `pdf_processor.py` - Procesamiento y chunking de PDFs
- `vector_store.py` - Gestión de embeddings y Pinecone
- `rag_system.py` - Sistema RAG completo
- `index_pdfs.py` - Script para indexar documentos
- `query_rag.py` - Script para hacer consultas
- `demo.py` - Demo interactivo

## Notas

- Coloca tus PDFs en la carpeta `pdfs/`
- Los datos vectoriales se guardan en `pinecone_db/`
- Ajusta el tamaño de chunks según tus necesidades en `pdf_processor.py`
