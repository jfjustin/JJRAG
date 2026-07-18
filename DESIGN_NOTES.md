APP design:

-> 1.Document_loading (types of doc support)

-> 2.Document_splitting to fit model and better efficiency

Choosed RecursiveCharacterTextsplitter since dealing with non-structured input documents.

-> 3.Vectorize storage for splitted documents, localize storage for efficiency

Choosed Faiss vector stroage for:
Fast and efficient for similarity search operations
No server/cloud dependency (can run locally)
Well-integrated with LangChain ecosystem
Suitable for projects without need for distributed architecture

-> 4.Achieve Query, choose model type (embedding or llm or chatbot); choose embedding model type (local(Ollama) or connected(OpenAI))

Incorporated encrypted API_key input for data security, through 'dotenv' and 'os' pkgs.

-> 5.UI design and MAIN()fucntion (StreamLit...)





Generals in APP Design:

2.1. Error processing:

1 (Try, except) for all functions,

2 (logging for different part of code error),

3 defensive programming in Streamlit vacancy

2.2. further optimization:

additional Local Save/Load Functionality (.save_local() & .load_local()) feature
Core Benefits:

Persistence between application runs without external services Eliminates reload/reprocessing of documents and embeddings Reduces API calls and associated costs Enables offline operation after initial setup

Implementation Details:

Saves entire vector store state including vectors, metadata, and indexes Typically uses pickle/binary formats for efficient storage Preserves document-embedding relationships and search capabilities

